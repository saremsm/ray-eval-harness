from __future__ import annotations

import argparse
import logging
import sys
import time

import ray

from coordinator import DistributedEvalCoordinator
from hooks import EarlyStoppingHook, LoggingHook
from scoring import DEFAULT_CONDITIONS
from types_ import EvalTask
from worker import HFWorker

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s", 
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Task Templates
_COMPLETIONS = [
    ("The capital of France is", "Paris"),
    ("The sky appears", "blue"),
    ("Water freezes at", "zero"),
    ("The sun rises in the", "east"),
    ("A triangle has", "three"),
    ("The opposite of hot is", "cold"),
    ("Humans breathe", "oxygen"),
    ("Ice is frozen", "water"),
    ("The color of grass is", "green"),
    ("Birds use wings to", "fly"),
    ("Fish live in", "water"),
    ("The Earth orbits the", "sun"),
    ("A week has", "seven"),
    ("Dogs are known as man's best", "friend"),
    ("The moon orbits the", "Earth"),
    ("Cats are known for their ability to", "purr"),
    ("Lightning is followed by", "thunder"),
    ("Bread is made from", "flour"),
    ("A compass points", "north"),
    ("The tallest mountain is", "Everest"),
]

def make_tasks(n: int) -> list[EvalTask]:
    """Generate n completion tasks, cycling through templates if n > 20."""
    n_templates = len(_COMPLETIONS)
    tasks = []
    for i in range(n):
        prompt, answer = _COMPLETIONS[i % n_templates]
        cycle = i // n_templates
        task_id = (
            f"task_{i:05d}" if cycle == 0
            else f"task_{i:05d}_c{cycle}"
        )
        tasks.append(
            EvalTask(
                task_id=task_id, 
                prompt=prompt,
                expected_answer=answer, 
                conditions=list(DEFAULT_CONDITIONS),
                metadata={"prompt": prompt, "expected": answer, "cycle": cycle},
            )
        )
    return tasks

# Summary
def _bar_chars() -> tuple[str, str]:
    """Block characters for the pass-rate bar, with an ASCII fallback. Windows
    consoles often default to cp1252, which can't encode U+2588/U+2591 and
    crashes print() mid-summary."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "█░".encode(encoding)
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"

def _build_fault_injection(
    backend: str, failure_rate: float, seed: int, decider_shards: int = 1
):
    """Pick the fault-injecting worker class and create the shared decider."""
    if failure_rate <= 0.0:
        return None, None
    # Fault-injecting subclasses live in their own module so the production
    from fault_injection import (
        FaultInjectingHFWorker,
        FaultInjectingVLLMWorker,
        ShardedFailureDecider,
    )
    worker_cls = (
        FaultInjectingVLLMWorker if backend == "vllm"
        else FaultInjectingHFWorker
    )
    decider = ShardedFailureDecider(seed=seed, n_shards=decider_shards)
    worker_kwargs = {"failure_rate": failure_rate, "decider": decider}
    return worker_cls, worker_kwargs

def print_summary(summary: dict, wall_elapsed: float) -> None:
    width = 62
    print(f"\n{'=' * width}")
    print("  Batch evaluation results")
    print(f"{'=' * width}")

    #Aggregator always returns the full shape; empty run shows total=0.
    if summary.get("total", 0) == 0:
        print (
            f"  No tasks completed. {summary.get('total_intake', 0)} "
            f"submitted; check worker logs for errors."
        )
        print(f"  Wall time: {wall_elapsed:.1f}s")
        print(f"{'=' * width}\n")
        return
    
    main_fields = [
        ("Total tasks",       summary.get("total")),
        ("Succeeded",         summary.get("succeeded")),
        ("Failed",            summary.get("failed")),
        ("Stopped early",     summary.get("stopped_early", 0)),
        ("Success rate",      f"{summary.get('success_rate', 0):.1%}"),
        ("Mean score",        f"{summary.get('mean_score', 0):.3f}"),
        ("Min / Max score",   f"{summary.get('min_score', 0):.3f} / "
                              f"{summary.get('max_score', 0):.3f}"),
        # Per-task latency is an ESTIMATE (batch wall time / batch size)
        ("Mean latency",      f"{summary.get('mean_latency_s', 0):.2f}s  "
                              "(est.: batch / batch_size)"),
        ("p99 latency",       f"{summary.get('p99_latency_s', 0):.2f}s  "
                              "(est.; see batch latency)"),
        ("Mean batch latency", f"{summary.get('mean_batch_latency_s', 0):.2f}s  "
                               "(measured, task-weighted)"),
        ("p99 batch latency", f"{summary.get('p99_batch_latency_s', 0):.2f}s  "
                              "(measured, task-weighted)"),
        ("Mean tokens",       f"{summary.get('mean_tokens_generated', 0):.1f}"),
        ("Throughput",        f"{summary.get('throughput_per_s', 0):.1f} tasks/s"),
        ("Wall time",         f"{wall_elapsed:.1f}s"),
        ("Results file",      summary.get("results_file")),
    ]
    for label, value in main_fields:
        print(f"  {label:<22} {value}")

    # Sorted ascending: worst-performing conditions first.
    condition_stats = summary.get("condition_stats", {})
    if condition_stats:
        print(f"\n  Per-condition breakdown")
        print(f"  {'Condition':<30} {'Pass rate':>9}  {'Mean awarded':>12}")
        print(f"  {'-' * 56}")
        full_char, empty_char = _bar_chars()
        for cond, stats in sorted(
            condition_stats.items(),
            key=lambda x: x[1]["pass_rate"],
        ):
            pass_rate = stats["pass_rate"]
            mean_aw = stats["mean_awarded"]
            bar_len = int(pass_rate * 20)
            bar = full_char * bar_len + empty_char * (20 - bar_len)
            print(
                f"  {cond:<30} {bar} {pass_rate:>5.1%}  {mean_aw:>8.3f}"
            )
        print(
            f"\n  Pass rate = fraction of tasks where condition passed.\n"
            f"  Mean awarded = mean weight given (depends on condition weight).\n"
            f"  Sorted ascending: worst-performing conditions appear first."
        )
    
    worker_stats = summary.get("worker_stats", [])
    if worker_stats:
        print(f"\n  Per-worker stats:")
        print(
            f"  {'Worker':<8} {'Completed':>10} {'Failed':>8} {'Poisoned':>10}"
        )
        print(f"  {'-' * 40}")
        for ws in sorted(worker_stats, key=lambda x: x["worker_id"]):
            print(
                f"  {ws['worker_id']:<8} "
                f"{ws['completed']:>10} "
                f"{ws['failed']:>8} "
                f"{'yes' if ws['poisoned'] else 'no':>10}"
            )
    print(f"{'=' * width}\n")

# Entry Point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed LLM eval harness with intervention hooks."
    )
    parser.add_argument(
        "--tasks", type=int, default=20,
        help="Number of eval tasks (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel EvalWorker actors (default: 3)",
    )
    parser.add_argument(
        "--failure-rate", type=float, default=0.0, dest="failure_rate",
        help=(
            "Fraction of batches to fail artificially. Triggers the "
            "fault-injecting worker subclass (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help=(
            "Seed for fault injection RNG. Same seed + failure_rate "
            "reproduces the same failure pattern (default: 0)"
        ),
    )
    parser.add_argument(
        "--model", type=str, default="distilgpt2",
        help="HuggingFace model name (default: distilgpt2)",
    )
    parser.add_argument(
        "--backend", type=str, default="hf", choices=["hf", "vllm"],
        help=(
            "Inference backend: 'hf' (CPU or single GPU) or 'vllm' "
            "(GPU required, default: hf)"
        ),
    )
    parser.add_argument(
        "--hf-device", type=int, default=-1, dest="hf_device",
        help=(
            "Device for HF backend: -1 for CPU, 0+ for the corresponding "
            "CUDA device. With N workers on a GPU, each claims num_gpus="
            "1/N - accounting, not isolation: the model must fit N times "
            "(default: -1)"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help=(
            "Tasks per actor call. Default depends on backend: 4 for HF, "
            "64 for vLLM. Increase for vLLM throughput tuning."
        ),
    )
    parser.add_argument(
        "--task-timeout", type=float, default=60.0, dest="task_timeout",
        help=(
            "Per-BATCH timeout in seconds, enforced inside the worker "
            "(StoppingCriteria on HF, asyncio.wait_for on vLLM). It "
            "bounds the whole batch, not each task; size it for "
            "batch_size * worst-case task time. A timeout raises and the "
            "batch is retried; the worker is replaced only if a health "
            "check fails (default: 60.0)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Run 3 tasks across 2 workers for quick verification",
    )
    parser.add_argument(
        "--hook", action="store_true",
        help="Run the mid-generation intervention demo after the main run",
    )                    
    parser.add_argument(
        "--output", type=str, default="results/results.jsonl",
        help="Path for JSONL results (default: results/results.jsonl)",
    )
    parser.add_argument(
        "--aggregator-shards", type=int, default=1,
        dest="aggregator_shards",
        help=(
            "Number of ResultsAggregator shard actors, keyed by a "
            "stable hash of task_id. Default: 1 - the measured best "
            "(bench/results/SHARDED.md): with batched "
            "record_batch the single actor keeps up, and every extra "
            "shard adds one actor call per shard touched per batch to "
            "the single-threaded driver loop (b=8 ceiling 17.8k -> "
            "9.5k -> 6.4k tasks/s at 1/4/8 shards). "
            "With N > 1, per-shard JSONLs are written next to "
            "--output as <stem>.shardK.jsonl and concatenated into "
            "--output at end of run; the concatenated file contains "
            "every result exactly once but is NOT globally ordered "
            "(grouped by shard, completion order within each shard)."
        ),
    )
    parser.add_argument(
        "--standby", type=int, default=0,
        help=(
            "Number of pre-loaded standby workers kept idle for O(1) "
            "replacement of hung/poisoned workers (default: 0 = every "
            "replacement blocks on a fresh model load, the no-standby "
            "behavior). GPU-memory cost: each standby holds a full "
            "model in memory and claims the same Ray resources as a "
            "regular worker, so budget for --workers + --standby "
            "actors; with --hf-device >= 0 the fractional-GPU claims "
            "are sized for --workers only, and standbys may not "
            "schedule on an already-full device."
        ),
    )
    parser.add_argument(
        "--decider-shards", type=int, default=1,
        dest="decider_shards",
        help=(
            "Number of FailureDecider shard actors behind the "
            "ShardedFailureDecider facade, keyed by a stable hash of "
            "the batch key. Only used when --failure-rate > 0. "
            "Sharding cannot change which batches fail: decisions are "
            "a pure function of (seed, batch_key, attempt, "
            "failure_rate) and routing is key-only, so attempt "
            "counters keep shard affinity (default: 1)."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        args.tasks = 3
        args.workers = 2
        logger.info("Dry run: 3 tasks, 2 workers")
    
    if args.tasks < 1:
        print(f"Error: --tasks must be at least 1, got {args.tasks}.", file=sys.stderr)
        sys.exit(1)
    
    if args.workers < 1:
        print(f"Error: --workers must be at least 1, got {args.workers}.", file=sys.stderr)
        sys.exit(1)

    if args.aggregator_shards < 1:
        print(
            f"Error: --aggregator-shards must be at least 1, "
            f"got {args.aggregator_shards}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.standby < 0:
        print(
            f"Error: --standby must be at least 0, got {args.standby}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.decider_shards < 1:
        print(
            f"Error: --decider-shards must be at least 1, "
            f"got {args.decider_shards}.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    ray.init(ignore_reinit_error=True)

    tasks = make_tasks(args.tasks)
    worker_cls, worker_kwargs = _build_fault_injection(
        args.backend, args.failure_rate, args.seed, args.decider_shards
    )
    coordinator = DistributedEvalCoordinator(
        n_workers=args.workers,
        model_name=args.model,
        backend=args.backend,
        max_retries=2,
        task_timeout=args.task_timeout,
        output_path=args.output,
        batch_size=args.batch_size,
        aggregator_shards=args.aggregator_shards,
        worker_cls=worker_cls,
        worker_kwargs=worker_kwargs,
        hf_device=args.hf_device,
        standby=args.standby,
    )

    logger.info(
        f"Starting: {args.tasks} tasks, {args.workers} workers, " 
        f"backend={args.backend}, model={args.model}, "
        f"failure_rate={args.failure_rate}, seed={args.seed}"
    )

    wall_start = time.perf_counter()
    summary = coordinator.run(tasks)
    wall_elapsed = time.perf_counter() - wall_start

    print_summary(summary, wall_elapsed)

    if args.hook: run_hooked_demo(args.model, args.hf_device)
    ray.shutdown()

def run_hooked_demo(model_name: str, hf_device: int) -> None:
    """Single-task hook demo."""
    print("\n" + "=" * 62)
    print("  Mid-generation intervention demo (hooked evaluation)")
    print("=" * 62)

    worker = HFWorker.remote(
        worker_id=99,
        model_name=model_name,
        device=hf_device,
    )
    ray.get(worker.health_check.remote())

    task = EvalTask(
        task_id="hook_demo",
        prompt="The capital of France is",
        expected_answer="Paris",
        conditions=list(DEFAULT_CONDITIONS),
    )

    stop_hook = EarlyStoppingHook(triggers=["Paris", "paris"])
    # INFO level so the per-token lines are visible under the default logging
    log_hook = LoggingHook(
        task_id=task.task_id, log_every_n=1, level=logging.INFO
    )

    print(f"  Prompt:          '{task.prompt}'")
    print(f"  Expected answer: '{task.expected_answer}'")
    print(f"  Hook:            stop as soon as 'Paris' appears\n")

    result = ray.get(
        worker.evaluate_with_hooks.remote(task, [stop_hook, log_hook])
    )

    print(f"  Response:        {result.response!r}")
    print(f"  Score:           {result.score:.3f}")
    print(f"  Tokens generated:{result.tokens_generated}")
    print(f"  Stopped early:   {result.stopped_early}")
    triggered = result.hook_state.get("triggered_by")
    if triggered:
        print(f"  Triggered by:    {triggered!r}")
    print("=" * 62 + "\n")
    
if __name__ == "__main__":
    main()
