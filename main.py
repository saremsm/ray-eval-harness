from __future__ import annotations

import argparse
import logging
import sys
import time

import ray

from coordinator import DistributedEvalCoordinator
from types_ import EvalTask, ScoringCondition
from hooks import EarlyStoppingHook, LoggingHook
from worker import HFWorker

logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",)
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

# Standard Rubric for tasks:
_CONDITIONS: list[ScoringCondition] = [
    ScoringCondition(
        name="contains_answer", 
        weight=0.5,
        description="Response contains expected answer",
    ),
    ScoringCondition(
        name="answer_at_end",
        weight=0.2,
        description="Answer appears near the end of the response",
    ),
    ScoringCondition(
        name="is_concise", 
        weight=0.2,
        description="Response is not excessively long",
    ),
    ScoringCondition(
        name="mentions_topic",
        weight=0.1,
        description="Response mentions a keyword from the prompt",
    ),
]

def make_tasks(n: int) -> list[EvalTask]:
    """Generates n completion tasks, cycling through templates if n>20."""
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
            EvalTask(task_id=task_id, prompt=prompt,
                expected_answer=answer, conditions=list(_CONDITIONS),
                metadata={"prompt": prompt, "expected": answer, "cycle": cycle},
            )
        )
    return tasks

# Summary
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
        ("Success rate",      f"{summary.get('success_rate', 0):.1%}"),
        ("Mean score",        f"{summary.get('mean_score', 0):.3f}"),
        ("Min / Max score",   f"{summary.get('min_score', 0):.3f} / "
                              f"{summary.get('max_score', 0):.3f}"),
        ("Mean latency",      f"{summary.get('mean_latency_s', 0):.2f}s"),
        ("p99 latency",       f"{summary.get('p99_latency_s', 0):.2f}s"),
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
        print(f"  {'Condition':<30} {'Pass rate':>9} {'Mean awarded':>12}")
        print(f"  {'-' * 56}")
        for cond, stats in sorted(
            condition_stats.items(),
            key=lambda x: x[1]["pass_rate"],
        ):
            pass_rate = stats["pass_rate"]
            mean_aw = stats["mean_awarded"]
            bar_len = int(pass_rate * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(
                f"  {cond:<30} {bar} {pass_rate:>5.1%}  {mean_aw:>8.3f}"
            )
        print(
            f"\n Pass rate = fraction of tasks where condition passed.\n"
            f"  Mean awarded = mean weight given (depends on condition weight).\n"
            f"  Sorted ascending: worst-performing conditions appear first."
        )
    
    worker_stats = summary.get("worker_stats", [])
    if worker_stats:
        print(f"\n  Per-worker stats:")
        print(
            f"  {'Worker':<8} {'Completed': >10} {'Failed':>8} {'Poisoned':>10}"
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
    
def run_hooked_demo(model_name: str) -> None:
    """Single-task hook demo."""
    print("\n" + "=" * 62)
    print("  Mid-generation intervention demo (hooked evaluation)")
    print("=" * 62)

    worker = HFWorker.remote(
        worker_id=99,
        model_name=model_name,
    )
    ray.get(worker.health_check.remote())

    task = EvalTask(
        task_id="hook_demo",
        prompt="The capital of France is",
        expected_answer="Paris",
        conditions=list(_CONDITIONS),
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
    if stop_hook.triggered_by:
        print(f"  Triggered by:    {stop_hook.triggered_by!r}")
    print("=" * 62 + "\n")

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
        help="Number of parallel Evalworker actors (default: 3)",
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
        help="Run the mid-generation intervention demo after batch eval",
    )                    
    parser.add_argument(
        "--output", type=str, default="results/results.jsonl",
        help="Path for JSONL results (default: results/results.jsonl)",
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
    
    ray.init(ignore_reinit_error=True)

    tasks = make_tasks(args.tasks)
    coordinator = DistributedEvalCoordinator(
        n_workers=args.workers,
        model_name=args.model,
        backend=args.backend,
        task_timeout=args.task_timeout,
        output_path=args.output,
    )

    logger.info(
        f"Starting: tasks={args.tasks}, workers={args.workers}, "
        f"backend={args.backend}, model={args.model}"
    )

    wall_start = time.perf_counter()
    summary = coordinator.run(tasks)
    wall_elapsed = time.perf_counter() - wall_start
    print_summary(summary, wall_elapsed)
    if args.hook: run_hooked_demo(args.model)
    ray.shutdown()
    
if __name__ == "__main__":
    main()
