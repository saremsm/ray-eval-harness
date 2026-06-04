# Distributed LLM Eval Harness

Ray-based eval harness for LLMs with mid-generation hooks. Built to explore the infra problems that come up in agentic evals: scheduling inference across heterogeneous workers, recovering from hung or crashed processes, and exposing a clean per-token hook point for interventions on `AsyncLLMEngine`.

## What it does

- Schedules batches across a pool of Ray actors using a work-stealing coordinator. Retries transient failures with exponential backoff, replaces workers that hang or crash (force-killing the old actor so its GPUs are actually freed), enforces the retry budget on hangs as well as exceptions, and isn't blocked by the deferred retry queue.
- Two interchangeable backends behind an `EvalBackend` Protocol: a CPU/GPU HuggingFace pipeline and a GPU vLLM backend on `AsyncLLMEngine`.
- A streaming intervention API (`evaluate_with_hooks`) with per-token callbacks. The HF backend uses `StoppingCriteria` so hooks run synchronously inside `generate()` and can actually halt the model. The vLLM backend uses `AsyncLLMEngine` so hooks observe the engine's continuous-batching scheduler in real time and abort with `engine.abort(request_id)`. Hook-observed state (e.g. which trigger fired) returns to the caller via `EvalResult.hook_state`.
- Deterministic fault injection coordinated through a shared actor, so the same `--seed` reproduces the same failure pattern across runs regardless of how Ray scheduled work.
- A rubric scorer with named, weighted conditions and per-condition aggregation.
- An aggregator actor that serializes concurrent writes from many workers into one JSONL stream without explicit locking.
- Unit tests for the pure modules and integration tests for the coordinator using an in-memory fake backend, so tests don't need GPUs or model loads.

## Architecture

```
                        +------------------------+
                        |  DistributedEvalCoord. |
                        |  (driver process)      |
                        +-----------+------------+
                                    | ray.wait / ray.get
              +---------------------+--------------------+
              |                     |                    |
        +-----v-----+         +-----v-----+        +-----v-----+
        |EvalWorker |         |EvalWorker |        |EvalWorker |
        |  (actor)  |         |  (actor)  |        |  (actor)  |
        +-----+-----+         +-----+-----+        +-----+-----+
              |                     |                    |
              +---------------------+--------------------+
                                    | add_result.remote(...)
                        +-----------v------------+
                        |  ResultsAggregator     |
                        |  (actor; JSONL writer) |
                        +------------------------+
```

Worker pools are homogeneous - all HF or all vLLM per run, selected by `--backend` (mixed pools are not built). The coordinator never touches worker state directly. All communication goes through Ray's actor mailbox, which serializes calls per-actor.

## Bugs found and fixed

Bugs found while hardening the harness. Each fix has a regression test; the pattern is the same one the Testing section describes: find, explain the mechanism, pin it with a test.

1. **Old actors were never killed on replacement.** `_replace_worker` dropped the handle and constructed a replacement - but a hung or poisoned GPU-claiming actor still holds its `num_gpus` claim, so the replacement can never schedule: every attempt times out at the 120s health barrier and the slot dies even though the GPU was one kill away from free. The GPU fault-tolerance benchmark below couldn't catch this because injected failures raise *without* poisoning, so replacement-under-GPU-contention was never exercised on GPU. Fix: force-`ray.kill(old, no_restart=True)` before constructing the replacement (this also errors out any evicted in-flight ref). Test: `test_replace_worker_kills_old_actor`.
2. **Hangs incremented the retry budget but never enforced it.** The hang-eviction path re-queued with `retry_count + 1`, but only the exception path *checked* the budget - a batch that deterministically wedged the engine looped forever at one 240s eviction cycle per lap. Fix: `_handle_timeouts` enforces `max_retries` and records terminal failures on exhaustion. Test: `test_hang_budget_exhaustion_records_terminal_failures`.
3. **The hooked demo read hook state that was never mutated.** Ray pickles hook objects into the actor; mutations happen on the worker's copies, so the driver's `stop_hook.triggered_by` stayed `None` and the "Triggered by" line could never print. Fix: hook-observable state returns via `EvalResult.hook_state`; the demo (and any caller) reads it there. Tests: `TestHookStateReturnPath`.
4. **Hang detection only ran when the cluster was quiet.** The timeout scan was gated on `ray.wait` returning nothing - under steady completions from healthy workers it was starved exactly when the system was busiest, so a hung ref could linger far past the 240s threshold. Fix: the scan runs every loop iteration (it's O(active) dict reads).
5. **Timeout -> poison -> full model reload was disproportionate.** When the HF `StoppingCriteria` deadline fires, the pipeline call returns cleanly - the loaded model is intact - yet the worker marked itself permanently poisoned, costing a model reload per slow batch. Fix: a batch timeout raises `TimeoutError` without poisoning; replacement happens only when a health check actually fails.
6. **`--hf-device 0 --workers N` silently stacked N unaccounted replicas on one GPU.** Fix: the coordinator claims `num_gpus = 1/N` per HF worker on GPU (with the standard caveat that Ray fractions are scheduling accounting, not memory isolation) and logs a warning.
7. Smaller items: aggregator writes are now fire-and-forget with an end-of-run drain (a blocking `ray.get` per batch stalled the single-threaded scheduler); the aggregator holds its file open instead of reopening per write; the p99 index used `int(0.99n)`, which is the max (p100) at n=100 - now nearest-rank; `mentions_topic` no longer awards near-free credit via stopwords like "The" (word-boundary matching on content words); the standard rubric now has one source of truth in `scoring.DEFAULT_CONDITIONS`; a module-level test assert became a real test.

## Design notes

A few choices that aren't obvious from reading the code:

**Backend Protocol with a runtime hasattr check.** Ray actor handles are remote references, not the actor itself, so `isinstance(handle, EvalBackend)` returns False even when the actor implements the Protocol. The Protocol exists for documentation and unit-test fakes; `validate_backend` does an attribute check at startup so a missing method fails before the first batch.

**Hung detection is per-ref, not per-worker.** A worker can issue many refs over its lifetime. Tracking timeouts per worker means a slow batch looks like a dead worker, which evicts workers that are actually fine. Per-ref means a healthy worker that finishes one batch and starts another resets cleanly.

**Health check on every failure path.** When a batch fails, the coordinator health-checks the worker before deciding to retry. Without this, a poisoned worker keeps accepting batches that fail instantly, and the retry budget gets burned. What is actually checked is per-backend:

- **HF**: `health_check` returns `not self._poisoned` - a flag reserved for unrecoverable states; batch timeouts don't set it.
- **vLLM**: `health_check` returns False if (a) the last `evaluate_batch`/`evaluate_with_hooks` raised a CUDA or engine-fatal error and no batch has succeeded since (`_last_error`; matched classes: `torch.cuda.OutOfMemoryError`, vLLM's `EngineDeadError`/`EngineGenerateError` (V1) or `AsyncEngineDeadError` (V0), plus `RuntimeError`s carrying CUDA fingerprints like "illegal memory access" or "device-side assert" - `TimeoutError` never counts, timeouts don't poison), or (b) the engine's `errored` property is truthy or raises, or (c) the engine's `check_health()` - sync or async, duck-typed - raises or fails to answer within `HEALTH_CHECK_TIMEOUT_S`. In the pinned vllm==0.15.0, `AsyncLLMEngine` is an alias of the V1 `AsyncLLM`, which exposes both `errored` and async `check_health()` (raising `EngineDeadError` when errored); the duck typing keeps V0-shaped engines (same two names) and engines exposing neither (assumed healthy) working. Before this, the vLLM `health_check` returned True unconditionally, so a dead engine burned the whole retry budget and the poison path never fired for that backend.

The integration test `test_poisoned_worker_replaced_on_first_failure` covers the coordinator path; `tests/test_worker_health.py` covers the per-backend checks.

**Two timeout layers, one budget.** The worker enforces a per-batch timeout via `StoppingCriteria` (HF) or `asyncio.wait_for` (vLLM) and raises on overrun. The coordinator enforces a separate per-ref hang threshold for the case where the actor itself becomes unreachable (process death, GC stall, network). The worker-side timeout always fires first on a live worker; the coordinator threshold only exists to catch workers whose own timeout machinery is dead, so it is derived from `--task-timeout` rather than fixed (a fixed 240s evicted healthy workers whenever `--task-timeout` exceeded it):

```
hang_threshold_s = max(HANG_THRESHOLD_MIN_S,
                       HANG_MULTIPLIER * task_timeout + HANG_MARGIN_S)
```

| Constant              | Value | Why                                                                 |
| --------------------- | ----: | ------------------------------------------------------------------- |
| `HANG_THRESHOLD_MIN_S`|   120 | Floor: tiny `--task-timeout` values must not make eviction trigger-happy - eviction costs a blocking model reload. |
| `HANG_MULTIPLIER`     |     2 | A batch may run its full budget and spend nearly as long again raising/serializing; 2x keeps a live worker's slowest legal batch inside the threshold. |
| `HANG_MARGIN_S`       |    30 | Actor-mailbox and scheduling latency, which does not shrink with small timeouts. |

The derived value is logged at startup and reported in the summary as `hang_threshold_s`. Different failure modes, different thresholds - and both paths consume and enforce the same retry budget.

**Per-backend batch sizes.** vLLM's continuous batching only saturates with many concurrent in-flight requests, so the coordinator hands `VLLMWorker` 64 tasks per call by default. HF runs forward-pass-bound, so 4 is plenty there. This single change is the difference between the harness bottlenecking the engine at 6 tasks/s and the engine running freely at ~120; see Performance below.

**Tuples in `pending`.** Retries re-append to `pending`, and a re-queued batch needs to remember how many times it's been tried. Keeping the count in the tuple means the budget is enforced on every retry path, including hangs.

**Aggregator as an actor, writes as fire-and-forget.** Many workers writing one JSONL file becomes "many workers calling `add_result.remote(...)` on one actor." Ray serializes actor methods for free - no `fcntl`, no threading lock. The coordinator doesn't block on each write: actor tasks from one caller execute in submission order, so the end-of-run `get_summary` is guaranteed to see every write; the write refs are drained at the end so exceptions still surface.

**Shared `FailureDecider` actor for deterministic injection.** The naive design - each worker holds its own seeded RNG - gives reproducible *per-worker* sequences but not reproducible *task-level* failures, because Ray's scheduler routes batches non-deterministically. A single decider actor that all fault-injecting workers query makes failure a function of `(seed, batch_content, attempt_number)` rather than which worker happened to receive the batch.

**`AsyncLLMEngine` for hooks.** Synchronous `LLM.generate` returns when all prompts are done with no per-token callback, and there can't be one without breaking continuous batching. `AsyncLLMEngine.generate` returns an async iterator of `RequestOutput`s, one per scheduling step, so hooks can observe one specific request while the engine batches others. `engine.abort(request_id)` is the clean way to stop a single in-flight request. This targets the API surface of the pinned vLLM version; the V1 engine's maintained class is `AsyncLLM`, and both the import path and the cumulative-output assumption behind the delta diffing should be re-verified on any vLLM upgrade.

## Layout

```
.
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point
├── coordinator.py               # work-stealing scheduler, retry, hang detection
├── worker.py                    # HFWorker and VLLMWorker actors
├── fault_injection.py           # FailureDecider + fault-injecting worker subclasses
├── aggregator.py                # ResultsAggregator actor (JSONL + stats)
├── scoring.py                   # rubric scorer + condition checkers
├── hooks.py                     # LoggingHook, EarlyStoppingHook
├── types_.py                    # dataclasses, enums, Protocols
├── utils.py                     # pure helpers (make_batches)
├── results/                     # JSONL output
└── tests/
    ├── test_coordinator.py      # helpers + end-to-end against a fake backend
    ├── test_fault_injection.py  # determinism, FailureDecider, seeding
    ├── test_scoring.py
    ├── test_hooks.py
    └── test_utils.py
```

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# CPU smoke test (no GPU needed)
python main.py --dry-run

# Larger HF run on CPU
python main.py --tasks 200 --workers 4 --backend hf

# HF run on a single GPU (each of N workers claims num_gpus=1/N;
# accounting, not isolation - the model must fit N times)
python main.py --tasks 200 --workers 4 --backend hf --hf-device 0

# vLLM run (requires GPU + a vLLM-compatible model)
python main.py --tasks 1000 --workers 1 --backend vllm \
    --model Qwen/Qwen2.5-1.5B

# Override the per-backend default batch size
python main.py --tasks 1000 --workers 1 --backend vllm \
    --model Qwen/Qwen2.5-1.5B --batch-size 128

# Mid-generation hook demo
python main.py --hook --backend hf
```

Inject failures to exercise the retry path. The fault-injection RNG is seeded, so `--seed 42` reproduces the same failure pattern across runs:

```bash
python main.py --tasks 200 --workers 1 --backend vllm \
    --model Qwen/Qwen2.5-1.5B --batch-size 16 \
    --failure-rate 0.3 --seed 42
```

## Performance

Cloud GPU. Lambda A10 (24GB VRAM), `Qwen/Qwen2.5-1.5B`, vLLM backend, 1 worker, 1000 tasks per row, greedy decoding, max_new_tokens=80. Numbers were measured on the pre-review code; none of the review fixes sit on the vLLM batch hot path (the write path became less blocking, if anything), but re-run before quoting them as current.

| `--batch-size` | Throughput (tasks/s) | Engine calls |
| -------------: | -------------------: | -----------: |
|              4 |                  6.0 |          250 |
|             16 |                 22.8 |           63 |
|             32 |                 44.5 |           32 |
|             64 |                 75.1 |           16 |
|            128 |                117.1 |            8 |
|            256 |                120.1 |            4 |

**Read the latency column with care.** `latency_seconds` is an estimate - batch wall time / batch size - so it *mechanically shrinks* as batch size grows; the apparent p99 "improvement" down the table is an artifact of the estimator, not a latency win. A task's real time-in-system grows with batch size (it waits for its whole batch). The summary now also reports **batch latency** (measured wall time of the batch a task rode in, task-weighted), which is the honest per-task figure; use throughput and batch latency to compare configurations, not the estimated per-task column.

Throughput climbs steeply with batch size because continuous batching only saturates when many requests are in flight; the default of 64 captures most of the available throughput (12.7× the batch=4 baseline) without committing all of GPU memory to in-flight KV cache.

From 128 to 256 the sweep gains ~2.5% (117.1 → 120.1 tasks/s) for real memory: past saturation, extra batch size buys KV-cache commitment, not speed, and a default that pre-spends the whole cache leaves nothing for longer generations or the hooked path sharing the engine.

The batch-4 row is the control: 5.91 tasks/s on the old harness vs 6.0 on this one, same hardware - the harness reproduces the old behavior when forced to the old batch size, so the climb up the table reflects a real change in how the harness drives the engine, not different hardware or a different model.

**Laptop CPU baseline.** Lenovo IdeaPad 82FE, Intel Core i5-1135G7, `distilgpt2`, HF backend, 4 workers, 50 tasks. Included for the smoke-test fault tolerance comparison, not for absolute throughput. `distilgpt2` scores poorly on the trivia rubric because it's an 82M-parameter model trained mostly on web text; that's the model, not the harness. (Note: the review tightened `mentions_topic`, so per-condition pass rates from older runs are not comparable to current ones.)

| Scenario (50 tasks, 4 workers)      | Outcome                        |
| ----------------------------------- | ------------------------------ |
| Clean run                           | 50/50 succeeded                |
| `--failure-rate 0.3 --seed 42`      | 50/50 after retries            |
| One worker killed mid-run           | 50/50; run continues at n-1    |
| Kill + `--failure-rate 0.3`         | 50/50; replacement + retries   |

## Reproducibility

Both backends use greedy decoding (`temperature=0.0` on vLLM; the HF text-generation pipeline defaults to `do_sample=False`). Outputs are **empirically stable across runs** (table below), but greedy decoding under continuous batching is not *guaranteed* bitwise-deterministic: batch composition changes kernel shapes and reduction orders, and composition varies with retry timing (backoff jitter deliberately uses an unseeded RNG). Failure *decisions* are fully deterministic; wall-clock scheduling is not.

Failure injection goes through a single shared `FailureDecider` actor that all fault-injecting workers query. The decider tracks per-batch attempt counts globally, so a decision is a function of `(seed, batch_content, attempt_number)` and not of which worker happened to receive the batch.

Verified on GPU (pre-review code): 200 tasks, 13 batches at batch_size=16, `--failure-rate 0.3 --seed 42`. Two independent runs on a fresh actor pool produced identical outcomes:

|                                 | seed 42, run 1 | seed 42, run 2 | seed 99 |
| ------------------------------- | -------------: | -------------: | ------: |
| Terminal failures (after retry) |             16 |             16 |       0 |
| Internal failure events         |             96 |             96 |      96 |
| `contains_answer` pass rate     |          74.5% |          74.5% |   75.0% |
| Wall time                       |          33.6s |          32.2s |   31.7s |

Same seed -> same outcome, including which specific batch exhausted its retry budget. Different seed -> different outcome (seed 99 happened to land all its failures on batches that recovered within the retry budget). Wall time varies by ~1s across seed-42 runs from system noise (vLLM CUDA graph capture, scheduler latency); reproducibility here means identical decisions at every injected-failure draw, not identical wall times or bitwise-identical logits.

The companion test `tests/test_fault_injection.py` covers the determinism guarantee at unit-test scale, including a regression for an earlier bug where Python's per-process-randomized `hash()` defeated the seeded RNG.

## Testing

```bash
pytest tests/
```

The `scoring`, `hooks`, and `utils` tests are pure Python and finish in well under a second. The coordinator tests cover both helpers and the full `run()` loop end-to-end. The end-to-end tests stub `ray.wait`, `ray.get`, and `ray.kill` and use an in-memory fake backend that satisfies `EvalBackend` structurally, so they exercise the real scheduler logic without a Ray cluster, GPU, or model load.

Real bugs surfaced by tests, in two waves. From the original development:

1. A deferred retry could promote back to `pending` after every active worker had gone idle, at which point nothing in the main loop dispatched the pending work and the run loop exited early. Fixed by including `pending` in the main loop condition and adding a dispatch-to-idle step after promotion.
2. `_handle_failure` only health-checked the worker on the terminal-failure branch. While retries remained, a poisoned worker kept receiving submissions that failed immediately, burning the retry budget. Fixed by running the health check on every failure path, before the retry decision.
3. The first attempt at deterministic fault injection seeded a per-worker RNG with `(seed, worker_id)`. This gave reproducible *per-worker* sequences but not reproducible *task-level* failures, because the scheduler routes batches non-deterministically. Replaced with the `FailureDecider` actor pattern; the regression test seeds via SHA-256 hashing of `(seed, batch_repr)` and verifies cross-run equivalence.

And the items in **Bugs found and fixed** above - most notably the unkilled-actor GPU leak and the unenforced hang budget, both of which lived in code paths the happy-path benchmarks never exercised. All have regression tests against the fake-backend integration harness or the unit suites.

## Known limitations

These are real and worth knowing about; none are showstoppers.

- **Whole-batch retry on partial failure.** When one task in a batch fails, the entire batch is re-queued. At vLLM batch sizes (64+), this re-runs many successful generations to recover one failure. Per-task retry, or splitting a failed batch in half on retry, would be the natural improvements.
- **The single aggregator actor caps throughput at thousands of QPS.** Sharded aggregators keyed by `task_id % N` would scale linearly but aren't implemented. The same shape applies to the `FailureDecider`, which is also a single actor by design (with an unbounded per-batch attempt map) and would need sharding and eviction for multi-thousand-worker injection.
- **Worker replacement blocks the scheduling loop.** The old actor is now killed immediately and the slot marked dead first, but constructing the replacement still blocks on model load - minutes at 70B scale - during which completed refs from healthy workers sit unharvested. A standby pool (workers pre-loaded and idle) or backgrounded replacement would make it O(1); neither is built.
- **`tokens_generated` is left at 0 on the HF batch path.** Reported accurately on the hooked path (counted from criterion calls) and on vLLM batch (from `len(token_ids)`). Re-tokenizing the response string round-trips through BPE merges and isn't guaranteed to match the generated count, so the field stays unset rather than reporting a wrong number. Plumbing token IDs out of the HF pipeline is the proper fix.
- **`tensor_parallel_size > 1` is wired through but only verified at `tp=1`.** The coordinator passes `tp` to both initial creation and worker replacement, the actor's `@ray.remote(num_gpus=tp)` claim follows from it, but a real multi-GPU run also wants `STRICT_PACK` placement groups for colocation; that piece isn't built. Single-GPU and CPU paths are exercised end-to-end on real hardware; multi-GPU and multi-node are designed-in but not run in anger.
- **Failed-batch results are only written to JSONL at terminal outcome.** Hang exhaustion and all-workers-dead are now recorded, but a run that aborts mid-retry (e.g. SIGKILL) still leaves no record of in-flight batches. Writing on first failure with a `pending_retry` flag and updating on terminal outcome would close the gap.
