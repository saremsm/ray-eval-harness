# Distributed LLM Eval Harness

Ray-based eval harness for LLMs with mid-generation hooks. Built to explore the infra problems that come up in agentic evals: scheduling inference across a pool of workers, recovering from hung or crashed processes (including whole-node loss), and exposing a clean per-token hook point for interventions on `AsyncLLMEngine`.

## What it does

- Schedules batches across a pool of Ray actors using a **work-queue coordinator**: a central pending queue with pull-based dispatch - each worker holds exactly one batch in flight, and completing (or failing) a batch pulls the next one off the queue for that slot. Workers never take work from each other; all routing is the driver handing the head of the queue to whichever slot just freed up. The coordinator retries failures with exponential backoff, re-queues a failed multi-task batch as two halves so a poisoned task is bisected to a singleton in ~log2(batch) failures instead of condemning its batchmates, replaces workers that hang or crash (force-killing the old actor so its GPUs are actually freed, then promoting a pre-loaded standby in O(1) when `--standby N` is set), enforces the retry budget on hangs as well as exceptions, and isn't blocked by the deferred retry queue.
- Two interchangeable backends behind an `EvalBackend` Protocol: a CPU/GPU HuggingFace pipeline and a GPU vLLM backend on `AsyncLLMEngine`.
- A streaming intervention API (`evaluate_with_hooks`) with per-token callbacks. The HF backend uses `StoppingCriteria` so hooks run synchronously inside `generate()` and can actually halt the model. The vLLM backend uses `AsyncLLMEngine` so hooks observe the engine's continuous-batching scheduler in real time and abort with `engine.abort(request_id)`. Hook-observed state (e.g. which trigger fired) returns to the caller via `EvalResult.hook_state`.
- Deterministic fault injection coordinated through a shared actor, so the same `--seed` reproduces the same failure pattern across runs regardless of how Ray scheduled work.
- A rubric scorer with named, weighted conditions and per-condition aggregation.
- A sharded results aggregator: a driver-side facade over N `ResultsAggregator` actors (`--aggregator-shards`, default 1 - the measured best; see Scaling) that serializes concurrent writes into per-shard JSONL streams without explicit locking, batches one actor call per shard per completed batch, and merges a single summary at end of run.
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
                                    | record_batch (one call per
                                    | shard per completed batch)
                        +-----------v------------+
                        |  ShardedAggregator     |
                        |  (driver-side facade)  |
                        +--+--------+--------+---+
                           |        |        |
                     +-----v--+ +---v----+ +-v------+
                     |Results | |Results | |Results |
                     |Aggr. 0 | |Aggr. 1 | |Aggr. N-1  (actors;
                     +--------+ +--------+ +--------+  JSONL shards)
```

Worker pools are homogeneous - all HF or all vLLM per run, selected by `--backend` (mixed pools are not built). The coordinator never touches worker state directly. All communication goes through Ray's actor mailbox, which serializes calls per-actor.

## Scaling

This section answers, with attached measurements only, what the harness's throughput ceiling is, which component sets it, how that changed across the scaling work, and how the system behaves when a whole node dies. Raw artifacts: `bench/results/BASELINE.md` (baseline), `bench/results/SHARDED.md` + `bench-results-sharded/` (shard sweep, post-batched-writes), `bench/multinode/MULTINODE_RESULTS.txt` + `bench/multinode/multinode_results.json` (multinode), `bench/results/VLLM_BATCH_SWEEP.md` (single-worker engine sweep).

### The saturation method

`bench/` measures **at what offered load the coordinator itself saturates.** The coordinator is a single-threaded driver loop - `ray.wait`, dispatch, aggregator submits - and at some completion rate that loop, not the GPUs, becomes the ceiling.

`bench/saturation.py` drives the *real* `DistributedEvalCoordinator` and the *real* aggregator actors, but replaces model workers with `FakeLatencyWorker` async actors whose `evaluate_batch` just `asyncio.sleep`s for `latency_s * (1 + jitter*U(-1,1))` and returns well-formed results with tiny outputs. Offered load is then a free dial - `workers * batch_size / latency_s` tasks/s - with no GPUs involved, and `--fail-rate > 0` routes through the existing `FailureDecider` so the retry machinery is part of what gets measured. `bench/sweep_saturation.py` runs the grid (latency {0.5, 0.1, 0.02, 0.005} s x workers {16, 64, 128, 256} x batch {1, 8, 64}); `bench/plot_saturation.py` draws achieved-vs-offered per batch size and prints the knee: the first point, in offered-load order, where achieved < `KNEE_RATIO` (0.8) x offered.

**Why completions/sec on the x-axis, not GPU count.** The coordinator never sees a GPU; it sees completion events. 8 fast workers finishing 64-task batches every 200 ms present exactly the same coordination load as 256 slow workers finishing the same batches every 6.4 s. Completions/sec is the load variable the coordinator actually saturates on, it's what the fake workers let us dial without hardware, and it makes the result portable: multiply a planned deployment's GPU count by its per-GPU completion rate and read the headroom straight off the curve. Worker count is a process count (every Ray actor is an OS process), so offered load past a few hundred actors is raised by lowering latency, not adding actors - which matters twice over, because worker count itself costs: at fixed offered load, 256 workers achieve materially less than 64 (batch=1 at offered 12,800: 2,687.3 vs 1,960.7 tasks/s, -27%).

**Instrumentation is measurement-only.** The coordinator carries a `_metrics` seam (`metrics.py`) defaulting to a no-op `NullMetrics` singleton; the bench swaps in `bench/recording.py`'s `RecordingMetrics` after construction. Timers cover `ray_wait` / `dispatch` / `agg_submit` / `loop_iter`; gauges sample `pending` / `active` / `deferred` / `standby` (live ready-pool size); counts track `completed` / `failed` / `retried` / `replaced`. One caveat every report states explicitly: **Ray exposes no actor mailbox depth**, so aggregator backlog is measured as submit-to-ready call latency, sampled every Nth call from the coordinator side. The actor drains its mailbox FIFO, so call latency *is* queue depth expressed in seconds - a proxy, and labeled as one in the JSON.

### Curve 1 - baseline: the per-task write path saturates first

Environment: Vultr dedicated-CPU, 32 vCPU, Python 3.12.3, ray 2.50.0; full grid, three independent sweeps, per-rep spread on achieved throughput <= 1.5% at every grid point. Raw reports: `bench/results/published/`; frozen analysis: `bench/results/BASELINE.md`.

| batch | ceiling (tasks/s, 3 reps)      | knee (offered t/s) |
|------:|--------------------------------|--------------------|
| 1     | 2,701.0 / 2,682.9 / 2,677.9    | 2,560              |
| 8     | 4,911.0 / 4,841.7 / 4,905.8    | 4,096 (marginal: re-crosses 80% at 5,120, permanently below from 6,400) |
| 64    | 4,873.3 / 4,760.7 / 4,848.3    | 8,192 (previous grid point 2,048 at 0.931; true crossing unsampled inside that bracket) |

**What saturated first: the per-task `add_result` submission path into the single `ResultsAggregator` actor.** Three measurements, in increasing directness:

1. The ceiling is per-*task*: batch 8 and batch 64 share a ~4,900 tasks/s ceiling despite an 8x difference in batches and dispatches per second, while batch 1 - one submit *and* one batch per task - halves it to ~2,700.
2. `agg_submit`'s share of the steady-state loop grows monotonically with saturation depth: 62.2% at the throughput peak, 75.0% at the b=64 knee, 88.9% at the deepest grid point, 90.05% in the long knee run - with `ray_wait` at 0.9-1.6% at those points. The coordinator was not slow at scheduling; it was spending nine-tenths of its time submitting results.
3. The aggregator's mailbox was seconds deep: the submit-to-ready proxy sat at p50 0.8-0.9 ms / p99 2-5 ms unsaturated, and p50 7,365.6 ms / p99 14,894.3 ms at the deepest grid point (p99 16,558 ms in the long knee run) - four orders of magnitude on the same code path.

Secondary, batch=1 only: below ~2,700 tasks/s the write path is not yet binding and per-batch scheduling overhead sets the ceiling instead - `dispatch` is the largest timer at every deep b=1 point (35.2% at the knee, 39.1% at the ceiling point).

Batching the writes attacked measurement 1's source directly: aggregator writes became one `record_batch` call per shard touched per completed batch, fire-and-forget with an end-of-run drain, instead of one blocking call per result.

### Curve 2 - after batched writes, plus the shard sweep

Same box class and grid, run after the write batching, at `--aggregator-shards` 1 / 4 / 8 (3 reps / 1 / 1; single-rep numbers carry no spread, and the shards=1 per-rep spread is <= 5.2%, worst at knee-adjacent b=64 points). Raw reports: `bench-results-sharded/`; full analysis: `bench/results/SHARDED.md`; overlay: `python -m bench.plot_saturation --compare bench-results-sharded/a1 bench-results-sharded/a4 bench-results-sharded/a8 --out bench/results/sharded_overlay.png`:

![Shard-sweep overlay: a1 vs a4 vs a8](bench/results/sharded_overlay.png)

| batch | shards=1 ceiling / knee               | shards=4            | shards=8            |
|------:|---------------------------------------|---------------------|---------------------|
| 1     | 2,568.3 t/s, knee 2,560               | 2,516.2, 2,560      | 2,473.7, 2,560      |
| 8     | **17,785.5 t/s, knee 20,480**         | 9,482.4, 10,240     | 6,406.6, 6,400      |
| 64    | 65,624.8 t/s, knee 32,768 (marginal)  | 47,521.8, 32,768    | 34,624.0, 32,768    |

The shards=1 curve already sits 3.6-13x above the baseline wherever the per-result path was the ceiling - the baseline bottleneck is gone at every shard count. Where write batching changes little (unsaturated batch=1), a1 matches the baseline within 0.5%, and the deep b=1 points run 1.0-5.1% *below* the baseline - `record_batch` pays a per-call payload/shard-hash cost that at batch=1 replaces a cheaper per-result call one-for-one.

**What saturates now: the single-threaded driver loop itself.** At the a1 batch=8 ceiling point no single timer dominates: `agg_submit` 35.8%, `dispatch` 33.8%, unaccounted 23.6%, `ray_wait` 6.8% (loop duty cycle 0.998). No actor is backed up behind it - the mailbox proxy p99 is <= ~0.3 s at the deepest points, versus 15-17 s in the baseline. The ceilings now scale with batch size because the residual cost is per completion *event*: in batch units they are ~2,568 / ~2,223 / ~1,025 completed batches/s at b = 1 / 8 / 64.

**Sharding the aggregator makes throughput strictly worse**, at every measured shard count and every saturated b>=8 grid point: the b=8 ceiling drops 47% then 64%, the b=8 knee halves at 4 shards and halves again at 8. The sharding intervention is itself the controlled experiment that proves the actor no longer saturates first: dividing it 4- and 8-ways would raise the ceiling if it did; instead, driver-side `agg_submit` share at the fixed b=8 ceiling point climbs 35.8% -> 64.7% -> 74.3% while loop iterations/s fall 2,473 -> 1,197 -> 813 and the per-shard mailbox proxy p50 *drops* to ~0.7-1.0 ms. Every extra shard converts one cheap actor-side merge into extra serialize-and-submit calls (a b=8 batch touches ~3.6 of 4 / ~5.2 of 8 shards in expectation), and that cost lands on the one thread that cannot be sharded. Batch=1 barely moves at any shard count (a singleton batch touches exactly one shard) and stays dispatch-bound.

### Chosen shard defaults

- `--aggregator-shards 1` - **the measured best**, not a placeholder: 1 beats 4 and 8 at every saturated grid point. The flag remains for re-measurement if a future change (e.g. moving scoring into the aggregator) makes the actor the bottleneck again.
- `--decider-shards 1` - **unmeasured**. Decisions are provably identical at any shard count (pure function of `(seed, batch_key, attempt, failure_rate)`, key-only routing, pinned by test), but the shard sweep faulted decider row was not run, so the default stays 1 with no performance claim attached.

### Fault-tolerance tax (measured at the baseline; not re-measured since)

At 10% injected batch failures on the l=0.02 / batch=8 grid row (20,000 tasks; `bench/results/published/clean_row/` vs `published/faulted/`):

| workers | clean (t/s) | faulted (t/s) | delta   |
|--------:|------------:|--------------:|---------|
|      16 |     4,911.0 |       3,658.1 | -25.5%  |
|      64 |     4,728.3 |       3,675.2 | -22.3%  |
|     128 |     4,462.7 |       3,026.3 | -32.2%  |
|     256 |     4,551.0 |       2,100.0 | -53.9%  |

The penalty grows with pool size while `agg_submit`'s share *shrinks* (72.2% -> 46.0% at w256): under faults, loop time moves into unaccounted (6.8-10.7% clean -> 34.9-42.1% faulted), which is where the blocking `ray.get(health_check, timeout=5.0)` poison probe in the failure path lives - it freezes the scheduling loop once per failure. batched writes fixed the write path; **this probe is still blocking and the faulted row has not been re-run post-batched-writes** (see Known limitations).

### Standby worker pool: O(1) replacement, paid in memory

Replacement used to block the single-threaded scheduling loop for up to `WORKER_INIT_TIMEOUT_S` x 3 attempts while the replacement loaded its model - minutes at 70B scale, during which completed refs from healthy workers sat unharvested. `--standby N` (default 0 = exactly the old blocking behavior) creates N extra workers from the same factory at startup; they load the model inside the same init barrier as the primaries and then idle. On replacement (hang eviction or failed health check), the coordinator force-kills the bad actor, promotes a standby with a handle swap - no construction, no `ray.get`, no model load - and starts a replacement standby *asynchronously*, polling its readiness ping with `ray.wait(timeout=0)` from the main loop. Refill failures retry up to `STANDBY_REFILL_MAX_ATTEMPTS` (3); after that the pool permanently runs one smaller. A replacement that finds the pool empty falls back to the blocking path and logs it. The promoted worker receives work through the existing dispatch path, so the one-batch-in-flight-per-slot invariant is untouched.

**Memory cost:** each standby holds a full model resident and claims the same Ray resources as a regular worker - budget memory and scheduling capacity for `--workers + --standby` actors. In the two-node run the standby's engine was a second full TinyLlama load on its node (2.05 GiB model weights per engine, per the run log). With `--hf-device` the fractional GPU claims are sized for `--workers` only, so standbys may not schedule on an already-full single device. The summary reports a `standby` block: configured size, promotions, refills started/completed/failed, empty-pool fallbacks, and the ready-pool size over time.

### Per-task retry semantics: bisection by splitting

Whole-batch retry re-ran B-1 healthy generations to recover one failure, and a deterministically bad task condemned all of its batchmates. Now, on any batch failure with retries remaining - transient exception, deterministic exception, or hang eviction - a multi-task batch re-queues as two midpoint halves (task order preserved, no RNG), each inheriting the parent's `retry_count + 1`; a singleton retries whole. Repeated failures binary-search the offender: a poisoned task is isolated to a singleton in ~log2(batch_size) failures while every half that excludes it succeeds. The budget is per task and splitting never resets it - each task terminates within `max_retries`, recorded exactly once. A deterministic *singleton* is terminal immediately: re-running it verbatim cannot change the outcome. Backoff (full jitter, cap 8 s) is drawn independently per enqueued half so halves decorrelate; hang requeues go straight to `pending` without backoff. The summary reports `retry_splits`; `retried` increments once per enqueued half. Fault-injection interaction: halves are *new* decider keys with fresh attempt counters, so a given `--seed` produces a different failure pattern than a pre-split-retry run, but per-task outcomes remain fully reproducible across runs at the same seed (pinned by test).

### Multi-node: two-node kill test, measured recovery

Two Lambda GPU boxes, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on vLLM 0.15.0 (V1 engine), 20,000 tasks at the vLLM default batch 64, `--standby 1`, one busy primary on the victim node; the worker node was killed mid-run from the head. Runbook: `bench/multinode/README.md`; measured artifact: `bench/multinode/MULTINODE_RESULTS.txt` / `multinode_results.json`.

| metric | value |
|--------|-------|
| tasks/s before the kill        | 108.5 (over 96 s) |
| tasks/s during recovery        | 43.2 (over 4 s)   |
| tasks/s after                  | 109.6 (over 86 s) |
| kill -> first batch on the replacement | **4.4 s** (sampler bounds 2.4-4.4 s) |
| completed / submitted          | **20,000 / 20,000 (100.00%)**, 0 duplicate task_ids, 0 terminal-failure rows |
| tasks retried due to the kill  | 64, in 1 retry event |
| rows produced on replacement workers | 9,504 |

Losing a whole node cost one batch (64 tasks) one retry and ~4 s of degraded throughput; the standby promotion ("Worker 0 replaced by standby (pool now 0 ready); refilling in background") restored full throughput and every task reached exactly one terminal state. The background refill had no free GPU to land on after the node died, so the pool ran with zero ready standbys for the rest of the run - expected and non-blocking. The run's placement preconditions (aggregator on the head, busy primary on the victim) were *checked*, not guaranteed - see Known limitations.

## Performance: vLLM batch-size sweep (a batching-mismatch discovery)

The interesting result here is not a speedup - it is that **the harness was starving its own engine.** The HF backend is forward-pass-bound and happy at 4 tasks per actor call, and an early version handed vLLM the same batch size; vLLM's continuous batching only saturates with many concurrent in-flight requests, so the engine idled at 6.0 tasks/s while capable of ~20x that on the same hardware. Making the default batch size per-backend (`DEFAULT_BATCH_SIZE_VLLM = 64`) let the same engine run at 75.1 tasks/s. Nothing got faster; a mismatch between scheduler granularity and engine design got removed.

Measured sweep (Lambda A10 24 GB, `Qwen/Qwen2.5-1.5B`, vLLM backend, 1 worker, 1,000 tasks per row, greedy decoding, max_new_tokens=80; artifact: `bench/results/VLLM_BATCH_SWEEP.md`, which also records that the numbers predate the review-stage fixes and should be re-run before being quoted as current):

| `--batch-size` | Throughput (tasks/s) | Engine calls |
| -------------: | -------------------: | -----------: |
|              4 |                  6.0 |          250 |
|             16 |                 22.8 |           63 |
|             32 |                 44.5 |           32 |
|             64 |                 75.1 |           16 |
|            128 |                117.1 |            8 |
|            256 |                120.1 |            4 |

The default of 64 captures most of the available throughput (12.5x the batch=4 row) without committing all of GPU memory to in-flight KV cache; from 128 to 256 the sweep gains almost nothing (117.1 -> 120.1 tasks/s) for real memory - past saturation, extra batch size buys KV-cache commitment, not speed.

**Read latency columns with care.** `latency_seconds` in run summaries is an estimate - batch wall time / batch size - so it *mechanically shrinks* as batch size grows; any apparent p99 "improvement" down the table is an artifact of the estimator. A task's real time-in-system grows with batch size (it waits for its whole batch). The summary also reports **batch latency** (measured wall time of the batch a task rode in, task-weighted), which is the honest per-task figure; use throughput and batch latency to compare configurations.

## Fault injection and reproducibility

Both backends use greedy decoding (`temperature=0.0` on vLLM; the HF text-generation pipeline defaults to `do_sample=False`). Outputs are empirically stable across runs, but greedy decoding under continuous batching is not *guaranteed* bitwise-deterministic: batch composition changes kernel shapes and reduction orders, and composition varies with retry timing (backoff jitter deliberately uses an unseeded RNG). Failure *decisions* are fully deterministic; wall-clock scheduling is not.

Verified once on GPU (commit `1af667c`): at 200 tasks / batch 16, two
seed-42 runs were identical down to which batch exhausted its retry
budget - 16/16 terminal failures, 96/96 events - and seed 99 diverges. A
first attempt at 50 tasks / batch 64 was invalidated as structurally
unable to exercise the path: at batch 64, 50 tasks is a single batch, so
there was nothing to vary.

Failure injection goes through a shared `FailureDecider` (`--decider-shards` actors behind a facade) that all fault-injecting workers query. The decider tracks per-batch attempt counts, so a decision is a function of `(seed, batch_key, attempt, failure_rate)` and not of which worker happened to receive the batch - the naive per-worker seeded RNG gives reproducible per-worker sequences but not reproducible task-level failures, because Ray routes batches non-deterministically. Sharding cannot change which batches fail: every shard gets the same seed, routing hashes the key only, so each key's attempt counter keeps shard affinity and the failure sequence at any shard count is identical to the single actor's (pinned by a 500-batch N=1-vs-N=4 equality test). With split-retry, halves are new keys with fresh attempt counters, so the reproducibility guarantee is stated - and tested - at the task level: same seed, same per-task outcomes, run after run (`tests/test_fault_injection.py`, including a regression for an earlier bug where Python's per-process-randomized `hash()` defeated the seeded RNG).

Exercise the retry path:

```bash
python main.py --tasks 200 --workers 1 --backend vllm \
    --model Qwen/Qwen2.5-1.5B --batch-size 16 \
    --failure-rate 0.3 --seed 42
```

The kill test above doubles as the current measured fault-tolerance result on real hardware: an injected whole-node loss with 100.00% completion and 0 terminal failures. (Earlier GPU and laptop fault-injection smoke tables were dropped from this README because their numbers have no results artifact in the repo; the determinism guarantee they illustrated is pinned by the test suite instead.)

## Bugs found and fixed

Bugs found while hardening the harness. Each fix has a regression test; the pattern is the same one the Testing section describes: find, explain the mechanism, pin it with a test.

1. **Old actors were never killed on replacement.** `_replace_worker` dropped the handle and constructed a replacement - but a hung or poisoned GPU-claiming actor still holds its `num_gpus` claim, so the replacement can never schedule: every attempt times out at the 120 s health barrier and the slot dies even though the GPU was one kill away from free. Injected failures raise *without* poisoning, so replacement-under-GPU-contention was never exercised on GPU before this. Fix: force-`ray.kill(old, no_restart=True)` before constructing the replacement (this also errors out any evicted in-flight ref). Test: `test_replace_worker_kills_old_actor`.
2. **Hangs incremented the retry budget but never enforced it.** The hang-eviction path re-queued with `retry_count + 1`, but only the exception path *checked* the budget - a batch that deterministically wedged the engine looped forever, one eviction cycle (then a fixed 240 s) per lap. Fix: `_handle_timeouts` enforces `max_retries` and records terminal failures on exhaustion. Test: `test_hang_budget_exhaustion_records_terminal_failures`.
3. **The hooked demo read hook state that was never mutated.** Ray pickles hook objects into the actor; mutations happen on the worker's copies, so the driver's `stop_hook.triggered_by` stayed `None`. Fix: hook-observable state returns via `EvalResult.hook_state`. Tests: `TestHookStateReturnPath`.
4. **Hang detection only ran when the cluster was quiet.** The timeout scan was gated on `ray.wait` returning nothing - under steady completions it was starved exactly when the system was busiest. Fix: the scan runs every loop iteration (it's O(active) dict reads).
5. **Timeout -> poison -> full model reload was disproportionate.** When the HF `StoppingCriteria` deadline fires, the pipeline call returns cleanly - the loaded model is intact - yet the worker marked itself permanently poisoned, costing a model reload per slow batch. Fix: a batch timeout raises `TimeoutError` without poisoning; replacement happens only when a health check actually fails.
6. **`--hf-device 0 --workers N` silently stacked N unaccounted replicas on one GPU.** Fix: the coordinator claims `num_gpus = 1/N` per HF worker on GPU (accounting, not memory isolation) and logs a warning.
7. Smaller items: aggregator writes are fire-and-forget with an end-of-run drain (a blocking `ray.get` per batch stalled the single-threaded scheduler); the aggregator holds its file open instead of reopening per write; the p99 index used `int(0.99n)`, which is the max (p100) at n=100 - now nearest-rank; `mentions_topic` no longer awards near-free credit via stopwords; the standard rubric has one source of truth in `scoring.DEFAULT_CONDITIONS`; a module-level test assert became a real test.

## Design notes

A few choices that aren't obvious from reading the code:

**Work-queue, not work-stealing.** Earlier revisions of this README (and the coordinator docstring) called the scheduler "work-stealing", which it never was: there are no per-worker deques and no worker ever takes work from another. The actual design is a single `pending` deque on the driver plus one in-flight batch per worker slot; a completion, failure, or hang-eviction event frees the slot and the driver hands it the next queued batch. It is pull-based in the sense that work moves only when a slot demands it - nothing is pre-assigned - which is what gives natural load balancing across heterogeneous worker speeds without any stealing protocol.

**Backend Protocol with a runtime hasattr check.** Ray actor handles are remote references, not the actor itself, so `isinstance(handle, EvalBackend)` returns False even when the actor implements the Protocol. The Protocol exists for documentation and unit-test fakes; `validate_backend` does an attribute check at startup so a missing method fails before the first batch.

**Hung detection is per-ref, not per-worker.** A worker can issue many refs over its lifetime. Tracking timeouts per worker means a slow batch looks like a dead worker, which evicts workers that are actually fine. Per-ref means a healthy worker that finishes one batch and starts another resets cleanly. Detection stays batch-granularity either way: a hang condemns the whole in-flight batch to the retry path (halved per split-retry) - there is no per-task attribution inside a hung ref.

**Health check on every failure path.** When a batch fails, the coordinator health-checks the worker before deciding to retry. Without this, a poisoned worker keeps accepting batches that fail instantly, and the retry budget gets burned. What is actually checked is per-backend:

- **HF**: `health_check` returns `not self._poisoned` - a flag reserved for unrecoverable states; batch timeouts don't set it.
- **vLLM**: `health_check` returns False if (a) the last `evaluate_batch`/`evaluate_with_hooks` raised a CUDA or engine-fatal error and no batch has succeeded since (`_last_error`; matched classes: `torch.cuda.OutOfMemoryError`, vLLM's `EngineDeadError`/`EngineGenerateError` (V1) or `AsyncEngineDeadError` (V0), plus `RuntimeError`s carrying CUDA fingerprints like "illegal memory access" or "device-side assert" - `TimeoutError` never counts, timeouts don't poison), or (b) the engine's `errored` property is truthy or raises, or (c) the engine's `check_health()` - sync or async, duck-typed - raises or fails to answer within `HEALTH_CHECK_TIMEOUT_S`. In the pinned vllm==0.15.0, `AsyncLLMEngine` is an alias of the V1 `AsyncLLM`, which exposes both `errored` and async `check_health()`; the duck typing keeps V0-shaped engines and engines exposing neither (assumed healthy) working. Before this, the vLLM `health_check` returned True unconditionally, so a dead engine burned the whole retry budget and the poison path never fired for that backend.

The integration test `test_poisoned_worker_replaced_on_first_failure` covers the coordinator path; `tests/test_worker_health.py` covers the per-backend checks. The probe itself is a blocking `ray.get(..., timeout=5.0)` on the driver thread - the measured fault-tolerance tax above is its cost, and making it non-blocking is still open.

**Two timeout layers, one budget.** The worker enforces a per-batch timeout via `StoppingCriteria` (HF) or `asyncio.wait_for` (vLLM) and raises on overrun. The coordinator enforces a separate per-ref hang threshold for the case where the actor itself becomes unreachable (process death, GC stall, network). The worker-side timeout always fires first on a live worker; the coordinator threshold only exists to catch workers whose own timeout machinery is dead, so it is derived from `--task-timeout` rather than fixed (a fixed 240 s evicted healthy workers whenever `--task-timeout` exceeded it):

```
hang_threshold_s = max(HANG_THRESHOLD_MIN_S,
                       HANG_MULTIPLIER * task_timeout + HANG_MARGIN_S)
```

The derived value is logged at startup and reported in the summary as `hang_threshold_s`. Different failure modes, different thresholds - and both paths consume and enforce the same retry budget.

**Sharded aggregator behind a plain-Python facade, writes as fire-and-forget.** Concurrent JSONL writing stays "call a method on an actor" - Ray serializes actor methods for free - but a driver-side `ShardedAggregator` facade fronts N `ResultsAggregator` actors, routed by `crc32(task_id) % N` (deliberately not the builtin `hash()`, which is salted per process). `record_batch` issues one actor call per shard touched per completed batch, never one per result. The coordinator doesn't block on writes; refs are drained at end of run, then `finalize()` concatenates the per-shard JSONLs into `--output` (every result exactly once, grouped by shard in per-shard completion order - **not** globally ordered for N > 1) and `get_summary()` merges: counters and sums added, min/max across shards, condition scores pooled, latency reservoirs concatenated. Each shard keeps a uniform latency reservoir capped at 100,000 samples, so merged latency *percentiles* are approximate once a shard exceeds that (means and counts stay exact). N=1 is a strict passthrough, behaviorally identical to the pre-shard single actor (pinned by test), and the measured best default (Scaling, curve 2).

**Shared `FailureDecider` actor for deterministic injection**, sharded the same way (`--decider-shards`) with a facade that duck-types the actor-handle surface (`should_fail.remote(...)`) - see Fault injection and reproducibility for the guarantees.

**`AsyncLLMEngine` for hooks.** Synchronous `LLM.generate` returns when all prompts are done with no per-token callback, and there can't be one without breaking continuous batching. `AsyncLLMEngine.generate` returns an async iterator of `RequestOutput`s, one per scheduling step, so hooks can observe one specific request while the engine batches others; `engine.abort(request_id)` is the clean way to stop a single in-flight request. This targets the pinned vLLM version's API surface; re-verify the import path and the cumulative-output assumption behind the delta diffing on any vLLM upgrade.

**Tuples in `pending`.** Retries re-append to `pending`, and a re-queued batch needs to remember how many times it's been tried. Keeping the count in the tuple means the budget is enforced on every retry path, including hangs.

## Constants

All in `coordinator.py` unless noted; verify against source, which is authoritative.

| Constant | Value | What it does |
| -------- | ----: | ------------ |
| `WORKER_INIT_TIMEOUT_S` | 120.0 | Init barrier for the whole pool (primaries + standbys); also the per-attempt `ray.get` timeout in blocking replacement and the async-refill readiness deadline. |
| `HANG_THRESHOLD_MIN_S`  | 120.0 | Floor of the derived hang threshold; tiny `--task-timeout` values must not make eviction trigger-happy. |
| `HANG_MULTIPLIER`       | 2.0   | A batch may run its full budget and spend nearly as long again raising/serializing. |
| `HANG_MARGIN_S`         | 30.0  | Actor-mailbox and scheduling latency, which does not shrink with small timeouts. |
| `RETRY_POLL_INTERVAL`   | 1.0   | `ray.wait` timeout when nothing is completing. |
| `STANDBY_REFILL_MAX_ATTEMPTS` | 3 | Consecutive async-refill failures before the standby pool permanently runs one smaller. |
| `DEFAULT_BATCH_SIZE_HF` | 4     | HF is forward-pass-bound at small batches. |
| `DEFAULT_BATCH_SIZE_VLLM` | 64  | Continuous batching needs many in-flight requests (see Performance). |
| `backoff_seconds` | min(8.0, 0.5 * 2^retry_count), full jitter | Retry backoff; unseeded RNG by design. |
| `max_retries` | 2 | Coordinator constructor default; `main.py` passes 2. Not a CLI flag. |
| `DEFAULT_TASK_TIMEOUT` (`worker.py`) | 60.0 | Worker-side per-batch timeout default. |
| `HEALTH_CHECK_TIMEOUT_S` (`worker.py`) | 4.0 | Deadline for the vLLM engine's own `check_health()`. |
| `KNEE_RATIO` (`bench/plot_saturation.py`) | 0.8 | Knee = first point with achieved < 0.8 x offered. |

## CLI

All flags in `main.py`:

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--tasks` | 20 | Number of eval tasks. |
| `--workers` | 3 | Number of parallel EvalWorker actors. |
| `--failure-rate` | 0.0 | Fraction of batches to fail artificially; > 0 selects the fault-injecting worker subclass. |
| `--seed` | 0 | Fault-injection seed; same seed + failure-rate reproduces per-task outcomes. |
| `--model` | `distilgpt2` | HuggingFace model name. |
| `--backend` | `hf` | `hf` or `vllm`. |
| `--hf-device` | -1 | HF device: -1 CPU, 0+ CUDA. With N workers on a GPU each claims `num_gpus=1/N` - accounting, not isolation. |
| `--batch-size` | per-backend | Tasks per actor call; defaults to 4 (HF) / 64 (vLLM). |
| `--task-timeout` | 60.0 | Per-**batch** timeout enforced inside the worker; size it for `batch_size * worst-case task time`. Also the input to the derived hang threshold. |
| `--dry-run` | off | 3 tasks, 2 workers, quick verification. |
| `--hook` | off | Run the mid-generation intervention demo after the main run. |
| `--output` | `results/results.jsonl` | JSONL results path. With `--aggregator-shards` > 1, per-shard files are written as `<stem>.shardK.jsonl` and concatenated here at end of run (not globally ordered). |
| `--aggregator-shards` | 1 | `ResultsAggregator` shard actors, keyed by a stable hash of `task_id`. 1 is the measured best (Scaling, `bench/results/SHARDED.md`). |
| `--standby` | 0 | Pre-loaded standby workers for O(1) replacement; each costs a full resident model plus a regular worker's Ray resource claim - budget for `--workers + --standby` actors. |
| `--decider-shards` | 1 | `FailureDecider` shard actors; only used when `--failure-rate` > 0. Cannot change which batches fail. |

`tensor_parallel_size` is a coordinator constructor argument (default 1), not a CLI flag.

## Layout

```
.
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point
├── coordinator.py               # work-queue scheduler, retry, hang detection
├── worker.py                    # HFWorker and VLLMWorker actors
├── fault_injection.py           # FailureDecider (+ sharded facade) + fault-injecting workers
├── aggregator.py                # ResultsAggregator shard actor + ShardedAggregator facade
├── scoring.py                   # rubric scorer + condition checkers
├── hooks.py                     # LoggingHook, EarlyStoppingHook
├── types_.py                    # dataclasses, enums, Protocols
├── metrics.py                   # Metrics Protocol + NullMetrics (no-op default)
├── utils.py                     # pure helpers (make_batches)
├── results/                     # JSONL output
├── bench/
│   ├── fake_worker.py           # FakeLatencyWorker (sleeps instead of inferring)
│   ├── recording.py             # RecordingMetrics + analysis helpers
│   ├── saturation.py            # single-point saturation run
│   ├── sweep_saturation.py      # latency x workers x batch grid
│   ├── plot_saturation.py       # achieved-vs-offered curves + knee (+ --compare overlays)
│   ├── multinode/               # two-node kill-test runbook, scripts, measure.py, results
│   └── results/                 # BASELINE.md, SHARDED.md, VLLM_BATCH_SWEEP.md, published/ reports
├── bench-results-sharded/       # shard-sweep raw reports (a1 x3 reps, a4, a8)
└── tests/
    ├── test_coordinator.py      # helpers + end-to-end against a fake backend
    ├── test_fault_injection.py  # determinism, FailureDecider (+ shard equivalence)
    ├── test_aggregator.py       # shard routing, merge, finalize, N=1 passthrough
    ├── test_worker_health.py    # per-backend health checks
    ├── test_metrics_null.py     # NullMetrics equivalence
    ├── test_bench.py            # bench harness (incl. @slow real-Ray run)
    ├── test_plot_saturation.py  # knee/overlay logic
    ├── test_multinode_measure.py# measure.py parsing
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

# Keep 1 pre-loaded standby worker for O(1) replacement of hung or
# poisoned workers (costs one extra resident model + worker resources)
python main.py --tasks 200 --workers 4 --backend hf --standby 1
```

Saturation bench (no GPUs; see Scaling for what it measures):

```bash
python -m bench.saturation --workers 128 --latency-s 0.02 --batch-size 8 \
    --tasks 200000 --fail-rate 0.0 --out bench/results/sat_w128_l020_b8.json
python -m bench.sweep_saturation --max-workers 128
python -m bench.plot_saturation                            # curves + knees
python -m bench.plot_saturation --compare a1/ a4/ a8/      # shard overlay
```

Two-node kill test: follow `bench/multinode/README.md` (setup scripts, firewall table, placement checks, `kill_node.sh`, `measure.py`).

## Testing

```bash
pytest tests/            # fast suite (slow tests deselected by default)
pytest tests/ -m slow    # real-Ray bench smoke test (~15s incl. cluster boot)
```

The `scoring`, `hooks`, and `utils` tests are pure Python and finish in well under a second. The coordinator tests cover both helpers and the full `run()` loop end-to-end. The end-to-end tests stub `ray.wait`, `ray.get`, and `ray.kill` and use an in-memory fake backend that satisfies `EvalBackend` structurally, so they exercise the real scheduler logic without a Ray cluster, GPU, or model load.

Real bugs surfaced by tests, from the original development:

1. A deferred retry could promote back to `pending` after every active worker had gone idle, at which point nothing dispatched the pending work and the run loop exited early. Fixed by including `pending` in the main loop condition and adding a dispatch-to-idle step after promotion.
2. `_handle_failure` only health-checked the worker on the terminal-failure branch. While retries remained, a poisoned worker kept receiving submissions that failed immediately, burning the retry budget. Fixed by running the health check on every failure path, before the retry decision.
3. The first attempt at deterministic fault injection seeded a per-worker RNG with `(seed, worker_id)` - reproducible *per-worker* sequences, not reproducible *task-level* failures, because the scheduler routes batches non-deterministically. Replaced with the `FailureDecider` actor pattern.

Plus everything in **Bugs found and fixed** - most notably the unkilled-actor GPU leak and the unenforced hang budget, both of which lived in code paths the happy-path benchmarks never exercised. All have regression tests.

## Known limitations

These are real and worth knowing about; none are showstoppers.

- **The coordinator is a single process, and its driver loop is now the first-saturating component.** With batched writes there is no actor backed up behind it; the ceiling is per-completion-event work on the one thread that cannot be sharded (dispatch + result submission roughly evenly split at depth), at ~2,568 / 17,785 / 65,625 tasks/s for batch 1 / 8 / 64 on the 32-vCPU bench box. Raising it means fewer driver-side calls per event (e.g. coalescing `record_batch` submissions across batches) or moving fan-out off the driver thread - not more actors behind it.
- **The failure-path poison probe still blocks the driver.** `_check_and_replace_if_poisoned` does a blocking `ray.get(health_check, timeout=5.0)` once per failure; the baseline measurement put the resulting fault tax at -25.5% peak throughput and -53.9% at 256 workers under 10% injected failures, and the faulted row has not been re-measured since batched writes, so the current cost is unknown but the blocking call is still there.
- **Split-retry is bisection, not attribution; true per-task retry is unbuilt.** A failed batch re-queues as halves, isolating one bad task in ~log2(batch) failures - but recovering it in a 64-batch still re-executes ~63 healthy task-runs across the shrinking retries, and the halved retry batches under-drive vLLM's continuous batching on the retry path. Per-task failure attribution from the worker would remove both costs. Splitting also changes which FailureDecider keys exist, so injected terminal-failure counts at a given `--seed` are not comparable between pre- and post-split-retry harnesses (each is individually reproducible).
- **Hang detection is batch-granularity.** The per-ref threshold can only condemn the whole in-flight batch; there is no visibility into which task inside a hung ref wedged the engine, so a single pathological task costs its batchmates retries until bisection isolates it.
- **Aggregator sharding is measured, slower, and it costs global order.** N=4/8 are strictly worse than 1 with batched writes (batch=8 ceiling 17,785.5 -> 9,482.4 -> 6,406.6 tasks/s); the default stays 1 as the measured best. With N > 1 the concatenated `results.jsonl` is grouped by shard, not globally ordered (re-sort by `task_id` if you need determinism), and merged latency percentiles are approximate once any shard exceeds its 100,000-sample reservoir. The `FailureDecider` sharding is provably outcome-identical but *unmeasured* at any count, and its per-batch attempt maps are unbounded per shard - long fault-heavy runs still want eviction of keys whose batches reached a terminal state. That piece remains unbuilt.
- **The standby pool trades memory for replacement latency, and only while it holds a ready worker.** Each standby is a full resident model plus a regular worker's Ray resource claim (budget `--workers + --standby`; with fractional HF claims sized for `--workers` only, standbys may not schedule on a full single GPU - and `--workers` equal to the cluster's GPU total plus `--standby 1` cannot pass the init barrier at all, as multinode runbook documents). A failure burst larger than N falls back to the blocking path; a refill that fails `STANDBY_REFILL_MAX_ATTEMPTS` times permanently shrinks the pool; refills restock the *pool*, not dead *slots*, so a slot lost while the pool was empty stays lost. The idle pool is not health-monitored: a standby that dies quietly is only discovered when promoted.
- **Actor placement is uncontrolled - the multinode failure mode.** The harness sets no scheduling strategy, so Ray's default packing decides where the aggregator and standbys land. If the aggregator is on a node that dies, its fire-and-forget `record_batch` refs are silently lost, the end-of-run drain raises, and the partial JSONL is stranded on the dead node - there is no aggregator fault tolerance. If the standby is on the victim node, promotion swaps in a dead handle (promotion does no health check), and once the pool is empty the blocking fallback freezes the driver for up to 3 x `WORKER_INIT_TIMEOUT_S` per lost slot. multinode run verified placement by inspection before the kill; a placement-group / node-affinity policy is a future stage.
- **`tokens_generated` is left at 0 on the HF batch path.** Reported accurately on the hooked path and on vLLM batch; re-tokenizing the response string isn't guaranteed to match the generated count, so the field stays unset rather than wrong. Plumbing token IDs out of the HF pipeline is the proper fix.
- **`tensor_parallel_size > 1` is wired through but only verified at `tp=1`.** The coordinator passes `tp` to creation and replacement and the `num_gpus=tp` claim follows, but a real multi-GPU-per-worker run also wants `STRICT_PACK` placement groups; that piece isn't built. Single-GPU, CPU, and two-node single-GPU-per-worker paths (multinode) are exercised on real hardware; `tp>1` is not.
- **Failed-batch results are only written to JSONL at terminal outcome.** Hang exhaustion and all-workers-dead are recorded, but a run that aborts mid-retry (e.g. SIGKILL) leaves no record of in-flight batches. Writing on first failure with a `pending_retry` flag and updating on terminal outcome would close the gap.
