# Distributed LLM Eval Harness

Ray-based eval harness for LLMs. Schedules batches of completion tasks across a pool of persistent worker actors with a work-stealing coordinator, scores responses against a weighted rubric, and streams results to JSONL through a single aggregator actor.

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
        | HFWorker  |         |VLLMWorker |        | HFWorker  |
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

The coordinator never touches worker state directly. All communication goes through Ray's actor mailbox, which serializes calls per-actor.

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

Cloud GPU. Lambda A10 (24GB VRAM), `Qwen/Qwen2.5-1.5B`, vLLM backend, 1 worker, 1000 tasks per row, greedy decoding, max_new_tokens=80.

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

**Laptop CPU baseline.** Lenovo IdeaPad 82FE, Intel Core i5-1135G7, `distilgpt2`, HF backend, 4 workers, 50 tasks. Included for the smoke-test fault tolerance comparison, not for absolute throughput. `distilgpt2` scores poorly on the trivia rubric because it's an 82M-parameter model trained mostly on web text; that's the model, not the harness.

| Scenario (50 tasks, 4 workers)      | Outcome                        |
| ----------------------------------- | ------------------------------ |
| Clean run                           | 50/50 succeeded                |
| `--failure-rate 0.3 --seed 42`      | 50/50 after retries            |
| One worker killed mid-run           | 50/50; run continues at n-1    |
| Kill + `--failure-rate 0.3`         | 50/50; replacement + retries   |

## Reproducibility

Both backends use greedy decoding (`temperature=0.0` on vLLM; the HF text-generation pipeline defaults to `do_sample=False`). Outputs are **empirically stable across runs** (table below), but greedy decoding under continuous batching is not *guaranteed* bitwise-deterministic: batch composition changes kernel shapes and reduction orders, and composition varies with retry timing.

Failure injection goes through a single shared `FailureDecider` actor that all fault-injecting workers query. The decider tracks per-batch attempt counts globally, so a decision is a function of `(seed, batch_content, attempt_number)` and not of which worker happened to receive the batch.

Verified on GPU: 200 tasks, 13 batches at batch_size=16, `--failure-rate 0.3 --seed 42`. Two independent runs on a fresh actor pool produced identical outcomes:

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

The `scoring` and `utils` tests are pure Python and finish in well under a second.
