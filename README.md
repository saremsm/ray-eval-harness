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

# Larger run on CPU
python main.py --tasks 200 --workers 4

# vLLM run (requires GPU + a vLLM-compatible model)
python main.py --tasks 1000 --workers 1 --backend vllm \
    --model Qwen/Qwen2.5-1.5B
```

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
