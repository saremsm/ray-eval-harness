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

## Testing

```bash
pytest tests/
```

The `scoring` and `utils` tests are pure Python and finish in well under a second.
