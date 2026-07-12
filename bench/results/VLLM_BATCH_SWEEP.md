# vLLM batch-size throughput sweep (single worker)

Results artifact extracted verbatim from the earlier README's
"Performance" section, so these numbers live in a results file rather
than only in prose that gets rewritten. No re-measurement was performed
for this extraction.

## Environment and provenance

Cloud GPU. Lambda A10 (24GB VRAM), `Qwen/Qwen2.5-1.5B`, vLLM backend,
1 worker, 1000 tasks per row, greedy decoding, max_new_tokens=80.
Numbers were measured on the pre-review code; none of the review fixes
sit on the vLLM batch hot path (the write path became less blocking, if
anything), but re-run before quoting them as current.

## Results

| `--batch-size` | Throughput (tasks/s) | Engine calls |
| -------------: | -------------------: | -----------: |
|              4 |                  6.0 |          250 |
|             16 |                 22.8 |           63 |
|             32 |                 44.5 |           32 |
|             64 |                 75.1 |           16 |
|            128 |                117.1 |            8 |
|            256 |                120.1 |            4 |

Throughput climbs steeply with batch size because continuous batching
only saturates when many requests are in flight; the default of 64
captures most of the available throughput (12.7x the batch=4 baseline)
without committing all of GPU memory to in-flight KV cache.

## Latency caveat

`latency_seconds` in the run summaries is an estimate - batch wall time
/ batch size - so it *mechanically shrinks* as batch size grows; any
apparent p99 "improvement" down the table is an artifact of the
estimator, not a latency win. A task's real time-in-system grows with
batch size (it waits for its whole batch). The summary also reports
**batch latency** (measured wall time of the batch a task rode in,
task-weighted), which is the honest per-task figure; use throughput and
batch latency to compare configurations, not the estimated per-task
column.
