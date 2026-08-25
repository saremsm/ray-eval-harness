# Docent ingestion for ray-eval-harness
docent_adapter.py maps the aggregator's per-task JSONL (EvalResult.to_dict) into
Docent AgentRuns: transcript = prompt + response; rubric scores under
metadata["scores"]; retry/fault machinery (failure_kind, worker_id, latencies)
in metadata. Prompts aren't in the JSONL; pass --tasks-json (dump via
main.make_tasks) or let the adapter rebuild the join. Requires Python >=3.11
for the docent SDK; DOCENT_API_KEY exported.

Run:
  python docent_adapter.py --results results/clean.jsonl --tasks-json tasks.json \
    --collection-name "<name>" --run-meta seed=42 backend=hf

Uploaded 2026-08-24: "REH clean seed42" and "REH faulted seed42"
(40 tasks each; faulted run: 20% injected faults, 39/40 succeeded,
1 terminal after split-retry). Analysis notes to follow.
