"""Ingest ray-eval-harness results into Docent (docs.transluce.org).

Maps the aggregator's JSONL EvalResult records (one per completed task,
schema: types_.EvalResult.to_dict) into Docent AgentRun objects and
uploads them as one collection per harness run.

The JSONL record does not carry the prompt (only task_id); prompts live
on EvalTask. main.make_tasks() is deterministic, so we rebuild the task
list the same way the run did and join on task_id. If your task source
changes, pass --tasks-json instead.

Usage:
    python docent_adapter.py --results results/results.jsonl --dry-run
    DOCENT_API_KEY=... python docent_adapter.py \
        --results 'results/results*.jsonl' \
        --collection-name "reh smoke run seed42" \
        --run-meta seed=42 backend=hf model=EleutherAI/pythia-70m

Verified against docent==0.1.80.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Iterator

REQUIRED_FIELDS = ("task_id", "score", "response", "failed", "condition_scores")


def iter_records(pattern: str) -> Iterator[dict[str, Any]]:
    """Yield records from every JSONL matching the glob (per-shard files
    from ShardedAggregator all match 'results/results*.jsonl')."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no files match {pattern!r}")
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{lineno}: bad JSON ({e})")
                missing = [f for f in REQUIRED_FIELDS if f not in rec]
                if missing:
                    sys.exit(f"{path}:{lineno}: record missing {missing}")
                rec["_source"] = f"{path}:{lineno}"
                yield rec


def load_prompts(tasks_json: str | None, n_hint: int) -> dict[str, dict[str, Any]]:
    """task_id -> {prompt, expected_answer, ...}. Prefer an explicit task
    dump; fall back to regenerating via main.make_tasks (deterministic)."""
    if tasks_json:
        with open(tasks_json, encoding="utf-8") as fh:
            tasks = json.load(fh)
        return {t["task_id"]: t for t in tasks}
    try:
        from main import make_tasks  # same module the run used
    except Exception as e:  # torch/ray imports may fail on a bare box
        print(f"warning: cannot import main.make_tasks ({e}); "
              "prompts will be marked unavailable", file=sys.stderr)
        return {}
    return {
        t.task_id: {"prompt": t.prompt, "expected_answer": t.expected_answer}
        for t in make_tasks(max(n_hint, 1))
    }


def to_agent_run(rec: dict[str, Any], task: dict[str, Any], run_meta: dict[str, str]):
    """One EvalResult record -> one Docent AgentRun.

    Transcript = the user prompt + the model's response (error text stands
    in for the response on failed records, flagged in metadata).
    Metadata carries everything Docent's DQL/rubrics can filter on:
    harness scores under 'scores' (Docent's convention), plus retry/fault
    machinery fields the harness alone can't cross-analyze.
    """
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message

    prompt = task.get("prompt") or "(prompt unavailable: task source not provided)"
    response = rec["response"] if not rec["failed"] else (
        rec.get("error") or "(failed: no response captured)"
    )
    messages = [
        parse_chat_message({"role": "user", "content": prompt}),
        parse_chat_message({"role": "assistant", "content": response}),
    ]

    scores: dict[str, float | bool] = {
        "rubric_total": float(rec["score"]),
        "failed": bool(rec["failed"]),
        **{f"cond_{k}": float(v) for k, v in rec["condition_scores"].items()},
    }
    metadata: dict[str, Any] = {
        "task_id": rec["task_id"],
        "external_id": rec["task_id"],        # our join key; Docent owns .id
        "scores": scores,
        "expected_answer": task.get("expected_answer"),
        "failure_kind": rec.get("failure_kind"),      # TRANSIENT/DETERMINISTIC/None
        "hooked": rec.get("hooked", False),
        "hook_state": rec.get("hook_state") or {},
        "stopped_early": rec.get("stopped_early", False),
        "worker_id": rec.get("worker_id"),
        "latency_seconds": rec.get("latency_seconds"),
        "batch_latency_seconds": rec.get("batch_latency_seconds"),
        "tokens_generated": rec.get("tokens_generated", 0),
        "source": rec["_source"],
        **run_meta,                            # seed, backend, model, run label
    }
    return AgentRun(
        name=rec["task_id"],
        transcripts=[Transcript(messages=messages)],
        metadata=metadata,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/results.jsonl",
                    help="glob for aggregator JSONL shard(s)")
    ap.add_argument("--tasks-json", default=None,
                    help="optional JSON dump of EvalTasks (task_id, prompt, ...)")
    ap.add_argument("--collection-name", default="ray-eval-harness run")
    ap.add_argument("--collection-desc", default="ingested by docent_adapter.py")
    ap.add_argument("--run-meta", nargs="*", default=[], metavar="K=V",
                    help="run-level metadata stamped on every AgentRun")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="AgentRuns per add_agent_runs call")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first mapped run and counts; no upload")
    args = ap.parse_args()

    run_meta = dict(kv.split("=", 1) for kv in args.run_meta)
    records = list(iter_records(args.results))
    prompts = load_prompts(args.tasks_json, n_hint=len(records))
    runs = [to_agent_run(r, prompts.get(r["task_id"], {}), run_meta)
            for r in records]

    n_fail = sum(r.metadata["scores"]["failed"] for r in runs)
    n_hook = sum(bool(r.metadata["hooked"]) for r in runs)
    print(f"mapped {len(runs)} agent runs "
          f"({n_fail} failed, {n_hook} hooked) from {args.results}")

    if args.dry_run:
        print("--- first mapped AgentRun ---")
        print(runs[0].model_dump_json(indent=2, exclude={"id"})[:2000])
        return

    from docent import Docent
    client = Docent(
        api_key=os.environ["DOCENT_API_KEY"],
        # self-hosting? uncomment:
        # server_url="http://localhost:8889", web_url="http://localhost:3001",
    )
    collection_id = client.create_collection(
        name=args.collection_name, description=args.collection_desc,
    )
    for i in range(0, len(runs), args.batch_size):
        client.add_agent_runs(collection_id, runs[i:i + args.batch_size])
        print(f"uploaded {min(i + args.batch_size, len(runs))}/{len(runs)}")
    print(f"done: collection {collection_id}")


if __name__ == "__main__":
    main()
