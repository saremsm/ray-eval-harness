"""Report recovery metrics for the two-node kill test. Usage: python
bench/multinode/measure.py [--logs bench/multinode/logs] [--kill-epoch E]
[--grace 30] [--json out.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

LOG_LINE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) (INFO|WARNING|ERROR) (.*)$")
RETRY_EXC = re.compile(r"Scheduling retry \d+/\d+ for (\d+) tasks")
RETRY_HANG = re.compile(r"Re-queued (\d+) tasks as")
FAILURE_MARKS = (
    "Batch failed on worker",
    "ref outstanding for",
)
PROMOTION_MARK = "replaced by standby"
READY_MARK = "workers ready and validated"


#
def parse_log(log_path: Path, t0_epoch: float) -> list[tuple[float, str, str]]:
    """(epoch, level, message) per stamped line. %H:%M:%S has no date: anchor on
    t0_epoch's calendar day, and roll forward a day when a timestamp lands well
    before t0 (midnight crossing)."""
    base = dt.datetime.fromtimestamp(t0_epoch, dt.timezone.utc)
    events = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOG_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            hh, mm, ss, level, msg = m.groups()
            stamp = base.replace(
                hour=int(hh), minute=int(mm), second=int(ss), microsecond=0
            )
            epoch = stamp.timestamp()
            if epoch < t0_epoch - 300:
                epoch += 86400.0
            events.append((epoch, level, msg))
    return events


def parse_progress(csv_path: Path) -> list[tuple[float, int, int]]:
    samples = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or (i == 0 and line.startswith("epoch")):
                continue
            e, t, r = line.split(",")
            samples.append((float(e), int(t), int(r)))
    return samples


def parse_results(jsonl_path: Path) -> dict:
    ids: dict[str, int] = {}
    failed = 0
    per_worker: dict[int, int] = {}
    n = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            ids[row["task_id"]] = ids.get(row["task_id"], 0) + 1
            if row.get("failed"):
                failed += 1
            wid = row.get("worker_id")
            if isinstance(wid, int):
                per_worker[wid] = per_worker.get(wid, 0) + 1
    dups = {k: v for k, v in ids.items() if v > 1}
    return {
        "rows": n,
        "unique": len(ids),
        "duplicates": dups,
        "failed_rows": failed,
        "per_worker": per_worker,
    }


#
def total_at(samples: list[tuple[float, int, int]], epoch: float) -> int:
    """Completed-task count at the last sample <= epoch (0 before any)."""
    best = 0
    for e, t, _ in samples:
        if e <= epoch:
            best = t
        else:
            break
    return best


def rate(samples, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    return (total_at(samples, end) - total_at(samples, start)) / (end - start)


def find_recovery(samples, kill_epoch: float) -> tuple[float | None, float | None]:
    """(upper_bound_epoch, lower_bound_epoch) of the first sample after the kill
    with replacement > 0, or (None, None) if none."""
    prev_e = None
    for e, _t, r in samples:
        if e < kill_epoch:
            prev_e = e
            continue
        if r > 0:
            return e, prev_e
        prev_e = e
    return None, None


def infer_kill_epoch(events) -> float | None:
    for e, level, msg in events:
        if level == "ERROR" and any(mark in msg for mark in FAILURE_MARKS):
            return e
    return None


def retried_tasks_in_window(events, start: float, end: float) -> tuple[int, int]:
    """(task_count_sum, event_count) for retry log lines in [start, end]."""
    total = 0
    n_events = 0
    for e, _level, msg in events:
        if not (start <= e <= end):
            continue
        for rx in (RETRY_EXC, RETRY_HANG):
            m = rx.search(msg)
            if m:
                total += int(m.group(1))
                n_events += 1
                break
    return total, n_events


#
def build_report(meta, events, samples, results, kill_epoch, kill_source,
                 grace: float) -> dict:
    ready = next((e for e, _l, m in events if READY_MARK in m), None)
    promotion = next((e for e, _l, m in events if PROMOTION_MARK in m), None)
    end_epoch = samples[-1][0] if samples else None

    rec_hi, rec_lo = (find_recovery(samples, kill_epoch)
                      if kill_epoch is not None else (None, None))
    recovery = rec_hi

    phases = {}
    if kill_epoch is not None and samples:
        before_start = ready if ready is not None else samples[0][0]
        phases["before"] = {
            "window_s": [before_start, kill_epoch],
            "tasks_per_s": rate(samples, before_start, kill_epoch),
        }
        during_end = recovery if recovery is not None else min(
            kill_epoch + 60.0, end_epoch
        )
        phases["during"] = {
            "window_s": [kill_epoch, during_end],
            "tasks_per_s": rate(samples, kill_epoch, during_end),
        }
        phases["after"] = {
            "window_s": [during_end, end_epoch],
            "tasks_per_s": rate(samples, during_end, end_epoch),
        }

    retry_end = (recovery if recovery is not None
                 else (kill_epoch + 180.0 if kill_epoch is not None else 0))
    retried, retry_events = (
        retried_tasks_in_window(events, kill_epoch - 2.0, retry_end + grace)
        if kill_epoch is not None else (0, 0)
    )

    submitted = meta["tasks"]
    completed = results["unique"]
    replacement_rows = sum(
        c for w, c in results["per_worker"].items() if w >= meta["workers"]
    )

    return {
        "kill_epoch": kill_epoch,
        "kill_epoch_source": kill_source,
        "workers_ready_epoch": ready,
        "promotion_log_epoch": promotion,
        "recovery": {
            "first_replacement_sample_epoch": recovery,
            "bounds_from_kill_s": (
                [None if rec_lo is None else max(0.0, rec_lo - kill_epoch),
                 rec_hi - kill_epoch]
                if recovery is not None and kill_epoch is not None else None
            ),
            "kill_to_first_replacement_batch_s": (
                recovery - kill_epoch
                if recovery is not None and kill_epoch is not None else None
            ),
        },
        "phases": phases,
        "retried_due_to_kill": {
            "tasks": retried,
            "retry_events": retry_events,
            "note": "sum over retry log events; a task re-queued twice counts twice",
        },
        "accounting": {
            "submitted": submitted,
            "completed_unique": completed,
            "completed_pct": 100.0 * completed / submitted if submitted else 0.0,
            "rows": results["rows"],
            "duplicate_task_ids": len(results["duplicates"]),
            "failed_rows": results["failed_rows"],
            "rows_on_replacement_workers": replacement_rows,
            "pass_100pct": completed == submitted and not results["duplicates"],
        },
    }


def print_report(r: dict) -> None:
    def fmt(x, suffix=""):
        return "n/a" if x is None else f"{x:.1f}{suffix}"

    print("=" * 62)
    print("  Two-node kill test")
    print("=" * 62)
    print(f"  kill epoch            {r['kill_epoch']}  ({r['kill_epoch_source']})")
    for name in ("before", "during", "after"):
        p = r["phases"].get(name)
        if p:
            dur = p["window_s"][1] - p["window_s"][0]
            print(f"  tasks/s {name:<8}      {p['tasks_per_s']:.1f}"
                  f"  (over {dur:.0f}s)")
    rec = r["recovery"]
    print(f"  kill -> first replacement batch: "
          f"{fmt(rec['kill_to_first_replacement_batch_s'], 's')}"
          + (f"  (bounds {rec['bounds_from_kill_s']})"
             if rec["bounds_from_kill_s"] else ""))
    if rec["first_replacement_sample_epoch"] is None:
        print("  WARNING: no completions attributed to a replacement worker."
              "\n           The kill likely took the standby or an idle actor"
              "\n           (README, Known failure 2). Re-roll for the demo.")
    acc = r["accounting"]
    print(f"  completed / submitted  {acc['completed_unique']} / "
          f"{acc['submitted']}  ({acc['completed_pct']:.2f}%)"
          f"   [{'PASS' if acc['pass_100pct'] else 'FAIL'}]")
    if acc["duplicate_task_ids"]:
        print(f"  DUPLICATE task_ids: {acc['duplicate_task_ids']} "
              f"(violates one-terminal-state-per-task)")
    if acc["failed_rows"]:
        print(f"  terminal FAILURE rows: {acc['failed_rows']} "
              f"(completed != succeeded; inspect results.jsonl)")
    print(f"  retried due to kill    {r['retried_due_to_kill']['tasks']} tasks "
          f"across {r['retried_due_to_kill']['retry_events']} retry events")
    print(f"  rows on replacements   {acc['rows_on_replacement_workers']}")
    print("=" * 62)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default="bench/multinode/logs")
    ap.add_argument("--kill-epoch", type=float, default=None)
    ap.add_argument("--grace", type=float, default=30.0,
                    help="seconds past recovery to still attribute retries to the kill")
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    args = ap.parse_args()

    logs = Path(args.logs)
    meta = json.loads((logs / "run_meta.json").read_text())
    events = parse_log(logs / "run.log", meta["t0_epoch"])
    samples = parse_progress(logs / "progress.csv")
    results = parse_results(logs / "results.jsonl")

    if args.kill_epoch is not None:
        kill_epoch, source = args.kill_epoch, "--kill-epoch"
    elif (logs / "kill_marker").exists():
        kill_epoch = float((logs / "kill_marker").read_text().strip())
        source = "kill_marker (head clock)"
    else:
        kill_epoch = infer_kill_epoch(events)
        source = ("inferred from first failure log line "
                  "(= detection time, LATER than the kill)")

    report = build_report(meta, events, samples, results, kill_epoch, source,
                          args.grace)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0 if report["accounting"]["pass_100pct"] else 1


if __name__ == "__main__":
    sys.exit(main())
