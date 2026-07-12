"""Tests for bench/multinode/measure.py and sample_progress.py."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench" / "multinode"))

import measure  # noqa: E402
import sample_progress  # noqa: E402


#
def _hms(epoch: float) -> str:
    # Emulate the head node's clock: harness logs in head-local time.
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%H:%M:%S")


def make_logs(tmp_path: Path, *, tasks=200, workers=2, kill_offset=120.0,
              recovery_offset=140.0, with_marker=True,
              failed_rows=0, duplicate_last=False):
    """A synthetic run: t0 anchored at 10:00:00 today, steady 10 tasks/s before the
    kill, a stall, then recovery on worker_id==workers."""
    t0 = dt.datetime.now().replace(
        hour=10, minute=0, second=0, microsecond=0
    ).timestamp()
    kill = t0 + kill_offset
    rec = t0 + recovery_offset

    log_lines = [
        f"{_hms(t0 + 5)} INFO All {workers} workers ready and validated (+1 standby)",
        f"{_hms(kill + 8)} ERROR Batch failed on worker 1 (TRANSIENT): "
        f"RayActorError: the actor died because its node has died.",
        f"{_hms(kill + 8)} WARNING Worker 1 (backend=vllm) health check "
        f"unreachable (RayActorError()); replacing worker for health reasons",
        f"{_hms(kill + 8)} WARNING Worker 1 replaced by standby (pool now 0 ready); "
        f"refilling in background",
        f"{_hms(kill + 8)} INFO Scheduling retry 1/2 for 64 tasks as two halves "
        f"(backoff: full jitter per enqueued half)",
        # A retry event well after recovery + grace: must NOT be attributed.
        f"{_hms(rec + 200)} INFO Scheduling retry 1/2 for 7 tasks",
    ]
    (tmp_path / "run.log").write_text("\n".join(log_lines) + "\n")

    samples = []
    for off in range(0, int(kill_offset) + 1, 2):
        samples.append((t0 + off, int(off * 10), 0))
    stalled_total = samples[-1][1]
    e = kill + 2
    while e < rec:
        samples.append((e, stalled_total, 0))
        e += 2
    for i, off in enumerate(range(0, 61, 2)):
        samples.append((rec + off, stalled_total + (i + 1) * 16, 16))
    (tmp_path / "progress.csv").write_text(
        "epoch,total,replacement\n"
        + "\n".join(f"{e:.3f},{t},{r}" for e, t, r in samples)
        + "\n"
    )

    rows = []
    for i in range(tasks):
        wid = workers if i >= tasks - 40 else i % workers
        rows.append({
            "task_id": f"task_{i:05d}",
            "score": 1.0,
            "failed": i < failed_rows,
            "worker_id": wid,
        })
    if duplicate_last:
        rows.append(dict(rows[-1]))
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )

    (tmp_path / "run_meta.json").write_text(json.dumps({
        "t0_epoch": t0, "tasks": tasks, "workers": workers,
        "standby": 1, "task_timeout": 60.0, "model": "m",
    }))
    if with_marker:
        (tmp_path / "kill_marker").write_text(f"{kill:.3f}\n")
    return t0, kill, rec


def load(tmp_path: Path, kill_epoch=None):
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    events = measure.parse_log(tmp_path / "run.log", meta["t0_epoch"])
    samples = measure.parse_progress(tmp_path / "progress.csv")
    results = measure.parse_results(tmp_path / "results.jsonl")
    if kill_epoch is None and (tmp_path / "kill_marker").exists():
        kill_epoch = float((tmp_path / "kill_marker").read_text().strip())
        source = "kill_marker"
    else:
        source = "arg"
    return meta, events, samples, results, kill_epoch, source


#
class TestParseLog:
    def test_reconstructs_epochs_on_t0_day(self, tmp_path):
        t0, kill, _ = make_logs(tmp_path)
        meta = json.loads((tmp_path / "run_meta.json").read_text())
        events = measure.parse_log(tmp_path / "run.log", meta["t0_epoch"])
        ready = next(e for e, _l, m in events if "ready and validated" in m)
        assert ready == pytest.approx(t0 + 5, abs=1.0)

    def test_midnight_wrap_rolls_forward(self, tmp_path):
        # t0 at 23:59:30 *UTC* (the head's clock - parse_log assumes a UTC head)
        t0 = dt.datetime.now(dt.timezone.utc).replace(
            hour=23, minute=59, second=30, microsecond=0
        ).timestamp()
        (tmp_path / "log").write_text("00:00:20 INFO after midnight\n")
        events = measure.parse_log(tmp_path / "log", t0)
        assert events[0][0] == pytest.approx(t0 + 50, abs=1.0)

    def test_unstamped_lines_ignored(self, tmp_path):
        t0 = dt.datetime.now().timestamp()
        (tmp_path / "log").write_text(
            "==== summary block ====\n10:00:00 INFO real line\nplain text\n"
        )
        events = measure.parse_log(tmp_path / "log", t0)
        assert len(events) == 1 and events[0][2] == "real line"


#
class TestWindows:
    def test_phase_rates(self, tmp_path):
        make_logs(tmp_path)
        meta, events, samples, results, kill, src = load(tmp_path)
        report = measure.build_report(
            meta, events, samples, results, kill, src, grace=30.0
        )
        # Steady 10/s before the kill; ~0 during.
        assert report["phases"]["before"]["tasks_per_s"] == pytest.approx(10.0, rel=0.15)
        assert report["phases"]["during"]["tasks_per_s"] < 1.5
        assert report["phases"]["after"]["tasks_per_s"] == pytest.approx(8.0, rel=0.2)

    def test_recovery_bounds_from_sampler(self, tmp_path):
        _t0, kill, rec = make_logs(tmp_path)
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        got = report["recovery"]["kill_to_first_replacement_batch_s"]
        assert got == pytest.approx(rec - kill, abs=2.5)

    def test_no_replacement_rows_flags_none(self, tmp_path):
        make_logs(tmp_path)
        # Zero out the replacement column: the kill took the standby.
        csv = tmp_path / "progress.csv"
        lines = csv.read_text().splitlines()
        fixed = [lines[0]] + [
            ",".join(line.split(",")[:2] + ["0"]) for line in lines[1:]
        ]
        csv.write_text("\n".join(fixed) + "\n")
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        assert report["recovery"]["first_replacement_sample_epoch"] is None
        # 'during' falls back to a bounded window rather than crashing.
        assert report["phases"]["during"]["window_s"][1] <= k + 60.0 + 1e-6

    def test_kill_marker_beats_inference(self, tmp_path):
        _t0, kill, _ = make_logs(tmp_path, with_marker=True)
        meta, events, samples, results, k, src = load(tmp_path)
        assert src == "kill_marker"
        # Inference lands on the first ERROR line, 8s later.
        inferred = measure.infer_kill_epoch(events)
        assert inferred == pytest.approx(kill + 8, abs=1.5)
        assert k == pytest.approx(kill, abs=0.01)


#
class TestRetriesAndAccounting:
    def test_retry_count_windowed(self, tmp_path):
        make_logs(tmp_path)
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        # The 64-task retry inside the window counts.
        assert report["retried_due_to_kill"]["tasks"] == 64
        assert report["retried_due_to_kill"]["retry_events"] == 1

    def test_hang_path_requeue_lines_also_counted(self):
        events = [
            (100.0, "INFO", "Re-queued 32 tasks as 2 entries (retry_count now 1, max 2)"),
            (101.0, "INFO", "Scheduling retry 1/2 for 16 tasks"),
            (999.0, "INFO", "Re-queued 5 tasks as 1 entry"),
        ]
        total, n = measure.retried_tasks_in_window(events, 90.0, 110.0)
        assert (total, n) == (48, 2)

    def test_accounting_pass_at_100pct(self, tmp_path):
        make_logs(tmp_path)
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        acc = report["accounting"]
        assert acc["pass_100pct"] and acc["completed_pct"] == 100.0
        assert acc["rows_on_replacement_workers"] == 40

    def test_duplicate_task_id_fails_invariant(self, tmp_path):
        make_logs(tmp_path, duplicate_last=True)
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        assert report["accounting"]["duplicate_task_ids"] == 1
        assert not report["accounting"]["pass_100pct"]

    def test_terminal_failures_reported_separately(self, tmp_path):
        make_logs(tmp_path, failed_rows=3)
        meta, events, samples, results, k, src = load(tmp_path)
        report = measure.build_report(meta, events, samples, results, k, src, 30.0)
        # Completion can be 100% while success is not: both surfaced.
        assert report["accounting"]["pass_100pct"]
        assert report["accounting"]["failed_rows"] == 3


#
class TestSampleProgress:
    def test_scan_counts_totals_and_replacements(self, tmp_path):
        p = tmp_path / "r.jsonl"
        rows = [{"task_id": "a", "worker_id": 0},
                {"task_id": "b", "worker_id": 2},
                {"task_id": "c", "worker_id": 3}]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        off, n, repl = sample_progress.scan_new_lines(str(p), 0, workers=2)
        assert (n, repl) == (3, 2)
        assert off == p.stat().st_size

    def test_partial_trailing_line_deferred(self, tmp_path):
        p = tmp_path / "r.jsonl"
        full = json.dumps({"task_id": "a", "worker_id": 0}) + "\n"
        partial = '{"task_id": "b", "worker'
        # write_bytes, not write_text: Windows text mode translates \n to \r\n.
        p.write_bytes((full + partial).encode())
        off, n, repl = sample_progress.scan_new_lines(str(p), 0, workers=2)
        assert (n, repl) == (1, 0)
        assert off == len(full.encode())
        # Complete the line: the next scan picks it up from the offset.
        p.write_bytes((full + partial + '_id": 2}\n').encode())
        off2, n2, repl2 = sample_progress.scan_new_lines(str(p), off, workers=2)
        assert (n2, repl2) == (1, 1)
        assert off2 == p.stat().st_size

    def test_missing_file_is_zero(self, tmp_path):
        off, n, repl = sample_progress.scan_new_lines(
            str(tmp_path / "nope.jsonl"), 0, workers=2
        )
        assert (off, n, repl) == (0, 0, 0)
