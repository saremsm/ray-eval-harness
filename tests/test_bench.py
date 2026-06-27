"""Bench harness tests."""

import argparse
import asyncio
import time

import pytest

from types_ import EvalResult, EvalTask

from bench.fake_worker import FakeLatencyWorkerImpl
from bench.recording import RecordingMetrics, percentile


def make_tasks(n: int) -> list[EvalTask]:
    return [EvalTask(task_id=f"t{i:03d}", prompt="x") for i in range(n)]


class TestFakeLatencyWorkerResults:
    def _run_batch(self, worker, tasks):
        return asyncio.run(worker.evaluate_batch(tasks))

    def test_returns_valid_results_in_input_order(self):
        worker = FakeLatencyWorkerImpl(
            worker_id=3, latency_s=0.001, jitter=0.0
        )
        tasks = make_tasks(5)
        results = self._run_batch(worker, tasks)

        assert len(results) == 5
        assert [r.task_id for r in results] == [t.task_id for t in tasks]
        for r in results:
            assert isinstance(r, EvalResult)
            assert r.succeeded
            assert r.score == 1.0
            assert r.worker_id == 3
            assert r.latency_seconds >= 0.0
            assert r.batch_latency_seconds is not None
            assert r.batch_latency_seconds >= 0.0
            # Round-trips through the aggregator's JSONL path.
            assert isinstance(r.to_dict(), dict)

    def test_sleeps_about_latency(self):
        worker = FakeLatencyWorkerImpl(
            worker_id=0, latency_s=0.05, jitter=0.0
        )
        start = time.perf_counter()
        self._run_batch(worker, make_tasks(2))
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.05
        assert elapsed < 1.0

    def test_jitter_stays_within_band(self):
        worker = FakeLatencyWorkerImpl(
            worker_id=0, latency_s=0.01, jitter=0.5, seed=7
        )
        for _ in range(10):
            results = self._run_batch(worker, make_tasks(1))
            # sleep = latency * (1 + jitter*U(-1,1)) in [0.005, 0.015]
            assert 0.005 <= results[0].batch_latency_seconds <= 0.5

    def test_ignores_backend_kwargs(self):
        """The coordinator's hf factory passes device=...; fake worker must accept
        and ignore it."""
        worker = FakeLatencyWorkerImpl(
            worker_id=0, device=-1, model_name="fake", task_timeout=60.0,
            latency_s=0.001,
        )
        results = self._run_batch(worker, make_tasks(1))
        assert results[0].succeeded

    def test_stats_track_completions(self):
        worker = FakeLatencyWorkerImpl(worker_id=1, latency_s=0.0)
        self._run_batch(worker, make_tasks(4))
        stats = asyncio.run(worker.get_stats())
        assert stats["completed"] == 4
        assert stats["failed"] == 0
        assert asyncio.run(worker.health_check()) is True


class TestRecordingHelpers:
    def test_percentile_nearest_rank(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert percentile(vals, 0.5) == 2.0
        assert percentile(vals, 0.99) == 4.0
        assert percentile([], 0.5) == 0.0

    def test_gauge_downsample_grid(self):
        rec = RecordingMetrics()
        rec.gauges["pending"] = [(100.0, 5.0), (100.1, 4.0), (100.7, 3.0)]
        series = rec.gauge_downsample("pending", 0.5)
        assert series[0] == (0.0, 4.0)   # last sample in [0, 0.5)
        assert series[-1] == (0.5, 3.0)  # last sample in [0.5, 1.0)


@pytest.mark.slow
class TestSaturationHarnessRealRay:
    def test_harness_runs_under_real_ray(self, tmp_path):
        """4 workers, 200 tasks against a real local cluster; the whole thing -
        ray.init included - must finish inside 30s."""
        import ray

        from bench.saturation import build_arg_parser, run_saturation

        ns = build_arg_parser().parse_args([
            "--workers", "4",
            "--latency-s", "0.02",
            "--batch-size", "8",
            "--tasks", "200",
            "--fail-rate", "0.0",
            "--out", str(tmp_path / "sat_test.json"),
        ])

        start = time.perf_counter()
        try:
            report = run_saturation(ns)
        finally:
            elapsed = time.perf_counter() - start
            ray.shutdown()

        assert elapsed < 30.0, (
            f"Harness took {elapsed:.1f}s; must run 200 tasks on 4 "
            "workers in under 30s"
        )

        assert report["achieved"]["tasks_recorded"] == 200
        assert report["achieved"]["succeeded"] == 200
        assert report["achieved"]["failed"] == 0
        assert report["counts"]["completed"] == 200
        assert report["counts"]["failed"] == 0
        assert report["offered_load_tasks_per_s"] == pytest.approx(
            4 * 8 / 0.02
        )
        assert report["achieved"]["throughput_tasks_per_s_wall"] > 0
        assert report["loop"]["iterations"] >= 200 // 8
        # All four timers produced steady-state totals.
        totals = report["steady_state"]["timer_totals_s"]
        for name in ("loop_iter", "ray_wait", "dispatch", "agg_submit"):
            assert name in totals
        # Gauge series exist, standby pinned at 0 (no standby pool yet).
        series = report["gauges"]["series"]
        for name in ("pending", "active", "deferred", "standby"):
            assert series[name], f"gauge series {name!r} empty"
        assert all(v == 0.0 for _, v in series["standby"])
        # Mailbox-proxy latency was sampled and documented as a proxy.
        assert "proxy" in report["agg_call_latency"]["note"]
        assert report["env"]["ray_version"] == ray.__version__

class TestAggProbeDefault:
    def test_probe_every_default_matches_post_d5_call_volume(self):
        """Pin the probe default: the probe's unit changed with batched writes from
        one add_result call per result to one record_batch call per batch per
        shard."""
        from bench.saturation import build_arg_parser

        ns = build_arg_parser().parse_args([])
        assert ns.agg_probe_every == 4
