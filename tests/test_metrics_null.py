"""NullMetrics equivalence. Two layers of proof: 1. 2."""

from unittest.mock import patch

import pytest

import coordinator as coord_mod
from metrics import (
    COUNT_NAMES,
    GAUGE_NAMES,
    NULL_METRICS,
    Metrics,
    NullMetrics,
    TIMER_NAMES,
)
from bench.recording import RecordingMetrics

# Reuse the fake-Ray harness from the coordinator tests: same scheduler.
from tests.test_coordinator import (
    _coordinator_with_fake_workers,
    _fake_ray_get,
    _fake_ray_wait,
)


class TestNullMetricsContract:
    def test_coordinator_defaults_to_null_singleton(self):
        coord, _ = _coordinator_with_fake_workers([["ok"]], n_tasks=1)
        assert coord._metrics is NULL_METRICS, (
            "Every coordinator must share the NullMetrics singleton by "
            "default; instrumentation is opt-in via bench/."
        )

    def test_null_metrics_satisfies_protocol(self):
        assert isinstance(NULL_METRICS, Metrics)
        assert isinstance(RecordingMetrics(), Metrics)

    def test_methods_are_noops(self):
        m = NullMetrics()
        for name in GAUGE_NAMES:
            assert m.gauge(name, 1.0) is None
        for name in COUNT_NAMES:
            assert m.count(name) is None
            assert m.count(name, 5) is None
        for name in TIMER_NAMES:
            with m.timer(name):
                pass  # enters and exits cleanly

    def test_timer_never_suppresses_exceptions(self):
        """A suppressing __exit__ would swallow worker failures inside timed blocks
        - the one way a 'no-op' could change control flow."""
        m = NullMetrics()
        with pytest.raises(RuntimeError, match="must propagate"):
            with m.timer("ray_wait"):
                raise RuntimeError("must propagate")
        assert m.timer("loop_iter").__exit__(
            RuntimeError, RuntimeError("x"), None
        ) is False

    def test_recording_timer_never_suppresses_exceptions(self):
        """RecordingMetrics must hold the same contract, or attaching it in the
        bench would change behavior relative to production."""
        m = RecordingMetrics()
        with pytest.raises(RuntimeError, match="must propagate"):
            with m.timer("ray_wait"):
                raise RuntimeError("must propagate")
        # The sample is still recorded on the exceptional path.
        assert len(m.timers["ray_wait"]) == 1


class TestNullVsRecordingEquivalence:
    """The same scheduling scenario."""

    PLAN = [
        ["raise_then_ok", "ok", "ok", "ok", "ok"],  # exercises retry path
        ["ok"] * 10,
    ]

    def _run(self, metrics=None) -> tuple[dict, object]:
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[list(p) for p in self.PLAN],
            n_tasks=12,
            max_retries=2,
        )
        if metrics is not None:
            coord._metrics = metrics
        with patch.object(coord_mod.ray, "wait", _fake_ray_wait), \
             patch.object(coord_mod.ray, "get", _fake_ray_get), \
             patch.object(coord_mod.ray, "kill"):
            summary = coord.run(tasks)
        return summary, coord

    def test_identical_summary_with_and_without_recording(self):
        base, _ = self._run(metrics=None)  # NullMetrics default
        rec = RecordingMetrics()
        recorded, _ = self._run(metrics=rec)

        for key in ("total", "succeeded", "failed"):
            assert base[key] == recorded[key], (
                f"summary[{key!r}] diverged between NullMetrics and "
                f"RecordingMetrics: {base[key]} vs {recorded[key]} - "
                "instrumentation changed behavior"
            )
        assert base["total"] == 12
        assert base["succeeded"] == 12

    def test_recording_captures_the_expected_events(self):
        """Sanity that the seam actually measures what it claims (the equivalence
        above would also pass if nothing were recorded)."""
        rec = RecordingMetrics()
        summary, _ = self._run(metrics=rec)

        assert summary["succeeded"] == 12
        assert rec.count_total("completed") == 12
        assert rec.count_total("retried") >= 1, (
            "The raise_then_ok plan must route through the retry path"
        )
        assert rec.count_total("failed") == 0
        assert rec.timer_count("loop_iter") >= 1
        assert rec.timer_count("ray_wait") >= 1
        assert rec.timer_count("dispatch") >= 1
        assert rec.timer_count("agg_submit") >= 1
        for gauge in GAUGE_NAMES:
            assert rec.gauges[gauge], f"gauge {gauge!r} never emitted"
        # Default --standby 0: no pool is ever populated.
        assert all(v == 0.0 for _, v in rec.gauges["standby"])
