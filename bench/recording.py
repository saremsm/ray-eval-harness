"""RecordingMetrics: a Metrics implementation that keeps timestamped samples in
memory for the saturation harness (saturation bench). Bench-only."""

from __future__ import annotations

import math
import time
from collections import defaultdict


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile, same convention as aggregator._percentile."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(n - 1, max(0, math.ceil(q * n) - 1))
    return sorted_values[idx]


class _RecordingTimer:
    """One-shot context manager appending (start_ts, duration)."""

    __slots__ = ("_sink", "_start")

    def __init__(self, sink: list) -> None:
        self._sink = sink
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.monotonic()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._sink.append((self._start, time.monotonic() - self._start))
        return False  # never suppress - same contract as NullMetrics


class RecordingMetrics:
    """Collects every timer/gauge/count sample with a monotonic timestamp."""

    def __init__(self) -> None:
        self.timers: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.gauges: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.counts: dict[str, list[tuple[float, int]]] = defaultdict(list)

    # Metrics interface
    def timer(self, name: str) -> _RecordingTimer:
        return _RecordingTimer(self.timers[name])

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name].append((time.monotonic(), value))

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name].append((time.monotonic(), n))

    # Analysis helpers
    def count_total(self, name: str) -> int:
        return sum(n for _, n in self.counts.get(name, []))

    def timer_span(self, name: str) -> tuple[float, float] | None:
        """(first_start, last_end) across all samples of a timer."""
        samples = self.timers.get(name, [])
        if not samples:
            return None
        first = samples[0][0]
        last = max(start + dur for start, dur in samples)
        return (first, last)

    def timer_total(
        self, name: str, t_lo: float | None = None, t_hi: float | None = None
    ) -> float:
        """Sum of durations for samples whose START falls in [t_lo, t_hi]."""
        total = 0.0
        for start, dur in self.timers.get(name, []):
            if t_lo is not None and start < t_lo:
                continue
            if t_hi is not None and start > t_hi:
                continue
            total += dur
        return total

    def timer_count(
        self, name: str, t_lo: float | None = None, t_hi: float | None = None
    ) -> int:
        return sum(
            1
            for start, _ in self.timers.get(name, [])
            if (t_lo is None or start >= t_lo)
            and (t_hi is None or start <= t_hi)
        )

    def steady_window(
        self, name: str = "loop_iter", trim: float = 0.2
    ) -> tuple[float, float] | None:
        """Middle (1 - 2*trim) of a timer's span: drops ramp-up (pipeline fill) and
        drain (last stragglers), where shares are unrepresentative."""
        span = self.timer_span(name)
        if span is None:
            return None
        t0, t1 = span
        width = t1 - t0
        return (t0 + trim * width, t1 - trim * width)

    def gauge_downsample(
        self, name: str, interval_s: float = 0.5
    ) -> list[tuple[float, float]]:
        """(t_rel, value) on a fixed grid: last sample in each interval.
        Downsampling to the grid gives the "sampled every 0.5s" series the report
        promises without a second thread."""
        samples = self.gauges.get(name, [])
        if not samples:
            return []
        t0 = samples[0][0]
        out: list[tuple[float, float]] = []
        bucket = 0
        last_val = samples[0][1]
        for ts, val in samples:
            b = int((ts - t0) / interval_s)
            while b > bucket:
                out.append((round(bucket * interval_s, 3), last_val))
                bucket += 1
            last_val = val
        out.append((round(bucket * interval_s, 3), last_val))
        return out
