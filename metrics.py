"""Measurement-only instrumentation interface for the coordinator (saturation
bench). Production runs never construct it."""

from __future__ import annotations

from typing import ContextManager, Protocol, runtime_checkable

TIMER_NAMES = ("ray_wait", "dispatch", "agg_submit", "loop_iter")
GAUGE_NAMES = ("pending", "active", "deferred", "standby")
COUNT_NAMES = ("completed", "failed", "retried", "replaced")


@runtime_checkable
class Metrics(Protocol):
    """Structural interface; the coordinator only ever calls these three."""

    def timer(self, name: str) -> ContextManager[None]: ...

    def gauge(self, name: str, value: float) -> None: ...

    def count(self, name: str, n: int = 1) -> None: ...


class _NoopTimer:
    """Shared, allocation-free context manager."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NOOP_TIMER = _NoopTimer()


class NullMetrics:
    """Do-nothing Metrics."""

    __slots__ = ()

    def timer(self, name: str) -> _NoopTimer:
        return _NOOP_TIMER

    def gauge(self, name: str, value: float) -> None:
        return None

    def count(self, name: str, n: int = 1) -> None:
        return None


# The singleton the coordinator defaults to.
NULL_METRICS = NullMetrics()
