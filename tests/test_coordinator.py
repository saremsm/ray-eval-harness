import time
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from collections import deque

import coordinator as coord_mod
from types_ import EvalResult, EvalTask, FailureKind
from coordinator import (
    DistributedEvalCoordinator,
    HUNG_REF_THRESHOLD_S,
    classify_failure,
    validate_backend,
)

# Helpers
def make_task(task_id: str = "t001") -> EvalTask:
    return EvalTask(
        task_id=task_id,
        prompt="The sky is",
        expected_answer="blue",
    )

def make_coordinator(**kwargs) -> DistributedEvalCoordinator:
    defaults = dict(n_workers=2, model_name="distilgpt2")
    defaults.update(kwargs)
    return DistributedEvalCoordinator(**defaults)

class _Ref:
    """Hashable stand-in for a Ray ObjectRef with a readable repr."""
    def __init__(self, label: str) -> None:
        self.label = label
    def __repr__(self) -> str:
        return f"FakeRef({self.label!r})"

class _FakeWorker:
    """Minimal worker stand-in for tests that don't need Ray."""
    def evaluate_batch(self, tasks): pass
    def health_check(self): return True
    def get_stats(self): return {}

class _CapturingAggregator:
    """records add_result payloads."""
    def __init__(self):
        self.results = []
        self.add_result = SimpleNamespace(
            remote=lambda r: (self.results.append(r), r)[1]
        )

# Validate Backend
class TestValidateBackend:
    def test_passes_when_all_methods_present(self):
        class GoodBackend:
            def evaluate_batch(self, tasks): pass
            def health_check(self): return True
            def get_stats(self): return {}

        validate_backend(GoodBackend())

    def test_raises_on_missing_health_check(self):
        class BadBackend:
            def evaluate_batch(self, tasks): pass
            def get_stats(self): return {}

        with pytest.raises(AttributeError, match="health_check"):
            validate_backend(BadBackend())

    def test_raises_for_plain_object(self):
        with pytest.raises(AttributeError):
            validate_backend(object())


# Classify Failure
class TestClassifyFailure:
    def test_generic_runtime_is_transient(self):
        assert classify_failure(RuntimeError("worker crashed")) == FailureKind.TRANSIENT

    def test_timeout_is_transient(self):
        assert classify_failure(TimeoutError("30s exceeded")) == FailureKind.TRANSIENT

    def test_token_length_is_deterministic(self):
        assert classify_failure(
            RuntimeError("token indices sequence length longer than 512")
        ) == FailureKind.DETERMINISTIC

    def test_index_out_of_range_is_deterministic(self):
        assert classify_failure(
            IndexError("index out of range")
        ) == FailureKind.DETERMINISTIC

class TestAssignNext:
    def test_submits_with_correct_retry_count(self):
        """retry_count from tuple must reach submit(), not default of 0."""
        coord = make_coordinator()
        submitted = []

        def mock_submit(widx, batch, retry_count=0):
            submitted.append((widx, batch, retry_count))

        batch = [make_task("t0")]
        # pending stores tuples; retry_count=1 simulates a re-queued batch.
        pending = deque([(batch, 1)])
        coord._assign_next(0, pending, mock_submit)

        assert len(submitted) == 1
        _, submitted_batch, submitted_rc = submitted[0]
        assert submitted_batch == batch
        assert submitted_rc == 1, (
            f"Expected retry_count=1 (from tuple), got {submitted_rc}. "
            "_assign_next must unpack the tuple, not use the default."
        )
        assert len(pending) == 0

    def test_idle_when_pending_empty(self):
        coord = make_coordinator()
        submitted = []
        coord._assign_next(0, deque(), lambda *a, **k: submitted.append(1))
        assert submitted == []

    def test_worker_idx_passed_through(self):
        coord = make_coordinator()
        submitted = []
        pending = deque([([make_task()], 0)])
        coord._assign_next(
            7, pending,
            lambda widx, *a, **k: submitted.append(widx),
        )
        assert submitted[0] == 7


# Handle Timeouts
class TestHandleTimeouts:
    """Tests use direct timestamp manipulation rather than monkeypatching."""

    def _make_coord(self, max_retries: int = 2) -> DistributedEvalCoordinator:
        coord = make_coordinator(max_retries=max_retries)
        coord._replace_worker = lambda workers, idx: None
        return coord

    def test_young_ref_not_evicted(self):
        coord = self._make_coord()
        ref = _Ref("r0")
        batch = [make_task("t0")]
        active = {ref: (0, batch, 0)}
        ref_timeouts = {ref: time.monotonic()}

        agg = _CapturingAggregator()        

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=deque(),
            submit=lambda *a, **k: None,
            aggregator=agg,
        )

        assert ref in active, "Recent ref must not be evicted"
        assert ref in ref_timeouts, "Recent ref's timestamp must be retained"
        assert agg.results == []

    def test_old_ref_evicted(self):
        coord = self._make_coord()
        ref = _Ref("r0")
        batch = [make_task("t0")]
        active = {ref: (0, batch, 0)}
        ref_timeouts = {ref: time.monotonic() - HUNG_REF_THRESHOLD_S - 1.0}

        agg = _CapturingAggregator()

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=deque(),
            submit=lambda *a, **k: None,
            aggregator=agg,
        )

        assert ref not in active, "Old ref must be removed from active"
        assert ref not in ref_timeouts, (
            "Old ref must be removed from ref_timeouts on eviction"
        )
        assert agg.results == []

    def test_evicted_batch_requeued_with_incremented_retry_count(self):
        coord = self._make_coord(max_retries=2)
        ref = _Ref("r0")
        batch = [make_task("t0")]
        active = {ref: (0, batch, 0)}
        ref_timeouts = {ref: time.monotonic() - HUNG_REF_THRESHOLD_S - 1.0}
        pending = deque()
        submitted = []

        def capture_submit(widx, b, retry_count=0):
            submitted.append((widx, b, retry_count))

        agg = _CapturingAggregator()
        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=pending,
            submit=capture_submit,
            aggregator=agg,
        )

        assert len(submitted) == 1, (
            "Expected exactly one submit call (re-queued batch immediately "
            f"consumed because pending was empty). Got: {submitted}"
        )
        _, submitted_batch, submitted_retry = submitted[0]
        assert submitted_batch == batch
        assert submitted_retry == 1, (
            f"retry_count must be 0 + 1 = 1, got {submitted_retry}. "
            "Hung timeouts must consume the retry budget."
        )
        assert agg.results == []

    def test_hang_budget_exhaustion_records_terminal_failures(self):
        """Hang past max_retries => recorded terminal failure, not re-queued.
        Regression: retry_count incremented on the hang path but never checked;
        an always-hanging batch looped forever."""
        coord = self._make_coord(max_retries=2)
        agg = _CapturingAggregator()
        ref = _Ref("r0")
        batch = [make_task("t0"), make_task("t1")]
        # retry_count == max_retries: the budget is spent.
        active = {ref: (0, batch, 2)}
        ref_timeouts = {ref: time.monotonic() - HUNG_REF_THRESHOLD_S - 1.0}
        pending = deque()
        submitted = []

        with patch.object(coord_mod.ray, "get", lambda r, timeout=None: r):
            coord._handle_timeouts(
                active=active,
                ref_timeouts=ref_timeouts,
                workers=[_FakeWorker()],
                pending=pending,
                submit=lambda widx, b, retry_count=0: submitted.append(b),
                aggregator=agg,
            )

        assert ref not in active
        assert submitted == [], (
            "Budget-exhausted batch must NOT be re-submitted"
        )
        assert len(pending) == 0
        assert len(agg.results) == 2, (
            "One terminal EvalResult per task in the hung batch"
        )
        assert all(r.failed for r in agg.results)
        assert all("retry budget exhausted" in r.error for r in agg.results)


    def test_ages_independent_across_refs(self):
        """One old ref must not cause young refs to be evicted in the same call."""
        coord = self._make_coord()
        agg = _CapturingAggregator()
        ref_old = _Ref("old")
        ref_young = _Ref("young")
        batch_old = [make_task("t_old")]
        batch_young = [make_task("t_young")]
        workers = [_FakeWorker(), _FakeWorker()]

        now = time.monotonic()
        active = {
            ref_old: (0, batch_old, 0),
            ref_young: (1, batch_young, 0),
        }
        ref_timeouts = {
            ref_old: now - HUNG_REF_THRESHOLD_S - 1.0,
            ref_young: now,
        }

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=workers,
            pending=deque(),
            submit=lambda *a, **k: None,
            aggregator=agg,
        )

        assert ref_old not in active
        assert ref_old not in ref_timeouts
        assert ref_young in active, (
            "Young ref must remain active despite ref_old's eviction in the "
            "same call. Eviction decisions must be per-ref, not global."
        )
        assert ref_young in ref_timeouts

    def test_no_eviction_when_no_refs_outstanding(self):
        coord = self._make_coord()
        active: dict = {}
        ref_timeouts: dict = {}
        pending: deque = deque()
        submitted = []

        agg = _CapturingAggregator()
        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=pending,
            submit=lambda *a, **k: submitted.append(1),
            aggregator=agg,
        )

        assert active == {}
        assert ref_timeouts == {}
        assert len(pending) == 0
        assert submitted == []
        assert agg.results == []

class _FakeRef:
    """Stand-in for ObjectRef."""

    def __init__(
        self,
        result=None,
        exception: Exception | None = None,
        resolve_at: float | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.resolve_at = resolve_at if resolve_at is not None else time.monotonic()

    def is_ready(self, now: float) -> bool:
        return now >= self.resolve_at


class _FakeAggregator:
    """Captures add_result and get_summary calls."""

    def __init__(self, total_tasks: int, output_path: str) -> None:
        self.total_tasks = total_tasks
        self.output_path = output_path
        self.results: list[EvalResult] = []

    @classmethod
    def remote(cls, total_tasks: int, output_path: str) -> "_FakeAggregator":
        return cls(total_tasks=total_tasks, output_path=output_path)

    class _Method:
        def __init__(self, fn):
            self.fn = fn

        def remote(self, *args, **kwargs) -> _FakeRef:
            try:
                return _FakeRef(result=self.fn(*args, **kwargs))
            except Exception as exc:
                return _FakeRef(exception=exc)

    @property
    def add_result(self):
        def _fn(result: EvalResult) -> None:
            self.results.append(result)
        return _FakeAggregator._Method(_fn)

    @property
    def get_summary(self):
        def _fn() -> dict:
            return {
                "total": len(self.results),
                "succeeded": sum(1 for r in self.results if r.succeeded),
                "failed": sum(1 for r in self.results if r.failed),
                "results_file": self.output_path,
            }
        return _FakeAggregator._Method(_fn)


class _FakeActorMethod:
    """Wraps a callable so .remote() returns a _FakeRef with optional delay."""

    def __init__(self, fn, delay: float = 0.0):
        self._fn = fn
        self._delay = delay

    def remote(self, *args, **kwargs) -> _FakeRef:
        try:
            result = self._fn(*args, **kwargs)
            return _FakeRef(
                result=result,
                resolve_at=time.monotonic() + self._delay,
            )
        except Exception as exc:
            return _FakeRef(
                exception=exc,
                resolve_at=time.monotonic() + self._delay,
            )


class _FakeBackendActor:
    """Worker actor stand-in that satisfies EvalBackend structurally."""
    def __init__(
        self,
        worker_id: int,
        plan: list[str] | None = None,
        delay: float = 0.0,
        **_unused,
    ) -> None:
        self.worker_id = worker_id
        self._plan = list(plan) if plan else []
        self._delay = delay
        self.completed = 0
        self.failed = 0
        self.poisoned = False

    @property
    def evaluate_batch(self):
        def _fn(tasks: list[EvalTask]) -> list[EvalResult]:
            if self.poisoned:
                self.failed += len(tasks)
                raise RuntimeError(f"worker {self.worker_id} is poisoned")

            action = self._plan.pop(0) if self._plan else "ok"
            if action == "raise":
                self.failed += len(tasks)
                raise RuntimeError("injected failure")
            if action == "raise_then_ok":
                self._plan.insert(0, "ok")
                self.failed += len(tasks)
                raise RuntimeError("injected failure")
            if action == "poison":
                self.poisoned = True
                self.failed += len(tasks)
                raise RuntimeError(f"worker {self.worker_id} poisoned itself")

            self.completed += len(tasks)
            return [
                EvalResult(
                    task_id=t.task_id,
                    score=1.0,
                    response="ok",
                    latency_seconds=0.01,
                    worker_id=self.worker_id,
                )
                for t in tasks
            ]
        return _FakeActorMethod(_fn, delay=self._delay)

    @property
    def health_check(self):
        def _fn() -> bool:
            return not self.poisoned
        return _FakeActorMethod(_fn, delay=0.0)

    @property
    def get_stats(self):
        def _fn() -> dict:
            return {
                "worker_id": self.worker_id,
                "model": "fake",
                "backend": "fake",
                "completed": self.completed,
                "failed": self.failed,
                "poisoned": self.poisoned,
            }
        return _FakeActorMethod(_fn, delay=0.0)


def _fake_ray_wait(refs, num_returns=1, timeout=None):
    """Stand-in for ray.wait."""
    deadline = time.monotonic() + (timeout if timeout is not None else 0.0)
    while True:
        now = time.monotonic()
        ready = [r for r in refs if r.is_ready(now)]
        if ready:
            return ready[:num_returns], [r for r in refs if r not in ready]
        if now >= deadline:
            return [], list(refs)
        time.sleep(min(0.01, max(0.0, deadline - now)))


def _fake_ray_get(ref_or_refs, timeout=None):
    """Stand-in for ray.get."""
    if isinstance(ref_or_refs, list):
        return [_fake_ray_get(r, timeout=timeout) for r in ref_or_refs]
    ref = ref_or_refs
    if timeout is not None:
        deadline = time.monotonic() + timeout
        while not ref.is_ready(time.monotonic()):
            if time.monotonic() >= deadline:
                raise TimeoutError("fake ray.get timeout")
            time.sleep(0.01)
    if ref.exception is not None:
        raise ref.exception
    return ref.result


@pytest.fixture
def patched_ray():
    """Patch ray.wait, ray.get."""
    with patch.object(coord_mod.ray, "wait", _fake_ray_wait), \
         patch.object(coord_mod.ray, "get", _fake_ray_get), \
         patch.object(coord_mod.ray, "kill") as kill_mock:
        yield {"kill": kill_mock}


def _coordinator_with_fake_workers(
    plan_per_worker: list[list[str]],
    n_tasks: int = 8,
    max_retries: int = 2,
    delay: float = 0.0,
    batch_size: int = 4,
) -> tuple[DistributedEvalCoordinator, list[EvalTask]]:
    """Build coordinator wired to fake workers and a fake aggregator."""
    n_workers = len(plan_per_worker)
    plan_queue = deque(plan_per_worker)

    class _BackendStub:
        @staticmethod
        def remote(**kwargs):
            plan = plan_queue.popleft() if plan_queue else None
            return _FakeBackendActor(plan=plan, delay=delay, **kwargs)

    coord = DistributedEvalCoordinator(
        n_workers=n_workers,
        model_name="fake-model",
        max_retries=max_retries,
        output_path="results/test.jsonl",
        batch_size=batch_size,
        aggregator_cls=_FakeAggregator,
    )
    coord._worker_cls = _BackendStub

    tasks = [make_task(f"t{i:03d}") for i in range(n_tasks)]
    return coord, tasks


class TestCoordinatorIntegration:
    """End-to-end run() tests against a fake backend."""

    def test_clean_run_completes_all_tasks(self, patched_ray):
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 8, ["ok"] * 8],
            n_tasks=8,
        )
        summary = coord.run(tasks)

        assert summary["total"] == 8, (
            f"Expected 8 results, got {summary['total']}"
        )
        assert summary["succeeded"] == 8
        assert summary["failed"] == 0


    def test_transient_failure_is_retried_and_succeeds(self, patched_ray):
        # Worker 0 fails on first call, then succeeds. Worker 1 is always healthy.
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[
                ["raise", "ok", "ok", "ok", "ok"],
                ["ok"] * 8,
            ],
            n_tasks=8,
            max_retries=2,
        )
        summary = coord.run(tasks)

        assert summary["total"] == 8, (
            "Retried batch must eventually be recorded by the aggregator"
        )
        assert summary["succeeded"] == 8
        assert summary["failed"] == 0

    def test_retry_exhaustion_records_failed_results(self, patched_ray):
        # Every call on worker 0 raises. Worker 1 succeeds.
        #   max_retries=1, worker 0's batch lands in failed bucket
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[
                ["raise"] * 10,
                ["ok"] * 10,
            ],
            n_tasks=8,
            max_retries=1,
        )
        summary = coord.run(tasks)

        assert summary["total"] == 8
        assert summary["failed"] >= 1, (
            "At least one batch should land in the failed bucket after "
            f"retry exhaustion. Summary: {summary}"
        )
        assert summary["succeeded"] + summary["failed"] == 8

    def test_poisoned_worker_replaced_on_first_failure(self, patched_ray):
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[
                ["poison"],          # worker 0: poisons itself on first call
                ["ok"] * 20,         # worker 1: always healthy
            ],
            n_tasks=8,
            max_retries=2,
        )
        summary = coord.run(tasks)

        assert summary["total"] == 8
        assert summary["succeeded"] == 8, (
            "After worker 0 poisons itself and is replaced, all retries "
            f"should succeed against the fresh replacement. Got {summary}"
        )
        assert summary["failed"] == 0
