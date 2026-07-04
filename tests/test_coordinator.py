import time
import pytest
import random
from unittest.mock import patch
from collections import deque

import coordinator as coord_mod
from aggregator import shard_index
from types_ import EvalResult, EvalTask, FailureKind
from coordinator import (
    DistributedEvalCoordinator,
    HANG_MARGIN_S,
    HANG_MULTIPLIER,
    HANG_THRESHOLD_MIN_S,
    classify_failure,
    backoff_seconds,
    validate_backend,
)

class TestHangThresholdDerivation:
    """hang_threshold_s = max(MIN, MULTIPLIER * task_timeout + MARGIN)."""

    def _derived(self, task_timeout: float) -> float:
        return max(
            HANG_THRESHOLD_MIN_S,
            HANG_MULTIPLIER * task_timeout + HANG_MARGIN_S,
        )

    def test_default_timeout(self):
        # 60s default: 2 * 60 + 30 = 150, above the 120 floor.
        coord = make_coordinator(task_timeout=60.0)
        assert coord.hang_threshold_s == self._derived(60.0) == 150.0

    def test_large_timeout_scales_above_old_fixed_value(self):
        # --task-timeout 300 previously evicted healthy workers at the fixed 240s.
        coord = make_coordinator(task_timeout=300.0)
        assert coord.hang_threshold_s == self._derived(300.0) == 630.0
        assert coord.hang_threshold_s > 300.0, (
            "Coordinator threshold must exceed the worker-side timeout, "
            "or it evicts workers whose batches are still legally running."
        )

    def test_small_timeout_hits_floor(self):
        # 2 * 10 + 30 = 50 would be trigger-happy; the floor wins.
        coord = make_coordinator(task_timeout=10.0)
        assert coord.hang_threshold_s == self._derived(10.0) == HANG_THRESHOLD_MIN_S

    def test_floor_is_sane(self):
        """Floor must be large enough that a healthy-but-slow batch isn't evicted."""
        assert HANG_THRESHOLD_MIN_S >= 30.0

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
    """Facade-shaped fake: records record_batch payloads."""

    def __init__(self):
        self.results = []
        self.batches = []

    def record_batch(self, results):
        self.batches.append(list(results))
        self.results.extend(results)
        return []

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

    def test_worker_batch_timeout_is_transient(self):
        """The worker's per-batch TimeoutError (no longer a poisoning RuntimeError)"""
        assert classify_failure(
            TimeoutError("Worker 3: batch of 4 exceeded 60.0s")
        ) == FailureKind.TRANSIENT

    def test_token_length_is_deterministic(self):
        assert classify_failure(
            RuntimeError("token indices sequence length longer than 512")
        ) == FailureKind.DETERMINISTIC

    def test_index_out_of_range_is_deterministic(self):
        assert classify_failure(
            IndexError("index out of range")
        ) == FailureKind.DETERMINISTIC

# Backoff Seconds

class TestBackoffSeconds:
    def test_first_retry_within_bound(self):
        rng = random.Random(0)
        for _ in range(100):
            assert 0.0 <= backoff_seconds(0, rng) <= 0.5
    def test_second_retry_within_bound(self):
        rng = random.Random(0)
        for _ in range(100):
            assert 0.0 <= backoff_seconds(1, rng) <= 1.0
    def test_cap_at_eight(self):
        rng = random.Random(0)
        for _ in range(100):
            assert 0.0 <= backoff_seconds(100, rng) <= 8.0
    def test_jitter_spreads_values(self):
        """Full jitter must produce a spread of values, not a single point."""
        rng = random.Random(42)
        samples = [backoff_seconds(3, rng) for _ in range(50)]
        assert len(set(round(s, 3) for s in samples)) > 10, (
            "Jitter should produce varied delays; if all samples cluster "
            "at one value, jitter isn't being applied."
        )
    def test_non_negative(self):
        rng = random.Random(0)
        for retry_count in range(20):
            assert backoff_seconds(retry_count, rng) >= 0

# Per-backend default batch size
class TestDefaultBatchSize:
    def test_hf_default(self):
        coord = make_coordinator(backend="hf")
        assert coord.batch_size == 4
    def test_vllm_default(self):
        coord = make_coordinator(backend="vllm")
        assert coord.batch_size == 64, (
            "vLLM default must be large enough to keep continuous batching "
            "saturated. 4 leaves the engine drained between batches."
        )
    def test_explicit_override(self):
        coord = make_coordinator(backend="vllm", batch_size=16)
        assert coord.batch_size == 16

# _assign_next
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
        ref_timeouts = {ref: time.monotonic() - coord.hang_threshold_s - 1.0}

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
        ref_timeouts = {ref: time.monotonic() - coord.hang_threshold_s - 1.0}
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

    def test_ages_independent_across_refs(self):
        """One old ref must not cause young refs to be evicted in the same call."""
        coord = self._make_coord()
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
            ref_old: now - coord.hang_threshold_s - 1.0,
            ref_young: now,
        }

        agg = _CapturingAggregator()
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
        assert agg.results == []

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

    def test_derived_threshold_evicts_below_old_fixed_240(self):
        """With a small task_timeout, a ref older than the derived threshold but
        younger than the old fixed 240s must trigger the hang path."""
        coord = make_coordinator(max_retries=2, task_timeout=10.0)
        coord._replace_worker = lambda workers, idx: None
        assert coord.hang_threshold_s < 240.0  # 120s floor here

        ref = _Ref("r0")
        batch = [make_task("t0")]
        active = {ref: (0, batch, 0)}
        # Age: past the derived threshold, well under 240.
        age = coord.hang_threshold_s + 10.0
        assert age < 240.0
        ref_timeouts = {ref: time.monotonic() - age}
        pending = deque()
        submitted = []

        agg = _CapturingAggregator()
        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=pending,
            submit=lambda widx, b, retry_count=0: submitted.append(
                (b, retry_count)
            ),
            aggregator=agg,
        )

        assert ref not in active, (
            "Ref past the derived threshold must be evicted even though "
            "it is younger than the old fixed 240s"
        )
        assert ref not in ref_timeouts
        # Exactly one terminal state for the batch: re-queued.
        assert len(submitted) == 1
        assert submitted[0][1] == 1, "hang must consume the retry budget"
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
        ref_timeouts = {ref: time.monotonic() - coord.hang_threshold_s - 1.0}
        pending = deque()
        submitted = []

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

# End-to-end integration tests against an in-memory fake backend.
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
    """Shard-actor fake: captures record_batch / get_summary / get_shard_state /."""

    instances: list["_FakeAggregator"] = []

    def __init__(self, total_tasks: int, output_path: str) -> None:
        self.total_tasks = total_tasks
        self.output_path = output_path
        self.results: list[EvalResult] = []
        self.record_batch_calls = 0
        self.closed = False
        _FakeAggregator.instances.append(self)

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
    def record_batch(self):
        def _fn(results: list[EvalResult]) -> None:
            self.record_batch_calls += 1
            self.results.extend(results)
        return _FakeAggregator._Method(_fn)

    @property
    def add_result(self):
        # Back-compat shim, same as the real actor's.
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

    @property
    def get_shard_state(self):
        # Pins the shard-state contract the facade merges.
        def _fn() -> dict:
            succ = [r for r in self.results if r.succeeded]
            batch_lat = [
                r.batch_latency_seconds for r in succ
                if r.batch_latency_seconds is not None
            ]
            per_worker: dict[int, int] = {}
            for r in self.results:
                per_worker[r.worker_id] = per_worker.get(r.worker_id, 0) + 1
            return {
                "count": len(self.results),
                "succeeded": len(succ),
                "failed": sum(1 for r in self.results if r.failed),
                "stopped_early": sum(1 for r in succ if r.stopped_early),
                "score_sum": sum(r.score for r in succ),
                "score_min": min(
                    (r.score for r in succ), default=float("inf")
                ),
                "score_max": max(
                    (r.score for r in succ), default=float("-inf")
                ),
                "latency_sum": sum(r.latency_seconds for r in succ),
                "latency_samples": [r.latency_seconds for r in succ],
                "latency_seen": len(succ),
                "batch_latency_samples": batch_lat,
                "batch_latency_seen": len(batch_lat),
                "tokens_total": sum(r.tokens_generated for r in succ),
                "per_worker": per_worker,
                "condition_scores": [
                    r.condition_scores for r in succ if r.condition_scores
                ],
            }
        return _FakeAggregator._Method(_fn)

    @property
    def close(self):
        def _fn() -> None:
            self.closed = True
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
        health_delay: float = 0.0,
        **_unused,
    ) -> None:
        self.worker_id = worker_id
        self._plan = list(plan) if plan else []
        self._delay = delay
        # Deferred-readiness modeling: the health_check ref resolves only after
        # health_delay seconds.
        self._health_delay = health_delay
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
                    batch_latency_seconds=0.01,
                    worker_id=self.worker_id,
                )
                for t in tasks
            ]
        return _FakeActorMethod(_fn, delay=self._delay)

    @property
    def health_check(self):
        def _fn() -> bool:
            return not self.poisoned
        return _FakeActorMethod(_fn, delay=self._health_delay)

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
    aggregator_shards: int = 1,
    standby: int = 0,
    refill_health_delay: float = 0.0,
) -> tuple[DistributedEvalCoordinator, list[EvalTask]]:
    """Build coordinator wired to fake workers and a fake aggregator."""
    n_workers = len(plan_per_worker)
    n_initial = n_workers + standby
    plan_queue = deque(plan_per_worker)

    class _BackendStub:
        created: list[_FakeBackendActor] = []

        @staticmethod
        def remote(**kwargs):
            plan = plan_queue.popleft() if plan_queue else None
            is_refill = (
                kwargs.get("worker_id", 0) >= n_workers
                and len(_BackendStub.created) >= n_initial
            )
            actor = _FakeBackendActor(
                plan=plan,
                delay=delay,
                health_delay=refill_health_delay if is_refill else 0.0,
                **kwargs,
            )
            _BackendStub.created.append(actor)
            return actor

        @staticmethod
        def options(**_kwargs):
            return _BackendStub  # ignore num_gpus etc. in tests

    coord = DistributedEvalCoordinator(
        n_workers=n_workers,
        model_name="fake-model",
        backend="hf",
        max_retries=max_retries,
        task_timeout=60.0,
        output_path="results/test.jsonl",
        batch_size=batch_size,
        aggregator_cls=_FakeAggregator,
        aggregator_shards=aggregator_shards,
        worker_cls=_BackendStub,
        standby=standby,
    )
    coord._test_worker_stub = _BackendStub

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

    def test_summary_reports_hang_threshold(self, patched_ray):
        """run() must surface the derived threshold so operators can see what
        eviction policy the run actually used."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 4, ["ok"] * 4],
            n_tasks=4,
        )
        summary = coord.run(tasks)
        assert summary["hang_threshold_s"] == coord.hang_threshold_s
        # _coordinator_with_fake_workers passes task_timeout=60.0.
        assert summary["hang_threshold_s"] == 150.0

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

    def test_replace_worker_kills_old_actor(self, patched_ray):
        """Must ray.kill the old actor before building the replacement: a dropped
        handle leaves a GPU-claiming actor holding its resources, so the
        replacement never schedules."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[
                ["poison"],
                ["ok"] * 20,
            ],
            n_tasks=8,
            max_retries=2,
        )
        summary = coord.run(tasks)
        kill_mock = patched_ray["kill"]
        assert kill_mock.called, (
            "Replacing a worker must force-kill the old actor to free "
            "its resources."
        )
        killed_actor = kill_mock.call_args[0][0]
        assert isinstance(killed_actor, _FakeBackendActor), (
            f"ray.kill must receive the old actor handle, got "
            f"{killed_actor!r}"
        )
        assert kill_mock.call_args[1].get("no_restart") is True
        assert summary["succeeded"] == 8

        # standby=0 default: blocking path, today's behavior.
        created = coord._test_worker_stub.created
        assert len(created) == 3, (
            "2 primaries + 1 synchronously constructed (blocking) "
            f"replacement expected at standby=0; got {len(created)}"
        )
        sb = summary["standby"]
        assert sb["configured"] == 0
        assert sb["promotions"] == 0
        assert sb["refills_started"] == 0
        assert sb["fallbacks"] == 0, (
            "standby=0 is today's behavior, not a pool fallback"
        )

    def test_run_terminates_when_all_workers_die(self, patched_ray, monkeypatch):
        """If every worker fails replacement, run() must terminate rather than spinning forever."""
        # Both workers poison on first call, and replacement always fails.
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["poison"], ["poison"]],
            n_tasks=8,
            max_retries=2,
        )
        # Force every replacement attempt to fail.
        monkeypatch.setattr(
            coord, "_replace_worker",
            lambda workers, idx, max_attempts=3: (
                workers.__setitem__(idx, None) or False
            ),
        )
        summary = coord.run(tasks)
        assert summary["total"] == 8, (
            "Every task must be accounted for in the summary, even when "
            "all workers die before completion."
        )
        assert summary["failed"] == 8
        assert summary["succeeded"] == 0


class TestStandbyPool:
    """Pre-loaded standby workers make replacement an O(1) handle swap."""

    def test_ctor_default_is_zero(self):
        coord = make_coordinator()
        assert coord.standby == 0

    def test_ctor_rejects_negative_standby(self):
        with pytest.raises(ValueError, match="standby"):
            make_coordinator(standby=-1)

    def test_startup_creates_standby_pool(self, patched_ray):
        """--standby N builds N extra workers from the same factory at startup; a
        clean run never touches them."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 8, ["ok"] * 8],
            n_tasks=8,
            standby=2,
        )
        summary = coord.run(tasks)
        assert summary["succeeded"] == 8

        created = coord._test_worker_stub.created
        assert len(created) == 4, "2 primaries + 2 standbys"
        # Standby worker_ids continue past the slot ids.
        assert [a.worker_id for a in created] == [0, 1, 2, 3]
        # Standbys stayed idle on a clean run.
        assert created[2].completed == 0 and created[3].completed == 0

        sb = summary["standby"]
        assert sb["configured"] == 2
        assert sb["promotions"] == 0
        assert sb["refills_started"] == 0
        assert sb["fallbacks"] == 0
        assert sb["final_pool_size"] == 2
        # Pool size over time: one startup sample at size 2, no changes.
        assert [size for _, size in sb["pool_timeline"]] == [2]

    def test_promotion_never_calls_the_blocking_creator(
        self, patched_ray, monkeypatch
    ):
        """With a ready standby, replacement must be a handle swap: the blocking
        construct-and-wait path must not run at all, and the promoted worker must
        receive the pending (retried) batch through the existing dispatch path."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["poison"], ["ok"] * 20],
            n_tasks=8,
            max_retries=2,
            standby=1,
        )

        def _forbidden(*_a, **_k):
            raise AssertionError(
                "blocking replacement ran despite a ready standby"
            )

        monkeypatch.setattr(coord, "_blocking_replace", _forbidden)
        summary = coord.run(tasks)

        assert summary["succeeded"] == 8
        sb = summary["standby"]
        assert sb["promotions"] == 1
        assert sb["fallbacks"] == 0

        created = coord._test_worker_stub.created
        # 2 primaries + 1 standby + 1 async refill; nothing else.
        assert len(created) == 4
        promoted = created[2]  # the startup standby (worker_id 2)
        assert promoted.completed > 0, (
            "The promoted standby must receive the re-queued batch "
            "through the normal dispatch path"
        )

    def test_refill_completes_on_a_later_loop_iteration(self, patched_ray):
        """After a promotion, the coordinator submits a replacement standby
        asynchronously and polls its readiness ping with ray.wait(timeout=0)"""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["poison"], ["ok"] * 30],
            n_tasks=24,
            max_retries=2,
            batch_size=4,
            delay=0.05,               # keeps the loop alive past the refill
            standby=1,
            refill_health_delay=0.05,
        )
        summary = coord.run(tasks)
        assert summary["succeeded"] == 24

        sb = summary["standby"]
        assert sb["promotions"] == 1
        assert sb["refills_started"] == 1
        assert sb["refills_completed"] == 1, (
            "The refill's readiness ping resolved during the run and "
            "must have been harvested by the non-blocking poll"
        )
        assert sb["refill_failures"] == 0
        assert sb["final_pool_size"] == 1
        # Pool size over time: 1 at startup -> 0 on promotion -> 1 on refill
        assert [size for _, size in sb["pool_timeline"]] == [1, 0, 1]
        # The refill became ready strictly after the promotion drained the pool.
        t_drain = sb["pool_timeline"][1][0]
        t_refill = sb["pool_timeline"][2][0]
        assert t_refill > t_drain

    def test_empty_pool_falls_back_to_blocking(self, patched_ray, caplog):
        """Two replacements, one standby, refill too slow: the second replacement
        finds the pool empty and must fall back to today's blocking path - and
        log that it did."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["poison"], ["poison"]],
            n_tasks=8,
            max_retries=2,
            standby=1,
            refill_health_delay=30.0,  # never ready within this run
        )
        with caplog.at_level("WARNING"):
            summary = coord.run(tasks)

        assert summary["succeeded"] == 8
        sb = summary["standby"]
        assert sb["promotions"] == 1
        assert sb["fallbacks"] == 1
        assert sb["refills_started"] == 1
        assert sb["refills_completed"] == 0
        assert sb["refills_in_flight_at_exit"] == 1
        assert any(
            "falling back to blocking replacement" in rec.message
            for rec in caplog.records
        ), "Empty-pool fallback must be logged"

        created = coord._test_worker_stub.created
        # 2 primaries + 1 standby + 1 refill + 1 blocking replacement.
        assert len(created) == 5

    def test_promotion_slot_bookkeeping(self, patched_ray):
        """Slot bookkeeping across a promotion: the dead actor gets no further
        batches, every task terminates exactly once."""
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["poison"], ["ok"] * 20],
            n_tasks=8,
            batch_size=4,
            max_retries=2,
            standby=1,
        )
        summary = coord.run(tasks)
        assert summary["succeeded"] == 8
        assert summary["failed"] == 0

        created = coord._test_worker_stub.created
        poisoned = created[0]
        # The poisoned actor saw exactly its first batch and nothing after
        assert poisoned.completed == 0
        assert poisoned.failed == 4

        # Every task reached exactly one terminal state: all executions that produced.
        assert sum(a.completed for a in created) == 8

        stats = summary["worker_stats"]
        assert len(stats) == 2, "exactly n_workers live slots at exit"
        ids = sorted(ws["worker_id"] for ws in stats)
        assert ids == [1, 2], (
            "slot 0 must now hold the promoted standby (worker_id 2); "
            f"got worker_ids {ids}"
        )


class TestCoordinatorShardedAggregator:
    """run() with aggregator_shards > 1: the facade fans results out to N shard."""

    def test_ctor_default_is_single_shard(self):
        coord = make_coordinator()
        assert coord.aggregator_shards == 1, (
            "Default must stay 1 until a sweep measures a better "
            "value."
        )

    def test_ctor_rejects_zero_shards(self):
        with pytest.raises(ValueError, match="aggregator_shards"):
            make_coordinator(aggregator_shards=0)

    def test_sharded_run_completes_all_tasks(self, patched_ray):
        _FakeAggregator.instances.clear()
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 8, ["ok"] * 8],
            n_tasks=8,
            batch_size=4,
            aggregator_shards=3,
        )
        summary = coord.run(tasks)

        # Merged summary comes from aggregator._summarize_state.
        assert summary["total"] == 8
        assert summary["succeeded"] == 8
        assert summary["failed"] == 0
        assert summary["aggregator_shards"] == 3
        assert len(summary["results_files"]) == 3
        assert "shard0" in summary["results_files"][0]

        shards = _FakeAggregator.instances
        assert len(shards) == 3

        # Every task reaches exactly one shard, exactly once.
        recorded = [r.task_id for s in shards for r in s.results]
        assert sorted(recorded) == sorted(t.task_id for t in tasks)

        # Routing is the facade's stable hash, per shard.
        for i, shard in enumerate(shards):
            for r in shard.results:
                assert shard_index(r.task_id, 3) == i, (
                    f"task {r.task_id} landed on shard {i} but hashes "
                    f"to {shard_index(r.task_id, 3)}"
                )

        # finalize() sealed every shard.
        assert all(s.closed for s in shards)

    def test_sharded_run_batches_shard_calls(self, patched_ray):
        """One record_batch call per shard per completed batch - never one call per
        result. 8 tasks in 2 batches of 4 across 3 shards can produce at most 6
        shard calls; 8 calls would mean the per-result path came back."""
        _FakeAggregator.instances.clear()
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 8, ["ok"] * 8],
            n_tasks=8,
            batch_size=4,
            aggregator_shards=3,
        )
        summary = coord.run(tasks)
        assert summary["total"] == 8

        shards = _FakeAggregator.instances
        total_calls = sum(s.record_batch_calls for s in shards)
        n_batches = 2
        assert total_calls <= n_batches * 3, (
            f"{total_calls} shard calls for {n_batches} batches x 3 "
            "shards max - record_batch must group per shard, not "
            "submit per result"
        )
        assert total_calls < 8, "looks like one call per result"

    def test_single_shard_run_uses_requested_output_path(self, patched_ray):
        """N=1 must be today's behavior: one shard actor at exactly the requested
        output path, no '.shardK' suffix, no merged-summary extras."""
        _FakeAggregator.instances.clear()
        coord, tasks = _coordinator_with_fake_workers(
            plan_per_worker=[["ok"] * 8, ["ok"] * 8],
            n_tasks=8,
        )
        summary = coord.run(tasks)
        shards = _FakeAggregator.instances
        assert len(shards) == 1
        assert shards[0].output_path == "results/test.jsonl"
        # Passthrough summary: the shard's own dict, untouched.
        assert summary["results_file"] == "results/test.jsonl"
        assert "results_files" not in summary
        assert "aggregator_shards" not in summary
