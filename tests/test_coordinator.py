import time
import pytest
from collections import deque

from types_ import EvalTask, FailureKind
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
        active = {ref: (0, batch)}
        ref_timeouts = {ref: time.monotonic()}

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=deque(),
            submit=lambda *a, **k: None,
        )

        assert ref in active, "Recent ref must not be evicted"
        assert ref in ref_timeouts, "Recent ref's timestamp must be retained"

    def test_old_ref_evicted(self):
        coord = self._make_coord()
        ref = _Ref("r0")
        batch = [make_task("t0")]
        active = {ref: (0, batch)}
        ref_timeouts = {ref: time.monotonic() - HUNG_REF_THRESHOLD_S - 1.0}

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=deque(),
            submit=lambda *a, **k: None,
        )

        assert ref not in active, "Old ref must be removed from active"
        assert ref not in ref_timeouts, (
            "Old ref must be removed from ref_timeouts on eviction"
        )

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
            ref_old: (0, batch_old),
            ref_young: (1, batch_young),
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

        coord._handle_timeouts(
            active=active,
            ref_timeouts=ref_timeouts,
            workers=[_FakeWorker()],
            pending=pending,
            submit=lambda *a, **k: submitted.append(1),
        )

        assert active == {}
        assert ref_timeouts == {}
        assert len(pending) == 0
        assert submitted == []
