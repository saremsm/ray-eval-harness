import pytest

from types_ import EvalTask, FailureKind
from coordinator import (
    DistributedEvalCoordinator,
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
