"""Per-backend health_check semantics."""

import asyncio
import time

import pytest

import worker as worker_mod
from types_ import EvalTask
from worker import HFWorkerImpl, VLLMWorkerImpl, _is_engine_fatal


# Fake exception classes whose NAMES match what vLLM / torch raise.
class EngineDeadError(Exception):
    pass


class AsyncEngineDeadError(Exception):
    pass


class EngineGenerateError(Exception):
    pass


class OutOfMemoryError(RuntimeError):  # torch.cuda.OutOfMemoryError shape
    pass


# Fake engines
class _FakeCompletion:
    def __init__(self, text: str = "blue", token_ids=(1, 2, 3)) -> None:
        self.text = text
        self.token_ids = list(token_ids)


class _FakeRequestOutput:
    def __init__(self) -> None:
        self.outputs = [_FakeCompletion()]


class _EngineBase:
    """Healthy default: errored False, no check_health, generate ok."""

    errored = False

    def generate(self, prompt, sampling_params, request_id):
        async def _stream():
            yield _FakeRequestOutput()

        return _stream()

    async def abort(self, request_id):
        pass


class _EngineErrored(_EngineBase):
    errored = True


class _EngineErroredPropertyRaises(_EngineBase):
    @property
    def errored(self):
        raise RuntimeError("engine core unreachable")


class _EngineAsyncCheckOK(_EngineBase):
    async def check_health(self):
        return None


class _EngineAsyncCheckRaises(_EngineBase):
    async def check_health(self):
        raise EngineDeadError("EngineCore died")


class _EngineAsyncCheckHangs(_EngineBase):
    async def check_health(self):
        await asyncio.sleep(30.0)


class _EngineSyncCheckOK(_EngineBase):
    def check_health(self):
        return None


class _EngineSyncCheckRaises(_EngineBase):
    def check_health(self):
        raise RuntimeError("background loop has errored")


class _EngineSyncCheckHangs(_EngineBase):
    def check_health(self):
        time.sleep(0.5)  # >> patched timeout; leaked thread dies quickly


class _EngineGenerateRaises(_EngineBase):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate(self, prompt, sampling_params, request_id):
        exc = self._exc

        async def _stream():
            raise exc
            yield  # pragma: no cover  (makes this an async generator)

        return _stream()


class _EngineGenerateFailsOnceThenOK(_EngineBase):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def generate(self, prompt, sampling_params, request_id):
        self.calls += 1
        fail = self.calls == 1
        exc = self._exc

        async def _stream():
            if fail:
                raise exc
            yield _FakeRequestOutput()

        return _stream()


class _EngineGenerateHangs(_EngineBase):
    def generate(self, prompt, sampling_params, request_id):
        async def _stream():
            await asyncio.sleep(30.0)
            yield _FakeRequestOutput()

        return _stream()


class _TestableVLLMWorker(VLLMWorkerImpl):
    """VLLMWorkerImpl with the engine injected instead of built."""

    def __init__(self, engine, task_timeout: float = 5.0) -> None:
        self._engine_to_inject = engine
        super().__init__(
            worker_id=0,
            model_name="fake-model",
            task_timeout=task_timeout,
        )

    def _build_engine(self):
        return self._engine_to_inject

    def _build_sampling_params(self):
        return None  # fakes ignore sampling params


def make_task(task_id: str = "t0") -> EvalTask:
    return EvalTask(
        task_id=task_id,
        prompt="The sky is",
        expected_answer="blue",
    )


@pytest.fixture
def small_health_timeout(monkeypatch):
    """Shrink the check_health budget so hang tests finish fast."""
    monkeypatch.setattr(worker_mod, "HEALTH_CHECK_TIMEOUT_S", 0.05)


# Engine attribute probing (errored / check_health)
class TestVLLMHealthCheckEngineProbes:
    def test_healthy_engine_without_check_health(self):
        w = _TestableVLLMWorker(_EngineBase())
        assert asyncio.run(w.health_check()) is True

    def test_errored_true_is_unhealthy(self):
        w = _TestableVLLMWorker(_EngineErrored())
        assert asyncio.run(w.health_check()) is False

    def test_errored_property_raising_is_unhealthy(self):
        w = _TestableVLLMWorker(_EngineErroredPropertyRaises())
        assert asyncio.run(w.health_check()) is False

    def test_async_check_health_ok(self):
        w = _TestableVLLMWorker(_EngineAsyncCheckOK())
        assert asyncio.run(w.health_check()) is True

    def test_async_check_health_raising_is_unhealthy(self):
        w = _TestableVLLMWorker(_EngineAsyncCheckRaises())
        assert asyncio.run(w.health_check()) is False

    def test_async_check_health_hanging_is_unhealthy(
        self, small_health_timeout
    ):
        w = _TestableVLLMWorker(_EngineAsyncCheckHangs())
        start = time.monotonic()
        assert asyncio.run(w.health_check()) is False
        assert time.monotonic() - start < 2.0, (
            "A hanging check_health must be cut off by "
            "HEALTH_CHECK_TIMEOUT_S, not awaited to completion."
        )

    def test_sync_check_health_ok(self):
        w = _TestableVLLMWorker(_EngineSyncCheckOK())
        assert asyncio.run(w.health_check()) is True

    def test_sync_check_health_raising_is_unhealthy(self):
        w = _TestableVLLMWorker(_EngineSyncCheckRaises())
        assert asyncio.run(w.health_check()) is False

    def test_sync_check_health_hanging_is_unhealthy(
        self, small_health_timeout
    ):
        w = _TestableVLLMWorker(_EngineSyncCheckHangs())
        start = time.monotonic()
        assert asyncio.run(w.health_check()) is False
        assert time.monotonic() - start < 2.0


# last_error lifecycle through evaluate_batch / evaluate_with_hooks
class TestVLLMLastErrorTracking:
    def _fail_batch(self, worker, exc_type):
        with pytest.raises(exc_type):
            asyncio.run(worker.evaluate_batch([make_task()]))

    def test_engine_dead_error_sets_last_error_and_health_false(self):
        w = _TestableVLLMWorker(
            _EngineGenerateRaises(EngineDeadError("EngineCore died"))
        )
        self._fail_batch(w, EngineDeadError)
        assert w._last_error is not None
        assert asyncio.run(w.health_check()) is False

    def test_cuda_runtime_error_sets_last_error(self):
        w = _TestableVLLMWorker(_EngineGenerateRaises(RuntimeError(
            "CUDA error: an illegal memory access was encountered"
        )))
        self._fail_batch(w, RuntimeError)
        assert asyncio.run(w.health_check()) is False

    def test_oom_by_class_name_sets_last_error(self):
        w = _TestableVLLMWorker(
            _EngineGenerateRaises(OutOfMemoryError("CUDA out of memory"))
        )
        self._fail_batch(w, OutOfMemoryError)
        assert asyncio.run(w.health_check()) is False

    def test_non_engine_error_does_not_set_last_error(self):
        w = _TestableVLLMWorker(
            _EngineGenerateRaises(ValueError("bad prompt encoding"))
        )
        self._fail_batch(w, ValueError)
        assert w._last_error is None
        assert asyncio.run(w.health_check()) is True

    def test_batch_timeout_does_not_set_last_error(self):
        """Per-batch asyncio.wait_for deadline: engine intact by design (timeouts
        don't poison), so health stays True."""
        w = _TestableVLLMWorker(_EngineGenerateHangs(), task_timeout=0.05)
        self._fail_batch(w, (TimeoutError, asyncio.TimeoutError))
        assert w._last_error is None
        assert asyncio.run(w.health_check()) is True

    def test_successful_batch_clears_last_error(self):
        w = _TestableVLLMWorker(
            _EngineGenerateFailsOnceThenOK(EngineDeadError("died once"))
        )
        self._fail_batch(w, EngineDeadError)
        assert asyncio.run(w.health_check()) is False

        results = asyncio.run(w.evaluate_batch([make_task()]))
        assert len(results) == 1
        assert w._last_error is None
        assert asyncio.run(w.health_check()) is True

    def test_hooks_path_sets_last_error_on_engine_error(self):
        w = _TestableVLLMWorker(
            _EngineGenerateRaises(EngineGenerateError("request blew up"))
        )
        with pytest.raises(EngineGenerateError):
            asyncio.run(w.evaluate_with_hooks(make_task(), hooks=[]))
        assert w._last_error is not None
        assert asyncio.run(w.health_check()) is False

    def test_get_stats_reports_last_error(self):
        w = _TestableVLLMWorker(
            _EngineGenerateRaises(EngineDeadError("EngineCore died"))
        )
        self._fail_batch(w, EngineDeadError)
        stats = asyncio.run(w.get_stats())
        assert stats["poisoned"] is True
        assert "EngineCore died" in stats["last_error"]


# _is_engine_fatal classifier
class TestIsEngineFatal:
    @pytest.mark.parametrize("exc", [
        EngineDeadError("dead"),
        AsyncEngineDeadError("dead (V0 name)"),
        EngineGenerateError("gen failed"),
        OutOfMemoryError("CUDA out of memory"),
        RuntimeError("CUDA error: device-side assert triggered"),
        RuntimeError("cuBLAS error: CUBLAS_STATUS_EXECUTION_FAILED"),
        RuntimeError("NCCL communicator was aborted"),
    ])
    def test_fatal(self, exc):
        assert _is_engine_fatal(exc) is True

    @pytest.mark.parametrize("exc", [
        ValueError("bad sampling params"),
        RuntimeError("worker crashed"),  # generic, no CUDA fingerprint
        TimeoutError("batch of 64 exceeded 60.0s"),
        # Guard: TimeoutError is never fatal even with cuda in the text.
        TimeoutError("cuda kernel still running at deadline"),
        # Python 3.10: asyncio.TimeoutError is a distinct class there.
        asyncio.TimeoutError("cuda kernel still running at deadline"),
    ])
    def test_not_fatal(self, exc):
        assert _is_engine_fatal(exc) is False


# HF path unchanged
class TestHFHealthCheckUnchanged:
    def test_hf_semantics_are_the_poison_flag(self):
        """HF health_check stays `not self._poisoned`; the health-check work must
        not have touched it. __new__ skips __init__ (no model load in tests)."""
        w = HFWorkerImpl.__new__(HFWorkerImpl)
        w._poisoned = False
        assert w.health_check() is True
        w._poisoned = True
        assert w.health_check() is False

    def test_hf_health_check_is_sync(self):
        import inspect
        assert not inspect.iscoroutinefunction(HFWorkerImpl.health_check)
