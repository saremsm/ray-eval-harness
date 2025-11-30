import hashlib
import random

import ray

from worker import HFWorkerImpl, VLLMWorkerImpl


def _stable_seed(*parts) -> int:
    """Tuple -> stable integer seed. random.Random takes only scalars; built-in
    hash() is per-process randomized (PYTHONHASHSEED)."""
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class FailureDeciderImpl:
    """Shared, deterministic fault-injection oracle."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._attempts: dict[tuple, int] = {}

    def should_fail(self, batch_key: tuple, failure_rate: float) -> bool:
        attempt = self._attempts.get(batch_key, 0)
        self._attempts[batch_key] = attempt + 1
        # failure_rate goes into the seed so different rates produce different
        seed = _stable_seed(self._seed, batch_key, attempt, failure_rate)
        rng = random.Random(seed)
        return rng.random() < failure_rate


FailureDecider = ray.remote(FailureDeciderImpl)


class FaultInjectingHFWorkerImpl(HFWorkerImpl):
    """HFWorkerImpl plus deterministic fault injection."""

    def __init__(self, *, failure_rate: float = 0.0, decider=None, **kwargs):
        super().__init__(**kwargs)
        self.failure_rate = failure_rate
        self._decider = decider

    def _should_fail(self, tasks) -> bool:
        if self._decider is None:
            # Fallback for direct instantiation without a decider.
            return random.random() < self.failure_rate
        batch_key = tuple(t.task_id for t in tasks)
        return ray.get(
            self._decider.should_fail.remote(batch_key, self.failure_rate)
        )

    def evaluate_batch(self, tasks):
        if self._should_fail(tasks):
            self.tasks_failed += len(tasks)
            raise RuntimeError(f"injected failure (worker {self.worker_id})")
        return super().evaluate_batch(tasks)


class FaultInjectingVLLMWorkerImpl(VLLMWorkerImpl):
    """VLLMWorkerImpl plus deterministic fault injection."""

    def __init__(self, *, failure_rate: float = 0.0, decider=None, **kwargs):
        super().__init__(**kwargs)
        self.failure_rate = failure_rate
        self._decider = decider

    async def _should_fail(self, tasks) -> bool:
        if self._decider is None:
            return random.random() < self.failure_rate
        batch_key = tuple(t.task_id for t in tasks)
        # ObjectRefs are awaitable inside async actor methods.
        return await self._decider.should_fail.remote(
            batch_key, self.failure_rate
        )

    async def evaluate_batch(self, tasks):
        if await self._should_fail(tasks):
            self.tasks_failed += len(tasks)
            raise RuntimeError(f"injected failure (worker {self.worker_id})")
        return await super().evaluate_batch(tasks)

# Ray actor wraps - one per leaf class, never the parent.
FaultInjectingHFWorker = ray.remote(FaultInjectingHFWorkerImpl)
FaultInjectingVLLMWorker = ray.remote(FaultInjectingVLLMWorkerImpl)
