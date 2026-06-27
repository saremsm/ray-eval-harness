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


def decider_shard_index(batch_key: tuple, n_shards: int) -> int:
    """Stable decider routing: hash(batch_key) % n_shards."""
    return _stable_seed(batch_key) % n_shards


class _ShardRouterMethod:
    """Makes ShardedFailureDecider.should_fail look exactly like an actor method."""

    def __init__(self, facade: "ShardedFailureDecider") -> None:
        self._facade = facade

    def remote(self, batch_key: tuple, failure_rate: float):
        return self._facade._shard_for(batch_key).should_fail.remote(
            batch_key, failure_rate
        )


class ShardedFailureDecider:
    """Plain-Python facade over N FailureDecider actors, sharded by a stable hash of
    batch_key (see decider_shard_index)."""

    def __init__(
        self,
        seed: int = 0,
        n_shards: int = 1,
        decider_cls=FailureDecider,
    ) -> None:
        if n_shards < 1:
            raise ValueError(f"n_shards must be >= 1, got {n_shards}")
        self.seed = seed
        self.n_shards = n_shards
        # Every shard gets the SAME seed: the decision seed already mixes in
        self._shards = [
            decider_cls.remote(seed=seed) for _ in range(n_shards)
        ]

    def shard_for(self, batch_key: tuple) -> int:
        return decider_shard_index(batch_key, self.n_shards)

    def _shard_for(self, batch_key: tuple):
        if self.n_shards == 1:
            return self._shards[0]
        return self._shards[self.shard_for(batch_key)]

    @property
    def should_fail(self) -> _ShardRouterMethod:
        return _ShardRouterMethod(self)


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
