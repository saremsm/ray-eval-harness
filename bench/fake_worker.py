"""FakeLatencyWorker: an async Ray actor that sleeps instead of running a model."""

from __future__ import annotations

import asyncio
import random

import ray

from types_ import EvalResult, EvalTask


class FakeLatencyWorkerImpl:
    """Structurally satisfies EvalBackend."""

    def __init__(
        self,
        worker_id: int,
        model_name: str = "fake-latency",
        task_timeout: float = 60.0,
        latency_s: float = 0.02,
        jitter: float = 0.1,
        failure_rate: float = 0.0,
        decider=None,
        seed: int = 0,
        **_ignored,
    ) -> None:
        self.worker_id = worker_id
        self.model_name = model_name
        self.task_timeout = task_timeout
        self.latency_s = latency_s
        self.jitter = jitter
        self.failure_rate = failure_rate
        self._decider = decider
        # Per-worker RNG for jitter only.
        self._rng = random.Random((seed, worker_id).__repr__())
        self.tasks_completed = 0
        self.tasks_failed = 0

    async def evaluate_batch(self, tasks: list[EvalTask]) -> list[EvalResult]:
        if self.failure_rate > 0.0 and self._decider is not None:
            batch_key = tuple(t.task_id for t in tasks)
            # ObjectRefs are awaitable inside async actor methods.
            if await self._decider.should_fail.remote(
                batch_key, self.failure_rate
            ):
                self.tasks_failed += len(tasks)
                raise RuntimeError(
                    f"injected failure (fake worker {self.worker_id})"
                )

        sleep_s = self.latency_s * (
            1.0 + self.jitter * self._rng.uniform(-1.0, 1.0)
        )
        sleep_s = max(0.0, sleep_s)
        await asyncio.sleep(sleep_s)

        per_task = sleep_s / len(tasks) if tasks else 0.0
        self.tasks_completed += len(tasks)
        return [
            EvalResult(
                task_id=task.task_id,
                score=1.0,
                response="ok",  # tiny output: serialization stays cheap
                latency_seconds=per_task,
                batch_latency_seconds=sleep_s,
                worker_id=self.worker_id,
                tokens_generated=1,
            )
            for task in tasks
        ]

    async def health_check(self) -> bool:
        return True

    async def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "model": self.model_name,
            "backend": "fake-latency",
            "completed": self.tasks_completed,
            "failed": self.tasks_failed,
            "poisoned": False,
        }


# num_cpus=0: fake workers spend their life in asyncio.sleep.
FakeLatencyWorker = ray.remote(num_cpus=0)(FakeLatencyWorkerImpl)
