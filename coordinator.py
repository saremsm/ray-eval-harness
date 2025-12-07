from __future__ import annotations

import logging
import time
from collections import deque

import ray

from aggregator import ResultsAggregator
from types_ import EvalResult, EvalTask, FailureKind

from utils import make_batches
from worker import HFWorker, VLLMWorker

logger = logging.getLogger(__name__)

# Wall-clock threshold for treating an outstanding ref as hung.
HUNG_REF_THRESHOLD_S = 240.0

# Poll interval when no refs are completing.
RETRY_POLL_INTERVAL = 1.0

# Per-backend batch-size defaults: HF over-batches past 4; vLLM's continuous
# batching starves below 64.
DEFAULT_BATCH_SIZE_HF = 4
DEFAULT_BATCH_SIZE_VLLM = 64

_DETERMINISTIC_PATTERNS = (
    "token indices sequence length",
    "index out of range",
)

def validate_backend(worker: object) -> None:
    """hasattr, not isinstance() (fails on Ray handles); fail at startup, not first batch."""
    required = ("evaluate_batch", "health_check", "get_stats")
    missing = [m for m in required if not hasattr(worker, m)]
    if missing:
        raise AttributeError(
            f"Worker {worker!r} is missing required methods: {missing}. "
            "All workers must satisfy the EvalBackend Protocol."
        )

def classify_failure(exc: Exception) -> FailureKind:
    """TRANSIENT = retry; DETERMINISTIC = error will reccur, don't bother."""
    msg = str(exc).lower()
    for pattern in _DETERMINISTIC_PATTERNS:
        if pattern in msg:
            return FailureKind.DETERMINISTIC
    return FailureKind.TRANSIENT

def backoff_seconds(retry_count: int) -> float:
    """exponential backoff, capped at 8s."""
    return min(8.0, 0.5 * (2 ** retry_count))

class DistributedEvalCoordinator:
    """Work-Stealing coordinator across a pool of EvalWorker actors. aggregator_cls."""
    def __init__(
        self,
        n_workers: int,
        model_name: str = "distilgpt2",
        backend: str = "hf",
        max_retries: int = 2,
        task_timeout: float = 60.0,
        output_path: str = "results/results.jsonl",
        batch_size: int | None = None,
        aggregator_cls=ResultsAggregator,
        worker_cls=None,
        worker_kwargs: dict | None = None,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.n_workers = n_workers
        self.model_name = model_name
        self.backend = backend
        self.max_retries = max_retries
        # task_timeout is per-BATCH.
        self.task_timeout = task_timeout
        self.output_path = output_path
        self.tensor_parallel_size = tensor_parallel_size
        if batch_size is None:
            batch_size = (
                DEFAULT_BATCH_SIZE_VLLM if backend == "vllm"
                else DEFAULT_BATCH_SIZE_HF
            )
        self.batch_size = batch_size

        self._aggregator_cls = aggregator_cls
        self._worker_cls = worker_cls or (
            VLLMWorker if backend == "vllm" else HFWorker
        )
        self._extra_worker_kwargs = worker_kwargs or {}
    
    # Public Interface

    def run(self, tasks: list[EvalTask]) -> dict:
        """run all tasks, return summary+worker_stats. results stream to JSONL non-
        blocking retry: failed batch -> deferred[] with wake-up time. back to
        pending[] when time hits. main loop keeps draining"""
        workers = self._create_workers()
        aggregator = self._aggregator_cls.remote( 
            total_tasks=len(tasks), 
            output_path=self.output_path,
        )
        # Ready to dispatch.
        pending: deque[tuple[list[EvalTask], int]] = deque(
            (batch, 0) for batch in make_batches(tasks, self.batch_size)
        )

        # Scheduled for a future wake-up. (ready_at, batch, retry_count).
        deferred: deque[tuple[float, list[EvalTask], int]] = deque()

        # ObjectRef -> (worker_index, batch, retry_count)
        active: dict = {}

        # Per-ref submission timestamp; keyed by ref, not worker.
        ref_timeouts: dict = {}

        def submit(
            worker_idx: int,
            batch: list[EvalTask],
            retry_count: int = 0,
        ) -> None:
            if workers[worker_idx] is None:
                # Slot is dead; re-queue so dispatch_pending_to_idle picks it up.
                pending.append((batch, retry_count))
                return
            ref = workers[worker_idx].evaluate_batch.remote(batch)
            active[ref] = (worker_idx, batch, retry_count)
            ref_timeouts[ref] = time.monotonic()

        def promote_ready_retries() -> None:
            """Move deferred batches whose wake-up has arrived into pending."""
            now = time.monotonic()
            still_waiting: list[tuple[float, list[EvalTask], int]] = []
            for ready_at, batch, retry_count in deferred:
                if ready_at <= now:
                    pending.append((batch, retry_count))
                else:
                    still_waiting.append((ready_at, batch, retry_count))
            deferred.clear()
            deferred.extend(still_waiting)

        def dispatch_pending_to_idle() -> None:
            """Hand pending work to idle workers - a deferred retry can promote back
            to pending after every worker has gone idle."""
            if not pending:
                return
            busy = {worker_idx for worker_idx, _, _ in active.values()}
            for i in range(self.n_workers):
                if not pending:
                    return
                if workers[i] is None or i in busy:
                    continue
                batch_to_send, rc = pending.popleft()
                submit(i, batch_to_send, rc)

        # Fill the pipeline
        available = [i for i in range(self.n_workers) if workers[i] is not None]
        while pending and available:
            batch, retry_count = pending.popleft()
            submit(available.pop(0), batch, retry_count)
        
        while active or deferred or pending:
            promote_ready_retries()
            dispatch_pending_to_idle()

            if not active:
                # Everything in flight has drained and we're only waiting on a deferred retry.
                time.sleep(RETRY_POLL_INTERVAL)
                continue

            done_refs, _ = ray.wait(
                list(active.keys()),
                num_returns=1,
                timeout=RETRY_POLL_INTERVAL,
            )

            if done_refs:
                done_ref = done_refs[0]
                worker_idx, batch, retry_count = active.pop(done_ref)
                ref_timeouts.pop(done_ref, None)

                try:
                    results: list[EvalResult] = ray.get(done_ref)
                    self._handle_success(results, aggregator)
                except Exception as exc:
                    should_retry = self._handle_failure(
                        exc=exc,
                        batch=batch,
                        worker_idx=worker_idx,
                        retry_count=retry_count,
                        workers=workers,
                        aggregator=aggregator,
                    )
                    if should_retry:
                        # Non-blocking: schedule for later, free the worker now.
                        ready_at = time.monotonic() + backoff_seconds(retry_count)
                        deferred.append((ready_at, batch, retry_count + 1))

                self._assign_next(worker_idx, pending, submit)
            else:
                # Scan every iteration, not only when ray.wait returns empty: under steady
                # completions that gate starves exactly when the system is busiest.
                self._handle_timeouts(
                    active=active,
                    ref_timeouts=ref_timeouts,
                    workers=workers,
                    pending=pending,
                    submit=submit,
                    aggregator=aggregator,
                )
        
        live_workers = [w for w in workers if w is not None]
        worker_stats = ray.get([w.get_stats.remote() for w in live_workers])
        summary = ray.get(aggregator.get_summary.remote())
        summary["worker_stats"] = worker_stats
        return summary

    # Hung-worker handling
    def _handle_timeouts(
        self,
        active: dict,
        ref_timeouts: dict,
        workers: list,
        pending: deque,
        submit,
        aggregator: object,
    ) -> None:
        """evict refs older than HUNG_REF_THRESHOLD_S, replace owning worker."""
        now = time.monotonic()
        # Snapshot before iterating: active is mutated during eviction.
        for ref in list(active.keys()):
            if ref not in active:
                continue

            submitted_at = ref_timeouts.get(ref, now)
            age = now - submitted_at
            if age < HUNG_REF_THRESHOLD_S:
                continue

            worker_idx, batch, retry_count = active.pop(ref)
            ref_timeouts.pop(ref, None)

            logger.error(
                f"Worker {worker_idx}: ref outstanding for {age:.0f}s "
                f"(threshold {HUNG_REF_THRESHOLD_S:.0f}s); replacing worker"
            )

            task_max_retries = (
                batch[0].max_retries
                if batch[0].max_retries is not None
                else self.max_retries
            )
            if retry_count < task_max_retries:
                pending.append((batch, retry_count + 1))
                logger.info(
                    f"Re-queued {len(batch)} tasks "
                    f"(retry_count now {retry_count + 1}, "
                    f"max {task_max_retries})"
                )
            else:
                logger.error(
                    f"Batch hung {retry_count + 1} time(s); retry budget "
                    f"exhausted - recording {len(batch)} terminal failures"
                )
                self._record(aggregator, [
                    EvalResult(
                        task_id=task.task_id,
                        score=0.0,
                        response="",
                        latency_seconds=0.0,
                        batch_latency_seconds=None,
                        failed=True,
                        worker_id=worker_idx,
                        error=(
                            f"batch hung >= {HUNG_REF_THRESHOLD_S:.0f}s "
                            f"{retry_count + 1} time(s); retry budget exhausted"
                        ),
                        failure_kind=FailureKind.TRANSIENT,
                    )
                    for task in batch
                ])

            self._replace_worker(workers, worker_idx)

            # Give the replacement its first batch right away if available.
            if pending:
                batch_to_send, rc = pending.popleft()
                submit(worker_idx, batch_to_send, rc)

    # Helpers
    def _record(
        self, 
        aggregator: object, 
        results: list[EvalResult]
    ) -> None:
        for result in results:
            ray.get(aggregator.add_result.remote(result))
    
    def _handle_success(self, results: list[EvalResult], aggregator: object,) -> None:
        self._record(aggregator, results)

    def _handle_failure(
        self,
        exc: Exception,
        batch: list[EvalTask],
        worker_idx: int,
        retry_count: int,
        workers: list,
        aggregator: object,
    ) -> bool:
        """returns True if batch should retry."""
        kind = classify_failure(exc)
        logger.error(f"Batch failed on worker {worker_idx} ({kind.name}): {exc}")
        
        self._check_and_replace_if_poisoned(workers, worker_idx)

        task_max_retries = (
            batch[0].max_retries
            if batch[0].max_retries is not None
            else self.max_retries
        )
        should_retry = (
            kind == FailureKind.TRANSIENT
            and retry_count < task_max_retries
        )
        if should_retry:
            logger.info(
                f"Scheduling retry {retry_count + 1}/{task_max_retries} "
                f"for {len(batch)} tasks "
                f"(backoff: {backoff_seconds(retry_count):.1f}s)"
            )
            return True

        # Terminal failure: one EvalResult per task.
        self._record(aggregator, [
            EvalResult(
                task_id=task.task_id,
                score=0.0,
                response="",
                latency_seconds=0.0,
                batch_latency_seconds=None,
                failed=True,
                worker_id=worker_idx,
                error=str(exc),
                failure_kind=kind,
            )
            for task in batch
        ])
        return False

    def _check_and_replace_if_poisoned(
        self, 
        workers: list, 
        worker_idx: int,
    ) -> None:
        """health-check the worker; replace if poisoned or unresponsive."""
        if workers[worker_idx] is None:
            return
        try:
            is_healthy = ray.get(
                workers[worker_idx].health_check.remote(),
                timeout=5.0,
            )
        except Exception:
            self._replace_worker(workers, worker_idx)
            return
        if not is_healthy:
            self._replace_worker(workers, worker_idx)

    def _assign_next(
        self, 
        worker_idx: int, 
        pending: deque, 
        submit,
    ) -> None:
        """Give freed worker its next batch, if any."""
        if pending:
            batch, retry_count = pending.popleft()
            submit(worker_idx, batch, retry_count)

    def _worker_factory(self):
        """Return (constructor_callable, extra_kwargs) for configured backend."""
        if self.backend == "vllm":
            tp = self.tensor_parallel_size
            return (
                self._worker_cls.options(num_gpus=tp).remote,
                {"tensor_parallel_size": tp, **self._extra_worker_kwargs},
            )
        return (
            self._worker_cls.remote,
            {**self._extra_worker_kwargs},
        )
    
    def _create_workers(self) -> list:
        logger.info(
            f"Creating {self.n_workers} workers "
            f"(backend={self.backend}, model={self.model_name}, "
            f"batch_size={self.batch_size})..."
        )
        remote_ctor, extra_kwargs = self._worker_factory()
        workers = [
            remote_ctor(
                worker_id=i,
                model_name=self.model_name,
                task_timeout=self.task_timeout,
                **extra_kwargs,
            )
            for i in range(self.n_workers)
        ]

        ray.get([w.health_check.remote() for w in workers])

        for worker in workers:
            validate_backend(worker)
        
        logger.info(f"All {self.n_workers} workers ready and validated")
        return workers
    
    def _replace_worker(
        self,
        workers: list,
        failed_idx: int,
        max_attempts: int = 3,
    ) -> bool:
        """replace a worker with bounded retries."""
        # TODO: still blocks on model load. need standby pool for big models.
        workers[failed_idx] = None  # nothing dispatches here while we work

        logger.warning(f"Replacing worker {failed_idx}")

        for attempt in range(max_attempts):
            try:
                remote_ctor, extra_kwargs = self._worker_factory()
                new_worker = remote_ctor(
                    worker_id=failed_idx,
                    model_name=self.model_name,
                    task_timeout=self.task_timeout,
                    **extra_kwargs,
                )
                ray.get(new_worker.health_check.remote(), timeout=120.0)
                validate_backend(new_worker)
                workers[failed_idx] = new_worker
                logger.info(
                    f"Replacement worker {failed_idx} ready "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                return True
            except Exception as exc:
                logger.error(
                    f"Replacement attempt {attempt + 1}/{max_attempts} "
                    f"for worker {failed_idx} failed: {exc}"
                )
                time.sleep(2.0 * (attempt + 1))

        logger.error(
            f"Worker {failed_idx} could not be replaced after {max_attempts} "
            f"attempts. Marking slot dead; coordinator will skip it."
        )
        return False
