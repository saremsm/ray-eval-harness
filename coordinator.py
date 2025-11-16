from __future__ import annotations

import logging
import time
from collections import deque

import ray

from aggregator import ResultsAggregator
from types_ import EvalResult, EvalTask, FailureKind

from utils import make_batches
from worker import HFWorker

logger = logging.getLogger(__name__)

# Wall-clock threshold for treating a worker as hung.
HUNG_WORKER_THRESHOLD_S = 240.0

# Poll interval when no refs are completing.
RETRY_POLL_INTERVAL = 1.0

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
        self, n_workers: int, model_name: str = "distilgpt2", max_retries: int = 2,
        output_path: str = "results/results.jsonl", batch_size: int = 4,
        aggregator_cls=ResultsAggregator,
    ) -> None:
        self.n_workers = n_workers
        self.model_name = model_name
        self.max_retries = max_retries
        self.output_path = output_path
        self.batch_size = batch_size

        self._aggregator_cls = aggregator_cls
        self._worker_cls = HFWorker
    
    # Public Interface

    def run(self, tasks: list[EvalTask]) -> dict:
        """run all tasks, return summary+worker_stats. results stream to JSONL"""
        workers = self._create_workers()
        aggregator = self._aggregator_cls.remote( 
            total_tasks=len(tasks), output_path=self.output_path,
        )
        # deque for O(1) popleft; pop(0) on a list is O(n).
        pending: deque[list[EvalTask]] = deque(
            make_batches(tasks, self.batch_size)
        )
        
        # ObjectRef -> (worker_index, batch)
        active: dict = {}

        # Consecutive quiet-poll tallies per worker; zeroed on eviction.
        timeout_counts: dict[int, int] = {i: 0 for i in range(self.n_workers)}

        def submit(worker_idx: int, batch: list[EvalTask]) -> None:
            if workers[worker_idx] is None:
                # Slot is dead; re-queue for another worker.
                pending.append(batch)
                return
            ref = workers[worker_idx].evaluate_batch.remote(batch)
            active[ref] = (worker_idx, batch)
        
        # Fill the pipeline: every live worker gets its first batch.
        available = [i for i in range(self.n_workers) if workers[i] is not None]
        while pending and available:
            batch = pending.popleft()
            submit(available.pop(0), batch)
        
        while active or pending:
            if not active:
                time.sleep(RETRY_POLL_INTERVAL)
                continue

            done_refs, _ = ray.wait(
                list(active.keys()),
                num_returns=1,
                timeout=RETRY_POLL_INTERVAL,
            )

            if done_refs:
                done_ref = done_refs[0]
                worker_idx, batch = active.pop(done_ref)

                try:
                    results: list[EvalResult] = ray.get(done_ref)
                    self._handle_success(results, aggregator)
                except Exception as exc:
                    self._handle_failure(
                        exc=exc,
                        batch=batch,
                        worker_idx=worker_idx,
                        workers=workers,
                        aggregator=aggregator,
                    )

                self._assign_next(worker_idx, pending, submit)
            else:
                # Scan every iteration, not only when ray.wait returns empty: under steady
                # completions that gate starves exactly when the system is busiest.
                self._handle_timeouts(
                    active=active,
                    timeout_counts=timeout_counts,
                    workers=workers,
                    pending=pending,
                    submit=submit,
                )
        
        live_workers = [w for w in workers if w is not None]
        worker_stats = ray.get([w.get_stats.remote() for w in live_workers])
        summary = ray.get(aggregator.get_summary.remote())
        summary["worker_stats"] = worker_stats
        return summary

    # Hung-worker handling
    def _handle_timeouts(self, active: dict, timeout_counts: dict,
        workers: list, pending: deque, submit,) -> None:
        """tally consecutive quiet polls per worker; evict and replace a worker once
        its tally crosses the threshold."""
        limit = int(HUNG_WORKER_THRESHOLD_S / RETRY_POLL_INTERVAL)
        for ref in list(active.keys()):
            worker_idx, batch = active[ref]
            timeout_counts[worker_idx] += 1
            if timeout_counts[worker_idx] < limit:
                continue

            active.pop(ref)
            logger.error(
                f"Worker {worker_idx}: {timeout_counts[worker_idx]} quiet "
                f"polls (~{HUNG_WORKER_THRESHOLD_S:.0f}s); replacing worker"
            )
            timeout_counts[worker_idx] = 0
            pending.append(batch)

            self._replace_worker(workers, worker_idx)

            # Give the replacement its first batch right away if available.
            if pending:
                batch_to_send = pending.popleft()
                submit(worker_idx, batch_to_send)

    # Helpers
    def _record(self, aggregator: object, results: list[EvalResult]) -> None:
        for result in results:
            ray.get(aggregator.add_result.remote(result))
    
    def _handle_success(self, results: list[EvalResult], aggregator: object,) -> None:
        self._record(aggregator, results)

    def _handle_failure(
        self, exc: Exception, batch: list[EvalTask], worker_idx: int,
        workers: list, aggregator: object,) -> None:
        """classify, then retry in place with backoff; record terminal failure when
        the budget is exhausted."""
        kind = classify_failure(exc)
        logger.error(f"Batch failed on worker {worker_idx} ({kind.name}): {exc}")

        task_max_retries = (
            batch[0].max_retries
            if batch[0].max_retries is not None
            else self.max_retries
        )
        if kind == FailureKind.TRANSIENT:
            for attempt in range(task_max_retries):
                # Blocking: the whole scheduling loop waits out this sleep.
                time.sleep(backoff_seconds(attempt))
                logger.info(
                    f"Retry {attempt + 1}/{task_max_retries} "
                    f"for {len(batch)} tasks"
                )
                try:
                    results = ray.get(
                        workers[worker_idx].evaluate_batch.remote(batch)
                    )
                    self._handle_success(results, aggregator)
                    return
                except Exception as retry_exc:
                    exc = retry_exc
            kind = classify_failure(exc)
        self._check_and_replace_if_poisoned(workers, worker_idx)

        # Terminal failure: one EvalResult per task.
        self._record(aggregator, [
            EvalResult(
                task_id=task.task_id, 
                score=0.0, 
                response="",
                latency_seconds=0.0, 
                failed=True, 
                worker_id=worker_idx,
                error=str(exc), 
                failure_kind=kind,
            )
            for task in batch
        ])

    def _check_and_replace_if_poisoned(
        self, workers: list, worker_idx: int,) -> None:
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

    def _assign_next(self, worker_idx: int, pending: deque, submit,) -> None:
        """Give freed worker its next batch, if any."""
        if pending:
            batch = pending.popleft()
            submit(worker_idx, batch)
    
    def _create_workers(self) -> list:
        logger.info(
            f"Creating {self.n_workers} workers "
            f"(model={self.model_name}, batch_size={self.batch_size})..."
        )
        workers = [
            self._worker_cls.remote(
                worker_id=i,
                model_name=self.model_name,
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
                new_worker = self._worker_cls.remote(
                    worker_id=failed_idx,
                    model_name=self.model_name,
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
