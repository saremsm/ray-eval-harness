from __future__ import annotations

import logging
from collections import deque
import ray
from aggregator import ResultsAggregator
from types_ import EvalResult, EvalTask
from utils import make_batches
from worker import HFWorker
logger = logging.getLogger(__name__)

def validate_backend(worker: object) -> None:
    """hasattr, not isinstance() (fails on Ray handles); fail at startup, not first batch."""
    required = ("evaluate_batch", "health_check", "get_stats")
    missing = [m for m in required if not hasattr(worker, m)]
    if missing:
        raise AttributeError(
            f"Worker {worker!r} is missing required methods: {missing}. "
            "All workers must satisfy the EvalBackend Protocol."
        )
    
class DistributedEvalCoordinator:
    """Work-Stealing coordinator across a pool of EvalWorker actors."""
    def __init__(
        self, n_workers: int, model_name: str = "distilgpt2",
        output_path: str = "results/results.jsonl", batch_size: int = 4,
        aggregator_cls=ResultsAggregator,
    ) -> None:
        self.n_workers = n_workers
        self.model_name = model_name
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

        def submit(worker_idx: int, batch: list[EvalTask]) -> None:
            ref = workers[worker_idx].evaluate_batch.remote(batch)
            active[ref] = (worker_idx, batch)
        
        # Fill the pipeline: every live worker gets its first batch.
        available = list(range(self.n_workers))
        while pending and available:
            batch = pending.popleft()
            submit(available.pop(0), batch)
        
        while active:
            done_refs, _ = ray.wait(
                list(active.keys()),
                num_returns=1,
            )
            done_ref = done_refs[0]
            worker_idx, batch = active.pop(done_ref)
        
            results: list[EvalResult] = ray.get(done_ref)
            self._handle_success(results, aggregator)
        
            self._assign_next(worker_idx, pending, submit)
        
        live_workers = [w for w in workers if w is not None]
        worker_stats = ray.get([w.get_stats.remote() for w in live_workers])
        summary = ray.get(aggregator.get_summary.remote())
        summary["worker_stats"] = worker_stats
        return summary
    
    # Helpers
    
    def _record(self, aggregator: object, results: list[EvalResult]) -> None:
        for result in results:
            ray.get(aggregator.add_result.remote(result))
    
    def _handle_success(self, results: list[EvalResult], aggregator: object,) -> None:
        self._record(aggregator, results)
    
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
