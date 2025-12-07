from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import ray

from types_ import EvalResult
from scoring import RubricScorer

logger = logging.getLogger(__name__)

def _percentile(sorted_values: list[float], q: float) -> float:
    """nearest-rank percentile: ceil(q*n)-1; int(q*n) indexes the max (p100) at n=100."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]

@ray.remote
class ResultsAggregator:
    """Collects EvalResults, computes runstats, writes JSONL."""
    def __init__(
        self, 
        total_tasks: int,
        output_path: str = "results/results.jsonl",
    ) -> None:
        self.total_tasks = total_tasks
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.perf_counter()

        self._count = 0
        self._succeeded = 0
        self._failed = 0
        self._score_sum = 0.0
        self._score_min = float("inf")
        self._score_max = float("-inf")
        self._latency_sum = 0.0
        self._latency_samples: list[float] = []
        self._batch_latency_samples: list[float] = []
        self._tokens_total = 0
        self._stopped_early_count = 0
        self._per_worker: dict[int, int] = {}
        self._all_condition_scores: list[dict[str, float]] = []

        logger.info(f"Aggregator: writing results to {self.output_path}")
    
    def add_result(self, result: EvalResult) -> None:
        self._count += 1
        self._per_worker[result.worker_id] = (
            self._per_worker.get(result.worker_id, 0) + 1
        )

        if result.succeeded:
            self._succeeded += 1
            self._score_sum += result.score
            self._score_min = min(self._score_min, result.score)
            self._score_max = max(self._score_max, result.score)
            self._latency_sum += result.latency_seconds
            self._latency_samples.append(result.latency_seconds)
            if result.batch_latency_seconds is not None:
                # task-weighted: each task contributes its batch's wall time.
                self._batch_latency_samples.append(
                    result.batch_latency_seconds
                )
            self._tokens_total += result.tokens_generated
            if result.stopped_early:
                self._stopped_early_count += 1
            if result.condition_scores:
                self._all_condition_scores.append(result.condition_scores)
        else:
            self._failed += 1
        
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - self.start_time
        rate = self._count / elapsed if elapsed > 0 else 0.0
        status = "OK  " if result.succeeded else "FAIL"
        logger.info(
            f"[{self._count:4d}/{self.total_tasks}] {status} "
            f"task={result.task_id} score={result.score:.3f} "
            f"latency={result.latency_seconds:.2f}s "
            f"worker={result.worker_id} rate={rate:.1f}/s"
        )
    
    def get_summary(self) -> dict:
        """full summary shape, includng zero-result case."""
        elapsed = time.perf_counter() - self.start_time
        mean_score = (
            self._score_sum / self._succeeded if self._succeeded > 0 else 0.0
        )
        mean_latency = (
            self._latency_sum / self._succeeded if self._succeeded > 0 else 0.0
        )
        mean_tokens = (
            self._tokens_total / self._succeeded if self._succeeded > 0 else 0.0
        )
        p99 = _percentile(sorted(self._latency_samples), 0.99)
        batch_sorted = sorted(self._batch_latency_samples)
        p99_batch = _percentile(batch_sorted, 0.99)
        mean_batch = (
            sum(batch_sorted) / len(batch_sorted) if batch_sorted else 0.0
        )

        condition_stats = RubricScorer.aggregate_condition_scores(
            self._all_condition_scores
        )

        success_rate = self._succeeded / self._count if self._count > 0 else 0.0
        throughput = self._count / elapsed if elapsed > 0 else 0.0

        return {
            "total": self._count,
            "total_intake": self.total_tasks,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "stopped_early": self._stopped_early_count,
            "success_rate": success_rate,
            "mean_score": mean_score,
            "min_score": (
                self._score_min if self._score_min != float("inf") else 0.0
            ),
            "max_score": (
                self._score_max if self._score_max != float("-inf") else 0.0
            ),
            "mean_latency_s": mean_latency,
            "p99_latency_s": p99,
            "mean_batch_latency_s": mean_batch,
            "p99_batch_latency_s": p99_batch,
            "mean_tokens_generated": mean_tokens, 
            "total_elapsed_s": elapsed,
            "throughput_per_s": throughput,
            "results_file": str(self.output_path),
            "per_worker_counts": self._per_worker,
            "condition_stats": condition_stats,
        }
