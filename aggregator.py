from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
import zlib
from pathlib import Path

import ray

from types_ import EvalResult
from scoring import RubricScorer

logger = logging.getLogger(__name__)

# Per-shard latency-sample reservoir cap (the reservoir cap). 100k floats is
# ~800 KB per shard - cheap - while covering every run this harness has
# actually done exactly.
RESERVOIR_CAP = 100_000


def _percentile(sorted_values: list[float], q: float) -> float:
    """nearest-rank percentile: ceil(q*n)-1; int(q*n) indexes the max (p100) at n=100."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(n - 1, max(0, math.ceil(q * n) - 1))
    return sorted_values[idx]


def shard_index(task_id: str, n_shards: int) -> int:
    """Stable shard routing: crc32(task_id) % n_shards."""
    return zlib.crc32(task_id.encode("utf-8")) % n_shards


def _reservoir_append(
    samples: list[float],
    seen_before: int,
    value: float,
    rng: random.Random,
) -> None:
    """Algorithm R: keep a uniform sample of size RESERVOIR_CAP. seen_before = how
    many values were observed before this one."""
    if len(samples) < RESERVOIR_CAP:
        samples.append(value)
        return
    j = rng.randint(0, seen_before)  # inclusive
    if j < RESERVOIR_CAP:
        samples[j] = value


def _summarize_state(
    state: dict,
    *,
    total_tasks: int,
    elapsed: float,
    results_file: str,
) -> dict:
    """Turn a (possibly merged) shard state into the summary dict."""
    count = state["count"]
    succeeded = state["succeeded"]

    mean_score = state["score_sum"] / succeeded if succeeded > 0 else 0.0
    mean_latency = state["latency_sum"] / succeeded if succeeded > 0 else 0.0
    mean_tokens = state["tokens_total"] / succeeded if succeeded > 0 else 0.0
    p99 = _percentile(sorted(state["latency_samples"]), 0.99)
    batch_sorted = sorted(state["batch_latency_samples"])
    p99_batch = _percentile(batch_sorted, 0.99)
    mean_batch = (
        sum(batch_sorted) / len(batch_sorted) if batch_sorted else 0.0
    )

    condition_stats = RubricScorer.aggregate_condition_scores(
        state["condition_scores"]
    )

    success_rate = succeeded / count if count > 0 else 0.0
    throughput = count / elapsed if elapsed > 0 else 0.0

    return {
        "total": count,
        "total_intake": total_tasks,
        "succeeded": succeeded,
        "failed": state["failed"],
        "stopped_early": state["stopped_early"],
        "success_rate": success_rate,
        "mean_score": mean_score,
        "min_score": (
            state["score_min"] if state["score_min"] != float("inf") else 0.0
        ),
        "max_score": (
            state["score_max"] if state["score_max"] != float("-inf") else 0.0
        ),
        "mean_latency_s": mean_latency,
        "p99_latency_s": p99,
        "mean_batch_latency_s": mean_batch,
        "p99_batch_latency_s": p99_batch,
        "mean_tokens_generated": mean_tokens,
        "total_elapsed_s": elapsed,
        "throughput_per_s": throughput,
        "results_file": results_file,
        "per_worker_counts": state["per_worker"],
        "condition_stats": condition_stats,
    }


def _merge_states(states: list[dict]) -> dict:
    """Merge per-shard raw states: counters summed, sample lists concatenated,
    min/max across shards."""
    merged: dict = {
        "count": 0,
        "succeeded": 0,
        "failed": 0,
        "stopped_early": 0,
        "score_sum": 0.0,
        "score_min": float("inf"),
        "score_max": float("-inf"),
        "latency_sum": 0.0,
        "latency_samples": [],
        "latency_seen": 0,
        "batch_latency_samples": [],
        "batch_latency_seen": 0,
        "tokens_total": 0,
        "per_worker": {},
        "condition_scores": [],
    }
    for s in states:
        merged["count"] += s["count"]
        merged["succeeded"] += s["succeeded"]
        merged["failed"] += s["failed"]
        merged["stopped_early"] += s["stopped_early"]
        merged["score_sum"] += s["score_sum"]
        merged["score_min"] = min(merged["score_min"], s["score_min"])
        merged["score_max"] = max(merged["score_max"], s["score_max"])
        merged["latency_sum"] += s["latency_sum"]
        merged["latency_samples"].extend(s["latency_samples"])
        merged["latency_seen"] += s["latency_seen"]
        merged["batch_latency_samples"].extend(s["batch_latency_samples"])
        merged["batch_latency_seen"] += s["batch_latency_seen"]
        merged["tokens_total"] += s["tokens_total"]
        for worker_id, n in s["per_worker"].items():
            merged["per_worker"][worker_id] = (
                merged["per_worker"].get(worker_id, 0) + n
            )
        merged["condition_scores"].extend(s["condition_scores"])
    return merged


class ResultsAggregatorImpl:
    """Collects EvalResults, computes runstats, writes JSONL."""

    def __init__(
        self,
        total_tasks: int,
        output_path: str = "results/results.jsonl",
    ) -> None:
        self.total_tasks = total_tasks
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("a", encoding="utf-8")
        self.start_time = time.perf_counter()

        self._count = 0
        self._succeeded = 0
        self._failed = 0
        self._score_sum = 0.0
        self._score_min = float("inf")
        self._score_max = float("-inf")
        self._latency_sum = 0.0
        # Uniform reservoirs (Algorithm R)
        self._latency_samples: list[float] = []
        self._latency_seen = 0
        self._batch_latency_samples: list[float] = []
        self._batch_latency_seen = 0
        self._reservoir_rng = random.Random(0xD5)
        self._tokens_total = 0
        self._stopped_early_count = 0
        self._per_worker: dict[int, int] = {}
        self._all_condition_scores: list[dict[str, float]] = []

        logger.info(f"Aggregator: writing results to {self.output_path}")

    def record_batch(self, results: list[EvalResult]) -> None:
        """Record a whole completed batch in ONE actor call: per-result counters +."""
        for result in results:
            self._ingest(result)
        self._fh.flush()

    def add_result(self, result: EvalResult) -> None:
        """Single-result back-compat shim; equivalent to record_batch([result])."""
        self.record_batch([result])

    def _ingest(self, result: EvalResult) -> None:
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
            _reservoir_append(
                self._latency_samples,
                self._latency_seen,
                result.latency_seconds,
                self._reservoir_rng,
            )
            self._latency_seen += 1
            if result.batch_latency_seconds is not None:
                # task-weighted: each task contributes its batch's wall time.
                _reservoir_append(
                    self._batch_latency_samples,
                    self._batch_latency_seen,
                    result.batch_latency_seconds,
                    self._reservoir_rng,
                )
                self._batch_latency_seen += 1
            self._tokens_total += result.tokens_generated
            if result.stopped_early:
                self._stopped_early_count += 1
            if result.condition_scores:
                self._all_condition_scores.append(result.condition_scores)
        else:
            self._failed += 1

        self._fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - self.start_time
        rate = self._count / elapsed if elapsed > 0 else 0.0
        status = "OK  " if result.succeeded else "FAIL"
        logger.info(
            f"[{self._count:4d}/{self.total_tasks}] {status} "
            f"task={result.task_id} score={result.score:.3f} "
            f"latency={result.latency_seconds:.2f}s "
            f"worker={result.worker_id} rate={rate:.1f}/s"
        )

    def get_shard_state(self) -> dict:
        """Raw counters and samples for driver-side merging. Lists are copied so the
        caller can't mutate actor state."""
        return {
            "count": self._count,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "stopped_early": self._stopped_early_count,
            "score_sum": self._score_sum,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "latency_sum": self._latency_sum,
            "latency_samples": list(self._latency_samples),
            "latency_seen": self._latency_seen,
            "batch_latency_samples": list(self._batch_latency_samples),
            "batch_latency_seen": self._batch_latency_seen,
            "tokens_total": self._tokens_total,
            "per_worker": dict(self._per_worker),
            "condition_scores": list(self._all_condition_scores),
        }

    def get_summary(self) -> dict:
        """full summary shape, including zero-result case."""
        elapsed = time.perf_counter() - self.start_time
        return _summarize_state(
            self.get_shard_state(),
            total_tasks=self.total_tasks,
            elapsed=elapsed,
            results_file=str(self.output_path),
        )

    def close(self) -> None:
        """Close the JSONL handle (idempotent)."""
        if not self._fh.closed:
            self._fh.close()


ResultsAggregator = ray.remote(ResultsAggregatorImpl)


class ShardedAggregator:
    """Plain-Python driver-side facade over N ResultsAggregator actors."""

    def __init__(
        self,
        total_tasks: int,
        output_path: str = "results/results.jsonl",
        n_shards: int = 1,
        aggregator_cls=ResultsAggregator,
    ) -> None:
        if n_shards < 1:
            raise ValueError(f"n_shards must be >= 1, got {n_shards}")
        self.total_tasks = total_tasks
        self.output_path = Path(output_path)
        self.n_shards = n_shards
        # Shared clock for merged elapsed/throughput.
        self._start_time = time.perf_counter()

        if n_shards == 1:
            # Identical to the single-actor layout: one actor, the exact requested path.
            self.shard_paths = [self.output_path]
        else:
            self.shard_paths = [
                self.output_path.with_name(
                    f"{self.output_path.stem}.shard{i}"
                    f"{self.output_path.suffix}"
                )
                for i in range(n_shards)
            ]
        # total_tasks is passed whole to every shard: it only feeds the
        self._shards = [
            aggregator_cls.remote(
                total_tasks=total_tasks, output_path=str(path)
            )
            for path in self.shard_paths
        ]

    def shard_for(self, task_id: str) -> int:
        return shard_index(task_id, self.n_shards)

    def record_batch(self, results: list[EvalResult]) -> list:
        """Group by shard, submit ONE record_batch call per shard touched, return
        the ObjectRefs (fire-and-forget; caller drains)."""
        if not results:
            return []
        if self.n_shards == 1:
            return [self._shards[0].record_batch.remote(results)]
        groups: dict[int, list[EvalResult]] = {}
        for result in results:
            groups.setdefault(self.shard_for(result.task_id), []).append(
                result
            )
        # Sorted for a deterministic submission order.
        return [
            self._shards[idx].record_batch.remote(group)
            for idx, group in sorted(groups.items())
        ]

    def get_summary(self) -> dict:
        if self.n_shards == 1:
            return ray.get(self._shards[0].get_summary.remote())
        states = ray.get(
            [shard.get_shard_state.remote() for shard in self._shards]
        )
        elapsed = time.perf_counter() - self._start_time
        summary = _summarize_state(
            _merge_states(states),
            total_tasks=self.total_tasks,
            elapsed=elapsed,
            results_file=str(self.output_path),
        )
        summary["results_files"] = [str(p) for p in self.shard_paths]
        summary["aggregator_shards"] = self.n_shards
        return summary

    def finalize(self) -> str:
        """Seal writes and materialise the single results file."""
        if self.n_shards == 1:
            return str(self.output_path)
        ray.get([shard.close.remote() for shard in self._shards])
        existing = [p for p in self.shard_paths if p.exists()]
        if existing:
            with self.output_path.open("w", encoding="utf-8") as out:
                for path in existing:
                    with path.open("r", encoding="utf-8") as fh:
                        shutil.copyfileobj(fh, out)
        return str(self.output_path)
