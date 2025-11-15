from __future__ import annotations

import logging
import time

import ray

from types_ import EvalResult, EvalTask
from scoring import RubricScorer

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 80

HF_MICRO_BATCH_SIZE = 8

# HuggingFace pipeline backend
@ray.remote
class HFWorker:
    """persistent worker: model loads once in __init__, reused across batches."""
    def __init__(self, worker_id: int, model_name: str = "distilgpt2",
        device: int = -1,) -> None:
        """device: -1 CPU, 0+ CUDA. GPU accounting is the coordinator's:
        _worker_factory claims fractional num_gpus (device >= 0) so Ray doesn't
        stack unaccounted replicas."""
        self.worker_id = worker_id
        self.model_name = model_name
        self.device = device
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._scorer = RubricScorer()
        # Retained for truly unrecoverable states / future circuit breaking.
        self._poisoned = False

        logger.info(
            f"Worker {worker_id}: loading '{model_name}' (HF, device={device})..."
        )

        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(
            "text-generation", model=model_name, device=device,
        )
        self._pipeline.tokenizer.pad_token_id = (
            self._pipeline.model.config.eos_token_id
        )

        logger.info(f"Worker {worker_id}: ready")
    def evaluate_batch(self, tasks: list[EvalTask]) -> list[EvalResult]:
        """return_full_text=False -> completion only. latency_seconds is an estimate
        (batch_latency/n); batch_latency_seconds is exact."""
        if self._poisoned:
            raise RuntimeError(
                f"Worker {self.worker_id} is poisoned. "
                "Coordinator must replace worker."
            )
        
        prompts = [task.prompt for task in tasks]
        
        start = time.perf_counter()
        try:
            raw_outputs = self._pipeline(prompts, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                batch_size=min(len(prompts), HF_MICRO_BATCH_SIZE),
                return_full_text=False,
            )
        except Exception:
            self.tasks_failed += len(tasks)
            raise

        batch_latency = time.perf_counter() - start

        estimated_per_task = batch_latency / len(tasks)
        results = []
        for task, output in zip(tasks, raw_outputs):
            response = output[0]["generated_text"]
            score, condition_scores = self._scorer.score(response, task)
            self.tasks_completed += 1
            results.append(EvalResult(
                task_id=task.task_id,
                score=score,
                response=response,
                latency_seconds=estimated_per_task,
                worker_id=self.worker_id,
                condition_scores=condition_scores,
            ))
        
        return results
    
    def health_check(self) -> bool:
        """True if not poisoned."""
        return not self._poisoned
    
    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "model": self.model_name,
            "backend": "hf",
            "completed": self.tasks_completed,
            "failed": self.tasks_failed,
            "poisoned": self._poisoned,
        }
