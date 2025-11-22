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

    def evaluate_with_hooks(
        self,
        task: EvalTask,
        hooks: list,  # list[InterventionHook]
    ) -> EvalResult:
        """per-token hooks via TextIteratorStreamer; generation runs in a background thread."""
        if self._poisoned:
            raise RuntimeError(f"Worker {self.worker_id} poisoned, replace me")

        from threading import Thread
        from transformers import TextIteratorStreamer

        tokenizer = self._pipeline.tokenizer
        model = self._pipeline.model
        inputs = tokenizer(task.prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        thread = Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            ),
            daemon=True,
        )

        start = time.perf_counter()
        thread.start()

        accumulated = ""
        stopped_early = False
        token_count = 0
        for delta in streamer:
            if not delta:
                continue
            accumulated += delta
            token_count += 1
            for hook in hooks:
                hook.on_delta(delta, accumulated)
            if any(hook.should_stop(accumulated) for hook in hooks):
                stopped_early = True
                break

        elapsed = time.perf_counter() - start
        response = accumulated
        score, condition_scores = self._scorer.score(response, task)
        self.tasks_completed += 1

        return EvalResult(
            task_id=task.task_id,
            score=score,
            response=response,
            latency_seconds=elapsed,
            hooked=True,
            worker_id=self.worker_id,
            condition_scores=condition_scores,
            tokens_generated=token_count,
            stopped_early=stopped_early,
        )

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
    
# vLLM Backend
@ray.remote(num_gpus=1)
class VLLMWorker:
    """vLLM worker: whole-GPU claim, synchronous LLM.generate."""

    def __init__(
        self,
        worker_id: int,
        model_name: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.worker_id = worker_id
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._scorer = RubricScorer()

        logger.info(
            f"Worker {worker_id}: loading '{model_name}' "
            f"(vLLM, tp={tensor_parallel_size})..."
        )

        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=model_name,
            dtype="auto",
            tensor_parallel_size=tensor_parallel_size,
        )
        # Greedy decoding: outputs stable run-to-run. 
        # Continuous batching doesn't guarantee bitwise determinism across batch.
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        logger.info(f"Worker {worker_id}: ready")

    def evaluate_batch(self, tasks: list[EvalTask]) -> list[EvalResult]:
        prompts = [task.prompt for task in tasks]

        start = time.perf_counter()
        try:
            outputs = self._llm.generate(prompts, self._sampling_params)
        except Exception:
            self.tasks_failed += len(tasks)
            raise

        batch_latency = time.perf_counter() - start
        estimated_per_task = batch_latency / len(tasks)

        results = []
        for task, output in zip(tasks, outputs):
            completion = output.outputs[0]
            response = completion.text
            score, condition_scores = self._scorer.score(response, task)
            self.tasks_completed += 1
            results.append(EvalResult(
                task_id=task.task_id,
                score=score,
                response=response,
                latency_seconds=estimated_per_task,
                worker_id=self.worker_id,
                condition_scores=condition_scores,
                tokens_generated=len(completion.token_ids),
            ))
        return results
    
    def evaluate_with_hooks(
        self,
        task: EvalTask,
        hooks: list,  # list[InterventionHook]
    ) -> EvalResult:
        """not supported: synchronous LLM.generate returns only terminal outputs."""
        raise NotImplementedError(
            "vLLM backend does not support hooks: synchronous generate has "
            "no per-token hook point"
        )
        
    def health_check(self) -> bool:
        return True

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "model": self.model_name,
            "backend": "vllm",
            "completed": self.tasks_completed,
            "failed": self.tasks_failed,
            "poisoned": False,
        }
