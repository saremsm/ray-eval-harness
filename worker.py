from __future__ import annotations

import asyncio
import logging
import random
import time

import ray

from types_ import EvalResult, EvalTask
from scoring import RubricScorer

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 80

HF_MICRO_BATCH_SIZE = 8
# Per-batch wall-clock budget for one evaluate call (seconds). 
DEFAULT_TASK_TIMEOUT = 60.0

# Stopping Criteria Helpers (HF Backend)
def _make_timeout_criterion(deadline: float):
    """StoppingCriteria that fires once monotonic time passes deadline."""
    from transformers import StoppingCriteria

    class _TimeoutCriterion(StoppingCriteria):
        def __init__(self):
            """device: -1 CPU, 0+ CUDA."""
            super().__init__()
            self.deadline = deadline
            self.timed_out = False

        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() >= self.deadline:
                self.timed_out = True
                return True
            return False

    return _TimeoutCriterion()


def _make_hook_criterion(hooks, tokenizer):
    """StoppingCriteria that runs hook callbacks after each step and halts when any
    hook signals stop."""
    from transformers import StoppingCriteria

    class _HookCriterion(StoppingCriteria):
        def __init__(self):
            super().__init__()
            self.hooks = hooks
            self.tokenizer = tokenizer
            # Set on first call: input_ids is prompt + 1 generated token.
            self.prompt_length = None
            self.accumulated = ""
            self.stopped_early = False
            self.token_count = 0

        def __call__(self, input_ids, scores, **kwargs):
            if self.prompt_length is None:
                self.prompt_length = input_ids.shape[1] - 1

            generated_ids = input_ids[0, self.prompt_length:]
            if generated_ids.numel() == 0:
                return False

            full_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            delta = full_text[len(self.accumulated):]
            if not delta:
                return False

            self.accumulated = full_text
            self.token_count = int(generated_ids.numel())

            for hook in self.hooks:
                hook.on_delta(delta, self.accumulated)

            if any(hook.should_stop(self.accumulated) for hook in self.hooks):
                self.stopped_early = True
                return True
            return False

    return _HookCriterion()

# HuggingFace pipeline backend
@ray.remote
class HFWorker:
    """persistent worker: model loads once in __init__, reused across batches."""
    def __init__(
        self, 
        worker_id: int, 
        model_name: str = "distilgpt2",
        task_timeout: float = DEFAULT_TASK_TIMEOUT,
        device: int = -1,
        failure_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        """device: -1 for CPU, 0+ for CUDA devices."""
        self.worker_id = worker_id
        self.model_name = model_name
        self.task_timeout = task_timeout
        self.device = device
        self._failure_rate = failure_rate
        self._rng = random.Random(seed + worker_id)
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
            "text-generation", 
            model=model_name, 
            device=device,
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
	
        if self._failure_rate > 0 and self._rng.random() < self._failure_rate:
            self.tasks_failed += len(tasks)
            raise RuntimeError(f"injected failure (worker {self.worker_id})")
        
        from transformers import StoppingCriteriaList

        prompts = [task.prompt for task in tasks]
        deadline = time.monotonic() + self.task_timeout
        timeout_criterion = _make_timeout_criterion(deadline)

        start = time.perf_counter()
        try:
            raw_outputs = self._pipeline(
                prompts, 
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                batch_size=min(len(prompts), HF_MICRO_BATCH_SIZE),
                return_full_text=False,
                stopping_criteria=StoppingCriteriaList([timeout_criterion]),
            )
        except Exception:
            self.tasks_failed += len(tasks)
            raise

        batch_latency = time.perf_counter() - start
        
        if timeout_criterion.timed_out:
            self.tasks_failed += len(tasks)
            raise TimeoutError(
                f"Worker {self.worker_id}: batch of {len(tasks)} exceeded "
                f"{self.task_timeout}s; partial outputs discarded"
            )
        
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
        """per-token hooks via TextIteratorStreamer; generation runs in a background
        thread."""
        if self._poisoned:
            raise RuntimeError(f"Worker {self.worker_id} poisoned, replace me")

        from transformers import StoppingCriteriaList

        criterion = _make_hook_criterion(
            hooks=hooks,
            tokenizer=self._pipeline.tokenizer,
        )

        start = time.perf_counter()
        try:
            outputs = self._pipeline(
                task.prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([criterion]),
                return_full_text=False,
            )
        except Exception:
            self.tasks_failed += 1
            raise

        elapsed = time.perf_counter() - start
        response = outputs[0]["generated_text"]
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
            tokens_generated=criterion.token_count,
            stopped_early=criterion.stopped_early,
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
    def __init__(
        self,
        worker_id: int,
        model_name: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        task_timeout: float = DEFAULT_TASK_TIMEOUT,
        failure_rate: float = 0.0,
        seed: int = 0,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.worker_id = worker_id
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.task_timeout = task_timeout
        self._failure_rate = failure_rate
        self._rng = random.Random(seed + worker_id)
        self.tensor_parallel_size = tensor_parallel_size
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._scorer = RubricScorer()
        self._request_counter = 0

        logger.info(
            f"Worker {worker_id}: loading '{model_name}' "
            f"(vLLM async, tp={tensor_parallel_size})..."
        )

        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype="auto",
            tensor_parallel_size=tensor_parallel_size,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        logger.info(f"Worker {worker_id}: ready")

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"w{self.worker_id}-r{self._request_counter}"

    async def evaluate_batch(self, tasks: list[EvalTask]) -> list[EvalResult]:
        """submit all prompts concurrently, gather terminal outputs."""
        if self._failure_rate > 0 and self._rng.random() < self._failure_rate:
            self.tasks_failed += len(tasks)
            raise RuntimeError(f"injected failure (worker {self.worker_id})")
        
        start = time.perf_counter()
        request_ids: list[str] = []

        async def _run_one(task: EvalTask) -> tuple[str, list[int]]:
            request_id = self._next_request_id()
            request_ids.append(request_id)
            final_output = None
            stream = self._engine.generate(
                task.prompt, self._sampling_params, request_id
            )
            async for output in stream:
                final_output = output
            assert final_output is not None, "engine produced no output"
            completion = final_output.outputs[0]
            return completion.text, list(completion.token_ids)

        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*[_run_one(t) for t in tasks]),
                timeout=self.task_timeout,
            )
        except Exception:
            for rid in request_ids:
                try:
                    await self._engine.abort(rid)
                except Exception:
                    pass
            self.tasks_failed += len(tasks)
            raise

        batch_latency = time.perf_counter() - start
        estimated_per_task = batch_latency / len(tasks)

        results = []
        failure_rate: float = 0.0,
        seed: int = 0,
        for task, (response, token_ids) in zip(tasks, outputs):
            score, condition_scores = self._scorer.score(response, task)
            self.tasks_completed += 1
            results.append(EvalResult(
                task_id=task.task_id,
                score=score,
                response=response,
                latency_seconds=estimated_per_task,
                worker_id=self.worker_id,
                condition_scores=condition_scores,
                tokens_generated=len(token_ids),
            ))
        return results

    async def evaluate_with_hooks(
        self,
        task: EvalTask,
        hooks: list,  # list[InterventionHook]
    ) -> EvalResult:
        request_id = self._next_request_id()
        start = time.perf_counter()
        stopped_early = False
        accumulated = ""
        token_count = 0

        try:
            stream = self._engine.generate(
                task.prompt, self._sampling_params, request_id
            )
            async for output in stream:
                completion = output.outputs[0]
                new_text = completion.text[len(accumulated):]
                token_count = len(completion.token_ids)

                if new_text:
                    for hook in hooks:
                        hook.on_delta(new_text, completion.text)

                accumulated = completion.text

                if any(hook.should_stop(accumulated) for hook in hooks):
                    stopped_early = True
                    await self._engine.abort(request_id)
                    break

        except Exception:
            self.tasks_failed += 1
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass
            raise

        elapsed = time.perf_counter() - start
        score, condition_scores = self._scorer.score(accumulated, task)
        self.tasks_completed += 1

        return EvalResult(
            task_id=task.task_id,
            score=score,
            response=accumulated,
            latency_seconds=elapsed,
            hooked=True,
            worker_id=self.worker_id,
            condition_scores=condition_scores,
            tokens_generated=token_count,
            stopped_early=stopped_early,
        )

    async def health_check(self) -> bool:
        return True

    async def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "model": self.model_name,
            "backend": "vllm",
            "completed": self.tasks_completed,
            "failed": self.tasks_failed,
            "poisoned": False,
        }
