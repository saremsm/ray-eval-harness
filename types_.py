from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable

# Enums
class FailureKind(Enum):
    """Failure classification for retry logic.""" 
    TRANSIENT = auto()
    DETERMINISTIC = auto()

# Scoring Types
@dataclass
class ScoringCondition:
    """One weighted condition in a scoring rubric."""
    name: str
    weight: float
    description: str

# Task + Result
@dataclass
class EvalTask:
    """One eval task. task_id and prompt required."""
    task_id: str
    prompt: str
    expected_answer: Optional[str] = None
    conditions: list[ScoringCondition] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    max_retries: Optional[int] = None

@dataclass
class EvalResult:
    """Result for one completed task."""
    task_id: str
    score: float
    response: str
    latency_seconds: float
    worker_id: int
    batch_latency_seconds: Optional[float] = None
    failed: bool = False
    hooked: bool = False
    error: Optional[str] = None
    failure_kind: Optional[FailureKind] = None
    condition_scores: dict[str, float] = field(default_factory=dict)
    tokens_generated: int = 0
    stopped_early: bool = False
    hook_state: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict:
        """failure_kind -> str so json.dumps just works."""
        return {
            "task_id": self.task_id,
            "score": self.score,
            "response": self.response,
            "latency_seconds": self.latency_seconds,
            "batch_latency_seconds": self.batch_latency_seconds,
            "failed": self.failed,
            "hooked": self.hooked,
            "worker_id": self.worker_id,
            "error": self.error,
            "failure_kind": (
                self.failure_kind.name if self.failure_kind else None
            ),
            "condition_scores": self.condition_scores,
            "tokens_generated": self.tokens_generated,
            "stopped_early": self.stopped_early,
            "hook_state": self.hook_state,
        }

# EvalBackend Protocol
@runtime_checkable    
class InterventionHook(Protocol):
    """Per-step callback for generation."""

    def on_delta(self, delta: str, accumulated: str) -> None:
        """Called after each generation step. delta = new text since prev call."""
        ...

    def should_stop(self, accumulated: str) -> bool:
        """Return True to halt. Sets stopped_early=True on result."""
        ...

@runtime_checkable
class EvalBackend(Protocol):
    """Structural interface for eval backends. @runtime_checkable doesn't help."""
    
    def evaluate_batch(self, tasks: list[EvalTask]) -> list[EvalResult]:
        """Evaluate batch. Returns results in input order."""
        ...
    
    def health_check(self) -> bool:
        """Doubles as init barrier (blocks on __init__)."""
        ...
    
    def get_stats(self) -> dict:
        ...
