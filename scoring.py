from __future__ import annotations

import re
import string
from typing import Callable

from types_ import EvalTask, ScoringCondition

# Negation Detection
_NEGATION_WORDS = {
    # Negators
    "not", "never", "no",
    # Contractions
    "isn't", "aren't", "wasn't", "weren't",
    "cannot", "can't", "couldn't", "won't",
    "wouldn't", "shouldn't",
    # The Do's
    "don't", "doesn't", "didn't",
    # The Have's
    "hasn't", "haven't", "hadn't",
    # Correctness
    "wrong", "incorrect", "false",
}
_NEGATION_WINDOW  = 6 #can catch "doesn't think the answer is X", but not unrelated 'not'. 
_PUNCT_STRIP = string.punctuation

# stopwords excluded from topic matching. without.
_STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "was", "were", "in", "on", "at",
    "to", "and", "or", "it", "its", "as", "by", "for", "with", "has",
    "have", "had", "be", "been", "this", "that",
}

def _is_negated(response: str, match_start: int) -> bool:
    """negation within _NEGATION_WINDOW words before match_start. punctuation
    stripped per candidate word, so 'false;' matches 'false'."""
    prefix = response[:match_start]
    preceding_words = prefix.strip().split()[-_NEGATION_WINDOW:]
    return any(
        w.lower().strip(_PUNCT_STRIP) in _NEGATION_WORDS
        for w in preceding_words
    )

# Condition Checkers
def contains_answer(response: str, task: EvalTask) -> bool:
    """case insensitive, not negated"""
    if task.expected_answer is None:
        return False
    needle = task.expected_answer.strip().lower()
    haystack = response.lower()
    idx = haystack.find(needle)
    if idx == -1:
        return False    
    return not _is_negated(response, idx)

def answer_at_end(response: str, task: EvalTask) -> bool: 
    """answer in last 30%"""
    if task.expected_answer is None:
        return False
    cutoff = max(0, int(len(response) * 0.7))
    tail = response[cutoff:].lower()
    needle = task.expected_answer.strip().lower()
    idx = tail.find(needle)
    if idx == -1:
        return False    
    return not _is_negated(tail, idx)

def no_hallucinated_numbers(response: str, task: EvalTask) -> bool:
    """no extra numbers besides expected."""
    if task.expected_answer is None:
        return True
    expected = task.expected_answer.strip()
    if not expected.lstrip("-").isdigit():
        return True
    numbers_in_response = set(re.findall(r"-?\d+", response))
    return numbers_in_response <= {expected}

def is_concise(response: str, task: EvalTask) -> bool:
    """under 150 words"""
    return len(response.split()) < 150

def mentions_topic(response: str, task: EvalTask) -> bool:
    """any of the 5 longest content keywords present in the response (word-boundary
    match; stopwords and short words excluded)."""
    words = [w.strip(_PUNCT_STRIP).lower() for w in task.prompt.split()]
    content = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    if not content:
        # Degenerate all-stopword prompt: fall back to the raw words.
        content = [w for w in words if w]
    keywords = sorted(set(content), key=len, reverse=True)[:5]
    return any(
        re.search(rf"\b{re.escape(w)}\b", response, re.IGNORECASE)
        for w in keywords
    )

# Default Rubric
DEFAULT_CONDITIONS: list[ScoringCondition] = [
    ScoringCondition(
        name="contains_answer",
        weight=0.5,
        description="Response contains expected answer (not negated)",
    ),
    ScoringCondition(
        name="answer_at_end",
        weight=0.2,
        description="Answer in last 30% of response",
    ),
    ScoringCondition(
        name="is_concise",
        weight=0.2,
        description="Response is under 150 words",
    ),
    ScoringCondition(
        name="mentions_topic",
        weight=0.1,
        description="Response mentions keyword from prompt",
    ),
]

_CONDITION_CHECKERS: dict[str, Callable[[str, EvalTask], bool]] = {
    "contains_answer": contains_answer,
    "answer_at_end": answer_at_end,
    "no_hallucinated_numbers": no_hallucinated_numbers,
    "is_concise": is_concise,
    "mentions_topic": mentions_topic,
}

# Scorer
class RubricScorer:
    """Scores response against rubric."""
    def __init__(
        self, 
        default_conditions: list[ScoringCondition] | None = None,
    ) -> None:
        self._defaults = default_conditions or DEFAULT_CONDITIONS

    def score(
        self, response: str, task: EvalTask
    ) -> tuple[float, dict[str, float]]:
        """Scores response. Unknown condition names score 0."""
        conditions = task.conditions if task.conditions else self._defaults
        total = 0.0
        details: dict[str, float] = {}

        for condition in conditions:
            checker = _CONDITION_CHECKERS.get(condition.name)
            if checker is None:
                details[condition.name] = 0.0
                continue
            
            passed = checker(response, task)
            awarded = condition.weight if passed else 0.0
            details[condition.name] = awarded
            total += awarded
        
        return min(1.0, max(0.0, total)), details
    
    @staticmethod
    def aggregate_condition_scores(
        all_condition_scores: list[dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """per-condition pass_rate + mean_awarded (all)"""
        if not all_condition_scores:
            return {}
        
        weight_sums: dict[str, float] = {}
        pass_counts: dict[str, int] = {}
        totals: dict[str, int] = {}

        for breakdown in all_condition_scores:
            for name, awarded in breakdown.items():
                weight_sums[name] = weight_sums.get(name, 0.0) + awarded
                totals[name] = totals.get(name, 0) + 1
                if awarded > 0.0:
                    pass_counts[name] = pass_counts.get(name, 0) + 1
        
        return {
            name: {
                "mean_awarded": weight_sums[name] / totals[name],
                "pass_rate": pass_counts.get(name, 0) / totals[name],
            }
            for name in weight_sums
        }
