from __future__ import annotations

from types_ import EvalTask

def make_batches(
    tasks: list[EvalTask], 
    batch_size: int
) -> list[list[EvalTask]]:
    """chunk task list for distribution across workers"""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    return [
        tasks[i : i + batch_size]
        for i in range(0, len(tasks), batch_size)
    ]
