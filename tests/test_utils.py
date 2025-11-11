import pytest
from types_ import EvalTask
from utils import make_batches

def t(tid: str) -> EvalTask:
    return EvalTask(task_id=tid, prompt="p")

class TestMakeBatches:
    def test_even_split(self):
        tasks = [t(f"t{i}") for i in range(8)]
        batches = make_batches(tasks, 4)
        assert len(batches) == 2
        assert all(len(b) == 4 for b in batches)

    def test_uneven_split_last_batch_smaller(self):
        tasks = [t(f"t{i}") for i in range(10)]
        batches = make_batches(tasks, 4)
        assert len(batches) == 3
        assert len(batches[-1]) == 2

    def test_empty_input(self):
        assert make_batches([], 4) == []

    def test_batch_larger_than_tasks(self):
        tasks = [t("t0"), t("t1")]
        assert make_batches(tasks, 10) == [tasks]
    
    def test_order_preserved(self):
        tasks = [t(f"t{i}") for i in range(6)]
        flat = [task for batch in make_batches(tasks, 2) for task in batch]
        assert [x.task_id for x in flat] == [x.task_id for x in tasks]
    
    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            make_batches([t("t0")], 0)
