import pytest

from fault_injection import FailureDeciderImpl, _stable_seed


class TestStableSeed:
    def test_returns_int(self):
        assert isinstance(_stable_seed(1, "x", (1, 2)), int)

    def test_deterministic(self):
        assert _stable_seed(1, "x", (1, 2)) == _stable_seed(1, "x", (1, 2))

    def test_different_inputs_differ(self):
        assert _stable_seed(1, "x") != _stable_seed(2, "x")
        assert _stable_seed(1, "x") != _stable_seed(1, "y")

    def test_handles_tuple_input(self):
        """Regression: random.Random rejects tuples directly. _stable_seed."""
        seed = _stable_seed(42, ("task_00000", "task_00001"), 0, 0.3)
        assert isinstance(seed, int)


class TestFailureDeciderImpl:
    def test_smoke_does_not_raise(self):
        """Regression: prior version passed a tuple to random.Random."""
        decider = FailureDeciderImpl(seed=42)
        batch_key = ("task_00000", "task_00001", "task_00002", "task_00003")
        result = decider.should_fail(batch_key, 0.3)
        assert isinstance(result, bool)

    def test_deterministic_across_fresh_deciders(self):
        """Same seed + same key + same attempt -> same decision."""
        d1 = FailureDeciderImpl(seed=42)
        d2 = FailureDeciderImpl(seed=42)
        key = ("a", "b", "c")
        assert d1.should_fail(key, 0.5) == d2.should_fail(key, 0.5)

    def test_attempt_counter_advances(self):
        """Successive calls on the same key advance attempt count."""
        d = FailureDeciderImpl(seed=42)
        key = ("a",)
        # Run enough attempts at rate=0.5 that we expect at least one True and one.
        decisions = [d.should_fail(key, 0.5) for _ in range(20)]
        assert any(decisions) and not all(decisions), (
            "attempts at rate=0.5 should produce a mix; "
            "if all match, the attempt counter isn't advancing."
        )

    def test_failure_rate_zero_never_fails(self):
        d = FailureDeciderImpl(seed=42)
        for i in range(100):
            assert not d.should_fail((f"t{i}",), 0.0)

    def test_failure_rate_one_always_fails(self):
        d = FailureDeciderImpl(seed=42)
        for i in range(100):
            assert d.should_fail((f"t{i}",), 1.0)

    def test_different_seeds_diverge(self):
        """Different seeds must produce visibly different patterns."""
        d1 = FailureDeciderImpl(seed=1)
        d2 = FailureDeciderImpl(seed=999)
        diffs = sum(
            d1.should_fail((f"t{i}",), 0.5)
            != d2.should_fail((f"t{i}",), 0.5)
            for i in range(200)
        )
        # At rate 0.5 with independent seeds, expect ~50% disagreement.
        assert diffs > 50, (
            f"Different seeds should disagree on a meaningful fraction; "
            f"got {diffs}/200."
        )

    def test_failure_rate_changes_pattern(self):
        """Same seed, different rates should give different patterns."""
        d_low = FailureDeciderImpl(seed=42)
        d_high = FailureDeciderImpl(seed=42)
        low_failures = sum(
            d_low.should_fail((f"t{i}",), 0.1) for i in range(200)
        )
        high_failures = sum(
            d_high.should_fail((f"t{i}",), 0.9) for i in range(200)
        )
        assert high_failures > low_failures + 100, (
            f"rate=0.9 should fail far more often than rate=0.1; "
            f"got {high_failures} vs {low_failures}."
        )

    def test_distinct_keys_are_independent(self):
        """Different batch keys advance independent attempt counters."""
        d = FailureDeciderImpl(seed=42)
        d.should_fail(("a",), 0.5)
        d.should_fail(("a",), 0.5)
        # ("b",) starts fresh at attempt 0
        d_b_first = FailureDeciderImpl(seed=42).should_fail(("b",), 0.5)
        assert d.should_fail(("b",), 0.5) == d_b_first
