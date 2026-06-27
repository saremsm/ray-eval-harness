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


# ShardedFailureDecider

from fault_injection import (  # noqa: E402  (grouped with the sharding tests)
    ShardedFailureDecider,
    decider_shard_index,
)


class _LocalDeciderHandle:
    """Wraps a FailureDeciderImpl so `.should_fail.remote(...)` executes."""

    def __init__(self, impl: FailureDeciderImpl) -> None:
        self.impl = impl

    @property
    def should_fail(self):
        impl = self.impl

        class _Method:
            @staticmethod
            def remote(batch_key, failure_rate):
                return impl.should_fail(batch_key, failure_rate)

        return _Method


def _local_decider_cls(created: list):
    """decider_cls stand-in constructing REAL FailureDeciderImpls."""

    class _Cls:
        @staticmethod
        def remote(seed: int):
            impl = FailureDeciderImpl(seed=seed)
            created.append(impl)
            return _LocalDeciderHandle(impl)

    return _Cls


def _make_batch_keys(n_batches: int, batch_size: int = 4) -> list[tuple]:
    return [
        tuple(
            f"task_{i * batch_size + j:05d}" for j in range(batch_size)
        )
        for i in range(n_batches)
    ]


class TestDeciderShardIndex:
    def test_matches_stable_seed_hash(self):
        for key in _make_batch_keys(20):
            for n in (1, 2, 3, 7, 16):
                assert decider_shard_index(key, n) == (
                    _stable_seed(key) % n
                )

    def test_deterministic_and_key_only(self):
        """Routing must depend on the batch_key alone."""
        f_a = ShardedFailureDecider(
            seed=1, n_shards=5, decider_cls=_local_decider_cls([])
        )
        f_b = ShardedFailureDecider(
            seed=999, n_shards=5, decider_cls=_local_decider_cls([])
        )
        for key in _make_batch_keys(100):
            assert f_a.shard_for(key) == f_b.shard_for(key)
            assert f_a.shard_for(key) == f_a.shard_for(key)

    def test_spreads_across_shards(self):
        """Sanity: 500 keys over 4 shards must not collapse onto one."""
        placements = {
            decider_shard_index(key, 4) for key in _make_batch_keys(500)
        }
        assert placements == {0, 1, 2, 3}


class TestShardedFailureDecider:
    def test_rejects_zero_shards(self):
        with pytest.raises(ValueError, match="n_shards"):
            ShardedFailureDecider(
                seed=0, n_shards=0, decider_cls=_local_decider_cls([])
            )

    def test_duck_types_the_actor_handle_surface(self):
        """Workers call `decider.should_fail.remote(batch_key, rate)`."""
        facade = ShardedFailureDecider(
            seed=42, n_shards=3, decider_cls=_local_decider_cls([])
        )
        result = facade.should_fail.remote(("a", "b"), 0.5)
        assert isinstance(result, bool)

    def test_500_batches_identical_sequence_n1_vs_n4(self):
        """Sharding must not change which batches fail: 500 batches, same seed, N=1
        vs N=4 -> identical failure sequences."""
        keys = _make_batch_keys(500)
        rate = 0.3

        bare = FailureDeciderImpl(seed=42)
        n1 = ShardedFailureDecider(
            seed=42, n_shards=1, decider_cls=_local_decider_cls([])
        )
        n4 = ShardedFailureDecider(
            seed=42, n_shards=4, decider_cls=_local_decider_cls([])
        )

        seq_bare = [bare.should_fail(k, rate) for k in keys]
        seq_n1 = [n1.should_fail.remote(k, rate) for k in keys]
        seq_n4 = [n4.should_fail.remote(k, rate) for k in keys]

        assert seq_n1 == seq_bare, "N=1 facade must match the bare Impl"
        assert seq_n4 == seq_bare, (
            "N=4 must produce the identical failure sequence - sharding "
            "changed an outcome"
        )
        # The sequence is non-trivial: rate 0.3 over 500 draws yields a mix.
        assert any(seq_bare) and not all(seq_bare)

    def test_retry_attempts_identical_across_shard_counts(self):
        """The one thing sharding COULD have broken: attempt counters."""
        keys = _make_batch_keys(60)
        rate = 0.5

        bare = FailureDeciderImpl(seed=7)
        n4 = ShardedFailureDecider(
            seed=7, n_shards=4, decider_cls=_local_decider_cls([])
        )

        seq_bare, seq_n4 = [], []
        for attempt_round in range(3):
            for k in keys:
                seq_bare.append(bare.should_fail(k, rate))
                seq_n4.append(n4.should_fail.remote(k, rate))
        assert seq_n4 == seq_bare
        assert any(seq_bare) and not all(seq_bare)

    def test_attempt_counters_have_shard_affinity(self):
        """Every attempt for a key lands on the same shard: exactly one shard's
        attempt map holds the key, with the full count."""
        created: list[FailureDeciderImpl] = []
        facade = ShardedFailureDecider(
            seed=0, n_shards=4, decider_cls=_local_decider_cls(created)
        )
        key = ("task_00000", "task_00001")
        for _ in range(3):
            facade.should_fail.remote(key, 0.5)

        holders = [
            impl for impl in created if key in impl._attempts
        ]
        assert len(holders) == 1, (
            f"key present on {len(holders)} shards; routing must pin "
            "each key to exactly one"
        )
        assert holders[0]._attempts[key] == 3
        assert created[facade.shard_for(key)] is holders[0]

    def test_all_shards_share_the_seed(self):
        created: list[FailureDeciderImpl] = []
        ShardedFailureDecider(
            seed=13, n_shards=4, decider_cls=_local_decider_cls(created)
        )
        assert len(created) == 4
        assert all(impl._seed == 13 for impl in created)
