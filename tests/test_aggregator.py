"""ShardedAggregator + ResultsAggregatorImpl tests."""

from __future__ import annotations

import json
import random
import zlib
from unittest.mock import patch

import pytest

import aggregator as agg_mod
from aggregator import (
    RESERVOIR_CAP,
    ResultsAggregatorImpl,
    ShardedAggregator,
    shard_index,
)
from types_ import EvalResult, FailureKind


# Local-actor shims

class _LocalHandle:
    """Wraps an Impl so handle.method.remote(...) executes synchronously."""

    def __init__(self, impl) -> None:
        self._impl = impl

    def __getattr__(self, name):
        fn = getattr(self._impl, name)

        class _Method:
            @staticmethod
            def remote(*args, **kwargs):
                return fn(*args, **kwargs)

        return _Method


def _local_aggregator_cls(created: list):
    """aggregator_cls stand-in constructing REAL Impls locally."""

    class _Cls:
        @staticmethod
        def remote(**kwargs):
            impl = ResultsAggregatorImpl(**kwargs)
            created.append(impl)
            return _LocalHandle(impl)

    return _Cls


class _CountingShard:
    """Interface-only shard fake: counts record_batch calls and captures."""

    def __init__(self, total_tasks: int, output_path: str) -> None:
        self.output_path = output_path
        self.batches: list[list[EvalResult]] = []

    def __getattr__(self, name):
        shard = self

        class _Method:
            @staticmethod
            def remote(*args, **kwargs):
                if name == "record_batch":
                    shard.batches.append(list(args[0]))
                return None

        return _Method


def _counting_cls(created: list):
    class _Cls:
        @staticmethod
        def remote(**kwargs):
            shard = _CountingShard(**kwargs)
            created.append(shard)
            return shard

    return _Cls


@pytest.fixture
def identity_ray_get():
    """The local handles already return values, so ray.get is identity (works for
    scalars and for the facade's list fan-out alike)."""

    def _get(x, timeout=None):
        return x

    with patch.object(agg_mod.ray, "get", _get):
        yield


@pytest.fixture
def frozen_clock(monkeypatch):
    """Controllable perf_counter inside aggregator.py."""
    state = {"now": 1_000.0}
    monkeypatch.setattr(agg_mod.time, "perf_counter", lambda: state["now"])

    def advance(dt: float) -> None:
        state["now"] += dt

    return advance


# Result generation

def _random_results(rng: random.Random, n: int) -> list[EvalResult]:
    """Mixed successes/failures with every summary-relevant field exercised: scores,
    latencies, optional batch latencies, tokens, early stops, condition scores,
    several workers."""
    out = []
    for i in range(n):
        failed = rng.random() < 0.25
        if failed:
            out.append(
                EvalResult(
                    task_id=f"t{i:05d}x{rng.randrange(10 ** 6)}",
                    score=0.0,
                    response="",
                    latency_seconds=0.0,
                    batch_latency_seconds=None,
                    failed=True,
                    worker_id=rng.randrange(8),
                    error="injected",
                    failure_kind=FailureKind.TRANSIENT,
                )
            )
            continue
        conditions = (
            {
                "contains_answer": rng.choice([0.0, 0.5]),
                "answer_at_end": rng.choice([0.0, 0.25]),
            }
            if rng.random() < 0.8
            else {}
        )
        out.append(
            EvalResult(
                task_id=f"t{i:05d}x{rng.randrange(10 ** 6)}",
                score=round(rng.random(), 6),
                response="resp",
                latency_seconds=rng.uniform(0.001, 2.0),
                batch_latency_seconds=(
                    None if rng.random() < 0.3 else rng.uniform(0.01, 5.0)
                ),
                worker_id=rng.randrange(8),
                tokens_generated=rng.randrange(0, 100),
                stopped_early=rng.random() < 0.2,
                condition_scores=conditions,
            )
        )
    return out


def _batches(rng: random.Random, results: list[EvalResult]):
    i = 0
    while i < len(results):
        size = rng.randint(1, 16)
        yield results[i : i + size]
        i += size


def _assert_summaries_equal(a: dict, b: dict) -> None:
    """Exact for structure/ints/strings; approx for floats (per-shard partial sums
    are floating-point non-associative)."""
    assert set(a.keys()) == set(b.keys())
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, float):
            assert va == pytest.approx(vb, rel=1e-9, abs=1e-12), key
        elif isinstance(va, dict) and key == "condition_stats":
            assert set(va.keys()) == set(vb.keys()), key
            for cond in va:
                for stat in va[cond]:
                    assert va[cond][stat] == pytest.approx(
                        vb[cond][stat], rel=1e-9
                    ), (key, cond, stat)
        else:
            assert va == vb, key


# Routing

class TestRouting:
    def test_shard_index_is_stable_crc32(self):
        for tid in ("t000", "task_00042", "b0001234", "", "日本語"):
            for n in (1, 2, 3, 7, 16):
                assert shard_index(tid, n) == (
                    zlib.crc32(tid.encode("utf-8")) % n
                )

    def test_routing_deterministic_across_facades(self, identity_ray_get):
        f1 = ShardedAggregator(
            total_tasks=0, output_path="results/x.jsonl", n_shards=5,
            aggregator_cls=_counting_cls([]),
        )
        f2 = ShardedAggregator(
            total_tasks=0, output_path="results/y.jsonl", n_shards=5,
            aggregator_cls=_counting_cls([]),
        )
        for i in range(200):
            tid = f"task_{i:05d}"
            assert f1.shard_for(tid) == f2.shard_for(tid)
            assert f1.shard_for(tid) == f1.shard_for(tid)

    def test_one_call_per_shard_per_batch(self, identity_ray_get):
        created: list[_CountingShard] = []
        facade = ShardedAggregator(
            total_tasks=100, output_path="results/x.jsonl", n_shards=4,
            aggregator_cls=_counting_cls(created),
        )
        rng = random.Random(3)
        batch = _random_results(rng, 40)
        refs = facade.record_batch(batch)

        touched = {i for i, s in enumerate(created) if s.batches}
        expected = {facade.shard_for(r.task_id) for r in batch}
        assert touched == expected
        assert len(refs) == len(touched)
        for i, shard in enumerate(created):
            assert len(shard.batches) <= 1, (
                f"shard {i} got {len(shard.batches)} calls for ONE "
                "batch - must be exactly one grouped call"
            )
            for r in (shard.batches[0] if shard.batches else []):
                assert facade.shard_for(r.task_id) == i
        # Union of groups == the batch, exactly once each.
        regrouped = [r for s in created for b in s.batches for r in b]
        assert sorted(r.task_id for r in regrouped) == sorted(
            r.task_id for r in batch
        )

    def test_empty_batch_submits_nothing(self, identity_ray_get):
        created: list[_CountingShard] = []
        facade = ShardedAggregator(
            total_tasks=0, output_path="results/x.jsonl", n_shards=4,
            aggregator_cls=_counting_cls(created),
        )
        assert facade.record_batch([]) == []
        assert all(not s.batches for s in created)

    def test_rejects_zero_shards(self):
        with pytest.raises(ValueError, match="n_shards"):
            ShardedAggregator(
                total_tasks=0, output_path="results/x.jsonl", n_shards=0,
                aggregator_cls=_counting_cls([]),
            )


# N=1 identity

class TestSingleShardIdentity:
    """N=1 must be behaviorally identical to the single-actor aggregator."""

    def test_summary_and_jsonl_identical_to_bare_impl(
        self, tmp_path, identity_ray_get, frozen_clock
    ):
        rng = random.Random(11)
        results = _random_results(rng, 120)
        batches = list(_batches(random.Random(12), results))

        direct = ResultsAggregatorImpl(
            total_tasks=len(results),
            output_path=str(tmp_path / "direct.jsonl"),
        )
        created: list[ResultsAggregatorImpl] = []
        facade = ShardedAggregator(
            total_tasks=len(results),
            output_path=str(tmp_path / "facade.jsonl"),
            n_shards=1,
            aggregator_cls=_local_aggregator_cls(created),
        )
        for batch in batches:
            direct.record_batch(batch)
            facade.record_batch(batch)
        frozen_clock(7.5)

        s_direct = direct.get_summary()
        s_facade = facade.get_summary()

        # Only the file path may differ; everything else exactly equal.
        assert s_direct.pop("results_file").endswith("direct.jsonl")
        assert s_facade.pop("results_file").endswith("facade.jsonl")
        assert s_direct == s_facade  # exact, no approx: same input order

        direct.close()
        created[0].close()
        assert (tmp_path / "direct.jsonl").read_bytes() == (
            tmp_path / "facade.jsonl"
        ).read_bytes()

    def test_no_shard_suffix_and_no_extra_keys(
        self, tmp_path, identity_ray_get
    ):
        out = tmp_path / "results.jsonl"
        created: list[ResultsAggregatorImpl] = []
        facade = ShardedAggregator(
            total_tasks=4, output_path=str(out), n_shards=1,
            aggregator_cls=_local_aggregator_cls(created),
        )
        assert created[0].output_path == out, (
            "N=1 must write the exact requested path - no '.shard0'"
        )
        summary = facade.get_summary()
        assert "results_files" not in summary
        assert "aggregator_shards" not in summary
        assert summary["results_file"] == str(out)

    def test_n1_passes_output_path_string_verbatim(
        self, identity_ray_get
    ):
        """The N=1 shard constructor and finalize() must see the caller's exact
        string, not a Path() round-trip of it."""
        raw = "results/./test.jsonl"
        created: list[_CountingShard] = []
        facade = ShardedAggregator(
            total_tasks=1, output_path=raw, n_shards=1,
            aggregator_cls=_counting_cls(created),
        )
        assert created[0].output_path == raw
        assert facade.finalize() == raw

    def test_finalize_is_a_noop_at_n1(self, tmp_path, identity_ray_get):
        out = tmp_path / "results.jsonl"
        created: list[ResultsAggregatorImpl] = []
        facade = ShardedAggregator(
            total_tasks=2, output_path=str(out), n_shards=1,
            aggregator_cls=_local_aggregator_cls(created),
        )
        facade.record_batch(_random_results(random.Random(1), 2))
        before = out.read_bytes()
        assert facade.finalize() == str(out)
        assert out.read_bytes() == before, "finalize must not rewrite N=1"
        assert not created[0]._fh.closed, (
            "N=1 finalize must not touch the shard (the single actor never closed)"
        )


# Merged == single (property test)

class TestMergedEqualsSingleShard:
    @pytest.mark.parametrize("seed", range(6))
    def test_merged_summary_equals_single_shard_summary(
        self, seed, tmp_path, identity_ray_get, frozen_clock
    ):
        rng = random.Random(seed)
        n_shards = rng.choice([2, 3, 5, 8])
        results = _random_results(rng, rng.randint(50, 200))
        batches = list(_batches(rng, results))

        single = ShardedAggregator(
            total_tasks=len(results),
            output_path=str(tmp_path / "single.jsonl"),
            n_shards=1,
            aggregator_cls=_local_aggregator_cls([]),
        )
        sharded = ShardedAggregator(
            total_tasks=len(results),
            output_path=str(tmp_path / "sharded.jsonl"),
            n_shards=n_shards,
            aggregator_cls=_local_aggregator_cls([]),
        )
        for batch in batches:
            single.record_batch(batch)
            sharded.record_batch(batch)
        frozen_clock(13.0)

        s_single = single.get_summary()
        s_sharded = sharded.get_summary()

        assert s_sharded.pop("aggregator_shards") == n_shards
        shard_files = s_sharded.pop("results_files")
        assert len(shard_files) == n_shards
        s_single.pop("results_file")
        s_sharded.pop("results_file")
        _assert_summaries_equal(s_single, s_sharded)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_merged_jsonl_is_a_permutation_of_the_single_file(
        self, seed, tmp_path, identity_ray_get
    ):
        rng = random.Random(100 + seed)
        results = _random_results(rng, 80)
        batches = list(_batches(rng, results))

        single_created: list[ResultsAggregatorImpl] = []
        single = ShardedAggregator(
            total_tasks=len(results),
            output_path=str(tmp_path / "single.jsonl"),
            n_shards=1,
            aggregator_cls=_local_aggregator_cls(single_created),
        )
        sharded = ShardedAggregator(
            total_tasks=len(results),
            output_path=str(tmp_path / "merged.jsonl"),
            n_shards=3,
            aggregator_cls=_local_aggregator_cls([]),
        )
        for batch in batches:
            single.record_batch(batch)
            sharded.record_batch(batch)

        single_created[0].close()
        merged_path = sharded.finalize()
        single_lines = (tmp_path / "single.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        merged_lines = (tmp_path / "merged.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert merged_path == str(tmp_path / "merged.jsonl")
        assert sorted(merged_lines) == sorted(single_lines), (
            "Concatenated shard output must contain exactly the same "
            "lines - same multiplicity - as the single-shard file, in "
            "some order."
        )
        for line in merged_lines:
            json.loads(line)  # every line valid JSONL


# Reservoir cap

class TestReservoirCap:
    def test_reservoir_caps_samples_but_not_sums(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(agg_mod, "RESERVOIR_CAP", 8)
        impl = ResultsAggregatorImpl(
            total_tasks=50, output_path=str(tmp_path / "r.jsonl")
        )
        rng = random.Random(0)
        values = [rng.uniform(0.001, 2.0) for _ in range(50)]
        impl.record_batch(
            [
                EvalResult(
                    task_id=f"t{i}",
                    score=1.0,
                    response="ok",
                    latency_seconds=v,
                    batch_latency_seconds=v * 2,
                    worker_id=0,
                )
                for i, v in enumerate(values)
            ]
        )
        state = impl.get_shard_state()
        assert len(state["latency_samples"]) == 8
        assert state["latency_seen"] == 50
        assert set(state["latency_samples"]) <= set(values), (
            "Reservoir must hold observed values only"
        )
        summary = impl.get_summary()
        # Means come from running sums - exact regardless of the cap.
        assert summary["mean_latency_s"] == pytest.approx(
            sum(values) / len(values)
        )
        # Percentile is approximate above the cap but stays in range.
        assert min(values) <= summary["p99_latency_s"] <= max(values)

    def test_default_cap_is_documented_value(self):
        assert RESERVOIR_CAP == 100_000


# Finalize (concatenation)

class TestFinalize:
    def test_concatenates_all_shards_and_keeps_shard_files(
        self, tmp_path, identity_ray_get
    ):
        created: list[ResultsAggregatorImpl] = []
        facade = ShardedAggregator(
            total_tasks=60,
            output_path=str(tmp_path / "results.jsonl"),
            n_shards=3,
            aggregator_cls=_local_aggregator_cls(created),
        )
        results = _random_results(random.Random(7), 60)
        for batch in _batches(random.Random(8), results):
            facade.record_batch(batch)

        summary = facade.get_summary()
        merged = facade.finalize()

        assert merged == str(tmp_path / "results.jsonl")
        merged_ids = [
            json.loads(line)["task_id"]
            for line in (tmp_path / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert sorted(merged_ids) == sorted(r.task_id for r in results), (
            "Every result exactly once in the concatenated file"
        )
        assert all(impl._fh.closed for impl in created)
        for path_str in summary["results_files"]:
            assert agg_mod.Path(path_str).exists(), (
                "per-shard files listed in the summary must survive "
                "finalize"
            )
        assert summary["results_file"] == merged

    def test_finalize_without_shard_files_writes_nothing(
        self, tmp_path, identity_ray_get
    ):
        """Interface-only fakes (as in the coordinator suite) never create shard
        files; finalize must not conjure an empty merged file."""
        out = tmp_path / "sub" / "results.jsonl"
        facade = ShardedAggregator(
            total_tasks=0, output_path=str(out), n_shards=2,
            aggregator_cls=_counting_cls([]),
        )
        assert facade.finalize() == str(out)
        assert not out.exists()


# Back-compat

class TestAddResultBackCompat:
    def test_add_result_equals_record_batch_of_one(
        self, tmp_path, frozen_clock
    ):
        results = _random_results(random.Random(21), 30)
        a = ResultsAggregatorImpl(
            total_tasks=30, output_path=str(tmp_path / "a.jsonl")
        )
        b = ResultsAggregatorImpl(
            total_tasks=30, output_path=str(tmp_path / "b.jsonl")
        )
        for r in results:
            a.add_result(r)
            b.record_batch([r])
        frozen_clock(2.0)
        sa, sb = a.get_summary(), b.get_summary()
        sa.pop("results_file")
        sb.pop("results_file")
        assert sa == sb
        a.close()
        b.close()
        assert (tmp_path / "a.jsonl").read_bytes() == (
            tmp_path / "b.jsonl"
        ).read_bytes()
