"""Unit tests for bench/plot_saturation.py. 2. Curves are keyed by (batch_size,
fail_rate) so fault-injected runs never merge into clean curves - merged, they
produce a knee that belongs to neither experiment. 4."""

import json
import os

import pytest

from bench.plot_saturation import (
    KNEE_RATIO,
    analyze,
    curve_label,
    curves_by_batch,
    load_reports,
)
from bench.plot_saturation import main as plot_main


def write_report(
    path,
    *,
    batch: int,
    offered: float,
    achieved: float,
    fail_rate: float = 0.0,
    workers: int = 8,
    agg_shards: int = 1,
    dec_shards: int = 1,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "args": {
            "workers": workers,
            "batch_size": batch,
            "fail_rate": fail_rate,
            "aggregator_shards": agg_shards,
            "decider_shards": dec_shards,
        },
        "offered_load_tasks_per_s": offered,
        "achieved": {"throughput_tasks_per_s_wall": achieved},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)


class TestLoadReportsRecursion:
    def test_finds_reports_in_nested_subdirectories(self, tmp_path):
        write_report(
            str(tmp_path / "rep1" / "a.json"),
            batch=8, offered=100.0, achieved=90.0,
        )
        write_report(
            str(tmp_path / "rep2" / "a.json"),
            batch=8, offered=100.0, achieved=91.0,
        )
        write_report(
            str(tmp_path / "top.json"),
            batch=8, offered=50.0, achieved=49.0,
        )
        reports = load_reports([str(tmp_path)])
        assert len(reports) == 3
        sources = {r["_source"] for r in reports}
        assert any(os.path.join("rep1", "a.json") in s for s in sources)
        assert any(os.path.join("rep2", "a.json") in s for s in sources)

    def test_skips_non_saturation_json(self, tmp_path):
        (tmp_path / "other.json").write_text('{"not": "a report"}')
        write_report(
            str(tmp_path / "sat.json"),
            batch=1, offered=10.0, achieved=9.0,
        )
        reports = load_reports([str(tmp_path)])
        assert len(reports) == 1

    def test_explicit_file_paths_still_work(self, tmp_path):
        p = str(tmp_path / "one.json")
        write_report(p, batch=1, offered=10.0, achieved=9.0)
        reports = load_reports([p])
        assert len(reports) == 1
        assert reports[0]["_source"] == p


class TestFailRateSeparation:
    def test_clean_and_faulted_runs_form_separate_curves(self, tmp_path):
        write_report(
            str(tmp_path / "clean" / "a.json"),
            batch=8, offered=6400.0, achieved=4911.0, fail_rate=0.0,
        )
        write_report(
            str(tmp_path / "faulted" / "a.json"),
            batch=8, offered=6400.0, achieved=3658.0, fail_rate=0.1,
        )
        curves = curves_by_batch(load_reports([str(tmp_path)]))
        assert set(curves) == {(8, 0.0, 1, 1), (8, 0.1, 1, 1)}
        assert curves[(8, 0.0, 1, 1)] == [(6400.0, 4911.0)]
        assert curves[(8, 0.1, 1, 1)] == [(6400.0, 3658.0)]

    def test_missing_fail_rate_defaults_to_clean(self, tmp_path):
        p = str(tmp_path / "old.json")
        write_report(p, batch=1, offered=10.0, achieved=9.0)
        with open(p, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        del r["args"]["fail_rate"]
        del r["args"]["aggregator_shards"]
        del r["args"]["decider_shards"]
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(r, fh)
        curves = curves_by_batch(load_reports([p]))
        # Older reports lack the shard args and key as (1, 1).
        assert set(curves) == {(1, 0.0, 1, 1)}

    def test_curve_label(self):
        assert curve_label(8, 0.0) == "batch=8"
        assert curve_label(8, 0.1) == "batch=8 fail=0.1"
        # Shard settings appear only when they differ from the default.
        assert curve_label(8, 0.0, 1, 1) == "batch=8"
        assert curve_label(8, 0.0, 4, 1) == "batch=8 agg=4"
        assert curve_label(8, 0.0, 1, 4) == "batch=8 dec=4"
        assert curve_label(8, 0.1, 8, 4) == "batch=8 fail=0.1 agg=8 dec=4"


class TestShardSettingSeparation:
    def test_shard_settings_form_separate_curves(self, tmp_path):
        # One tree holding an a1 and an a4 sweep (the sweep layout) must yield one
        write_report(
            str(tmp_path / "a1" / "rep1" / "x.json"),
            batch=8, offered=25600.0, achieved=17785.0, agg_shards=1,
        )
        write_report(
            str(tmp_path / "a4" / "rep1" / "x.json"),
            batch=8, offered=25600.0, achieved=9482.0, agg_shards=4,
        )
        curves = curves_by_batch(load_reports([str(tmp_path)]))
        assert set(curves) == {(8, 0.0, 1, 1), (8, 0.0, 4, 1)}
        assert curves[(8, 0.0, 1, 1)] == [(25600.0, 17785.0)]
        assert curves[(8, 0.0, 4, 1)] == [(25600.0, 9482.0)]

    def test_decider_shards_also_key(self, tmp_path):
        write_report(
            str(tmp_path / "d1.json"),
            batch=8, offered=100.0, achieved=90.0,
            fail_rate=0.1, dec_shards=1,
        )
        write_report(
            str(tmp_path / "d4.json"),
            batch=8, offered=100.0, achieved=90.0,
            fail_rate=0.1, dec_shards=4,
        )
        curves = curves_by_batch(load_reports([str(tmp_path)]))
        assert set(curves) == {(8, 0.1, 1, 1), (8, 0.1, 1, 4)}

    def test_sharded_curve_labeled_in_analyze_output(self, capsys):
        curves = {(8, 0.0, 4, 1): [(25600.0, 9482.0)]}
        analyze("t", curves)
        out = capsys.readouterr().out
        assert "batch=8 agg=4" in out


class TestCompareMultiway:
    def _tree(self, tmp_path, name, achieved, agg_shards):
        write_report(
            str(tmp_path / name / "x.json"),
            batch=8, offered=25600.0, achieved=achieved,
            agg_shards=agg_shards,
        )
        return str(tmp_path / name)

    def test_compare_accepts_three_paths(self, tmp_path, capsys):
        a1 = self._tree(tmp_path, "a1", 17785.0, 1)
        a4 = self._tree(tmp_path, "a4", 9482.0, 4)
        a8 = self._tree(tmp_path, "a8", 6407.0, 8)
        plot_main(["--compare", a1, a4, a8, "--no-plot"])
        out = capsys.readouterr().out
        assert "[a1] batch=8:" in out
        assert "[a4] batch=8 agg=4:" in out
        assert "[a8] batch=8 agg=8:" in out

    def test_compare_still_works_with_two_paths(self, tmp_path, capsys):
        a = self._tree(tmp_path, "before", 17785.0, 1)
        b = self._tree(tmp_path, "after", 9482.0, 4)
        plot_main(["--compare", a, b, "--no-plot"])
        out = capsys.readouterr().out
        assert "[before]" in out
        assert "[after]" in out

    def test_compare_rejects_single_path(self, tmp_path, capsys):
        a = self._tree(tmp_path, "only", 17785.0, 1)
        with pytest.raises(SystemExit):
            plot_main(["--compare", a, "--no-plot"])
        assert "at least two paths" in capsys.readouterr().err


class TestKneeAnalysis:
    def test_knee_is_first_point_below_ratio(self, capsys):
        curves = {
            (8, 0.0): [
                (100.0, 95.0),           # 0.95
                (200.0, 150.0),          # 0.75 <- knee
                (400.0, 150.0),          # 0.375
            ]
        }
        analyze("t", curves)
        out = capsys.readouterr().out
        assert "knee at offered 200.0" in out
        assert "achieved 150.0" in out
        assert "re-crosses" not in out
        assert "max achieved 150.0" in out

    def test_no_knee_reported_when_all_points_track_offered(self, capsys):
        curves = {(1, 0.0): [(100.0, 95.0), (200.0, 190.0)]}
        analyze("t", curves)
        out = capsys.readouterr().out
        assert "no knee" in out
        assert f"{KNEE_RATIO:.0%}" in out

    def test_marginal_knee_recross_is_flagged(self, capsys):
        # Real shape from the published batch=8 grid: 256 slow workers dip below 80%.
        curves = {
            (8, 0.0): [
                (2048.0, 1816.0),   # 0.887
                (4096.0, 3273.0),   # 0.799 <- printed knee
                (5120.0, 4402.0),   # 0.860 <- re-cross
                (6400.0, 4886.0),   # 0.763
                (10240.0, 4653.0),  # 0.454
            ]
        }
        analyze("t", curves)
        out = capsys.readouterr().out
        assert "knee at offered 4096.0" in out
        assert "re-crosses 80% at offered 5120.0" in out
        assert "marginal" in out

    def test_faulted_curve_labeled_in_output(self, capsys):
        curves = {(8, 0.1): [(6400.0, 3658.0)]}
        analyze("t", curves)
        out = capsys.readouterr().out
        assert "batch=8 fail=0.1" in out
