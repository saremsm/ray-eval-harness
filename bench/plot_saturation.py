"""Plot achieved-vs-offered saturation curves from bench/saturation.py JSON reports."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

KNEE_RATIO = 0.8


def load_reports(paths: list[str]) -> list[dict]:
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith(".json")
            )
        else:
            files.append(path)
    reports = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        if "offered_load_tasks_per_s" not in r:
            continue  # not a saturation report
        r["_source"] = f
        reports.append(r)
    return reports


def curves_by_batch(reports: list[dict]) -> dict[int, list[tuple[float, float]]]:
    """batch_size -> [(offered, achieved)] sorted by offered load."""
    curves: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in reports:
        batch = int(r["args"]["batch_size"])
        offered = float(r["offered_load_tasks_per_s"])
        achieved = float(r["achieved"]["throughput_tasks_per_s_wall"])
        curves[batch].append((offered, achieved))
    for batch in curves:
        curves[batch].sort()
    return dict(sorted(curves.items()))


def analyze(label: str, curves: dict[int, list[tuple[float, float]]]) -> None:
    for batch, points in curves.items():
        knee = next(
            (
                (off, ach)
                for off, ach in points
                if ach < KNEE_RATIO * off
            ),
            None,
        )
        max_achieved = max(ach for _, ach in points)
        knee_txt = (
            f"knee at offered {knee[0]:.1f} tasks/s "
            f"(achieved {knee[1]:.1f})"
            if knee
            else f"no knee (all points >= {KNEE_RATIO:.0%} of offered)"
        )
        print(
            f"[{label}] batch={batch}: {knee_txt}; "
            f"max achieved {max_achieved:.1f} tasks/s "
            f"over {len(points)} points"
        )


def plot(
    datasets: list[tuple[str, dict[int, list[tuple[float, float]]]]],
    out: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    linestyles = ["-", "--", ":", "-."]
    all_offered: list[float] = []
    for i, (label, curves) in enumerate(datasets):
        style = linestyles[i % len(linestyles)]
        for batch, points in curves.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            all_offered.extend(xs)
            name = f"batch={batch}" if len(datasets) == 1 else (
                f"{label} batch={batch}"
            )
            ax.plot(xs, ys, style, marker="o", label=name)
    if all_offered:
        lim = sorted(all_offered)
        ax.plot(lim, lim, color="gray", linewidth=0.8,
                label="achieved = offered")
        ax.plot(lim, [KNEE_RATIO * x for x in lim], color="gray",
                linewidth=0.8, linestyle="--",
                label=f"{KNEE_RATIO:.0%} of offered (knee line)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offered load (tasks/s)")
    ax.set_ylabel("achieved throughput (tasks/s)")
    ax.set_title("Coordinator saturation: achieved vs offered")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Plot saturation curves and report knees."
    )
    p.add_argument("paths", nargs="*",
                   default=[os.path.join("bench", "results")],
                   help="Report files or directories "
                        "(default: bench/results/)")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="Overlay two result sets (files or directories); "
                        "for before/after comparisons (shard sweep)")
    p.add_argument("--out", type=str,
                   default=os.path.join("bench", "results", "saturation.png"))
    p.add_argument("--no-plot", action="store_true", dest="no_plot",
                   help="Analysis only; skip matplotlib")
    ns = p.parse_args(argv)

    if ns.compare:
        a, b = ns.compare
        datasets = [
            (os.path.basename(a.rstrip("/")) or a,
             curves_by_batch(load_reports([a]))),
            (os.path.basename(b.rstrip("/")) or b,
             curves_by_batch(load_reports([b]))),
        ]
    else:
        datasets = [("results", curves_by_batch(load_reports(ns.paths)))]

    any_points = False
    for label, curves in datasets:
        if curves:
            any_points = True
            analyze(label, curves)
        else:
            print(f"[{label}] no saturation reports found")
    if not any_points:
        return
    if not ns.no_plot:
        plot(datasets, ns.out)


if __name__ == "__main__":
    main()
