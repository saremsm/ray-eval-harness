"""Sweep bench/saturation.py over a fixed grid. Grid: latency {0.5, 0.1, 0.02,
0.005}s x workers {16, 64, 128, 256} x batch {1, 8, 64}."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

LATENCIES_S = (0.5, 0.1, 0.02, 0.005)
WORKER_COUNTS = (16, 64, 128, 256)
BATCH_SIZES = (1, 8, 64)


def result_path(
    out_dir: str,
    workers: int,
    latency_s: float,
    batch: int,
    aggregator_shards: int = 1,
    decider_shards: int = 1,
) -> str:
    """Per-point report path."""
    ms = int(round(latency_s * 1000))
    shard_suffix = (
        ""
        if aggregator_shards == 1 and decider_shards == 1
        else f"_a{aggregator_shards}_d{decider_shards}"
    )
    return os.path.join(
        out_dir,
        f"sat_w{workers}_l{ms:03d}_b{batch}{shard_suffix}.json",
    )


def point_cmd(
    ns: argparse.Namespace,
    workers: int,
    latency_s: float,
    batch: int,
    n_tasks: int,
    out: str,
) -> list[str]:
    """argv for one grid point."""
    return [
        sys.executable, "-m", "bench.saturation",
        "--workers", str(workers),
        "--latency-s", str(latency_s),
        "--batch-size", str(batch),
        "--tasks", str(n_tasks),
        "--fail-rate", str(ns.fail_rate),
        "--aggregator-shards", str(ns.aggregator_shards),
        "--decider-shards", str(ns.decider_shards),
        "--out", out,
    ]


def tasks_for(
    ns: argparse.Namespace, workers: int, latency_s: float, batch: int
) -> int:
    if ns.tasks is not None:
        return ns.tasks
    offered = workers * batch / latency_s
    n = int(offered * ns.target_duration)
    return max(ns.min_tasks, min(ns.max_tasks, n))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Grid sweep for the coordinator saturation bench."
    )
    p.add_argument("--max-workers", type=int, default=256, dest="max_workers",
                   help="Skip grid points whose worker count exceeds what "
                        "this box can host (each actor is a process; a "
                        "30-vCPU box handles a few hundred at most). To "
                        "push offered load higher, lower latency instead.")
    p.add_argument("--out-dir", type=str, default=os.path.join("bench", "results"),
                   dest="out_dir")
    p.add_argument("--fail-rate", type=float, default=0.0, dest="fail_rate")
    p.add_argument("--aggregator-shards", type=int, default=1,
                   dest="aggregator_shards",
                   help="Forwarded to every point (shard sweeps). "
                        "Use one --out-dir per setting.")
    p.add_argument("--decider-shards", type=int, default=1,
                   dest="decider_shards",
                   help="Forwarded to every point; only matters when "
                        "--fail-rate > 0. Use one --out-dir per "
                        "setting.")
    p.add_argument("--tasks", type=int, default=None,
                   help="Fixed task count per point (default: sized from "
                        "offered load and --target-duration)")
    p.add_argument("--target-duration", type=float, default=20.0,
                   dest="target_duration",
                   help="Target seconds of offered load per point")
    p.add_argument("--min-tasks", type=int, default=2000, dest="min_tasks")
    p.add_argument("--max-tasks", type=int, default=200000, dest="max_tasks")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the plan without running anything")
    ns = p.parse_args(argv)

    os.makedirs(ns.out_dir, exist_ok=True)
    grid = list(itertools.product(LATENCIES_S, WORKER_COUNTS, BATCH_SIZES))
    ran = skipped_existing = skipped_capped = failed = 0

    for latency_s, workers, batch in grid:
        out = result_path(
            ns.out_dir, workers, latency_s, batch,
            ns.aggregator_shards, ns.decider_shards,
        )
        if workers > ns.max_workers:
            skipped_capped += 1
            continue
        if os.path.exists(out):
            print(f"skip (exists): {out}")
            skipped_existing += 1
            continue

        n_tasks = tasks_for(ns, workers, latency_s, batch)
        cmd = point_cmd(ns, workers, latency_s, batch, n_tasks, out)
        print(f"run: {' '.join(cmd)}")
        if ns.dry_run:
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"FAILED (rc={proc.returncode}): {out}", file=sys.stderr)
            failed += 1
        else:
            ran += 1

    print(
        f"sweep done: {ran} ran, {skipped_existing} existing, "
        f"{skipped_capped} over --max-workers cap, {failed} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
