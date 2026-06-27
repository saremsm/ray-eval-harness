"""Saturation bench: find where the coordinator saturates."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time

import ray

from aggregator import ResultsAggregator
from coordinator import DistributedEvalCoordinator
from fault_injection import ShardedFailureDecider
from types_ import EvalTask

from bench.fake_worker import FakeLatencyWorker
from bench.recording import RecordingMetrics, percentile

GAUGE_SAMPLE_INTERVAL_S = 0.5


# Aggregator call-latency probe
class AggregatorCallProbe:
    """Measures record_batch submit-to-ready latency from the driver side."""

    def __init__(self, every: int = 32) -> None:
        self.every = max(1, every)
        self.samples: list[float] = []
        self._n = 0
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._drain, name="agg-call-probe", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            ref, t_submit = item
            try:
                ray.get(ref)
            except Exception:
                pass  # a failed write still measures the round trip
            self.samples.append(time.monotonic() - t_submit)

    def observe(self, ref) -> None:
        self._n += 1
        if self._n % self.every == 0:
            self._q.put((ref, time.monotonic()))

    def close(self, timeout: float = 30.0) -> None:
        self._q.put(None)
        self._thread.join(timeout=timeout)

    def wrap(self, aggregator_cls):
        """Return an aggregator_cls stand-in producing probed handles."""
        probe = self

        class _ProbedMethod:
            def __init__(self, method) -> None:
                self._method = method

            def remote(self, *args, **kwargs):
                t = time.monotonic()
                ref = self._method.remote(*args, **kwargs)
                probe.observe(ref)
                return ref

        class _HandleProxy:
            def __init__(self, handle) -> None:
                self._handle = handle

            def __getattr__(self, name):
                method = getattr(self._handle, name)
                if name in ("record_batch", "add_result"):
                    return _ProbedMethod(method)
                return method

        class _WrappedAggregatorCls:
            @staticmethod
            def remote(**kwargs):
                return _HandleProxy(aggregator_cls.remote(**kwargs))

        return _WrappedAggregatorCls

    def latency_report(self) -> dict:
        s = sorted(self.samples)
        return {
            "note": (
                "record_batch submit-to-ready latency measured "
                "coordinator-side; Ray exposes no actor mailbox depth, "
                "so FIFO call latency is the proxy for aggregator "
                "queue depth (one sample = one shard call, not one "
                "result)"
            ),
            "probe_every_nth_call": self.every,
            "samples": len(s),
            "p50_s": percentile(s, 0.50),
            "p90_s": percentile(s, 0.90),
            "p99_s": percentile(s, 0.99),
            "max_s": s[-1] if s else 0.0,
        }


def make_bench_tasks(n: int) -> list[EvalTask]:
    return [
        EvalTask(task_id=f"b{i:07d}", prompt="x") for i in range(n)
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Coordinator saturation bench (measurement only)."
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--latency-s", type=float, default=0.02, dest="latency_s",
                   help="Mean fake batch latency (seconds)")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--tasks", type=int, default=2000)
    p.add_argument("--fail-rate", type=float, default=0.0, dest="fail_rate",
                   help="Injected batch failure rate via the shared "
                        "decider; > 0 exercises the retry machinery "
                        "under load")
    p.add_argument("--jitter", type=float, default=0.1,
                   help="Latency jitter fraction: sleep = latency_s * "
                        "(1 + jitter * U(-1, 1))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aggregator-shards", type=int, default=1,
                   dest="aggregator_shards",
                   help="ResultsAggregator shard actors behind the "
                        "ShardedAggregator facade (default 1 = the "
                        "pre-shard single actor); the dial the shard sweep "
                        "sweep turns")
    p.add_argument("--decider-shards", type=int, default=1,
                   dest="decider_shards",
                   help="FailureDecider shard actors; only used "
                        "when --fail-rate > 0. Sharding provably "
                        "cannot change which batches fail (key-only "
                        "routing, shared seed), so faulted overlays "
                        "stay comparable across shard counts")
    p.add_argument("--agg-probe-every", type=int, default=32,
                   dest="agg_probe_every",
                   help="Sample every Nth record_batch call for the "
                        "call-latency proxy (bounds probe overhead)")
    p.add_argument("--out", type=str, default=None,
                   help="JSON report path (default: bench/results/"
                        "sat_w{W}_l{ms}_b{B}.json)")
    return p


def default_out_path(ns: argparse.Namespace) -> str:
    ms = int(round(ns.latency_s * 1000))
    return os.path.join(
        "bench", "results",
        f"sat_w{ns.workers}_l{ms:03d}_b{ns.batch_size}.json",
    )


def run_saturation(ns: argparse.Namespace) -> dict:
    """Run one configuration and return the report dict."""
    try:
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
        )
    except TypeError:  # older/newer ray without include_dashboard kwarg
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    tmpdir = tempfile.mkdtemp(prefix="sat_bench_")
    jsonl_path = os.path.join(tmpdir, "results.jsonl")

    decider = (
        ShardedFailureDecider(seed=ns.seed, n_shards=ns.decider_shards)
        if ns.fail_rate > 0.0 else None
    )
    worker_kwargs = {
        "latency_s": ns.latency_s,
        "jitter": ns.jitter,
        "failure_rate": ns.fail_rate,
        "decider": decider,
        "seed": ns.seed,
    }

    recording = RecordingMetrics()
    probe = AggregatorCallProbe(every=ns.agg_probe_every)

    coordinator = DistributedEvalCoordinator(
        n_workers=ns.workers,
        model_name="fake-latency",
        backend="hf",  # CPU factory path; FakeLatencyWorker ignores device
        max_retries=2,
        task_timeout=max(60.0, 20.0 * ns.latency_s),
        output_path=jsonl_path,
        batch_size=ns.batch_size,
        aggregator_cls=probe.wrap(ResultsAggregator),
        aggregator_shards=ns.aggregator_shards,
        worker_cls=FakeLatencyWorker,
        worker_kwargs=worker_kwargs,
    )
    # The only bench hook into the coordinator: the metrics seam.
    coordinator._metrics = recording

    tasks = make_bench_tasks(ns.tasks)
    wall_start = time.monotonic()
    summary = coordinator.run(tasks)
    wall_s = time.monotonic() - wall_start
    probe.close()

    # Offered vs achieved
    offered = ns.workers * ns.batch_size / ns.latency_s
    achieved_wall = summary["total"] / wall_s if wall_s > 0 else 0.0

    # Loop rate over the loop_iter span
    loop_span = recording.timer_span("loop_iter")
    loop_iters = recording.timer_count("loop_iter")
    loop_span_s = (loop_span[1] - loop_span[0]) if loop_span else 0.0
    iters_per_s = loop_iters / loop_span_s if loop_span_s > 0 else 0.0

    # Steady-state timer shares (middle 60% of the loop_iter span)
    window = recording.steady_window("loop_iter", trim=0.2)
    t_lo, t_hi = window if window else (None, None)
    loop_total = recording.timer_total("loop_iter", t_lo, t_hi)
    inner = {
        name: recording.timer_total(name, t_lo, t_hi)
        for name in ("ray_wait", "dispatch", "agg_submit")
    }
    shares = {
        name: (dur / loop_total if loop_total > 0 else 0.0)
        for name, dur in inner.items()
    }
    shares["unaccounted"] = max(
        0.0,
        1.0 - sum(shares[n] for n in ("ray_wait", "dispatch", "agg_submit")),
    )
    window_s = (t_hi - t_lo) if window else 0.0
    duty_cycle = loop_total / window_s if window_s > 0 else 0.0

    report = {
        "args": {k: v for k, v in vars(ns).items()},
        "env": {
            "ray_version": ray.__version__,
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "offered_load_tasks_per_s": offered,
        "achieved": {
            "tasks_recorded": summary["total"],
            "succeeded": summary["succeeded"],
            "failed": summary["failed"],
            "wall_time_s": wall_s,
            "throughput_tasks_per_s_wall": achieved_wall,
            "throughput_tasks_per_s_aggregator": summary["throughput_per_s"],
        },
        "saturated": achieved_wall < 0.8 * offered,
        "loop": {
            "iterations": loop_iters,
            "span_s": loop_span_s,
            "iterations_per_s": iters_per_s,
        },
        "steady_state": {
            "window_definition": "middle 60% of the loop_iter span",
            "window_s": window_s,
            "loop_duty_cycle": duty_cycle,
            "timer_totals_s": {"loop_iter": loop_total, **inner},
            "share_of_loop_iter": shares,
        },
        "agg_call_latency": probe.latency_report(),
        "gauges": {
            "interval_s": GAUGE_SAMPLE_INTERVAL_S,
            "series": {
                name: recording.gauge_downsample(
                    name, GAUGE_SAMPLE_INTERVAL_S
                )
                for name in ("pending", "active", "deferred", "standby")
            },
        },
        "counts": {
            name: recording.count_total(name)
            for name in ("completed", "failed", "retried", "replaced")
        },
        "results_jsonl": jsonl_path,
    }
    return report


def main(argv: list[str] | None = None) -> None:
    ns = build_arg_parser().parse_args(argv)
    out = ns.out or default_out_path(ns)
    ns.out = out

    report = run_saturation(ns)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    a = report["achieved"]
    print(
        f"offered {report['offered_load_tasks_per_s']:.1f} tasks/s | "
        f"achieved {a['throughput_tasks_per_s_wall']:.1f} tasks/s "
        f"({a['tasks_recorded']} tasks in {a['wall_time_s']:.1f}s) | "
        f"loop {report['loop']['iterations_per_s']:.0f} it/s | "
        f"saturated={report['saturated']}"
    )
    shares = report["steady_state"]["share_of_loop_iter"]
    print(
        "steady-state loop share: "
        + ", ".join(f"{k}={v:.1%}" for k, v in shares.items())
    )
    lat = report["agg_call_latency"]
    print(
        f"record_batch call latency (mailbox proxy): "
        f"p50={lat['p50_s'] * 1000:.1f}ms p99={lat['p99_s'] * 1000:.1f}ms "
        f"({lat['samples']} samples)"
    )
    print(f"wrote {out}")
    ray.shutdown()


if __name__ == "__main__":
    main()
