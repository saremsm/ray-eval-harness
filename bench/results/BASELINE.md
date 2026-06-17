# Coordinator saturation baseline

Frozen reference for the before/after overlays. Every number below is
computed from the JSON reports under `bench/results/published/` (three
independent full-grid sweeps `rep1/ rep2/ rep3/`, one long knee run
`knee_long.json`, and the clean/faulted comparison row `clean_row/`
`faulted/`). No numbers from memory, no re-runs.

Environment (recorded in every report's `env` block): Vultr
dedicated-CPU, 32 vCPU, Python 3.12.3, ray 2.50.0. The stray
`bench/results/sat_w8_l050_b8.json` is a 16-vCPU / Python 3.10.6
Windows smoke run from a different machine and is **excluded** from
everything below.

Grid: latency {0.5, 0.1, 0.02, 0.005} s x workers {16, 64, 128, 256} x
batch {1, 8, 64}; offered load = `workers * batch / latency` tasks/s;
`fail_rate = 0` throughout the grid. Values below are the mean of the
three reps; per-rep spread on achieved throughput is <= 1.5% at every
grid point (worst case: batch=8 knee-region points, e.g. w16/l0.02/b8 at
4,841.7-4,911.0).

## Achieved vs offered

Reproduce the raw curves/knees with:
`python -m bench.plot_saturation bench/results/published/rep1 bench/results/published/rep2 bench/results/published/rep3 --no-plot`

### batch = 1

| workers | latency (s) | offered (t/s) | achieved (t/s, mean of 3) | achieved/offered |
|--------:|------------:|--------------:|--------------------------:|-----------------:|
|      16 |         0.5 |            32 |                      31.5 |            0.985 |
|      64 |         0.5 |           128 |                     120.2 |            0.939 |
|      16 |         0.1 |           160 |                     153.7 |            0.961 |
|     128 |         0.5 |           256 |                     232.0 |            0.906 |
|     256 |         0.5 |           512 |                     428.0 |            0.836 |
|      64 |         0.1 |           640 |                     594.5 |            0.929 |
|      16 |        0.02 |           800 |                     707.9 |            0.885 |
|     128 |         0.1 |         1,280 |                   1,134.0 |            0.886 |
|     256 |         0.1 |         2,560 |                   1,864.2 |        **0.728** |
|      16 |       0.005 |         3,200 |                   2,121.6 |            0.663 |
|      64 |        0.02 |         3,200 |                   2,541.2 |            0.794 |
|     128 |        0.02 |         6,400 |                   2,388.0 |            0.373 |
|      64 |       0.005 |        12,800 |                   2,687.3 |            0.210 |
|     256 |        0.02 |        12,800 |                   1,960.7 |            0.153 |
|     128 |       0.005 |        25,600 |                   2,402.9 |            0.094 |
|     256 |       0.005 |        51,200 |                   1,977.4 |            0.039 |

### batch = 8

| workers | latency (s) | offered (t/s) | achieved (t/s, mean of 3) | achieved/offered |
|--------:|------------:|--------------:|--------------------------:|-----------------:|
|      16 |         0.5 |           256 |                     247.9 |            0.968 |
|      64 |         0.5 |         1,024 |                     956.6 |            0.934 |
|      16 |         0.1 |         1,280 |                   1,204.8 |            0.941 |
|     128 |         0.5 |         2,048 |                   1,815.7 |            0.887 |
|     256 |         0.5 |         4,096 |                   3,273.0 |        **0.799** |
|      64 |         0.1 |         5,120 |                   4,401.7 |            0.860 |
|      16 |        0.02 |         6,400 |                   4,886.2 |            0.763 |
|     128 |         0.1 |        10,240 |                   4,652.7 |            0.454 |
|     256 |         0.1 |        20,480 |                   4,497.7 |            0.220 |
|      16 |       0.005 |        25,600 |                   4,823.2 |            0.188 |
|      64 |        0.02 |        25,600 |                   4,697.3 |            0.183 |
|     128 |        0.02 |        51,200 |                   4,610.4 |            0.090 |
|      64 |       0.005 |       102,400 |                   4,719.5 |            0.046 |
|     256 |        0.02 |       102,400 |                   4,529.3 |            0.044 |
|     128 |       0.005 |       204,800 |                   4,642.3 |            0.023 |
|     256 |       0.005 |       409,600 |                   4,480.4 |            0.011 |

### batch = 64

| workers | latency (s) | offered (t/s) | achieved (t/s, mean of 3) | achieved/offered |
|--------:|------------:|--------------:|--------------------------:|-----------------:|
|      16 |         0.5 |         2,048 |                   1,906.8 |            0.931 |
|      64 |         0.5 |         8,192 |                   4,692.5 |        **0.573** |
|      16 |         0.1 |        10,240 |                   4,827.5 |            0.471 |
|     128 |         0.5 |        16,384 |                   4,608.3 |            0.281 |
|     256 |         0.5 |        32,768 |                   4,458.1 |            0.136 |
|      64 |         0.1 |        40,960 |                   4,718.1 |            0.115 |
|      16 |        0.02 |        51,200 |                   4,842.1 |            0.095 |
|     128 |         0.1 |        81,920 |                   4,608.7 |            0.056 |
|     256 |         0.1 |       163,840 |                   4,471.7 |            0.027 |
|      16 |       0.005 |       204,800 |                   4,745.6 |            0.023 |
|      64 |        0.02 |       204,800 |                   4,738.5 |            0.023 |
|     128 |        0.02 |       409,600 |                   4,646.0 |            0.011 |
|      64 |       0.005 |       819,200 |                   4,701.4 |            0.006 |
|     256 |        0.02 |       819,200 |                   4,463.1 |            0.005 |
|     128 |       0.005 |     1,638,400 |                   4,594.1 |            0.003 |
|     256 |       0.005 |     3,276,800 |                   4,446.8 |            0.001 |

**The grid reached saturation.** At the deepest points achieved
throughput is 0.1-4% of offered while staying flat at the ceiling
(batch=64 achieves 4,446.8 mean at 3,276,800 offered - 737x
oversubscribed - vs 4,827.5 at 10,240 offered). No follow-up grid is
needed for the clean baseline.

Two grid effects to keep in mind when reading the curves:
- **Duplicate offered loads are different configurations.** Offered
  3,200 (b1) appears as both w16/l0.005 (0.663) and w64/l0.02 (0.794);
  offered 12,800 as w64/l0.005 (2,687.3) and w256/l0.02 (1,960.7).
- **Worker count itself costs.** At fixed offered load, 256 workers
  achieve materially less than 64 (b1 @ 12,800: 2,687.3 vs 1,960.7,
  -27%; b64 ceiling points: w16 ~4,750-4,840 vs w256 ~4,450-4,470).
  This is why the next grid, if one were needed, would raise offered
  load via lower latency and larger batch, not more actors.

## Knees

Knee = first point, in offered-load order, with achieved < 0.8 x
offered (`bench/plot_saturation.py`, `KNEE_RATIO = 0.8`). Identical in
all three reps.

| batch | knee (offered t/s) | knee point | achieved at knee | ratio | note |
|------:|-------------------:|-----------|-----------------:|------:|------|
|     1 | 2,560 | w256, l=0.1 | 1,848.4-1,882.7 | 0.72-0.74 | one rep re-crosses 80% at offered 3,200 (rep1 w64/l0.02: 2,563.2) |
|     8 | 4,096 | w256, l=0.5 | 3,272.0-3,273.8 | 0.799 | **marginal**: re-crosses at offered 5,120 (0.860), permanently below from 6,400 (0.763) |
|    64 | 8,192 | w64, l=0.5 | 4,665.8-4,721.7 | 0.573 | previous grid point is 2,048 (0.931); the true crossing is inside (2,048, 8,192), unsampled |

The b=8 and b=64 knees are consistent with a single per-task ceiling of
~4,900 tasks/s: 0.8 x offered crosses that ceiling at offered ~6,100,
which sits inside b=8's sustained-saturation onset (5,120 -> 6,400) and
inside b=64's unsampled bracket. The b=1 knee is genuinely lower
(~2,560) because batch=1 pays per-batch scheduling costs per task (see
breakdown below).

## Max achieved throughput

| scope | max achieved (t/s) | file |
|-------|-------------------:|------|
| **overall / batch=8** | **4,911.0** | `rep1/sat_w16_l020_b8.json` (w16, l=0.02, offered 6,400; reps: 4,841.7 / 4,905.8) |
| batch=64 | 4,873.3 | `rep1/sat_w16_l100_b64.json` (w16, l=0.1, offered 10,240; reps: 4,760.7 / 4,848.3) |
| batch=1 | 2,701.0 | `rep1/sat_w64_l005_b1.json` (w64, l=0.005, offered 12,800; reps: 2,682.9 / 2,677.9) |

Ceiling summary: **~4,900 tasks/s for batch >= 8; ~2,700 for batch = 1.**
Batching 8 -> 64 moves the ceiling by under 1% (4,886.2 vs 4,842.1 best
grid means); batching 1 -> 8 nearly doubles it - the ceiling cost is
per-*task*, not per-batch.

## Timer breakdown at the knees

Steady-state (middle 60% of the `loop_iter` span), share of `loop_iter`,
mean of 3 reps. Loop duty cycle >= 0.997 at every knee - the driver loop
is busy essentially 100% of the time, so shares of `loop_iter` are
shares of the whole loop.

| timer | b=1 knee (w256/l0.1) | b=8 knee (w256/l0.5) | b=64 knee (w64/l0.5) |
|-------|---------------------:|---------------------:|---------------------:|
| `loop_iter` total (abs) | 14.08 s (duty 0.997) | 12.47 s (duty 0.999) | 12.97 s (duty 1.000) |
| `ray_wait`   | 16.0% (2.26 s) | 34.0% (4.23 s) | 17.6% (2.28 s) |
| `dispatch`   | 35.2% (4.96 s) | 8.5% (1.06 s)  | 3.6% (0.47 s)  |
| `agg_submit` | 25.6% (3.60 s) | 49.7% (6.20 s) | **75.0% (9.73 s)** |
| unaccounted  | 23.1% (3.25 s) | 7.9% (0.98 s)  | 3.8% (0.49 s)  |
| loop iterations/s | 2,176 | 492 | 118 |
| `add_result` proxy p50 / p99 | 0.9 ms / 36.7 ms | 2.4 ms / 43.7 ms | **4,930 ms / 9,923 ms** |

## What saturates first

**The per-task `add_result` submission path to the single
`ResultsAggregator` actor.** Three measurements from the attached files,
in increasing directness:

1. **The ceiling is per-task.** Batch 8 and batch 64 share a ~4,900
   tasks/s ceiling (4,911.0 vs 4,873.3 max) despite an 8x difference in
   batches, completion events, and dispatches per second; batch 1 - one
   `add_result`-per-task *and* one batch per task - halves it to
   2,701.0. Only a cost paid once per task at any batch size fits.
2. **`agg_submit` is the loop at depth.** Its share of `loop_iter`
   grows monotonically with saturation depth: 62.2% at the throughput
   peak (`sat_w16_l020_b8`, mean of 3), 75.0% at the b=64 knee,
   **88.9%** at the deepest grid point (`sat_w256_l005_b64`: 12.54 s of
   the 14.11 s steady-state loop; rep1 89.04%), and **90.05%** in the
   long knee run (`knee_long.json`, w128/l0.005/b64: `agg_submit`
   11.878 s of `loop_iter` 13.191 s, vs `ray_wait` 0.9%). The
   coordinator is not slow at scheduling - `ray_wait` is 0.9-1.6% at
   those points - it is spending nine-tenths of its time submitting
   results.
3. **The aggregator's mailbox is seconds deep.** The `add_result`
   submit-to-ready proxy (FIFO actor, so call latency = queue depth in
   seconds) is p50 0.8-0.9 ms / p99 2-5 ms at unsaturated b=1 points,
   but p50 **7,365.6 ms** / p99 **14,894.3 ms** at `sat_w256_l005_b64`
   (mean of 3; rep1: 7,420.7 / 14,736.0 ms) and p99 **16,558 ms** in
   `knee_long.json` - a four-orders-of-magnitude blowup on the same
   code path.

Secondary, batch=1 only: below ~2,700 tasks/s the b>=8 bottleneck is
not yet binding, and the b=1 ceiling is set by per-batch scheduling
overhead instead - `dispatch` is the largest timer at every deep b=1
point (35.2% at the knee, 39.1% at the ceiling point
`sat_w64_l005_b1`, vs `agg_submit` at 25-32%).

The first target follows directly: batch the aggregator writes (one
`record`-style call per completed batch, or buffered flushes), which
attacks measurements 2 and 3 at their source.

## Appendix: faulted-row anchors for the shard-sweep overlay

Same grid row (l=0.02, b=8), `fail_rate=0.1`, 20,000 tasks, from
`published/clean_row/` vs `published/faulted/` (clean_row files match
rep1 bit-for-bit on achieved throughput). Reproduce:
`python -m bench.plot_saturation --compare bench/results/published/clean_row bench/results/published/faulted`

| workers | clean (t/s) | faulted (t/s) | delta | unaccounted loop share, clean -> faulted |
|--------:|------------:|--------------:|------:|-----------------------------------------:|
|      16 |     4,911.0 |       3,658.1 | -25.5% | 6.8% -> 34.9% |
|      64 |     4,728.3 |       3,675.2 | -22.3% | 10.1% -> 42.1% |
|     128 |     4,462.7 |       3,026.3 | -32.2% | 10.7% -> 40.4% |
|     256 |     4,551.0 |       2,100.0 | **-53.9%** | 10.1% -> 41.7% |

The fault penalty grows with pool size while `agg_submit`'s share
*shrinks* (72.2% -> 46.0% at w256): under faults the loop's time moves
into unaccounted (~41%), which is where the blocking
`ray.get(health_check)` poison probe lives - the second target
(non-blocking poison checks).
