# Shard sweeps on the 32-vCPU host

Everything below runs on the same environment as BASELINE (Vultr
dedicated-CPU, 32 vCPU, Python 3.12, ray 2.50.0) or the overlay is not
comparable. Run inside tmux; the clean grids are hours of wall clock.

## 0. Sync + verify

From the local repo root:

    tar --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache \
        --exclude='results/*.jsonl' --exclude='bench/results/*.json' \
        -czf - . | ssh vultr "mkdir -p ray-eval-harness && tar -xzf - -C ray-eval-harness"

On the host:

    cd ray-eval-harness && source venv/bin/activate
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
    python -m pytest tests -q

## 1. Clean grid x aggregator shards (fail_rate = 0)

One out-dir per (setting, rep). Do NOT mix settings in a directory:
plot_saturation keys curves by (batch, fail_rate) only and would merge
different shard settings into one bogus curve. The `_a{A}_d{D}` filename
suffix prevents silent skip-collisions but is not a substitute for
separate directories.

a1 is the regression anchor (must reproduce BASELINE within its <=1.5%
per-rep spread) and gets 3 reps like BASELINE. a4/a8 get 1 rep minimum;
add reps 2-3 if host time is available.

    tmux new -s d4b
    for rep in 1 2 3; do
      python -m bench.sweep_saturation --aggregator-shards 1 \
          --out-dir bench/results/sharded/a1/rep$rep
    done
    python -m bench.sweep_saturation --aggregator-shards 4 \
        --out-dir bench/results/sharded/a4/rep1
    python -m bench.sweep_saturation --aggregator-shards 8 \
        --out-dir bench/results/sharded/a8/rep1

Budget: ~40-50 min per full-grid rep on the 32-vCPU box (deep batch=1
points dominate: up to 200k tasks at ~2k tasks/s each). 5 reps total
=~ 3.5-4.5 h. Sweeps resume: re-running the same command skips
completed points.

Sanity mid-flight (any time): the a1 knees must match BASELINE.

    python -m bench.plot_saturation bench/results/sharded/a1/rep1 --no-plot
    # b=1 knee at offered 2560, b=8 at 4096 (marginal), b=64 at 8192

## 2. Faulted row x decider shards (l=0.02, b=8, 20k tasks)

Same row as BASELINE's appendix. Keep --aggregator-shards pinned at 1
here so decider sharding is the ONLY variable moving; the
aggregator-shard effect on the faulted row can be a follow-up once the sweep
picks a default.

    for d in 1 4; do
      for w in 16 64 128 256; do
        python -m bench.saturation --workers $w --latency-s 0.02 \
            --batch-size 8 --tasks 20000 --fail-rate 0.1 --seed 0 \
            --aggregator-shards 1 --decider-shards $d \
            --out bench/results/sharded/faulted_d$d/sat_w${w}_l020_b8_a1_d$d.json
      done
    done

Budget: ~10 min total. Determinism check (must hold, it is the decider-sharding
guarantee): per worker count, the `counts` blocks (completed / failed /
retried) of faulted_d1 and faulted_d4 must be identical.

    python - << 'EOF'
    import json, glob
    for w in (16, 64, 128, 256):
        a = json.load(open(glob.glob(f"bench/results/sharded/faulted_d1/sat_w{w}_*.json")[0]))
        b = json.load(open(glob.glob(f"bench/results/sharded/faulted_d4/sat_w{w}_*.json")[0]))
        assert a["counts"] == b["counts"], (w, a["counts"], b["counts"])
        print(w, "ok", a["counts"])
    EOF

## 3. Fetch

From the local machine:

    scp -r vultr:~/ray-eval-harness/bench/results/sharded ./bench-results-sharded


## Tooling notes

- plot_saturation `--compare` takes exactly TWO paths (nargs=2). The
  baseline-vs-a4-vs-a8 overlay is three-way, and curves are not keyed
  by shard setting - both are expected plot-script changes inside the
  the shard sweep, not prerequisites.
- Reports carry their full args (including shard settings), so the
  analysis reads settings from JSON, never from filenames.
- The record_batch mailbox proxy samples every 4th call (see the
  --agg-probe-every default rationale in bench/saturation.py); at the
  2000-task sweep floor that is ~62 samples per point.
