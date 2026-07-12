#!/usr/bin/env bash
# launch the 20K-task vLLM run against the two-node cluster.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/bench/multinode/logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate" 2>/dev/null || true

export RAY_ADDRESS="auto"   # connect to the `ray start` cluster; error if none

TASKS="${TASKS:-20000}"
STANDBY="${STANDBY:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-60}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

TOTAL_GPUS="$(python - <<'EOF'
import ray
ray.init(address="auto", logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
EOF
)"
WORKERS="${WORKERS:-$TOTAL_GPUS}"

echo "Cluster GPUs: $TOTAL_GPUS  workers: $WORKERS  standby: $STANDBY"
if [ "$TOTAL_GPUS" -lt "$((WORKERS + STANDBY))" ]; then
    echo "PREFLIGHT FAIL: workers ($WORKERS) + standby ($STANDBY) > GPUs ($TOTAL_GPUS)." >&2
    echo "The coordinator's 120s init barrier waits on standby health checks;" >&2
    echo "an unschedulable standby aborts the run before the first batch." >&2
    echo "Either provision WORKERS+STANDBY GPUs, or rerun with:" >&2
    echo "  WORKERS=$((TOTAL_GPUS - STANDBY)) bash bench/multinode/run.sh" >&2
    echo "(README.md, 'Known failure 1'.)" >&2
    exit 1
fi

# Rotate any old results file: the aggregator opens --output in APPEND mode.
OUT="$LOG_DIR/results.jsonl"
if [ -s "$OUT" ]; then mv "$OUT" "$OUT.$(date +%s)"; fi
: > "$LOG_DIR/progress.csv"
rm -f "$LOG_DIR/kill_marker"

ray status | tee "$LOG_DIR/ray_status_before.txt"

T0="$(python -c 'import time; print(f"{time.time():.3f}")')"
python - "$LOG_DIR/run_meta.json" <<EOF
import json, sys, time
json.dump({
    "t0_epoch": $T0,
    "tasks": $TASKS,
    "workers": $WORKERS,
    "standby": $STANDBY,
    "task_timeout": $TASK_TIMEOUT,
    "model": "$MODEL",
}, open(sys.argv[1], "w"), indent=2)
EOF
echo "T0 epoch: $T0  (pass this to kill_node.sh)"

python bench/multinode/sample_progress.py \
    --results "$OUT" --out "$LOG_DIR/progress.csv" \
    --workers "$WORKERS" --interval 2 &
SAMPLER_PID=$!
# Snapshot actor placement mid-warmup: shows which node hosts each VLLMWorker.
( sleep 60; ray list actors --detail > "$LOG_DIR/actors_t60.txt" 2>&1 || true ) &
SNAP_PID=$!
trap 'kill "$SAMPLER_PID" "$SNAP_PID" 2>/dev/null || true' EXIT

set +e
python main.py \
    --backend vllm \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --workers "$WORKERS" \
    --standby "$STANDBY" \
    --task-timeout "$TASK_TIMEOUT" \
    --output "$OUT" \
    2>&1 | tee "$LOG_DIR/run.log"
RC=${PIPESTATUS[0]}
set -e

ray status | tee "$LOG_DIR/ray_status_after.txt" || true
echo
echo "exit code: $RC"
echo "log:      $LOG_DIR/run.log"
echo "results:  $OUT"
echo "next:     python bench/multinode/measure.py --logs $LOG_DIR"
exit "$RC"
