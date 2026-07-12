#!/usr/bin/env bash
# kill the worker node's Ray processes at T+120s. Error path (expected
# primary, ~1-30s after the kill). 2. Hang path (backstop, at hang_threshold_s
# = max(120, 2*task_timeout + 30) = 150s for --task-timeout 60).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/bench/multinode/logs"
mkdir -p "$LOG_DIR"

TARGET="${1:?usage: kill_node.sh <worker-private-ip>|--local [T0-epoch] [delay]}"
T0="${2:-}"
DELAY="${3:-120}"

if [ -n "$T0" ]; then
    NOW="$(python3 -c 'import time; print(time.time())')"
    WAIT="$(python3 -c "print(max(0.0, $T0 + $DELAY - $NOW))")"
    echo "Sleeping ${WAIT}s until T0+${DELAY}s ..."
    sleep "$WAIT"
else
    echo "No T0 given; sleeping ${DELAY}s from now ..."
    sleep "$DELAY"
fi

KILL_EPOCH="$(python3 -c 'import time; print(f"{time.time():.3f}")')"
if [ "$TARGET" = "--local" ]; then
    ray stop --force
else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "ubuntu@$TARGET" '~/venv/bin/ray stop --force'
fi
echo "$KILL_EPOCH" > "$LOG_DIR/kill_marker"
echo "Killed Ray on $TARGET at epoch $KILL_EPOCH (written to $LOG_DIR/kill_marker)"
echo "Watch run.log for: 'Batch failed on worker' / 'replacing worker' /"
echo "'replaced by standby'; or 'ref outstanding for' if the hang backstop fired."
