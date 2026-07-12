#!/usr/bin/env bash
# worker-node setup for the two-node kill test.
set -euo pipefail

HEAD_IP="${1:?usage: setup_worker.sh <head-private-ip> [own-private-ip] [HF_HOME]}"
OWN_IP="${2:-}"
export HF_HOME="${3:-${HF_HOME:-$HOME/hf-home}}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

PY="$(command -v python3.11 || command -v python3)"
echo "Using interpreter: $PY ($("$PY" --version 2>&1))"
if [ ! -d "$HOME/venv" ]; then "$PY" -m venv "$HOME/venv"; fi
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

mkdir -p "$HF_HOME"
echo "Pre-downloading $MODEL into HF_HOME=$HF_HOME ..."
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("model present in cache")
EOF

ray stop --force >/dev/null 2>&1 || true
EXTRA=()
if [ -n "$OWN_IP" ]; then EXTRA+=(--node-ip-address="$OWN_IP"); fi
ray start --address="$HEAD_IP:6379" \
    --min-worker-port=10002 \
    --max-worker-port=10100 \
    "${EXTRA[@]}"

sleep 3
ray status
echo
echo "Worker joined $HEAD_IP:6379. Run the eval from the head node."
