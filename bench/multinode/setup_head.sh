#!/usr/bin/env bash
# head-node setup for the two-node kill test. Model:
# TinyLlama/TinyLlama-1.1B-Chat-v1.0 - 1.1B params, ~2.2 GB in fp16, ungated,
# supported by the pinned vllm==0.15.0; fits an A10 (24 GB) with room for KV
set -euo pipefail

HEAD_IP="${1:?usage: setup_head.sh <head-private-ip> [HF_HOME]}"
export HF_HOME="${2:-${HF_HOME:-$HOME/hf-home}}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# --- venv + pinned deps -------------------------------------------------
PY="$(command -v python3.11 || command -v python3)"
echo "Using interpreter: $PY ($("$PY" --version 2>&1))"
if [ ! -d "$HOME/venv" ]; then "$PY" -m venv "$HOME/venv"; fi
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

# --- model pre-download into the (possibly shared) HF_HOME -------------- If.
mkdir -p "$HF_HOME"
echo "Pre-downloading $MODEL into HF_HOME=$HF_HOME ..."
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("model present in cache")
EOF

# --- start the Ray head (HF_HOME is exported in this shell) -------------
ray stop --force >/dev/null 2>&1 || true
ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --min-worker-port=10002 \
    --max-worker-port=10100

sleep 3
ray status
echo
echo "Head up at $HEAD_IP:6379. Now run setup_worker.sh $HEAD_IP on the other box."
