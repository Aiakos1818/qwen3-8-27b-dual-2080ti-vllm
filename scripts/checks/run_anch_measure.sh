#!/usr/bin/env bash
# One anchor-measure run: fresh engine, run one ~86k resident request while
# sampling per-GPU memory.used. Paths/model come from .env (repo root).
#
# Usage: run_anch_measure.sh <tag> <cadence_tokens> <anchors>
# Env: LOG_DIR (default /tmp/vllm_logs), VLLM_PYTHON, VLLM_BASE_URL.
set -u

TAG=$1
CKPT=$2
ANCH=$3
SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)          # scripts/checks
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)             # repo root
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi
PY=${VLLM_PYTHON:-python}
BASE=${VLLM_BASE_URL:-http://localhost:8000/v1}
LOG_DIR=${LOG_DIR:-/tmp/vllm_logs}
mkdir -p "$LOG_DIR"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
pkill -9 -f '[v]llm.entrypoints' 2>/dev/null
sleep 2

VLLM_MAMBA_CKPT_TOKENS=$CKPT VLLM_MAMBA_CKPT_ANCHORS=$ANCH VLLM_PIN_MIN_TOKENS=0 \
  "$REPO_ROOT/scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh" > "$LOG_DIR/anch_${TAG}_server.log" 2>&1 &
ok=""
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/models" 2>/dev/null)
  if [ "$code" = 200 ]; then ok=1; break; fi
  sleep 5
done
if [ -z "$ok" ]; then echo "TAG=$TAG FAILED START"; exit 1; fi

# baseline idle
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$LOG_DIR/anch_${TAG}_idle.txt"

# background sampler every 1s during the run
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$LOG_DIR/anch_${TAG}_samples.txt"; sleep 1; done ) &
SPID=$!

timeout 300 "$PY" "$REPO_ROOT/scripts/probes/resident_once.py" > "$LOG_DIR/anch_${TAG}_run.log" 2>&1
kill $SPID 2>/dev/null
echo "=== TAG=$TAG ckpt=$CKPT anchors=$ANCH done ==="
