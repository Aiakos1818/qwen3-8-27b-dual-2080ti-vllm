#!/usr/bin/env bash
# One cadence C: clean engine -> healthy -> run scenario -> kill. Append TSV.
#
# Usage: run_cmatrix_one.sh <cadence_tokens>
# Starts the repo 100K profile with VLLM_MAMBA_CKPT_TOKENS=$C, waits for the
# engine, runs scripts/revert_cmatrix.py once, appends its RESULT rows to a TSV,
# then stops the engine. Paths/model come from .env (repo root).
#
# Env: LOG_DIR (default /tmp/vllm_logs), VLLM_PYTHON, VLLM_BASE_URL.
set -u

C=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi
PY=${VLLM_PYTHON:-python}
BASE=${VLLM_BASE_URL:-http://localhost:8000/v1}
LOG_DIR=${LOG_DIR:-/tmp/vllm_logs}
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/cmatrix_${C}.log
SERVER_LOG=$LOG_DIR/server_100k.log
RES=$LOG_DIR/cmatrix_results.tsv

# stop any existing engine
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
pkill -9 -f '[v]llm.entrypoints' 2>/dev/null
sleep 2

VLLM_MAMBA_CKPT_TOKENS=$C "$REPO_ROOT/scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh" > "$SERVER_LOG" 2>&1 &
ok=""
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/models" 2>/dev/null)
  if [ "$code" = 200 ]; then ok=1; break; fi
  sleep 5
done
if [ -z "$ok" ]; then echo "C=$C ENGINE FAILED TO START" | tee "$RES" >/dev/null; exit 1; fi

size=$(grep -a 'GPU KV cache size' "$SERVER_LOG" | tail -1 | grep -oE '[0-9,]+ tokens' | tr -d ' ,')
conc=$(grep -a 'Maximum concurrency' "$SERVER_LOG" | tail -1 | grep -oE '[0-9.]+x' | head -1)

timeout 900 "$PY" "$REPO_ROOT/scripts/revert_cmatrix.py" > "$LOG" 2>&1
ec=$?
echo "=== C=$C kv=$size conc=$conc exit=$ec ===" | tee -a "$RES"
grep '^RESULT' "$LOG" >> "$RES"

# stop engine for the next C
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
pkill -9 -f '[v]llm.entrypoints' 2>/dev/null
exit 0
