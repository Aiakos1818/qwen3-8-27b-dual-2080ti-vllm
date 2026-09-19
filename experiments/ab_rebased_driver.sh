#!/usr/bin/env bash
# Single-phase A/B: run the V1 model runner on the production profile and bench it.
set -uo pipefail

T=$(cd -P "$(dirname "$0")" && pwd)
REPO=$(cd "$T/.." && pwd)
TOOLS="$REPO/scripts/tools"
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

LOG="$T/runner_rebased.log"
: >"$LOG"

echo "[rebased] launching VLLM_USE_V2_MODEL_RUNNER=0 (see run_256k_mtp6_v2.sh:83)"
setsid bash "$T/run_256k_mtp6_v2.sh" --head8bit >"$LOG" 2>&1 &
echo "[rebased] launcher pid=$!"

if "$TOOLS/wait_server.sh" "$LOG" 900 8000; then
  pid=$(grep -oE "APIServer pid=[0-9]+" "$LOG" | head -1 | cut -d= -f2)
  echo "[rebased] server pid=${pid:-?}  'Using V2 Model Runner' lines: $(grep -c 'Using V2 Model Runner' "$LOG")"
  wpid=$(grep -oE "Worker pid=[0-9]+" "$LOG" | head -1 | cut -d= -f2)
  if [ -n "${wpid:-}" ]; then
    echo "[rebased] worker $wpid env:"
    tr '\0' '\n' <"/proc/$wpid/environ" 2>/dev/null |
      grep -E "VLLM_USE_V2_MODEL_RUNNER|VLLM_SM75_SPEC_SYNC_MODE|VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE" |
      sed 's/^/[rebased]   /'
  fi
  "$VLLM_PYTHON" "$REPO/benchmarks/run_context_ttft.py" \
    --model "$SERVED_MODEL_NAME" \
    --word-counts 30000 --max-tokens 256 --runs 3 --timeout 3600 \
    --output $T/ab_rebased.json 2>&1 | tee $T/bench_rebased.out | tail -12
else
  echo "[rebased] server not ready; no bench. log tail:"
  tail -30 "$LOG"
fi

echo "[rebased] stopping server"
"$TOOLS/stop_server.sh" 8000 2>&1 | tail -4
echo "[rebased] DONE"
