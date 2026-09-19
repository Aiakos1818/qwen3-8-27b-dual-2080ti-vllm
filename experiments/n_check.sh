#!/usr/bin/env bash
# Follow-up n checks after the 128K sweep: does the n=4/5 win hold at short
# context (31.5K, the §6 baseline prompt) and at the production context (250K)?
set -uo pipefail
T=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$T/.." && pwd)
cd "$REPO_ROOT"
KEY="${VLLM_API_KEY:?set VLLM_API_KEY}"
OUT=$T/n_check_results.txt
: > "$OUT"

run() {  # run <n> <words> <label>
  local n=$1 words=$2 label=$3
  echo "=== n=$n  $label ===" >> "$OUT"
  ./scripts/tools/stop_server.sh 8000 >/dev/null 2>&1
  sleep 5
  local L=$T/nc_${label}_n$n.log
  echo "$L" > $T/current_server_log
  MAX_MODEL_LEN=262144 KV_CACHE_MEMORY_BYTES=5300000000 SPEC_NUM_TOKENS=$n \
    nohup setsid bash $T/run_ab_128k.sh > "$L" 2>&1 < /dev/null &
  sleep 20
  bash scripts/tools/wait_server.sh "$L" 900 8000 >> "$OUT" 2>&1
  echo "  pool: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$L" | head -1)" >> "$OUT"
  for r in 1 2; do
    echo "  --- run $r ---" >> "$OUT"
    python3 $T/decode_profile.py --url http://127.0.0.1:8000 --model qwen38-27b \
      --api-key "$KEY" --words "$words" --max-tokens 512 --seed 95031 >> "$OUT" 2>&1
  done
}

# 31.5K context: the same prompt shape as the section 6 baseline
for n in 3 4 5; do run "$n" 30300 short; done
# 250K context: the C3 prompt
for n in 3 5; do run "$n" 240000 long; done
echo "CHECK_DONE" >> "$OUT"
