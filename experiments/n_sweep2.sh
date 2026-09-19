#!/usr/bin/env bash
# n sweep at 128K context, no offload (the 15 GiB host is marginal with a 4.3 GiB
# staging region). Each config pays one prefill; the second run of the same prompt
# hits the in-session prefix cache, so it is prefill-free.
set -uo pipefail
T=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$T/.." && pwd)
cd "$REPO_ROOT"
KEY="${VLLM_API_KEY:?set VLLM_API_KEY}"
OUT=$T/n_sweep2_results.txt
: > "$OUT"

read -ra N_LIST_ARR <<< "${N_LIST:-3 2 4 5}"
for n in "${N_LIST_ARR[@]}"; do
  echo "=== n=$n ===" >> "$OUT"
  ./scripts/tools/stop_server.sh 8000 >/dev/null 2>&1
  sleep 5
  L=$T/ns_n$n.log
  echo "$L" > $T/current_server_log
  MAX_MODEL_LEN=131072 KV_CACHE_DTYPE=fp8_e4m3 KV_CACHE_MEMORY_BYTES=3000000000 \
    SPEC_NUM_TOKENS=$n nohup setsid bash $T/run_ab_128k.sh > "$L" 2>&1 < /dev/null &
  sleep 20
  bash scripts/tools/wait_server.sh "$L" 900 8000 >> "$OUT" 2>&1
  echo "  pool: $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$L" | head -1)" >> "$OUT"
  for r in 1 2; do
    echo "  --- run $r ---" >> "$OUT"
    python3 $T/decode_profile.py --url http://127.0.0.1:8000 --model qwen38-27b \
      --api-key "$KEY" --words 120000 --max-tokens 512 --seed 95031 >> "$OUT" 2>&1
  done
done
echo "SWEEP_DONE" >> "$OUT"
