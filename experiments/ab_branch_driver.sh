#!/usr/bin/env bash
# Paired A/B of the pre-rebase branch and the rebased branch, same session.
# Each side: boot the production 256K profile (MTP n=6, head8bit, no thinking
# budget), then 6 runs at 31.2K. The benchmark seeds are deterministic
# (seed = word_count*100 + run), so run N faces the same prompt on both sides.
set -uo pipefail

T=$(cd -P "$(dirname "$0")" && pwd)
REPO=$(cd "$T/.." && pwd)
SRC=$(cd "$REPO/../zyYuc-sandbox/src/vllm-0271" && pwd)
TOOLS="$REPO/scripts/tools"
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

LAUNCH="$T/run_256k_mtp6_v2.sh"   # same launcher as the logged baseline

phase() {
  local name=$1 branch=$2 out=$3
  echo "[ab] ===== $name  branch=$branch ====="
  if ! git -C "$SRC" checkout -q "$branch"; then
    echo "[ab] $name: checkout failed"; return 1
  fi
  echo "[ab] $name HEAD=$(git -C "$SRC" log --oneline -1 | cut -c1-45)"
  local log="$T/runner_$name.log"
  : >"$log"
  setsid bash "$LAUNCH" --head8bit >"$log" 2>&1 &
  if ! "$TOOLS/wait_server.sh" "$log" 900 8000; then
    echo "[ab] $name: server not ready; log tail:"
    tail -20 "$log"
    "$TOOLS/stop_server.sh" 8000 2>&1 | tail -2
    return 1
  fi
  "$VLLM_PYTHON" "$REPO/benchmarks/run_context_ttft.py" \
    --model "$SERVED_MODEL_NAME" \
    --word-counts 30000 --max-tokens 256 --runs 6 --timeout 3600 \
    --output "$out" 2>&1 | tee "$T/bench_${name}6.out" | tail -3
  echo "[ab] $name: stopping"
  "$TOOLS/stop_server.sh" 8000 2>&1 | tail -2
  sleep 5
}

phase old 2080ti_dual_qwen38-27B $T/ab_old6.json
phase new rebase-main-tip $T/ab_new6.json
git -C "$SRC" checkout -q 2080ti_dual_qwen38-27B
echo "[ab] ALL DONE; main tree back on $(git -C "$SRC" rev-parse --abbrev-ref HEAD) $(git -C "$SRC" log --oneline -1 | cut -c1-12)"
