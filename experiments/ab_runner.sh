#!/usr/bin/env bash
# A/B: V2 model runner (production env) vs V1, same 256K profile + MTP n=6 + head8bit.
set -uo pipefail

T=$(cd -P "$(dirname "$0")" && pwd)
OUTDIR=$T
REPO=$(cd "$T/.." && pwd)
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a
PY="${VLLM_PYTHON:?}"
BENCH="$REPO/benchmarks/run_context_ttft.py"

run_mode() {
  local mode=$1 script=$2
  local log="$OUTDIR/runner_${mode}.log"
  echo "[driver] ==== $mode  ($script) ===="
  : >"$log"
  setsid bash "$script" --head8bit >"$log" 2>&1 &
  local pid=$!
  echo "[driver] $mode pid=$pid"
  local t=0 ok=0
  while [ "$t" -lt 900 ]; do
    if grep -q "Application startup complete" "$log" 2>/dev/null; then ok=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[driver] $mode: process died early"; break; fi
    sleep 15
    t=$((t + 15))
  done
  echo "[driver] $mode ready=$ok waited=${t}s"
  grep -m1 -iE "Using V[12] Model Runner|does not yet support" "$log" | sed "s/^/[driver:$mode] /"
  if [ "$ok" = 1 ]; then
    "$PY" "$BENCH" --word-counts 8100 --max-tokens 256 --runs 3 --timeout 3600 \
      --output "$OUTDIR/ab_runner_${mode}.json" 2>&1 | tail -6 | sed "s/^/[bench:$mode] /"
  fi
  local pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "${pgid:-}" ] && kill -TERM -- "-$pgid" 2>/dev/null
  sleep 10
  if kill -0 "$pid" 2>/dev/null; then
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "${pgid:-}" ] && kill -KILL -- "-$pgid" 2>/dev/null
    sleep 3
  fi
  echo "[driver] $mode finished (leftover procs: $(pgrep -fc '[v]llm.entrypoints' 2>/dev/null || echo 0))"
}

run_mode v2 $T/run_256k_mtp6_v2.sh
run_mode v1 $T/run_256k_mtp6_v1.sh
echo "[driver] ALL DONE"
