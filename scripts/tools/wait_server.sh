#!/usr/bin/env bash
# Event-driven "wait until the server is up, or give up the moment it fails".
#
# Why this exists: the engine needs minutes to load weights and compile, and the
# real failure modes (cudaHostRegister poisoning the CUDA context, /dev/shm
# exhaustion, engine-init errors) happen *during* that window. Sleeping a fixed
# 30s and then checking both wastes time and still misses the moment of failure.
# This polls every 2s and returns as soon as `/health` answers 200 or a fatal
# pattern shows up in the log.
#
# Usage:
#   wait_server.sh <logfile> [max_seconds] [port] [launcher_pid]
#
# If launcher_pid is given and that process exits before the server is ready,
# this returns fatal immediately (covers failures whose log line we do not
# pattern-match), printing the last log lines.
#
# Exit codes:
#   0 = ready           (printed: "[wait] READY after Ns")
#   1 = fatal found     (printed: the matching log line, or the dead-pid tail)
#   2 = timed out
#
# Port defaults to $PORT from .env (falling back to 8000).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi

LOG=${1:?usage: wait_server.sh <logfile> [max_seconds] [port] [launcher_pid]}
MAX=${2:-600}
PORT=${3:-${PORT:-8000}}
PID=${4:-}
POLL=2

# Fatal patterns. Deliberately excludes the benign per-start
# "Cannot use FA version 2 ..." ERROR lines emitted on SM75.
#
# The config-validation group matters because a bad flag (e.g. an unsupported
# speculative method under the V2 runner) fails while VllmConfig is built, before
# the engine exists -- so none of the engine/worker patterns below ever fire and
# the process just exits. Without these the wait would sit until the timeout.
ERR_RE='CUDA error: invalid argument|Engine core initialization failed|EngineCore failed to start|Insufficient space in /dev/shm|Worker failed with error|WorkerProc hit an exception|torch\.AcceleratorError|died unexpectedly|pydantic_core\._pydantic_core\.ValidationError|ValidationError: [0-9]+ validation error|Value error, .*(not (yet )?support|does not support)|Unsupported speculative method|is not supported yet'

start=$(date +%s)
while true; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$MAX" ]; then
    echo "[wait] TIMEOUT after ${MAX}s (log: $LOG)"
    exit 2
  fi

  if [ -f "$LOG" ]; then
    # ``|| true``: with ``set -e``/``pipefail`` a grep with no match would exit
    # the script instead of continuing to poll (the healthy case has no match).
    hit=$(grep -aE "$ERR_RE" "$LOG" 2>/dev/null \
          | grep -avE 'fa_utils|Cannot use FA' | tail -1) || true
    if [ -n "$hit" ]; then
      echo "[wait] FATAL after $((now - start))s:"
      echo "$hit"
      exit 1
    fi
  fi

  if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
    echo "[wait] FATAL after $((now - start))s: launcher pid $PID exited before readiness:"
    tail -n 5 "$LOG" 2>/dev/null || true
    exit 1
  fi

  code=$(curl -s -m 2 "http://127.0.0.1:${PORT}/health" -o /dev/null -w '%{http_code}' 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "[wait] READY after $((now - start))s"
    exit 0
  fi

  sleep "$POLL"
done
