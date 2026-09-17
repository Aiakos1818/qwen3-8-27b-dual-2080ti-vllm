#!/usr/bin/env bash
# Stop one vLLM instance precisely -- by the port it serves, not by a name pattern.
#
# Why not `pkill -f vllm...`: that matches any process whose command line merely
# contains the string (including the shell running the pkill itself), so with two
# instances on different ports it can hit the wrong one or none. This finds the
# instance the way the kernel sees it -- the pid listening on the port -- and stops
# its process tree (api_server + EngineCore + one worker per rank), leaving other
# instances alone.
#
# Usage:
#   stop_server.sh                        # port from $PORT in .env, else 8000
#   stop_server.sh 8001                   # positional port
#   stop_server.sh --port 8001 --dry-run  # show what would be killed, send nothing
#   stop_server.sh --list                 # every running vLLM instance
#   stop_server.sh 8001 --force           # skip the graceful SIGTERM wait
#   stop_server.sh 8001 --no-clean-shm    # keep its /dev/shm staging file
#   stop_server.sh --pid 12345            # stop exactly this pid
#
# The CPU offload region is a real file in /dev/shm (/dev/shm/vllm_offload_<id>.mmap,
# one CPU_BYTES_TO_USE-sized object). This branch never unlinks it -- the startup
# log says "Created/Opened existing mmap file" and never "Unlinked mmap file" --
# so it survives the process and keeps holding tmpfs (and therefore RAM) until
# somebody removes it. Stopping therefore removes this instance's own file by
# default, and prints the /dev/shm usage before/after so the freed space is
# visible; files belonging to other instances are never touched.
#
# --dry-run prints the pid, its process group, every member and the signal that
# would be sent -- nothing is executed -- so you can confirm the target first.
# An explicit --port/positional always beats $PORT from .env.
#
# Exit codes:
#   0 = stopped (or, for --dry-run/--list, the target was identified)
#   1 = found but still alive after the timeout (leftovers printed)
#   2 = nothing to stop (no instance on that port)
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)

ARG_PORT=
PID=
DRY=0
FORCE=0
CLEAN_SHM=1
TIMEOUT=30
MODE=stop

usage() {
  sed -n '2,/^set -Eeuo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port) ARG_PORT=${2:?--port needs a value}; shift 2 ;;
    --pid) PID=${2:?--pid needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --force|-f) FORCE=1; shift ;;
    --clean-shm) CLEAN_SHM=1; shift ;;
    --no-clean-shm) CLEAN_SHM=0; shift ;;
    --list) MODE=list; shift ;;
    -h|--help) usage; exit 0 ;;
    ''|*[!0-9]*) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *) ARG_PORT=$1; shift ;;
  esac
done

# .env is read *after* the arguments on purpose: it sets PORT for the profiles,
# and letting it overwrite an explicit --port would silently target the wrong
# instance.
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi
PORT=${ARG_PORT:-${PORT:-8000}}

MARKERS='vllm.entrypoints.openai.api_server'

# ---------------------------------------------------------------- instance list
# /proc is scanned directly: `pgrep -f` would also match shells and log tails
# that merely mention the module, and mixing those up is how the wrong process
# gets killed. argv[0] must be a python interpreter.
instance_cmdlines() {
  local pid cmdline
  for entry in /proc/[0-9]*; do
    pid=${entry#/proc/}
    [ -r "$entry/cmdline" ] || continue
    cmdline=$(tr '\0' ' ' < "$entry/cmdline" 2>/dev/null) || continue
    case "$cmdline" in
      *"$MARKERS"*) ;;
      *) continue ;;
    esac
    case "${cmdline%% *}" in
      *python*) ;;
      *) continue ;;
    esac
    printf '%s %s\n' "$pid" "$cmdline"
  done
}

port_of_cmdline() {
  sed -n 's/.*--port \([0-9]\+\)\( .*\|$\)/\1/p' <<<"$1"
}

engine_id_of_cmdline() {
  sed -n 's/.*"engine_id":"\([^"]*\)".*/\1/p' <<<"$1"
}

model_of_cmdline() {
  local model
  model=$(sed -n 's/.*--model \([^ ]*\).*/\1/p' <<<"$1")
  printf '%s' "${model##*/}"
}

if [ "$MODE" = list ]; then
  found=0
  while read -r pid cmdline; do
    [ -n "${pid:-}" ] || continue
    printf '  pid %-8s port %-6s engine %-34s model %s\n' \
      "$pid" "$(port_of_cmdline "$cmdline")" \
      "$(engine_id_of_cmdline "$cmdline")" "$(model_of_cmdline "$cmdline")"
    found=$((found + 1))
  done < <(instance_cmdlines)
  [ "$found" -gt 0 ] || echo "  (no vLLM instance found)"
  exit 0
fi

# ------------------------------------------------------------------ locate pid
if [ -z "$PID" ]; then
  # The listening socket is the authoritative answer for "which process serves
  # this port", so prefer it; fall back to the cmdline scan when ss is absent.
  if command -v ss >/dev/null 2>&1; then
    PID=$(ss -ltnpH "sport = :$PORT" 2>/dev/null |
          grep -oP 'pid=\K[0-9]+' | head -1 || true)
  fi
  if [ -z "$PID" ]; then
    while read -r pid cmdline; do
      if [ "$(port_of_cmdline "$cmdline")" = "$PORT" ]; then
        PID=$pid
        break
      fi
    done < <(instance_cmdlines)
  fi
fi

if [ -z "$PID" ] || [ ! -d "/proc/$PID" ]; then
  echo "[stop] no vLLM instance on port $PORT (nothing to stop)"
  exit 2
fi

CMDLINE=$(tr '\0' ' ' < "/proc/$PID/cmdline")
if [[ "$CMDLINE" != *"$MARKERS"* ]]; then
  echo "[stop] refusing: pid $PID does not look like a vLLM server:"
  echo "       ${CMDLINE:0:160}"
  exit 1
fi
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
ENGINE=$(engine_id_of_cmdline "$CMDLINE")

# ------------------------------------------------------------------- members
# ps right-aligns its columns, so trim before taking the pid.
mapfile -t MEMBERS < <(ps -eo pid=,pgid=,cmd= | awk -v g="$PGID" '$2 == g {print}')
member_pids() {
  local line
  for line in "${MEMBERS[@]}"; do
    awk '{print $1}' <<<"$line"
  done
  return 0   # an empty list is the success case, not an error under set -e
}

my_pgid=$(ps -o pgid= -p $$ | tr -d ' ')
group_safe=1
[ "$PGID" = "$my_pgid" ] && group_safe=0   # would kill the calling shell too
for line in "${MEMBERS[@]}"; do
  # Everything in the group must belong to this instance. The multiprocessing
  # helpers are vLLM's own children, so they count as belonging.
  case "$line" in
    *"$MARKERS"*|*VLLM::*) ;;
    *multiprocessing.resource_tracker*|*multiprocessing.spawn*|\
    *multiprocessing.forkserver*) ;;
    *)
      group_safe=0
      echo "[stop] unrelated process in the group: $(tr -s ' ' <<<"$line" | cut -c1-120)"
      ;;
  esac
done

echo "[stop] port $PORT  pid $PID  pgid $PGID  engine ${ENGINE:--}"
echo "[stop] processes ($((${#MEMBERS[@]}))):"
for line in "${MEMBERS[@]}"; do
  printf '         %s\n' "$(tr -s ' ' <<<"$line" | cut -c1-150)"
done
if [ "$group_safe" = 1 ]; then
  echo "[stop] plan: SIGTERM $PID, then the group -$PGID if needed, then SIGKILL"
else
  echo "[stop] plan: SIGTERM $PID, then its descendants, then SIGKILL"
fi
if [ "$CLEAN_SHM" = 1 ]; then
  echo "[stop] plan: remove /dev/shm/vllm_offload_${ENGINE:-<no engine_id>}.mmap"
          # (after the processes are gone; skipped if another process maps it)
fi

if [ "$DRY" = 1 ]; then
  echo "[stop] --dry-run: nothing was signalled"
  exit 0
fi

# ---------------------------------------------------------------------- signal
descendants() {
  ps -eo pid=,ppid= | awk -v r="$1" '
    { kids[$2] = kids[$2] " " $1 }
    END {
      stack[++top] = r
      while (top > 0) {
        cur = stack[top--]
        n = split(kids[cur], arr, " ")
        for (i = 1; i <= n; i++) if (arr[i] != "") { print arr[i]; stack[++top] = arr[i] }
      }
    }'
}

signal_group() {  # $1 = signal, to the whole process group
  kill "-$1" "-$PGID" 2>/dev/null || true
}

signal_leader() {
  kill "-$1" "$PID" 2>/dev/null || true
}

signal_tree() {   # leaves first, then the leader
  local pid
  while read -r pid; do
    [ -n "$pid" ] && kill "-$1" "$pid" 2>/dev/null || true
  done < <(descendants "$PID" | tac)
  signal_leader "$1"
}

alive_members() {
  local pid
  for pid in $(member_pids); do
    [ -d "/proc/$pid" ] && printf '%s ' "$pid"
  done
  return 0   # same: no survivors must not trip `set -e`
}

wait_gone() {  # $1 = seconds; returns 0 once no member is alive
  local i left
  for i in $(seq "$1"); do
    left=$(alive_members)
    [ -z "$left" ] && return 0
    sleep 1
  done
  return 1
}

if [ "$group_safe" = 1 ]; then
  polite() { signal_group "$1"; }
else
  polite() { signal_tree "$1"; }
fi

if [ "$FORCE" = 1 ]; then
  echo "[stop] --force: SIGKILL"
  polite KILL
else
  # The api_server shuts its workers down itself when it gets SIGTERM alone, so
  # try that first; escalate to the whole group/tree only if it does not.
  echo "[stop] SIGTERM $PID (graceful)"
  signal_leader TERM
  if ! wait_gone 10; then
    left=$(alive_members)
    echo "[stop] still alive after 10s (${left:-none}); SIGTERM the rest"
    polite TERM
    if ! wait_gone $((TIMEOUT > 5 ? TIMEOUT - 10 : 5)); then
      left=$(alive_members)
      echo "[stop] still alive (${left:-none}); SIGKILL"
      polite KILL
      wait_gone 5 || true
    fi
  fi
fi

# ----------------------------------------------------------------- check result
left=$(alive_members)
if [ -n "$left" ]; then
  echo "[stop] LEFT BEHIND: $left"
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then
  echo "[stop] process tree gone but port $PORT is still listening"
  exit 1
fi
echo "[stop] stopped (port $PORT free)"

shm_report() {  # "used / total (free)" in GiB, straight from df
  # mawk has no `**` operator, so divide by the literal.
  df -B1 /dev/shm 2>/dev/null | awk 'NR == 2 {
    printf "%.1f / %.1f GiB (%.1f free)", $3 / 1073741824, $2 / 1073741824, \
           $4 / 1073741824
  }'
}

echo "[stop] /dev/shm after stop: $(shm_report)"
if [ "$CLEAN_SHM" = 1 ]; then
  if [ -z "$ENGINE" ]; then
    echo "[stop] cmdline carries no engine_id; /dev/shm left alone"
  else
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/shm_staging.sh"
    vllm_clean_shm_staging "$ENGINE"
    if [ -e "/dev/shm/vllm_offload_${ENGINE}.mmap" ]; then
      echo "[stop] staging file still present (mapped by another process?)"
    else
      echo "[stop] staging file gone; /dev/shm now: $(shm_report)"
    fi
  fi
else
  echo "[stop] --no-clean-shm: staging file (if any) left in place"
fi
exit 0
