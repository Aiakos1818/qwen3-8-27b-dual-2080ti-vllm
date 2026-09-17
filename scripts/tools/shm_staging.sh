#!/usr/bin/env bash
# /dev/shm staging-file hygiene for the offload profiles.
#
# The CPU KV offload region is an mmap of /dev/shm/vllm_offload_<engine_id>.mmap,
# which is RAM in full. A cleanly exited engine unlinks its own file, but a
# killed one leaves it behind (the profile README warns about exactly this), and
# a profile whose engine_id changed cannot reclaim the old name either. A stale
# file then starves the next launch ("Insufficient space in /dev/shm ..."), so
# every offload profile calls this before starting.
#
# Files another process still maps are never touched: unlinking them would not
# hurt the running engine (POSIX keeps the mapping), but a new worker of that
# engine could no longer join it.

vllm_clean_shm_staging() {
  local engine_id="${1:?engine_id required}"
  local our_file="/dev/shm/vllm_offload_${engine_id}.mmap"
  local f
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    if [ "$f" = "$our_file" ]; then
      rm -f "$f"
      continue
    fi
    if grep -qsF -- "$f" /proc/[0-9]*/maps; then
      echo "[info] /dev/shm: keeping $f (mapped by a running process)" >&2
    else
      echo "[info] /dev/shm: removing orphaned staging file $f" >&2
      rm -f "$f"
    fi
  done
}
