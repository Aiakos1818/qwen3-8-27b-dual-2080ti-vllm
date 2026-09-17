#!/usr/bin/env bash
# /dev/shm staging-file hygiene for the offload profiles.
#
# The CPU KV offload region is an mmap of /dev/shm/vllm_offload_<engine_id>.mmap,
# which is RAM in full. A cleanly exited engine unlinks its own file, but a killed
# one leaves it behind, and a profile whose engine_id changed cannot reclaim the
# old name either. So every offload profile drops *its own* file before starting.
#
# Nothing else is ever touched: another engine's staging file is its business, and
# if it makes /dev/shm too small then vllm_check_offload_sizing() refuses the
# launch and names the holder rather than deleting someone else's state. A live
# file (one a process still maps) is left alone for the same reason: the preflight
# then reports "an instance of this profile is already running".

vllm_clean_shm_staging() {
  local engine_id="${1:?engine_id required}"
  local our_file="/dev/shm/vllm_offload_${engine_id}.mmap"
  [ -e "$our_file" ] || return 0
  if grep -qsF -- "$our_file" /proc/[0-9]*/maps; then
    echo "[info] /dev/shm: $our_file is mapped by a running process; leaving it" >&2
    return 0
  fi
  echo "[info] /dev/shm: removing stale staging file $our_file" >&2
  rm -f "$our_file"
}
