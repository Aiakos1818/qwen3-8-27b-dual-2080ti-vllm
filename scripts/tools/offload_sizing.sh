#!/usr/bin/env bash
# Sizing preflight shared by the offload profiles.
#
# The CPU tier is a pre-faulted mmap of /dev/shm, i.e. RAM in full, so before
# starting an engine the profile has to know the host can back it. vLLM checks
# the free space of /dev/shm itself (check_shm_free_space in shm_broadcast.py)
# and dies with a RuntimeError if it cannot, but by then the process is already
# half up and the message is easy to miss in the log; this runs ahead of the
# launch and distinguishes the two very different causes:
#
#   * the /dev/shm mount is simply too small  -> remount it (needs root)
#   * the mount is big enough but busy        -> something else holds it: a stale
#     staging file, or another engine still running. Neither is deleted here;
#     the error names the holders so the operator can decide.
#
# Errors refuse the launch (ALLOW_UNSAFE_LAUNCH=1 overrides). Warnings never do.
# Call vllm_clean_shm_staging() first: it drops the profile's own stale staging
# file (nothing else), so the free-space verdict matches what a launch would see.
#
# Inputs (exported by the profile):
#   OFFLOAD_LABEL                 label for the staging line, e.g. "RAM tier"
#   OFFLOAD_STAGING_BYTES         CPU tier / promotion staging size
#   OFFLOAD_CHAIN_BYTES           bytes of one full-length context on disk
#   OFFLOAD_CHAIN_CHUNKS          chunks per full-length context
#   OFFLOAD_RAM_CHAINS            full contexts the CPU tier targets (>= 1)
#   OFFLOAD_SSD_BYTES             0 = no disk tier
#   OFFLOAD_SSD_CHAINS            full contexts the disk ring targets (0 = n/a)
#   OFFLOAD_POOL_BYTES            --kv-cache-memory-bytes
#   OFFLOAD_POOL_BYTES_PER_TOKEN  measured, for a rough pool estimate
# Optional:
#   OFFLOAD_MIN_FREE_RAM_BYTES    physical headroom required (default 4 GiB)
#   CHECK_ONLY / ALLOW_UNSAFE_LAUNCH / MAX_MODEL_LEN / KV_ENGINE_ID
#
# Returns 0 when the launch may proceed, 1 when it must not.

_offload_gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

_offload_shm_holders() {
  find /dev/shm -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null |
    sort -rn | head -5 |
    while read -r size path; do
      printf '           %8s  %s\n' "$(_offload_gib "$size") GiB" "$path"
    done
}

# A running engine's staging file is unlinked once all workers have mapped it,
# so it no longer appears in /dev/shm but its pages still occupy the tmpfs. Sum
# such mappings out of /proc/*/maps so the error can name the actual holder.
_offload_shm_mapped_staging() {
  command -v python3 >/dev/null 2>&1 || return 0
  python3 - <<'PY' 2>/dev/null
import glob
import os

regions = {}
for path in glob.glob("/proc/[0-9]*/maps"):
    try:
        lines = open(path, errors="ignore").read().splitlines()
    except OSError:
        continue
    for line in lines:
        if "/dev/shm/vllm_offload_" not in line:
            continue
        parts = line.split()
        try:
            start, end = (int(x, 16) for x in parts[0].split("-"))
        except (ValueError, IndexError):
            continue
        name = parts[5].removesuffix(" (deleted)")
        size, mappers = regions.get(name, (0, 0))
        # Each process maps the whole region, so report one size plus the count
        # of processes holding it rather than summing across them.
        regions[name] = (max(size, end - start), mappers + 1)
for name, (size, mappers) in sorted(
    regions.items(), key=lambda item: item[1][0], reverse=True
)[:5]:
    print(size, mappers, name)
PY
}

vllm_check_offload_sizing() {
  local label="${OFFLOAD_LABEL:?OFFLOAD_LABEL is required}"
  local staging="${OFFLOAD_STAGING_BYTES:?OFFLOAD_STAGING_BYTES is required}"
  local chain="${OFFLOAD_CHAIN_BYTES:?OFFLOAD_CHAIN_BYTES is required}"
  local chain_chunks="${OFFLOAD_CHAIN_CHUNKS:?OFFLOAD_CHAIN_CHUNKS is required}"
  local ram_chains="${OFFLOAD_RAM_CHAINS:?OFFLOAD_RAM_CHAINS is required}"
  local ssd="${OFFLOAD_SSD_BYTES:-0}"
  local ssd_chains="${OFFLOAD_SSD_CHAINS:-0}"
  local pool="${OFFLOAD_POOL_BYTES:?OFFLOAD_POOL_BYTES is required}"
  local per_token="${OFFLOAD_POOL_BYTES_PER_TOKEN:?OFFLOAD_POOL_BYTES_PER_TOKEN is required}"
  local reserve="${OFFLOAD_MIN_FREE_RAM_BYTES:-4294967296}"

  local tier_chunks=$(( staging / 55800000 ))
  local tier_chains=$(( staging / chain ))
  local ssd_ring=0
  [ "$ssd" -gt 0 ] && ssd_ring=$(( ssd / chain ))
  local pool_tokens=$(( pool / per_token ))
  local problems=0

  echo "[profile] context=${MAX_MODEL_LEN:-?}  chain=$chain_chunks chunks ($(_offload_gib "$chain") GiB)"
  echo "[profile] $label=$tier_chunks chunks ($(_offload_gib "$staging") GiB) = ~$tier_chains full context(s)"
  [ "$ssd" -gt 0 ] && echo "[profile] disk ring ~$ssd_ring contexts ($(_offload_gib "$ssd") GiB)"
  echo "[profile] GPU pool ~$pool_tokens tokens (rough; the boot log's \"GPU KV cache size\" is authoritative)"

  if [ "$tier_chunks" -lt "$(( ram_chains * chain_chunks ))" ]; then
    echo "[warn] $label holds $tier_chunks chunks but $ram_chains full context(s)" >&2
    echo "       need $(( ram_chains * chain_chunks )): fewer long sessions stay" >&2
    echo "       restorable than intended" >&2
  fi
  if [ "$ssd" -gt 0 ] && [ "$ssd_ring" -lt "$ssd_chains" ]; then
    echo "[warn] disk budget holds ~$ssd_ring full context(s); the profile targets $ssd_chains" >&2
    echo "       (raise VLLM_SSD_MAX_BYTES)" >&2
  fi
  if [ "$pool_tokens" -lt "${MAX_MODEL_LEN:-0}" ]; then
    echo "[warn] GPU pool ~$pool_tokens tokens < MAX_MODEL_LEN ${MAX_MODEL_LEN:-?}: a" >&2
    echo "       full-length request may not fit (raise KV_CACHE_MEMORY_BYTES)" >&2
  fi

  # /dev/shm: the region is pre-faulted in full, so the mount must be both large
  # enough and free enough. Free space is what the engine itself checks.
  local shm_total shm_free
  shm_total=$(df -B1 --output=size /dev/shm 2>/dev/null | tail -1)
  shm_free=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1)
  if [ -z "${shm_total:-}" ] || [ "$shm_total" -lt "$staging" ]; then
    echo "[error] /dev/shm is $(_offload_gib "${shm_total:-0}") GiB but the $label needs" >&2
    echo "        $(_offload_gib "$staging") GiB. With root:" >&2
    echo "        mount -o remount,size=$(( staging / 1048576 + 2048 ))M /dev/shm" >&2
    problems=$((problems + 1))
  fi
  if [ -z "${shm_free:-}" ] || [ "$shm_free" -lt "$staging" ]; then
    echo "[error] /dev/shm has only $(_offload_gib "${shm_free:-0}") GiB free of" >&2
    echo "        $(_offload_gib "${shm_total:-0}") GiB, but the $label needs $(_offload_gib "$staging") GiB" >&2
    echo "        (vLLM would fail with \"Insufficient space in /dev/shm\")." >&2
    if [ -n "${KV_ENGINE_ID:-}" ] && grep -qsF "/dev/shm/vllm_offload_${KV_ENGINE_ID}.mmap" /proc/[0-9]*/maps; then
      echo "        An instance of this profile is already running: its staging file" >&2
      echo "        holds the space. Stop it (or use a different PORT/engine id)." >&2
    else
      echo "        Free the space and retry. Only this profile's own stale staging" >&2
      echo "        file is removed automatically; anything else is left alone, so a" >&2
      echo "        stale /dev/shm/vllm_offload_*.mmap must be removed by hand." >&2
    fi
    echo "        current /dev/shm holders:" >&2
    _offload_shm_holders >&2
    local mapped
    mapped=$(_offload_shm_mapped_staging)
    if [ -n "$mapped" ]; then
      echo "        mapped staging regions (unlinked, so they do not show above):" >&2
      while read -r size mappers path; do
        printf '           %8s  %s (%s process(es))\n' \
          "$(_offload_gib "$size") GiB" "$path" "$mappers" >&2
      done <<<"$mapped"
    fi
    problems=$((problems + 1))
  fi

  # The region is pre-faulted before the engine starts, and it stays un-pinned
  # when RLIMIT_MEMLOCK is small, so physical headroom has to be real.
  local avail_kb need_kb
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  need_kb=$(( staging / 1024 + reserve / 1024 ))
  if [ "${avail_kb:-0}" -lt "$need_kb" ]; then
    echo "[error] MemAvailable $(_offload_gib "$(( ${avail_kb:-0} * 1024 ))") GiB < $label" >&2
    echo "        + $(_offload_gib "$reserve") GiB headroom ($(_offload_gib "$(( need_kb * 1024 ))") GiB):" >&2
    echo "        the region is pre-faulted, so startup would fail or OOM-kill" >&2
    problems=$((problems + 1))
  fi
  local cgroup_limit cgroup_usage cgroup_free
  cgroup_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)
  cgroup_usage=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true)
  if [[ "$cgroup_limit" =~ ^[0-9]+$ ]] && [[ "$cgroup_usage" =~ ^[0-9]+$ ]]; then
    cgroup_free=$(( cgroup_limit - cgroup_usage ))
    if [ "$cgroup_free" -lt $(( staging + 1073741824 )) ]; then
      echo "[warn] cgroup headroom $(_offload_gib "$cgroup_free") GiB < $label + 1 GiB" >&2
      echo "       (vLLM logs this too and still tries: usage may be reclaimable)" >&2
    fi
  fi
  local memlock_kb
  memlock_kb=$(ulimit -l)
  if [ "$memlock_kb" != "unlimited" ] && [ "$memlock_kb" -lt $(( staging / 1024 )) ]; then
    echo "[warn] RLIMIT_MEMLOCK=$(_offload_gib "$(( memlock_kb * 1024 ))") GiB < $label: host" >&2
    echo "       registration fails and DMA stays unpinned (needs commit bf78fc276);" >&2
    echo "       keep RAM headroom so the region is never swapped out" >&2
  fi

  if [ "$problems" -gt 0 ] && [ "${ALLOW_UNSAFE_LAUNCH:-0}" != "1" ]; then
    echo "[error] refusing to launch ($problems sizing check(s) failed);" >&2
    echo "        set ALLOW_UNSAFE_LAUNCH=1 to start anyway" >&2
    return 1
  fi
  return 0
}
