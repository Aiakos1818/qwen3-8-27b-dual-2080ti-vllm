#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp8_e4m3 KV, 500.8K context, RAM-only KV offload
# sized for TWO full-length contexts (CPU tier as the store, no secondary tier).
#
# The rung between "no offload" and "RAM + SSD": the CPU tier alone keeps whole
# chains, so an evicted session is restored from RAM without touching the disk
# (faster than the SSD rung: PCIe-bound instead of disk-bound). Its size is the
# whole point: 2 x ceil(500800 / 1600) chunks = 626 chunks at ~55.8 MB each =
# 34.9 GB, i.e. two full-length contexts can be resident at once and either can
# be restored. Everything else is still recomputed, and a third session evicts
# the least recently restored one (RAM-only keeps no second copy anywhere).
#
# That tier is a hard /dev/shm reservation, pre-faulted at startup, and the
# default tmpfs is 50% of RAM: a 64 GB host defaults to 32 GiB, which is 0.6 GB
# short of two chains — remount /dev/shm to ~36 GiB (the check below prints the
# exact command). RAM headroom still matters: 34.9 GB of staging + the engine
# fits a 64 GB host, but not a smaller one.
#
# Validated on a small-scale run (40K prompts, 2 x 26-chunk chains, 61-chunk
# tier): A -> B -> A restored 36,800/39,170 (94%) in 3 s with 1.44 GB of
# CPU_to_GPU traffic and zero disk traffic.
#
# This profile targets the upstream-based branch (sm75-upstream) and uses only
# upstream knobs: prefix caching plus upstream's tiered offload with an empty
# secondary tier list.
#
# Hosts whose RLIMIT_MEMLOCK is below CPU_BYTES_TO_USE fail cudaHostRegister for
# the staging region. That is tolerated (unpinned DMA), but only with the
# sticky-error fix in vllm/v1/kv_offload/cpu/gpu_worker.py (commit bf78fc276) —
# without it the JIT warmup dies with "CUDA error: invalid argument". Keep real
# RAM headroom: an unpinned staging that gets swapped out destroys the restore
# path.
#
# Paths / model come from .env (copy config/vllm-500k-RAMx2.env.example).
# CHECK_ONLY=1 runs the sizing checks and exits without touching anything.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi

: "${MODEL_PATH:?set MODEL_PATH in .env}"
: "${VLLM_PYTHON:?set VLLM_PYTHON in .env}"
: "${FLASHQLA_PATH:?set FLASHQLA_PATH in .env}"
: "${SERVED_MODEL_NAME:=qwen38-27b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${OMP_NUM_THREADS:=8}"

# --- geometry (measured, one chunk per block) ------------------------------
# 1600 tokens and ~55.8 MB per chunk (both ranks' shards); the engine derives the
# real kv_bytes_per_chunk from the canonical layout, which lands slightly under
# 55.8 MB, so chunk counts round up in our favour.
CHUNK_TOKENS=1600
CHUNK_BYTES=55800000

# --- profile knobs ---------------------------------------------------------
# Calibrate to the "GPU KV cache size" line after a launch;
# scripts/tools/kv_pool_sizing.py reports the pool a profile needs.
# 9.6e9 pool -> 525,229 tokens (measured / logs), i.e. ~1.05x of one 500.8K
# request at ~17.9 KiB/token.
: "${MAX_MODEL_LEN:=500800}"
: "${KV_CACHE_MEMORY_BYTES:=9600000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
# CPU tier: both the store and the promotion staging. Sized for two full-length
# chains (2 x CHAIN_CHUNKS x CHUNK_BYTES = 34.9 GB), so two long sessions can be
# resident and either one restored. Costs /dev/shm and RAM in full.
CHAIN_CHUNKS=$(( (MAX_MODEL_LEN + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
CHAIN_BYTES=$(( CHAIN_CHUNKS * CHUNK_BYTES ))
: "${CPU_BYTES_TO_USE:=$(( 2 * CHAIN_BYTES ))}"
: "${SPEC_NUM_TOKENS:=3}"
# A KV load failure cannot really happen without a disk tier; recompute keeps
# the two offload rungs identical and is the safer default anyway.
: "${KV_LOAD_FAILURE_POLICY:=recompute}"
# Stable engine id: names the /dev/shm staging file. Give every profile its own
# id; two instances must not share one.
: "${KV_ENGINE_ID:=qwen38-27b-500k-RAMx2}"

# --- sizing self-checks ----------------------------------------------------
# Measured bytes per pooled token for this model/layout (fp8_e4m3, TP2, MTP=3).
POOL_BYTES_PER_TOKEN=18278
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

TIER_CHUNKS=$(( CPU_BYTES_TO_USE / CHUNK_BYTES ))
TIER_CHAINS=$(( CPU_BYTES_TO_USE / CHAIN_BYTES ))
POOL_TOKENS=$(( KV_CACHE_MEMORY_BYTES / POOL_BYTES_PER_TOKEN ))

echo "[profile] context=$MAX_MODEL_LEN  chain=$CHAIN_CHUNKS chunks ($(gib "$CHAIN_BYTES") GiB)"
echo "[profile] RAM tier=$TIER_CHUNKS chunks ($(gib "$CPU_BYTES_TO_USE") GiB) = ~$TIER_CHAINS full context(s)"
echo "[profile] GPU pool ~$POOL_TOKENS tokens"

problems=0
if [ "$TIER_CHUNKS" -lt "$(( 2 * CHAIN_CHUNKS ))" ]; then
  echo "[warn] RAM tier holds $TIER_CHUNKS chunks but two full contexts need" >&2
  echo "       $(( 2 * CHAIN_CHUNKS )): fewer long sessions stay restorable than intended" >&2
fi
if [ "$POOL_TOKENS" -lt "$MAX_MODEL_LEN" ]; then
  echo "[warn] GPU pool ~$POOL_TOKENS tokens < MAX_MODEL_LEN: a full-length request" >&2
  echo "       may not fit (raise KV_CACHE_MEMORY_BYTES)" >&2
fi

# The tier is a pre-faulted mmap of /dev/shm: a tmpfs that is too small fails
# mid-startup, so refuse early with an actionable message.
SHM_BYTES=$(df -B1 --output=size /dev/shm 2>/dev/null | tail -1)
if [ -z "${SHM_BYTES:-}" ] || [ "$SHM_BYTES" -lt "$CPU_BYTES_TO_USE" ]; then
  echo "[error] /dev/shm is $(gib "${SHM_BYTES:-0}") GiB but the RAM tier needs" >&2
  echo "        $(gib "$CPU_BYTES_TO_USE") GiB. With root:" >&2
  echo "        mount -o remount,size=$(( CPU_BYTES_TO_USE / 1048576 + 2048 ))M /dev/shm" >&2
  problems=$((problems + 1))
fi
AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
# Tier + 4 GiB for the engine, the page cache and the OS.
NEED_KB=$(( CPU_BYTES_TO_USE / 1024 + 4 * 1024 * 1024 ))
if [ "${AVAIL_KB:-0}" -lt "$NEED_KB" ]; then
  echo "[error] MemAvailable $(gib "$(( ${AVAIL_KB:-0} * 1024 ))") GiB < RAM tier + 4 GiB" >&2
  echo "        ($(gib "$(( NEED_KB * 1024 ))") GiB): the tier is pre-faulted, so" >&2
  echo "        startup would fail or OOM-kill" >&2
  problems=$((problems + 1))
fi
MEMLOCK_KB=$(ulimit -l)
if [ "$MEMLOCK_KB" != "unlimited" ] && [ "$MEMLOCK_KB" -lt $(( CPU_BYTES_TO_USE / 1024 )) ]; then
  echo "[warn] RLIMIT_MEMLOCK=$(gib "$(( MEMLOCK_KB * 1024 ))") GiB < RAM tier: host" >&2
  echo "       registration fails and DMA stays unpinned (needs commit bf78fc276);" >&2
  echo "       keep RAM headroom so the tier is never swapped out" >&2
fi

if [ "$problems" -gt 0 ] && [ "${ALLOW_UNSAFE_LAUNCH:-0}" != "1" ]; then
  echo "[error] refusing to launch ($problems sizing check(s) failed);" >&2
  echo "        set ALLOW_UNSAFE_LAUNCH=1 to start anyway" >&2
  exit 1
fi
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "[profile] CHECK_ONLY=1: sizing checks done, not launching"
  exit 0
fi

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# API auth: inherit VLLM_API_KEY from the environment (empty = no auth).
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

KV_XFER="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$KV_ENGINE_ID\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE,\"eviction_policy\":\"lru\",\"secondary_tiers\":[]}}"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype half --tensor-parallel-size 2 --device-ids 0,1
  --kv-cache-dtype fp8_e4m3
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --enable-prefix-caching --max-num-seqs 1
  --enable-prompt-tokens-details
  --max-num-batched-tokens 1024 --enable-chunked-prefill
  --skip-mm-profiling
  --limit-mm-per-prompt '{"image":20,"video":1}'
  --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}'
  --reasoning-parser qwen3
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}"
  --kv-transfer-config "$KV_XFER"
  --disable-uvicorn-access-log
)
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

# The staging file is re-opened by a fixed engine id (it is never unlinked), so
# drop any stale one before the engine picks it up.
rm -f "/dev/shm/vllm_offload_${KV_ENGINE_ID}.mmap"

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
