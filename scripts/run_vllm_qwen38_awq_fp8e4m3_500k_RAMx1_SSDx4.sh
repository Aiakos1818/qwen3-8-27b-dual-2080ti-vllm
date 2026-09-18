#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp8_e4m3 KV, 500.8K context, tiered KV offload:
# RAM holds ONE full-length context (promotion staging), the disk holds FOUR.
#
# Why a 64 GB host: a promoted chain is retained in the CPU tier and stays pinned
# until the request consumes it, so the staging must hold one whole chain —
# ceil(500800 / 1600) = 313 chunks at ~55.8 MB each = 17.5 GB, +10% headroom for
# the tier's own LRU = 19.2 GB. That is a hard /dev/shm reservation, pre-faulted
# at startup, and the default tmpfs is 50% of RAM: 32 GB would leave it 0.3 GiB
# short of a single chain, 64 GB leaves it comfortable (32 GiB). Below the chain
# size, restores of a full-length prompt silently yield 0 hits.
#
# Top rung of the three 500K profiles: 500k (GPU only, no offload) →
# 500k_RAMx2 (CPU tier as the store, two contexts) → this one (RAM staging +
# disk ring, the only rung that keeps several long sessions restorable).
#
# Disk tier: a 4-context ring. Past VLLM_SSD_MAX_BYTES the least recently
# restored blocks are evicted (LRU by file mtime), so the directory can never
# fill the partition; the newest four full-length contexts stay restorable, and
# the RAM tier only has to hold the one being promoted.
#
# Validated at the mechanism level on a small run (40K prompts, 61-chunk RAM
# tier): A -> B -> A restored 94% in 3 s from the offload tier, and the disk
# ring/byte budget behaves as documented in docs/upstream-branch.md 5.5.
#
# This profile targets the upstream-based branch (2080ti_dual_qwen38-27B) and uses only
# upstream knobs: prefix caching plus upstream's tiered offload.
#
# Hosts whose RLIMIT_MEMLOCK is below CPU_BYTES_TO_USE fail cudaHostRegister for
# the staging region. That is tolerated (unpinned DMA), but only with the
# sticky-error fix in vllm/v1/kv_offload/cpu/gpu_worker.py (commit bf78fc276) —
# without it the JIT warmup dies with "CUDA error: invalid argument". Keep real
# RAM headroom: an unpinned staging that gets swapped out destroys the restore
# path.
#
# Paths / model come from .env (copy config/vllm-500k-RAMx1-SSDx4.env.example).
# CHECK_ONLY=1 runs the sizing checks and exits without touching anything.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=tools/shm_staging.sh
source "$SCRIPT_DIR/tools/shm_staging.sh"
# shellcheck source=tools/offload_sizing.sh
source "$SCRIPT_DIR/tools/offload_sizing.sh"
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
# Staging: exactly one full-length context (313 chunks). It only has to hold the
# chain being promoted, so no headroom is needed; every stored context also lives
# on disk. Costs /dev/shm and RAM in full — pre-faulted before the engine starts.
CHAIN_CHUNKS=$(( (MAX_MODEL_LEN + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
CHAIN_BYTES=$(( CHAIN_CHUNKS * CHUNK_BYTES ))
: "${CPU_BYTES_TO_USE:=$(( 1 * CHAIN_BYTES ))}"
: "${SPEC_NUM_TOKENS:=3}"
# Disk tier: 4 full-length contexts (4 x 17.5 GB = 69.9 GB).
: "${VLLM_SSD_ROOT:=/home/aiakos/Qwen3.8-27B-Deploy/ssd_kv}"
: "${VLLM_SSD_MAX_BYTES:=$(( 4 * CHAIN_BYTES ))}"
# Keep the directory across restarts (the budget reclaims what it needs, and the
# eviction order is read back from the files' mtimes). Set 1 to wipe it.
: "${VLLM_SSD_CLEAN_START:=0}"
# A KV load failure (missing/short file on disk) either recomputes the affected
# tokens or fails the request (vLLM's default). Recompute is the safer choice
# for offload: the disk tier is best-effort, and 0-hit degradation beats an
# aborted request.
: "${KV_LOAD_FAILURE_POLICY:=recompute}"
# Stable engine id: names the /dev/shm staging file and the SSD session dir.
# Give every profile its own id; two instances must not share one.
: "${KV_ENGINE_ID:=qwen38-27b-500k-RAMx1-SSDx4}"

# --- sizing preflight ------------------------------------------------------
# Shared with the other offload profiles (scripts/tools/offload_sizing.sh). It
# refuses to launch when the host cannot back the pre-faulted /dev/shm staging
# region — the mount's total size, its *free* space (a stale or another
# instance's staging file can eat it), and physical headroom — and warns when a
# capacity target is undersized. CHECK_ONLY=1 stops after the checks.
vllm_clean_shm_staging "$KV_ENGINE_ID"
OFFLOAD_LABEL="staging"
OFFLOAD_STAGING_BYTES=$CPU_BYTES_TO_USE
OFFLOAD_CHAIN_BYTES=$CHAIN_BYTES
OFFLOAD_CHAIN_CHUNKS=$CHAIN_CHUNKS
OFFLOAD_RAM_CHAINS=1
OFFLOAD_SSD_BYTES=$VLLM_SSD_MAX_BYTES
OFFLOAD_SSD_CHAINS=4
OFFLOAD_POOL_BYTES=$KV_CACHE_MEMORY_BYTES
OFFLOAD_POOL_BYTES_PER_TOKEN=18278
vllm_check_offload_sizing || exit 1
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
# Verify spec drafts on the native FlashInfer decode path so speculative
# decode keeps full cudagraphs (SM75 has no fused GDN decode / TRT-LLM).
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# API auth: inherit VLLM_API_KEY from the environment (empty = no auth).
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

KV_XFER="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$KV_ENGINE_ID\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE,\"eviction_policy\":\"lru\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$VLLM_SSD_ROOT\",\"max_bytes\":$VLLM_SSD_MAX_BYTES}]}}"

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

# A fixed engine id names the run's SSD session dir; leave its contents alone
# unless asked to start clean, so a restart can still restore blocks.
mkdir -p "$VLLM_SSD_ROOT"
if [ "$VLLM_SSD_CLEAN_START" = "1" ]; then
  rm -rf "${VLLM_SSD_ROOT:?}"/*
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
