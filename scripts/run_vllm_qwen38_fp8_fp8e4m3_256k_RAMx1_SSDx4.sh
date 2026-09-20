#!/usr/bin/env bash
# Qwen3.8-27B FP8 (e4m3) weights + fp8_e4m3 KV, 256K context (262,144), tiered
# KV offload: RAM holds ONE full-length context (promotion staging), the disk
# holds FOUR.
#
# The offload rung of the FP8 256K profile: a long prompt stays restorable across
# restarts instead of being recomputed. The GPU-only profile
# (run_vllm_qwen38_fp8_fp8e4m3_256k.sh) is the serving default; this one trades
# a bigger /dev/shm reservation for a warm cache.
#
# Why a bigger host than 15 GiB: a promoted chain is retained in the CPU tier
# until the request consumes it, so the staging must hold one whole chain --
# ceil(262144 / 1600) = 164 chunks at ~55.8 MB each = 9.15 GB. That is a hard
# /dev/shm reservation, pre-faulted at startup, so it needs ~10 GB of tmpfs and
# therefore roughly a 32 GB host (the default tmpfs is half of RAM). On the
# 15 GiB host this profile's preflight refuses to launch; the 128K offload
# profile is the one that fits there (4.58 GB). Below the chain size, restores
# of a full-length prompt silently yield 0 hits, which is why the staging is
# sized to a whole chain and not less.
#
# Memory: FP8 weights leave less GPU headroom than AWQ, so MTP is OFF and the
# GPU pool is 4.9e9 (>= 4,831,346,688 needed for 262,144 @ fp8_e4m3, no MTP).
# MTP6 would need >= 5,341,052,928 and was measured to OOM with FP8 weights.
#
# Disk tier: a 4-context ring. Past VLLM_SSD_MAX_BYTES the least recently
# restored blocks are evicted (LRU by file mtime), so the directory can never
# fill the partition; the newest four full-length contexts stay restorable, and
# the RAM tier only has to hold the one being promoted.
#
# Hosts whose RLIMIT_MEMLOCK is below CPU_BYTES_TO_USE fail cudaHostRegister for
# the staging region. That is tolerated (unpinned DMA), but only with the
# sticky-error fix in vllm/v1/kv_offload/cpu/gpu_worker.py (commit bf78fc276) --
# without it the JIT warmup dies with "CUDA error: invalid argument".
#
# Paths come from .env; the FP8 checkpoint is derived from MODEL_PATH's
# directory (override with FP8_MODEL_PATH).
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

# --- model -----------------------------------------------------------------
FP8_MODEL_PATH="${FP8_MODEL_PATH:-$(dirname "$MODEL_PATH")/Qwen3.8-27B-FP8}"
if [ ! -f "$FP8_MODEL_PATH/config.json" ]; then
  echo "$0: FP8 checkpoint not found at $FP8_MODEL_PATH" >&2
  exit 2
fi

# --- geometry (measured, one chunk per block) ------------------------------
CHUNK_TOKENS=1600
CHUNK_BYTES=55800000

# --- profile knobs ---------------------------------------------------------
: "${MAX_MODEL_LEN:=262144}"
: "${KV_CACHE_MEMORY_BYTES:=4900000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
# Staging: exactly one full-length context (164 chunks).
CHAIN_CHUNKS=$(( (MAX_MODEL_LEN + CHUNK_TOKENS - 1) / CHUNK_TOKENS ))
CHAIN_BYTES=$(( CHAIN_CHUNKS * CHUNK_BYTES ))
: "${CPU_BYTES_TO_USE:=$(( 1 * CHAIN_BYTES ))}"
# MTP off: FP8 weights do not leave room for the MTP6 geometry at 256K.
: "${SPEC_NUM_TOKENS:=0}"
# Disk tier: 4 full-length contexts (4 x 9.15 GB = 36.6 GB).
: "${VLLM_SSD_ROOT:=/home/aiakos/Qwen3.8-27B-Deploy/ssd_kv}"
: "${VLLM_SSD_MAX_BYTES:=$(( 4 * CHAIN_BYTES ))}"
: "${VLLM_SSD_CLEAN_START:=0}"
: "${KV_LOAD_FAILURE_POLICY:=recompute}"
: "${KV_ENGINE_ID:=qwen38-27b-fp8-256k-RAMx1-SSDx4}"

# --- sizing preflight ------------------------------------------------------
vllm_clean_shm_staging "$KV_ENGINE_ID"
OFFLOAD_LABEL="staging"
OFFLOAD_STAGING_BYTES=$CPU_BYTES_TO_USE
OFFLOAD_CHAIN_BYTES=$CHAIN_BYTES
OFFLOAD_CHAIN_CHUNKS=$CHAIN_CHUNKS
OFFLOAD_RAM_CHAINS=1
OFFLOAD_SSD_BYTES=$VLLM_SSD_MAX_BYTES
OFFLOAD_SSD_CHAINS=4
OFFLOAD_POOL_BYTES=$KV_CACHE_MEMORY_BYTES
# fp8_e4m3, no MTP: measured 4.9e9 / 286,249 tokens on this host.
OFFLOAD_POOL_BYTES_PER_TOKEN=17118
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
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

KV_XFER="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$KV_ENGINE_ID\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE,\"eviction_policy\":\"lru\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$VLLM_SSD_ROOT\",\"max_bytes\":$VLLM_SSD_MAX_BYTES}]}}"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$FP8_MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype half --tensor-parallel-size 2 --device-ids 0,1
  --quantization fp8
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
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>","default_thinking_token_budget":8000}'
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
  --kv-transfer-config "$KV_XFER"
  --disable-uvicorn-access-log
)
if [ "${SPEC_NUM_TOKENS:-0}" -gt 0 ]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

mkdir -p "$VLLM_SSD_ROOT"
if [ "$VLLM_SSD_CLEAN_START" = "1" ]; then
  rm -rf "${VLLM_SSD_ROOT:?}"/*
fi

echo "[launch] fp8 weights + fp8_e4m3 KV, 256K RAMx1 SSDx4, model=$FP8_MODEL_PATH max_len=$MAX_MODEL_LEN pool=$KV_CACHE_MEMORY_BYTES n=$SPEC_NUM_TOKENS" >&2
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
