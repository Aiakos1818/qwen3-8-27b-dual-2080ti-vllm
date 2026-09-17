#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp8_e4m3 KV, 16K context, small pool, tiered KV
# offload (CPU staging primary + disk/fs secondary) — the quick offload bench.
#
# The pool holds ~11K tokens for a 16K context, so a handful of requests force
# eviction and a restore in seconds rather than minutes. Use it to iterate on
# the offload path; use the 128K profile for realistic sizes.
#
# Promoted chunks are retained in the CPU tier, so CPU_BYTES_TO_USE must hold a
# whole chain: ceil(MAX_MODEL_LEN / 1600) chunks at ~55.8 MB each (measured,
# one chunk per block). 16K -> 11 chunks -> 0.61e9; the 2.4e9 default leaves
# room for several sessions.
#
# Hosts whose RLIMIT_MEMLOCK is below CPU_BYTES_TO_USE fail cudaHostRegister
# for the staging region. That is tolerated (unpinned DMA), but only with the
# sticky-error fix in vllm/v1/kv_offload/cpu/gpu_worker.py (commit bf78fc276)
# — without it the JIT warmup dies with "CUDA error: invalid argument".
#
# Paths / model come from .env (copy config/vllm.env.example).
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

# --- profile knobs ---------------------------------------------------------
: "${MAX_MODEL_LEN:=16384}"
: "${KV_CACHE_MEMORY_BYTES:=900000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${CPU_BYTES_TO_USE:=2400000000}"
: "${SPEC_NUM_TOKENS:=3}"
# Disk tier. Upstream has no quota knob: the tier grows with the working set,
# so watch the directory size during long tests.
: "${VLLM_SSD_ROOT:=${REPO_ROOT}/ssd_kv}"
: "${VLLM_SSD_CLEAN_START:=1}"
# Stable engine id: names the /dev/shm staging file and the SSD session dir.
# Give every profile its own id; two instances must not share one.
: "${KV_ENGINE_ID:=qwen38-27b-16k-ssd}"

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

KV_XFER="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$KV_ENGINE_ID\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE,\"eviction_policy\":\"lru\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$VLLM_SSD_ROOT\"}]}}"

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

# A fixed engine id re-opens the previous run's staging file (it is never
# unlinked) and would reuse the previous run's SSD session dir; start clean.
mkdir -p "$VLLM_SSD_ROOT"
if [ "$VLLM_SSD_CLEAN_START" = "1" ]; then
  rm -rf "${VLLM_SSD_ROOT:?}"/*
fi
rm -f "/dev/shm/vllm_offload_${KV_ENGINE_ID}.mmap"

# A chain only restores if it fits in the CPU tier while it is promoted.
CHAIN_CHUNKS=$(( (MAX_MODEL_LEN + 1599) / 1600 ))
TIER_CHUNKS=$(( CPU_BYTES_TO_USE / 55800000 ))
if [ "$TIER_CHUNKS" -lt "$CHAIN_CHUNKS" ]; then
  echo "[warn] CPU staging holds ~$TIER_CHUNKS chunks but a ${MAX_MODEL_LEN}-token" >&2
  echo "       chain needs $CHAIN_CHUNKS: restores past ~$((TIER_CHUNKS * 1600)) tokens" >&2
  echo "       will yield 0 hits (raise CPU_BYTES_TO_USE)." >&2
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
