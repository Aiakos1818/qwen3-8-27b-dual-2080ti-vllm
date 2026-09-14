#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + fp8_e4m3 KV, 100K context pool.
#
# Small-scale testbed for the KV-optimization features: prefix-cache keep-alive
# (pin), Mamba/GDN durable anchors, and session park/resume to RAM (and,
# optionally, SSD). See docs/kv-optimization/vllm_01_保活.md, vllm_02_锚点.md,
# vllm_03_offload_ram.md.
#
# Paths / model come from .env (copy config/vllm.env.example). All KV knobs can
# be overridden in .env or the environment.
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

# --- KV optimization knobs -------------------------------------------------
: "${MAX_MODEL_LEN:=102400}"                 # 100*1024 = 102.4k
: "${KV_CACHE_MEMORY_BYTES:=2300000000}"     # calibrate to "GPU KV cache size"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${CPU_BYTES_TO_USE:=2400000000}"          # host staging (RAM parking / bounce); <=2e9 breaks cudaHostRegister
: "${SPEC_NUM_TOKENS:=3}"                    # MTP speculative tokens
: "${VLLM_PIN_MIN_TOKENS:=16000}"            # keep-alive pin threshold (0 = off)
: "${VLLM_MAMBA_CKPT_TOKENS:=32000}"         # anchor cadence
: "${VLLM_MAMBA_CKPT_ANCHORS:=3}"            # anchors kept per chain
: "${VLLM_HOSTTIER_EVICT_SMALL_TOKENS:=32000}"  # sessions < this park first
# Stable engine id (default is a random UUID per launch): names the host-tier
# SSD session dir and the /dev/shm staging file. A fixed id lets
# VLLM_SSD_CLEAN_START=1 clear the previous run's sessions. Keep it unique
# per profile; two instances must not share one.
: "${KV_ENGINE_ID:=qwen38-27b-100k}"

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PIN_MIN_TOKENS VLLM_MAMBA_CKPT_TOKENS VLLM_MAMBA_CKPT_ANCHORS
export VLLM_HOSTTIER_EVICT_SMALL_TOKENS
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

# Optional SSD tier: set VLLM_SSD_ROOT in .env to park sessions on disk instead
# of (or in addition to) RAM. Without it the CPU staging pool is used directly.
if [ -n "${VLLM_SSD_ROOT:-}" ]; then
  export VLLM_SSD_ROOT
  export VLLM_SSD_QUOTA_BYTES="${VLLM_SSD_QUOTA_BYTES:-8589934592}"   # 8 GiB
  export VLLM_SSD_MAX_MBPS="${VLLM_SSD_MAX_MBPS:-800}"
  export VLLM_SSD_CLEAN_START="${VLLM_SSD_CLEAN_START:-1}"
fi

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
  --max-num-batched-tokens 4096 --enable-chunked-prefill
  --no-async-scheduling
  --skip-mm-profiling
  --limit-mm-per-prompt '{"image":20,"video":1}'
  --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}'
  --reasoning-parser qwen3
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}"
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
  --cpu-offload-gb 0
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"$KV_ENGINE_ID\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE}}"
  --disable-uvicorn-access-log
)
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

# With a stable engine id the /dev/shm staging file outlives a crashed run and
# would be re-opened (never unlinked) by the next launch. Remove it so every
# launch creates a fresh one of the expected size.
rm -f "/dev/shm/vllm_offload_${KV_ENGINE_ID}.mmap"

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
