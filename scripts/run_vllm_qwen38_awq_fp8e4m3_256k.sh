#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp8_e4m3 KV, 256K context, SSD-only host tier.
#
# Keep-alive pin + Mamba/GDN anchors + two-tier GPU/SSD session offload (chunked
# streaming). See docs/kv-optimization/vllm_01_保活.md, vllm_02_锚点.md,
# vllm_04_offload_ssd.md.
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
# --max-model-len 262144 (=256*1024), pool 5.6e9. Measured: 262144 needs a safe
# pool >= 5,431,296,000; 5.6e9 saves VRAM (safe ceiling ~272,000). Diagnose with
# scripts/kv_pool_sizing.py (docs/kv-optimization/GPU_MEMORY_CALCULATION.md §4.5).
: "${MAX_MODEL_LEN:=262144}"
: "${KV_CACHE_MEMORY_BYTES:=5600000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${CPU_BYTES_TO_USE:=2400000000}"
: "${SPEC_NUM_TOKENS:=3}"
: "${VLLM_PIN_MIN_TOKENS:=16000}"
: "${VLLM_MAMBA_CKPT_TOKENS:=32000}"
: "${VLLM_MAMBA_CKPT_ANCHORS:=3}"
: "${VLLM_HOSTTIER_EVICT_SMALL_TOKENS:=32000}"
: "${VLLM_SSD_ROOT:=/tmp/vllm_ssd}"
: "${VLLM_SSD_QUOTA_BYTES:=68719476736}"   # 64 GiB
: "${VLLM_SSD_MAX_MBPS:=800}"
: "${VLLM_SSD_CLEAN_START:=1}"

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# API auth: inherit VLLM_API_KEY from the environment (empty = no auth).
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export VLLM_PIN_MIN_TOKENS VLLM_MAMBA_CKPT_TOKENS VLLM_MAMBA_CKPT_ANCHORS
export VLLM_HOSTTIER_EVICT_SMALL_TOKENS
# Two-tier GPU/SSD session offload (chunked, no session-size limit).
export VLLM_SSD_ROOT VLLM_SSD_QUOTA_BYTES VLLM_SSD_MAX_MBPS VLLM_SSD_CLEAN_START
export VLLM_SSD_ONLY=1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

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
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU_BYTES_TO_USE}}"
  --disable-uvicorn-access-log
)
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
