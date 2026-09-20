#!/usr/bin/env bash
# Qwen3.8-27B FP8 (e4m3) weights + fp8_e4m3 KV, 256K context
# (262,144 = the model's own max_position_embeddings, so no YaRN is involved).
#
# This is the FP8-weight counterpart of run_vllm_qwen38_awq_fp8e4m3_256k.sh and
# is meant as the higher-precision rung: the AWQ-INT4 checkpoint is ~4-bit,
# this one keeps the released FP8 weights. On SM75 there is no FP8 tensor core,
# so the FP8 GEMMs are dequantised to FP16 at runtime -- expect this profile to
# be slower than the AWQ one; its value is fidelity, not throughput.
#
# No --head8bit option here: the FP8 checkpoint was never run through
# scripts/tools/quantize_lm_head.py, so its lm_head stays bf16. The int8-head
# variants (and the switch) exist only for the AWQ checkpoints.
#
# Memory: measured on this host, FP8 weights leave ~6.4 GiB/GPU free after
# loading, and fp8_e4m3 KV costs ~16.8 KB/token/GPU without MTP. A 262,144
# request therefore needs >= 4,831,346,688 (kv_pool_sizing.py); 4.9e9 is used
# here. MTP is OFF by default: the MTP6 geometry needs >= 5,341,052,928 and was
# measured to OOM on this host with FP8 weights (269,228-token pool then a
# failed allocation). Enable with SPEC_NUM_TOKENS=6 only after re-checking the
# pool against the actual free memory.
#
# Paths come from .env; the FP8 checkpoint is derived from MODEL_PATH's
# directory (override with FP8_MODEL_PATH).
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

# --- model -----------------------------------------------------------------
# .env's MODEL_PATH points at the AWQ yarn checkpoint; this profile wants the
# released FP8 one. Derive it from the same directory unless overridden.
FP8_MODEL_PATH="${FP8_MODEL_PATH:-$(dirname "$MODEL_PATH")/Qwen3.8-27B-FP8}"
if [ ! -f "$FP8_MODEL_PATH/config.json" ]; then
  echo "$0: FP8 checkpoint not found at $FP8_MODEL_PATH" >&2
  exit 2
fi

# --- profile knobs ---------------------------------------------------------
# Calibrate to the "GPU KV cache size" line after a launch;
# scripts/tools/kv_pool_sizing.py reports the pool a profile needs.
: "${MAX_MODEL_LEN:=262144}"
: "${KV_CACHE_MEMORY_BYTES:=4900000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${SPEC_NUM_TOKENS:=0}"

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
# 262,144 is the model's own limit, so this is not strictly needed here.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
# API auth: inherit VLLM_API_KEY from the environment (empty = no auth).
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

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
  --disable-uvicorn-access-log
)
if [ "${SPEC_NUM_TOKENS:-0}" -gt 0 ]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

echo "[launch] fp8 weights + fp8_e4m3 KV, model=$FP8_MODEL_PATH max_len=$MAX_MODEL_LEN pool=$KV_CACHE_MEMORY_BYTES n=$SPEC_NUM_TOKENS" >&2
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
