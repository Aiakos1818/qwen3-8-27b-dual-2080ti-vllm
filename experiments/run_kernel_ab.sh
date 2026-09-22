#!/usr/bin/env bash
# One-off launcher for the W4A16 trunk-kernel A/B (Marlin vs Humming) on the AWQ
# checkpoint. The leg is selected by LINEAR_BACKEND:
#   auto     -> trunk Marlin (default), head int8 Humming
#   humming  -> trunk + head Humming (--linear-backend humming)
# Everything else (shape / KV / MTP / head) is identical between legs, so the
# only difference is which kernel implements the compressed-tensors WNA16 trunk.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
# .env sets MODEL_PATH unconditionally; honour an explicit override.
MODEL_PATH_IN="${MODEL_PATH:-}"
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi
if [ -n "$MODEL_PATH_IN" ]; then
  MODEL_PATH="$MODEL_PATH_IN"
fi

: "${MODEL_PATH:?set MODEL_PATH in .env}"
: "${VLLM_PYTHON:?set VLLM_PYTHON in .env}"
: "${FLASHQLA_PATH:?set FLASHQLA_PATH in .env}"
: "${SERVED_MODEL_NAME:=qwen38-27b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${OMP_NUM_THREADS:=8}"

: "${MAX_MODEL_LEN:=131072}"
: "${KV_CACHE_MEMORY_BYTES:=3000000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${SPEC_NUM_TOKENS:=6}"
: "${LINEAR_BACKEND:=auto}"
: "${SPEC_METHOD:=mtp}"
: "${PROMPT_LOOKUP_MIN:=2}"
: "${PROMPT_LOOKUP_MAX:=4}"
: "${ENABLE_THINKING:=1}"

: "${HEAD8BIT:=1}"

EXTRA_LD_LIBRARY_PATH=""
if [ "$HEAD8BIT" = 1 ]; then
  case "$MODEL_PATH" in
    *-head8bit) ;;
    *) MODEL_PATH="$MODEL_PATH-head8bit" ;;
  esac
fi
EXTRA_LD_LIBRARY_PATH=$(
  ls -d "$(dirname "$(dirname "$VLLM_PYTHON")")"/lib/python*/site-packages/nvidia/cu13/lib 2>/dev/null | head -1
)
if [ -z "$EXTRA_LD_LIBRARY_PATH" ] || [ ! -d "$EXTRA_LD_LIBRARY_PATH" ]; then
  echo "$0: venv cu13 lib dir not found" >&2
  exit 2
fi

: "${RUNNER_V2:=1}"

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="$RUNNER_V2"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}"
export PYTHONPATH="$FLASHQLA_PATH"

if [ "$SPEC_METHOD" = "ngram_gpu" ]; then
  SPEC_CONFIG="{\"method\":\"ngram_gpu\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"prompt_lookup_min\":$PROMPT_LOOKUP_MIN,\"prompt_lookup_max\":$PROMPT_LOOKUP_MAX}"
else
  SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}"
fi
if [ "$ENABLE_THINKING" = 1 ]; then
  THINKING_CONFIG='{"enable_thinking":true}'
else
  THINKING_CONFIG='{"enable_thinking":false}'
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
  --max-num-batched-tokens 1024 --enable-chunked-prefill
  --skip-mm-profiling
  --limit-mm-per-prompt '{"image":20,"video":1}'
  --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}'
  --reasoning-parser qwen3
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>","default_thinking_token_budget":8000}'
  --default-chat-template-kwargs "$THINKING_CONFIG"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
  --speculative-config "$SPEC_CONFIG"
  --disable-uvicorn-access-log
)
if [ "$LINEAR_BACKEND" != "auto" ]; then
  ARGS+=(--linear-backend "$LINEAR_BACKEND")
fi
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
