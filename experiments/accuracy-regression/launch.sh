#!/usr/bin/env bash
# accuracy-regression launcher
#   usage: launch.sh <B0|W4|W4y|H8f|H8>
#
# 5-config precision A/B. All share: max-model-len 262144, no MTP, TP=2,
# single sequence, same chat template / parsers. Only the intended variable
# changes between neighbouring configs:
#   B0  -> W4   weight quant (FP8 -> AWQ INT4), default rope, fp16 KV
#   W4  -> W4y  rope (default -> yarn), AWQ, fp16 KV
#   W4y -> H8f  lm_head (bf16 -> int8), AWQ+yarn, fp16 KV
#   H8f -> H8   KV dtype (fp16 -> fp8_e4m3), production
#   B0  -> H8   total production delta
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
# shellcheck disable=SC1091
if [ -f "$REPO_ROOT/.env" ]; then source "$REPO_ROOT/.env"; fi

: "${VLLM_PYTHON:?set VLLM_PYTHON in .env}"
: "${FLASHQLA_PATH:?set FLASHQLA_PATH in .env}"
: "${CHAT_TEMPLATE:?set CHAT_TEMPLATE in .env}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${OMP_NUM_THREADS:=8}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${SERVED_MODEL_NAME:=qwen38-27b}"

CONFIG="${1:?usage: launch.sh <B0|W4|W4y|H8f|H8>}"
MODELS_DIR="$(dirname "$MODEL_PATH")"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

EXTRA_LD_LIBRARY_PATH=""
HEAD8BIT=0
QUANT=""
case "$CONFIG" in
  B0)
    MODEL="$MODELS_DIR/Qwen3.8-27B-FP8"
    QUANT="fp8"
    KV="float16"
    POOL="${POOL:-9600000000}"
    ;;
  W4)
    MODEL="$MODELS_DIR/Qwen3.8-27B-AWQ-INT4"
    KV="float16"
    POOL="${POOL:-9600000000}"
    ;;
  W4y)
    MODEL="$MODELS_DIR/Qwen3.8-27B-AWQ-INT4-yarn512k"
    KV="float16"
    POOL="${POOL:-9600000000}"
    ;;
  H8f)
    MODEL="$MODELS_DIR/Qwen3.8-27B-AWQ-INT4-yarn512k-head8bit"
    KV="float16"
    POOL="${POOL:-9600000000}"
    HEAD8BIT=1
    ;;
  H8)
    MODEL="$MODELS_DIR/Qwen3.8-27B-AWQ-INT4-yarn512k-head8bit"
    KV="fp8_e4m3"
    POOL="${POOL:-5300000000}"
    HEAD8BIT=1
    ;;
  *)
    echo "unknown config: $CONFIG (want B0|W4|W4y|H8f|H8)" >&2
    exit 2
    ;;
esac

# Optional overrides for feasibility experiments / the KV-isolation pair.
KV="${KV_OVERRIDE:-$KV}"

if [ "$HEAD8BIT" = 1 ]; then
  EXTRA_LD_LIBRARY_PATH=$(
    ls -d "$(dirname "$(dirname "$VLLM_PYTHON")")"/lib/python*/site-packages/nvidia/cu13/lib 2>/dev/null | head -1
  )
  if [ -z "$EXTRA_LD_LIBRARY_PATH" ] || [ ! -d "$EXTRA_LD_LIBRARY_PATH" ]; then
    echo "$0: --head8bit: venv cu13 lib dir not found" >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# eval runs without auth so the harness needs no key
export VLLM_API_KEY=""
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}"
export PYTHONPATH="$FLASHQLA_PATH"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype half --tensor-parallel-size 2 --device-ids 0,1
  --kv-cache-dtype "$KV"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-memory-bytes "$POOL"
  --enable-prefix-caching --max-num-seqs 1
  --enable-prompt-tokens-details
  --max-num-batched-tokens 1024 --enable-chunked-prefill
  --skip-mm-profiling
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
  --max-logprobs "${MAX_LOGPROBS:-1000}"
  --disable-uvicorn-access-log
)
if [ "${LOGIT_EVAL:-0}" = 1 ]; then
  # dense-only: token-id prompts, no detokenisation of the top-k
  # (decoded_token is the dominant per-entry cost).
  ARGS+=(--language-model-only --skip-tokenizer-init)
elif [ "${LOGIT_EVAL:-0}" = 2 ]; then
  # probes: generation-side `logprobs` needs a tokenizer, so keep it, but
  # skip the vision tower and the chat/tool machinery.
  ARGS+=(--language-model-only)
else
  ARGS+=(
    --limit-mm-per-prompt '{"image":20,"video":1}'
    --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}'
    --reasoning-parser qwen3
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>","default_thinking_token_budget":8000}'
    --default-chat-template-kwargs '{"enable_thinking":true}'
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
  )
fi
if [ -n "$QUANT" ]; then ARGS+=(--quantization "$QUANT"); fi
if [ "${LOGIT_EVAL:-0}" = 0 ] && [ -n "${CHAT_TEMPLATE:-}" ]; then ARGS+=(--chat-template "$CHAT_TEMPLATE"); fi

echo "[launch] config=$CONFIG model=$MODEL kv=$KV pool=$POOL max_len=$MAX_MODEL_LEN" >&2
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
