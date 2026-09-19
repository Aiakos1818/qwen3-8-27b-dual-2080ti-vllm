#!/usr/bin/env bash
# Temp launcher for A/B tests. Same as scripts/run_vllm_qwen38_awq_fp8e4m3_500k.sh
# but overrides given on the command line WIN over .env (the profile script
# sources .env first, so its ${VAR:-default} silently discards the caller's env).
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# capture caller overrides before .env clobbers them
IN_SYNC="${VLLM_SM75_SPEC_SYNC_MODE-}"
IN_V2="${VLLM_USE_V2_MODEL_RUNNER-}"
IN_GDNK="${VLLM_GDN_DECODE_KERNEL-}"
IN_SPECN="${SPEC_NUM_TOKENS-}"
IN_MAXLEN="${MAX_MODEL_LEN-}"
IN_KVDT="${KV_CACHE_DTYPE-}"
IN_KVMEM="${KV_CACHE_MEMORY_BYTES-}"

source "$REPO_ROOT/.env"

[ -n "$IN_SYNC" ] && export VLLM_SM75_SPEC_SYNC_MODE="$IN_SYNC"
[ -n "$IN_V2" ] && export VLLM_USE_V2_MODEL_RUNNER="$IN_V2"
[ -n "$IN_GDNK" ] && export VLLM_GDN_DECODE_KERNEL="$IN_GDNK"
[ -n "$IN_SPECN" ] && SPEC_NUM_TOKENS="$IN_SPECN"
[ -n "$IN_MAXLEN" ] && MAX_MODEL_LEN="$IN_MAXLEN"
[ -n "$IN_KVMEM" ] && KV_CACHE_MEMORY_BYTES="$IN_KVMEM"

: "${MODEL_PATH:?}"; : "${VLLM_PYTHON:?}"; : "${FLASHQLA_PATH:?}"
: "${SERVED_MODEL_NAME:=qwen38-27b}"
: "${HOST:=0.0.0.0}"; : "${PORT:=8000}"
: "${CUDA_HOME:=/usr/local/cuda}"; : "${OMP_NUM_THREADS:=8}"
: "${MAX_MODEL_LEN:=500800}"
: "${KV_CACHE_MEMORY_BYTES:=9600000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${SPEC_NUM_TOKENS:=3}"
: "${KV_CACHE_DTYPE:=fp8_e4m3}"
[ -n "$IN_KVDT" ] && KV_CACHE_DTYPE="$IN_KVDT"

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype half --tensor-parallel-size 2 --device-ids 0,1
  --kv-cache-dtype "$KV_CACHE_DTYPE"
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
  --disable-uvicorn-access-log
  --enable-logging-iteration-details
)
if [ -z "${NO_SPEC:-}" ]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}")
fi
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi
if [ -n "${CG_MODE:-}" ]; then
  ARGS+=(--compilation-config "{\"cudagraph_mode\":\"$CG_MODE\"}")
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

echo "[run_test] sync_mode=$VLLM_SM75_SPEC_SYNC_MODE v2_runner=$VLLM_USE_V2_MODEL_RUNNER spec_n=$SPEC_NUM_TOKENS"
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
