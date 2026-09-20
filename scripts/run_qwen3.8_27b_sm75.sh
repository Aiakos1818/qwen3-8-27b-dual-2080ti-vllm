#!/usr/bin/env bash
# Verified launch profile for Qwen3.8-27B on 2x RTX 2080 Ti 22GB + NVLink.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi
LD_LIBRARY_PATH=$(printenv LD_LIBRARY_PATH || true)
PYTHONPATH=$(printenv PYTHONPATH || true)

: "$MODEL_PATH"
: "$VLLM_PYTHON"
: "$SERVED_MODEL_NAME"
: "$HOST"
: "$PORT"
: "$FLASHQLA_PATH"
: "$CHAT_TEMPLATE"
: "$CUDA_HOME"
: "$OMP_NUM_THREADS"
: "$VLLM_USE_DEEP_GEMM"
: "$VLLM_USE_FLASHINFER_SAMPLER"
: "$VLLM_QWOPUS_MTP_BF16_DRAFT"
: "$VLLM_SM75_SPEC_SYNC_MODE"
: "$VLLM_USE_V2_MODEL_RUNNER"

export CUDA_HOME OMP_NUM_THREADS VLLM_USE_DEEP_GEMM VLLM_USE_FLASHINFER_SAMPLER
export VLLM_QWOPUS_MTP_BF16_DRAFT VLLM_SM75_SPEC_SYNC_MODE VLLM_USE_V2_MODEL_RUNNER
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

# --- model -----------------------------------------------------------------
# Base route = the released FP8 checkpoint (default rope, 262,144 native). 180K
# is within the native window, so no YaRN. .env's MODEL_PATH points at the AWQ
# yarn checkpoint, so derive the FP8 one from the same directory; override with
# FP8_MODEL_PATH.
FP8_MODEL_PATH="${FP8_MODEL_PATH:-$(dirname "$MODEL_PATH")/Qwen3.8-27B-FP8}"

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$FP8_MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dtype half \
  --tensor-parallel-size 2 \
  --device-ids 0,1 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 180000 \
  --enable-prefix-caching \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":20,"video":1}' \
  --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}' \
  --reasoning-parser qwen3 \
  # Thinking budget: force </think> after 8000 reasoning tokens. Clients that send
  # no thinking_token_budget (opencode does not) would otherwise spend the whole
  # max_tokens on reasoning and get cut off before the tool call. Reasoning is
  # counted from the prompt too, so a turn left unterminated recovers immediately.
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>","default_thinking_token_budget":8000}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --chat-template "$CHAT_TEMPLATE" \
  --chat-template-content-format string \
  --gpu-memory-utilization 0.93 \
  --kv-cache-memory-bytes 4G \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.06}' \
  --cpu-offload-gb 0 \
  --disable-uvicorn-access-log
