#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp8_e4m3 KV, 500.8K context, no KV offload.
#
# The base profile of this branch: prefix caching only, one long request per
# engine. 9.6e9 pool -> 525,229 tokens of capacity (measured), i.e. ~1.05x of
# one full 500.8K request.
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
# Calibrate to the "GPU KV cache size" line after a launch;
# scripts/tools/kv_pool_sizing.py reports the pool a profile needs.
: "${MAX_MODEL_LEN:=500800}"
: "${KV_CACHE_MEMORY_BYTES:=9600000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
# Speculative depth: 5 beats 3 on decode throughput at every context measured
# (2026-09-18: 31.5K +21%, 128K +23%, 250K +37%). The per-round cost grows
# sub-linearly with n because only the verify runs all 64 layers, while the
# acceptance does drop (40% vs 59%). n=6 is faster again at every context
# measured (31.5K +4.8%, 215K +13% together with fp16 KV, 250K +4.7%,
# 449K +5.4%), so the 500K rungs default to it: the 9.6e9 pool still covers a
# full 500,800 request (509,877 tokens measured at n=6).
# n=7 does start -- the earlier "fails to start" was a KV pool 0.09 GiB short --
# but it is ~60% worse per token, so n stays at 5-6.
: "${SPEC_NUM_TOKENS:=6}"

# Optional: --head8bit runs the int8 lm_head checkpoint variant (docs section
# 6.15): +17~21% net decode throughput at no measurable acceptance cost.  Like
# the yarn directory it only rewrites the shard that holds lm_head, so the
# unchanged shards are symlinks; nothing is copied.  It runs through the Humming
# kernel, whose NVRTC JIT needs the venv's cu13 lib dir on LD_LIBRARY_PATH, which
# is appended to the export below.
EXTRA_LD_LIBRARY_PATH=""
for _arg in "$@"; do
  case "$_arg" in
    --head8bit)
      case "$MODEL_PATH" in
        *-head8bit) ;;
        *) MODEL_PATH="${MODEL_PATH}-head8bit" ;;
      esac
      EXTRA_LD_LIBRARY_PATH=$(
        ls -d "$(dirname "$(dirname "$VLLM_PYTHON")")"/lib/python*/site-packages/nvidia/cu13/lib 2>/dev/null | head -1
      )
      if [ -z "$EXTRA_LD_LIBRARY_PATH" ] || [ ! -d "$EXTRA_LD_LIBRARY_PATH" ]; then
        echo "$0: --head8bit: venv cu13 lib dir not found" >&2
        exit 2
      fi
      ;;
    -h | --help)
      echo "usage: $(basename "$0") [--head8bit]"
      echo "  --head8bit  run the int8 lm_head checkpoint variant (docs/upstream-branch.md 6.15)"
      exit 0
      ;;
    *)
      echo "$0: unknown option: $_arg" >&2
      exit 2
      ;;
  esac
done

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
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}"
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
  --disable-uvicorn-access-log
)
if [ -n "${CHAT_TEMPLATE:-}" ]; then
  ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
