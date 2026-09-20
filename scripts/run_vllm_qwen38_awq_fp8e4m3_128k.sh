#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 (default rope) + fp8_e4m3 KV, 128K context (131,072),
# no KV offload.
#
# The short-context rung: same stack as the 256K production profile with the
# window cut to 128K, so the KV pool and the per-card footprint drop. Note this
# is not half of the 256K pool: the GDN/mamba state adds ~0.74 GiB of fixed
# cost, so 131,072 needs >= 3,068,264,448 (scripts/tools/kv_pool_sizing.py);
# 3.2e9 is used here. Use the 256K profile when the wider window is needed, and
# the 128K/256K/500K SSDx4 rungs when the KV must survive a restart.
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

# --- model -----------------------------------------------------------------
# 262,144 is the model's native max_position_embeddings, so no context
# extension is needed: run the released default-rope checkpoint. The -yarn512k
# directory differs only in config.json's rope_parameters (weights are the same
# symlinked shards), and YaRN's mscale is a constant attention scale applied at
# every position, so within the native window it is pure precision cost with no
# benefit. Measured logit-level: dropping YaRN recovers 1.9%-3.6% top-1
# agreement at short context and more at long context (see
# reports/2026-09-sm75-optimization/accuracy-regression/). Use the -yarn512k
# checkpoint only for >262,144. Override with NATIVE_MODEL_PATH.
NATIVE_MODEL_PATH="${NATIVE_MODEL_PATH:-$(dirname "$MODEL_PATH")/Qwen3.8-27B-AWQ-INT4}"

# --- profile knobs ---------------------------------------------------------
# Calibrate to the "GPU KV cache size" line after a launch;
# scripts/tools/kv_pool_sizing.py reports the pool a profile needs.
: "${MAX_MODEL_LEN:=131072}"
: "${KV_CACHE_MEMORY_BYTES:=3200000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
# Speculative depth: 6 beats 5 (and 5 beats 3) at every context measured:
# 31.5K +4.8%, 215K +13% together with fp16 KV, 250K +4.7%, 449K +5.4%.  The
# per-round cost grows sub-linearly with n because only the verify runs all 64
# layers while the draft head runs one, so the extra draft is largely amortised.
# n=6 needs ~1% more KV than n=5 (extra speculative slots); the 256K/128K
# geometry carries a little more budget for that.  n=7 does start -- the earlier
# "fails to start" was a KV pool 0.09 GiB short -- but it is ~60% worse per
# token, so n stays at 6.
: "${SPEC_NUM_TOKENS:=6}"

# lm_head int8 is ON by default (docs/upstream-branch.md 6.15): +17~21% net
# decode throughput at no measurable acceptance cost. --disable-head8bit runs
# the unquantised (bf16) head instead.
HEAD8BIT=1
for _arg in "$@"; do
  case "$_arg" in
    --disable-head8bit)
      HEAD8BIT=0
      ;;
    -h | --help)
      echo "usage: $(basename "$0") [--disable-head8bit]"
      echo "  --disable-head8bit  run the bf16 lm_head checkpoint instead of the int8 variant"
      exit 0
      ;;
    *)
      echo "$0: unknown option: $_arg" >&2
      exit 2
      ;;
  esac
done

# The int8 head is a checkpoint variant: only the shard holding lm_head is
# rewritten, the rest are symlinks. Its Humming kernel needs the venv's cu13
# lib dir on LD_LIBRARY_PATH, which is appended to the export below.
EXTRA_LD_LIBRARY_PATH=""
if [ "$HEAD8BIT" = 1 ]; then
  case "$NATIVE_MODEL_PATH" in
    *-head8bit) ;;
    *) NATIVE_MODEL_PATH="$NATIVE_MODEL_PATH-head8bit" ;;
  esac
  EXTRA_LD_LIBRARY_PATH=$(
    ls -d "$(dirname "$(dirname "$VLLM_PYTHON")")"/lib/python*/site-packages/nvidia/cu13/lib 2>/dev/null | head -1
  )
  if [ -z "$EXTRA_LD_LIBRARY_PATH" ] || [ ! -d "$EXTRA_LD_LIBRARY_PATH" ]; then
    echo "$0: --disable-head8bit: venv cu13 lib dir not found" >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS CUDA_HOME
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_QWOPUS_MTP_BF16_DRAFT="${VLLM_QWOPUS_MTP_BF16_DRAFT:-1}"
export VLLM_SM75_SPEC_SYNC_MODE="${VLLM_SM75_SPEC_SYNC_MODE:-safe}"
# Verify spec drafts on the native FlashInfer decode path so speculative
# decode keeps full cudagraphs (SM75 has no fused GDN decode / TRT-LLM).
export VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE="${VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE:-1}"
# 262,144 is the model's own limit, so this is not strictly needed here; the
# 500K rungs do need it. Kept so every profile shares one env block.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
# API auth: inherit VLLM_API_KEY from the environment (empty = no auth).
export VLLM_API_KEY="${VLLM_API_KEY:-}"
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${EXTRA_LD_LIBRARY_PATH:+:$EXTRA_LD_LIBRARY_PATH}"
export PYTHONPATH="$FLASHQLA_PATH"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$NATIVE_MODEL_PATH"
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
  # Thinking budget: force </think> after 8000 reasoning tokens. Clients that send
  # no thinking_token_budget (opencode does not) would otherwise spend the whole
  # max_tokens on reasoning and get cut off before the tool call. Reasoning is
  # counted from the prompt too, so a turn left unterminated recovers immediately.
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>","default_thinking_token_budget":8000}'
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
