#!/usr/bin/env bash
# Qwen3.8-27B AWQ-INT4 + YARN + fp16 KV, 225,280 context (220K), no KV offload.
#
# Same KV budget as the 500.8K profile (9.6e9) but with fp16 KV, which halves
# the per-token KV (measured ~18.6 KB fp8 -> ~37.2 KB fp16 at n=5) so the
# reachable context drops to ~248K; 225,280 leaves ~14% headroom on the pool and
# keeps the "at least one full-length request" check (which wants ~4% more than
# the pool average) at 8.7e9 < 9.6e9.  Per-card VRAM is the same as the 500.8K
# profile (~21.1 GiB), which is measured to work on this host.
#
# Why fp16 KV: on Turing the fp8 software dequantisation costs more than the
# bandwidth it saves, so the attention kernel is ~18% faster with fp16 at 250K
# (1.456 -> 1.170 ms, same harness).  n=6 is worth ~5% over n=5 at every context
# measured.  Together they are the last two decode levers that do not need a
# different checkpoint.
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
: "${MAX_MODEL_LEN:=225280}"
: "${KV_CACHE_MEMORY_BYTES:=9600000000}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
# Speculative depth: 5 beats 3 on decode throughput at every context measured
# (2026-09-18: 31.5K +21%, 128K +23%, 250K +37%). The per-round cost grows
# sub-linearly with n because only the verify runs all 64 layers, while the
# acceptance does drop (40% vs 59%). n >= 7 fails to start (verify batch out
# of the captured shapes); n=6 measured ~5% faster again.
: "${SPEC_NUM_TOKENS:=6}"

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
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export PYTHONPATH="$FLASHQLA_PATH"

ARGS=(
  --host "$HOST" --port "$PORT"
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype half --tensor-parallel-size 2 --device-ids 0,1
  --kv-cache-dtype float16
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
