#!/usr/bin/env bash
# 512k + KV anchors (AWQ-INT4, fp8_e4m3 KV, MTP=3, pool 9.6e9).
# Same MTP as the 435k production profile (block_size=1600, cadence 32000).
# The pool must leave >= K free slots at full length (anchor deadlock criterion,
# docs/kv-optimization/vllm_02_锚点.md 4.2). Cadence auto-aligns to the block
# size (floor), so the raw 32000 is valid at 1600 and would floor to 31680 at
# MTP1's 1584. 9.7e9 OOMs at request time.
# SSD offload on: the pool is ~96% full, so a revert needs S to spill/restore
# rather than be evicted.
export OMP_NUM_THREADS=8
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_QWOPUS_MTP_BF16_DRAFT=1
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PIN_MIN_TOKENS=16000
# Auto-aligned to the block size (MTP3 -> 1600 -> stays 32000).
export VLLM_MAMBA_CKPT_TOKENS=32000
export VLLM_MAMBA_CKPT_ANCHORS=3
# Offload: with the pool ~96% full a revert would otherwise evict the resident
# session instead of reusing it; the SSD tier lets S spill and be restored.
export VLLM_SSD_ROOT=${VLLM_SSD_ROOT:-/home/aiakos/Qwen3.8-27B-Deploy/ssd_kv}
export VLLM_SSD_QUOTA_BYTES=${VLLM_SSD_QUOTA_BYTES:-68719476736}
export VLLM_SSD_MAX_MBPS=${VLLM_SSD_MAX_MBPS:-800}
export VLLM_SSD_CLEAN_START=${VLLM_SSD_CLEAN_START:-1}
export VLLM_SSD_ONLY=1
export VLLM_HOSTTIER_EVICT_SMALL_TOKENS=32000
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/venv/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PYTHONPATH=/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/src/FlashQLA-SM70-SM75

exec /home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model /home/aiakos/Qwen3.8-27B-Deploy/models/Qwen3.8-27B-AWQ-INT4-yarn512k \
  --served-model-name qwen38-27b \
  --dtype half --tensor-parallel-size 2 --device-ids 0,1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 524288 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-memory-bytes 9600000000 \
  --enable-prefix-caching --max-num-seqs 1 \
  --enable-prompt-tokens-details \
  --max-num-batched-tokens 1024 --enable-chunked-prefill \
  --no-async-scheduling \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":20,"video":1}' \
  --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}' \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  --cpu-offload-gb 0 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":4000000000}}' \
  --disable-uvicorn-access-log
