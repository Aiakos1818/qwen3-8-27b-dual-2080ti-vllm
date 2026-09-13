#!/bin/bash
set -e

# 配置 = INT4 W4A16 (AWQ-INT4) + YARN rope + 最大安全上下文 + fp8_e4m3 KV
#      + 保活(pin) + Mamba/GDN 锚点 + 两层 GPU/SSD 会话 offload(分块流式)
#   - 权重: Qwen3.8-27B-AWQ-INT4-yarn512k (~10.5GiB/卡)
#   - 池吃满本机每卡预算: --kv-cache-memory-bytes 9600000000 (8.94 GiB/卡)。
#     本机可启动的最大池 = 9.6e9; 9.7e9 首次请求即 OOM。
#   - --max-model-len 500800 (=313*1600 整块) = 池 9.6e9 的安全上限
#     (容量 ~525,816 token 减去 16 块 headroom)。用 scripts/kv_pool_sizing.py
#     诊断(见 docs/kv-optimization/GPU_MEMORY_CALCULATION.md §4.5)。
#     注: 512k(524288) 需安全池 9,999,155,200, 本机放不下 -> 满长深回退会重算;
#     要可靠深回退请用 500800 或更小。
#   - 锚点: VLLM_MAMBA_CKPT_TOKENS=32000/K=3(覆盖近尾 96k, 每 cadence 1 个锚点)。
#   - 保活: 结束且 >= VLLM_PIN_MIN_TOKENS(16000) 的会话整链 pin; 准入压力时
#     最小保活会话先落盘(分块, GPU 块逐块释放), 回来时 restore。
#   - SSD: quota 64 GiB=68719476736 B, 限速 800 MiB/s 保护系统盘, 启动清残留。
#   - 启动 flakiness: TP worker warmup 偶发 CUDA invalid argument/OOM; 重试即可,
#     重试前清理 /dev/shm/vllm_offload_*.mmap 残留。

export OMP_NUM_THREADS=8
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_QWOPUS_MTP_BF16_DRAFT=1
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Keep-alive (auto pin finished long sessions).
export VLLM_PIN_MIN_TOKENS=16000
# Durable Mamba/GDN checkpoint anchors (deep/truncated revert reuse).
export VLLM_MAMBA_CKPT_TOKENS=32000
export VLLM_MAMBA_CKPT_ANCHORS=3
# Two-tier GPU/SSD session offload (chunked, no session-size limit).
export VLLM_SSD_ROOT=${VLLM_SSD_ROOT:-/home/aiakos/Qwen3.8-27B-Deploy/ssd_kv}
export VLLM_SSD_QUOTA_BYTES=${VLLM_SSD_QUOTA_BYTES:-68719476736}
export VLLM_SSD_MAX_MBPS=${VLLM_SSD_MAX_MBPS:-800}
export VLLM_SSD_CLEAN_START=${VLLM_SSD_CLEAN_START:-1}
export VLLM_SSD_ONLY=1
# Eviction tiers: sessions < 32k are parked first, >= 32k by oldest-first
# (aligned with keep-alive 16k / anchors 32k).
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
  --max-model-len 500800 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES:-9600000000} \
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
