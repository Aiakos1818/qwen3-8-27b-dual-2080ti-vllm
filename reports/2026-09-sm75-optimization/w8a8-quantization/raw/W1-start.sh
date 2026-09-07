#!/bin/bash
export TRITON_CACHE_DIR=/home/<user>/.cache/triton/W1
export TORCH_EXTENSIONS_DIR=/home/<user>/.cache/torch_extensions/W1
export TRITON_SM75_BF16_DOT_AS_F16=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_QWOPUS_MTP_BF16_DRAFT=1
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONPATH=/home/<user>/FlashQLA-SM70-SM75-0280
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
exec /home/<user>/vllm-env-0280-qwopus/bin/python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8001 --model /home/<user>/models/Qwen3.8-27B-INT8-W8A8-imatrix-MTP --served-model-name qwen-local --dtype half --tensor-parallel-size 2 --device-ids 0,1 --kv-cache-dtype float16 --max-model-len 65536 --enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 4096 --enable-chunked-prefill --no-async-scheduling --skip-mm-profiling --limit-mm-per-prompt '{"image":20,"video":1}' --mm-processor-kwargs '{"min_pixels":100352,"max_pixels":501760}' --reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"\u601d\u8003","reasoning_end_str":"\u7ed3\u675f"}' --default-chat-template-kwargs '{"enable_thinking":true}' --enable-auto-tool-choice --tool-call-parser qwen3_xml --chat-template /home/<user>/models/Qwen-Fixed-Chat-Templates/chat_template.jinja --chat-template-content-format string --gpu-memory-utilization 0.93 --kv-cache-memory-bytes 4G --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.06}' --cpu-offload-gb 0 --disable-uvicorn-access-log
