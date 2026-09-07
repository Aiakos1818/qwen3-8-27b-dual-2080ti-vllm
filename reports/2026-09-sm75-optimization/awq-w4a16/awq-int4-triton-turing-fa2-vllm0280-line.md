# AWQ INT4 + Triton-Turing FA2：vLLM 0.28.0 独立试验线

日期：2026-09-06  
目标：不恢复 2026-06 的旧 service；在当前 **vLLM 0.28.0** 环境中直接加载旧 AWQ INT4 27B 参数画像，并把 Triton-Turing 的 **FA2 prefill kernel 嵌入 vLLM 热路径**。AWQ-Marlin 负责权重 GEMM，Turing FA2 负责 full-attention prefill，paged-KV writer 与 decode 路径各保留已验证的最优实现。

## 1. 结论先行：能做什么，不能把什么混为一谈

### 1.1 可以直接做

1. 以 vLLM 0.28.0 加载旧基线模型 `/home/<user>/models/Qwopus3.6-27B-v2-AWQ-4bit`，使用 `--quantization awq_marlin`、TP2、128K、MTP3、FP16 KV，得到“同模型、升级后 runtime”的 AWQ 对照。
2. 在这个 AWQ 基线中替换标准 Triton 为 `triton-turing`，但仍保持 FlashInfer attention，判断 compiler fork 是否改善其它 Triton kernel。
3. 在 FP16 KV 下分别测试标准 Triton 和 Triton-Turing 的 `TRITON_ATTN`，测 Turing software pipeline 对 vLLM paged attention 的净收益。
4. 用 AWQ 减少的 weight 显存，测试更大的 FP16 KV、DFlash2/draft、更多并发或更大的 prefill chunk；这些是 AWQ 最现实的端到端收益来源。

### 1.2 主路线：利用 Triton-Turing FA2，而非围绕官方 backend 开关

目标不是调用 vLLM 的 `FLASH_ATTN` flag，而是在 vLLM 0.28 的 attention runner 中实际调用 Triton-Turing 的 SM75 FA2 forward kernel。第一版的热路径应是：

```text
AWQ W4A16 weights ──Marlin linear──> FP16 hidden/QKV
                                      │
                       full-attention prefill q_len > 1
                                      │
                         Triton-Turing FA2 forward
                                      │
                  vLLM native FP16 paged-KV cache writer
                                      │
              FlashInfer paged decode / prefix extend / q_len = 1
                                      │
                    existing FlashQLA GDN + MTP paths stay intact
```

vLLM 官方 `FLASH_ATTN` backend 的 SM80 gate 只说明它不是本项目的接入点；它不否定 Triton-Turing 在 SM75 上的 FA2 实现。AWQ 也不需要去“改变”这个 gate。为避免混淆，三条路径严格区分：

| 名称 | 实际含义 | SM75 + AWQ 的结论 |
|---|---|---|
| `FLASH_ATTN` | vLLM 官方 FlashAttention backend / FlashAttention-2 等 | 不是本线的接入点 |
| `TRITON_ATTN` | vLLM 现有 Triton paged-attention backend | 诊断对照：帮助判断 paged layout 和 compiler 的差异 |
| Triton-Turing FA2 | `triton-turing` 仓库中的 SM75 software-pipelined FA2 forward kernel | **本线主目标**：作为新的 vLLM prefill operator/backed dispatch 接入 |

### 1.3 AWQ INT4 不等于会用上 Triton-Turing 的 pure-INT4 MMA

旧笔记中的 AWQ 是 `awq_marlin`：典型形式是 **W4A16**，即 4-bit 权重、FP16 activation。vLLM 的 AWQ-Marlin 路线会在模型加载时重排 AWQ weights，并调用专用 Marlin linear kernel；单纯替换 Triton 编译器通常不会改写该 GEMM。

Triton-Turing README 中的 `m8n8k32` pure INT4 MMA 是整数 activation × 整数 weight 的内核能力。要真正命中它，需要另做 W4A8/W4A4 的 activation quantization、校准、打包格式和 fused linear backend；它不是现成 AWQ W4A16 checkpoint 的无损替换。

所以此线要分成两个可验证研发目标：

```text
AWQ W4A16 + Marlin                 -> 先获得显存/吞吐基线
Triton-Turing FA2 prefill backend  -> 优化 attention，而非 AWQ GEMM
Triton-Turing W4A8/W4A4 linear     -> 独立的量化与 GEMM 项目，最后再做
```

## 2. 老 Obsidian 基线如何正确使用

读取的笔记为 `notes/2026-06-08-linux-qwen-27b-baseline.md`。它提供的是一个可复现的**加载参数画像**，不是可直接搬用的当前性能数字：模型、vLLM 版本、Triton、FlashInfer、patch、温度和输入集都已经变化。

| 旧笔记项 | 旧值 | 0.28.0 迁移处理 |
|---|---|---|
| 模型 | `Qwopus3.6-27B-v2-AWQ-4bit` | 先用同一路径，保证是完整 checkpoint |
| 量化 | `awq_marlin` | 保留，作为 W4A16 Marlin 基线 |
| TP | 2 | 保留，GPU 0/1 |
| 最大上下文 | 128000 | 保留为首轮上限 |
| `max-num-batched-tokens` | 8192 | 保留为首轮；再 sweep 4096/8192/16384 |
| `max-num-seqs` | 2 | 保留为首轮；另做 1/2/4 工作画像 |
| `gpu-memory-utilization` | 0.90 | 保留为首轮；**不要**同时指定 KV bytes |
| KV dtype | 未显式指定，实际为 FP16/auto 画像 | 首轮显式 `float16`，便于与 Turing FA2 对照 |
| FlashQLA GDN | `flashqla_legacy` | 仅在 0.28 runtime patch/extension 真正加载后保留 |
| MTP | 3 | 保留为首轮，同时 sweep 0..5 |
| FlashInfer sampler | 1 | 作为一个独立变量，测 0/1 |
| SM75 sync | `nosync` | 默认先用当前 0.28 已验证的 `safe`；只有确定性和长压测通过后才测试 `nosync` |
| `performance-mode` / `optimization-level` | 老 fork 私有参数 | **不迁移**；上游 0.28 没有等价安全替换 |
| `max-cudagraph-capture-size=4` | 老接口 | 转为当前 `compilation-config` 的 PIECEWISE / capture `[4]` |

旧笔记没有记录同条件 raw benchmark。因此它用于“参数与功能回归对照”，而不是拿 2026-06 的数字直接宣称 0.28 提速。历史 0.24 FP8-KV 严格 A/B 的 decode 数字也不适用于本 AWQ checkpoint，不能混入第一张对比表。

## 3. 首个可运行配置：AWQ-Marlin 0.28 基线 A0

这是前台测试配置，不是 systemd unit，也不恢复旧 service。它需要整个 TP2 占用两张卡；因此只能在当前 8000 服务停掉后的维护窗口运行，或使用另一台有两张空闲 GPU 的机器。启动前先确认模型目录是完整的，不能以 2026-09-01 的“下载中”记录代替检查。

```bash
# 只读预检：在目标机执行
AWQ_MODEL=/home/<user>/models/Qwopus3.6-27B-v2-AWQ-4bit
VLLM_PY=/home/<user>/vllm-env-0280-qwopus/bin/python

test -s "$AWQ_MODEL/config.json"
test -f "$AWQ_MODEL/qwen3.6-enhanced.jinja"
"$VLLM_PY" -c 'import vllm, torch; print(vllm.__version__, vllm.__file__); print(torch.cuda.get_device_capability(0))'
"$VLLM_PY" -m vllm.entrypoints.openai.api_server --help | rg 'awq_marlin|kv-cache-dtype|compilation-config|speculative-config'
```

```bash
# A0：Qwopus AWQ W4A16 / Marlin、FP16 KV、vLLM 0.28.0
# 启动前只需按既有维护步骤停掉占用 GPU0/1 的当前服务；不恢复任何老 unit。
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/home/<user>/FlashQLA-SM70-SM75
export OMP_NUM_THREADS=8
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_QWOPUS_MTP_BF16_DRAFT=1
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export TORCH_EXTENSIONS_DIR=/home/<user>/.cache/torch_extensions/vllm-0280-awq-a0
export TRITON_CACHE_DIR=/home/<user>/.cache/triton/vllm-0280-awq-a0

/home/<user>/vllm-env-0280-qwopus/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8001 \
  --model /home/<user>/models/Qwopus3.6-27B-v2-AWQ-4bit \
  --served-model-name qwen-awq-a0 \
  --dtype float16 --quantization awq_marlin \
  --tensor-parallel-size 2 --device-ids 0,1 \
  --max-model-len 128000 \
  --kv-cache-dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 --max-num-seqs 2 \
  --enable-chunked-prefill --enable-prefix-caching \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template /home/<user>/models/Qwopus3.6-27B-v2-AWQ-4bit/qwen3.6-enhanced.jinja \
  --chat-template-content-format string \
  --trust-remote-code \
  --override-generation-config '{"temperature":0.5,"top_p":0.85,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.06}' \
  --cpu-offload-gb 0 --disable-uvicorn-access-log
```

这条命令有意不包含以下误导性做法：

- 不设 `VLLM_ATTENTION_BACKEND=FLASH_ATTN`；SM75 下它会被 vLLM 0.28 拒绝。
- 不设 `--kv-cache-dtype fp8_e4m3`；A0 的目的就是获得与旧 AWQ/FP16-KV 画像一致的 FA2/Triton 对照。FP8 KV 在后续单独测试。
- 不移植旧 `--performance-mode interactivity` 和 `--optimization-level 3`；它们是旧 fork 语义，不是上游 0.28 调参。
- 不指定 `--kv-cache-memory-bytes`；否则会改变旧配置的显存分配逻辑，且覆盖 `gpu-memory-utilization`。

### A0 必须核验的日志与运行时事实

```text
vllm.__version__ == 0.28.0
vllm.__file__ 指向预期 0.28 源码/venv，而不是残留 site-packages
quantization=awq_marlin，且线性层日志确实走 Marlin/AWQ kernel
attention=FLASHINFER（不是 silent fallback）
FlashQLA legacy GDN 已加载，且没有 compute_70 JIT error
MTP=3 已初始化，且 metrics 有 draft/accepted 计数
KV dtype=float16，记录实际 cache token 数与每卡峰值显存
```

## 4. AWQ + Triton-Turing 的正交试验顺序

| Case | 权重 GEMM | 编译器 | attention | KV | 目的 |
|---|---|---|---|---|---|
| A0 | AWQ-Marlin W4A16 | 标准 Triton | FlashInfer | FP16 | 0.28 AWQ 迁移基线 |
| A1 | AWQ-Marlin W4A16 | triton-turing | FlashInfer | FP16 | fork 是否加速其它 Triton kernel；最重要的低风险试验 |
| A2 | AWQ-Marlin W4A16 | triton-turing | **`TURING_FA2` prefill + FlashInfer decode** | FP16 | 主路线：真正使用仓库 FA2 forward，而不是只装编译器 |
| A3 | AWQ-Marlin W4A16 | 标准 Triton | TRITON_ATTN | FP16 | 诊断对照：vLLM Triton paged-attention 的基线 |
| A4 | AWQ-Marlin W4A16 | triton-turing | TRITON_ATTN | FP16 | 诊断对照：Turing compiler 对现有 paged-attention 的净收益 |
| A5 | AWQ-Marlin W4A16 | 标准/turing 各一 | FlashInfer/TRITON 各一 | FP8 E4M3 | 只在 FP16 路径正确后看容量与转码的影响 |
| A6 | W4A8/W4A4 新 checkpoint 或新量化 backend | triton-turing | 同 A4 | FP16/INT8 KV | 评估 pure-INT4 MMA；这是独立研究，不可与 A0-A5 混报 |

每个 case 运行同一组：4K、20K、60K prompt；128 和 512 completion；温度 0/固定 seed；生产采样；冷/热 prefix；tool-call。每项至少 5 个稳态重复，保留 TTFT、prefill tok/s、decode tok/s、E2E、每卡显存、MTP acceptance 分位置、cache hit、日志和 profiler trace。

### 4.1 A2 是研发主线；A1/A3/A4 是归因对照

A2 不依赖 A4 的 generic `TRITON_ATTN` 先赢。Triton-Turing 的 FA2 是要直接嵌入 prefill 热路径的目标内核；vLLM `TRITON_ATTN` 的 paged layout、cache conversion 和 decode 实现可能掩盖它的收益，因此 A3/A4 只能用来解释结果，不能决定是否尝试 A2。

### 4.2 先测 A1 的原因

若 A1 比 A0 快，说明 `triton-turing` 对当前模型的其它 Triton 组件有真实贡献；若 A4 慢，只能说明 vLLM `TRITON_ATTN` 的布局/调度不合适，不能把 compiler fork 或 A2 的专用 FA2 路线整体判死。

若 A1 与 A0 持平，同时 profiler 显示 AWQ-Marlin GEMM、FlashInfer、GDN 或 TP collective 占主导，也符合预期：AWQ-Marlin 不是通过 `@triton.jit` 的纯 INT4 kernel 自动获得加速。

## 5. Triton-Turing FA2 的正确接入设计：A2

### 5.1 为什么 A4 不等于仓库的 FA2

vLLM 的 `TRITON_ATTN` 是 paged-KV serving backend；Triton-Turing 仓库中的 FA2 forward tutorial通常处理的是变长/连续 QKV attention。即使二者都用 Triton，也不能假设 vLLM 会自动调用仓库的 FA2 kernel。

### 5.2 A2 所需的最小 custom backend

在独立的 vLLM 0.28 工作树中新增 `TURING_FA2` backend；不要修改 `FlashAttentionBackend` 的 SM80 gate。最小范围：

1. **支持范围先收窄为 FP16、SM75、full-attention prefill。** 不处理 FP8 KV、不承诺 decode、不在第一版覆盖 prefix-cache extend。
2. 从当前 Qwen layer config 获取 head_dim、Q/KV heads、rope 后 QKV，使用 Triton-Turing FA2 varlen forward 处理纯 prefill chunk。
3. KV 写入先复用 vLLM 原生、已验证的 FP16 writer，作为数据布局 oracle；不要在第一版实现 FP8 writer。
4. decode（`q_len=1`）、prefix-cache hit/extend、MTP draft/target 的复杂 paged 场景继续走 FlashInfer，且日志必须计数说明何时走 FA2、何时走 paged decode。
5. 对每个 full-attention/KV-cache group 明确选择 backend；混合 GDN 层继续使用 FlashQLA/Mamba 既有路径。
6. 与 FlashInfer 对同一 FP16 Q/K/V 做数值对照；温度 0 的端到端 token 序列、tool call 和 60K 长文都通过后才比较速度。

### 5.3 A2 的成功定义

只有同时满足下列三点才叫“FA2 打开成功”：

- profiler 中存在 Triton-Turing 生成的 FA2 prefill kernel，而非仅环境里安装了 fork；
- 纯 prefill 的 attention kernel 与 FlashInfer 数值一致（定义误差阈值）且端到端功能回归通过；
- 相对 A0/A1，在目标 20K/60K 工作负载的 **prefill 或 TTFT 中位数**有明确正收益，p95 不恶化超过 10%。

否则只能称为“装了 Triton-Turing”或“TRITON_ATTN 可启动”，不是 FA2 加速已接入。

## 6. AWQ 释放的显存应该如何用

以下每项都要与 A0 分开试，不能一次全打开：

| Sweep | 候选 | 目标 |
|---|---|---|
| KV 总量 | 由 0.90 自动预算；或固定 4/5/6 GiB | 记录最大上下文、并发和 OOM，不假设单 token 更快 |
| KV dtype | FP16 → FP8 E4M3 → INT8/INT4（仅 0.28 支持时） | 分开测容量、质量和速度 |
| MTP | 0/1/2/3/4/5 | AWQ 余量可能改变 MTP graph/cache 的最优点 |
| DFlash2 | W4A16 draft / 原 draft | AWQ 主模型腾出显存后的 end-to-end speculative 候选 |
| `max-num-seqs` | 1/2/4 | 单请求延迟与 aggregate throughput 分开报告 |
| `max-num-batched-tokens` | 4096/8192/16384 | 长 prefill TTFT 与峰值 activation 取舍 |
| CUDA graph | `[4]`、实际命中的 `[4,8]` 等 | 以日志中真实 MTP/并发 shape 为准 |
| FlashInfer sampler | 0/1 | 老参数画像的变量，不预设哪一边更快 |
| SM75 spec sync | `safe`/`nosync`（仅支持时） | `nosync` 必须以确定性和长稳压测试证明无竞态 |

## 7. 与当前 Qwen3.8 AWQ/DFlash2 线的关系

本文件的 A0-A6 选择 **旧 Qwopus AWQ**，因为用户要求和 2026-06-08 旧笔记作可比对照。它回答的是：升级到 0.28.0 后，旧 AWQ + Marlin 参数画像能否为 Triton-Turing FA2 prefill 提供足够显存和适合的端到端工作负载。

当前 Qwen3.8 的候选 AWQ（例如普通 AWQ INT4、带 MTP 的 AWQ 或 DFlash2 组合）应在 A0-A3 得出结论后作为第二张表测试。否则一次比较会同时改变：模型权重、MTP head、chat template、拒答行为、GDN 结构、量化 pack、draft 和 runtime，无法判断提速来自哪里。

## 8. 开测前的真实 blocker

本机在本次调研时无法通过无密钥 SSH 读取 `<server>`，所以尚未确认：

- 旧 AWQ 模型目录今天是否完整；
- 当前 0.28.0 venv 的实际 `vllm.__file__`、Triton、FlashInfer、FlashQLA import；
- 当前生产服务能否安全让出双卡给前台 8001 试验；
- 0.28 本地 patch 中 `safe/nosync` 的精确语义和可用枚举。

这些是启动前的只读 P0，不是要求恢复旧 service。确认后先前台运行 A0；随后运行 A1，并行推进 A2 的 FA2 prefill adapter；A3/A4 只作为 paged-attention 归因对照。

## 9. 资料口径

- 旧加载参数与硬件快照：Obsidian `notes/2026-06-08-linux-qwen-27b-baseline.md`。
- 0.28.0 direct FA backend 兼容性：vLLM v0.28.0 的 `flash_attn.py` 源码，而不是旧版本经验。
- Triton-Turing 的 SM75 FA2 和 pure-INT4 能力：项目 README；其 operator microbenchmark 不等于 AWQ W4A16 + vLLM 服务端到端结果。
