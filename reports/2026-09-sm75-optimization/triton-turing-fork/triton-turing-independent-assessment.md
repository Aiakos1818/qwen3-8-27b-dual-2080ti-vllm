# triton-turing 接入调研（独立判断版）

日期：2026-09-06
方法：只基于一手材料独立分析 —— triton-turing 仓库（README/setup.py/FA2 tutorial/commit 记录）、
qwen3-8-27b-dual-2080ti-vllm 仓库（README/启动脚本/patch/环境锁）、vLLM v0.27.1 源码
（attention backends、GDN 层、Marlin 支持门）、Qwen3.8-27B 架构公开资料、2080 Ti 硬件规格。
不引用本工作区其他调研结论。

---

## 0. 核心结论（TL;DR）

1. **triton-turing 是编译器级替换，不是加速插件。** 它是 Triton 3.7.0 的 fork，装上后替换
   `triton` pip 包，所有 `@triton.jit` kernel 重新用 SM75 感知的方式编译（恢复 Tensor Core
   MMA + 首次实现的软件流水线）。它**不提供**新的 attention backend，也不改写 cuBLAS GEMM。

2. **本服务里真正走 Triton 的热路径很窄。** 全注意力（16/64 层）走 FlashInfer CUDA，
   GDN prefill（48 层）走 FlashQLA CUDA，线性层 GEMM 走 cuBLAS FP16（FP8 权重反量化后）。
   fork 直接受益的只有：GDN decode 的 FLA Triton 递推内核（48 层、每 token 执行）、
   MTP draft、RoPE/RMSNorm/SiLU/采样/反量化等辅助小 kernel。
   **预期端到端收益：decode 约 +0~10%，prefill 约 +0~5%。值得做（低风险、且是后续一切的基础），
   但不要指望它单独带来量级变化。**

3. **最大的加速空间在权重字节数，不在编译器。** 2080 Ti 单卡 616 GB/s，TP2 下每卡每 token
   搬运 13.5 GB（27B FP8），raw decode 上限 ≈ 45 tok/s；实测 ~100 tok/s 说明 MTP=3 已放大
   ~2.2×，**decode 已贴近 FP8 权重的带宽屋顶**。要突破只有两条路：
   (a) 权重改 INT4（字节减半 → raw ~85 tok/s → 含 MTP 约 150~190 tok/s，潜在 1.5~2×）；
   (b) 提高 MTP 接受率。
   关键独立发现：**vLLM 0.27.1 的 Marlin 已支持 SM75**（`marlin_utils.py`：
   `device_capability < 75 → []`，AWQ uint4 / GPTQ uint4b8 均在列）。
   所以 INT4 第一步**不需要 fork**：直接找/做一个 Qwen3.8-27B 的 AWQ/GPTQ INT4 checkpoint，
   `--quantization awq_marlin` 即可 A/B。fork 的 pure-INT4 MMA（m8n8k32，219 TOPS）
   属于 W4A8 路线的第二阶段优化（需要 activation 量化 + 自定义 Triton GEMM 挂进 vLLM 量化层）。

4. **fork 的 FA2 优势不覆盖本模型的注意力形状。** 其 FA2 实测在 head_dim 64（+21~26% vs
   CUTLASS）和 128（+4~8%）；本模型全注意力层是 **GQA 24/4、head_dim 256**，超出实测包络。
   Turing 64 KB/CTA 共享内存下 d256 必须缩小 tile，MMA 效率下降。全注意力又只占 1/4 层数，
   即使 attention 快 20%，端到端 prefill 也只 +5% 左右。**TRITON_ATTN 路线定性为"测一下，别押注"。**

5. **推荐执行顺序**：nsys 剖析 → 只换编译器（保留 FlashInfer）→ 参数 sweep →
   INT4 checkpoint A/B → TRITON_ATTN A/B → （仅当剖析证明热点在 GEMM 时）自定义 INT4-MMA GEMM。

---

## 1. 一手事实盘点

### 1.1 硬件与服务（来自服务仓库 README/脚本/环境锁）

| 项 | 值 |
|---|---|
| GPU | 2× 魔改 RTX 2080 Ti 22GB（SM75 / CC 7.5），NVLink NV2（2 条，单条约 25.8 GB/s） |
| 显存带宽 | ~616 GB/s/卡（GDDR6 256-bit） |
| CPU/内存 | Ryzen 7 5700X 8C16T / 32 GiB（编译并行度受限） |
| 栈 | Ubuntu 24.04，driver 580.159.03，CUDA 13.0，Py3.12，torch 2.13.0+cu130 |
| vLLM | 0.27.1（commit 6e448d0）+ 本仓库 patch（GDN flashqla_legacy、MTP 兼容、SM75 FlashInfer/采样修正） |
| Triton | 3.7.1（torch 2.13 的依赖 pin） |
| Attention | FlashInfer 0.6.16.post3（SM75 可用；FLASH_ATTN 被 SM80 门拒绝） |
| GDN prefill | FlashQLA-SM70-SM75（`gdn_prefill_backend=flashqla_legacy`） |
| 模型 | Qwen3.8-27B FP8（dense，VLM，带 MTP head） |
| 关键参数 | TP2、`--dtype half`、`--quantization fp8`、`--kv-cache-dtype fp8_e4m3`、180K ctx、KV 4G/卡、max-num-seqs=1、max-num-batched-tokens=4096、MTP=3、PIECEWISE graph [4]、prefix cache + chunked prefill |
| 实测 | prefill ~1100~1340 tok/s；decode ~84~101 tok/s；TTFT 2.6s@2.8K → 53s@59K |

### 1.2 模型架构（HF/SGLang 公开资料）

- **48 层 Gated DeltaNet**（线性递推，head_dim 128）+ **16 层 Gated Attention**（GQA 24/4，
  head_dim 256，64 维 rotary）。共 64 层，dense 27B。
- 推论：
  - 全注意力 KV 只存在于 16 层 → KV 显存/格式优化的作用面是 1/4 层；
  - GDN 状态是固定大小矩阵（不随上下文增长）→ 180K 长上下文的可行性主要来自混合架构；
  - GDN decode 每层每 token 做一次状态递推 → 延迟敏感的小 kernel，正是软件流水线的适用域。

### 1.3 triton-turing（仓库一手信息）

- **基础版本：Triton 3.7.0**（setup.py `TRITON_VERSION = "3.7.0"`），比服务的 3.7.1 低一个 patch。
- 原理：上游 Triton 在 SM75 会发 MMA 指令，但软件流水线依赖 Ampere 专属的 `cp.async`，
  在 Turing 上缺失 → kernel 延迟暴露、Tensor Core 空转。fork 用
  `ld.global → st.shared → bar.sync` 多级流水补上（Turing 首次），`num_stages` 可配
  （甜点=2，受 64 KB/CTA 限制会静默 clamp，可用
  `TRITON_SM75_DUMP_PIPELINE_DEPTH=1` / `kernel.metadata.sm75_pipeline_slots` 观测）。
- 能力清单（Titan RTX 微基准，非本工作负载）：
  - FA2 forward：d64 +21~26% vs CUTLASS CUDA，d128 +4~8%；vs xformers SDPA 1.7~2.2×
  - FA2 backward：d64 +35~41%，d128 **-13~16%**（唯一输的地方，64KB smem 限制）
  - FP16 GEMM：≈ cuBLAS 的 84~86%（**不快过 cuBLAS**）
  - Grouped/MoE GEMM：+5~10%（2 stage 优于 3 stage）；pipeline 对 MoE +20%
  - INT8 GEMM（m8n8k16）：~1.8× cuBLAS INT8
  - **INT4 MMA（m8n8k32）：219 TOPS ≈ 2× INT8，Triton 生态首个可用纯 int4 matmul；
    cuBLAS 在 Turing 上没有 INT4 GEMM**
  - bf16 dot → fp16 Tensor Core（opt-in `TRITON_SM75_BF16_DOT_AS_F16=1`，bf16 GEMM 12~18×；
    改变数值语义，默认关）
- 安装：源码构建（cmake≥3.20、ninja、pybind11；LLVM 静态库自动下载 ~1.8GB），
  `pip install -e .` 装出的就是 `triton` 包；另有独立 `python/triton_kernels` 子包。
- 状态：社区维护（43 star），活跃（最近提交就在当天），无 tag/release wheel。

### 1.4 vLLM 0.27.1 侧的关键源码事实（本次逐一核实）

- attention backends 目录：`flash_attn.py`（SM80+ 门）、`flashinfer.py`、`flex_attention.py`、
  `gdn_attn.py`、`triton_attn.py`（paged、GQA、CUDA-graph 兼容、**FP8 KV 支持但被 SM89+ 门拦住**，
  门在 `TritonAttentionImpl.__init__`：`kv_cache_dtype.startswith("fp8") and not cap>=89 → raise`）。
  选择顺序 FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX，SM75+FP8 KV 下只剩 FlashInfer。
- GDN 层（`qwen_gdn_linear_attn.py`）：
  - **decode 走 `fused_recurrent_gated_delta_rule_packed_decode`（FLA，Triton kernel）** ← fork 直接受益
  - prefill 默认 FlashInfer `gdn_prefill`，可选 FLA Triton / cuTeDSL；本服务经 patch 强制
    `flashqla_legacy`（CUDA）← fork 不受益
- Marlin（`quantization/utils/marlin_utils.py`）：`device_capability < 75 → []`，
  **即 SM75 受支持**；AWQ（uint4+zp）与 GPTQ（uint4b8）量化类型均在列。
  → **INT4 W4A16 在本服务硬件上是现成可测路径，无需自研 kernel。**
- torch 2.13.0+cu130 依赖 triton==3.7.1（环境锁佐证）→ 装 fork 需要处理版本 pin。

---

## 2. 热路径归属：fork 到底能加速谁

| 组件 | 层数/频率 | 实现 | fork 受益？ | 备注 |
|---|---|---|---|---|
| 全注意力 prefill+decode | 16 层 | FlashInfer CUDA | ✗ | 除非换 TRITON_ATTN（§4.3） |
| GDN prefill | 48 层 | FlashQLA legacy CUDA | ✗ | 长上下文 prefill 的主力 |
| **GDN decode** | **48 层 × 每 token** | **FLA Triton 递推** | **✓** | 延迟暴露小 kernel，流水线适用域 |
| 线性 GEMM（QKV/O/MLP） | 64 层 | cuBLAS FP16（FP8 dequant 后） | ✗ | fork FP16 GEMM 只有 cuBLAS 的 84~86% |
| MTP draft（1 个小 forward） | 每步 | Triton + cuBLAS | 部分 ✓ | |
| RoPE/RMSNorm/SiLU/采样/FP8 dequant | 每层 | Triton 小 kernel | ✓ | 单个占比小，合计可观 |
| TP allreduce | 每层 | NCCL over NVLink | ✗ | |

**Roofline 核算（decode）**：27B FP8 ≈ 27 GB 权重 → TP2 每卡 13.5 GB/token；
616 GB/s → ~45 tok/s raw；实测 ~100 tok/s ⇒ MTP 有效放大 ≈ 2.2×。
**decode 已贴近 FP8 权重带宽屋顶**：编译器优化只能救"延迟暴露"的部分（GDN 递推、小 kernel），
救不了权重搬运本身。prefill 是算力受限，但主力 GEMM 在 cuBLAS、GDN prefill 在 FlashQLA，
同样绕不开 Triton。

**结论：单换编译器 = 低风险小收益（decode +0~10% / prefill +0~5% 量级）；
量级收益必须来自权重字节数（INT4）或 MTP 接受率。**

---

## 3. 接入方式（工程步骤，服务器 Linux）

### 3.1 构建与安装

```bash
# 独立目录构建，不动生产 venv
git clone https://github.com/Chennesxu/triton-turing.git /home/<user>/triton-turing-src
cd /home/<user>/triton-turing-src
git checkout <固定 commit>          # 仓库活跃更新，必须 pin

# 构建依赖（装进目标 venv 或独立构建 venv）
pip install -r python/requirements.txt   # cmake>=3.20,<4 / ninja / pybind11 / lit

# 关键：5700X 8 核，限制并行度（-j32 曾拖垮 sshd 的经验教训）
MAX_JOBS=8 pip install -e . --no-build-isolation

# 版本 pin 处理：把 setup.py 的 TRITON_VERSION 从 "3.7.0" 改成 "3.7.1+tt"
# （PEP 440：==3.7.1 可匹配 3.7.1+tt，满足 torch 2.13 的 triton==3.7.1）
# 如 vLLM 导入 triton_kernels，一并：pip install -e python/triton_kernels
```

验证：`python -c "import triton; print(triton.__version__, triton.__file__)"`
应显示 `3.7.1+tt` 且路径指向源码树。

### 3.2 版本风险（3.7.0 vs 3.7.1）

- fork 基于 3.7.0，落后服务一个 patch。3.7.1 的修复若涉及 vLLM 所用 kernel 的 codegen，
  可能出现 JIT 失败或数值差异。缓解：pin fork commit；出问题时可 cherry-pick 3.7.1 diff
  或向 fork 提 PR 追平。
- fork 活跃（当天仍有提交）→ 每次评估都记录 commit SHA。

### 3.3 Canary 与 A/B 纪律

- TP2 实例独占双卡：**8000 与 canary 8001 不能并存**（除非 canary 用单卡小模型）。
  A/B 采用停 8000 → 起 8001 → 测完恢复的维护窗口制。
- 独立 `TRITON_CACHE_DIR`（fork 与标准 triton 绝不共用编译缓存）；
  **JIT 预热完成后再计时**（首请求编译时间不得计入推理指标）；缓存目录跨重启保留。
- 每个候选先过功能关：`/health`、streaming、thinking、XML tool call、temp=0 确定性序列、
  长前缀命中、MTP draft/accepted 计数、30 分钟稳定；再谈速度。
- 速度口径：固定语料 2.8K/5.6K/8.5K/19.8K/59.2K token，128/512 completion，
  每点 ≥5 次重复取中位数（p95 需 ≥20 次）；用现有 `benchmarks/run_context_ttft.py`。
- 回退：`pip install triton==3.7.1` + 重启服务（unit 不变）。

### 3.4 观测手段

- `TRITON_SM75_DUMP_PIPELINE_DEPTH=1`：确认热点 kernel 真的拿到了 ≥2 级流水
  （注意 clamp：要求的深度 ≠ 得到的深度）。
- `nsys profile` / torch.profiler：先把 TTFT 与 decode 的时间拆成
  FlashInfer attention / FlashQLA GDN / GDN Triton / cuBLAS GEMM / NCCL / MTP / CPU gap，
  **只优化占比够大的项**（attention 若只占 20%，快 30% 也就端到端 +6%）。

---

## 4. 实验矩阵（单变量，独立设计）

### 4.1 编译器解耦组（最重要）

| ID | 编译器 | attention | KV | 回答的问题 |
|---|---|---|---|---|
| B0 | triton 3.7.1 | FlashInfer | fp8_e4m3 | 基线冻结（5 次复测） |
| **C1** | **triton-turing** | **FlashInfer（不变）** | fp8_e4m3 | **fork 经由 GDN decode/MTP/辅助 kernel 的净收益** |
| C2 | triton 3.7.1 | TRITON_ATTN | fp16 | vLLM Triton paged attention 本体水平 |
| C3 | triton-turing | TRITON_ATTN | fp16 | 软件流水线对 paged attention 的净贡献 |
| C4 | triton-turing | TRITON_ATTN | fp8_e4m3（需解除 SM89 门 + 内核内反量化） | 与生产 KV 格式对齐后的真实水平 |

判断规则：
- C1 > B0 ⇒ fork 有独立价值，保留；C1 ≈ B0 ⇒ fork 的价值只在 attention 路线，转 C2/C3。
- C3 ≤ C2 ⇒ 流水线对 vLLM paged kernel 无益，C4 不必做。
- C4 相对 B0 在目标上下文（≥19K）TTFT 中位数明确赢且 p95 不恶化 >10% 才考虑切换。
- FP16 KV 组（C2/C3）容量约为 FP8 一半，最大上下文降到 64~80K 档测，不与 180K 直接比。

### 4.2 INT4 权重组（最大潜在收益，与 fork 正交）

| ID | 权重 | 编译器 | attention | KV | 目的 |
|---|---|---|---|---|---|
| W0 | FP8（=B0） | 3.7.1 | FlashInfer | fp8_e4m3 | 基线 |
| W1 | **AWQ INT4（awq_marlin）** | 3.7.1 | FlashInfer | fp8_e4m3 | **纯权重格式收益**（带宽减半） |
| W2 | GPTQ INT4（gptq_marlin） | 3.7.1 | FlashInfer | fp8_e4m3 | AWQ vs GPTQ 打包/精度对比 |
| W3 | W1 胜者 | triton-turing | FlashInfer | fp8_e4m3 | fork 对 INT4 路径的叠加收益 |
| W4 | W1 胜者 + 更大 KV 预算/并发 | — | — | — | INT4 释放的显存怎么用（容量/并发，非单 token 速度） |
| W5 | W4A8 自定义 Triton GEMM（fork int4 MMA） | triton-turing | — | — | 第二阶段研发：activation 量化 + 挂进量化层 |

前置：确认/制作 Qwen3.8-27B 的 AWQ/GPTQ INT4 checkpoint（社区若有带 MTP head 的版本优先；
没有则用 GPTQModel/AutoAWQ 自量化，group=128）。
质量 gate：int4 对 27B 混合模型（尤其 GDN 层）的敏感度未知，必须跑业务题集
（长文摘要、数学、代码、中文工具调用）对比 FP8 基线，达标才谈速度。
预期：decode raw ~45→~85 tok/s（带宽减半，GEMM 计算开销上升会吃掉一部分），
含 MTP 约 150~190 tok/s；prefill 因权重读取减半 + Marlin 计算密集，预计 +10~30%。

### 4.3 参数组（与 fork 正交，便宜，先做）

| 变量 | sweep | 说明 |
|---|---|---|
| `max-num-batched-tokens` | 2048/4096/8192 | prefill chunk 越大 GEMM M 越大；盯峰值显存 |
| `kv-cache-memory-bytes` | 4G/5G/6G | **显式设置时 gpu-memory-utilization 不参与 KV 推导**；扩 prefix cache 驻留 |
| MTP `num_speculative_tokens` | 0/1/2/3/4/5 | 记录**分位置** acceptance；后段坍缩则不算赢 |
| `kv-cache-dtype` | fp8_e4m3（保）/ int8（精度选项）/ fp16（对照） | int8 同容量、7-bit 尾数 vs fp8 的 3-bit，长文精度更好 |
| `gpu-memory-utilization` | 0.93/0.95 | 仅在去掉 kv bytes 显式值时有效 |
| `max-num-seqs` | 1（长上下文画像）/ 2/4（吞吐画像，ctx 降到 32~64K） | 两个画像分开优化，不混报 |
| async scheduling / stream_interval | off/on；1/2/4 | 消除 GPU 空隙与 CPU 开销；验证与 MTP/GDN patch 兼容 |

### 4.4 换模型组（若业务允许）

| 方案 | 收益来源 | 代价 |
|---|---|---|
| Qwen3-30B-A3B（MoE，3B 激活） | decode 权重流量 ~9× 少；fork 的 grouped GEMM +20% 流水线直接加成 MoE prefill | 能力/长文/工具调用质量 gate；MoE 路由对 INT4 更敏感 |
| 27B INT4 单卡（TP1） | 去掉每层 allreduce；22GB 装 15GB 权重 + KV 紧张但可行 | 单卡算力减半，需实测；MTP 状态路径需验证 |
| 小模型 ×2 单卡副本 | 并发吞吐翻倍 | 单请求能力下降 |

---

## 5. 预期收益汇总（独立估计，均需实测确认）

| 措施 | decode | prefill/TTFT | 风险 | 工作量 |
|---|---|---|---|---|
| C1 只换编译器 | +0~10% | +0~5% | 低（数值漂移、版本差） | 半天 |
| 参数组（batched tokens/KV/MTP） | ±（MTP 决定） | +0~10% | 低 | 1~2 天 |
| W1 INT4（awq_marlin） | **×1.5~2** | +10~30% | 中（精度） | 1~3 天（含量化与回归） |
| C4 TRITON_ATTN+fork（fp8 KV） | ±5% | ±(0~10)%，d256 可能为负 | 中 | 2~4 天 |
| W5 自定义 W4A8 int4-MMA GEMM | 在 W1 上再 +10~30% | 同左 | 高（研发） | 1~3 周 |
| 换 MoE 模型 | ×3~5（相对 27B dense） | 视模型 | 高（能力） | 3~7 天 |

---

## 6. 风险清单

1. **数值**：MMA 重排 + 流水线改变累加顺序 → logits 微小漂移属正常；temp=0 序列
   与基线不一致时需先排除 bug 再接受。bf16-as-fp16 flag 对本服务（`--dtype half`）
   理论上不命中，但一旦误开会改变语义，环境里显式置 0。
2. **head_dim 256 在 fork FA2 实测包络之外**：任何"TRITON_ATTN 提速"的结论必须先有
   d256 微基准，再看端到端。
3. **版本漂移**：fork 3.7.0-base vs 服务 3.7.1；vLLM 0.27.1 对 triton 特性有隐式依赖
   （如 tensor descriptor 路径在 SM75 的回退行为）。pin commit + 完整功能回归。
4. **JIT/图捕获**：换编译器后首次启动重编译全部 kernel（分钟级）；CUDA graph 捕获必须
   发生在预热之后；编译缓存持久化。
5. **构建资源**：8 核机上 ninja 并行度 ≤8；LLVM 1.8GB 下载走代理/镜像。
6. **社区 fork 可持续性**：43 star、无 release。生产依赖前评估：跟上游 3.7.x 的 rebase
   成本、或把 SM75 patch 回合进标准 triton 的可行性。
7. **A/B 有效性**：双卡独占、热降频、前缀冷热都会污染数据；每个数据点记录
   GPU 温度/时钟与前缀命中状态。

## 7. Go/No-Go 判定线

- 候选保留条件（全部满足）：功能回归零失败；目标指标中位数明确赢；关键 p95 恶化 <10%；
  MTP acceptance ≥ 基线 95% 且无静默退化；30 分钟稳定无 Xid/OOM/restart；回退已演练。
- 研发项（C4 的 FP8 writer、W5 的 W4A8 GEMM）启动条件：前置对照组证明方向有正收益
  （C3>C2；W1 已赢且 profile 显示 GEMM 占 decode 大头）。

## 8. 推荐执行顺序

1. **P0**：B0 冻结 + nsys 剖析（TTFT 与 decode 各一份），拿到时间占比表。
2. **P1**：C1（只换编译器，FlashInfer 不动）——信息价值最高、风险最低的一步。
3. **P2**：参数组 sweep（batched tokens、KV bytes、MTP 0~5）——不依赖 fork。
4. **P3**：W1/W2 INT4 A/B + 质量 gate——最大潜在收益。
5. **P4**：C2/C3/C4 TRITON_ATTN 线——仅当 P0 剖析显示全注意力占比足够大才深做。
6. **P5**：W5 自定义 W4A8 int4-MMA GEMM / 专用 FA2 prefill op——仅当 P0/P3 证明值得。
7. **P6**：胜出组合 2 小时混合负载 + 3 次重启 → 生产切换（systemd unit 更新 + 回退演练）。
