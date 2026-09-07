# vLLM 双 2080 Ti（SM75）优化整体报告

> 覆盖时段：2026-09-06 17:37 ～ 2026-09-07 22:28（按 11 份源文件的**创建时间**排序还原逻辑顺序）
> 性质：**只整理数据，未修改任何线上服务**。本报告为纯汇总，所有数字均可回溯到源报告与服务器原始 JSON。

---

## 0. TL;DR（一页结论）

1. **Triton-Turing 编译器 fork：无实用价值，已回退。** 三组独立对照（C1 vs B0、W2 vs W1、W4 vs W3）端到端偏差均 ≤0.4%。根因：本服务热路径几乎全走 CUDA kernel（FlashInfer / FlashQLA / cuBLAS / CUTLASS / Marlin），fork 只能影响占比很小的 Triton kernel（GDN decode 递推、辅助小 kernel）。
2. **TRITON_ATTN 在 SM75 长上下文不可用**：XL 档 TTFT 比 FlashInfer 慢 148%，prefill 吞吐 -60%。
3. **fork 的 FA2 forward 在 head_dim=256 上完全不可用**：共享内存需求 69,632 B > Turing 64 KB 硬限制，autotune 配置空间不含 BN=16，全部 OOM。
4. **PyTorch SDPA 在 SM75 d256 无可用快速 kernel，已排除**：正确 GQA 口径下无任何快速 kernel，展开 KV 口径也只有 FlashInfer 的一半速度。
5. **FlashInfer 被确认为 SM75 d256 prefill 当前最优可用实现**（8K~59K 全程稳定 16.5~16.9 TFLOPS）。
6. **最大真实收益来自权重量化轨道：W8A8（imatrix）W1 配置 vs FP8 基线 B0，TTFT -20%~-35%，prefill +24%~+54%**（XL 档 52.98s → 35.76s）。B0 XL 重测 52.98s 与 2026-08-25 生产参考（53.02s）一致，锚定该收益为真实。
7. **KV cache 量化（FP8/INT8 KV）在固定 4 GiB KV 预算下无收益**，prefill 反而慢 2%~21%。
8. **MTP=3 是最优投机深度**：MTP=4/5 深层位置接收率坍缩（位置 4-5 仅 12%~35%），TTFT 反而慢 1%~8%。
9. **新 imatrix-MTP 模型比旧 SmoothQuant 模型 TTFT 快 8%~11%**，prefill 峰值 +11.6%，代价是 decode -13%。
10. **最终推荐配置**：Qwen3.8-27B-INT8-W8A8-imatrix-MTP + W8A8(CutlassInt8ScaledMM) + FP16 KV + 65K 上下文 + MTP3 + CUDA graph [4] + FlashInfer + FlashQLA GDN + 官方 Triton 3.7.1（即 W1 配置）；需要 180K 长上下文时用 W7（FP8 KV，TTFT 代价 2%~3%）。

---

## 1. 测试环境与基线

### 1.1 硬件与软件

| 项 | 值 |
|---|---|
| GPU | 2 × 魔改 RTX 2080 Ti **22 GB**（SM75 / CC 7.5），NVLink NV2（单条约 25.8 GB/s） |
| 显存带宽 | ~616 GB/s/卡（GDDR6 256-bit） |
| CPU / 内存 | Ryzen 7 5700X 8C16T / 32 GiB（编译并行度受限，-j32 曾拖垮 sshd） |
| 系统栈 | Ubuntu 24.04，driver 580.159.03，CUDA 13.0，Python 3.12，torch 2.13.0+cu130 |
| vLLM | 基准测试环境统一为 **0.28.0**（服务仓库日志文件名记 0.27.1，基准数据以 0.28.0 环境为准） |
| Triton | 官方 3.7.1（torch pin）；fork 为 3.7.0+git82007a85（Chennesxu/triton-turing） |
| Attention | FlashInfer 0.6.16.post3（SM75 可用；FLASH_ATTN 被 SM80 能力门拒绝） |
| GDN prefill | FlashQLA-SM70-SM75（gdn_prefill_backend=flashqla_legacy） |

### 1.2 模型

| 项 | 值 |
|---|---|
| 模型 | Qwen3.8-27B（VLM，dense，带 MTP head），FP8 权重为主 |
| 架构 | 64 层 = **16 层全注意力**（GQA 24/4，head_dim=**256**，每 4 层 1 个 FA）+ **48 层 GDN 线性注意力**（head_dim 128） |
| 推论 | 全注意力 KV 只占 1/4 层；GDN 状态固定大小（不随上下文增长）→ 180K 长上下文可行性主要来自混合架构；GDN decode 每层每 token 一次状态递推，是延迟敏感小 kernel |

### 1.3 生产基线 B0（FP8 画像）

| 参数 | 值 |
|---|---|
| 权重 / 量化 | Qwen3.8-27B FP8（MarlinFP8ScaledMM 路径） |
| Attention / GDN | FlashInfer / FlashQLA legacy |
| KV cache | fp8_e4m3，4 GiB/卡，max_model_len 180,000 |
| 并行 / 调度 | TP2，max-num-seqs=1，max-num-batched-tokens=4096，chunked prefill，prefix caching |
| 投机 / 图 | MTP=3，PIECEWISE CUDA graph capture [4] |
| 历史实测（2026-09-02，FlashInfer） | TTFT 2.62s@2.7K / 4.53s@5.4K / 6.51s@8.1K / 15.23s@19K；prefill ~1086~1298 tok/s |
| 生产参考（2026-08-25） | XL(59.2K) TTFT **53.02s / 1117 tok/s**（P2.5 用作锚点） |

---

## 2. 总时间线（按文件创建时间）

| # | 创建时间 | 文件 | 阶段 | 一句话结果 |
|---|---|---|---|---|
| 1 | 09-06 17:37 | triton-turing-integration-handoff | 初次集成交接 | 目标：fork 替换 triton + 强制 TRITON_ATTN；已完成源码克隆/LLVM 下载/FP8 限制 patch；**编译未完成**（-j32 拖垮 sshd 教训） |
| 2 | 09-06 18:41 | triton-turing-integration-rollback-summary | 初次集成实测 | 编译成功、验证版可启动，但 TTFT 全面**慢 1%~11%** → **完整回退**（triton 3.7.1 + 原 patch 撤销 + 服务恢复，18:41:09 health=200） |
| 3 | 09-06 18:58 | triton-turing-full-acceleration-matrix | 方法论升级 | 制定单变量全量试验矩阵（B0 冻结、C1-C7 编译器/backend 解耦、K0-K7 KV、Q0-Q7 量化、S0-S8 MTP、L/T 调度画像）；指出此前测试是**耦合组合**，C1（只换编译器）是关键缺失对照 |
| 4 | 09-06 19:09 | awq-int4-triton-turing-fa2-vllm0280-line | 独立试验线规划 | AWQ W4A16（awq_marlin）+ 把 fork FA2 作为自定义 TURING_FA2 prefill 算子的 A0-A6 路线；澄清 W4A16≠pure-INT4 MMA |
| 5 | 09-06 19:18 | triton-turing-independent-assessment | 独立评估（一手材料） | 热路径归属分析：fork 收益预期 decode +0~10% / prefill +0~5%；roofline：decode 已贴 FP8 权重带宽屋顶；**INT4 W4A16 经 Marlin 在 SM75 现成可测**；FA2 实测包络不含 d256 |
| 6 | 09-06 23:27 | p0-p1-benchmark-results | **P0+P1 系统基准** | C1 vs B0 持平（≤0.1%）→ fork 无独立价值；W8A8 轨道 TTFT 全面胜 W4A16；W4A16 decode 2×、显存 -30%；W6a(TRITON_ATTN) XL 慢 148% |
| 7 | 09-07 00:08 | p2-d256-fa2-microbench | P2 算子级验证 | fork FA2 d256 **全配置 OOM**（69,632>65,536 B）；FlashInfer d256 仅 11.5~16.9 TF；SDPA 正确 GQA 口径无可用快速 kernel（E1/E1b 追测） |
| 8 | 09-07 00:22 | triton-turing-final-assessment | fork 评估终结 | 结论：fork 无实用价值（架构不匹配，非质量问题）；生产 venv 回退官方 3.7.1；canary venv 与源码保留 |
| 9 | 09-07 09:40 | p25-ttft-followup-report | **P2.5 追补测量** | SDPA 路线正式排除（正确 GQA 口径无快速 kernel）；FlashInfer 确认为 d256 最优；B0 XL 重测 52.8~53.4s（与 08-25 参考 53.02s 一致）确立生产基线锚点 → W8A8 的 XL 收益待重测确认 |
| 10 | 09-07 16:27 | w8a8-benchmark-summary | **W8A8 五变体对比** | 重测 B0（XL 52.98s，锚定 08-25）后确认 **W1 真实赢 ~33% XL TTFT**；KV 量化负收益；MTP4<MTP3；fork 引擎(W9) 更差 |
| 11 | 09-07 22:28 | mtp3-vs-mtp5-ttft-benchmark | MTP 深度 + 新旧模型 | 新 imatrix-MTP 模型 TTFT 快 8~11%；**MTP=3 最优**，MTP=5 接收率 44.8%（vs 62.9%）、TTFT 慢 3~5% |

**逻辑主线**：
初次"编译器+backend 一起换"的耦合实验失败回退 → 升级为单变量矩阵 + 热路径归属分析（预判 fork 收益有限、真正杠杆是权重字节数）→ P0 验证预判（C1 持平）→ P1 整数权重双轨道（W8A8 赢 TTFT / W4A16 赢 decode）→ P2 关闭 FA2 线（d256 OOM）→ fork 退役 → P2.5 追补测量（SDPA 排除 + B0 XL 基线锚点）→ W8A8 重测坐实 W1 的 ~33% XL 收益 → MTP 深度与新旧模型收尾。

---

## 3. 分阶段过程与数据

### 阶段 1（09-06 17:37–18:41）Triton-Turing 初次集成：实测与回退

**做了什么**：克隆 triton-turing（commit 82007a85）、下载 LLVM 1.8GB、patch vLLM triton_attn.py 解除 SM75 FP8 KV 限制、限并行度编译安装 fork、让 TRITON_ATTN 复用 vLLM 原生 reshape_and_cache_flash 写 FP8 KV（验证版），180K/4GiB/TP2/MTP3 参数下成功启动并通过真实生成请求。

**初测 TTFT（同脚本、128 completion、每档 1 次）**：

| 目标 words | FlashInfer 基线 | triton-turing 验证版 | 偏差 |
|---:|---:|---:|---:|
| 2,700 | 2.62 s | 2.645 s | +0.9% |
| 5,400 | 4.53 s | 4.648 s | +2.6% |
| 8,100 | 6.51 s | 6.837 s | +5.0% |
| 19,000 | 15.23 s | 16.951 s | **+11.3%** |

**结果**：未提速 → 完整回退（triton 恢复 3.7.1、triton_attn.py 恢复 HEAD、systemd 服务恢复 enabled+active，18:41:09 health=200 且日志确认 FlashInfer）。
**教训**（直接催生阶段 2 的方法论升级）：该测试同时改了编译器 + attention backend + KV writer，是耦合组合，单次数据不能归因。

### 阶段 2（09-06 18:58–19:18）三份规划/评估：矩阵、AWQ 线、独立评估

1. **全量加速矩阵**（18:58）：B0 冻结基线；C1-C7 编译器/backend/KV 解耦组（**C1=只换编译器、FlashInfer 不动**，被点名"此前漏掉的关键对照"）；K0-K7 KV 格式；Q0-Q7 量化/并行；S0-S8 MTP sweep；L（长上下文单请求）/T（多请求吞吐）双画像。三条铁律：单变量、一时刻只加载一个 TP2 实例、先功能数值后速度。
2. **AWQ INT4 + FA2 独立线**（19:09）：A0-A6 案例；主目标是把 fork FA2 forward 作为自定义 TURING_FA2 prefill 算子嵌入（A2），TRITON_ATTN 仅作诊断对照；AWQ W4A16 走 Marlin，与 pure-INT4 MMA 严格区分。
3. **独立评估**（19:18，仅基于一手材料）：
   - 热路径归属：全注意力(16层)=FlashInfer CUDA、GDN prefill(48层)=FlashQLA CUDA、线性 GEMM=cuBLAS/Marlin —— **fork 直接受益的只有 GDN decode FLA Triton 递推 + MTP draft + 辅助小 kernel**；预期端到端 decode +0~10%、prefill +0~5%。
   - Roofline：27B FP8 → TP2 每卡 13.5 GB/token ÷ 616 GB/s ≈ **45 tok/s raw**；实测 ~100 tok/s ⇒ MTP 放大 ~2.2×，**decode 已贴 FP8 权重带宽屋顶**，编译器救不了权重搬运。
   - 关键独立发现：**vLLM Marlin 支持 SM75**（capability<75 才排除）→ INT4 W4A16 是现成可测路径，无需自研 kernel。
   - 风险提示：fork FA2 实测只覆盖 d64/d128，本模型 d256 在包络外。

### 阶段 3（09-06 23:27）P0 + P1 系统基准

**矩阵**（S/M/L/XL = 2.8K/8.5K/19.8K/59.2K tokens，每档 2 次，128 completion；D 档 = 2048 tokens 纯 decode）：

| ID | 权重 | Triton | Attention | KV | 线性 kernel（日志确认） |
|---|---|---|---|---|---|
| B0 | FP8 | 标准 3.7.1 | FlashInfer | fp8_e4m3 | MarlinFP8ScaledMM |
| C1 | FP8 | fork 3.7.0+git | FlashInfer | fp8_e4m3 | MarlinFP8ScaledMM |
| W1 | W8A8 INT8 | fork(editable) | FlashInfer | float16 | CutlassInt8ScaledMM |
| W2 | W8A8 INT8 | fork+BACKENDS_IN_TREE | FlashInfer | float16 | CutlassInt8ScaledMM |
| W3 | W4A16 AWQ | fork(editable) | FlashInfer | float16 | MarlinLinearKernel(WNA16) |
| W4 | W4A16 AWQ | fork+IN_TREE | FlashInfer | float16 | MarlinLinearKernel(WNA16) |
| W6a | W8A8 INT8 | fork+IN_TREE | **TRITON_ATTN** | float16 | CutlassInt8ScaledMM |

**TTFT（秒，均值）**：

| 档 | B0 | C1 | W1 | W2 | W3 | W4 | W6a |
|---|---:|---:|---:|---:|---:|---:|---:|
| S 2.8K | 2.59 | 2.59 | 2.08 | 2.08 | 2.29 | 2.35 | 2.28 |
| M 8.5K | 6.42 | 6.42 | 4.39 | 4.37 | 5.71 | 5.71 | 5.59 |
| L 19.8K | 14.70 | 14.70 | 9.70 | 9.69 | 13.48 | 13.45 | 15.98 |
| XL 59.2K | 52.98 † | ≈53.0 † | 35.62 | 35.65 | 48.18 | 48.00 | 88.41 |

**Prefill（tok/s，均值）**：

| 档 | B0 | W1 | W3 | W6a |
|---|---:|---:|---:|---:|
| S | 1099 | 1365 | 1244 | 1250 |
| M | 1316 | 1928 | 1480 | 1513 |
| L | 1317 | 2037 | 1467 | 1237 |
| XL | 1118 † | 1656 | 1229 | 670 |

**D 档 decode（2048 tokens，bench_d.py；该口径已废弃，decode 横向对比以 W8A8 汇总的 128-token 短生成口径为准）**：B0 49.6 / W1 37.2 / W2 37.2 / W3 76.3 / W4 78.3 / W6a ~27 tok/s。

**当时结论**：① fork 净效应 ≤0.4%（C1/W2/W4 三组对照全持平）；② 胜出轨道 W8A8（按 TTFT 优先规则），W4A16 有 decode 2× 与显存 -30% 优势；③ TRITON_ATTN 长上下文性能崩溃（XL +148%、prefill -60%）。
† B0/C1 的 XL 行为 2026-09-07 重测值（与 2026-08-25 生产参考 53.02s 一致）。

### 阶段 4（09-07 00:08–00:22）P2 d256 FA2 微基准 + fork 终结评估

**P2 微基准（单卡 2080 Ti，GQA 24/4，d=256，causal，fp16）**：

| Provider | N=1024 | N=4096 | N=16384 | 状态 |
|---|---|---|---|---|
| Triton-Turing FA2 | OOM | OOM | OOM | ❌ 全配置失败（Required 69,632 > 65,536 B；autotune 空间 BM∈[64,128]×BN∈[32,64,128] 不含 BN=16；pipeline 全被 clamp/关闭） |
| FlashInfer | 0.559 ms / 11.5 TF | 6.93 ms / 14.9 TF | 97.5 ms / 16.9 TF | ✅ 可用但不强 |
| PyTorch SDPA(mem_efficient) | 无可用 kernel | 无可用 kernel | 无可用 kernel | ❌ 排除（展开 KV 口径 8.2~8.7 TF，为 FlashInfer 一半；E1/E1b） |

**终结评估结论**：fork 技术上真实（SM75 软件流水线、INT4 MMA 等），但与本服务**架构不匹配**：d256 超出 FA2 包络且 64KB 放不下；热路径不在 Triton；decode 贴带宽屋顶；bf16-as-fp16 代理不命中（服务用 fp16）。**生产 venv 回退官方 Triton 3.7.1**（force-reinstall，health=200 验证）；canary venv 保留 fork 供未来 W4A8 路线；源码保留。

### 阶段 5（09-07 09:40）P2.5 追补测量（服务器实测，09:00–09:45 维护窗口）

**E1：SDPA 路线正式排除**。正确 GQA 口径下 torch 在 SM75 d256 无任何快速 kernel（FLASH/EFFICIENT/CUDNN 全部 "No available kernel"，只剩 math 兜底）；把 K/V 物理展开到 24 头后 mem_efficient 只有 **8.2~8.8 TFLOPS，是 FlashInfer 的一半**。

**E1 正确口径 d256 长上下文微基准（单卡，fp16，causal）**：

| 形状 | N=8192 | N=16384 | N=32768 | N=59240 |
|---|---|---|---|---|
| FlashInfer GQA 24/4 | 24.5 ms / 16.8 TF | 98.6 ms / 16.7 TF | 390 ms / 16.9 TF | 1284 ms / 16.8 TF |
| FlashInfer 12/2（TP2 per-GPU） | 12.5 ms / 16.5 TF | 49.6 ms / 16.6 TF | 199 ms / 16.6 TF | **645 ms / 16.7 TF** |
| SDPA 展开 24Q | 48.7 ms / 8.5 TF | 192 ms / 8.6 TF | 764 ms / 8.6 TF | 2489 ms / 8.7 TF |
| SDPA 展开 12Q | 25.2 ms / 8.2 TF | 98.5 ms / 8.4 TF | 387 ms / 8.5 TF | 1255 ms / 8.6 TF |

**E2/A0：生产配置忠实复刻（官方 Triton 3.7.1，FP8，FP8 KV，bt=4096）**：

| 档 | TTFT run1/run2 | prefill tok/s | 2026-08-25 生产参考 |
|---|---|---:|---|
| S 2.8K | 2.64 / 2.58 s | 1080–1101 | 2.59s ✅ |
| M 8.4K | 6.41 / 6.39 s | 1319–1322 | 6.45s ✅ |
| L 19.8K | 14.95 / 15.02 s | 1316–1323 | 14.78s ✅ |
| XL 59.2K | **52.81 / 53.42 s** | **1109–1122** | **53.02s ✅** |

四档与 08-25 参考吻合（±2%）→ B0 XL 行以本测为锚点（52.8~53.4s）；W1 的 XL=35.62s 对应 ~35% 的 XL TTFT 改善，由阶段 6 重测确认。

**XL TTFT 构成估算**：per-GPU 59K 每层 full-attention 645 ms × 16 层 ≈ 10.3 s（约 19~29%），其余大头在 GEMM/GDN/NCCL —— attention 已近天花板，继续压 TTFT 的空间在权重轨道与 GEMM。

**E2/A1：bt=8192**：S/M 与 4096 持平（2.58s/6.4s），L 档中止，尾部出现一次 HTTP 500（未排查）→ 悬而未决，不算通过也不算失败。
**遗留观察**：A1 19.7K prefill 期间 GPU0 持续 100% 而 GPU1 0%（10 秒采样），与 500 是否相关未排查。
**服务状态**：实验全部停止、双卡显存归零后拉起原服务，/health、/v1/models（max_model_len 180000）、真实生成请求均验证通过。

### 阶段 6（09-07 16:27）W8A8 五变体对比（模型：Qwen3.8-27B-INT8-W8A8-imatrix-MTP）

| 变体 | 引擎 | KV cache | max_model_len | MTP | 说明 |
|---|---|---|---|---|---|
| B0（重测） | 官方 0.28.0 | FP8 (fp8_e4m3) | 180,000 | 3 | FP8 权重生产基线 |
| **W1** | 官方 0.28.0 | FP16(默认) | 65,536 | 3 | **W8A8 推荐配置** |
| W7 | 官方 0.28.0 | FP8 (fp8_e4m3) | 180,000 | 3 | W8A8 + 长上下文 |
| W9 | **Triton-Turing fork** | INT8 (per_token_head) | 180,000 | 3 | fork 引擎对照 |
| W10 | 官方 0.28.0 | FP8 (fp8_e4m3) | 180,000 | **4** | MTP 深度对照 |

公共参数：TP2、gpu-mem-util 0.93、kv-cache-memory-bytes 4G、max-num-seqs 1、bt 4096、chunked prefill、prefix caching、FlashQLA legacy、PIECEWISE graph [4]。

> 注：B0（重测）与 W1 的 KV/上下文条件不对齐（B0=FP8 KV+180K，W1=FP16 KV+65K），横向对比主要是权重量化路径的对比；XL 档有 2026-08-25 生产参考（53.02s）锚定，可信度最高。

**TTFT（秒，均值；B0/W1/W7/W10 各 2 runs，W9 取 run3 避开冷启动异常）**：

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---|---:|---:|---:|---:|---:|
| 2.84K | 2.59 | **2.08** | 2.13 | 2.23 | 2.19 |
| 5.64K | — | — | — | 3.79 | **3.38** |
| 8.45K | 6.37 | **4.39** | 4.48 | 5.54 | 4.69 |
| 19.77K | 14.92 | **9.71** | 10.11 | — | 10.45 |
| 59.24K | **52.98** | **35.76** | 37.73 | — | 38.75 |

**Prefill（tok/s）**：

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---|---:|---:|---:|---:|---:|
| 2.84K | 1,097 | **1,365** | 1,337 | 1,277 | 1,303 |
| 5.64K | — | — | — | 1,488 | **1,671** |
| 8.45K | 1,328 | **1,928** | 1,888 | 1,526 | 1,803 |
| 19.77K | 1,325 | **2,037** | 1,956 | — | 1,892 |
| 59.24K | 1,118 | **1,656** | 1,570 | — | 1,529 |

**Decode（tok/s，128 tokens 短生成均值；KV 短、数值偏高，仅横向对比）**：

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---|---:|---:|---:|---:|---:|
| 2.84K | 98.6 | 82.3 | 76.6 | 73.6 | 77.6 |
| 8.45K | 115.1 | 79.1 | 83.7 | 62.3 | 92.8 |
| 19.77K | 90.4 | 79.4 | 66.3 | — | 92.6 |
| 59.24K | 92.4 | 66.7 | 75.6 | — | 71.5 |

**相对 B0 的提升**：

| 输入长度 | W1 TTFT | W7 TTFT | W10 TTFT | W1 Prefill | W7 Prefill | W10 Prefill |
|---|---:|---:|---:|---:|---:|---:|
| 2.84K | -20% | -18% | -15% | +24% | +22% | +19% |
| 8.45K | -31% | -30% | -26% | +45% | +42% | +36% |
| 19.77K | -35% | -32% | -30% | +54% | +48% | +43% |
| 59.24K | **-33%** | -29% | -27% | +48% | +40% | +37% |

**结论**：
1. **W1 是最优配置**：TTFT -20%~-35%、prefill +24%~+54%，不需 fork，稳定性最好。B0 重测 XL=52.98s 与 08-25 参考（53.02s）吻合，坐实了 W1 的 XL 收益是真实的 ~33%。
2. **KV 量化负收益**：FP8 KV(W7) prefill 慢 2~5%、TTFT 慢 2~3%；INT8 KV(W9) 慢 6~21% 且有冷启动异常（S 档 run1/2 达 28.5s/59.1s）。固定 4G KV 预算下省显存无实际收益。
3. **MTP=4 不如 MTP=3**：W10 各档 TTFT 慢 1~8%、prefill 低 5~7%；vLLM 警告 num_speculative_tokens>1 会对同一 MTP 层多次前向、降低接收率。
4. **fork 引擎无额外收益**：W9 全面差于 W1，且需额外设 TRITON_BACKENDS_IN_TREE=1。
5. 推荐：生产=W1；显存受限=W7；不推荐 W9/W10。

### 阶段 7（09-07 22:28）MTP=3 vs MTP=5 + 新旧模型对比

**两组模型**：A=旧 Qwen3.8-27B-SmoothQuant-W8A8-INT8；B=新 Qwen3.8-27B-INT8-W8A8-imatrix-MTP（含 MTP 权重）。
**配置**：8000 服务、TP2、fp8_e4m3 KV、180K、bt 4096、max-num-seqs 1；A-MTP3 / B-MTP3（graph [4]）/ B-MTP5（graph [6]）；上下文 2.7K~60K 词，每组 3 次（20K/60K 为 2 次），128 tokens，temp 0.6。

**TTFT（秒）**：

| 上下文 | A-MTP3 | B-MTP3 | B-MTP5 | B-MTP3 vs A | B-MTP5 vs B-MTP3 |
|---|---:|---:|---:|---:|---:|
| 2.7K | 2.23 | 2.09 | 2.19 | **+6.3%** | -4.8% |
| 5.4K | 3.60 | 3.32 | 3.42 | **+7.8%** | -3.0% |
| 8.1K | 5.02 | 4.56 | 4.75 | **+9.2%** | -4.2% |
| 10.8K | 6.58 | 5.92 | 6.07 | **+10.0%** | -2.5% |
| 13.5K | 8.14 | 7.29 | 7.52 | **+10.4%** | -3.2% |
| 16.2K | 9.88 | 8.82 | 9.09 | **+10.7%** | -3.1% |
| 20K | 12.08 | 10.97 | 11.36 | **+9.2%** | -3.6% |
| 60K | 44.79 | 41.11 | 42.25 | **+8.2%** | -2.8% |

**Prefill 峰值（tok/s）**：A-MTP3 1,726 → B-MTP3 **1,927（+11.6%）**；B-MTP5 1,867。60K 档：1,393 → 1,517（+8.9%）→ 1,476。
**Decode（tok/s，均值）**：A-MTP3 82.0 / B-MTP3 71.4（**-13%**）/ B-MTP5 80.3。

**MTP 接收率（服务日志）**：

| 配置 | 位置1 | 位置2 | 位置3 | 位置4 | 位置5 | 平均接收率 | 平均接受长度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MTP=3 | ~0.82 | ~0.57 | ~0.50 | — | — | **62.9%** | ~2.9 tokens |
| MTP=5 | ~0.80 | ~0.56 | ~0.42 | ~0.27 | ~0.21 | **44.8%** | ~3.2 tokens |

**结论**：① 新 imatrix-MTP 模型 TTFT 全程快 8~11%、prefill 峰值 +11.6%，代价是 decode -13%；② **MTP=3 是最佳搭配**——MTP=5 位置 4-5 接收率仅 12~35%，draft 算力大量浪费，TTFT 反而慢 3~5%（decode 虽略快 80.3 vs 71.4，不足以弥补）；vLLM 对 MTP>1 有官方警告。
**最优配置推荐**：imatrix-MTP 模型 + MTP=3 + graph [4]；60K TTFT 41.1s、prefill 1,517 tok/s。

---

## 4. 横向总表（跨报告合并）

### 4.1 TTFT 总对比（秒，越低越好；†=重测值）

| 变体 | 配置要点 | S ~2.8K | 5.6K | M ~8.4K | L ~19.8K | XL ~59.2K |
|---|---|---:|---:|---:|---:|---:|
| B0 | FP8 权重+FP8 KV+180K+MTP3（生产基线） | 2.59 | — | 6.42 | 14.70 | **52.98** |
| C1 | B0 + 只换 fork 编译器 | 2.59 | — | 6.42 | 14.70 | ≈53.0 † |
| **W1** | **W8A8 + FP16 KV + 65K + MTP3（推荐）** | **2.08** | — | **4.39** | **9.71** | **35.76** |
| W7 | W8A8 + FP8 KV + 180K + MTP3 | 2.13 | — | 4.48 | 10.11 | 37.73 |
| W9 | W8A8 + fork 引擎 + INT8 KV | 2.23 | 3.79 | 5.54 | — | — |
| W10 | W8A8 + FP8 KV + MTP4 | 2.19 | 3.38 | 4.69 | 10.45 | 38.75 |
| W3 | W4A16 AWQ + FP16 KV | 2.29 | — | 5.71 | 13.48 | 48.18 |
| W6a | W8A8 + TRITON_ATTN | 2.28 | — | 5.59 | 15.98 | 88.41 |
| A-MTP3 | 旧 SmoothQuant 模型（fp8 KV, 180K） | 2.23 | 3.60 | 5.02 | 12.08(20K) | 44.79(60K) |
| B-MTP3 | 新 imatrix-MTP 模型（fp8 KV, 180K） | 2.09 | 3.32 | 4.56 | 10.97(20K) | 41.11(60K) |
| B-MTP5 | 新 imatrix-MTP 模型 + MTP5 | 2.19 | 3.42 | 4.75 | 11.36(20K) | 42.25(60K) |

> 口径注：各报告档位 token 数略有差异（P0/P1：2700/8100/19000/57000 词；W8A8 另加 5400 词；MTP 报告 2700~60000 词），跨表 XL 档不完全同口径。

### 4.2 各路线最终裁决

| 优化路线 | 裁决 | 关键数据 | 来源阶段 |
|---|---|---|---|
| Triton-Turing 编译器 fork | ❌ 无价值，已回退 | 三组对照 ≤0.4%；热路径不在 Triton | P0/P1/终结评估 |
| TRITON_ATTN backend | ❌ SM75 长上下文不可用 | XL TTFT +148%、prefill -60%（670 vs 1656 tok/s） | P1 W6a |
| fork FA2 forward (d256) | ❌ 硬件不可行 | 69,632 B > 64 KB smem，全配置 OOM | P2 |
| PyTorch SDPA prefill | ❌ 排除 | 正确 GQA 口径无可用 kernel / 展开口径 8.2~8.7 TF（FlashInfer 一半） | P2.5 E1 |
| FlashInfer attention | ✅ 确认为 d256 当前最优可用 | 16.5~16.9 TF 全程稳定 | P2.5 E1 |
| W8A8 INT8 权重（imatrix） | ✅ **最大真实收益** | TTFT -20~-35%、prefill +24~+54%（XL 52.98→35.76s） | W8A8 汇总 |
| W4A16 AWQ 权重 | ⚠️ 场景备选（decode 2×、显存 -30%，TTFT 劣于 W8A8） | decode 76.3 vs 37.2（废弃口径）/ 质量未评 | P1 W3 |
| KV 量化（FP8/INT8 KV） | ❌ 固定 4G 预算下负收益 | prefill -2~-21% | W8A8 汇总 |
| MTP=4 / MTP=5 | ❌ 劣于 MTP=3 | TTFT +1~8%；接收率 44.8% vs 62.9% | W10 / MTP 报告 |
| 新 imatrix-MTP 模型 | ✅ 优于旧 SmoothQuant | TTFT +8~11%、prefill 峰值 +11.6%（decode -13%） | MTP 报告 |
| bt=8192 | ⏸ 未决 | S/M 无收益，L 中止 + HTTP 500 | P2.5 A1 |
| prefix caching 运营化 | ⏳ 待实施（零代码改动） | 命中时 XL TTFT 53s→~1s 量级 | P2.5 清单 |

---

## 5. 总体结论

1. **编译器路线（Triton-Turing fork）彻底关闭**：对本服务端到端贡献 ≤0.4%，核心卖点 FA2 在 d256 上受 64KB 共享内存硬限制不可用；已回退官方 Triton 3.7.1，fork 仅保留在 canary venv 供未来 W4A8 路线。
2. **Attention backend 路线彻底关闭**：TRITON_ATTN（长上下文 -148%）、fork FA2（OOM）、PyTorch SDPA（无可用 kernel）全部排除；**FlashInfer 是 SM75 d256 当前最优可用 prefill 实现**，且 XL 档 attention 仅占 TTFT 的 19~29%——换 attention 已无空间。
3. **权重量化是本轮唯一产生量级收益的杠杆**：W8A8（imatrix）W1 配置相比 FP8 生产基线，TTFT 缩短 20~35%、prefill 提升 24~54%（XL 52.98s→35.76s）。机理：SM75 无 FP8 Tensor Core，FP8 权重需反量化后走 FP16 GEMM，而 W8A8 可直接用 INT8 Tensor Core。
4. **KV cache 量化不要做**（固定 4 GiB 预算下）：FP8/INT8 KV 的 prefill 反而慢 2~21%，省下的显存没有转化为容量收益。
5. **投机解码深度 MTP=3 封顶**：4/5 的深层位置接收率坍缩（12~35%），TTFT 净损失 1~8%；vLLM 对 MTP>1 亦有官方警告。
6. **模型换代（imatrix-MTP）带来 8~11% TTFT 提升**，与 W8A8 轨道叠加；代价是 decode -13%（长输出场景需权衡）。
7. **W4A16（AWQ）是"短输入长输出"场景的备选**：decode 约 2× 于 W8A8、显存 -30%，但 TTFT 全面劣于 W8A8，且量化质量（thinking/tool call）尚未评测。
8. **质量回归是所有量化/换模型结论的前置条件**：本轮所有报告均为性能数据，W8A8/W4A16 的业务题集质量对比（长文摘要、数学、代码、中文工具调用）未见记录，生产切换前必须补齐。

## 6. 最终推荐配置与遗留问题

### 6.1 推荐配置（综合 W8A8 汇总 + MTP 报告）

| 项 | 推荐值 | 依据 |
|---|---|---|
| 模型 | Qwen3.8-27B-INT8-W8A8-imatrix-MTP | MTP 报告 6.3 |
| 权重 kernel | W8A8（CutlassInt8ScaledMM） | W1 |
| KV cache | FP16 + 65K 上下文（生产首选）；需 180K 长上下文时 FP8 KV（TTFT 代价 2~3%） | W1 / W7 |
| MTP | 3 | W10、B-MTP5 均劣 |
| CUDA graph | PIECEWISE [4] | 各报告一致 |
| attention / GDN | FlashInfer / FlashQLA legacy | P2.5 / 生产画像 |
| Triton | 官方 3.7.1（fork 已回退） | 终结评估 |
| 调度 | max-num-seqs=1、bt=4096、chunked prefill、prefix caching | 生产画像 |

### 6.2 遗留 / 未决事项

| 事项 | 状态 |
|---|---|
| 质量回归（W8A8/W4A16 对 thinking/tool call/长文的影响） | 未做，生产切换前置条件 |
| bt=8192 的 L/XL 档 | 中止（HTTP 500 未排查）；S/M 已证无收益 |
| A1 期间 GPU1 0% 占用观察（TP2 应双卡交替满载） | 未排查，与 500 是否相关未知 |
| W4A16 长输出场景 | 性能优势明确（decode 2×），质量待评 |
| prefix caching 运营化（固定系统提示/常用长前缀预热） | 零代码改动，命中即 TTFT 数量级下降，建议尽快落地 |
| GEMM/NCCL profiler trace（需 --profiler-config 挂载 /start_profile） | 未捕获；attention 仅 19~29%，下一个靶点未定位 |
| 服务最终状态 | 09-07 09:45 已恢复原 FP8 配置并验证；其后 W8A8/MTP 测试直接在 8000 服务上换模型进行，文件未记录 22:28 之后的最终生产配置（最新测试为 imatrix-MTP 模型） |

### 6.3 原始数据索引

本次战役的原始 JSON 随报告一并存放于各支线 `raw/` 子目录：

| 支线 | raw/ 内容 |
|---|---|
| w8a8-quantization | B0_retest / W1 / W7 / W9 / W10 基准 JSON + W1 启动脚本 |
| mtp-and-model | A-MTP3 / B-MTP3 / B-MTP5 / 20K-60K 补充 |
| attention-backend | A0 生产复刻基线、E1/E1b 微基准、W6a TRITON_ATTN 数据 |
| awq-w4a16 | W3 / W4 基准 JSON、AWQ 上下文梯度数据 |

---

*本报告由 11 份源报告按创建时间（2026-09-06 17:37 → 2026-09-07 22:28）顺序整合而成；所有数字均引自源报告，未对线上服务做任何修改。*
