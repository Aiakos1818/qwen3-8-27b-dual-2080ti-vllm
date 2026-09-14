# Qwen3.8-27B：双 RTX 2080 Ti 22GB + NVLink 的 vLLM 部署（含 KV 优化）

> **本 fork 在基础部署之上新增 KV 优化分支**：长会话保活（keep-alive pin）/ Mamba 锚点 /
> GPU↔RAM/SSD 分块流式 offload（435K 上下文，无会话大小上限）+ 观测面板。
> 详见 [`docs/kv-optimization/`](docs/kv-optimization/README.md)。

这是一个独立的开源部署项目，面向 2 张魔改 RTX 2080 Ti 22GB、并且两卡之间已连接双 NVLink 的用户。

目标是把一套正在运行的 Qwen3.8-27B 长上下文配置完整公开：硬件、驱动、加速路径、补丁、Jinja 模板、环境变量、systemd 和完整加载参数都在这里；并额外公开面向**长上下文多会话 agent** 的 KV 缓存优化。

适合：单机双卡、单请求优先、180K / 435K 上下文、个人/小团队 API、长文档与代码任务。

不包含：模型权重、API Key、内网地址、个人目录、SSH 或隧道配置。

## 本 fork 新增

在基础部署之上，本 fork（[Aiakos1818](https://github.com/Aiakos1818)）追加了 **KV 缓存优化**，面向长上下文多会话 agent：

- **会话保活（keep-alive pin）**：结束的长会话整链 pin 出可驱逐池；准入放不下时按“缓存最小的先出”释放，回来仍全命中。
- **Mamba/GDN 锚点**：按 cadence（默认 32k）落持久状态快照，中间分叉从最近锚点续跑，最迟-K 窗口防卡死。
- **GPU↔RAM/SSD 分层 offload**：会话整链 park/restore，**分块流式**使会话大小不再受 CPU staging 限制。
- **观测面板**：Prometheus 指标 + `scripts/monitor_host_tier.py` 实时面板。

补丁 `patches/vllm-v0.27.1-kv-offload-2080ti.patch` 必须在基础补丁 `patches/vllm-v0.27.1-sm75-qwen3.8.patch` **之后**应用。完整设计与实测见
[`docs/kv-optimization/`](docs/kv-optimization/README.md)，启动 profile 见下方「KV 优化」一节。

## 已验证环境

| 分类 | 参数 |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti 22GB（22,528 MiB / 卡） |
| GPU 架构 | Turing / SM75 / Compute Capability 7.5 |
| GPU 互联 | NV2：每张卡 2 条 NVLink；单条实测约 25.781 GB/s |
| CPU | Intel Xeon E5-2696 v3，18 核 36 线程 @2.30GHz |
| 内存 | 15 GiB（swap 64 GiB） |
| OS / Kernel | Ubuntu 24.04.4 LTS / Linux 7.0.0-31-generic |
| NVIDIA Driver | 580.173.02 |
| CUDA Runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.27.2.dev0+g6e448d0ea（上游 commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac，即 v0.27.1 + 本仓库 patch） |
| Transformers / Triton | 5.16.1 / 3.7.1 |
| FlashInfer | 0.6.16.post3 |
| NCCL | 2.29.7 |

## 这套配置的目标

- TP=2：两张卡共同加载 Qwen3.8-27B。
- FP8 权重 + fp8_e4m3 KV Cache：把更多显存留给上下文。
- max-model-len=180000：服务最大上下文 180K；启动日志可用 KV Cache 约 195K tokens。
- max-num-seqs=1：优先长上下文单请求，不按高并发路线配置。
- Prefix Cache + Chunked Prefill：改善固定系统提示词和超长输入。
- MTP=3 + PIECEWISE CUDA Graph：降低部分解码与固定形状调度开销。
- flashqla_legacy：为 SM70/SM75 的 Qwen GDN prefill 提供兼容加速路径。
- Qwen3 thinking、XML tool calling、修复版 chat template：全部包含在启动配置中。

## 目录

~~~text
config/       环境变量样例（含 100K / 435K-SSD KV 优化 profile）
docs/         打补丁、加速组件和引用说明
docs/kv-optimization/  KV 优化专题（保活 / 锚点 / RAM+SSD offload / 面板）
patches/      已验证工作树导出的 vLLM / FlashQLA patch（含 KV 优化补丁）
scripts/      启动、硬件检查、KV 启动 profile、监控与验证脚本
systemd/      常驻服务模板
templates/    qwen3.8-froggeric-v22.3 Jinja 模板源文件
reports/      2026-09 优化战役报告（整体报告 + 6 条支线，含原始 JSON）
~~~

## 快速开始

### 1. 验证硬件

~~~bash
bash scripts/verify_hardware.sh
~~~

关键拓扑应包含：

~~~text
GPU0  GPU1
GPU0   X   NV2
GPU1  NV2   X
~~~

没有 NV2 也可能运行，但双卡 TP 的通信条件与本配置不同。

### 2. 获取上游源码并套用补丁

按 docs/PATCHING.md 锁定 vLLM、FlashQLA 和 FlashInfer 版本，并应用本仓库 patch。

### 3. 填写你的路径

~~~bash
cp config/vllm.env.example .env
~~~

至少修改：

~~~bash
MODEL_PATH=/你的/Qwen3.8-27B-FP8/模型目录
VLLM_PYTHON=/你的/venv/bin/python
FLASHQLA_PATH=/你的/FlashQLA-SM70-SM75
~~~

CHAT_TEMPLATE 默认指向本仓库内的 templates/qwen3.8-froggeric-v22.3.jinja。

### 4. 启动

~~~bash
bash scripts/run_qwen3.8_27b_sm75.sh
~~~

## 跑通后的性能验收

> **数据来源**：本节及下一节（含 `reports/`）的实测数据来自原项目（zyYuc）在其机器
> （AMD Ryzen 7 5700X / 32 GiB）上的运行；本 fork 的运行机器为 Intel Xeon E5-2696 v3 / 15 GiB，
> 硬件不同，下列数字仅供形态参考，未在本机复测。本 fork 自己的实测见
> [`docs/kv-optimization/`](docs/kv-optimization/README.md)。

这一步是“AI 能否真的帮你跑到相近速度”的关键，而不是只看到服务能启动。这里的首字时间严格按模型流式输出的第一个思考字符或答案字符计算；Qwen 开始输出 think 中第一个字符，就视为首字。

~~~bash
python benchmarks/run_context_ttft.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen-local \
  --word-counts 2700 5400 8100 19000 57000 \
  --runs 3 \
  --max-tokens 128 \
  --output my-benchmark-result.json
~~~

测试方法、真实流式原始结果和验收范围都在 benchmarks/README.md。下面是 2026-08-25 在当前线上服务重新跑出的上下文梯度结果；旧的 20K 合成压测表已移除，不再作为首页代表速度。

| 实际输入 | 平均首字时间（TTFT） | 平均 Prefill 速度 | 平均 Decode 速度 |
| :-- | --: | --: | --: |
| 2.84K tokens | 2.59 s | 1,099.8 tok/s | 97.3 tok/s |
| 5.64K tokens | 4.48 s | 1,260.2 tok/s | 94.3 tok/s |
| 8.45K tokens | 6.45 s | 1,311.2 tok/s | 101.1 tok/s |
| 19.77K tokens | 14.78 s | 1,337.6 tok/s | 101.3 tok/s |
| 59.24K tokens | 53.02 s | 1,117.3 tok/s | 84.4 tok/s |

首字的定义：流式 SSE 收到第一个非空 reasoning_content、reasoning 或 content 字符；Qwen 开始输出 think 内的第一个字符即计入首字。DSH 全任务平均首字应使用 DSH 的原始任务集单独复测，不能与本合成上下文梯度混用。

复现时应先锁定 docs/environment-lock.md，再按 benchmarks/README.md 的方法测试。相同硬件的合理验收范围是约 ±10%；显著偏离时按顺序检查：NVLink 是否为 NV2、TP 是否为 2、补丁是否生效、FlashQLA legacy 是否被日志选中、是否使用 FP8 KV、是否有其他 GPU 占用。

## 2026-09 性能更新：W8A8 vs FP8（同硬件、同 180K 条件）

2026-09-07 在同一台双 2080 Ti 上，把权重从 FP8 换成 W8A8（imatrix），保持 fp8_e4m3 KV + 180K 上下文不变，与上方 2026-08-25 基线同条件对比。W8A8 两列分别为 MTP3 与 MTP5。

**首字时间 TTFT（越低越好）**

| 实际输入 | FP8 + MTP3（上次基线） | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 2.59 s | **2.09 s（-19%）** | 2.19 s（-15%） |
| 5.64K tokens | 4.48 s | **3.32 s（-26%）** | 3.42 s（-24%） |
| 8.45K tokens | 6.45 s | **4.56 s（-29%）** | 4.75 s（-26%） |
| ~20K tokens | 14.78 s | **10.97 s（-26%）** | 11.35 s（-23%） |
| ~60K tokens | 53.02 s | **41.11 s（-23%）** | 42.24 s（-20%） |

**Prefill 速度（越高越好）**

| 实际输入 | FP8 + MTP3 | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 1,099.8 tok/s | **1,361（+24%）** | 1,298（+18%） |
| 5.64K tokens | 1,260.2 | **1,701（+35%）** | 1,652（+31%） |
| 8.45K tokens | 1,311.2 | **1,855（+41%）** | 1,780（+36%） |
| ~20K tokens | 1,337.6 | **1,900（+42%）** | 1,836（+37%） |
| ~60K tokens | 1,117.3 | **1,517（+36%）** | 1,476（+32%） |

**Decode 速度（128-token 流式口径，越高越好）**

| 实际输入 | FP8 + MTP3 | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 97.3 tok/s | 67.9（-30%） | 86.9（-11%） |
| 5.64K tokens | 94.3 | 67.7（-28%） | 85.5（-9%） |
| 8.45K tokens | 101.1 | 74.0（-27%） | 77.2（-24%） |
| ~20K tokens | 101.3 | 78.9（-22%） | 70.7（-30%） |
| ~60K tokens | 84.4 | 75.3（-11%） | 85.2（+1%） |

- **W8A8（imatrix）权重量化是本轮最大单项收益**：首字时间降 19~29%，prefill 升 18~42%（SM75 无 FP8 Tensor Core，FP8 权重要反量化走 FP16 GEMM，W8A8 直接走 INT8 Tensor Core）。
- **代价是 decode 略慢**（-11%~-30%，128-token 口径）；MTP5 的 decode 更接近 FP8 基线。长输出优先的场景可看 W4A16（decode 约 2×，TTFT 劣于 W8A8，见 reports）。
- **MTP3 是甜点位**：MTP5 深层位置接收率坍缩（平均 44.8% vs 62.9%），不建议。
- 若能接受 65K 短上下文 + FP16 KV，W8A8+MTP3 的 ~60K 首字时间进一步降到 **35.76 s（-33%）**。
- 机理、全部变体数据、被排除的路线（TRITON_ATTN / FA2 d256 / SDPA / Triton-Turing fork）见 [reports/2026-09-sm75-optimization/](reports/2026-09-sm75-optimization/00-consolidated-report.md)。
- **W8A8 未做业务侧质量回归，切换前请先评测。**

## KV 优化：长会话保活 / Mamba 锚点 / GPU↔SSD 分层 offload

上面是**基础部署**（180K 单请求）。在其之上，本仓库还公开一套面向**长上下文多会话 agent**
场景的 KV 缓存优化分支，补丁为 `patches/vllm-v0.27.1-kv-offload-2080ti.patch`
（在基础补丁 `vllm-v0.27.1-sm75-qwen3.8.patch` **之后**应用）。

解决的问题：`max-num-seqs=1` 下，前缀缓存池满时旧会话整链被驱逐，下次重发**全量重算**；
从历史中间 revert/截断重发也会整段重算。

| 机制 | 作用 | 文档 |
| --- | --- | --- |
| **会话保活（keep-alive pin）** | 结束且 ≥16k token 的会话整链 pin 出可驱逐池；准入放不下时按“缓存最小的先出”释放，回来仍全命中。 | [vllm_01_保活](docs/kv-optimization/vllm_01_保活.md) |
| **Mamba/GDN 锚点** | 按 cadence（默认 32k）落持久状态快照；中间分叉从最近锚点续跑，最迟-K 窗口防卡死。 | [vllm_02_锚点](docs/kv-optimization/vllm_02_锚点.md) |
| **GPU↔RAM/SSD 分层 offload** | 会话整链 park 到 RAM 或 SSD，回来 restore；**分块流式**使会话大小不再受 CPU staging 限制。 | [vllm_03 RAM](docs/kv-optimization/vllm_03_offload_ram.md) · [vllm_04 SSD](docs/kv-optimization/vllm_04_offload_ssd.md) |
| **观测面板** | Prometheus 指标 + `scripts/monitor_host_tier.py` 实时面板（保活 / 锚点 / RAM+SSD 池 / I/O）。 | [vllm_05 面板](docs/kv-optimization/vllm_05_kv信息面板.md) |

启动 profile：

~~~bash
# 100K 池，小尺度实验台（RAM parking；SSD 可选）
bash scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh

# 435K 上下文，SSD-only 两层 offload（分块流式，无会话大小上限）
cp config/vllm-435k-ssd.env.example .env   # 填 MODEL_PATH / VLLM_PYTHON / VLLM_SSD_ROOT
bash scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh
~~~

实测（2×RTX 2080Ti 22GB，TP=2，AWQ-INT4 + fp8_e4m3 KV，KV 池 9e9 B → 489,789 tokens）：

- 435K 会话（281 slot ≈ 13.8 GiB，9 个 chunk）park→resume：**cached=390,400，sha 一致，0 NaN**；
  真 NVMe 写 **27.1 GiB** / 读 **13.3 GiB**，写 0.93 GiB/s、读 1.68 GiB/s。
- 保活开启后，55k 会话被挤压仍 **52,800/52,800 全命中**（旧行为为 0/全量重算）。
- 深回退从锚点续跑：72k 截断 cached=64,000（整段重算需 43s）。
- 写中 SIGKILL 原子性、分块流式功能矩阵 17/17（1 soft）通过。

复现/验证脚本：`scripts/monitor_host_tier.py`、`scripts/ssd_matrix.py`、
`scripts/ssd_100k_check.py`、`scripts/ssd_435k_check.py`、`scripts/ssd_crash_check.py`、
`scripts/correctness_check.py`。开发期定向探测脚本（`probe_*` / `revert_*` / `resident_*` /
`test_kv_100k.py` / `offload_matrix.py` / `capture_proxy.py` 等）同样收录于 `scripts/`，
路径与模型经 `MODEL_PATH` / `VLLM_BASE_URL` / `VLLM_PYTHON` 参数化。
指标与 env 总表见 [vllm_05 面板](docs/kv-optimization/vllm_05_kv信息面板.md)。

> 文档中的字节一律 1024（`GiB`/`MiB/s`），token 一律 1000（`k=1000`）。

## 完整加载参数与用途

| 参数 | 当前值 | 用途 |
| --- | --- | --- |
| --dtype | half | 运行时 FP16 计算 dtype。 |
| --tensor-parallel-size | 2 | 两张 GPU 做 Tensor Parallel。 |
| --device-ids | 0,1 | 明确使用 GPU 0、1。 |
| --quantization | fp8 | FP8 权重加载。 |
| --kv-cache-dtype | fp8_e4m3 | 用 FP8 E4M3 存 KV Cache，降低 KV 显存。 |
| --max-model-len | 180000 | 单请求上下文上限。 |
| --gpu-memory-utilization | 0.93 | vLLM 目标使用每卡 93% 显存。 |
| --kv-cache-memory-bytes | 4G | 显式限制 KV Cache 显存预算。 |
| --max-num-seqs | 1 | 单并发、长上下文优先。 |
| --max-num-batched-tokens | 4096 | 限制单轮调度 token，平衡峰值显存与延迟。 |
| --enable-prefix-caching | 开启 | 缓存重复系统提示词和前缀。 |
| --enable-chunked-prefill | 开启 | 长输入分块 prefill。 |
| --no-async-scheduling | 开启 | 关闭异步调度，保持这套兼容路径。 |
| --speculative-config | mtp / 3 | 每步最多预测 3 个 token。 |
| --compilation-config | PIECEWISE / [4] | 只捕获 size=4 CUDA Graph。 |
| --additional-config | flashqla_legacy | SM75 GDN prefill 后端。 |
| --reasoning-parser | qwen3 | Qwen3 thinking 输出解析。 |
| --tool-call-parser | qwen3_xml | Qwen3 XML tool calling 解析。 |
| --chat-template | qwen3.8-froggeric-v22.3.jinja | 使用本仓库修复模板。 |
| --override-generation-config | T=0.6，top_p=0.95，top_k=20，repetition=1.06 | 默认采样参数。 |

## 环境变量与加速路径

| 环境变量 | 值 | 作用 |
| --- | --- | --- |
| OMP_NUM_THREADS | 8 | 本机 Xeon E5-2696 v3 为 18 核 36 线程，取 8 以控制线程开销。 |
| VLLM_USE_DEEP_GEMM | 0 | 关闭此配置未使用的 DeepGEMM 路径。 |
| VLLM_USE_FLASHINFER_SAMPLER | 0 | 关闭 FlashInfer top-k/top-p sampler。 |
| VLLM_QWOPUS_MTP_BF16_DRAFT | 1 | Qwen3.5 MTP draft 层兼容设置。 |
| VLLM_SM75_SPEC_SYNC_MODE | safe | SM75 speculative decoding 保守同步模式。 |
| VLLM_USE_V2_MODEL_RUNNER | 1 | 使用 V2 model runner。 |
| PYTHONPATH | FLASHQLA_PATH | 让 vLLM 能导入 FlashQLA SM75 GDN backend。 |

完整来源、固定 commit、补丁边界和引用见 docs/ACCELERATION_AND_ATTRIBUTION.md。完整环境锁定清单见 docs/environment-lock.md。

## systemd 常驻服务

1. 把仓库放到 /opt/qwen3-8-27b-dual-2080ti-vllm，或修改 unit 中路径。
2. 把 config/vllm.env.example 复制为 /etc/qwen3.8-vllm.env 并填写路径。
3. 修改 systemd/qwen3.8-27b-vllm.service.example 中的 Linux 用户。
4. 安装并启动：

~~~bash
sudo cp systemd/qwen3.8-27b-vllm.service.example /etc/systemd/system/qwen3.8-27b-vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3.8-27b-vllm
sudo systemctl status qwen3.8-27b-vllm --no-pager
~~~

## 重要说明

- 模型权重不在本仓库内。请从拥有相应许可的来源获取模型。
- 这是 SM75 / 2080 Ti 的特化配置，不能把它当作 H100、4090、A100 或无 NVLink 双卡的通用最优参数。
- FP8 权重和 FP8 KV Cache 需要自行做业务精度回归，尤其是长上下文、数学、代码和工具调用。W8A8 / W4A16 量化 checkpoint 同理（见 reports/2026-09-sm75-optimization/）。
- 本仓库只公开部署配置和已导出的本地补丁；上游组件遵循各自许可证。

## 引用与致谢

感谢并请引用：vLLM、PyTorch、Hugging Face Transformers、FlashInfer、FlashQLA-SM70-SM75、NCCL、Triton-Turing（SM75 fork，reports 战役引用）。详细链接、commit 和许可证在 docs/ACCELERATION_AND_ATTRIBUTION.md。

本 fork 在其上追加了 **KV 优化分支**（会话保活 / Mamba 锚点 / GPU↔RAM/SSD 分层 offload 与观测面板，见 `docs/kv-optimization/` 与 `patches/vllm-v0.27.1-kv-offload-2080ti.patch`），由 [Aiakos1818](https://github.com/Aiakos1818) 贡献，按本仓库 MIT 许可发布。
