# vLLM KV 优化文档索引

本目录是 [qwen3-8-27b-dual-2080ti-vllm](../../README.md) 部署项目的 **KV 缓存优化**专题：
面向 Qwen3.8-27B（混合 GDN/Mamba，2×RTX 2080Ti 22GB，TP=2）在**长上下文多会话 agent 场景**
下的会话保活、Mamba 锚点、RAM/SSD 分层 offload，以及观测面板。

对应补丁：`../../patches/vllm-v0.27.1-kv-offload-2080ti.patch`
（在 `patches/vllm-v0.27.1-sm75-qwen3.8.patch` 基础补丁之上应用）。

## 单位约定

- **字节/比特一律 1024**（IEC）：显存、RAM、KV 池、staging、quota、磁盘、I/O 量、带宽
  一律 `KiB/MiB/GiB`、`MiB/s`；换算 `1 GiB = 2^30 B`。
- **token/上下文一律 1000**：`k = 1000`；`--max-model-len`、池容量等精确值给整数，
  历史 `×1024` 命名加注（如 `102400 = 100×1024 = 102.4k`）。
- 块数/slot/条数：无单位，十进制。

> 说明：vLLM 上游（`mem_constants`、`format_gib`、nvidia-smi）本就是 IEC；自研 host-tier
> 的限速/日志与面板已统一到 IEC。env/CLI 的原始字节整数（如 `9000000000`）不改值，仅显示换算。

## 专题文档

| 文档 | 内容 |
|---|---|
| [vllm_01_保活](vllm_01_保活.md) | 前缀缓存为何在池满时整链失效；自动 keep-alive pin 整链、准入压力“缓存最小的先出”释放；pin 空闲块的 adoption 守卫。 |
| [vllm_02_锚点](vllm_02_锚点.md) | revert/截断重发（从历史中间分叉）整段重算的根因；按 cadence（32k）落 Mamba 持久快照，最迟-K 窗口、同等保护、容量/卡死口径、512k 规划。 |
| [vllm_03_offload_ram](vllm_03_offload_ram.md) | 会话级 park/spill 到 RAM：语义、架构、连接器只作拷贝通道、容量对照、实测；**跨层两档驱逐策略的权威描述（§9）**。 |
| [vllm_04_offload_ssd](vllm_04_offload_ssd.md) | 两层 GPU↔SSD + 分块流式（无会话大小上限）：架构、配置、实现不变式、435k 真 NVMe 验收、bug 修复、限制与复现。 |
| [vllm_05_kv信息面板](vllm_05_kv信息面板.md) | Prometheus 指标全集、`scripts/monitor_host_tier.py` 实时面板、`RAMTRACE` 轨迹、env 配置总表。 |

## 知识参考项

| 文档 | 内容 |
|---|---|
| [GPU 显存计算](GPU_MEMORY_CALCULATION.md) | Qwen3.8-27B 双 2080Ti 的显存/KV 池精确计算：层结构、page_size、混合分组、KV 需求公式与验证、参数关系、预算表、OOM 案例、推荐配置。KV 池容量与 `--kv-cache-memory-bytes` 标定参考。 |

## 相关入口

- 启动脚本：`scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh`、`scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh`
- FP8 权重 profile：`scripts/run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh`（见下）
- 观测面板：`scripts/monitor_host_tier.py`
- 验证脚本：`scripts/ssd_matrix.py`、`scripts/ssd_100k_check.py`、`scripts/ssd_435k_check.py`、`scripts/ssd_crash_check.py`、`scripts/correctness_check.py`
- 环境变量样例：`config/vllm-435k-ssd.env.example`、`config/vllm-fp8-100k.env.example`
- 打补丁步骤：`docs/PATCHING.md`

## 与权重量化无关（FP8 实测）

KV 优化操作的是 KV 块，与权重是 AWQ-INT4 / FP8 无关。FP8 权重 100K 全流程实测
（RAM + SSD）见 [`reports/2026-09-fp8-kv/`](../../reports/2026-09-fp8-kv/)。切换量化类型
只需：改 `--quantization` + 模型路径、重标定 `KV_CACHE_MEMORY_BYTES`（权重变大）、把
venv 的 `ninja` 放进 PATH（FP8 启用 `norm_quant`/`act_quant` 融合）。唯一行为差异：
restore 后的深回退锚点命中（见报告 §5）。

> 文档中按名称提到的 `probe_*` / `revert_*` / `resident_*` 等脚本为开发期定向探测，
> 未随本仓库发布；结论已全部写入上述专题文档。
