# 优化战役报告

本目录存放 SM75 双卡优化战役的全部报告与原始数据，按"支线"组织：每条支线一个子目录，含 README（结论先行）、原始报告与 `raw/`（原始 JSON）。

## 2026-09-sm75-optimization（2026-09-06 ~ 09-07）

环境：2× RTX 2080 Ti 22GB（SM75）+ NVLink，vLLM 0.28.0，同一 27B 基座模型（不同量化 checkpoint）。

| 支线 | 状态 | 一句话结论 |
|---|---|---|
| [w8a8-quantization](2026-09-sm75-optimization/w8a8-quantization/) | ✅ 推荐 | W8A8(imatrix) vs FP8：TTFT -19%~-35%，prefill +24%~+54% |
| [mtp-and-model](2026-09-sm75-optimization/mtp-and-model/) | ✅ MTP3 | 新 imatrix-MTP 模型比旧 SmoothQuant 快 8~11%；MTP3 接收率最高，MTP5 深层坍缩 |
| [attention-backend](2026-09-sm75-optimization/attention-backend/) | ✅ 无空间 | TRITON_ATTN / FA2(d256) / SDPA 全排除，FlashInfer 为最优可用 |
| [triton-turing-fork](2026-09-sm75-optimization/triton-turing-fork/) | 🔴 OPEN（移交） | 端到端 ≤0.4% 无加速；架构错配（d256 超 64KB 共享内存），移交社区 |
| [awq-w4a16](2026-09-sm75-optimization/awq-w4a16/) | ⚠️ 备选 | decode ~2×、显存 -30%，TTFT 劣于 W8A8，质量待评 |
| [int8-kvcache](2026-09-sm75-optimization/int8-kvcache/) | ⚠️ 基本未测 | 仅 W9（per_token_head）部分数据，格式空间未展开 |

入口：[00-consolidated-report.md](2026-09-sm75-optimization/00-consolidated-report.md)（整体时间线 + 跨支线总表）、[p0-p1-systematic-benchmark.md](2026-09-sm75-optimization/p0-p1-systematic-benchmark.md)（P0+P1 系统基准）。

## 2026-09-fp8-kv（2026-09-12）

[FP8 权重 × KV 优化 100K 实测](2026-09-fp8-kv/)。验证 KV 优化分支（保活 / Mamba 锚点 /
GPU↔RAM/SSD 分层 offload）在 **FP8 权重**（block-wise dynamic e4m3）上照常工作。

| 检查 | 结果 |
|---|---|
| 权重显存 / KV 容量 | 14.96 GiB/卡；`GPU KV cache size 106,288`（与 AWQ 同池同值） |
| RAM restore 正确性 | sha `db8b8e836881534b` 与 baseline 一致，0 NaN |
| RAM offload 矩阵 | `spills=8 restores=4 evictions=2 drops=2`（与 AWQ 一致） |
| SSD 真盘 park/resume | `R cached=81600 / 10.7s`，sha 一致，写 9.34 GiB / 读 2.85 GiB |
| SSD 强制分块矩阵 | **18/18 PASS**；`S3 restore + 深回退 keep2=30400 keep3=46400`、`S3b 二次停车 keep2=30400` |
| 单元测试 | 139 passed（含新增回归测试与真 LRU 测试） |

结论：**KV 优化与权重量化无关**，切换只需改 `--quantization`、重标定 KV 池、把 `ninja` 放进 PATH。

测试中发现并修复了三个锚点问题（报告 §5）：

1. MTP 的 `use_eagle()` 让 full-attention finder 多丢一个 block，使 cadence 锚点不可达
   → 每个 cadence 同时保留 `C` 与 `C - block_size` 两个锚点，常驻深回退 `keep=2` 由
   28800 提升到 **30400**。
2. 基类 head-free 在 mamba 保留逻辑之前释放 head 区间，pre-cadence 状态（`C - block_size`）
   在被 pin 前就被置 null → 覆写 `MambaManager._remove_blocks_in_range` 保留 durable boundary
   块。restore 后深回退由 `cached=0` 修复为 **`cached=30400`**（RAM 与真 NVMe SSD 均验证）。
3. 恢复会话的前缀命中只认领恢复点一个状态块，旧锚点成为无主闲置块 → 首次缓存时
   `_adopt_cached_durable_anchors` 把缓存中仍在的 cadence 锚点重新认领进窗口/保活 entry。
   二次停车后深回退由 `cached=0` 修复为 **`cached=30400`**（`p03 test2`）。

## 2026-09-435k-kv（2026-09-13）

[435k / 512k KV 复验报告](2026-09-435k-kv/)。三个锚点修复落地后，用最终代码在真 NVMe SSD 上
复跑 AWQ-INT4 长上下文：

| 检查 | 结果 |
|---|---|
| 435k 常驻深回退 | `V0 cached=352000` ✅ |
| 435k SSD 恢复 + 深回退 | `R cached=384000`（sha 一致）→ `V2 cached=352000` ✅ |
| 512k 满长 prefill（MTP3） | 499,018 token，无 OOM ✅ |
| 512k SSD 分块恢复 | `R cached=496000`（99.4%，sha 一致，45.7s vs 重算 1490.7s）✅ |
| 512k 深回退（MTP3） | `V cached=480000` ✅（MTP1/515k 贴近上限时会 miss，见报告）|

结论：435k 与 512k（MTP3）三个修复端到端全部 PASS；cadence 现自动向下对齐到 block_size，
用户无需知道 block_size。

## 声明

- 所有数据在上述硬件实测，不构成对其他环境的承诺。
- 模型权重 / 量化 checkpoint 不包含在本仓库。
- W8A8 / W4A16 未做业务侧质量回归，切换前需自行评测。
- 引用与鸣谢（FlashQLA-SM70-SM75、Triton-Turing fork 等，均 MIT）：见 [docs/ACCELERATION_AND_ATTRIBUTION.md](../docs/ACCELERATION_AND_ATTRIBUTION.md)。
