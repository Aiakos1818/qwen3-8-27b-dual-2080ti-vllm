# 优化战役报告

本目录存放 SM75 双卡优化战役的全部报告与原始数据，按"支线"组织：每条支线一个子目录，含 README（结论先行）、原始报告，部分支线附 `raw/`（原始 JSON）。

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

## 声明

- 所有数据在上述硬件实测，不构成对其他环境的承诺。
- 模型权重 / 量化 checkpoint 不包含在本仓库。
- 部分报告引用的原始数据（`benchmarks/2026-09-06-*`、`benchmarks/2026-09-07-*`、若干顶层
  `reports/2026-09-0*.md`）产生于实验机器，**未随仓库发布**；正文已内联结论与关键表格，那些
  链接仅作溯源标注。
- W8A8 / W4A16 未做业务侧质量回归，切换前需自行评测。
- 引用与鸣谢（FlashQLA-SM70-SM75、Triton-Turing fork 等，均 MIT）：见 [docs/ACCELERATION_AND_ATTRIBUTION.md](../docs/ACCELERATION_AND_ATTRIBUTION.md)。
