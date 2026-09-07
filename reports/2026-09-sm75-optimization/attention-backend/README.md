# 支线三：Attention backend — ✅ 无空间，FlashInfer 确认为最优可用

## 结论

SM75 d256（Qwen3.8-27B 全注意力层 head_dim=256，GQA 24/4）上，三条替代路线全部实测排除：

| 路线 | 结果 | 依据 |
|---|---|---|
| TRITON_ATTN（vLLM 原生） | ❌ XL TTFT +148%（35.62→88.41s），prefill -60% | W6a 端到端 |
| Triton-Turing fork FA2 | ❌ 全 OOM：69,632 B > 65,536 B（64KB 共享内存硬限制）；autotune 空间不含 BN=16 | P2 微基准 |
| PyTorch SDPA | ❌ 无可用快速 kernel：原生 GQA 形状全部 `No available kernel`；GQA 展开口径 8.2~8.7 TF，低于 FlashInfer | E1/E1b 微基准 |

FlashInfer 是该硬件最优可用：16.5~16.9 TF（GQA 24/4，d256，causal，fp16，8K~59K 全程稳定）。

## 口径说明

- 模型真实形状是 GQA（Hq=24, Hkv=4）。用 96/96 方阵测量的是"GQA 展开"口径（KV 冗余复制 4 倍），其 8.2~8.7 TF 与 FlashInfer 的 16.5~16.9 TF 不可直接类比。
- d256 prefill 瓶颈（FlashInfer 11.5~16.9 TF）是 XL 档 TTFT 的主因；上游关注：vLLM Issue #38918。
- XL 档 attention 仅占 TTFT 的 19~29%（E1 kernel 时间外推），换 attention 已无空间。

## 文件

- [p2-d256-fa2-microbench.md](p2-d256-fa2-microbench.md) — P2 微基准报告
- [p25-ttft-followup-report.md](p25-ttft-followup-report.md) — P2.5 追补测量（口径验证 + 剩余手段验证）
- raw/ — E1/E1b 微基准 JSON、A0 生产基线锚点、W6a TRITON_ATTN 数据
