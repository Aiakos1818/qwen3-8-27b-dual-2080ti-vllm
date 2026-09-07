# P2 阶段：d256 FA2 微基准测试报告

> 日期：2026-09-07
> 服务器：<user>@<server>
> 硬件：2 × RTX 2080 Ti 22GB（SM75）+ NVLink
> vLLM 版本：0.28.0
> Triton-Turing fork：3.7.0+git82007a85（editable install）
> 执行计划参照：`inbox/2026-09-06-qwen38-awq-triton-turing-sequential-execution-plan.md` §5

---

## 1. 背景与目标

P0/P1 结论显示：
- Triton-Turing fork 在当前 FlashInfer + CUDA GEMM 路径下端到端收益 ≤0.4%
- fork 的 FA2 forward 在 d64/d128 有实测收益（+21~26% / +4~8% vs CUTLASS）
- Qwen3.8-27B 全注意力层 head_dim=256，超出 fork README 基准范围
- P2 目标：算子级验证 fork FA2 在 d256 下是否可行，若可行再考虑接入 vLLM prefill 路径

## 2. 模型架构确认

```
Qwen3.8-27B (VLM, FP8):
  hidden_size: 5120
  num_attention_heads: 24
  num_key_value_heads: 4  (GQA 6:1)
  head_dim: 256  (显式设定，不是 hidden_size/num_heads)
  num_hidden_layers: 64
  layer_types: 16× full_attention + 48× linear_attention
  pattern: [LA, LA, LA, FA, LA, LA, LA, FA, ...] (每4层1个FA)
```

## 3. 共享内存约束分析

Turing SM75 硬限制：**64 KB/CTA** 共享内存（65536 B）。

FA2 forward kernel 对 d256 的共享内存需求（仅 Q/K/V tiles，不含寄存器中的 acc/m_i/l_i）：

| BLOCK_M | BLOCK_N | num_stages | smem (B) | 可用？ |
|---------|---------|------------|----------|--------|
| 32      | 16      | 1          | 32768    | ✅     |
| 32      | 16      | 2          | 49152    | ✅     |
| 32      | 16      | 3          | 65536    | ✅ (上限) |
| 64      | 16      | 1          | 49152    | ✅     |
| 64      | 16      | 2          | 65536    | ✅ (上限) |
| 64      | 32      | 1          | 65536    | ✅ (上限) |
| 128     | 任意     | 任意       | >65536   | ❌     |

**问题**：fork 的 FA2 forward autotune 配置空间为 `BM∈[64,128], BN∈[32,64,128]`，**不包含 BN=16**。autotune 扫描的所有配置都会超出 64KB 限制。

## 4. 实测结果

### 4.1 Triton-Turing FA2 Forward — ❌ 全部失败

直接调用 fork tutorial 的 `attention()` 函数（含 autotune），在 d=256 下：

```
OutOfResources: shared memory, Required: 69632, Hardware limit: 65536
```

Pipeline 深度 dump（`TRITON_SM75_DUMP_PIPELINE_DEPTH=1`）显示：

| Autotune Config | Pipeline 状态 | smem 可用 | 说明 |
|-----------------|--------------|-----------|------|
| BM=64 BN=64 ns=2 | **关闭** | 32768 B / 需要 65536 B | 其他变量占用后剩余空间不足 |
| BM=64 BN=128 ns=2 | **关闭** | 0 B / 需要 65536 B | 完全不够 |
| BM=128 BN=128 ns=2 | **关闭** | 0 B / 需要 131072 B | 远超限制 |

**最终编译出的 kernel**：69632 B > 65536 B → 运行时 OOM。

所有 N_CTX（1024 ~ 16384）均失败，无一例外。

### 4.2 FlashInfer — ✅ 可用但性能不强

GQA 24/4, d=256, causal, fp16（单卡 RTX 2080 Ti）：

| N_CTX | 时间 (ms) | TFLOPS |
|-------|----------|--------|
| 1024  | 0.559    | 11.52  |
| 2048  | 1.872    | 13.77  |
| 4096  | 6.929    | 14.88  |
| 8192  | 24.320   | 16.95  |
| 16384 | 97.525   | 16.91  |

### 4.3 PyTorch SDPA — ❌ 无可用快速 kernel（正确 GQA 形状实测）

按模型真实 GQA 形状（Hq=24, Hkv=4, d=256, causal, fp16，单卡 RTX 2080 Ti）实测：

| 形状 | N_CTX | 结果 |
|---|---|---|
| 原生 GQA (24/4) | 8192 ~ 59240 | `No available kernel`（mem_efficient 快速路径未被选中） |
| 原生 GQA (12/2, TP2) | 8192 ~ 59240 | `No available kernel` |
| GQA 展开 (96/96) | 8192 / 16384 / 32768 / 59240 | 8.6 / 8.5 / 8.8 / 8.7 TF（kernel 口径） |

GQA 展开口径需把 KV 从 4 头冗余复制成 24 头（4× 冗余计算），其吞吐低于 FlashInfer 的 16.5~16.9 TF。
结论：SM75 d256 上 SDPA 没有可用的快速 kernel，不构成 FlashInfer 的替代路径。

## 5. 分析与结论

### 5.1 Triton-Turing FA2 在 d256 上不可用

**根本原因**：fork 的 FA2 forward autotune 配置空间不含 BN=16（最小的 tile 在 N 维度是 32），导致所有可用配置在 d=256 下需要 ≥65536 B 共享内存（含 acc/m_i/l_i 等其他占用后超出 64KB 硬限制）。

即使能手动指定 BN=16 的配置，由于 tile 太小（BM=64 × BN=16），MMA 效率会极低，且 pipeline 最多只能用 2 stages（64KB 刚好填满），无法像 d64/d128 那样获得 +21~26% 的收益。

### 5.2 FlashInfer d=256 性能低于预期

FlashInfer 在 SM75 上 d=256 的 prefill 吞吐仅 11.5~16.9 TFLOPS，是 XL 档（59K context）TTFT 的主要瓶颈（B0 重测 52.98s）。

FlashInfer 的 SM75 d=256 路径缺少 tile tuning 优化——它主要针对 Ampere+ 的共享内存（≥100KB）设计。

### 5.3 P2 决策

| 决策项 | 结果 | 依据 |
|--------|------|------|
| fork FA2 d=256 可用？ | ❌ 不可用 | 共享内存 OOM（69632 > 65536） |
| 尝试 hybrid FA2 prefill 接入？ | ❌ 放弃 | 底层 kernel 不可用 |
| 考虑 PyTorch SDPA 作为 prefill backend？ | ❌ 排除 | 正确 GQA 形状下无可用快速 kernel（见 §4.3） |
| P2 FA2 专项 | **关闭** | 前置条件不满足 |

## 6. 下一步建议

### 6.1 短期（不改 fork）
1. **调查 FlashInfer d=256 SM75 性能低下的原因**——是否有 FlashInfer 配置选项可以优化 tile 大小
2. **关注 vLLM 上游对 d=256 SM75 的支持进展**——Issue #38918 显示社区已关注此问题

### 6.2 中期（需要 fork 改进）
1. **在 fork FA2 forward 中添加 BN=16 的 autotune 配置**——可能让 d=256 在 64KB 内编译通过
2. **即使编译通过，预期性能也不佳**——tile 太小，MMA 效率低；fork README 的 d128 数据也显示 pipeline 收益从 +48% 降至 +4-8%
3. **更好的方向：fork 的 bf16 dot as fp16 功能**——`TRITON_SM75_BF16_DOT_AS_F16=1` 可让 bf16 操作走 Tensor Core（12-17× 加速），但这涉及数值精度问题

### 6.3 不建议的方向
- ❌ 继续尝试在 fork FA2 上手动调参 d=256——64KB 共享内存是硬限制，tile 必须很小，性能不可能好
- ❌ P3 整数 W4A4——P2 已关闭，前置条件不满足
- ❌ 使用 fork FA2 backward——forward 都不可用，backward 更不可能（d128 backward 已需要 ~82KB）

## 7. 原始数据

```
raw/
  E1-sdpa_microbench.json            # E1: FlashInfer vs SDPA（原生 GQA 形状，SDPA 无可用 kernel）
  E1b-sdpa_expanded.json              # E1b: SDPA GQA 展开口径（8.2~8.8 TF）
  A0-benchmark.raw.json               # 生产基线锚点（B0 重测）
  W6a-benchmark.raw.json              # TRITON_ATTN 交叉数据
```

Pipeline 深度 dump 日志（stderr）：
- BM=64 BN=64 ns=2: `not pipelined; one slot needs 65536 B, 32768 B available`
- BM=64 BN=128 ns=2: `not pipelined; one slot needs 65536 B, 0 B available`
- BM=128 BN=128 ns=2: `not pipelined; one slot needs 131072 B, 0 B available`
- 最终编译: `Required: 69632, Hardware limit: 65536` → OOM
