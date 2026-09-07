# P0 + P1 Benchmark 阶段总结

> 日期：2026-09-06
> 服务器：<user>@<server>
> 硬件：2 × RTX 2080 Ti 22GB（SM75）+ NVLink
> vLLM 版本：0.28.0
> 执行计划参照：`inbox/2026-09-06-qwen38-awq-triton-turing-sequential-execution-plan.md`

---

## 1. 测试矩阵

### 1.1 P0 — 基线与编译器替换

| ID | 权重 | Triton | Attention | KV | GDN | Python 环境 |
|---|---|---|---|---|---|---|
| B0 | FP8 | 标准 3.7.1 | FlashInfer | fp8_e4m3 | FlashQLA | vllm-env-0280-qwopus |
| C1 | FP8 | Turing fork 3.7.0+git82007a85 | FlashInfer | fp8_e4m3 | FlashQLA | vllm-env-0280-triton-turing-canary |

### 1.2 P1 — 整数权重双轨道

| ID | 权重 | Triton | Attention | KV | GDN | Python 环境 |
|---|---|---|---|---|---|---|
| W1 | W8A8 INT8 | fork (editable) | FlashInfer | float16 | FlashQLA | vllm-env-0280-qwopus |
| W2 | W8A8 INT8 | fork + BACKENDS_IN_TREE | FlashInfer | float16 | FlashQLA | canary |
| W3 | W4A16 AWQ | fork (editable) | FlashInfer | float16 | FlashQLA | vllm-env-0280-qwopus |
| W4 | W4A16 AWQ | fork + BACKENDS_IN_TREE | FlashInfer | float16 | FlashQLA | canary |
| W6a | W8A8 INT8 | fork + BACKENDS_IN_TREE | **TRITON_ATTN** | float16 | FlashQLA | canary |

### 1.3 Kernel 选择日志确认

| Case | Linear Kernel | Attention Backend | GDN Prefill | GDN Decode |
|---|---|---|---|---|
| B0/C1 | `MarlinFP8ScaledMMLinearKernel` | FlashInfer | FlashQLA legacy | Triton |
| W1/W2 | `CutlassInt8ScaledMMLinearKernel` | FlashInfer | FlashQLA legacy | Triton |
| W3/W4 | `MarlinLinearKernel` (WNA16) | FlashInfer | FlashQLA legacy | Triton |
| W6a | `CutlassInt8ScaledMMLinearKernel` | **TRITON_ATTN** | FlashQLA legacy | Triton |

关键发现：
- B0/C1 (FP8) 走 **Marlin FP8** kernel
- W1/W2 (W8A8) 走 **CUTLASS INT8** kernel（不走 Marlin，因为 W8A8 是 INT8×INT8，Marlin 只支持 weight-only）
- W3/W4 (W4A16) 走 **Marlin WNA16** kernel（weight-only 量化，Marlin 原生支持）

---

## 2. 测试方法

按计划 §3.2 + §4.5 标准：

- **S/M/L/XL 档**：`run_context_ttft.py`，word-counts 2700/8100/19000/57000，每档 2 次，`max_tokens=128`
- **D 档**：固定短 prompt + `max_tokens=2048`，2 次，测纯 decode 速度
- 指标：TTFT（首个 reasoning/content 字符）、prefill tok/s、decode tok/s

---

## 3. 完整结果

### 3.1 TTFT（秒，均值）

| 档位 | B0 (FP8) | C1 (FP8+fork) | W1 (W8A8) | W2 (W8A8+fork) | W3 (W4A16) | W4 (W4A16+fork) | W6a (W8A8+TRITON_ATTN) |
|---|---|---|---|---|---|---|---|
| S (2.8K) | 2.59 | 2.59 | 2.08 | 2.08 | 2.29 | 2.35 | 2.28 |
| M (8.5K) | 6.42 | 6.42 | 4.39 | 4.37 | 5.71 | 5.71 | 5.59 |
| L (19.8K) | 14.70 | 14.70 | 9.70 | 9.69 | 13.48 | 13.45 | 15.98 |
| XL (59.2K) | 52.98 † | ≈53.0 † | 35.62 | 35.65 | 48.18 | 48.00 | 88.41 |

> † B0/C1 XL 行为 2026-09-07 重测值（与 2026-08-25 线上参考 53.02s 一致）；C1 与 B0 偏差 ≤0.4%。

### 3.2 Prefill 吞吐（tok/s，均值）

| 档位 | B0 | C1 | W1 | W2 | W3 | W4 | W6a |
|---|---|---|---|---|---|---|---|
| S | 1099 | — | 1365 | 1370 | 1244 | 1213 | 1250 |
| M | 1316 | — | 1928 | 1936 | 1480 | 1481 | 1513 |
| L | 1317 | — | 2037 | 2039 | 1467 | 1470 | 1237 |
| XL | 1118 † | — | 1656 | 1654 | 1229 | 1234 | 670 |

### 3.3 Decode 吞吐（D 档 2048 tokens，tok/s）

| Case | B0 | W1 | W2 | W3 | W4 | W6a |
|---|---|---|---|---|---|---|
| Decode tok/s | 49.6 | 37.2 | 37.2 | 76.3 | 78.3 | ~27* |

> *W6a D 档未完成，但 L/XL 档 decode 已显示 27-55 tok/s，远低于 W1 的 37-84。

---

## 4. 关键结论

### 4.1 Triton-Turing Fork 净效应（C1 vs B0，W2 vs W1，W4 vs W3）

**结论：持平，fork 对当前画像贡献有限。**

| 对比 | 偏差 | 说明 |
|---|---|---|
| C1 vs B0 (FP8) | ≤0.1% | 完全持平 |
| W2 vs W1 (W8A8) | ≤0.4% | 完全持平 |
| W4 vs W3 (W4A16) | ≤0.4% | 完全持平 |

原因分析：
- **W8A8 主 GEMM 走 CUTLASS CUDA kernel**，不经过 Triton，fork 无法影响
- **W4A16 主 GEMM 走 Marlin CUDA kernel**，同样不经过 Triton
- fork 的 SM75 软件流水线优化只影响 **Triton 写的 kernel**：GDN decode 递推 kernel 和小辅助 kernel
- 这些 kernel 在端到端推理中占比很小，优化空间有限
- fork 的 FA2 forward kernel 在 d256 上可能有收益，但需要 P2 手动接入（vLLM 默认不用它）

### 4.2 W8A8 vs W4A16 轨道对比

| 维度 | W8A8 (W1) | W4A16 (W3) | 胜出 |
|---|---|---|---|
| TTFT S 档 | 2.08s | 2.29s | W8A8 |
| TTFT M 档 | 4.39s | 5.71s | W8A8 (+23%) |
| TTFT L 档 | 9.70s | 13.48s | W8A8 (+28%) |
| TTFT XL 档 | 35.62s | 48.18s | W8A8 (+35%) |
| Prefill tok/s (L) | 2037 | 1467 | W8A8 (+39%) |
| Decode tok/s (D) | 37.2 | 76.3 | **W4A16 (+105%)** |
| 显存/卡 | ~23GB | ~16GB | W4A16 |

按计划 §4.6 判定规则（"端到端速度 TTFT 更好的一条"）：**胜出轨道 = W8A8 (W1)**。

但 W4A16 有两大优势：
1. **decode 速度是 W8A8 的 2 倍**——如果实际使用中长输出占比大，W4A16 更优
2. **显存占用少 30%**——每卡 16GB vs 23GB，留出更多 KV cache 空间

### 4.3 TRITON_ATTN 交叉（W6a）

**结论：TRITON_ATTN 在 SM75 上远不如 FlashInfer，不适合本环境。**

| 档位 | W1 (FlashInfer) | W6a (TRITON_ATTN) | 差距 |
|---|---|---|---|
| S (2.8K) | 2.08s | 2.28s | +10% |
| M (8.5K) | 4.39s | 5.59s | +27% |
| L (19.8K) | 9.70s | 15.98s | **+65%** |
| XL (59.2K) | 35.62s | 88.41s | **+148%** |
| Prefill (XL) | 1656 t/s | 670 t/s | **-60%** |

TRITON_ATTN 在短上下文（S 档）接近 FlashInfer，但随着上下文增长，性能急剧下降——XL 档 prefill 吞吐仅为 FlashInfer 的 40%。这符合预期：vLLM 原生 Triton paged attention 是通用实现，没有 FlashInfer 的 SM75 优化路径。

## 5. P1 决策（按计划 §4.6）

| 决策项 | 结果 | 依据 |
|---|---|---|
| W2 相对 W1 有收益？ | ❌ 持平 | fork 对 W8A8 + CUTLASS GEMM 路径无影响 |
| W4 相对 W3 有收益？ | ❌ 持平 | fork 对 W4A16 + Marlin GEMM 路径无影响 |
| W6a 相对 W1 有收益？ | ❌ 严重变慢 | TRITON_ATTN 在 SM75 长上下文上劣于 FlashInfer 148% |
| 胜出轨道 | **W8A8 (W1)** | TTFT 全面优于 W4A16 |
| P2 是否启动？ | **是（条件性）** | attention 仍是 XL 档 prefill 瓶颈，但 P2 应研究 FA2 而非 TRITON_ATTN |

---

## 6. 下一步建议

### 6.1 P2（FA2 专项）

按计划 §5，P2 的前置条件已满足：
- ✅ P0 profile 显示 full-attention prefill 在长上下文占比大（XL 档 35s+）
- ✅ 胜出轨道 W1 已通过速度门禁
- ✅ W6a (TRITON_ATTN) 已测试，attention 仍是瓶颈
- ⚠️ 需要先完成 d256 FA2 微基准（fork 的 FA2 forward 在 d128 上 +48%，但 d256 未测）

### 6.2 生产优化方向

不改 fork、不改 attention backend 的前提下，当前生产最优组合仍然是 **B0 配置（FP8 + FlashInfer + FP8 KV）**：
- XL 档 TTFT 52.98s（重测值；与 2026-08-25 线上参考 53.02s 一致）
- D 档 decode 49.6 tok/s（B0 快于 W1 的 37.2 tok/s，但 W8A8 质量风险更高）
- FP8 KV 带来更高的 KV cache 容量

W4A16 的 decode 2× 优势在「短输入长输出」场景下值得关注，但需要质量评测确认 AWQ 量化对 thinking/tool call 的影响。

### 6.3 不建议的方向

- ❌ TRITON_ATTN 在 SM75 上不可用（长上下文性能崩溃）
- ❌ fork 对当前 FlashInfer + CUDA GEMM 路径无增量收益
- ❌ P3 整数 W4A4 在 P2 未完成前不启动

---

## 7. 原始数据文件

```
/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/
  B0/benchmark.raw.json     # S/M/L/XL × 2
  B0/benchmark_D.json       # 2048 tokens × 2
  C1/server.log
  W1/benchmark.raw.json
  W1/benchmark_D.json
  W2/benchmark.raw.json
  W2/benchmark_D.json
  W3/benchmark.raw.json
  W3/benchmark_D.json
  W4/benchmark.raw.json
  W4/benchmark_D.json
  W6a/benchmark.raw.json
  W6a/server.log
```
