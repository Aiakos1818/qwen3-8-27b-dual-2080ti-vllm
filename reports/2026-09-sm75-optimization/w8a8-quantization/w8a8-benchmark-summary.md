# W8A8 基准测试汇总报告

> 测试日期：2026-09-06 ~ 2026-09-07
> 硬件：2× RTX 2080 Ti (SM75, 22GB ×2)
> 模型：`Qwen3.8-27B-INT8-W8A8-imatrix-MTP`
> vLLM 版本：v0.28.0
> 测试脚本路径：`/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/`

---

## 一、变体配置一览

| 变体 | venv 环境 | 引擎 | KV Cache 格式 | max_model_len | MTP tokens | 线性算子 | 备注 |
|------|----------|------|---------------|---------------|------------|---------|------|
| **B0** | `vllm-env-0280-qwopus` | 官方 v0.28.0 | FP8 (fp8_e4m3) | 180,000 | 3 | MarlinFP8ScaledMM | FP8 基准（重测） |
| **W1** | `vllm-env-0280-qwopus` | 官方 v0.28.0 | FP16 (默认) | 65,536 | 3 | CutlassInt8ScaledMM | W8A8 基准（FP16 KV） |
| **W7** | `vllm-env-0280-qwopus` | 官方 v0.28.0 | FP8 (fp8_e4m3) | 180,000 | 3 | CutlassInt8ScaledMM | W8A8 + FP8 KV + 180k |
| **W9** | `vllm-env-0280-triton-turing-canary` | Triton-Turing Fork | INT8 (per_token_head) | 180,000 | 3 | CutlassInt8ScaledMM | Fork 引擎 + INT8 KV |
| **W10** | `vllm-env-0280-qwopus` | 官方 v0.28.0 | FP8 (fp8_e4m3) | 180,000 | 4 | CutlassInt8ScaledMM | W8A8 + FP8 KV + MTP=4 |

### 公共启动参数

所有变体共享以下参数：

```
--model /home/<user>/models/Qwen3.8-27B-INT8-W8A8-imatrix-MTP
--dtype half --tensor-parallel-size 2 --device-ids 0,1
--gpu-memory-utilization 0.93 --kv-cache-memory-bytes 4G
--enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 4096
--enable-chunked-prefill --no-async-scheduling
--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
--compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
```

### 环境变量

```
TRITON_SM75_BF16_DOT_AS_F16=0
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_QWOPUS_MTP_BF16_DRAFT=1
VLLM_SM75_SPEC_SYNC_MODE=safe
VLLM_USE_V2_MODEL_RUNNER=1
PYTHONPATH=/home/<user>/FlashQLA-SM70-SM75-0280
```

> **W9 额外**：`TRITON_BACKENDS_IN_TREE=1`（Fork 引擎需要显式启用 in-tree Triton backends）

### venv 路径

| venv | 路径 | 说明 |
|------|------|------|
| `vllm-env-0280-qwopus` | `/home/<user>/vllm-env-0280-qwopus/` | 官方 vLLM v0.28.0 + QwOpus 补丁 |
| `vllm-env-0280-triton-turing-canary` | `/home/<user>/vllm-env-0280-triton-turing-canary/` | Triton-Turing Fork（SM75 优化分支） |

### 启动脚本路径

| 变体 | 脚本路径 |
|------|---------|
| B0 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/B0/start.sh` |
| W1 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W1/start.sh` |
| W7 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W7/start.sh` |
| W9 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W9/start.sh` |
| W10 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W10/start.sh` |

### 测试脚本

| 测试类型 | 脚本路径 | 说明 |
|---------|---------|------|
| TTFT (S/M/L/XL) | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/run_context_ttft.py` | `--word-counts 2700 5400 8100 19000 57000 --runs 2 --max-tokens 128` |

> **D 档测试 (`bench_d.py`) 数据已废弃**：该脚本计算的 Decode 速度将 thinking 阶段时间纳入分母，而 `completion_tokens` 包含 reasoning tokens，导致每次结果随机波动、数值虚低，不具备参考价值。报告仅使用 TTFT 测试（128 tokens 短生成）的数据。

---

## 二、TTFT 测试结果（max_tokens=128）

> 每档取所有 run 的**平均值**
> B0/W1/W7/W10: 2 runs；W9: 3 runs（S 档 run1/2 冷启动异常，取 run3）
> Decode 速度 = 128 tokens 短生成的平均速度（KV Cache 短，attention 计算量小，数值偏高）

### S 档 (~2.84K tokens)

| 变体 | 平均 TTFT (s) | 平均 Prefill (tok/s) | 平均 Decode (tok/s) |
|------|-------------|--------------------|-------------------|
| **B0** | 2.59 | 1,097 | 98.6 |
| **W1** | **2.08** | **1,365** | 82.3 |
| **W7** | 2.13 | 1,337 | 76.6 |
| **W9** | 2.23 | 1,277 | 73.6 |
| **W10** | 2.19 | 1,303 | 77.6 |

### 5.64K tokens

| 变体 | 平均 TTFT (s) | 平均 Prefill (tok/s) | 平均 Decode (tok/s) |
|------|-------------|--------------------|-------------------|
| **W9** | 3.79 | 1,488 | 69.7 |
| **W10** | **3.38** | **1,671** | 75.2 |

> B0/W1/W7 未跑此档位

### M 档 (~8.45K tokens)

| 变体 | 平均 TTFT (s) | 平均 Prefill (tok/s) | 平均 Decode (tok/s) |
|------|-------------|--------------------|-------------------|
| **B0** | 6.37 | 1,328 | 115.1 |
| **W1** | **4.39** | **1,928** | 79.1 |
| **W7** | 4.48 | 1,888 | 83.7 |
| **W9** | 5.54 | 1,526 | 62.3 |
| **W10** | 4.69 | 1,803 | 92.8 |

### L 档 (~19.77K tokens)

| 变体 | 平均 TTFT (s) | 平均 Prefill (tok/s) | 平均 Decode (tok/s) |
|------|-------------|--------------------|-------------------|
| **B0** | 14.92 | 1,325 | 90.4 |
| **W1** | **9.71** | **2,037** | 79.4 |
| **W7** | 10.11 | 1,956 | 66.3 |
| **W9** | — | — | — |
| **W10** | 10.45 | 1,892 | 92.6 |

### XL 档 (~59.24K tokens)

| 变体 | 平均 TTFT (s) | 平均 Prefill (tok/s) | 平均 Decode (tok/s) |
|------|-------------|--------------------|-------------------|
| **B0** | 52.98 | 1,118 | 92.4 |
| **W1** | **35.76** | **1,656** | 66.7 |
| **W7** | 37.73 | 1,570 | 75.6 |
| **W9** | — | — | — |
| **W10** | 38.75 | 1,529 | 71.5 |

> W9 未跑 L/XL 档（只跑了 S/M 2700/5400/8100 词）

---

## 三、汇总对比表

### 平均 TTFT (s) — 越低越好

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---------|-----|-----|-----|-----|------|
| 2.84K | 2.59 | **2.08** | 2.13 | 2.23 | 2.19 |
| 5.64K | — | — | — | 3.79 | **3.38** |
| 8.45K | 6.37 | **4.39** | 4.48 | 5.54 | 4.69 |
| 19.77K | 14.92 | **9.71** | 10.11 | — | 10.45 |
| 59.24K | 52.98 | **35.76** | 37.73 | — | 38.75 |

### 平均 Prefill 速度 (tok/s) — 越高越好

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---------|-----|-----|-----|-----|------|
| 2.84K | 1,097 | **1,365** | 1,337 | 1,277 | 1,303 |
| 5.64K | — | — | — | 1,488 | **1,671** |
| 8.45K | 1,328 | **1,928** | 1,888 | 1,526 | 1,803 |
| 19.77K | 1,325 | **2,037** | 1,956 | — | 1,892 |
| 59.24K | 1,118 | **1,656** | 1,570 | — | 1,529 |

### 平均 Decode 速度 (tok/s, 128 tokens 短生成) — 越高越好

| 输入长度 | B0 | W1 | W7 | W9 | W10 |
|---------|-----|-----|-----|-----|------|
| 2.84K | 98.6 | 82.3 | 76.6 | 73.6 | 77.6 |
| 8.45K | 115.1 | 79.1 | 83.7 | 62.3 | 92.8 |
| 19.77K | 90.4 | 79.4 | 66.3 | — | 92.6 |
| 59.24K | 92.4 | 66.7 | 75.6 | — | 71.5 |

> 注：此 Decode 速度为 128 tokens 短生成的平均值，KV Cache 短、attention 计算量小，数值偏高，仅适合横向对比各变体在相同条件下的差异，不代表长生成场景的实际 Decode 速度。

---

## 四、相对 B0 的提升幅度

### TTFT 提升对比 (→ 负数 = 更快)

| 输入长度 | W1 vs B0 | W7 vs B0 | W10 vs B0 |
|---------|---------|---------|----------|
| 2.84K | -20% | -18% | -15% |
| 8.45K | -31% | -30% | -26% |
| 19.77K | -35% | -32% | -30% |
| 59.24K | -33% | -29% | -27% |

### Prefill 提升对比 (→ 正数 = 更快)

| 输入长度 | W1 vs B0 | W7 vs B0 | W10 vs B0 |
|---------|---------|---------|----------|
| 2.84K | +24% | +22% | +19% |
| 8.45K | +45% | +42% | +36% |
| 19.77K | +54% | +48% | +43% |
| 59.24K | +48% | +40% | +37% |

---

## 五、关键结论

### 1. W1 是最优配置

**W8A8 官方 v0.28.0 + FP16 KV Cache + 65k 上下文 + MTP=3** 在 TTFT 和 Prefill 两个维度都是最优的：
- TTFT 比 B0 缩短 **20-35%**
- Prefill 吞吐提升 **24-54%**
- 不需要 Fork 引擎，稳定性最好

### 2. KV Cache 量化对 Prefill 有负面影响

- **FP8 KV (W7)**：Prefill 比 W1 慢 2-5%，TTFT 慢 2-3%
- **INT8 KV (W9)**：Prefill 比 W1 慢 6-21%，且 S 档冷启动出现异常（28.5s/59.1s）
- KV Cache 量化节省的显存在固定 `kv-cache-memory-bytes 4G` 下没有实际收益

### 3. MTP=4 不如 MTP=3

- W10 (MTP=4) 各档 TTFT 均比 W1 慢 1-8%
- W10 Prefill 吞吐比 W1 下降 5-7%
- vLLM 启动时已警告：`num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate`
- MTP=4 的额外 draft forward 开关超过了投机采样收益

### 4. Fork 引擎无额外收益

- W9 (Triton-Turing Fork) 的 TTFT 和 Prefill 均比 W1 (官方) 差
- Fork 引擎需要额外设置 `TRITON_BACKENDS_IN_TREE=1`，增加部署复杂度
- 在当前 SM75 + W8A8 配置下，Fork 引擎没有带来性能提升

### 5. 最终推荐

| 场景 | 推荐配置 |
|------|---------|
| 生产环境 | **W1**：W8A8 + FP16 KV + 65k + MTP3 |
| 显存受限 | **W7**：W8A8 + FP8 KV + 180k + MTP3（TTFT 略慢但可接受） |
| 不推荐 | W9 (INT8 KV) 和 W10 (MTP=4)：均劣于 W1 |

---

## 六、附录：原始数据文件路径

| 变体 | TTFT 数据 | 启动脚本 |
|------|---------|---------|
| B0 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/B0_retest/benchmark.raw.json` | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/B0/start.sh` |
| W1 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W1/benchmark.raw.json` | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W1/start.sh` |
| W7 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W7/benchmark.raw.json` | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W7/start.sh` |
| W9 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W9_ttft.json` | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W9/start.sh` |
| W10 | `W10_ttft.json` + `W10_ttft_LX.json` | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/W10/start.sh` |
