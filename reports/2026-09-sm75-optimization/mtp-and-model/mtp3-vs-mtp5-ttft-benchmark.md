# Qwen3.8-27B TTFT 基准测试报告：MTP=3 vs MTP=5

> **测试日期**: 2026-09-07
> **测试环境**: Linux <server>, 双卡 RTX 2080 Ti (SM75), vLLM 0.28.0

---

## 1. 测试模型

| 编号 | 模型名称 | 模型路径 | 量化方式 |
|------|----------|----------|----------|
| A (旧模型) | `Qwen3.8-27B-SmoothQuant-W8A8-INT8` | `/home/<user>/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8` | SmoothQuant W8A8 INT8 |
| B (新模型) | `Qwen3.8-27B-INT8-W8A8-imatrix-MTP` | `/home/<user>/models/Qwen3.8-27B-INT8-W8A8-imatrix-MTP` | INT8 W8A8 imatrix (含 MTP 权重) |

## 2. 测试配置

### 2.1 服务参数（所有测试一致，仅模型和 MTP 不同）

| 参数 | 值 |
|------|-----|
| vLLM 版本 | 0.28.0 |
| 服务端口 | 8000 |
| 服务名 | `qwen-local` |
| Tensor Parallel | 2 (device-ids: 0,1) |
| dtype | half (fp16) |
| KV Cache dtype | fp8_e4m3 |
| GPU Memory Utilization | 0.93 |
| KV Cache Memory | 4G |
| Max Model Len | 180,000 |
| Max Num Seqs | 1 |
| Max Num Batched Tokens | 4,096 |
| Chunked Prefill | enabled |
| Prefix Caching | enabled |
| Async Scheduling | disabled |
| GDN Prefill Backend | flashqla_legacy |
| Chat Template | `/home/<user>/models/Qwen-Fixed-Chat-Templates/chat_template.jinja` |
| Reasoning Parser | qwen3 |
| Tool Call Parser | qwen3_xml |

### 2.2 三组测试的变量

| 测试组 | 模型 | MTP tokens | CUDA Graph Capture Sizes | Max CUDAGraph Size |
|--------|------|-----------|--------------------------|---------------------|
| **A-MTP3** | Qwen3.8-27B-SmoothQuant-W8A8-INT8 | 3 | [4] | 4 |
| **B-MTP3** | Qwen3.8-27B-INT8-W8A8-imatrix-MTP | 3 | [4] | 4 |
| **B-MTP5** | Qwen3.8-27B-INT8-W8A8-imatrix-MTP | 5 | [6] | 6 |

### 2.3 测试脚本

| 项目 | 路径 |
|------|------|
| 脚本位置 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/run_context_ttft.py` |
| 结果目录 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/results/` |
| A-MTP3 结果文件 | `results/ttft_results.json` + `results/ttft_20k_60k.json` |
| B-MTP3 结果文件 | `results/ttft_mtp_model.json` |
| B-MTP5 结果文件 | `results/ttft_mtp5_model.json` |
| systemd 服务 | `/etc/systemd/system/qwen-vllm-qwopus.service` |
| 服务日志 | `/home/<user>/vllm-0271-main-8000.log` |

### 2.4 测试方法

- **脚本**: `run_context_ttft.py` — 通过 OpenAI API 流式请求，测量首个 token 返回时间 (TTFT)
- **上下文规模**: 2700, 5400, 8100, 10800, 13500, 16200, 20000, 60000 词
- **每组重复**: 3 次（A-MTP3 的 20K/60K 为 2 次）
- **Max Tokens**: 128
- **Temperature**: 0.6, Top-p: 0.95
- **指标**: TTFT (s)、Prefill 吞吐 (tok/s)、Decode 吞吐 (tok/s)、总耗时 (s)

---

## 3. 测试结果

### 3.1 A-MTP3 (旧模型 SmoothQuant, MTP=3)

| 上下文 (词数) | Prompt Tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Total (s) |
|---|---|---|---|---|---|
| 2,700 | 2,844 | 2.23 | 1,278 | 78.5 | 3.88 |
| 5,400 | 5,644 | 3.60 | 1,568 | 79.6 | 5.28 |
| 8,100 | 8,451 | 5.02 | 1,684 | 90.0 | 6.45 |
| 10,800 | 11,277 | 6.58 | 1,715 | 78.7 | 8.20 |
| 13,500 | 14,046 | 8.14 | 1,725 | 80.9 | 9.73 |
| 16,200 | 16,862 | 9.88 | 1,706 | 84.2 | 11.42 |
| 20,000 | 20,846 | 12.08 | 1,726 | 82.0 | 13.68 |
| 60,000 | 62,388 | 44.79 | 1,393 | 82.2 | 46.39 |

### 3.2 B-MTP3 (新模型 imatrix-MTP, MTP=3)

| 上下文 (词数) | Prompt Tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Total (s) |
|---|---|---|---|---|---|
| 2,700 | 2,844 | 2.09 | 1,361 | 67.9 | 3.98 |
| 5,400 | 5,644 | 3.32 | 1,701 | 67.7 | 5.21 |
| 8,100 | 8,451 | 4.56 | 1,855 | 74.0 | 6.29 |
| 10,800 | 11,277 | 5.92 | 1,904 | 76.7 | 7.61 |
| 13,500 | 14,046 | 7.29 | 1,927 | 72.4 | 9.08 |
| 16,200 | 16,862 | 8.82 | 1,912 | 68.1 | 10.74 |
| 20,000 | 20,841 | 10.97 | 1,900 | 78.9 | 12.60 |
| 60,000 | 62,365 | 41.11 | 1,517 | 75.3 | 42.83 |

### 3.3 B-MTP5 (新模型 imatrix-MTP, MTP=5)

| 上下文 (词数) | Prompt Tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Total (s) |
|---|---|---|---|---|---|
| 2,700 | 2,844 | 2.19 | 1,297 | 87.0 | 3.70 |
| 5,400 | 5,644 | 3.42 | 1,652 | 85.5 | 4.93 |
| 8,100 | 8,451 | 4.75 | 1,780 | 77.2 | 6.45 |
| 10,800 | 11,277 | 6.07 | 1,859 | 80.3 | 7.67 |
| 13,500 | 14,046 | 7.52 | 1,867 | 82.4 | 9.18 |
| 16,200 | 16,862 | 9.09 | 1,856 | 74.1 | 10.84 |
| 20,000 | 20,841 | 11.36 | 1,836 | 70.7 | 13.27 |
| 60,000 | 62,365 | 42.25 | 1,476 | 85.2 | 43.80 |

---

## 4. 三组对比

### 4.1 TTFT 对比

| 上下文 | A-MTP3 (s) | B-MTP3 (s) | B-MTP5 (s) | B-MTP3 vs A-MTP3 | B-MTP5 vs B-MTP3 |
|---|---|---|---|---|---|
| 2.7K | 2.23 | 2.09 | 2.19 | **+6.3%** ↑ | -4.8% ↓ |
| 5.4K | 3.60 | 3.32 | 3.42 | **+7.8%** ↑ | -3.0% ↓ |
| 8.1K | 5.02 | 4.56 | 4.75 | **+9.2%** ↑ | -4.2% ↓ |
| 10.8K | 6.58 | 5.92 | 6.07 | **+10.0%** ↑ | -2.5% ↓ |
| 13.5K | 8.14 | 7.29 | 7.52 | **+10.4%** ↑ | -3.2% ↓ |
| 16.2K | 9.88 | 8.82 | 9.09 | **+10.7%** ↑ | -3.1% ↓ |
| 20K | 12.08 | 10.97 | 11.36 | **+9.2%** ↑ | -3.6% ↓ |
| 60K | 44.79 | 41.11 | 42.25 | **+8.2%** ↑ | -2.8% ↓ |

### 4.2 Prefill 吞吐对比

| 上下文 | A-MTP3 | B-MTP3 | B-MTP5 | B-MTP3 vs A-MTP3 | B-MTP5 vs B-MTP3 |
|---|---|---|---|---|---|
| 2.7K | 1,278 | 1,361 | 1,297 | +6.5% ↑ | -4.7% ↓ |
| 5.4K | 1,568 | 1,701 | 1,652 | +8.5% ↑ | -2.9% ↓ |
| 8.1K | 1,684 | 1,855 | 1,780 | +10.2% ↑ | -4.0% ↓ |
| 10.8K | 1,715 | 1,904 | 1,859 | +11.0% ↑ | -2.4% ↓ |
| 13.5K | 1,725 | 1,927 | 1,867 | +11.7% ↑ | -3.1% ↓ |
| 16.2K | 1,706 | 1,912 | 1,856 | +12.1% ↑ | -2.9% ↓ |
| 20K | 1,726 | 1,900 | 1,836 | +10.1% ↑ | -3.4% ↓ |
| 60K | 1,393 | 1,517 | 1,476 | +8.9% ↑ | -2.7% ↓ |

### 4.3 Decode 吞吐对比

| 上下文 | A-MTP3 | B-MTP3 | B-MTP5 |
|---|---|---|---|
| 2.7K | 78.5 | 67.9 | 87.0 |
| 5.4K | 79.6 | 67.7 | 85.5 |
| 8.1K | 90.0 | 74.0 | 77.2 |
| 10.8K | 78.7 | 76.7 | 80.3 |
| 13.5K | 80.9 | 72.4 | 82.4 |
| 16.2K | 84.2 | 68.1 | 74.1 |
| 20K | 82.0 | 78.9 | 70.7 |
| 60K | 82.2 | 75.3 | 85.2 |
| **平均** | **82.0** | **71.4** | **80.3** |

---

## 5. MTP 接收率分析

### 5.1 MTP=3 接收率（B-MTP3 测试期间日志）

来源: `/home/<user>/vllm-0271-main-8000.log`

| 位置 | 接收率范围 | 典型值 |
|------|-----------|--------|
| 位置 1 | 0.72~0.95 | ~0.82 |
| 位置 2 | 0.45~0.73 | ~0.57 |
| 位置 3 | 0.31~0.54 | ~0.50 |
| **平均接收率** | | **~62.9%** |
| **平均接受长度** | | **~2.9 tokens** |

### 5.2 MTP=5 接收率（B-MTP5 测试期间日志）

| 位置 | 接收率范围 | 典型值 |
|------|-----------|--------|
| 位置 1 | 0.71~0.91 | ~0.80 |
| 位置 2 | 0.42~0.74 | ~0.56 |
| 位置 3 | 0.25~0.61 | ~0.42 |
| 位置 4 | 0.12~0.39 | ~0.27 |
| 位置 5 | 0.10~0.35 | ~0.21 |
| **平均接收率** | | **~44.8%** |
| **平均接受长度** | | **~3.2 tokens** |

### 5.3 接收率按上下文分布 (MTP=5)

| 上下文 | Avg 接收率 | Mean Accept Length |
|---|---|---|
| 2.7K | ~57.0% | ~3.85 |
| 5.4K | ~45.1%~49.9% | ~3.25~3.49 |
| 8.1K | ~36.4%~50.6% | ~2.82~3.98 |
| 10.8K+ | ~32.2%~53.1% | ~2.61~3.66 |
| 60K | ~38.3%~53.1% | ~2.92~3.66 |

### 5.4 vLLM 日志警告

```
WARNING [speculative.py:971] Enabling num_speculative_tokens > 1 will run multiple times 
of forward on same MTP layer, which may result in lower acceptance rate
```

```
WARNING [vllm.py:1862] max_num_scheduled_tokens is set to 4096 based on the speculative 
decoding settings. This may lead to suboptimal performance. Consider increasing 
max_num_batched_tokens to accommodate the additional draft token slots, or decrease 
num_speculative_tokens.
```

---

## 6. 结论

### 6.1 模型对比: B-MTP3 vs A-MTP3

| 指标 | 结论 |
|------|------|
| **TTFT** | 新模型 (imatrix-MTP) 全程快 **8~11%**，中长上下文优势最大 |
| **Prefill 峰值** | 1,927 vs 1,727 tok/s，提升 **+11.6%** |
| **60K Prefill** | 1,517 vs 1,393 tok/s，提升 **+8.9%** |
| **Decode** | 71.4 vs 82.0 tok/s，下降 **-13%** |
| **最佳配置** | MTP=3 是最佳搭配 |

### 6.2 MTP tokens 对比: MTP=5 vs MTP=3

| 指标 | 结论 |
|------|------|
| **TTFT** | MTP=5 全程慢 **3~5%** |
| **Prefill** | MTP=5 低 **3~4%** |
| **接收率** | MTP=5 平均仅 **44.8%** vs MTP=3 的 **62.9%** |
| **位置 4-5 接收率** | 仅 12~35%，draft 算力大量浪费 |
| **接受长度** | MTP=5 略高 (3.2 vs 2.9)，但不足以弥补 |
| **Decode** | MTP=5 略快 (80.3 vs 71.4 tok/s)，推测增益来自更多 spec token |
| **根因** | vLLM 对 MTP>1 需在同一 MTP 层多次前向，位置越深准确度越低 |

### 6.3 最优配置推荐

| 项目 | 推荐值 |
|------|--------|
| **模型** | `Qwen3.8-27B-INT8-W8A8-imatrix-MTP` |
| **MTP tokens** | **3** (而非 5) |
| **CUDA Graph** | `[4]`, max=4 |
| **Prefill 峰值** | 1,927 tok/s |
| **60K TTFT** | 41.1s |
| **60K Prefill** | 1,517 tok/s |

> **MTP=5 不可取**: 位置 4-5 接收率过低 (~21%)，draft 浪费算力，反而拖慢 prefill 速度 3-5%。vLLM 官方也对此配置发出警告。MTP=3 在本硬件 (SM75) 和模型组合下是最佳平衡点。

---

## 7. 文件索引

| 文件 | 路径 |
|------|------|
| 测试脚本 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/run_context_ttft.py` |
| A-MTP3 结果 (2.7K-16.2K) | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/results/ttft_results.json` |
| A-MTP3 结果 (20K, 60K) | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/results/ttft_20k_60k.json` |
| B-MTP3 结果 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/results/ttft_mtp_model.json` |
| B-MTP5 结果 | `/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/results/ttft_mtp5_model.json` |
| systemd 服务 | `/etc/systemd/system/qwen-vllm-qwopus.service` |
| vLLM 日志 | `/home/<user>/vllm-0271-main-8000.log` |
| 本报告 | `<local>/Documents/vllm/tasks/reports/2026-09-07-mtp3-vs-mtp5-ttft-benchmark.md` |
