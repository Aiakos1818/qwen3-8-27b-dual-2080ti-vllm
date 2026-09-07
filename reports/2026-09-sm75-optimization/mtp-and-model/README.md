# 支线二：MTP 深度与新旧模型 — ✅ MTP3 最优

## 结论

- 同模型（imatrix-MTP）下 **MTP3 优于 MTP5**：MTP5 深层位置（第 4/5 位）接收率坍缩到 ~27%/~21%，平均接收率 62.9% → 44.8%，TTFT 反而慢 3~5%。
- **新 imatrix-MTP 模型比旧 SmoothQuant W8A8 模型 TTFT 快 8~11%**，prefill 峰值 +11.6%，代价是 decode -13%。
- vLLM 官方警告：speculative decoding 不建议超过 3~4 个 draft token。

## 实测数据（imatrix-MTP 模型，fp8_e4m3 KV，180K）

| 上下文 | MTP3 TTFT | MTP5 TTFT | MTP3 prefill | MTP5 prefill |
|---|---|---|---|---|
| 2.7K | 2.09 s | 2.19 s | 1,361 | 1,298 tok/s |
| 8.1K | 4.56 s | 4.75 s | 1,855 | 1,780 tok/s |
| 20K | 10.97 s | 11.35 s | 1,900 | 1,836 tok/s |
| 60K | 41.11 s | 42.24 s | 1,517 | 1,476 tok/s |

接收率（MTP3 / MTP5，服务日志）：位置1 ~82%/~80%，位置2 ~57%/~56%，位置3 ~50%/~42%，位置4 —/~27%，位置5 —/~21%；平均 62.9% / 44.8%。

新旧模型对比（MTP3）：2.7K 2.23→2.09，8.1K 5.02→4.56，20K 12.08→10.97，60K 44.79→41.11 s。

## 文件

- [mtp3-vs-mtp5-ttft-benchmark.md](mtp3-vs-mtp5-ttft-benchmark.md) — 原始报告
- raw/ — ttft_results.json（A-MTP3）、ttft_mtp_model.json（B-MTP3）、ttft_mtp5_model.json（B-MTP5）、ttft_20k_60k.json
