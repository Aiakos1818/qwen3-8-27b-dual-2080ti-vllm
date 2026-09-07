# 支线六：INT8 KV Cache — ⚠️ 基本未测（可探索空间）

## 现状

INT8 KV Cache 的格式空间没有系统性测试。2026-09 战役只测了一个变体的部分数据：

| 变体 | KV 量化 | 结果 |
|---|---|---|
| W9 | INT8 per_token_head，W8A8 权重，180K | S 2.23s / 5.6K 3.79s / M 5.54s（S 档 run1/2 冷启动异常 28.5s/59.1s，取 run3）；L/XL 未测；判负优化（M 档比 W1 慢 +26%） |

原始数据：[../w8a8-quantization/raw/W9-ttft.json](../w8a8-quantization/raw/W9-ttft.json)

## 未测矩阵

| 维度 | 选项 | 已测 |
|---|---|---|
| 粒度 | per_token / per_channel / per_token_head | 仅 per_token_head（部分） |
| 权重组合 | FP8 / W8A8 / W4A16 | 仅 W8A8 |
| 上下文 | 65K / 180K | 仅 180K |
| MTP | 3 / 4 / 5 | 仅 3 |

## 价值

INT8 KV 相对 FP16 KV 省一半 KV 显存；配合 W8A8 权重，可能同时保住 180K 长上下文与更大 KV 容量（当前 180K 方案 W7 用的是 FP8 KV）。
若要探索，建议从 W7 条件下的 per_token 起步，测 S/M/L/XL + decode，并注意冷启动异常（W9 曾出现）。
