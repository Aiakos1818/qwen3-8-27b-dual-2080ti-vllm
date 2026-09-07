# 支线五：W4A16（AWQ）— ⚠️ 备选（decode 导向场景，质量待评）

## 结论

W4A16（AWQ，Marlin WNA16 kernel）与 W8A8 互补：decode 约 2× 快、显存约 -30%，但 TTFT 明显差于 W8A8。
适合"短输入、长输出"场景；**未做质量评测**（AWQ 量化对 thinking/tool call 的影响需验证）。

## 实测数据（vs W8A8 W1）

| 维度 | W8A8 (W1) | W4A16 (W3) |
|---|---|---|
| TTFT S/M/L/XL | 2.08 / 4.39 / 9.71 / 35.76 s | 2.29 / 5.71 / 13.48 / 48.18 s |
| Decode（D 档 2048 tokens） | 37.2 tok/s | **76.3 tok/s（+105%）** |
| 显存/卡 | ~23 GB | ~16 GB（-30%） |

## 文件

- [awq-int4-triton-turing-fa2-vllm0280-line.md](awq-int4-triton-turing-fa2-vllm0280-line.md) — 原始报告（含 W4A4 P3 规划）
- raw/ — W3/W4 原始 JSON + AWQ 上下文梯度数据
