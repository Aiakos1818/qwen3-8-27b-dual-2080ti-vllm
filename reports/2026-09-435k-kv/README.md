# 435k / 512k KV 复验报告（三个锚点修复后）

本报告补充 [`docs/kv-optimization/vllm_04_offload_ssd.md`](../../docs/kv-optimization/vllm_04_offload_ssd.md)
§6.4：那份 435k 验收采集于三个锚点修复（`c07c1c9` MTP eagle-drop 对齐、head-prefix
free 保留、恢复会话认领旧锚点）**之前**。这里用最终代码在真 NVMe SSD 上复跑。

## 1. 环境

- 模型：`Qwen3.8-27B-AWQ-INT4-yarn512k`（AWQ-INT4，`--kv-cache-dtype fp8_e4m3`，TP=2，
  2 × RTX 2080Ti 22GB）。
- 保活/锚点：`VLLM_PIN_MIN_TOKENS=16000`、`VLLM_MAMBA_CKPT_ANCHORS=3`、
  `VLLM_HOSTTIER_EVICT_SMALL_TOKENS=32000`；SSD tier（真 NVMe `ssd_kv`，quota 64 GiB，
  限速 800 MiB/s，`SSD_ONLY=1`）。
- 测试脚本：`scripts/ssd_435k_revert_check.py`（435k）、`scripts/ssd_512k_anchor_check.py`
  （512k）。S 常驻 → T 挤出 S 到 SSD → R 整链恢复 → V 深回退到 cadence 锚点。

## 2. 435k（PASS）

`--max-model-len 435200`、池 `9.0e9` → 容量 489,789 token。T 从 407k 缩到 162k
（S 常驻后仅剩 ~97k，162k 足以触发 spill，省约 10 min）。

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S 常驻 | 386,822 | 0 | 786.5s | — |
| **V0 深回退 keep=22（常驻）** | 354,770 | **352,000** | 10.4s | — |
| T 小请求（挤出 S） | 162,430 | 0 | 198.8s | — |
| **R 恢复（SSD 分块）** | 386,832 | **384,000** | 53.0s | `db8b8e836881534b` |
| **V2 深回退 keep=22（restore 后）** | 354,770 | **352,000** | 9.9s | — |

- `SSD-435K-REVERT-DONE`：常驻深回退、restore、restore 后深回退**全部命中 352000 锚点**。
- 指标：`stores=3`、`restores=2`、写 32.0 GiB、读 25.8 GiB；`nvidia-smi` 峰值 ≤21.8 GiB/卡。
- 修复前：restore 后深回退 `cached=0`（整段重算，见 FP8 报告 §5）。

## 3. 512k（满长 prefill + SSD 恢复 PASS；深回退未命中）

512k profile 的 OOM 回退梯：MTP 3→1、batched 4096→1024、池 `9.6e9`（`9.7e9` 首次请求
即 OOM）→ 容量 **536,624** token。注意本配置自动选 **mamba/attention block_size=1584**
（435k 是 1600），故锚点 cadence 必须是 1584 的整数倍：`VLLM_MAMBA_CKPT_TOKENS=31680`
（32000 会因非整数倍被引擎**整体禁用锚点**并打印 warning）。

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S 满长常驻 | 515,046 | 0 | 1504.5s | — |
| T 小请求（挤出 S） | 114,348 | 0 | 137.5s | — |
| **R 恢复（SSD 分块）** | 515,056 | **513,216**（99.6%） | 41.1s | `db8b8e836881534b` |
| V 深回退 keep=30（restore 后） | 482,994 | **0** | 1354.3s | — |

- 结论：**满长 515k 不 OOM**；SSD 分块恢复正确（99.6%、sha 一致、41s vs 重算 1504s，约 36×）。
- **深回退未命中**：池 536,624 对 515,046 只剩 ~21k free。V（482,994）一来，准入压力把
  常驻链 **spill 到 SSD**；而 SSD 里存的是 515k 的完整链，`find` 只匹配"请求 ≥ 存储链"
  （更长或同链），**不匹配更短的纯前缀**，故 V 无法 restore，从 0 重算。
- 对比 435k：池 1.13×、剩 103k，V 不需挤走常驻链，直接在 GPU 前缀缓存命中。
- 这是"内存余量 + SSD 只认长链"的组合限制，非锚点机制缺陷。可后续改进方向：让 SSD
  `find` 也接受更短前缀（restore 会多载入尾部，需权衡），或为 revert 预留 headroom。

## 4. 结果文件

- `awq_435k_postfix_check.txt`：435k 复跑原始输出。
- `awq_512k_anchor_check.txt`：512k 复跑原始输出。
