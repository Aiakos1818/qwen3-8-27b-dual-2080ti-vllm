# 435k / 512k KV 复验报告（三个锚点修复后）

本报告补充 [`docs/kv-optimization/vllm_04_offload_ssd.md`](../../docs/kv-optimization/vllm_04_offload_ssd.md)
§6.4：那份 435k 验收采集于三个锚点修复（`07bf075` MTP eagle-drop 对齐、`8850f9d`
head-prefix free 保留、`996970e` 恢复会话认领旧锚点）**之前**。这里用最终代码在真 NVMe
SSD 上复跑。

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

## 3. 512k（MTP3：满长 prefill + SSD 恢复 + 深回退全 PASS）

512k profile 现与生产一致用 **MTP3**：block_size 自动选 **1600**，cadence 32000（MTP1 会选
1584；本分支 cadence 已自动向下对齐到 block_size，32000 在两种情况下都能用）。OOM 回退梯：
池 `9.6e9`（`9.7e9` 首次请求即 OOM）→ 容量 **525,816** token。

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S 满长常驻（31 轮） | 499,018 | 0 | 1490.7s | — |
| T 小请求（挤出 S） | 114,348 | 0 | 137.8s | — |
| **R 恢复（SSD 分块）** | 499,028 | **496,000**（99.4%） | 45.7s | `db8b8e836881534b` |
| **V 深回退 keep=30** | 482,994 | **480,000** | 16.6s | — |

- `SSD-512K-ANCHOR-DONE`：满长 prefill 不 OOM；SSD 分块恢复正确（99.4%、sha 一致、45.7s vs
  重算 1490.7s，约 33×）；**restore 后深回退命中 480000 锚点**（`stores=2`，R 未被 spill）。
- 深回退能否命中取决于**恢复后常驻链是否仍在 GPU**。MTP3/S=499k 时池剩 ~26.8k slot，R 恢复
  后仍常驻，V 直接命中其前缀。
- 对比 MTP1（`awq_512k_anchor_check.txt`，S=515k、池 536,624、仅剩 ~21.6k slot）：V 的准入
  把常驻链 spill 到 SSD，而 SSD `find` 只认"更长/同链"、不认更短纯前缀 → 重算。即池贴近上限
  时才会触发该限制。
- 结论：MTP3 下 512k 全流程 PASS。极上限（S≥515k、余量 < ~6 块）长 prefill 会**自我抢占**
  （日志 `alloc gate ... need=3 free=0`），释放 durable 窗口、近尾锚点丢失 → 深回退重算；
  需留 **余量 ≥ 16 块（25,600 token）**。池容量口径与推荐值见
  [`docs/kv-optimization/GPU_MEMORY_CALCULATION.md` §4.5](../../docs/kv-optimization/GPU_MEMORY_CALCULATION.md)
  与工具 `scripts/kv_pool_sizing.py`（9.6e9 池安全上限 ≈ 500,800 token）。另修复：durable 窗口
  改为按 token 位置淘汰（保留近尾 K 个，而非最早插入的），否则恢复会话只留最早几个边界
  （`30400..96000`）而非近尾（`448k..512k`）。

## 4. 结果文件

- `awq_435k_postfix_check.txt`：435k 复跑原始输出。
- `awq_512k_anchor_check.txt`：512k **MTP1** 复跑原始输出（深回退未命中）。
- `awq_512k_mtp3_check.txt`：512k **MTP3** 复跑原始输出（全 PASS）。
