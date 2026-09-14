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
- 测试脚本：`scripts/checks/ssd_435k_revert_check.py`（435k）、`scripts/checks/ssd_512k_anchor_check.py`
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

**2026-09-14 复验（锚点粒度改为每 cadence 只留 `cadence − block_size` 一个）**：S 386,822 →
V0 命中 352,000（10.5s）→ T 162,430 → R 恢复 384,000（sha `db8b8e836881534b`，59.7s）→
V2 命中 352,000（9.9s），`SSD-435K-REVERT-DONE`。RAMTRACE 窗口 = `[318400, 350400, 382400]`
（K=3 个锚点，较此前 6 个减半），V 的 `diag lookup hits=[352000,352000,352000,352000]`
（Mamba 组命中）。原始输出 `awq_435k_cblock_check.txt`。

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
  与工具 `scripts/tools/kv_pool_sizing.py`（9.6e9 池安全上限 ≈ 500,800 token）。另修复：durable 窗口
  改为按 token 位置淘汰（保留近尾 K 个，而非最早插入的），否则恢复会话只留最早几个边界
  （`30400..96000`）而非近尾（`448k..512k`）。

## 4. 结果文件

- `awq_435k_postfix_check.txt`：435k 复跑原始输出。
- `awq_435k_cblock_check.txt`：435k 复跑原始输出（每 cadence 1 个锚点，2026-09-14）。
- `awq_512k_anchor_check.txt`：512k **MTP1** 复跑原始输出（深回退未命中）。
- `awq_512k_mtp3_check.txt`：512k **MTP3** 复跑原始输出（全 PASS）。
- `awq_435k_upstream_sync_check.txt`：移植上游 3 个 mamba/GDN 修复后的 435k 尺度复验（见 §5）。

## 5. 上游同步后的复验（2026-09-14，`upstream-sync`）

在移植上游 #51812 / #56196 / #49436 之后，用 `scripts/checks/ssd_435k_revert_check.py`
在 **9.6e9 profile**（`--max-model-len 500800`、池 9.6e9、容量 525,229 token、真 NVMe
`ssd_kv`、staging 2.4e9 = 43 槽 / chunk 21、MTP3、锚点 3）上复跑；原始输出
`awq_435k_upstream_sync_check.txt`。

| 请求 | prompt | cached | wall | 说明 |
|---|---|---|---|---|
| S 常驻 | 384,704 | 35,200 | 790.0s | — |
| **V0 深回退 keep=22（常驻）** | 352,652 | **350,400** | 8.8s | 锚点命中 |
| T 小请求（挤出 S） | 160,312 | 0 | 204.7s | — |
| **R 恢复（SSD 分块）** | 384,714 | **382,400** | 47.9s | sha `db8b8e836881534b`，与移植前基线逐字节一致 |
| **V2 深回退（restore 后）** | 352,652 | **350,400** | 8.7s | 恢复后重新认领锚点 |

`METRICS stores=3 restores=2`、写 30.5 GiB、读 24.8 GiB、盘上 6.07 GB；`SSD-435K-REVERT-DONE`。
服务日志除启动期 `fa_utils` 的 FA2 提示（SM75 不支持 FA2，既有）外无 ERROR。同期
`GET /host_tier_info` 显示 gpu 2 条链（398k/366k，各 9 anchors）、ssd 1 条（174k），与
S/T 布局一致。

> 注：本次验收跑在 **9.6e9 profile**（当时生产在跑的实例）上。同一时段 `435k_ssd` profile
> 在本机起不来（`cudaHostRegister` 失败 → 毒化 CUDA context → warmup 报 `CUDA error:
> invalid argument` → 引擎挂死），**用移植前的代码同样复现**，故与本次移植无关。
> 该问题后来定位为 **`/dev/shm` 被残留 staging 文件塞满**，已修复，见 §6。

## 6. 最终验收（2026-09-14，固定 `KV_ENGINE_ID` 修复后）

根因：驱动对 `cudaHostRegister` 失败给出的原话是 `Failed to allocate physical memory`——
`/dev/shm`（本机 7.8 GiB）被历次崩溃残留的 `vllm_offload_<engine_id>.mmap`（2.23 GiB/个，
只有正常退出才 unlink）塞满后，驱动无法为 staging 区落页，返回 `CUDA_ERROR_INVALID_VALUE`
并**毒化 CUDA context**（失败后下一次 CUDA 调用必报 `invalid argument`）。旧 `engine_id` 是
每次启动的随机 UUID，残留永不回收 → 表现为"时好时坏 / 清一下或重启就能起"。

修复：每个 profile 固定 `KV_ENGINE_ID` + 启动前 `rm -f` 自己的 staging 文件（README §5）。

修复后在同一台机器、`435k_ssd` profile（固定 id `qwen38-27b-435k`，`/dev/shm` 29%）复跑
`scripts/checks/ssd_435k_revert_check.py`：**全 PASS**，原始输出 `awq_435k_final_accept.txt`。

| 请求 | prompt | cached | wall | 说明 |
|---|---|---|---|---|
| S 常驻 | 384,704 | 0 | 818.9s | — |
| **V0 深回退（常驻）** | 352,652 | **350,400** | 8.9s | 锚点命中 |
| T 小请求（挤出 S） | 160,312 | 0 | 208.4s | — |
| **R 恢复（SSD 分块）** | 384,714 | **382,400** | 60.8s | sha `db8b8e836881534b` |
| **V2 深回退（恢复后）** | 352,652 | **350,400** | 8.7s | 重新认领锚点 |

`METRICS stores=3 restores=2`、`SSD-435K-REVERT-DONE`；服务日志除既有 `fa_utils` FA2 提示外
无 ERROR。另外验证：固定 id 后 SSD 会话目录稳定（`ssd_kv/qwen38-27b-435k_e82d15bfcaae`），
预先放入的 `dummy_stale.txt` 在重启后被 `VLLM_SSD_CLEAN_START=1` 清掉——`clean_start` 现在
真正能清**上一次运行**的会话。
