# vLLM KV 优化 04：Offload 到 SSD（两层 GPU/SSD，分块流式）

> 2026-09 实现并实测。目标：把自建 host-tier 从「GPU + RAM parking」升级为
> **两层 GPU ↔ SSD**——CPU host 区只作传输期 bounce buffer，parked 会话全部落盘；
> 再升级为**分块流式**，使任意大小会话（直至 max-model-len）都能 park/resume。
> 保留保活 pin、Mamba 边界状态、cadence 锚点与 restore 后深回退等全部会话语义。
>
> 相关文档：
> - 保活：[`vllm_01_保活.md`](vllm_01_保活.md)
> - 锚点：[`vllm_02_锚点.md`](vllm_02_锚点.md)
> - RAM offload：[`vllm_03_offload_ram.md`](vllm_03_offload_ram.md)
> - KV 信息面板：[`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)
> - 运行脚本：`scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh`

---

## 1. 背景

原方案（见 [`vllm_03_offload_ram.md`](vllm_03_offload_ram.md)）在 GPU 显存压力下把最小保活
会话整链 spill 到 `/dev/shm` 的 CPU 槽位（park-to-RAM），resume 时再整链 load 回。RAM 容量
受机器限制（本机 `/dev/shm` 7.8 GiB，50k 会话约 1.9 GiB），无法驻留多个大会话。

SSD 方案去掉 RAM 停车层：**会话要么在 GPU，要么在 SSD**；CPU host 区只在传输过程中短暂占用。

```
GPU KV (保活/mamba边界/锚点)
   │ spill（准入压力）              ▲ restore（resume）
   ▼                               │
CPU staging（仅传输中占用）        │
   │ 异步写 O_DIRECT               │ 异步读
   ▼                               │
SSD 会话仓（配额 + LRU + 进程内索引）
```

- `VLLM_SSD_ROOT` 未设置时自动退回原 RAM parking（旧路径保留、回归通过）。
- 启用 SSD 时连接器的 native offload 仍关闭（`native_store_enabled=False`），
  避免其分配器覆盖 parked 槽位（历史 NaN 根因，见 03 §8）。

---

## 2. 架构与数据流

### 2.1 存储模型

- CPU staging 池 = `cpu_bytes_to_use` 个槽（每槽 `kv_bytes_per_chunk` ≈ 54 MiB，
  含双 rank 分片），来自 `/dev/shm/vllm_offload_<engine_id>.mmap` 共享区。
- scheduler 进程通过 `create_scheduler_view()` 建 `rank=None` 零拷贝视图，
  直接对 staging 槽做文件 I/O（无需 worker 参与）。
- 磁盘上每个会话一个目录：`<root>/<engine_id>_<hash>/sessions/<sid>/<idx>.bin`，
  每槽一个文件；`sid` 做了文件系统安全化（sha1 后缀）。
- 进程内索引按 **tail hash / 前缀 hash** 匹配（与 RAM 路径同规则，取最长链）。
  每条会话记录保留 tokens/槽数/字节/**锚点数**（供面板逐条显示，见
  [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md) §1.3）。

### 2.2 Spill（GPU 压力淘汰保活会话）

```
候选(仍 GPU-pinned) ── alloc staging ──► add_external_store(GPU→CPU)
    │                                        │ 完成：confirm_spill_ssd()
    │                                        ▼  立即 unpin/free GPU 块
    │                                   evict_for(配额 LRU) ──► submit_store(staging→SSD)
    │                                        │ 完成：free staging + 索引生效
    └── staging 不足：留在 pending 队列，下一步重试（不丢会话）
```

- GPU 块在 GPU→CPU 拷贝完成即释放（数据已在 staging），不必等落盘。
- staging 槽持有到 **SSD 写完** 才释放（防止并发迁移复用覆盖）。
- 写文件走 temp+rename 原子替换；索引只在全部文件成功后生效。

### 2.3 Restore（resume 命中 SSD）

```
索引命中 ── 保证 GPU free ≥ need（必要时 spill 其他，保护本会话）
   │
   ├─ alloc staging ──► take_for_restore(sid) ──► submit_load(SSD→staging)
   │                        （取出索引，防止本步重复匹配）
   │                    完成：alloc 新 GPU 块 + add_external_load(staging→GPU)
   │                    完成：register_restored_blocks + free staging + 删会话文件
   └─ staging 不足：下一步重试
```

- 恢复目标会话不被配额 LRU 驱逐；在飞读/写的会话也受保护。
- 恢复完成后文件即删（数据已在 GPU）；再次被淘汰会重新落盘。

### 2.4 失败处理

任一步失败 → 释放 staging/GPU 块、删除该会话文件、计数 `drops`，请求按
普通路径重算（不崩溃、不残留索引）。SSD 读失败会先删坏文件，避免死循环重试。

---

## 3. 配置（`vllm/envs.py`）

| env | 默认 | 说明 |
|---|---|---|
| `VLLM_SSD_ROOT` | `""` | SSD 会话仓根目录；空 = 禁用（RAM 回退） |
| `VLLM_SSD_QUOTA_BYTES` | `8 GiB` (8589934592) | 配额；超限按 LRU 删整会话目录 |
| `VLLM_SSD_READ_THREADS` | `8` | 读优先 I/O 线程 |
| `VLLM_SSD_WRITE_THREADS` | `8` | 写优先 I/O 线程 |
| `VLLM_SSD_MAX_MBPS` | `0` | 聚合读写限速（0 = 不限） |
| `VLLM_SSD_CLEAN_START` | `1` | 启动清空本 engine 的会话目录 |
| `VLLM_SSD_CHUNK_SLOTS` | `0` | 每 chunk 的 slot 数；0 = staging 的一半（双缓冲） |
| `VLLM_SSD_ONLY` | `0` | 1 = 必须有 `VLLM_SSD_ROOT`+staging，否则启动失败（禁用 RAM 回退） |
| `VLLM_HOSTTIER_EVICT_SMALL_TOKENS` | `64000` | 驱逐分档阈值（见 03 §9；0 = 单档最旧优先） |
| `VLLM_DISABLE_HOSTTIER` | `0` | 全局 kill-switch（含 SSD） |

> **`engine_id`**（`--kv-transfer-config` 的字段，profile 里由 `KV_ENGINE_ID` 注入）：决定
> SSD 会话目录名（`<root>/<engine_id 的 _safe_name>/sessions`）与 `/dev/shm` staging 文件名
> （`vllm_offload_<engine_id>.mmap`）。**每个 profile 必须固定且唯一**——随机 UUID 会让上次的
> 会话目录变成孤儿、staging 残留逐次累积（详见 README §5 与本文 §8）。

`cpu_bytes_to_use` = staging 池总大小（双 rank）。**分块流式下不必放得下整个会话**，
只决定每次搬多少、要搬几趟：默认 `chunk = staging//2`（双缓冲，让多会话 / restore+spill
各占一个在飞 chunk）；staging 越小 → chunk 越小 → I/O 往返越多。硬性下限仅 `chunk ≥ 1`。

| 部署 | staging | chunk | 可 offload 会话 |
|---|---|---|---|
| 100k / 256k / 435k / pool9.6e9（当前 profile） | 2.4e9（43 slot） | 21 slot ≈ 1.1 GiB | 无上限（435k 会话 ≈ 14 chunk） |

> **staging 下限（本机 kernel 7.0 / 驱动 580）**：过小（≤2e9，≤35 slot）会让 TP1 的
> `cudaHostRegister` 返回 `cudaErrorInvalidValue`，进而毒化 CUDA context 使 warmup 失败
> （`qwen_triton_warmup` 里一个 1 元素 `torch.full` 报 `CUDA error: invalid argument`）。
> 实测 **2.4e9（43 slot）稳定**；4e9（71 slot）亦稳定。历史实测见 §7/§8。

启动示例：

```bash
export VLLM_SSD_ROOT=/path/to/ssd_kv
export VLLM_SSD_QUOTA_BYTES=68719476736   # 64 GiB
export VLLM_SSD_ONLY=1
# --kv-transfer-config ... "cpu_bytes_to_use":2400000000
```

---

## 4. 实现与不变式

| 文件 | 内容 |
|---|---|
| **新** `vllm/v1/core/host_tier_ssd.py` | `HostTierSSDStore`：索引/配额 LRU/线程池/限速/`take_for_restore`/`poll`/`discard`；分块 API `begin_store`/`append_store`/`abort_store`/`submit_load_range`/`finish_restore`（commit 标记）；`evict_sort_key`/`evict_for`；复用 `tiering/fs` 的 `DualQueueThreadPool` + `batch_store_block`/`batch_load_block`（O_DIRECT 自动回退、原子 temp+rename、C 扩展 `fs_io_C`） |
| `vllm/v1/core/sched/scheduler.py` | 分块 spill/restore 状态机（`_advance_ssd_spill`/`_advance_ssd_restore`/`_finish_ssd_copy`/`_finish_ssd_chunk`）；restore 优先占 staging；attention 先传、mamba 锚点后传；`_request_remaining_blocks` 与准入 gate 同参 |
| `vllm/v1/core/kv_cache_manager.py` | `release_spill_blocks`（逐块释放）/`spill_hold_done`/`hold_restored_blocks`/`release_restored_hold`；`confirm_spill_ssd()`（K=1 路径） |
| `vllm/v1/kv_offload/cpu/spec.py` | `create_scheduler_view()`（scheduler 侧共享区视图） |
| `vllm/distributed/.../offloading_connector.py` | `cpu_engine_id()` / `create_scheduler_kv_region()` |
| `vllm/v1/metrics/{stats,loggers}.py` | SSD 指标（见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)） |
| `vllm/envs.py` | `VLLM_SSD_*` 声明 |
| `tests/v1/core/test_host_tier_ssd.py` | 12 个单测（索引/配额 LRU/槽位生命周期/失败清理/在飞保护/分块/两档驱逐/锚点快照） |
| `scripts/checks/ssd_matrix.py` | 单次启动功能矩阵 |
| `scripts/checks/ssd_100k_check.py` / `scripts/checks/ssd_435k_check.py` | 真 NVMe park/resume |
| `scripts/checks/ssd_crash_check.py` | 写中 SIGKILL 原子性 + 启动清理 |

### 关键不变式

1. staging 槽在对应 SSD 写完前绝不释放；GPU 块在 GPU→CPU 完成后即可释放。
2. 索引只在文件全部落盘（原子 rename）后生效；索引取出（restore 中）的会话不再匹配。
3. 恢复目标与在飞会话不被配额驱逐。
4. 暂存不足只**推迟**迁移（下一步重试），仅当会话 > staging 容量才 drop+告警。
5. 每次会话迁移 all-or-nothing，失败回退为普通重算。

---

## 5. 分块流式（无会话大小上限，2026-09 追加）

整会话版要求 staging ≥ 最大会话（435k 需 ~14 GiB，本机不可行）。分块流式把一次会话传输
拆成 K 个 chunk，CPU 区退化为双缓冲 bounce buffer，任意会话（≤ max-model-len）均可 park/resume。

### 5.1 传输模型
- `chunk = VLLM_SSD_CHUNK_SLOTS`（默认 `staging//2`，双缓冲）。
- **传输顺序：attention 组先、mamba 锚点最后**。空闲块从队列头驱逐，锚点最后
  释放 → 位于队尾 → restore 后深回退锚点存活最久（修复 keep2=0 的回归，见 §7）。
- 会话文件仍按传输顺序编号（`<sid>/00000.bin…`），恢复按同序回读。

### 5.2 Spill
`begin_store`（整会话配额预留）→ 逐 chunk：`alloc staging` →
`add_external_store`（GPU→CPU）→ `release_spill_blocks`（该 chunk GPU 块立即
释放）+ `append_store(..., commit=last)` → 释放 staging、前进。全部 commit 后
入索引、计数一次。

### 5.3 Restore
`take_for_restore`（取出索引）→ 逐 chunk：确保 GPU 空闲 ≥ chunk（不足先 spill
其他，protect 目标；**GPU 块在读开始前分配**，消除检查/分配竞态）→
`alloc staging` → `submit_load_range`（SSD→staging）→ `add_external_load`
（staging→GPU）→ `hold_restored_blocks`（插 hash + pin，防止分块间隙被驱逐）→
释放 staging、前进。全部完成后 `release_restored_hold` + `finish_restore`
（删文件）、标记 restored。staging 竞争时 restore 优先，spill 用剩余。

### 5.4 失败/中断
- append/read/load 失败：`abort_store`/`discard` 删半成品、释放 staging 与
  hold 块、按会话计一次 drop；请求走重算，无残留索引。
- 请求在 restore 中途中止：已恢复 chunk 保留为缓存（`release_restored_hold`），
  其余 `discard`，不再继续读。
- 崩溃：文件原子 rename + `.commit` 标记；`CLEAN_START=1` 启动清目录。
- spill 未 commit 前不可 restore（不入索引）；resume 落在 spill 中途的罕见情形
  走重算（文档记录）。

---

## 6. 实测

### 6.1 盘带宽（本机）

| 介质 | O_DIRECT 写 | O_DIRECT 读 | 备注 |
|---|---|---|---|
| NVMe（Colorful CN600 476 GiB） | **0.93 GiB/s** | **1.68 GiB/s** | 系统盘，171 GiB 空闲 |
| tmpfs（`/dev/shm`） | 2.51 GiB/s | 5.68 GiB/s | 用于功能矩阵，无磨损 |

### 6.2 单元测试（149 passed，2026-09 更新）

- `tests/v1/core/test_host_tier_ssd.py`：12 例（chunked roundtrip / abort 释放配额 /
  load range 边界 / 两档 LRU 驱逐 / `touch` 刷新 recency / 锚点数与 `snapshot()`）。
- `tests/v1/core/test_host_tier_spill.py`：16 例（release 部分释放+abort 不重复 unpin /
  hold+release restored blocks / 两档 LRU 驱逐 / `find` 刷新 recency / 锚点传播 /
  `matches_pinned_chain` 刷新 / `host_tier_info`）。
- `tests/v1/core/test_prefix_caching.py`：89 例回归。
- `tests/v1/core/test_mamba_align_chunk_split.py`：28 例（含 `_remove_blocks_in_range`
  保留 pre-cadence 锚点、恢复会话重新认领缓存锚点、抢占后不重认领、cadence 自动对齐
  到 block_size 的回归测试）。
- `tests/entrypoints/serve/host_tier/test_host_tier_api.py`：4 例（`/host_tier_info`
  的 200 / 无引擎 / 不支持 / 引擎异常 503）。

### 6.3 功能矩阵（tmpfs 假 SSD，强制分块）**18/18（1 soft）**

`VLLM_SSD_ROOT=/dev/shm/ssd_test`、quota 3e9、**staging 4e9（71 slot）、chunk 8**
（强制 50k 会话走 4-5 个 chunk；≤2e9 会触发 `cudaHostRegister` 毒化，见 §4.3）：

| 场景 | 结果 |
|---|---|
| S3 test：A→B 挤出→SSD 分块存→resume 分块 restore→深回退 | `keep2=32000 keep3=48000` ✅ |
| S3b test2：两轮停车/恢复 + 深回退 | `cycle2 keep2=32000` ✅ |
| S1/S2：baseline 与 park/resume sha | `db8b8e836881534b` ✅ |
| S2 SSD tier 生效 | `restores+1`、`cached=48000` ✅ |
| S4 配额 LRU | `stores+3`、`evictions+2`、`sessions=1` ✅ |
| S5 并发（4×36k×2 轮） | `P15-BAD []`、0 NaN ✅ |
| S3 control / NaN / 存活 | ✅ |

> 为保住 restore 后 Mamba 锚点存活，最终矩阵把 GPU 池从 106k 提到 128k token（2.77e9），
> 因此 S2 的 S+T 不再挤出 SSD（soft 项 `restores+0`，属配置放宽，非回归）；SSD park/resume
> 由 S3/S3b/S4 覆盖。矩阵须在**干净池**上先跑 S3；S3b 紧随其后、共享前缀缓存，作为回归
> smoke check（干净的机制验证见 FP8 报告的 RAM/真 NVMe `test2`）。
>
> 注：上表 S3/S3b 的 `32000/48000` 为**未开 MTP** 时的边界。部署默认 MTP3（eagle drop）下
> 边界为 `30400/46400`，见 [`vllm_02_锚点.md`](vllm_02_锚点.md) §2.3；`scripts/checks/ssd_matrix.py` 两种
> 取值均接受。

### 6.4 真 NVMe 435k 验收（分块流式）**PASS**

部署口径：`--max-model-len 435200`、`--kv-cache-memory-bytes 9000000000`（容量
**489,789 token**，1.13×）、保活 16000、锚点 32000/K=3、staging 4e9（71 slot）、
chunk 35、SSD quota 64 GiB、`MAX_MBPS=800 MiB/s`、`SSD_ONLY=1`。

`scripts/checks/ssd_435k_check.py`（S≈390k 常驻 → T≈407k 挤出 S → R=S+tail 恢复）：

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S resident | 392,670 | 0 | 826.0s | — |
| T bigger（挤出 S） | 407,690 | 0 | 917.0s | — |
| **R resumed（SSD 分块恢复）** | 392,680 | **390,400** | **46.8s** | `db8b8e836881534b` |

- 指标：`stores=2`、`restores=1`、写 **27.1 GiB**、读 **13.3 GiB**；磁盘 **13.8 GiB**。
- 恢复 wall 46.8s = SSD 读 13.3 GiB @限速800 MiB/s（~17s）+ 剩余 2,280 token prefill + GPU load
  + 32 token 输出；对比重算 826s → **17.7×**。
- 会话 = 272 attn + 9 锚点 = 281 slot ≈13.8 GiB，分 **9 个 chunk** 传输。
- 全程无 OOM、`nvidia-smi` 峰值 21.2GiB/卡以内。

> 显存标定：`9.6e9` 时容量 521,633 token 但 prefill OOM（峰值 20.96GiB/21.48，仅剩 20MiB）；
> 降到 `9.0e9` 后容量 489,789 token、峰值 ~20.4GiB，留 ~0.5-0.8GiB 余量。附带结论：本机 fp8 KV
> 下 **512k 也几乎可行**（按 17.9 KiB/token 只需 ~9.65e9 B = 8.99 GiB 池）。

> **注（2026-09-13）**：上表采集于三个锚点修复（MTP eagle-drop 对齐、head-prefix free 保留、
> 恢复会话认领旧锚点）**之前**。修复后的 435k 端到端复跑（含 restore 后深回退锚点命中）
> 见 [`reports/2026-09-435k-kv/`](../../reports/2026-09-435k-kv/README.md)：S 386,822 →
> T 162,430 挤出 → R `cached=384,000`（sha 一致）→ 深回退 `352000`。同报告还含 512k 满长
> 复验与"深回退未命中"的成因。表内数字如与复跑不一致以复跑为准。

### 6.5 中断原子性

`scripts/checks/ssd_crash_check.py`：写中 SIGKILL → 18 个完整 `.bin` + 1 个临时文件（无索引引用）；
重启 `CLEAN_START=1` 后本 `engine_id` 的 sessions 目录清空（要求 profile 固定
`KV_ENGINE_ID`，见 README §5）。`SSD-CRASH-OK`。

---

## 7. 实现期间修复的问题

**整会话版：**
1. **restore 无限重试**：SSD 会话 restore 后仍留在索引，下一步 `find` 又命中 → 每步重读
   整会话、请求永不完成。修复：`take_for_restore()` 在恢复开始时把会话移出索引，读完成后删文件。
2. **准入活锁（真卡死）**：pressure relief 用「空命中」估算 `need=44`，而准入 gate 用实际命中
   算 `need=46`，`free=44` → 请求永久卡在 gate 下。修复：`_request_remaining_blocks` 接受命中
   块/命中长度，与 gate 使用同一组参数。

**分块流式：**
3. **GPU 块竞态导致引擎崩溃**：restore 在发起 SSD 读之前检查空闲块，但在读完成后才分配；
   期间活跃请求可把块占走 → `get_new_blocks` 抛 `ValueError: Cannot get 5 free blocks from
   the pool` → EngineDead。修复：**在读开始前分配该 chunk 的 GPU 块**（检查与分配同一同步调用内）。
4. **restore 后深回退锚点失效**：mamba 锚点位于第一个 chunk，释放最早 → 位于空闲队列头部 →
   下次分配时被优先驱逐；深回退 `keep2=0 / keep3=32000`。修复：**传输顺序改为 attention 先、
   mamba 锚点最后**，释放最晚 → 队列尾部 → 保留到深回退。修复后 `keep2=32000 / keep3=48000`。

---

## 8. 限制与风险

- **分块流式已解除 staging 容量限制**：CPU 区只需 2 个 chunk；大会话在传输期间逐块释放/占用
  GPU。435k 会话 ≈13.8 GiB，限速 800 MiB/s 下写 ~17s（后台）/读 ~17s（恢复路径）。
- **restore 后的锚点已随会话认领**（三个锚点修复后）：恢复会话在首次缓存时由
  `_adopt_cached_durable_anchors` 把仍在缓存中的 cadence 状态块 `touch` 认领进 durable
  window 与保活 entry，此后生命周期与常驻一致（受 pin/K 保护、随链 spill/restore）。残留
  限制：若恢复时该锚点已被普通缓存挤出，则无法认领，该深回退点退化为重算（无正确性问题）。
- **并发抖动**：多个大会话 + 小配额会频繁 LRU（每次 park/resume 数 GB I/O）。建议配额 ≥ 并发
  大会话数，或只对空闲会话落盘。
- **消费级 NVMe 寿命**：用 `VLLM_SSD_MAX_MBPS` 限速 + 配额兜底；本方案按需 LRU。
- **无跨重启恢复**：索引在进程内，启动不扫描磁盘；`CLEAN_START=1` 只清**本 `engine_id`**
  的 sessions 目录。所以 profile 固定 `KV_ENGINE_ID`（见 README §5）：若沿用每次启动随机的
  UUID，上次的会话目录会成为孤儿（永不回收、永不清理），且 `/dev/shm` 的 staging 残留会逐次
  累积——满 7.8 GiB 后 `cudaHostRegister` 失败并毒化 CUDA context，表现为启动挂死。
- **O_DIRECT**：tmpfs 在本内核可用（probe=True），不可用时自动回退 buffered。
- 系统盘与 OS/模型权重共用：独立目录 + 限速，避免影响其他负载。
- **锚点 cadence**：C=32000/K=3 覆盖近尾 96k；更深的截断重发无锚点可用（见 02）。
- **cadence 自动对齐到 block_size**：`VLLM_MAMBA_CKPT_TOKENS` 会被**向下取整**到实际
  block_size 的整数倍（`MambaManager` 与 `scheduler` 用同一个对齐值），用户无需知道
  block_size。例：MTP3 → block 1600，32000 不变；MTP1 → block 1584，32000 → 31680。
  启动日志打印 `requested/effective/block_size`。
- **满池下深回退**：能否命中取决于恢复后常驻链是否仍在 GPU。512k MTP3（池 525,816、
  S 499,018、剩 ~26.8k slot）R 恢复后仍常驻 → V 命中 480000。池贴近上限时（余量 < ~6 块，
  如 S 515k）长 prefill 会**自我抢占**（`alloc gate ... need=3 free=0`），
  `pop_blocks_for_free` 释放 durable 窗口，近尾锚点丢失 → 深回退重算；尝试"抢占时保留锚点"
  会在重新调度时死锁（锚点占位使准入失败），故需**留足余量**（余量 ≥ 16 块 = 25,600 token）。
  余量口径用 `scripts/tools/kv_pool_sizing.py` 计算（见 `GPU_MEMORY_CALCULATION.md` §4.5）。另：
  durable 窗口现按 token 位置淘汰（保留近尾 K 个，而非最早插入的），见
  [`reports/2026-09-435k-kv/`](../../reports/2026-09-435k-kv/README.md)。
- 启动 flakiness（TP warmup CUDA invalid argument）与实现无关；重试前清理
  `/dev/shm/vllm_offload_*.mmap` 残留（失败的 worker 会留下 ~858 MiB 文件，多次失败会拖垮后续启动）。

---

## 9. 复现

```bash
# 启动（435k + SSD 分块，真实 NVMe）
bash scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh
# 标定：日志 "GPU KV cache size" >= 435200*1.05；nvidia-smi ~21G/卡

# 功能矩阵（tmpfs，强制分块：在 100k 启动配置上加）
#   VLLM_SSD_ROOT=/dev/shm/ssd_test VLLM_SSD_QUOTA_BYTES=3000000000 \
#   VLLM_SSD_CHUNK_SLOTS=8 VLLM_SSD_ONLY=1 RAMTRACE=1
#   --kv-transfer-config ... "cpu_bytes_to_use":2400000000
#   --kv-cache-memory-bytes 2770000000 --max-num-seqs 4
python scripts/checks/ssd_matrix.py

# 100k 真 NVMe（需引擎已按 §3 配置启动）
python scripts/checks/ssd_100k_check.py

# 435k 真盘验收
python scripts/checks/ssd_435k_check.py

# 中断原子性
python scripts/checks/ssd_crash_check.py

# 单元测试
cd /path/to/vllm
python -m pytest tests/v1/core/test_host_tier_ssd.py \
  tests/v1/core/test_host_tier_spill.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_mamba_align_chunk_split.py -q --noconftest
```

---

## 10. 驱逐策略

跨层两档驱逐策略（`VLLM_HOSTTIER_EVICT_SMALL_TOKENS`、`evict_sort_key`、四个选择点）的
权威描述见 [`vllm_03_offload_ram.md`](vllm_03_offload_ram.md) §9；其中 SSD 配额不足时由
`HostTierSSDStore.evict_for` 执行（in-flight 会话不驱逐）。

## 11. 指标与监控

Prometheus 指标、`scripts/tools/monitor_host_tier.py` 面板、`RAMTRACE` 轨迹与 env 配置总表见
[`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)。
