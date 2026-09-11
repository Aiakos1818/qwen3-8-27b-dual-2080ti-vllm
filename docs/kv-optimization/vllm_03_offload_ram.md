# vLLM KV 优化 03：Offload 到 RAM（会话级 park / spill）

> 本文件只讲 host-tier（RAM）的 spill/restore。GPU 显存压力下把最小保活会话整链
> spill 到 `/dev/shm` 的 CPU 槽位（park-to-RAM），resume 时整链 load 回。
>
> 相关文档：
> - 保活：[`vllm_01_保活.md`](vllm_01_保活.md)
> - 锚点：[`vllm_02_锚点.md`](vllm_02_锚点.md)
> - SSD offload（两层 GPU/SSD）：[`vllm_04_offload_ssd.md`](vllm_04_offload_ssd.md)
> - KV 信息面板：[`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)

## 1. 目标与语义

在 100k 会话、单并发、长会话反复续聊/回退的场景下，GPU KV 池被“保活 pin 的整链”占满时，
新请求只能淘汰旧的保活会话。原行为是**直接释放回收**（内容丢失，回来时整段重算）。
本特性把“淘汰”改为**先整链存到 CPU 内存（RAM），再释放 GPU 块**；被淘汰会话回来时
**整链从 RAM 载回 GPU 并登记为前缀缓存**，从而以“秒级”恢复代替“整段重算”。

语义（与用户逐条确认）：
1. 保活链正常情况下仍 pinned 在 GPU，行为不变。
2. 准入压力淘汰最小保活会话时：**先 store 到 RAM，store 完成后才 unpin/free GPU 块**（正确性优先）。
3. 会话回来（新请求是其续写/重发）→ 从 RAM 整链 load 回 GPU。
4. RAM 满时：
   - 从 RAM 中**最小**的（非保护）会话开始丢；
   - load X 时 **X 永不丢**（先丢 X 以外的 RAM 会话）；
   - 仍装不下则放弃被淘汰的会话（优雅降级为整段重算）。
5. 淘汰顺序：GPU 侧候选按**从小到大**选出（够腾地方即可），store 按**从大到小**处理
   （大者先存、先腾 GPU，且优先保住大会话）。**跨层驱逐策略（两档）见 §9。**
6. PIN=0（无保活）或无 offload 配置时，行为与旧路径完全一致。

## 2. 关键调研结论：为什么自建 tier

vLLM fork 自带两条 CPU offload 连接器（`OffloadingConnector`/`CPUOffloadingSpec`、
`SimpleCPUOffloadConnector`），但**都不能直接满足本场景**：

- 二者都是**请求生命周期驱动**：只对“运行中的请求”做 store/load，且 CPU 侧按键/块 LRU
  管理；对“已完成 → 保活 → 被淘汰”的空闲整链不可见。
- 实测（PIN=0，A86k→B90k 覆盖→resume A）：连接器**确实 store 且驻留**了 A 的链，但
  resume 的 CPU lookup 对 **hybrid/mamba 链**无法恢复（跨组对齐/可达性限制），
  仍然整段重算。
- 因此：**自建存储/命中/驱逐，连接器只当跨进程异步拷贝通道**。

## 3. 架构与数据流

```
准入压力 (scheduler.schedule)
  └─ _spill_keepalive_entries(need, protect=X)
       ├─ take_spill_candidates: 从最小保活 entry 依次取出（仍保持 GPU pin）
       ├─ _build_spill_store_job: 逐组 GPU block_ids + group_sizes/indices
       ├─ alloc_ram_slots(n)（不足则 evict_ram_for 丢最小非 X）
       └─ connector.add_external_store(src=GPU, dst=CPU slots)   # 入队一次
  ...
每步 build_connector_meta: 把 external_store 塞进 store_jobs（仅一次）
worker.prepare_store_kv/start_kv_transfers: CPUOffloadingWorker.submit_store
  ...
完成经 completed_jobs 回 scheduler._drain_spill_completions
       └─ kv_cache_manager.confirm_spill: unpin/free GPU 块 + 记录 _ram_sessions

resume X (fresh request 前缀匹配)
  └─ scheduler._maybe_begin_restore(X)
       ├─ find_ram_session(X)（tail/前缀匹配）
       ├─ free < need → 先 spill 最小保活（protect=X）
       ├─ allocate_restore_blocks + connector.add_external_load(CPU slots → GPU blocks)
       └─ 本步跳过请求（re-queue），load 完成后再调度
  ...
_drain_spill_completions: register_restored_blocks
       └─ 给新块写回原 hash 并放回空闲缓存池 → 普通前缀命中，仅算尾段
```

要点：
- store 未完成前 GPU 块不下发（`_spill_hold` 保持 pinned），避免被提前覆写。
- CPU 槽粒度 = GPU 块（`blocks_per_chunk=1`），单会话占 `len(链)` 个槽。
- 被淘汰会话的“可还原信息”只保留**不可变元数据**（各组 hash、长度）+ CPU 槽，不引用会被复用的块对象。

## 4. 配置

启用（在启动命令追加）：
```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"cpu_bytes_to_use":<bytes>}}'
```
- `cpu_bytes_to_use` 决定 CPU 槽容量（`num_blocks = bytes / 每槽字节`，`blocks_per_chunk` 保持 1）。
  想缓存 K 个 ~90k 会话，取 `K × 单会话 GPU KV 字节`（实测单会话 ≈55–58 槽）。
- `VLLM_PIN_MIN_TOKENS` 默认 0（保活**默认关**，需显式 opt-in；0 时无保活 entry，
  本特性也无 spill 候选，等于不触发）。100k 启动脚本显式设为 16000 以保持部署行为。
- host-tier 启用门控（三重与门；`VLLM_DISABLE_HOSTTIER` 只是 kill switch）：
  | connector | `cpu_bytes_to_use` | `VLLM_DISABLE_HOSTTIER` | 结果 |
  |---|---|---|---|
  | 无 | — | — | 关（默认；不进 host-tier 分支） |
  | 有 | 必填且 >0 | 未设/False | 开 |
  | 有 | 必填且 >0 | 任意非空（含 `"0"`） | 关 |
- `/dev/shm` 需容纳 `cpu_bytes_to_use`（本机 15 GiB RAM，注意上限）。
- 调试：`RAMTRACE=1` 输出到 `$RAMTRACE_LOG`（默认 `/tmp/vllm_ramtrace.log`；spill/evict/restore 轨迹）。

### 4.1 容量对照（本机：2×2080Ti、fp8_e4m3、TP2、block_size 1600）
| `cpu_bytes_to_use` | CPU 槽数（MTP off） | CPU 槽数（MTP on） |
|---|---|---|
| 1.0e9 | 19 | — |
| 1.6e9 | — | 28 |
| 1.9e9 | 36 | 34 |
| 2.4e9 | — | 43 |

观测 ≈ **54 MiB/槽**（每槽 1 个 GPU 块；会话槽数 = `ceil(tokens/1600)`，约 50k 会话 ≈ 32 槽）。
按目标驻留会话数 N 配置：`cpu_bytes_to_use ≈ N × 1.8e9`（对应 ~50k 会话、留 ~10% 余量）。

> **锚点开销**：开启 `VLLM_MAMBA_CKPT_TOKENS` 后，spill 会连同 Mamba cadence 锚点一起存储
> （这样 restore 后的深回退仍能命中锚点，见 §6c）。每会话额外占用最多
> `VLLM_MAMBA_CKPT_ANCHORS`（默认 3）个槽 × mamba 组数（本模型 3）≈ 最多 +9 槽。
> 例：50k 会话 = 31 attn + 3 边界状态 + 最多 3 锚点 ≈ 37 槽，故一个 50k 会话需
> `cpu_bytes_to_use ≳ 2.1e9`。不带锚点则仍为 ~34 槽。

## 5. 代码地图

| 文件 | 关键点 |
|---|---|
| `vllm/v1/core/kv_cache_manager.py` | pin 时按组捕获 `grp_blocks`；`_spill_hold`/`_ram_sessions`；`set_ram_capacity`/`alloc_ram_slots`/`free_ram_slots`；`take_spill_candidates`/`abort_spill`/`confirm_spill`；`evict_ram_for`；`find_ram_session`/`allocate_restore_blocks`/`register_restored_blocks` |
| `vllm/v1/core/sched/scheduler.py` | `_spill_keepalive_entries`、`_build_spill_store_job`、`_maybe_begin_restore`、`_drain_spill_completions`；准入压力接线；restore 请求跳过 `hit_diverged` 重对齐 |
| `.../kv_connector/v1/offloading/common.py` | `OffloadingConnectorMetadata.external_load_jobs` |
| `.../offloading/scheduler.py` | 外部 store/load 入队（**每步只注入一次**）、按 worker 完成计数回收、`take_external_completed` |
| `.../offloading/worker.py` | 执行外部 load，但**不经 `finished_recving`**（避免 base scheduler 断言） |
| `.../offloading_connector.py` | `cpu_capacity()`、`add_external_{store,load}`、`take_external_completed` |

均带注释、默认关闭、语法/引擎实测通过。

## 6. 实测结果

环境：2×RTX 2080Ti、100k 池（`--kv-cache-memory-bytes 2300000000`）、单并发。

### 6a. 无 MTP（A/B/C 大会话，3.7 GiB tier）
| 场景 | 结果 |
|---|---|
| A(86,319)→B(92,271) | A 被 spill：`job=28 n_slots=55` → `confirm_spill` → B 完成（83s），引擎健康 |
| resume A | restore → **cached=86,240 / 1.0s**（原整段 76s） |
| A,B,C(96,762) + RAM 满 | `ram evict dropped=A`（最小非保护）→ B park → 引擎健康 |
| resume B | restore → **cached=90,944 / 2.6s** |

### 6b. 开 MTP（100k 池 + ~50k tier，`cpu_bytes_to_use=1.9e9` → 容量 34 槽）
会话 s1/s2/s3 各 ≈50,238 tokens（≈31 槽），ping-pong + RAM 满：

| 步骤 | 结果 |
|---|---|
| s1 / s2 | 各一次 spill（`job n_slots=31 tokens=49600`）→ 正常完成（~39s） |
| **resume s1** | restore → **cached=48,000 / 2.7s** ✅ |
| s3 | 触发对旧会话的 spill/park → 正常完成 |
| **resume s2** | restore → **cached=48,000 / 2.5s** ✅ |
| resume s3 / s1b / s2b | cached=0（该会话已被 RAM 满策略丢弃）→ 整段重算 ~39s（优雅降级） |
| 普通保活复用（不触发 spill，19k 会话 resume） | **cached=17,600 / 2.1s** ✅（`trusted_local` 修复） |

全程：引擎保持 200、无死锁；CPU 占用恒 ≤ 34 槽；store 完成前 GPU 块保持 pinned。
测试脚本：`scripts/offload_matrix.py`。

### 6c. 输出正确性校验（greedy 生成比对）
对同一提示，比较“显存命中”与“spill→restore”两条路径的 greedy 输出（`scripts/correctness_check.py`）：

| 配置 | baseline（显存命中） | offload（restore） | 结论 |
|---|---|---|---|
| 无 MTP | sha `db8b8e83…` / `\n\nOK` | sha `db8b8e83…` / `\n\nOK` | ✅ 一致 |
| 开 MTP | sha `db8b8e83…` / `\n\nOK` | sha `db8b8e83…` / `\n\nOK` | ✅ 一致 |

开 MTP 的 offload 场景：`cached=48000`、`num_nans_in_logits=0`。

**restore × 锚点（深回退）**：`VLLM_MAMBA_CKPT_TOKENS=32000` 下，
A(~48k) 被 B 挤出→spill(37 槽，含锚点)→resume restore(`cached=48000`)→
**截断回退 keep=2 命中 `cached=32000`**（与常驻控制组一致），keep=3 `cached=48000`。
`n_slots` 含锚点：`grp_sizes=[2,2,2,31]`。

> 历史：修复前开 MTP restore 会产出 **NaN logits**。根因是**自建 host-tier 的 CPU 槽位与
> 连接器自身 native offload 的分配器冲突**（共用同一 CPU 缓冲，连接器 native store 覆盖
> parked 槽位）。修复：启用 host-tier 时置 `connector_scheduler.native_store_enabled = False`。
> 详见 §8 第 1 条。

### 6d. 回归（PIN=0 / 无 connector / kill-switch / 并发）
| 配置 | 结果 |
|---|---|
| 无 connector（plain） | prefix 复用正常（R `cached=48,000`，sha `db8b8e83…`），启动 ~180s |
| connector + `VLLM_PIN_MIN_TOKENS=0` | **无 spill/restore 轨迹**；resume `cached=0` 整段重算，输出正确，引擎 200 |
| MTP + CKPT 深回退（大池 3.15e9，J≈70k） | revert `cached=67,200 / 3.8s` |
| `VLLM_DISABLE_HOSTTIER=1` | `ram capacity slots=0`、**无 spill/restore**；输出 sha 与 baseline 一致 |
| 并发压力（`max-num-seqs 4`，4×~36k×2 轮） | 14+ 次 spill/1 restore；`Corrupted: 0`、`NAN=0`、两轮 sha 一致、引擎 200 |

结论：关掉保活、不用 connector、或 kill-switch 时，行为与旧路径一致，无回归；并发下无数据竞争。

> 注：本表数据采集于“三功能默认全关”收尾之前；收尾后不设任何 env 且无 connector 即全默认，
> 100k 脚本显式开保活/锚点以保持原行为。

## 7. 限制与待办

- **与连接器 native offload 互斥（已自动处理）**：启用 host-tier 时置
  `connector_scheduler.native_store_enabled = False`，连接器仅作拷贝通道。
- `trusted_local`：仅当 restore 链未发散时才跳过 connector 的 hybrid 重对齐；一旦发散即
  reconcile（可能整段重算，但保证正确、避免 NaN）。
- **容量开销**：锚点随会话 spill，一个 50k 会话 ≈ 37 槽（§4.1）；tier 小于该值时该会话
  无法 park（直接释放、优雅降级）。
- **并发抖动**：多个大会话 + 小 tier（只装 1 个）时，准入压力会 spill↔evict 反复抖动；
  功能正确（0 NaN/0 corrupted），但吞吐下降；建议 tier 容量 ≥ 并发大会话数。
- Mamba/GDN 边界状态块 + cadence 锚点均纳入 spill（`pin_request_auto`）；锚点与边界块
  重合时去重，避免重复 unpin。EAGLE 下取边界再下一块以避免 spec 状态污染。
- **观测**：Prometheus 指标见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)。
- CPU 侧未做跨进程多 rank 共享（直接用 worker 私有 CPU 缓冲，按 rank 一致槽号拷贝）。

## 8. 收尾计划（2026-09）

1. **MTP/EAGLE 解锁（已完成）**
   - 结论：开 MTP 下 baseline 与 restore 的 greedy 输出一致（sha `db8b8e83…`）、
     `num_nans_in_logits=0`，门控已移除，host-tier 在 MTP 下默认启用。
   - **真根因**：自建 host-tier 的 CPU 槽位与连接器自身 native offload 的分配器**冲突**
     ——两者共用同一 CPU 缓冲，连接器在请求结束时的 native store 会覆盖我们 parked 的槽位
     （含 mamba 状态块），restore 读到被覆盖的数据 → GDN/注意力 NaN。
   - **修复**：启用 host-tier 时 `connector_scheduler.native_store_enabled = False`
     （仅作拷贝通道）；修复后 store/load 校验和端到端一致、0 NaN、输出一致。
2. **本轮收尾修复（全部已实现并实测）**
   - **restore 源槽位持有到 load 完成**：把 slots 存进 `_restore_jobs`，
     `_drain_spill_completions` 完成后再释放。
   - **准入死锁修复**：`_get_num_evictable_blocks` 排除 `pinned > 0`（pinned 保活块不算可驱逐）。
   - **锚点重复 unpin 崩溃**：边界状态块同时是 cadence 锚点时按对象去重。
   - **abort 清理**：restore 中途 abort 不再残留 `_restored_req_ids`；spill 入队失败回滚槽位。
   - **锚点纳入 spill**：`pin_request_auto` 把锚点并入 per-group 链，restore 后深回退仍命中锚点（§6c）。
3. **深回退(junction revert)+ MTP 组合** —— 已验证（§6d：大池 J≈70k `cached=67,200/3.8s`）。
4. **`cpu_bytes_to_use` ↔ 会话容量对照表** —— 已补（§4.1，含锚点开销）。
5. **观测指标** —— 已加（见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)）。
6. **单元测试** —— `tests/v1/core/test_host_tier_spill.py`，`pytest --noconftest` 运行。
7. PIN=0 全回归、无 MTP 全回归、kill-switch、并发压力均已完成（§6c/§6d）。
8. **默认值收尾（2026-09，已完成）**：三功能全部默认关闭；保活改为显式 opt-in
   （`VLLM_PIN_MIN_TOKENS` 默认由 16000 改为 0，100k 脚本显式 export 16000）。
   7 个新变量统一注册进 `vllm/envs.py`（见 §8.1）。

### 8.1 环境变量一览（均已注册进 `vllm/envs.py`，默认全关）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_PIN_MIN_TOKENS` | `0`（关） | 保活阈值；>0 时结束且 tokens≥阈值的请求整链 pin（显式 opt-in） |
| `VLLM_MAMBA_CKPT_TOKENS` | `0`（关） | 锚点节奏（tokens）；须为 mamba block_size 整数倍 |
| `VLLM_MAMBA_CKPT_ANCHORS` | `3` | 每请求锚点上限（`max(1,·)`；TOKENS=0 时惰性） |
| `VLLM_DISABLE_HOSTTIER` | `False` | 仅 kill switch；启用仍需 connector + `cpu_bytes_to_use`（§4 门控表） |
| `VLLM_SPILL_NO_MAMBA` | `False` | 诊断：spill/保活时不捕获 Mamba 边界状态块 |
| `VLLM_RESTORE_TRUST` | `False` | 诊断：信任刚 restore 的链，跳过 connector 重对齐 |
| `RAMTRACE` | `False` | 诊断：写 host-tier spill/restore 轨迹（下述） |
| `RAMTRACE_LOG` | `/tmp/vllm_ramtrace.log` | 轨迹输出路径 |

- `RAMTRACE=1` → `$RAMTRACE_LOG`（默认 `/tmp/vllm_ramtrace.log`）：`diag lookup`(逐组命中)、`sched`(n_new/local/look/nblocks)、
  `pin g*`(各组块数)、`spill groups`/`register_restored`(逐组块数)、`mamba cap`(捕获的
  GDN 状态块)、`restore blocks`(还原块 id)、`NAN target`(目标 logits 的 NaN 数)、
  `alloc gate`/`mamba defer`(准入/对齐阻塞原因)。
- `VLLM_DISABLE_HOSTTIER=1`：整体关闭 host-tier（退回直接释放，已实测）。
- `VLLM_SPILL_NO_MAMBA=1`：关闭 Mamba 边界状态捕获（诊断用）。
- `VLLM_RESTORE_TRUST=1`：恢复对“刚 restore 链”的信任（旧行为；诊断用）。

---

## 9. 跨层驱逐策略（两档）——权威描述

host-tier 的所有“挑被驱逐会话”的决策统一为**两档**策略，由
`VLLM_HOSTTIER_EVICT_SMALL_TOKENS`（默认 64000，0 = 关闭分档）划分：

- 档 0（`tokens < 阈值`，小会话，重算便宜）：优先驱逐。
- 档 1（`tokens >= 阈值`）：小会话耗尽后才动。
- 每档内按**最旧优先**（`parked_at` / `last_used`）。

统一入口 `evict_sort_key(tokens, age)`（`host_tier_ssd.py`），作用于：

| 选择点 | 位置 | 说明 |
|---|---|---|
| GPU 保活会话 → host（spill） | `kv_cache_manager.take_spill_candidates` | 准入压力、restore 腾 GPU 共用 |
| restore 前腾 GPU 的驱逐列表 | 同上（`_spill_keepalive_entries` 调用） | 先算要驱逐的会话，再逐个 spill |
| RAM 空间不足释放 | `kv_cache_manager.evict_ram_for` | `protect`（恢复目标 X）不驱逐 |
| SSD 配额不足释放 | `HostTierSSDStore.evict_for` | in-flight 会话不驱逐（详见 04） |

年龄字段：GPU 保活条目与 RAM 会话在 park 时写 `parked_at`（`time.monotonic()`）；
SSD 会话用既有 `last_used`。启动日志
`HostTierSSD: staging_slots=.. chunk_slots=.. evict_small=<n> block_size=<n>` 打印生效阈值。
435k 生产脚本取 `VLLM_HOSTTIER_EVICT_SMALL_TOKENS=32000`（对齐保活 16k / 锚点 32k）。

**测试**：`test_evict_ram_for_two_tier_then_oldest`、
`test_take_spill_candidates_two_tier_then_oldest`、`test_evict_prefers_small_then_oldest`；
`VLLM_HOSTTIER_EVICT_SMALL_TOKENS=0` 退化为纯最旧优先（已手工核验）。

> 说明：`_spill_keepalive_entries` 对已选出的集合仍按“大→小”顺序发起 store
> 任务（每次 store 释放最多 GPU、并给大会话最大 host 空间机会），该处理顺序
> 与上面的**选择**顺序无关。

SSD 侧的配额驱逐与分块流式实现见 [`vllm_04_offload_ssd.md`](vllm_04_offload_ssd.md)。
