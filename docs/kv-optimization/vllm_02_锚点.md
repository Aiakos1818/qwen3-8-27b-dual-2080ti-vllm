# vLLM KV 优化 02：锚点（Mamba/GDN 持久检查点快照）

> 目标：让混合 GDN/Mamba 模型在 **revert/截断重发（从历史中间分叉续跑）** 时，能从最近的
> 持久状态快照（“锚点”）续算，而不是整段重算；并保证 restore（含 offload）之后深回退
> 仍能命中锚点。
>
> 相关文档：
> - 保活（前缀缓存保护）：[`vllm_01_保活.md`](vllm_01_保活.md)
> - RAM offload：[`vllm_03_offload_ram.md`](vllm_03_offload_ram.md)
> - SSD offload：[`vllm_04_offload_ssd.md`](vllm_04_offload_ssd.md)
> - KV 信息面板：[`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)

---

## 1. 背景与动机（revert + resend 实测）

### 1.1 真实负载形态

opencode 每次请求 = **全量历史消息 + 当前尾部**（标准 chat）。实测一次真实 revert+resend：

- revert 前（12 msg）：…`user 'ls'` → `assistant`(tool_call) → `tool`(结果)
- revert 后（10 msg）：**同 msgs[0..8] 逐字不变** + 最后一条 user 文本 `ls`→`pwd` 改写；
  该轮 assistant tool-call 与 tool 结果被删掉。

⇒ revert 语义 = **截断到被改消息之前 + 在尾部替换改写后的内容**。理论上应命中“直到被
revert 消息之前”的全部历史，只重算被改写/再生成的尾部。

### 1.2 大会话 truncating revert（关键实测）

`revert_clean.log`，每次请求为多消息历史，满历史 H 一次发完（finish → keep-alive pin）：

| 步骤 | prompt | cached | wall | 说明 |
|---|---|---|---|---|
| H 满历史（58.3k） | 58,309 | 0 | 46.9s | 全 prefill，写入并 pin |
| H 相同整段重放 | 58,309 | **56,000** | 3.1s | 整段复用 OK（尾 ~2.2k 不缓存） |
| P revert keep=8（只改最后一条 user） | 58,311 | **56,000** | 2.6s | 改写处落在从不缓存的尾部 → 不影响 |
| P revert keep=4（截断到 ~27k 历史再改写） | 30,235 | **0** | 21.5s | **前缀未复用，整段重算** |
| 再次发送同一个 P keep=4 | 30,235 | — | 崩溃 | 相同长度第 2 次 → allocator 双释放 |
| P revert keep=6 | 46k 级 | — | 卡死 | `Waiting=1, Running=0` 永久停摆 |

结论：**本 build 上跨“变短”的前缀复用（revert 典型形态）不生效**；更糟的是随后会触发
allocator 错误或调度卡死（该崩溃/卡死与其修复见 [`vllm_01_保活.md`](vllm_01_保活.md) §6）。

### 1.3 根因（HITDBG 插桩坐实）

在 `HybridKVCacheCoordinator.find_longest_cache_hit` 打印各组候选/返回值（H 58.3k resident）：

| 探针 | FullAttention 组 | Mamba 组 | 协调器 final |
|---|---|---|---|
| H 首次（无缓存） | 0 | 0 | 0 |
| H 相同整段重放 | 56000 | 56000 | **56000** ✅ |
| P keep=4 截断 30.2k | **27200** | **0** | **0** ❌ |
| P keep=8 只改尾 | 56000 | 56000 | **56000** ✅ |

⇒ **全注意力组对“任意块边界”都能复用（中间续接点 27200 可命中）；Mamba/GDN `align` 组只在
它自己落过 SSM 状态快照的边界（典型 = 上次请求的尾续接点）上能命中，历史中间没有状态快照
→ 该组返回 0 → 混合协调器取各组最小值 → 整体归零 → 整段重新 prefill。**

即：不是前缀被挤出/哈希不一致，而是 **`align` 缓存不支持“从历史中间分叉续跑”**；等长整段
重放/续写（续接点=尾边界）恰在支持集内。根因再加一层：`align` 采用 **rolling 状态**——每步
把 running state 写进“最后一块”，推进后即把上一状态块 free/null，故历史中间的边界状态是
**瞬态**，最终只有“请求尾那个快照块”留在缓存。全密集落快照不可行：58k 需 ~36 attention +
~36 mamba 快照 ≈ 72 块 > 池容量(~66)。

### 1.4 对最初问题的答案

- **设计上**：revert+resend 不应整段重算——revert 点之前逐字未变，应整段命中只 prefill 尾。
- **本 build 实测（修复前）**：几乎全部重新 prefill（短会话阈值；长会话截断 `cached=0`；
  重算后继续同链可能 EngineDead/卡死）。
- keep-alive 能保住“整段等长重放”，但**救不了截断复用** → 需要锚点。

---

## 2. A1 设计与实现：按 cadence 的持久快照

折中 = **按 cadence 做持久快照（检查点）**：

1. `scheduler._mamba_block_aligned_split`：prefill chunk 终点强制落在每个 cadence 边界
   （`VLLM_MAMBA_CKPT_TOKENS`，默认 0=关），保证该处有真实状态落块。
2. `MambaManager.remove_skipped_blocks`：不再 free 端点为 cadence 倍数的状态块，改为
   `block_pool.pin_block` 保留并留在前缀缓存（pin 后的块不可驱逐）。
3. 内存代价：每 32k 一个持久快照块，512k 链 ~16 块，可忽略。

实测（100k 池，C=32000，90k 链，全链路无崩溃）：

| 请求 | prompt | cached | wall | 说明 |
|---|---|---|---|---|
| full resident / identical replay | 86,319 | 83,200 | 4.1s | 基线正常 |
| revert 截断到 ~72k（分叉在 64k 锚点之后） | 72,296 | **64,000** | **9.3s** | 从 64k 锚点续跑，只重算 8k（原先整段 43s） |
| revert 截断到 ~30k（低于首个 32k 锚点） | 30,215 | 0 | 21s | 无锚点则该段从 0 重算 |
| full replay（8 次发散请求后） | 86,319 | 70,400 | — | 引擎存活（尾部缓存被挤，属池压力） |

启用：`VLLM_MAMBA_CKPT_TOKENS=32000`（须为 block_size 的倍数）。

### 2.1 cadence 粒度实测定优

每条 `C` 值 = 干净重启引擎一次；探针顺序 resident → 相同重放 → revert_J72k →
revert_J58k → revert_J30k → 相同重放 2：

| C (tokens) | resident | 相同重放 | revert_J72k | revert_J58k | revert_J30k | 重放2 | 结论 |
|---|---|---|---|---|---|---|---|
| 64000 | 77.7s | 83,200/4.7s | **64,000**/9.4s | 0/46.6s | 0 | 56,000 | 首锚点过高，J58k 无益 |
| 32000 | 77.5s | 83,200/4.3s | **64,000**/9.3s | **32,000**/24.2s | 0(30k<32k) | 56,000 | J58k 也能复用 |
| 16000 | **卡死**（resident 86k prefill 150s 无进度） | — | — | — | — | — | 途中持久 pin 超出 ~1.04x 头寸 → 准入死锁 |

**结论：32000 是最优且最小的可用粒度**——比 64000 多覆盖 J58k，比 16000 不会因
“单条长 prefill 运行中按 16k pin 走的块超过池余量(~4%)”而卡死。

### 2.2 修复：最迟-K 滚动窗口 + 生命周期释放

1. **最迟-K 滚动窗口**：每请求最多保留 `VLLM_MAMBA_CKPT_ANCHORS`（默认 3）个运行中锚点；
   新锚点达到上限时释放最老的一个 → 单请求在跑占用**平摊有界**，不再随 C 变小而单调抽干
   free（这正是 16k 卡死的根因）。释放走 `unpin + free`，锚点退回普通 cached 块。
2. **生命周期释放**：请求结束（`pop_blocks_for_free`）时对该请求全部锚点执行同样的
   `unpin + free` → 不产生永久 pin/池泄漏。

修复后实测（100k 池，86k 链，resident 均能完成、无卡死）：

| C (tokens) | resident | identical replay | revert J72k | revert J58k |
|---|---|---|---|---|
| 16000（原卡死） | **77.5s 完成** | 83,200/4.3s | 64,000/9.3s | **48,000**/10.3s（独立探测） |
| 8000 | **78.2s 完成** | 83,200/4.3s | 64,000/9.3s | — |

调参：`VLLM_MAMBA_CKPT_TOKENS`（cadence，须为 block_size 倍数）、
`VLLM_MAMBA_CKPT_ANCHORS`（最迟-K 窗口大小，默认 3）。

### 2.3 MTP（eagle drop）下的锚点边界（2026-09 修复）

MTP 推测解码会让 `SpeculativeConfig.use_eagle()` 返回 True，于是
`FullAttentionManager.find_longest_cache_hit` 在返回前 **多丢一个 block**
（`hit_length -= min(alignment_tokens, block_size)`）。后果：在 `C` 处截断的请求，
协调器最终只认到 `C - block_size`，而持久锚点在 `C` 上不可达 → Mamba 组退回更早的状态
（实测 28800，而不是 32000；restore 后甚至 0）。

修复（本分支）：**每个 cadence 同时保留两个锚点**——`C` 与 `C - block_size`：

1. `scheduler._mamba_block_aligned_split` 额外在 `C - block_size` 停一个 prefill chunk，
   保证该处有真实 SSM 快照；
2. `MambaManager` 的持久保留条件改为 `_is_durable_boundary(end_tokens)`：
   `end_tokens % C == 0` 或 `(end_tokens + block_size) % C == 0`；
3. 窗口上限内部翻倍（`VLLM_MAMBA_CKPT_ANCHORS × 2`），使该 env 仍表示“近尾覆盖的
   cadence 数”。

实测（FP8 100k、MTP3）：`p03 control` 常驻截断 `keep=2 → cached=30400`（修复前 28800）。

#### 2.3.1 基类 head-free 先于保留逻辑（restore 后深回退，2026-09 修复）

只保留双锚点还不够：`MambaManager.remove_skipped_blocks` 先调用基类的
`remove_skipped_blocks`，而基类按 `get_num_skipped_tokens() = num_computed_tokens - 1`
先把 head 区间整段释放（`_remove_blocks_in_range`，从高到低、遇 null 停），然后 mamba
自己的保留逻辑才去读 `last_state_block_idx`。于是 `C - block_size` 处的 pre-cadence 状态
（如 processed=32000 时释放区间 `[0,19)` 顶端的 idx=18/30400）**在被 pin 前就已被释放并置 null**，
不进保活 entry / spill → 会话经 RAM/SSD restore 后深回退为 `cached=0`。cadence 锚点
（idx=19）之所以幸存，是因为它上方 idx=20 是 null，基类在到达它之前就 break 了。

修复：`MambaManager` 覆写 `_remove_blocks_in_range`——区间内命中 `_is_durable_boundary()`
的块改为 `_retain_durable_anchor()`（pin + 入窗）而不是 `free_blocks`，其余保持原语义
（含 null-break）；只影响 mamba align 组，`RSWAManager`/FA 不受影响。保留的块不进
`freed`（保持分配 ref=1），结束由 `pin_request_auto` 统一降 ref，避免 ref 变负。

实测（FP8 100k、MTP3、RAM 与真 NVMe SSD）：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `p03 control` keep=2（常驻） | 28800 | **30400** |
| `p03 test` keep=2（RAM restore） | 0（25.8s 重算） | **30400**（1.8s） |
| `p03 test` keep=2（真 NVMe SSD restore） | 0 | **30400** |
| `ssd_matrix` S3 keep=2 / keep=3 | 0 / 46400 | **30400 / 46400** |

回归测试：`tests/v1/core/test_mamba_align_chunk_split.py::test_durable_boundaries_survive_head_free`
（去掉覆写即失败）。代价：每 cadence 多 pin 一个 pre-cadence 锚点，保活 entry 略大，
LRU 淘汰顺序变化，但命中总数不变（见 `reports/2026-09-fp8-kv/` §3.3）。

#### 2.3.2 恢复会话不认领旧锚点（二次停车后深回退，2026-09 修复）

Mamba 前缀命中只认领**一个**状态块（恢复点，如 48000）；更早的持久锚点
（30400/32000）留在公共缓存里成为"无主闲置块"——不进 `req_to_blocks`、不进窗口、
不进保活 entry，也不受 K 保护，随时可被普通缓存挤掉。后果：

- 第一次 restore 后深回退仍可能命中（锚点还躺在 GPU 缓存里，best-effort，p03 test 实测如此）；
- 但该会话再次停车时 entry 里没有锚点（续聊 34 车位 vs 新会话 40），二次 restore 后
  GPU 上的旧锚点已被挤出 → 深回退退化为整段重算（实测 `cached=0`、43s）。

修复：`MambaManager` 在请求**首次缓存且此前已有前缀命中**时调用
`_adopt_cached_durable_anchors`——按 hash 从尾向头扫 cadence 边界，把缓存中仍在的
状态块 `touch`（ref 0→1）→ `_retain_durable_anchor`（pin + 入窗，同一
`VLLM_MAMBA_CKPT_ANCHORS × 2` 上限）。之后生命周期与新会话锚点完全一致：运行期受
pin/K 保护，结束由 `take_durable_window` 交给保活 entry（34→40），spill/restore 随链走。
每请求只认领一次，且仅在**首次调度**（`num_preemptions == 0`）时认领：被抢占后的
锚点已是该请求自己的空闲块，再 pin 会把它们从空闲队列抽走，导致其下一次分配
在池边界上饿死（`need=3 avail=1` 活锁；tmpfs 矩阵 S3 实测）。

实测（FP8 100k、MTP3、RAM 与真 NVMe SSD）：`p03_restore_revert.py test2`
（A→B→A→revert→B→A→revert）两轮 `keep=2` 均 `cached=30400`（修复前第二轮 `0`）；
trace 中恢复会话 re-park 的 entry 为 `[3,3,3,30]`（含 3 组各 2 锚点），
修复前 `[1,1,1,30]`。

回归测试：`test_resumed_session_reclaims_cached_anchors`（去掉认领即失败）与
`test_preempted_request_does_not_re_adopt_anchors`（去掉 `num_preemptions == 0`
门控即失败）。
代价：续聊/回退会话的 entry 由 34 增至 40 车位，RAM 淘汰时机提前；
`offload_matrix` 的淘汰受害者随之变化（3/4 → 2/4，见 `reports/2026-09-fp8-kv/` §3.3）。

---

## 3. 语义与约束

### 3.1 最终默认值

决定因素 = 最迟-K 窗口的**近尾覆盖深度 ≈ K·C**（K 由 ~1.04x headroom 限制为 3）：

| 目标 | 依据 |
|---|---|
| **C=32000、K=3**（默认推荐） | 覆盖 ~96k ≈ 覆盖整个满池会话；近尾浅回退接近全复用；最坏回补 ≤32k 秒级 |
| C=16000、K=3 | 回补细一半，但近尾覆盖仅 ~48k，更深的回退在干净态会落空整算 |
| 更小（8k） | 覆盖仅 ~24k，除非加大 K（受 headroom 限制） |

推荐：**默认 `VLLM_MAMBA_CKPT_TOKENS=32000`、`VLLM_MAMBA_CKPT_ANCHORS=3`**。

### 3.2 锚点与缓存链“同等保护”

拍板语义：**链被保活则锚点受保护，链随压力释放则锚点一并释放，链不保活则锚点也不保活**；
不加独立上限（有界性由保活 entry 的池压力释放自然保证）。

实现（改后均有注释）：
1. `MambaManager.take_durable_window(request_id)`：请求结束时把手头存活的锚点移交，
   不再自己 release；
2. `kv_cache_manager.pin_request_auto`：在 pin 整链时把这些锚点也**并入同一保活 entry**
   （对每个锚点 `free_blocks` 使其与链块同为 pinned-idle），entry 的 token/块数把锚点计入；
   ⇒ 之后 `release_pins_smallest`（准入压力，最小先出）会连同锚点一起 unpin；
   ⇒ 无独立注册表、无永久 pin 泄漏、无需新 env 上限；
   ⇒ 若链不保活（PIN=0 / 长度不足 / abort），锚点仍随链释放。

验证（100k 大池 3.15e9）：
- **C=32000/K=3（默认）**：resident(96,762)→revert(J≈70k) ⇒ **cached=67,200 / 3.8s**；
- C=3200/K=16：adoption 生效（三组各 5 个锚点并入 entry），但该 revert 仍 `cached=0`
  ——根因是**每组长 prefill 结束时窗口里只存活最近 ~K 个锚点且都靠尾**，分叉点(J≈70k)
  超出其近尾覆盖。

限制：锚点复用受两重约束——(1) 存活窗口≈最近 ~K 个锚点，覆盖近尾约 K·C；(2) 需三组共有、
且 ≤ 分叉点。默认 C=32000/K=3 覆盖 100k 级会话的中浅回退；更深/极细 cadence 优雅降级
（整段重算，无崩溃/卡死）。

---

## 4. 容量与卡死口径

### 4.1 锚点“显存/容量”实测

结论 = **1 个 cache 块（slot）的池容量，而非新增显存**。

- 布局探针：100k 池每 worker `num_blocks=82`，KV slab = 134,348,800 B ⇒
  **每块 1,638,400 B ≈ 1.56 MiB/卡**（容量记账单位）。
- 显存 A/B：同一 86k resident，`C=0` 与 `C=32000(K=16)` 两次全新引擎，`nvidia-smi`
  全程一致：**GPU0 14,918 MiB / GPU1 14,909 MiB** ⇒ KV 启动一次性预留，**锚点不新增 GPU
  分配**；pin 时刻 `num_free_blocks` 不变（43→43）。
- 准确说法：**每锚点 = 在固定预留池里占住 1 个 block slot（≈1.56 MiB 容量/卡）**；
  卡死判据是 slot 数，不是字节。

口径备忘：
- “slot=1600 token”是**逻辑跨度**；
- “1.56 MiB/块”是**物理字节**，Mamba 锚点块只存“边界状态快照”而非 1600 token 逐 token KV；
- 早期 `free slot ≈ (cache_tokens − used)/1600` 只是近似，精确应读 `num_gpu_blocks − used`。

### 4.2 卡死判据 100k 对照验证

命题：**满上下文时若仍有 ≥ 在跑锚点数 K 的 free slot，则不会因锚点卡死。**

统一负载：近满长 resident **96,762 tokens**，`C=3200`、`K=16`、关 keep-alive。

| 运行 | KV 池(bytes) | cache tokens / 并发 | 满长时 free slot(估) | 结果 |
|---|---|---|---|---|
| R1 | 2.3e9 | 106,288 / 1.04x | ≈6 < 16 | **卡死**（GPU 0%、无完成） |
| R2 | 3.15e9（临时） | 146,470 / 1.43x | ≈31 ≥ 16 | **完成 95.0s，无卡死** |

⇒ 判据成立：free slot ≥ K（16）则 16 锚点不卡；不足则卡。

---

## 5. 512k 启用规划

- **现状即生效**：会话保活默认开（脚本显式 `VLLM_PIN_MIN_TOKENS=16000`）；崩溃修复
  (touch 守卫) 无条件生效；**Mamba 检查点默认关**（`VLLM_MAMBA_CKPT_TOKENS` 默认 0）。
- 启用检查点需在 512k 脚本前 export：`VLLM_MAMBA_CKPT_TOKENS=32000`（可加 `ANCHORS`）。
- 锚点成本几乎可忽略：K=16 ≈ 16 slot ≈ ~25 MiB/卡容量；真正约束是**运行中 free slot ≥ K**
  ——512k 现池按 ~1.0x 配，无 slot 余量，需先小步实测（临时加大 bytes 或降 MTP spec 腾位），
  确认满长 prefill + 目标 K 不卡后再定默认值。
- 待办：①（已完成 §6.1）极细 cadence 中途覆盖；②（已完成 2026-09-13）512k 满长 + 锚点实测：
  515k 满长不 OOM、SSD 分块恢复 99.6%（sha 一致）；深回退在满池下未命中（池仅剩 ~21k，
  常驻链被 spill 到 SSD，而 SSD 只认"更长/同链"，见
  [`reports/2026-09-435k-kv/`](../../reports/2026-09-435k-kv/README.md)）；③ 若需细粒度，
  将“free slot 数”从近似换成引擎真值探针。

---

## 6. 极细 cadence 根因与处置

### 6.1 根因：极细 cadence 下“结束时释放”丢弃锚点

现象：C=3200、K=16 下，resident(96,762)→revert(J≈70k) 返回 `cached=0`（59s 全量重算），
而 C=16k/32k 可复用。插桩定位：

| 实验 | revert 结果 |
|---|---|
| 默认（finish 时 unpin+free 回 cached，PIN 关） | cached=0；mamba find `deepest=-1` |
| 默认（finish 时释放，**PIN 开=16000**） | cached=0（keep-alive 不能保护被释放的锚点） |
| 诊断开关：finish 后**保持 pin** | **cached=67,200 / 3.7s**（@67200 锚点命中） |

结论：锚点创建/注册正常；丢失发生在 **release-at-finish**：极细 cadence 锚点很多、被释放后
回到 free/cached 即被后续“分配新块”按 LRU 弹出并清哈希；命中要求“三组共有的 ≤分叉点 锚点”，
任一组的被清即可整体归零。keep-alive 本身 pin 的是请求链、不含锚点，故开/关都救不了。

### 6.2 处置：锚点与缓存链“同等保护”

即 §3.2 的实现。验证（100k 大池 3.15e9）：
- **C=32000/K=3（默认）**：resident(96,762)→revert(J≈70k) ⇒ **cached=67,200 / 3.8s**；
- C=3200/K=16：adoption 生效，但该 revert 仍 `cached=0`（近尾窗口限制，见 §3.2）。

据此维持 §3.1 推荐 C≥16000。

---

## 7. 配置与指标

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_MAMBA_CKPT_TOKENS` | `0`（关） | 锚点节奏（tokens）；须为 mamba block_size 整数倍；推荐 32000 |
| `VLLM_MAMBA_CKPT_ANCHORS` | `3` | 每请求保留的“近尾 cadence”数（`max(1,·)`；每个 cadence 实际保留 `C` 与 `C-block_size` 两个锚点，内部上限 ×2；TOKENS=0 时惰性） |
| `VLLM_SPILL_NO_MAMBA` | `False` | 诊断：spill/保活时不捕获 Mamba 边界状态块 |

指标（Prometheus，细则见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)）：
`vllm:keep_alive_anchors`（保活链持有的锚点块数）、
`vllm:keep_alive_anchor_sessions`（至少含 1 锚点的保活链数）。

---

## 8. 复现（锚点相关脚本/产物）

- 矩阵驱动：`scripts/revert_lib.py`、`scripts/revert_matrix.py`、`scripts/revert_pinprobe.py`
- A1 检查点 / cadence 矩阵：`scripts/revert_ckpt.py`、`scripts/revert_cmatrix.py`
- resident 探针：`scripts/resident_once.py`、`scripts/resident_big.py`
- 结果：`logs/cmatrix_results.tsv`、`logs/anch_*.txt`（显存 A/B）
- 日志：`logs/revert_ckpt*.log`、`logs/cmatrix_*.log`、`logs/revert_clean.log`
- 崩溃/卡死修复相关（见 [`vllm_01_保活.md`](vllm_01_保活.md) §6）：`logs/server_100k.log`

测试：`tests/v1/core/test_prefix_caching.py`（89 例）、
`tests/v1/core/test_mamba_align_chunk_split.py`（23 例，含 §2.3.1 的
`test_durable_boundaries_survive_head_free` 与 §2.3.2 的
`test_resumed_session_reclaims_cached_anchors`、
`test_preempted_request_does_not_re_adopt_anchors`）等，`pytest --noconftest` 运行。
