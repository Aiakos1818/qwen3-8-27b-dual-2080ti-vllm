# vLLM KV 优化 01：会话保活（keep-alive / 前缀缓存保护）

> 对象：`Qwen3.8-27B-AWQ-INT4-yarn512k`（INT4 权重 + fp8_e4m3 KV），Qwen3.5 混合
> GDN/Mamba 架构的自研 vLLM 分支（`vllm` 源码树，0.27.2.dev0；构建见 `docs/PATCHING.md`）。
> 部署形态：2×RTX 2080 Ti（TP2）、`max-model-len 102400`、KV 池 ~106,288 tokens、
> `--enable-prefix-caching`、`--max-num-seqs 1`、chunked prefill、MTP 常开。
> 时间：2026-09。本文件记录“多会话 agent 用法下前缀缓存为何失效、以及如何保活”。
>
> 相关文档：
> - 锚点（Mamba 检查点）：[`vllm_02_锚点.md`](vllm_02_锚点.md)
> - RAM offload：[`vllm_03_offload_ram.md`](vllm_03_offload_ram.md)
> - SSD offload：[`vllm_04_offload_ssd.md`](vllm_04_offload_ssd.md)
> - KV 信息面板：[`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)

---

## 1. 背景与目标

Agent 场景：会话 A 用掉长上下文后暂停，随后会话 B/C 占用池，之后回到 A 继续。
期望：A 的 KV 前缀在暂停期间仍可复用，恢复只算新增部分。

最初问题（池 512k 假设）：A 200k + B 100k 后回 A，A 能增长到池满前不重算；
一旦第 3 会话挤压到 A 的开头，A 是否整段重算？

本仓 100k 池脚本 `scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh` 即以小尺度复现上述问题，
目标是**保活会话的 KV（pin），并让压力下的释放遵循“缓存最小的先出”**。

---

## 2. 复现与验证过程

### 2.1 100k 池启动脚本与校准

- 由 512k 脚本复制，仅改 `--max-model-len 102400`（=100×1024=102.4k，沿用 512k/240k 命名）与
  `--kv-cache-memory-bytes`。
- KV bytes 校准：`102400 token 需 ~2.05 GiB`；1.9e9 字节（1.77 GiB）只够 ~84.8k，引擎拒绝启动
  （`To serve at least one request ... estimated maximum model length is 84800`）。
  提至 **2.3e9 字节**后：日志 `GPU KV cache size: 106,288 tokens`、`Maximum concurrency 1.04x`。

### 2.2 主测试 `scripts/checks/test_kv_100k.py`（全自动）

观测手段（已确认该分支支持）：
- chat 接口 `usage.prompt_tokens_details.cached_tokens` = 从 block0 起连续命中的 token 数；
- `/metrics` 计数器、`server_100k.log` 的 `GPU KV cache usage / Prefix cache hit rate`。

阶段：warmup → A0(≈0.45P) → A0 原样 replay（期望全命中）→ B(0.35P) → C(0.4P) → replay A0
→ D1/D2(0.6P) 再挤 → replay A0。

关键结果：

| 阶段 | cached_tokens | 说明 |
|---|---|---|
| A0(47.9k) 首次 | 0 | 全 prefill ~36s |
| A0 replay | 44,800 | 前缀缓存整段复用，~3s |
| B、C 填池后 replay A0 | **0** | 整段重算 ~36s（危险） |
| D 挤空后 replay A0 | 0 | 全量重算（符合预期） |

### 2.3 定向探测（隔离复现驱逐方向）

`probe_kv_evdir`、`probe_evict_isolate`、`probe_one_page`、`probe_interleave`：

| 操作（引擎干净启动） | A(100k) replay 结果 |
|---|---|
| A 原样重发 | 97,600（≈61 页全命中，3.6s） |
| 中间插 C=2k（池未满，不驱逐） | 97,600（正常） |
| 中间插 C=4.5k（需挤 ~1-3 页） | **0**（整段重算 ~95s） |
| 中间插 C=9k（原 512k 讨论尺度） | 0 |
| 小 F=2k 穿插（A=40k、池有余量） | 38,400（不受影响） |

### 2.4 引擎级插桩结论（重要）

对 `BlockPool`（alloc/free 顺序、hash 摘除）与 scheduler（每请求命中量）加 DBG 后的判定：

1. **A 没有任何一块被正式摘哈希（无 `evict`）** —— `_remove_cached_block_hashes` 全程无
   A 页被清。
2. 但 C 运行后引擎侧 `local_hit_tokens` 从 97,600 直降 **0**（命中判定失败）。
3. **free 队列顺序正反翻转（`single_type.free` 的 `reversed`、scheduler 延迟释放）
   结果完全不变** —— 说明这不是经典的“头部 vs 尾部驱逐顺序”问题；A 的失效与
   它“哪一端被挤”无关，而是该 build **GDN/Mamba ‘align’ 实验性前缀缓存**在
   “池接近满 + 另一请求需复用少量页”时令整条链不可命中。
4. 附带结论：前缀命中只认 **从 block0 连续命中、遇 miss 即 break**，故链上任一
   前缀页失效即整链不可复用（缓存后段无法“救回”）。

→ 因此“把驱逐方向从头部改为尾部”这类单点修改**不能解决恢复全量重算**；必须让受保护
会话的页**永不进入可复用集合**（保活），或让失效根因消失（本分支暂无简单修复）。

---

## 3. 前缀缓存行为补充（实测）

> 这两节来自同 rig 的延伸调查（原 `opencode_revert_resend_前缀缓存调查.md` §2/§3），
> 属通用前缀缓存行为，与保活/锚点的动机相关。

### 3.1 缓存窗口/阈值现象

| 请求 prompt tokens | 首次 cached | 相同重放 cached |
|---|---|---|
| 2,152（短会话） | 0 | **0** |
| 5,142 | 0 | 3,200 |
| 8,138 | 3,200 | 6,400 |
| 14,144 | 6,400 | 11,200 |

- 重放命中 ≈ prompt − (~1.7–2.9k)（≈1–2 块；尾部/末端若干块不可缓存）。
- **短于约 2 个完整块（~3.2k tokens）的请求，重放 cached 恒为 0**——与 revert 无关的
  阈值效应。

### 3.2 续写（同会话增量，前缀不同尾）可行

同内容前缀、不同长度/尾部的请求（T6k 之后发 T12k）：`T12000 first cached=6400`
——证明“共享前缀 + 追加不同内容”的前缀复用存在（约 ≥5k 规模时）。

---

## 4. 方案选型与最终策略

采用（用户确认）：

- **自动保活启发式**：结束的请求若 `num_tokens >= VLLM_PIN_MIN_TOKENS`（默认 16,384），
  自动把其整条缓存前缀链 **pin**（移出可驱逐集合）。不做 API 字段（避免侵入 OpenAI 协议）。
- **整链保活**：因本 build 任一页被复用即整链失效，只保前 K tokens 无效，故 pin 整链。
- **无上限**：不设全局 pinned 上限。
- **释放唯一时机 = 准入**：新请求放不下（free 不足）时，按 **pinned token 数升序**逐条
  unpin（“缓存最小的先出”），直到放得下或全放完（之后退回 vLLM 原等待/重算语义）。
- **同会话合并**：新链若包含旧链为前缀（同会话续用/重发），用新链替换旧条目，避免
  重复保活无限增长。

---

## 5. 代码改动明细（改动处均已加注释）

分支源码：`vllm/vllm/`（克隆目录见 `docs/PATCHING.md`）

### 5.1 `v1/core/kv_cache_utils.py` — `KVCacheBlock`
- 新增字段 `pinned: int = 0`（保活 pin 计数；0=未保活）。见 `kv_cache_utils.py:125` 注释。

### 5.2 `v1/core/block_pool.py`
- `free_blocks`：`ref_cnt` 降到 0 的 **pinned** 块跳过入队（不进入 free/驱逐队列）——
  `block_pool.py:733` 注释。
- `pin_block(block)`：保活一块。若该块空闲（`ref_cnt==0`）且已在 free 队列则先
  `free_block_queue.remove` 摘链，再 `pinned += 1`；带 docstring（`block_pool.py:748`）。
- `unpin_block(block)`：释放一次保活；归零且空闲时 `append` 回 free 队列，恢复为普通
  可驱逐缓存项（`block_pool.py:763`，含断言防重复入链）。
- `evict_blocks`：跳过 pinned 块，防止 hash 侧强制失效打穿保活（`block_pool.py:795`）。

### 5.3 `v1/core/kv_cache_manager.py`
- `__init__` 新增保活注册表 `self._auto_pin_entries`（见 `kv_cache_manager.py:190` 注释）：
  每条 `{tail, blocks, grp_blocks, tokens, num_blocks, anchors, parked_at}`。
- `pin_request_auto(request)`：
  - 遍历**所有** single-type manager（attention + GDN）的 `req_to_blocks`，取仍有 hash 的
    整链块并 pin（含“任一页复用即整链失效，故逐组都保”的注释）；
  - 以 `request.block_hashes` 判断**子集替换**：旧链尾 hash 出现在新链哈希集合中（同会话
    续用/重发）→ 先 unpin 旧条目，再 pin 新链，避免重复保活无限增长。
  - 结束时把 Mamba durable 锚点并入同一 entry（见 [`vllm_02_锚点.md`](vllm_02_锚点.md)）。
- `release_pins_smallest(need_free_blocks)`：按 `tokens` 升序 unpin，直到 free ≥ 需求
  或清空（“缓存最小的先出”）。
- 辅助：`_unpin_entry` / `num_pinned_entries` / `num_pinned_tokens` / `num_pinned_blocks`
  / `num_pinned_anchors` / `num_pinned_anchor_sessions`。

### 5.4 `v1/core/sched/scheduler.py`
- 请求结束释放前调用 `_should_keep_alive(request)` → `kv_cache_manager.pin_request_auto`：
  - **必须在 free 之前**执行（此时 `req_to_blocks` 仍存活，可快照整链）——`_free_blocks`
    内注释。
  - 启发式在 `_should_keep_alive`：成功结束（`FINISHED_STOPPED` / `FINISHED_LENGTH_CAPPED`）、
    `num_tokens >= VLLM_PIN_MIN_TOKENS`（默认 0=关，脚本显式 16000）。
- 准入减压：WAITING 请求 `allocate_slots` 前，若存在保活且 `free < 剩余所需块数`，先
  `release_pins_smallest` 再分配——`# Keep-alive pressure relief` 注释段。

---

## 6. 稳健性：pin 空闲块的 adoption 守卫

> 来源：延伸调查 §5 + §8.3。此修复同时服务保活与锚点（两者都会把空闲缓存块置为
> pinned-idle 态），因此放在本篇。

**现象**：多次发散链复用/整段重算后，一次大段重算触发
`RuntimeError: remove() called on an invalid block`（`FreeBlockQueue.remove`）→ EngineDead。

**根因**：保活/检查点把空闲缓存块置为 **ref_cnt=0 且不在 free 队列**（pin 态）；后续请求
命中该块时 `touch()` 一律按“空闲在队”去 `free_block_queue.remove`，对不在队的 pin 块抛异常。

**修复**：`touch()` 对 `pinned>0` 的空闲块**跳过 unlink**（直接 ref++ 让 adopt 请求共享），
pin 持续持有该块直到 unpin/释放。

**验证**（此前必崩路径均通过、引擎存活）：
- C=16k 全矩阵（含 J30 大段重算，此前崩）：J30 `cached=16,000/10.9s`、全程无崩；
- 无检查点模式下的历史崩点：`H(58k)→P keep4(截断 0)→再次 P keep4` 此前 EngineDead，
  现在 `cached=27,200/2.6s`、引擎存活。

---

## 7. 配置与开关

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_PIN_MIN_TOKENS` | `0`（关） | 结束请求达到该 token 数即自动保活；`0` = 关闭（回归旧行为）；100k/435k 脚本显式设 `16000` |
| `VLLM_DISABLE_HOSTTIER` | `False` | 全局 kill-switch（含 RAM/SSD tier） |

无需 CLI / OpenAI 协议改动。启用保活需引擎开启前缀缓存（`--enable-prefix-caching`）。
完整 env 表见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)。

---

## 8. 验证结果（100k 池 106,288 tokens）

### 8.1 保活生效 `scripts/probes/probe_keepalive.py`（默认 16k 阈值）

| 步骤 | A 恢复 cached | 说明 |
|---|---|---|
| A(55k) 首次 | 0 | 全 prefill 43.5s |
| A replay intact | 52,800 | 全命中 |
| B(25k) 新会话（fit，自动保活） | — | — |
| **A replay AFTER B** | **52,800** | 无保活时旧行为为 0（B 会挤 A 头→整链失效） |
| C(45k)：free(A55+B25 pinned)=26k < 45k → 准入触发释放 | — | 释放**最小的 B**（25k） |
| **A replay AFTER C** | **52,800** | A 仍保活，全命中（未被释放） |

⇒ “缓存最小的先出”成立：C 被准入时释放的是 B 而非 A。

### 8.2 开关回归 `scripts/probes/probe_keepalive_off.py`（`VLLM_PIN_MIN_TOKENS=0`）

| 步骤 | A 恢复 cached |
|---|---|
| A(55k) → B(25k) → A replay | **0**（旧行为复现，全量 43s 重算） |

⇒ 特性与开关均验证有效。

### 8.3 备注
- 引擎日志全程无异常（除启动阶段 FA2 不支持 7.5 算力的常规 ERROR）。
- 部署 `scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh` 头注已同步记录方案与实测。

---

## 9. 运维缓解建议

> 来源：延伸调查 §7。部分已被锚点（[`vllm_02_锚点.md`](vllm_02_锚点.md)）取代。

- **浅 revert**：续接点≈尾边界已实测复用，约束“别深回退”；
- **深回退接受有界重算**：30k 级 ~20s，真机 512k 深回退为分钟级固有成本；
- `--no-enable-prefix-caching`：行为确定、无崩溃，代价为每轮全量；
- 深回退改走“上一条总结 + 新尾部”式压缩会话，避开中间分叉。

---

## 10. 产物清单

> 目录归拢（2026-09）：python 脚本、启动 profile 与配套配置均在 `scripts/`；
> 日志与结果由脚本写入 `LOG_DIR`（默认 `/tmp/vllm_logs`）或各自的 `*_OUT` 环境变量。

| 文件 | 作用 |
|---|---|
| `scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh` | 100k 池启动脚本（池已校准，头注含文档指针） |
| `scripts/checks/test_kv_100k.py` | 主自动化测试（全命中 / 驱逐后恢复 / 全驱逐） |
| `scripts/probes/probe_kv_100k.py` `scripts/probes/probe_kv_evdir.py` `scripts/probes/probe_evict_isolate.py` `scripts/probes/probe_one_page.py` `scripts/probes/probe_interleave.py` | 驱逐/失效定向探测 |
| `scripts/probes/probe_keepalive.py` `scripts/probes/probe_keepalive_off.py` | 保活特性验证与关闭开关回归 |
| `scripts/capture/oc_fixture.json` `scripts/capture/capture_opencode.json` `scripts/capture/capture_proxy.py` | opencode 负载捕获配置与代理 |
| `$LOG_DIR/server_100k.log`、`*_run*.log`、`test_kv_100k_results.json` | 各次运行日志/结果留档 |

---

## 11. 局限与后续方向

1. **本 build 的失效根因尚未修复**：保活是“绕开”，而非解决 GDN/Mamba ‘align’ 前缀缓存
   在池满时整链失效的根因；如需根治，建议在 `get_computed_blocks`（读侧命中）与
   `cache_blocks / move_block_hashes / CoW`（写侧）定位链失效点。
2. 自动保活无上限 + 准入释放：稳态下池会被“最大的一批会话”占据，新请求每次触发
   “放最小、跑一个、又保活它”的循环；若有大量互不相关大会话，建议后续加总预算或
   释放后再保活退避。
3. **保活指标已补齐**：`vllm:keep_alive_{entries,blocks,tokens,anchors,anchor_sessions}`
   （见 [`vllm_05_kv信息面板.md`](vllm_05_kv信息面板.md)）。
4. 迁移到上游 vLLM 时，pin 语义需按目标版本 BlockPool/调度接口重写。
