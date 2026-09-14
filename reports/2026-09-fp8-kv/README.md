# FP8 权重 × KV 优化：100K 实测报告（2026-09）

> 目的：验证 [KV 优化分支](../..)（会话保活 / Mamba 锚点 / GPU↔RAM/SSD 分层 offload / 观测面板）
> 在 **FP8 权重**模型上是否照常工作，并给出切换量化类型所需的适配。
>
> 结论先行：**KV 优化与权重量化方式无关**。从 AWQ-INT4 换到 FP8 只需改模型路径 +
> `--quantization fp8`、重标定 KV 池预算、并把 venv 的 `ninja` 放进 PATH（FP8 会启用
> `norm_quant`/`act_quant` 融合，需 JIT 编译）。**无需改动任何 KV 优化代码**。
> 测试中发现并修复了三个锚点问题（详见 §5）：**MTP eagle-drop 导致锚点边界错位**
> （常驻深回退 `keep=2` 由 28800 提升到 30400）、**基类 head-free 先于锚点保留逻辑
> 释放 pre-cadence 状态**（restore 后深回退由 `cached=0` 修复为 `cached=30400`）、
> **恢复会话不认领旧锚点**（二次停车后深回退由 `cached=0` 修复为 `cached=30400`）。

## 1. 环境与配置

| 项 | 值 |
|---|---|
| 权重 | `Qwen3.8-27B-FP8`（block-wise dynamic e4m3，`weight_block_size=[128,128]`，磁盘 29 GB） |
| 量化 | `--quantization fp8`（对比：AWQ 用 compressed-tensors 自动识别） |
| KV dtype | `fp8_e4m3`（未变） |
| GPU / TP | 2 × RTX 2080 Ti 22GB，TP=2 |
| vLLM | 0.27.2.dev0+g6e448d0ea（含 KV 补丁） |
| MTP | method=mtp，num_speculative_tokens=3，draft layer quantization=fp8 |
| profile | `scripts/run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh` |
| 池 | `--kv-cache-memory-bytes 2300000000`、`--max-model-len 102400`、`max-num-seqs 1` |
| KV 开关 | `VLLM_PIN_MIN_TOKENS=16000`、`VLLM_MAMBA_CKPT_TOKENS=32000`、`ANCHORS=3`、`VLLM_HOSTTIER_EVICT_SMALL_TOKENS=32000` |

## 2. 标定（启动日志）

- 权重显存：`Model loading took 14.96 GiB`（每卡）。AWQ-INT4 为 10.51 GiB → **+4.45 GiB/卡**。
- KV 容量：`GPU KV cache size: 106,288 tokens`（与 AWQ 同池同值 → KV dtype 未变，容量公式不变）。
- block size 仍自动定为 `1600`，mamba page padding `0.25%`。
- FP8 专属：`Enabled custom fusions: norm_quant, act_quant`；`FlashInfer ... kv_cache_dtype=torch.float8_e4m3fn, arch=sm75`。
- 因此切换权重后 **只需重标定 KV 池字节数**（权重变大、留给 KV 的空间变小），KV 侧的
  block/mamba page/容量公式全部不变。

完整启动日志片段：`fp8_100k_startup.txt`。

## 3. Phase 2 — 保活 + 锚点 + RAM parking

引擎：100k 池、`cpu_bytes_to_use=4e9`（71 槽）、RAM parking（无 SSD）。

### 3.1 正确性（greedy 输出 sha）

| 模式 | prompt | cached | out | sha | head |
|---|---|---|---|---|---|
| baseline（resident） | 48152 | 46400 | 55 | `db8b8e836881534b` | `\n\nOK` |
| offload（S→T 挤出→restore） | 48152 | 46400 | 55 | `db8b8e836881534b` | `\n\nOK` |

两条路径 sha 完全一致，且与 AWQ 模型相同（`db8b8e836881534b`）→ RAM restore 字节正确、`NAN=0`。
（注：`scripts/checks/correctness_check.py` 新增 `KV_CHECK_MAX_TOKENS` 环境变量，默认 32；FP8 在该 prompt 下需
64 才能产出非空 content，本测试用 64。）

### 3.2 RAM spill/restore 指标

P3 代码（真 LRU 淘汰）在干净引擎上复测 `scripts/checks/correctness_check.py offload`：`R-restored cached=46400`、
sha 一致；`host_tier_spills=1, restores=1, evictions=0, drops=1`；`keep_alive_entries=1,
tokens=62400, anchors=6, anchor_sessions=1`。（修复 1–2 时代的旧值 `spills=2, restores=1,
drops=0`，随负载/时序略有差异，正确性不变。）

### 3.3 RAM offload 矩阵（`scripts/checks/offload_matrix.py`）

| 请求 | prompt | cached（修复 1–2 后 / 修复 3 后） |
|---|---|---|
| s1 / s2 / s3 | 50238 | 0（冷） |
| resume-s1 | 50255 | **48000** / **48000** |
| resume-s2 | 50255 | **48000** / **48000** |
| resume-s3 | 50255 | 0 / 0 |
| resume-s1b | 50255 | 0 / 0 |
| resume-s2b | 50255 | **48000** / 0 |

两个版本各自在**全新引擎**上复跑（`ramtrace_fix6d.log` / `ramtrace_prefix3.log`）。
三个锚点修复后保活 entry 变大（新会话 40 车位；续聊会话由 34 增到 40），RAM 满时的
淘汰受害者随之移动：`resume-s2b` 由中变丢，**命中总数 3/4 → 2/4**。
这是"续聊会话携带锚点"的既定代价（淘汰按时间，见
`docs/kv-optimization/vllm_02_锚点.md` §2.3.2）；矩阵是 71 槽容量的边界压力测试，
命中数对 entry 大小敏感，不是正确性指标。单会话生产配置（`max-num-seqs=1`）不受影响。

P3 真 LRU 代码复跑矩阵（`fp8_100k_offload_matrix_p3.txt`）同样 **2/4**：LRU 只改变同档内
受害者的选择依据（最近使用），不改变"entry 变大 → 满池边界受害者移动"的结果。

### 3.4 锚点（Mamba 深回退）

- `p03 control`（A 常驻 → 截断回退）：`keep=2 → cached=30400`（MTP eagle-drop 后的边界；
  修复前为 28800，见 §5），`keep=3 → cached=46400`。
- `p03 test`（A → B 挤出 → A restore → 截断回退）：`keep=2 → cached=30400`（修复前 0），
  `keep=3 → cached=46400`（见 §5）。
- `p03 test2`（A → B → A → 回退 → B → A → 回退，两轮停车/恢复）：两轮
  `keep=2 → cached=30400`（修复 3 前第二轮 0，见 §5.3）。

## 4. Phase 3 — SSD 分块 offload

### 4.1 真 NVMe 100K（`scripts/checks/ssd_100k_check.py`）

引擎：`VLLM_SSD_ROOT=<NVMe>/ssd_kv`、quota 8 GiB、`MAX_MBPS=800`、`SSD_ONLY=1`、
`staging_slots=71 / chunk_slots=35`、`O_DIRECT=True`。

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S resident | 84241 | 0 | 88.1s | — |
| T bigger（挤出 S） | 87217 | 0 | 93.5s | — |
| **R resumed（SSD restore）** | 84251 | **81600** | **10.7s** | `db8b8e836881534b` |

`stores=4, restores=1`（本次紧跟 `test2` 运行，多出的一次 store 来自该测试的会话；
首次干净运行时为 `stores=2`）；写 **9.34 GiB**、读 **2.85 GiB**；restore 后磁盘文件已删（`DISK_BYTES 0`）。
restore sha 与 baseline 一致 → **SSD 分块 park/resume 字节正确**。真 NVMe 上的二次停车/恢复
深回退另见 `fp8_100k_p03_test2_ssd.txt`（两轮 restore `cached=46400`、回退 `30400`）。

### 4.2 中断原子性（`scripts/checks/ssd_crash_check.py`）

写中 SIGKILL → `finals=19 temps=1`（无索引引用半成品）；`clean_start` 后目录清空。**`SSD-CRASH-OK`**。

### 4.3 tmpfs 强制分块矩阵（`scripts/checks/ssd_matrix.py`）

**18/18 PASS（1 soft）**：

| 检查 | 结果 |
|---|---|
| S1 baseline sha | `db8b8e836881534b` ✓ |
| S2 SSD park/resume sha 与 baseline 一致 | ✓（`cached=46400`；`restores+0` 为 soft，见下注） |
| S4 配额 LRU | ✓（`stores+3, evictions+3, sessions=1`） |
| S5 并发（4×36k×2 轮） | ✓ `P15-BAD []`，0 mismatch |
| no NaN logits | ✓ `nans=0` |
| **S3 restore + 深回退锚点** | ✓ `keep2=30400 keep3=46400`（修复前 `keep2=0`） |
| **S3b 二次停车/恢复 + 深回退锚点** | ✓ `cycle2 keep2=30400` |

> 复跑注意：`VLLM_SSD_ROOT` 指向 tmpfs 时，若 `CPU_BYTES_TO_USE` 过小（如 `1e9`，
> staging 17 槽），本机（kernel 7.0 / 驱动 580）上 TP1 的 `cudaHostRegister` 会返回
> `cudaErrorInvalidValue`，进而毒化 CUDA context 使 warmup 失败。把 staging 调大
> （`4e9`，71 槽）即可稳定启动；`VLLM_SSD_CHUNK_SLOTS=8` 仍强制分块传输。
>
> **更新（2026-09-14）**：真实 SSD（非 tmpfs）下同样复现；实测阈值在 35~43 槽之间——
> `2e9`（35 槽）失败、**`2.4e9`（43 槽）稳定**。当前 profile 取 `cpu_bytes_to_use=2.4e9`
> （chunk 21），两实例共存时 `/dev/shm` 占用 2×2.4e9 ≈ 4.5 GiB。
>
> S2 的 `restores+0`（soft）与 §6.3 同因：为保住 restore 后 Mamba 锚点存活，矩阵把 GPU
> 池提到 128k token，S+T 不再挤出 SSD；SSD park/resume 由 S3/S3b/S4 覆盖。S3b 紧跟 S3
> 运行、共享前缀缓存，作为回归 smoke check；干净的机制验证见 §3.4 的 RAM `test2`。

## 5. 三个锚点问题（均已修复）

### 5.1 MTP eagle-drop 导致锚点边界错位（常驻深回退）

MTP 推测解码让 `SpeculativeConfig.use_eagle()` 返回 True，于是
`FullAttentionManager.find_longest_cache_hit` 返回前 **多丢一个 block**
（`hit_length -= min(alignment_tokens, block_size)`）：

```
DBG FA find max_len=32136 pre_drop=32000 drop_eagle=True post=30400
DBG coord group idx=0 FullAttentionSpec new_hit=30400
DBG coord group idx=1 MambaSpec          max_len=30400 new_hit=28800
DBG coord final hit=28800
```

FA 组本来命中了 32000，但 eagle drop 把它压到 **30400**；协调器再拿 30400 去查 Mamba，
持久锚点在 32000 不可达 → 退回 28800。**与 FP8 无关**（`use_eagle()` 只看 `method`）。

修复：每个 cadence 同时保留 `C` 与 `C - block_size` 两个锚点。scheduler 额外在
`C - block_size` 停一个 prefill chunk，`MambaManager` 的持久保留条件同步覆盖该边界，
窗口上限内部 ×2。常驻截断 **`keep=2 → cached=30400`**（修复前 28800）。

### 5.2 基类 head-free 先于锚点保留逻辑释放 pre-cadence 状态（restore 后深回退）

修复 §5.1 后，`p03 control` 正常，但 `p03 test` / 矩阵 S3（会话经 RAM/SSD restore 后深回退）
仍为 `cached=0`。插桩确认：pre-cadence 状态块（如 30400 / idx=18）在 `remove_skipped_blocks`
的 mamba 保留逻辑读它之前就已被置 null。

根因在基类调用顺序：

```
MambaManager.remove_skipped_blocks()
 ├─ super().remove_skipped_blocks()      # 先跑，释放 head 区间
 │    num_skipped = get_num_skipped_tokens() = num_computed_tokens - 1
 │    _remove_blocks_in_range(0, num_skipped // block_size)   # 从高到低，遇 null 停
 └─ if align: 读 blocks[last_state_block_idx]   # 后跑，块已被基类释放
```

processed=32000 时释放区间为 `[0,19)`，pre-cadence 状态块 idx=18（30400）正好在区间顶端
→ 被基类先 `free_blocks` + `null`，随后 mamba 保留逻辑读到 null 而跳过。cadence 锚点
idx=19 之所以幸存，是因为它上面的 idx=20 是 null，基类在到达它之前就 break 了。

修复（`MambaManager._remove_blocks_in_range` 覆写）：在该区间内遇到
`_is_durable_boundary()` 的块时，改为 `_retain_durable_anchor()`（pin + 入窗）而不是释放，
其余保持原语义（含 null-break）；该覆写只影响 mamba align 组，`RSWAManager` 不受影响。

### 5.3 恢复会话不认领旧锚点（二次停车后深回退）

§5.2 修复后 `p03 test` 单轮通过，但会话**第二次**停车/恢复后深回退仍退化为整段重算。
根因：Mamba 前缀命中只认领**一个**状态块（恢复点，如 48000）；30400/32000 的旧锚点
留在公共缓存里成为"无主闲置块"——不进 `req_to_blocks`、不进窗口、不进保活 entry，
不受 K 保护，会被普通缓存挤出。第一次 restore 后深回退仍能命中只是"锚点还躺在 GPU
缓存里"的 best-effort；再次停车时 entry 里没有锚点（续聊 34 车位 vs 新会话 40），
二次 restore 后旧锚点已被挤出 → `cached=0`（43s 重算）。

修复：`MambaManager` 在请求**首次缓存且此前已有前缀命中**时调用
`_adopt_cached_durable_anchors`——按 hash 从尾向头扫 cadence 边界，把缓存中仍在的
状态块 `touch`（ref 0→1）→ `_retain_durable_anchor`（pin + 入窗，同一
`VLLM_MAMBA_CKPT_ANCHORS × 2` 上限）。此后生命周期与新会话锚点完全一致：运行期受
pin/K 保护，结束由 `take_durable_window` 交给保活 entry（34→40），spill/restore 随链走。
只认领一次，且仅在**首次调度**（`num_preemptions == 0`）时认领——被抢占后的锚点已是
该请求自己的空闲块，再 pin 会把它们从空闲队列抽走，使其下一次分配在池边界上饿死
（tmpfs 矩阵 S3 实测 `need=3 avail=1` 活锁，加门控后消失）。

实测（RAM、trace `ramtrace_fix6c.log`）：`test2` 两轮 `keep=2` 均 `cached=30400`
（修复前第二轮 0）；恢复会话 re-park 的 entry 为 `[3,3,3,30]`（3 组各 2 锚点 + 兜底），
修复前 `[1,1,1,30]`；`adopt anchor` 日志可见每轮恢复认领 6 个锚点。

### 5.4 验证

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `p03 control` keep=2（常驻） | 28800 | **30400** |
| `p03 test` keep=2（RAM restore） | 0（25.8s 重算） | **30400**（1.8s） |
| `p03 test` keep=2（真 NVMe SSD restore） | 0 | **30400** |
| `p03 test2` keep=2 第二轮（RAM） | 0（43s 重算） | **30400**（1.9s） |
| 矩阵 S3 keep=2 / keep=3 | 0 / 46400 | **30400 / 46400** |
| 矩阵 S3b 第二轮 keep=2 | — | **30400**（见 §4.3） |

正确性不变：`scripts/checks/correctness_check.py baseline/offload` sha 均为 `db8b8e836881534b`、
0 NaN；`ssd_100k_check` restore sha 一致；单元测试新增
`test_durable_boundaries_survive_head_free`、
`test_resumed_session_reclaims_cached_anchors` 与
`test_preempted_request_does_not_re_adopt_anchors`（各自去掉对应修复即失败）。

代价：每 cadence 多 pin 一个 pre-cadence 锚点；续聊/回退会话的 entry 由 34 增至
40 车位，RAM 淘汰时机提前（矩阵命中总数 3/4 → 2/4，见 §3.3）；窗口上限仍为
`VLLM_MAMBA_CKPT_ANCHORS × 2`。

## 6. 切换量化类型的适配清单（实测确认）

1. `MODEL_PATH` → FP8 目录；`--quantization fp8`（AWQ 可省略，自动识别）。
2. **重标定 `KV_CACHE_MEMORY_BYTES`**：权重每卡 +4.45 GiB → 非KV基线变大；100k 池沿用
   `2.3e9` 仍够（KV 容量 106,288 tokens 不变）。435k 在 FP8 下不可行（权重 ~15 GiB +
   KV ~8.4 GiB > 21.5 GiB）。
3. **`ninja` 必须在 PATH**：FP8 启用 `norm_quant`/`act_quant` 融合，Triton 需要 `ninja`；
   profile 已把 `dirname $VLLM_PYTHON` 前置到 PATH（这是 FP8 与 AWQ 的一个关键差异）。
4. `VLLM_QWOPUS_MTP_BF16_DRAFT=1` 保留；draft 层按 target 的 fp8 量化加载。
5. KV 优化 env / 补丁 / 脚本**零改动**。
6. 正确性 sha 因模型权重不同需按模型重新基线（本 FP8 恰好与 AWQ 相同：`db8b8e83…`）。

## 7. 复现

```bash
# 启动 FP8 100K + KV 优化（RAM parking）
cp config/vllm-fp8-100k.env.example .env   # 填 MODEL_PATH / VLLM_PYTHON / FLASHQLA_PATH
bash scripts/run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh
# 启用真 NVMe SSD tier：在 .env 设 VLLM_SSD_ROOT / VLLM_SSD_ONLY=1 / quota / MAX_MBPS

export MODEL_PATH=<fp8 model> SERVED_MODEL_NAME=qwen38-27b \
  VLLM_PYTHON=<venv python> RAMTRACE_LOG=<path> KV_CHECK_MAX_TOKENS=64
python scripts/checks/correctness_check.py baseline
python scripts/checks/correctness_check.py offload
python scripts/checks/ssd_100k_check.py
python scripts/checks/ssd_matrix.py --expected-sha db8b8e836881534b   # tmpfs root 时 staging 用 4e9，见 §4.3
python scripts/checks/ssd_crash_check.py
```

`scripts/checks/p03_restore_revert.py` 支持 `control|test|test2` 三种模式（`test2` 为两轮停车/恢复）。

单元测试（140 passed，含 pre-cadence 保留、恢复会话认领、抢占不重认领、真 LRU 三个回归测试）：

```bash
cd /path/to/vllm
python -m pytest tests/v1/core/test_host_tier_ssd.py \
  tests/v1/core/test_host_tier_spill.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_mamba_align_chunk_split.py -q --noconftest
```

## 8. 已知启动抖动

1. FP8 路径下 TP worker warmup 偶发 `torch.AcceleratorError: CUDA error: invalid argument`
   （`qwen_triton_warmup`），与 AWQ 的已知 flakiness 相同，重试即可；重试前清
   `/dev/shm/vllm_offload_*.mmap` 与 `/dev/shm/sem.mp-*`。
2. `cudaHostRegister`（CPU staging mmap）失败会**毒化 CUDA context**：即使代码只打 warning
   继续，后续 `torch.full` 等调用也会以同一 `invalid argument` 失败。tmpfs SSD root +
   过小 staging（`CPU_BYTES_TO_USE=1e9`）在本机稳定触发 rank=1 注册失败；调大 staging
   或换真 NVMe root 即可（见 §4.3 注）。
   两者均与 KV 优化实现无关。

## 9. 结果文件

- `fp8_100k_startup.txt`：三套引擎的启动标定日志。
- `fp8_100k_correctness_baseline.txt` / `fp8_100k_correctness_offload.txt`
- `fp8_100k_offload_matrix.txt`（最终代码、post-gate；修复 1–2 后对照见 §3.3）
- `fp8_100k_p03_control.txt` / `fp8_100k_p03_test.txt` / `fp8_100k_p03_test2.txt`
- `fp8_100k_p03_test2_ssd.txt`（真 NVMe SSD 两轮停车/恢复）
- `fp8_100k_ssd_100k_check.txt`
- `fp8_100k_ssd_crash_check.txt`
- `fp8_100k_ssd_matrix.txt`
