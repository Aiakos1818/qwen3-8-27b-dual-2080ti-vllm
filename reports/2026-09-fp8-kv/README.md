# FP8 权重 × KV 优化：100K 实测报告（2026-09）

> 目的：验证 [KV 优化分支](../..)（会话保活 / Mamba 锚点 / GPU↔RAM/SSD 分层 offload / 观测面板）
> 在 **FP8 权重**模型上是否照常工作，并给出切换量化类型所需的适配。
>
> 结论先行：**KV 优化与权重量化方式无关**。从 AWQ-INT4 换到 FP8 只需改模型路径 +
> `--quantization fp8`、重标定 KV 池预算、并把 venv 的 `ninja` 放进 PATH（FP8 会启用
> `norm_quant`/`act_quant` 融合，需 JIT 编译）。**无需改动任何 KV 优化代码**。
> 测试中发现并修复了一个 **MTP eagle-drop 导致锚点边界错位**的问题（常驻深回退
> `keep=2` 由 28800 提升到 30400）；restore 后的深回退仍受 MTP speculative-block
> 复用限制（详见 §5）。

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
（注：`correctness_check.py` 新增 `KV_CHECK_MAX_TOKENS` 环境变量，默认 32；FP8 在该 prompt 下需
64 才能产出非空 content，本测试用 64。）

### 3.2 RAM spill/restore 指标

`host_tier_spills=2, restores=1, drops=0, evictions=0`；`keep_alive_entries=1, tokens=52800`。

### 3.3 RAM offload 矩阵（`offload_matrix.py`）

| 请求 | prompt | cached |
|---|---|---|
| s1 / s2 / s3 | 50238 | 0（冷） |
| resume-s1 | 50255 | **48000** |
| resume-s2 | 50255 | 0（被 RAM 满策略丢弃） |
| resume-s3 | 50255 | **48000** |
| resume-s1b | 50255 | **48000** |
| resume-s2b | 50255 | 0（丢弃） |

`spills=8, restores=4, evictions=2, drops=2` —— 与 AWQ 文档 §6b 的语义/量级一致。

### 3.4 锚点（Mamba 深回退）

- `p03 control`（A 常驻 → 截断回退）：`keep=2 → cached=30400`（MTP eagle-drop 后的边界；
  修复前为 28800，见 §5），`keep=3 → cached=46400`。
- `p03 test`（A → B 挤出 → A restore → 截断回退）：`keep=2 → cached=0`（restore 限制，见 §5）。

## 4. Phase 3 — SSD 分块 offload

### 4.1 真 NVMe 100K（`ssd_100k_check.py`）

引擎：`VLLM_SSD_ROOT=<NVMe>/ssd_kv`、quota 8 GiB、`MAX_MBPS=800`、`SSD_ONLY=1`、
`staging_slots=71 / chunk_slots=35`、`O_DIRECT=True`。

| 请求 | prompt | cached | wall | sha |
|---|---|---|---|---|
| S resident | 84241 | 0 | 82.1s | — |
| T bigger（挤出 S） | 87217 | 0 | 88.0s | — |
| **R resumed（SSD restore）** | 84251 | **81600** | **12.7s** | `db8b8e836881534b` |

`stores=2, restores=1`；写 **6.43 GiB**、读 **3.16 GiB**；restore 后磁盘文件已删（`DISK_BYTES 0`）。
restore sha 与 baseline 一致 → **SSD 分块 park/resume 字节正确**。

### 4.2 中断原子性（`ssd_crash_check.py`）

写中 SIGKILL → `finals=19 temps=1`（无索引引用半成品）；`clean_start` 后目录清空。**`SSD-CRASH-OK`**。

### 4.3 tmpfs 强制分块矩阵（`ssd_matrix.py`，staging 17 / chunk 8）

**16/17 PASS（1 soft）**：

| 检查 | 结果 |
|---|---|
| S1 baseline sha | `db8b8e836881534b` ✓ |
| S2 SSD park/resume sha 与 baseline 一致 | ✓（`restores+1`, `cached=46400`） |
| S4 配额 LRU | ✓（`stores+3, evictions+2, sessions=1`） |
| S5 并发（4×36k×2 轮） | ✓ `P15-BAD []`，0 mismatch |
| no NaN logits | ✓ `nans=0` |
| **S3 restore + 深回退锚点** | ✗ `keep2=0 keep3=46400`（见 §5） |

## 5. 唯一行为差异：restore 后的深回退锚点（已定位根因，常驻侧已修复）

`p03 test` 与矩阵 S3 均显示：**会话经 RAM/SSD restore 后，`keep=2` 的截断回退不再命中
Mamba 锚点（cached=0）**。

### 5.1 根因（插桩确认）

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
持久锚点在 32000 不可达 → 退回 28800（restore 后连 28800 也没有 → 0）。
**与 FP8 无关**（`use_eagle()` 只看 `method`，AWQ + MTP 同样如此）。

### 5.2 修复（常驻侧）

每个 cadence 同时保留 `C` 与 `C - block_size` 两个锚点：scheduler 额外在
`C - block_size` 停一个 prefill chunk，`MambaManager` 的持久保留条件同步覆盖该边界，
窗口上限内部 ×2。见 [`vllm_02_锚点.md`](../../docs/kv-optimization/vllm_02_锚点.md) §2.3。

- `p03 control`（常驻截断）：**`keep=2 → cached=30400`**（修复前 28800），`keep=3 → 46400`。

### 5.3 仍存在的限制

pre-cadence 状态（`C - block_size`）位于 MTP 的 speculative-block 复用窗口内，会在被 pin
前就被复用覆盖，因此**不进入保活 entry / spill**。所以 **restore + 深回退仍为 `cached=0`**。

- 影响：restore 后的深回退退化为整段重算（**无正确性问题**，sha 仍一致、0 NaN）；
  **浅回退（keep=3，尾部）正常命中**（`cached=46400`）。
- 不修复该点需要改动 MTP 的 speculative-block 分配/复用路径，风险较高，留作后续工作。
  不影响保活、RAM/SSD park/resume、常驻深回退与浅回退。

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
python scripts/correctness_check.py baseline
python scripts/correctness_check.py offload
python scripts/ssd_100k_check.py
python scripts/ssd_matrix.py --expected-sha db8b8e836881534b
python scripts/ssd_crash_check.py
```

单元测试（110 passed）：

```bash
cd zyYuc-sandbox/src/vllm-0271
python -m pytest tests/v1/core/test_host_tier_ssd.py \
  tests/v1/core/test_host_tier_spill.py tests/v1/core/test_prefix_caching.py -q --noconftest
```

## 8. 已知启动抖动

FP8 路径下 TP worker warmup 偶发 `torch.AcceleratorError: CUDA error: invalid argument`
（`qwen_triton_warmup`），与 AWQ 的已知 flakiness 相同，重试即可（本次多次重试后成功）；
重试前清 `/dev/shm/vllm_offload_*.mmap`。与 KV 优化实现无关。

## 9. 结果文件

- `fp8_100k_startup.txt`：三套引擎的启动标定日志。
- `fp8_100k_correctness_baseline.txt` / `fp8_100k_correctness_offload.txt`
- `fp8_100k_offload_matrix.txt`
- `fp8_100k_p03_control.txt` / `fp8_100k_p03_test.txt`
- `fp8_100k_ssd_100k_check.txt`
- `fp8_100k_ssd_crash_check.txt`
- `fp8_100k_ssd_matrix.txt`
