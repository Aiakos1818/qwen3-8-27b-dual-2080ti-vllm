# Qwen3.8-27B 双 2080Ti vLLM 显存使用详细计算

## 1. 硬件环境

| 项目 | 值 |
|---|---|
| GPU | 双 RTX 2080Ti 22 GiB, NVLink 互联 |
| 每卡总显存 | 22528 MiB (22.00 GiB) |
| GPU 架构 | SM75 (Turing) |
| FP8 支持 | 无原生 FP8 Tensor Core, 使用 Marlin kernel 做 weight-only FP8 |
| 系统 | Ubuntu, Python 3.12 |
| vLLM | v0.27.2.dev0+g6e448d0ea (zyYuc 自定义版本) |

## 2. 模型架构

Qwen3.8-27B 是 **混合架构 (Hybrid)** 模型，结合了传统 Attention 和线性注意力 (Mamba-like)。

### 2.1 层结构

| 层类型 | 数量 | 说明 |
|---|---|---|
| Full Attention | 16 | 每 4 层中的第 4 层 (full_attention_interval=4) |
| Linear Attention (Mamba) | 48 | Gated Delta Net, 每 4 层中的前 3 层 |
| MTP Draft | 1 | Multi-Token Prediction 推测解码层 |
| **总计** | **64 + 1 MTP** | |

### 2.2 Full Attention 层参数

| 参数 | 值 | 说明 |
|---|---|---|
| num_attention_heads | 24 | |
| num_key_value_heads | 4 | GQA, head 数 |
| head_dim | 256 | 每个 head 的维度 |
| head_size_v | 256 | V head 维度 (= head_dim) |
| dtype | fp8_e4m3 | KV Cache 量化格式, 1 byte/元素 |
| partial_rotary_factor | 0.25 | |

### 2.3 Linear Attention (Mamba/Gated Delta Net) 层参数

| 参数 | 值 | 说明 |
|---|---|---|
| linear_num_key_heads | 16 | |
| linear_num_value_heads | 48 | |
| linear_key_head_dim | 128 | |
| linear_value_head_dim | 128 | |
| linear_conv_kernel_dim | 4 | 卷积核大小 |
| mamba_ssm_dtype | float32 | 状态使用 FP32 (4 bytes) |
| num_speculative_tokens | 3 | MTP 推测解码, num_speculative_blocks=3 |

### 2.4 模型权重大小

| 项目 | 值 |
|---|---|
| 检查点总大小 (磁盘) | 28.75 GiB |
| 每卡 GPU 权重 (TP=2, 实测) | 14.96 GiB (15319 MiB) |
| 量化格式 | FP8 (Marlin kernel, weight-only) |

> 实测来源: v4 启动日志 `Model loading took 14.96 GiB` (每 worker, 即每卡)。
> 注: 此前引用的 "profiling 10.27 GiB" 数值有误，15319 MiB 才是 GPU 上真实权重占用。

## 3. KV Cache 内存计算

### 3.1 平台自动调整 Block Size

模型默认 `block_size=16`，但平台代码 (`platforms/interface.py:909`) 会自动调大：

```
attn_block_size = kernel_block_alignment_size × ceil(mamba_page_size / (kernel_block_alignment_size × attn_page_size_1_token))
```

对于此模型，block_size 被设为 **1600**，原因是：
- Mamba page size (1,634,304 bytes) 远大于默认 attention page
- 需要 attention page_size >= mamba page_size
- 日志: "Setting attention block size to 1600 tokens to ensure that attention page size is >= mamba page size."

Mamba page size 随后被 padding 到与 attention page size 相等:
- 日志: "Padding mamba page size by 0.25% to ensure that mamba page size and attention page size are exactly equal."
- 1,634,304 → 1,638,400 bytes (= 1600 tokens × 1024 B/token)

### 3.2 Full Attention 层 page_size_bytes

公式 (`kv_cache_interface.py:220`):

```
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
```

计算 (TP=2, 每卡 2 个 KV head):
```
page_size_bytes = 2 × 1600 × 2 × 256 × 1 = 1,638,400 bytes = 1.5625 MiB
```

每层每个 block 存储 K 和 V 两个 tensor, 每个大小为 `block_size × num_kv_heads × head_size × dtype_size`。

### 3.3 Linear Attention (Mamba) 层 state 计算

Gated Delta Net 的状态包含两部分 (TP=2, 每卡一半):

**Conv State (卷积状态, fp16)**:
```
conv_dim = linear_key_head_dim × linear_num_key_heads × 2 + linear_value_head_dim × linear_num_value_heads
         = 128 × 16 × 2 + 128 × 48
         = 4,096 + 6,144
         = 10,240

conv_state_shape = (conv_kernel_dim - 1 + num_spec, conv_dim / tp_size)
                 = (4 - 1 + 3, 10,240 / 2)
                 = (6, 5,120)
conv_state_numel = 6 × 5,120 = 30,720 elements  (fp16, 2 bytes)
```

**Temporal State (时间状态, fp32)**:
```
temporal_state_shape = (linear_num_value_heads / tp_size, linear_value_head_dim, linear_key_head_dim)
                     = (48 / 2, 128, 128)
                     = (24, 128, 128)
temporal_state_numel = 24 × 128 × 128 = 393,216 elements  (fp32, 4 bytes)
```

**Page Size (unpadded)**:
```
page_size_bytes = 30,720 × 2 (fp16) + 393,216 × 4 (fp32)
                = 61,440 + 1,572,864
                = 1,634,304 bytes ≈ 1.559 MiB
```

**Page Size (padded to match attention)**:
```
page_size_padded = 1,638,400 bytes = 1.5625 MiB
Padding = (1,638,400 - 1,634,304) / 1,634,304 = 0.25%
```

**Max Memory Usage (mamba_cache_mode="align", prefix caching 时默认)**:
```
max_memory_usage = page_size_padded × (2 + num_speculative_blocks)
                 = 1,638,400 × (2 + 3)
                 = 8,192,000 bytes = 7.8125 MiB   (每层)
```

源码: `kv_cache_interface.py:735-736` (align 模式)

Mamba 状态是**固定大小**，不随序列长度增长。

### 3.4 混合模型分组 (KV Cache Groups)

源码: `kv_cache_utils.py:1205-1280` (`_get_kv_cache_groups_uniform_page_size`)

**分组过程**:

1. 按 KVCacheSpec 类型分桶:
   - 桶 0: 16 个 FullAttentionSpec 层
   - 桶 1: 48 个 MambaSpec 层

2. 加上 MTP 层 (1层), 满注意力变为 17 层

3. 计算 group_size:
   ```
   min_num_layers = min(17, 48) = 17
   max_num_layers = max(17, 48) = 48
   48 >= 17 × 1.5 = 25.5 → group_size = min_num_layers = 17
   ```

4. 分组结果:
   - Full Attention: 17 层 → 1 组 (16 真实 + 1 MTP)
   - Linear Attention: 48 层 → cdiv(48, 17) = 3 组
   - 最后一组需要 padding: `17 - 48 % 17 = 17 - 14 = 3` 层
   - 浪费: 3 / 48 = **6.25%**

日志确认: "Add 3 padding layers, may waste at most 6.25% KV cache memory"

**最终分组结构**:
| 组号 | 层数 | 类型 |
|---|---|---|
| Group 0 | 17 | Full Attention (16 真实 + 1 MTP padding) |
| Group 1 | 17 | Linear Attention (16 真实 + 1 padding) |
| Group 2 | 17 | Linear Attention (16 真实 + 1 padding) |
| Group 3 | 17 | Linear Attention (16 真实 + 1 padding) |

group_size = 17, 每组 17 层共享一个 KV Cache memory pool。

### 3.5 KV Cache 需求总量计算 (精确)

源码: `kv_cache_utils.py:1939-1950` (`_max_memory_usage_bytes_from_groups` 通用分支)

```
total = group_size × page_size × blocks_needed
group_size  = 17
page_size   = 1,638,400 bytes   (unified: attention page @1600 = mamba padded page)
blocks_needed = Σ_groups cdiv(group_max_usage, page_size)
              = cdiv(attention_max, page_size) + 3 × cdiv(mamba_max, page_size)
              = ceil(max_model_len / 1600) + 3 × 5
              = blocks + 15
```

- attention group max usage = `ceil(max_model_len/1600) × 1,638,400`
  (源码 `kv_cache_interface.py:266-271`)
- 每个 mamba group max usage = `1,638,400 × 5` (align 模式, 7.35 节公式) → cdiv = 5
- `15 = 3 mamba groups × (2 + 3 spec blocks)`, 固定项

**精确需求公式**:
```
needed_bytes(max_model_len) = 27,852,800 × (ceil(max_model_len/1600) + 15)
                             (27,852,800 = 17 × 1,638,400)
```

### 3.6 公式验证 (三个独立数据点全部吻合)

| max_model_len | blocks | needed (精确) | 实测证据 |
|---|---|---|---|
| 245760 | 154 | 27,852,800 × 169 = 4,707,123,200 B = 4.384 GiB | v3-a 报错 "4.38 GiB" ✓ |
| 233600 | 146 | 27,852,800 × 161 = 4,484,300,800 B = 4.182 GiB | ≤ 4.5G available ✓ |
| 235200 | 147 | 27,852,800 × 162 = 4,512,153,600 B = 4.201 GiB | > 4.5G → 报错 "estimated max 233600" ✓ |

**262144 (164 blocks) 精确需求**:
```
needed = 27,852,800 × 179 = 4,985,651,200 bytes = 4.6433 GiB = 4,760.65 MiB
```

**结论**: `--kv-cache-memory-bytes ≥ 4,985,651,200`，取 **5,000,000,000**
(精确余量 14,348,800 B ≈ 13.7 MiB，校验为纯比较，无不确定性)。

## 4. 参数关系

### 4.1 `--gpu-memory-utilization`

源码: `worker/utils.py:414`

```python
requested_memory = ceil(total_memory × gpu_memory_utilization)
```

这是 vLLM 认为自己可用的**总 GPU 内存预算**, 覆盖模型权重 + KV Cache + CUDA Graphs + 推理临时 buffer。

| util | requested_memory (MiB) | 占总显存 |
|---|---|---|
| 0.90 | 20,275 | 90.0% |
| 0.92 | 20,726 | 92.0% |
| 0.93 | 20,951 | 93.0% |
| 0.95 | 21,402 | 95.0% |

当指定 `--kv-cache-memory-bytes` 时, 此参数仅影响 vLLM 的总预算检查, **不控制实际 KV Cache 分配**。

### 4.2 `--kv-cache-memory-bytes`

源码: `gpu_worker.py:474-496`

指定此参数时:
1. **跳过内存 profiling** (不再自动计算可用 KV Cache)
2. **直接使用指定值**作为 KV Cache 预算，**无任何扣减**
3. `profile_run` 仍会执行 (用于编译模型, 见 `gpu_worker.py:477`)，但不影响 KV 分配

**无 MM IPC 扣减的实证**:
- `mm_ipc_gpu_memory_gb` 默认值 = 0 (`config/multimodal.py:225`)
- `reserve_mm_ipc_gpu_memory` 在 `reserved_bytes <= 0` 时原样返回 (`gpu_ipc_memory.py:230-231`)
- v3-a 报错 "available KV cache memory (4.19 GiB)" = 4,500,000,000 bytes = 4.1910 GiB，**与原始值完全相等**
- (此前文档 "MM IPC 扣减 287 MiB" 是误判: 把 4.5G 报错的数字错安到了 4.8G 配置上)

### 4.3 `--max-model-len` 校验

源码: `kv_cache_utils.py:751-788` (`_check_enough_kv_cache_memory`)

启动时校验:
```
needed_memory = _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
available_memory = return_value of determine_available_memory()

if needed_memory > available_memory:
    estimated_max_len = binary_search(available_memory)
    raise ValueError(
        f"To serve at least one request with the model's max seq len ({max_model_len}), "
        f"({format_gib(needed_memory)} GiB KV cache is needed, which is larger than "
        f"the available KV cache memory ({format_gib(available_memory)} GiB). "
        f"Based on the available memory, the estimated maximum model length is {estimated_max_len}."
    )
```

### 4.4 显存布局与约束

GPU 显存布局 (nvidia-smi, GPU 0):
```
GPU Total:    22528 MiB
├─ Driver Reserved: 529 MiB  (22528 - 21965 - 34, CUDA driver 占用, 不可用)
├─ Used:          21965 MiB  (vLLM 进程全部分配)
└─ Free:             34 MiB
→ 进程实际可用上限 = 21999 MiB
```

**v3 实测分解** (245760 + KV 4.8G + batched-tokens 4096, 启动时日志 `Model loading took 14.96 GiB`):
```
Model Weights:         15319 MiB  (14.96 GiB, 每卡, 启动日志)
KV Cache:              4578 MiB  (4,800,000,000 bytes, 无扣减)
Context+NCCL+Graphs+Temp: 2068 MiB (= 21965 - 上述两项; 含 CUDA graphs 105 MiB)
非KV 基线 (batched 4096): 17387 MiB, 余量 34 MiB
```

**v4 实测分解** (262144 + KV 5.0G + batched-tokens 2048):
```
Model Weights:         15319 MiB
KV Cache:              4768 MiB  (5,000,000,000 bytes)
Context+NCCL+Graphs+Temp: 1722 MiB
非KV 基线 (batched 2048): 17041 MiB
Used: 21809 MiB, Free: 190 MiB  (比 v3 的 34 MiB 更宽裕)
```

> **batched-tokens 4096→2048 释放了 346 MiB** (17387 - 17041)，其中临时激活池
> 随 `max-num-batched-tokens` 线性增长；PyTorch 缓存分配器保留池只增不缩。

**核心约束**:
```
非KV占用 + KV Cache ≤ 21999 MiB
v4 实测: 17041 + 4768 = 21809, 余量 190 MiB ✓
```

### 4.5 durable 深回退的安全池余量 (重要)

§3.5 的公式是**启动校验的最低值**: 它把 attention 组恰好配到 `max_model_len`，
attention 组余量为 0。durable 锚点(每请求)与 chunked-prefill 的 CoW 需要额外块，
一旦请求到达池上限就会**自我抢占** (`alloc gate ... need=3 free=0`)，抢占会释放
durable 窗口 → 深回退重算。因此要**安全**运行，池必须比最低值多留 `H` 块:

```
min_pool_bytes(max_len)  = group_size × page_size × ( ceil(max_len/block_size) + mamba_blocks )
safe_pool_bytes(max_len) = group_size × page_size × ( ceil(max_len/block_size) + mamba_blocks + H )
```

- 本部署 (MTP3): `group_size=17`、`page_size=1,638,400`、`slot=27,852,800`、
  `block_size=1600`、`mamba_blocks=3×(2+num_spec)=15`。即
  `safe_pool_bytes ≈ 27,852,800 × (ceil(max_len/1600) + 15 + H)`。
- **锚点模型**: 每个 cadence 只保留 `cadence - block_size` 一个锚点（MTP/eagle 使
  full-attention 命中器丢一块，复用 cadence 会 reconcile 到 `cadence - block_size`），
  所以 `K = VLLM_MAMBA_CKPT_ANCHORS` 个锚点覆盖 `K` 个 cadence。
- **H 启发式**: `H = COW(7) + mamba_groups × K`。默认 (3 组, K=3) → **16 块**，与
  499k PASS / 515k FAIL 的实测吻合。K<3 覆盖变浅（更深回退重算）；K 过大则 H 变大、
  池可能装不下。
- 等价规则: 启动日志 `GPU KV cache size: N tokens` 必须满足
  `N ≥ max_len + H × block_size`。

**反向 (由池推最大安全上下文)**:
```
slots          = floor(pool_bytes / slot_bytes)
max_safe_len   = (slots − mamba_blocks − H) × block_size
```

**工具**: `scripts/kv_pool_sizing.py`（纯标准库；**只给启动脚本**，参数全从里面读）。
默认做两项一致性检查并给建议：
```bash
# 默认：检查 profile 的池/上下文是否匹配（两项检查）
python scripts/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh
# 可选：覆盖上下文 / 池
python scripts/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh --max-len 500k
python scripts/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh --pool-bytes 9.6e9
```
输出：
- `[池检查]`：当前池 vs `safe_pool(max-model-len)` → 不足 / 偏小（可启动但满池会自我抢占）/ 符合；不符给推荐池。
- `[上下文检查]`：`max-model-len` vs `max_safe_len(池)` → 符合 / 超出；不符给推荐上下文。**未定义池则跳过**。
- 退出码：两项均符合 0，否则 1（便于 CI）。

**`--feasible`（实际部署一次测 OOM）**：用 profile 原样启动一次（日志 `/tmp/kv_pool_sizing_feasible.log`，
健康探测每 10s，瞬态失败重试 ≤3 次，`CUDA out of memory` 不重试），健康后读 `nvidia-smi` 的
used − 池字节得**真实非KV**，随即退出部署并清理，再判定 `非KV + 安全池 ≤ 21.5 GiB`。
```bash
python scripts/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh --max-len 512k --feasible
# 不部署的替代：--log <已有启动日志>（估算）或 --non-kv-gib <实测值>
python scripts/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh \
    --max-len 512k --feasible --log /path/to/server_c5.log
```
`--log` 用 `Model loading took X GiB` + 基线(2.0 GiB) 估非KV；`--non-kv-gib` 直接给实测值
（= 运行中 `nvidia-smi` used − 池字节）。

**标定示例** (AWQ-512k, MTP3, H=16):

| max_len | 最低池 | 安全池 | 现有 profile | 结论 |
|---|---|---|---|---|
| 512,000 | 9.33e9 | **9.78e9** | 9.6e9 | ✗ 本机 9.78e9 启动 OOM；9.6e9 只能安全到 500,800 |
| 500,000 | 9.14e9 | **9.58e9** | 9.6e9 | ✓ 刚好 |
| 435,200 | 7.99e9 | **8.44e9** | 9.0e9 | ✓ 有余量 |
| 262,144 | 4.99e9 | **5.43e9** | 5.0e9 | ✗ 需提到 5.43e9 |
| 102,400 | 2.20e9 | **2.65e9** | 2.3e9 | ✗ 需提到 2.65e9 |

> 注: `block_size` 由 MTP `num_spec` 决定、与 `max_len` 无关 (MTP3→1600, MTP1→1584)，
> 从启动日志 `Setting attention block size to N tokens` 读取。换模型/TP/量化时脚本用
> `--profile` 的 `--model` 读 config.json 重算 `page_size`/`group_size`/`mamba_blocks`。
> `max-num-seqs>1` 时脚本按 `(N−1)×(attn_blocks+mamba_blocks)` 追加并发余量。


## 5. 各上下文长度内存预算表

约束 (4.4 节): `非KV占用 + KV ≤ 21999 MiB`，v3 非KV 基线 = 17387 MiB (@ batched-tokens 4096)，余量 34 MiB。

KV 需求按 3.5 节精确公式: `27,852,800 × (blocks + 15)` bytes。

| 上下文 | blocks (÷1600) | KV 需求 (精确) | kv-cache-memory-bytes | check 余量 | Δ vs v3 (4.8G) | 可行性 |
|---|---|---|---|---|---|---|
| 175,000 | 113 | 3,565,158,400 B (3.32 GiB) | 3,600,000,000 | 34.8 MiB | -1144 MiB | 可行 |
| 200,000 | 125 | 3,899,392,000 B (3.63 GiB) | 3,950,000,000 | 50.6 MiB | -810 MiB | 可行 |
| 233,600 | 146 | 4,484,300,800 B (4.18 GiB) | 4,500,000,000 | 15.7 MiB | -287 MiB | 可行 (v3-a 报错确认) |
| 245,760 | 154 | 4,707,123,200 B (4.38 GiB) | 4,800,000,000 (v3) | 92.8 MiB | — | 已验证可运行 (Free 34 MiB) |
| 262,144 | 164 | 4,985,651,200 B (4.64 GiB) | 5,000,000,000 (v4) | 13.7 MiB | **+191 MiB** | **已验证** (实测释放 346 MiB, Free 190 MiB) |

**"Δ vs v3" 为显存净增量**：KV 增大但非KV不变时为负余量，必须通过降低
`max-num-batched-tokens` (释放临时激活池) 或降低 MTP 来抵消，否则启动 OOM。
v4 实测: batched-tokens 4096→2048 释放 346 MiB ≥ 所需 157 MiB，启动成功，
且 `GPU KV cache size: 262,144 tokens` (164 blocks 精确命中)。

## 6. OOM 案例分析

### 6.1 v2: 245760 + 0.95 util + 无 kv-cache-memory-bytes

**参数**: `--max-model-len 245760 --gpu-memory-utilization 0.95`

**行为**: 自动 profiling, vLLM 将 ~95% 显存分配给模型 + KV Cache。

**结果**: OOM on first inference
```
GPU 1: Tried to allocate 56,623,104 bytes (54 MiB), free: 41,287,680 (39 MiB)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB.
```

**分析**:
- 模型 + KV Cache 占用 ~20.98 GiB
- 推理只剩 ~30 MiB, 不够 `buf10 = empty_strided_cuda((s18, 8704))` 的 54 MiB
- 根本原因: 0.95 util 太激进, 没有为推理临时 buffer 留空间

### 6.2 v3-a: 245760 + 0.90 util + 4.5G kv-cache-memory-bytes

**参数**: `--max-model-len 245760 --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 4500000000`

**结果**: 启动失败
```
ValueError: To serve at least one request with the model's max seq len (245760),
(4.38 GiB KV cache is needed, which is larger than the available KV cache memory (4.19 GiB).
Based on the available memory, the estimated maximum model length is 233600.
```

**分析**:
- `4500000000 bytes = 4.19 GiB` 即原始值，**无 MM IPC 扣减** (4.2 节)
- 245760 tokens 需要 4.38 GiB, 差 ~0.19 GiB
- 4.5G 恰好是 233600 (146 blocks) 的上限 (余量 <1 MiB)
- `gpu-memory-utilization` 设了 `kv-cache-memory-bytes` 后被跳过，不影响 KV Cache 分配

### 6.3 v3-b: 245760 + 0.92 util + 4.8G kv-cache-memory-bytes

**参数**: `--max-model-len 245760 --gpu-memory-utilization 0.92 --kv-cache-memory-bytes 4800000000`

**状态**: 已验证可运行 (GPU 利用率 84%)

**分析**:
- `4800000000 bytes = 4577.6 MiB = 4.47 GiB` 即原始值，无扣减
- 245760 (154 blocks) 需求 4.38 GiB → check 通过，余量 ~95 MiB
- 实测运行: Used 21965 MiB / Free 34 MiB，非KV 基线 17387 MiB (4.4 节)
- `gpu-memory-utilization` 在此场景下被忽略 (日志: "skipped memory profiling. This does not respect the gpu_memory_utilization config")

## 7. 推荐配置表

### 7.1 当前 profile 一览

| 模型 | 上下文 | 池 (kv-cache-memory-bytes) | 脚本 |
|---|---|---|---|
| FP8 | 180,000 | 4e9 B (3.73 GiB) | `scripts/run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh` (`MAX_MODEL_LEN=180000`) |
| AWQ-INT4 | 102,400 | 2.3e9 B | `scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh` |
| AWQ-INT4 | 262,144 | 5.6e9 B | `scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh` |
| AWQ-INT4 | 435,200 | 9.0e9 B | `scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh` |
| AWQ-INT4 | 500,800 | 9.6e9 B | `scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh` |

**池容量计算方法** (3.5 节精确公式, 无 MM IPC 扣减):
- 启动校验最低值: `27,852,800 × (ceil(max_len/1600) + 15)`
- 深回退安全值: 再加 `H` 块 (见 §4.5)；直接诊断用 `scripts/kv_pool_sizing.py <run.sh>`。

> 注: `--gpu-memory-utilization` 设了 `kv-cache-memory-bytes` 后被跳过
> (日志: "skipped memory profiling. This does not respect the gpu_memory_utilization config")，
> 此处保留仅为通过启动检查，不控制实际分配。

### 7.2 安全建议

- **核心约束** (4.4 节): `非KV占用 + KV ≤ 21999 MiB`
- 加大上下文 → KV 增大 → 必须同步削减非KV，否则启动 OOM:
  1. 降 `--max-num-batched-tokens` (4096→2048→1024): 释放临时激活池
  2. 降 MTP `num_speculative_tokens` (3→1): 释放 MTP draft (BF16) + 推测 temp
- 代价: 降 batched-tokens 使长 prompt 的 prefill 分块变慢 (TTFT 上升)，decode 不受影响
- 本机 AWQ 可启动的最大池 = 9.6e9 (8.94 GiB/卡)；9.7e9 首次请求即 OOM
  → 最大安全上下文 500,800 (见 §4.5)

### 7.3 已验证的稳定配置

| 配置 | 参数 | 状态 |
|---|---|---|
| FP8 + 180,000 | `scripts/run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh` (4e9 B, `MAX_MODEL_LEN=180000`) | 稳定 |
| AWQ-INT4 + 100k | `scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh` (2.3e9 B) | 稳定 (小池实验台) |
| AWQ-INT4 + 256k | `scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh` (5.6e9 B) | 已验证 |
| AWQ-INT4 + 435k | `scripts/run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh` (9.0e9 B) | 已验证 |
| AWQ-INT4 + 500.8k | `scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh` (9.6e9 B) | 已验证 |

### 7.4 启动 OOM 回退阶梯

若启动时 `torch.OutOfMemoryError` (KV 分配阶段)，依次尝试:
1. `--max-num-batched-tokens 1024`
2. `--kv-cache-memory-bytes` 降到该上下文的最低需求 (见 §4.5)
3. `num_speculative_tokens: 3 → 1`
4. 降 `--max-model-len`

## 附录 A: 关键源码位置

| 功能 | 文件 | 行号 |
|---|---|---|
| KV Cache 内存计算 | `vllm/v1/core/kv_cache_utils.py` | 1890-1950 |
| 内存校验 | `vllm/v1/core/kv_cache_utils.py` | 751-788 |
| 分组逻辑 | `vllm/v1/core/kv_cache_utils.py` | 1205-1280 |
| Block size 调整 | `vllm/platforms/interface.py` | 890-940 |
| gpu_memory_utilization | `vllm/v1/worker/utils.py` | 409-429 |
| kv-cache-memory-bytes | `vllm/v1/worker/gpu_worker.py` | 474-496 |
| MM IPC 扣减 | `vllm/multimodal/gpu_ipc_memory.py` | 155-188 |
| FullAttentionSpec | `vllm/v1/kv_cache_interface.py` | 266-271 |
| MambaSpec | `vllm/v1/kv_cache_interface.py` | 729-738 |
| Mamba state shape | `vllm/model_executor/layers/mamba/mamba_utils.py` | 247-268 |

## 附录 B: 模型配置 (config.json 关键字段)

```json
{
  "num_hidden_layers": 64,
  "num_attention_heads": 24,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "full_attention_interval": 4,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 48,
  "linear_conv_kernel_dim": 4,
  "mamba_ssm_dtype": "float32",
  "mtp_num_hidden_layers": 1,
  "max_position_embeddings": 262144
}
```
