# 上游分支（`sm75-upstream`）：开分支 / 移植 / 部署记录

本文记录"在**上游 vLLM main** 上重建 SM75 部署"这条线的工作：分支怎么开的、
移植了什么、编译环境怎么修的、怎么部署、128K 下 offload 实测结果。

---

## 1. 分支结构

| 项 | 值 |
|---|---|
| 源码目录 | `/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/src/vllm-0271` |
| 新分支 | `sm75-upstream`（基于上游 `main`） |

提交序列：

```
bf78fc276  kv offload: clear the sticky CUDA error via the runtime, not the torch binding
737fea73b  kv offload: clear the sticky CUDA error after a failed cudaHostRegister
49f68ba24  sm75/qwen3.8: port SM75 deployment changes to upstream main
fbf2c5e8b  [Frontend] Add per-request metrics to Responses API (#55084)   ← 上游 main 基点
```

- `49f68ba24`：9 文件、+235/−19（见 §2）。
- `737fea73b` → `bf78fc276`：上一条补丁用错 API（`torch._C._cudart` 没有
  `cudaGetLastError`），在它本该恢复的失败路径上抛 `AttributeError` 把 worker 打崩；
  后一条改为经 `ctypes` 从已加载的 CUDA runtime 取符号（见 §5.1）。

---

## 2. 移植范围：只搬 SM75 部署改动，不搬 KV 层

`49f68ba24` 只包含让 SM75 + Qwen3.8 跑起来所必需的 9 个文件：

| 文件 | 作用 |
|---|---|
| `vllm/config/reasoning.py` | `default_thinking_token_budget` 等 reasoning 配置 |
| `vllm/envs.py` | 新增 `VLLM_QWOPUS_MTP_BF16_DRAFT`、`VLLM_SM75_SPEC_SYNC_MODE`（`auto`/`safe`/`nosync`） |
| `.../layers/mamba/gdn/qwen_gdn_linear_attn.py` | FlashQLA legacy（SM70/SM75）GDN prefill kernel 接入 + Triton decode 回退 |
| `vllm/model_executor/models/qwen3_5_mtp.py` | Qwen3.5 MTP draft 的 bf16 加载路径 |
| `vllm/v1/attention/backends/flashinfer.py` | FlashInfer 在 SM75 上的支持判定 |
| `vllm/v1/attention/backends/gdn_attn.py` | GDN attention backend 选择 |
| `vllm/v1/engine/input_processor.py` | 输入处理相关适配 |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | SM75 采样路径 |
| `vllm/v1/worker/gpu_model_runner.py` | SM75 spec-decode 同步等 runner 适配 |

移植范围只限于"让 SM75 + Qwen3.8 跑起来"所必需的改动：不使用任何额外优化层，
长会话复用完全交给上游自带的 **tiering offload**（见 §5）。

---

## 3. 编译环境（本机实测，均为 venv 内改动，不入 git）

| 项 | 值 |
|---|---|
| venv | `/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/venv` |
| 安装版本 | `vllm 0.26.1rc1.dev2278+g49f68ba24`（editable） |
| torch / transformers | `2.13.0+cu130` / `5.16.1` |
| FlashInfer | `0.6.18.post1`（原 `0.6.16.post3`） |
| tilelang / apache-tvm-ffi | `0.1.12` / `0.1.11` |
| CUDA toolkit（pip） | `cuda-toolkit 13.0.3.0`；`nvidia-cuda-nvcc/nvvm/crt/cudart` **13.0.88**（runtime 13.0.96） |

为编译成功所做的修补：

1. **CUDA 工具链降级**：`nvidia-cu13` 的 nvcc/nvvm/crt 从 13.3 降到 **13.0.88**（13.3 的头文件与 torch cu130 不匹配）。
2. **补 20 个无版本号 `.so` 链接**于 `.../nvidia/cu13/lib`（`libcudart.so` → `libcudart.so.13` 之类），否则链接期找不到。
3. `.../nvidia/cu13/lib64` → `lib` 符号链接。
4. **`.deps` 路径重写**：160 个文件里残留的 `/home/aiakos/zyYuc-sandbox` 改为 `/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox`。
5. **手动 clone cutlass 源码**到 `.deps/cutlass-src`（`v4.7.1`，HEAD `cb4247394`），配合 `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` 规避离线构建时的下载。

构建参数：`TORCH_CUDA_ARCH_LIST=7.5`、`MAX_JOBS=6`、`CUDA_HOME` 指向 pip 的 cu13，耗时约 **52 分钟**。
可复现脚本：`scripts/setup/build_vllm.sh`（`VLLM_SRC=<vLLM checkout> bash …`；
`CHECK_ONLY=1` 只跑预检，会核对上表 5 项修补是否到位）。

本机 FlashQLA checkout 取分支 `sm75-qwen3.8` 的 HEAD（`7c30b56`，已含本地 SM75 改动，
即 `patches/flashqla-sm70-sm75-local.patch` 的内容），以 editable 方式装在当前 checkout
路径上（`pip check` 干净）。

`flash_qla` 的 `setup.py` 已把 `tilelang` / `apache-tvm-ffi` 从 `==0.1.8` / `==0.1.9`
放宽为 `>=`：内核按这两个版本开发，但在 tilelang 0.1.12 + apache-tvm-ffi 0.1.11
（与 flashinfer 0.6.18 共存）上实测可跑，硬钉会让 pip 报冲突
（见 `docs/environment-lock.md`）。重装时用 `pip install -e . --no-deps`，避免把
tilelang 降级回去。

---

## 4. 部署

### 4.1 启动 profile

| 脚本 | 用途 |
|---|---|
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k.sh` | 基础 profile：500.8K 上下文，无 offload，9.6e9 池 |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_128k_ssd.sh` | 128K 上下文 + 上游两层 offload（CPU+disk），**本文主要验证对象** |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_16k_ssd.sh` | 16K + 小池 offload 快速实验台（秒级触发驱逐/恢复） |
| `config/vllm-128k-ssd.env.example` | 128K profile 的配置模板 |

三个 profile 都从 `.env` 取路径（`MODEL_PATH` / `VLLM_PYTHON` / `FLASHQLA_PATH` /
`CHAT_TEMPLATE`），profile 参数在脚本内有默认值、可用环境变量覆盖。

共同参数：`--dtype half`、TP=2、`--device-ids 0,1`、`--kv-cache-dtype fp8_e4m3`、
`--gdn_prefill_backend=flashqla_legacy`、MTP `num_speculative_tokens=3`、
`--max-num-seqs 1`、`--max-num-batched-tokens 1024` + chunked prefill。

### 4.2 实测启动与后端确认

| 配置 | 冷启到 `/health` 200 |
|---|---|
| 最小配置（9.6e9 池，500800） | **144 s** |
| 128K + offload（3.0e9 池，CPU 4.6e9） | **146 s** |

日志确认：`Using FlashQLA legacy (SM70/SM75) GDN prefill kernel`、
`Using FLASHINFER attention backend out of ['FLASHINFER', 'TRITON_ATTN']`、
MTP acceptance ~79%、工具调用模板与 `qwen3_xml` parser 正常。

### 4.3 池容量（实测，用于核对 `kv-cache-memory-bytes`）

| `max-model-len` | 池 | 实测 `GPU KV cache size` | 并发 |
|---|---|---|---|
| 500,800 | 9.6e9 | 525,229 tokens | 1.05× |
| 131,072 | 3.0e9 | 144,584 tokens | 1.10× |

---

## 5. 128K 上游 offload 实测

配置：池 3.0e9（144,584 tokens）、CPU staging 见下、fs 磁盘层 `/home/aiakos/Qwen3.8-27B-Deploy/ssd_kv`。
测试方式：发 A 建立会话 → 发 B（不同内容）把 A 挤出 GPU → 重发 A，看 `cached_tokens`。

### 5.1 关键前提：`cudaHostRegister` 与粘性错误

本机 `RLIMIT_MEMLOCK` 仅 **8192 KB（软硬都是）**，staging 区域注册失败时上游只打
warning、**不清除 CUDA 粘性错误**，于是该线程的下一次 CUDA 调用（JIT warmup 的
`torch.full`）会以 `CUDA error: invalid argument` 崩掉。

修复即 §1 的两条提交：失败后调用 `cudaGetLastError()` 清除。注意
`torch.cuda.cudart()` **只暴露 9 个符号**（`cudaError`、`cudaGetErrorString`、
`cudaHostRegister`、`cudaHostUnregister`、`cudaMemGetInfo`、`cudaProfilerStart/Stop`、
`cudaStreamCreate/Destroy`），没有 `cudaGetLastError`，所以必须经 `ctypes` 从已加载的
runtime 取符号。已复现验证：强制注册失败后，不清理则 `torch.full` 报
`CUDA error: operation not supported`，清理后正常。

实测注册成功/失败都出现过：staging **2.4e9 时失败**（走 unpinned DMA），
**4.6e9 时成功**（走 pinned）。两种情况都能跑，但成功后恢复明显更快。

### 5.2 主结论：CPU staging 层必须装得下**整条链**

上游的 tiering 会把**促销（promote）回来的 chunk 保留在 CPU 层**，而不是只把它当
流式 bounce buffer。因此：

```
CPU_BYTES_TO_USE >= ceil(MAX_MODEL_LEN / 1600) × 55.8 MB
                                  ↑ 实测 chunk 几何：55.8 MB/chunk，1 chunk = 1 block = 1600 tokens
```

128K 需要 **82 chunks ≈ 4.58e9**。装不下时，`primary_tier.prepare_write()` 返回
None（`vllm/v1/kv_offload/tiering/manager.py:455`），计数
`vllm:kv_offload_tiering_promotion_allocation_failures`，**该次恢复整体作废**。

### 5.3 实测结果

| 场景 | 链大小 vs CPU 层 | A 重发 `cached_tokens` | 耗时 |
|---|---|---|---|
| 10K × 2，池 0.9e9 / CPU 2.4e9 | 7 blocks ≤ 43 | **8,000 / 10,004（80%）** | 1.5 s |
| 40K × 4，池 3.0e9 / CPU 2.4e9 | 25 blocks ≤ 43 | **36,800 / 39,170（94%）** | **3 s**（冷启 31 s） |
| 120K × 2，池 3.0e9 / CPU **2.4e9** | 75 blocks > 43 | **0**（见下） | 132 s（=冷启） |
| 120K × 2，池 3.0e9 / CPU **4.6e9** | 75 blocks ≤ 82 | **118,400 / 120,000（98.7%）** | **5 s**（冷启 132 s，**26×**） |

第三行的代价特别值得注意：那一轮 **offload 写了 11.24 GB、从磁盘读回 2.4 GB、
分配失败 3 次，最终收益为 0**（`external_prefix_cache_hits_total = 0`）。
即 staging 配置不足时，offload 不是"没用"，而是**净亏 I/O**。

对应的启动自检：脚本在 `CPU_BYTES_TO_USE` 装不下一条满长链时会打印 `[warn]`
（避免这种静默 0 收益）。

---

## 6. 已知问题与注意事项

| 问题 | 说明 / 处置 |
|---|---|
| `RLIMIT_MEMLOCK = 8192 KB`（软硬同） | 无法在不提权的情况下调高（`sudo -n` 要密码；`systemd-run --user -p LimitMEMLOCK=infinity` 报 Unknown assignment）。靠 §5.1 的补丁降级运行 |
| 磁盘层没有配额参数 | 上游的 fs 二级层不支持配额/限速；磁盘层随工作集增长（实测一次测试就到 8.5–15 GB），长测试要盯 `du -sh ssd_kv` |
| 磁盘层目录随 `engine_id` | 不固定 `engine_id` 时每次启动都新目录（孤儿累积）；128K profile 固定为 `qwen38-27b-128k-ssd` 并在启动前清空 |
| `flash_qla` 依赖钉死导致 pip 冲突 | 已修：`setup.py` 放宽为 `>=` 并重装 editable（§3）；启动脚本仍设 `PYTHONPATH`，但已非必需 |
| `scripts/tools/kv_pool_sizing.py` 原先解析不了本仓库全部 profile | 已修（见下） |

`kv_pool_sizing.py` 的修复（`a323e60`）：原来只认 `${VAR:-默认}`，导致
`"$VAR"` / `${VAR:=默认}` 全留字面量（**6 个 profile 全部 ValueError**）；
参数正则 `[^\s\\]+` 遇反斜杠截断，导致转义 JSON 参数（`speculative-config`）
解析失败、**MTP 恒为 0**（连带块大小算成 1568 而非 1600）。修复后 6 个 profile
均正确报告 `MTP=3 block=1600`；128K profile 经其校验为
"池 3.0e9 → 安全上限 142,400 ≥ 131,072 ✓"。

---

## 7. 待办与未覆盖

1. 两个 `sm75-upstream`（vLLM 仓库 / 本仓库）都**未推送**到任何远端。
2. offload 尚未压测的场景：磁盘层长期增长与驱逐（上游无配额参数）、池接近满时的恢复、
   `max-num-seqs > 1` 的并发、pinned 与 unpinned DMA 的性能差、多轮 restore 的长时间稳定性。
