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
| `scripts/run_vllm_qwen38_awq_fp8e4m3_128k_RAMx1_SSDx4.sh` | 128K 上下文 + 上游两层 offload（RAM 1 条链 staging + 磁盘 4 条链的环），**本文主要验证对象** |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k_RAMx2.sh` | 500K + 纯 RAM offload（CPU 层即 store，2 条链） |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k_RAMx1_SSDx4.sh` | 500K + RAM staging（1 条链）+ 磁盘 4 条链的环 |
| `config/vllm-128k-RAMx1-SSDx4.env.example` | 128K profile 的配置模板 |

profile 命名规律：`<模型>_<量化>_<上下文>[_RAMx<N>[_SSDx<M>]]` —— 后缀即容量
（`<N>` 个满长上下文常驻 RAM / `<M>` 个在磁盘上成环）；文件名与容量一一对应，
尺寸在脚本内由 `MAX_MODEL_LEN` 推导。

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

### 5.4 磁盘层写满时的行为（实测）

下面是**未设上限**（`max_bytes=0`，上游默认行为）时的情况；设了上限见 §5.5。
把 128K profile 的 `VLLM_SSD_ROOT` 指到 1.6 GB 的 tmpfs 让它真实写满
（一条 120K 链约需 4.5 GB）：

| 观察 | 结果 |
|---|---|
| 写失败形态 | `ERROR [thread_pool.py:181] Job N block I/O failed: [Errno 5] Input/output error: '<path>'`，每个新请求持续出现（本轮 **324 次**） |
| 写入撞停点 | **1.615 GB**（tmpfs 98%） |
| 进行中的请求 | **不受影响**，正常返回 |
| 后续复用 | 退化为**全量重算**（132~133 s，与冷启一致）；磁盘层仍有少量命中（43 chunk / 2.4 GB 读），但请求侧收益≈0 |
| 请求被中止 / 短读 / load 失败 | **0 / 0 / 0** |

即：**写满不会崩、不会中止请求，只是 offload 静默失效并持续刷 ERROR 日志**。
（本机只能在 tmpfs 上触发，errno 是 `EIO` 而非 `ENOSPC`，代码路径相同：
`batch_store_block` 抛 OSError → `thread_pool.py:181` 记录 → job 失败。）

由此得到两条运维约束：

1. **未设上限时回收只能靠重启或停机清理**：那时磁盘层没有 TTL/淘汰，只能靠
   `VLLM_SSD_CLEAN_START=1` 在启动时清空整个目录（两个 profile 现在默认 0 = 不清，
   因为上限会自己回收，见 §5.5）；**外部清理必须在服务停止时做** —— 运行中删文件会让
   索引与磁盘不一致，可能触发 load 失败。
2. **把 `VLLM_SSD_ROOT` 放到有配额的文件系统**，隔离写满对同分区其他数据的影响。

profile 已把 `kv_load_failure_policy` 设为 `recompute`（vLLM 默认 `fail`，即中止受影响的
请求）：磁盘层是尽力而为的，退化为重算优于报错。注意本轮**没有触发 load 失败路径**（写失败
的 chunk 在索引里直接是 MISS），所以两种策略的差异目前只有代码路径支撑
（`vllm/v1/core/sched/scheduler.py` 的 "Failing N request(s) due to KV load failure"）。

> **排查提示**：单请求场景下 A→B→A 的第三次很容易被 **GPU 前缀缓存**冒领 —— 实测出现过
> 118,400/120,000、3 s 的"假恢复"，其实是 A 的块还在 GPU 池里，与磁盘无关。判断命中来源
> 必须看 `tiering_*` 指标，不能只看 `cached_tokens`。

---

### 5.5 磁盘层字节上限 + LRU 淘汰（实现，已实测）

上游 fs 层没有容量概念（无配额、无 TTL、无淘汰），占用随累计 spill 单调增长
（实测 ~4.5 GB / 条 120K 链）。本线补了 **tier 自管字节预算**：配置
`max_bytes` 后，tier 在**自己的 I/O 线程**上淘汰整块文件（不会阻塞调度线程），
空间不够就删到够为止。配套还做了：

- **LRU 而非 FIFO**：晋升（磁盘→CPU/GPU）只把块拷进 primary，**磁盘副本保留**
  （`_complete_promotion` 不删 secondary 副本）→ 被反复恢复的块就是有价值的块，
  所以按"最近使用"淘汰。recency 就是文件 mtime，恢复成功时 `os.utime` 刷新；
  重启后从磁盘读回，顺序依然正确。
- **写满兜底**：写如果仍以 `ENOSPC`/`EDQUOT`/`EIO` 失败（例如同分区被别的数据占满），
  就再淘汰一批并重试（`evict_retries`，默认 3 次）。
- **不淘汰在途 load 的块**（in-flight 路径集合），避免自己制造 load 失败。
- **启动盘点**：init 时扫描目录（~10–30 ms / 3000 文件），所以 `CLEAN_START=0`
  的续用场景也正确；顺带清掉被 kill 的进程留下的孤儿 `.tmp`。
- **指标**：`vllm:kv_offload_tiering_fs_{used_bytes,evictions,evicted_bytes,skipped_store_bytes}`。

实测（128K profile，`VLLM_SSD_MAX_BYTES=6.05 GiB`，一条 120K 链需 4.58 GB）：

| 步骤 | cached | 用时 | 磁盘占用 |
|---|---|---|---|
| A（冷启） | 0 / 120,000 | 132 s | 4.3 GiB |
| A 重发 | **118,400（98.7%）** | **3 s** | 4.3 GiB |
| C（新链，触发淘汰） | 0 / 120,000 | 132 s | **6.1 GiB（被 cap 住）** |
| C 重发 | **118,400（98.7%）** | **3 s** | 6.1 GiB |

- **0 次 `block I/O failed`、0 次请求中止、0 次短读**（对比 §5.4 未设上限时的 324 次）。
- 淘汰计数 46 块 / 2.39 GiB，`used_bytes` 稳定在 6.02 GiB ≤ cap；被淘汰的正是
  最旧的 A（C 更新），最新链 C 完好 → LRU 行为符合预期。

注意两点：**上限假设 tier 独占该目录**（多实例共享 `root_dir` 时互相删块无法判断谁在用，
此时不要设 `max_bytes`）；**cap 至少要装得下一条满长链**，否则每次 store 都会把自己
链头淘汰掉，收益退化为 0（此时 tier 会打一条 `warning_once` 并跳过该批 store）。

---

### 5.6 500K 部署三档（64 GB 主机目标）

三个 profile 共用同一套 KV/池参数（`MAX_MODEL_LEN=500800`、`KV_CACHE_MEMORY_BYTES=9.6e9`
→ 525,229 tokens = 1.05×），只有 offload 档位不同：

| 档 | 脚本 | tier 配置 | 容量语义 |
|---|---|---|---|
| 500k | `run_vllm_qwen38_awq_fp8e4m3_500k.sh` | 无 offload | 只有 GPU 池；重复的长 prompt 全量重算 |
| 500k_RAMx2 | `run_vllm_qwen38_awq_fp8e4m3_500k_RAMx2.sh` | `TieringOffloadingSpec` + `secondary_tiers: []` | CPU 层**就是 store**：2 条满长链 = 626 chunks = **32.5 GiB**；恢复只走 PCIe（不经磁盘）；淘汰即丢 |
| 500k_RAMx1_SSDx4 | `run_vllm_qwen38_awq_fp8e4m3_500k_RAMx1_SSDx4.sh` | 同上 + `fs` secondary（`max_bytes`） | RAM 只做晋升 staging（1 条链 = 313 chunks = **16.3 GiB**），磁盘是 **4 条链的 LRU 环**（65.1 GiB）；唯一能保住多个长会话、且跨重启保留的档 |

尺寸全部由 `MAX_MODEL_LEN` 推导（`CHAIN_CHUNKS = ceil(MAX_MODEL_LEN/1600)`、
`CHAIN_BYTES = CHAIN_CHUNKS × 55.8 MB`），改上下文长度会自动跟随；
`CPU_BYTES_TO_USE` / `VLLM_SSD_MAX_BYTES` 也都可以用环境变量覆盖。

**RAM 是硬开销**：tier 是 `/dev/shm` 上启动前预 fault 的 mmap，而 tmpfs 默认 = RAM 的 50%：

- `RAMx1_SSDx4`：16.3 GiB ✓ 直接落在 64 GB 主机的 32 GiB tmpfs 内，**无需改系统**；
- `RAMx2`：32.5 GiB **略超**默认 32 GiB → 需 `mount -o remount,size=36864M /dev/shm`
  （自检会打印精确命令）。

**装机自检**（三个 offload profile 共用 `scripts/tools/offload_sizing.sh`）：

| 检查 | 级别 |
|---|---|
| `/dev/shm` **总量** ≥ staging | error（打印精确的 `mount -o remount,size=...M`） |
| `/dev/shm` **剩余** ≥ staging | error（同名文件被 unlink 后仍在 `/proc/*/maps` 里，会连同可见文件一起列出持有者） |
| `MemAvailable` ≥ staging + 4 GiB | error（staging 启动前整块预 fault） |
| tier / 链 / 磁盘环 / GPU 池 是否配得上目标 | warn |
| memlock、cgroup 余量 | warn（不阻断，与 vLLM 自身语义一致） |

不合格**拒绝启动**（exit 1），`CHECK_ONLY=1` 只检查不启动，`ALLOW_UNSAFE_LAUNCH=1` 强制放行。
`/dev/shm` 清理策略：**只删自己 engine id 的陈旧文件**（且该文件没有进程映射时）；别的实例的文件
一律不动 —— 空间不够就报错并点名持有者，由人决定。

实测（本机，128K 档）：给自己的 id 放一个 1 GiB 陈旧文件 → `CHECK_ONLY` 打印
`removing stale staging file ...` 并正常通过（exit 0）；放一个**别的 id** 的 4 GiB 文件 →
**不删**，报 `has only 3.8 GiB free of 7.8 GiB` 并列出持有者，exit 1。

**RAM-only 档已实测**（小规模代跑：40K prompt、2 × 26 chunks、61-chunk tier、小池）：
`A → B → A` 恢复 **36,800 / 39,170（94%）/ 3 s**，`kv_offload_total_bytes{CPU_to_GPU}=1.44 GB`
（正好是那条链），磁盘文件数不变 → 证明"空 secondary 的 tiering = 纯 RAM 档"可用。

> 注意 tier 必须装得下**同时要命中的链**：同一轮里 43-chunk tier 因 A+B 需 52 chunks 而
> 互相挤出 → 0 命中（与 §5.2 的"链必须完整"是同一条规则）。这就是 `RAMx2` 按 2 条链配、
> `RAMx1_SSDx4` 的 RAM 只当 staging（而磁盘兜住多会话）的原因。

本机（16 GB / 7.8 GiB shm）实测：两档都按预期拒绝启动并打印精确的 `mount -o remount` 命令；
参数在 vLLM 侧解析正常（日志确认 `max_model_len: 500800` 与 tier 配置）。**真实 500K 恢复
要等内存到位**（见 §7）。

### 5.7 信息面板 `scripts/tools/monitor_kv_offload.py`

本分支没有 fork 的 `GET /host_tier_info`（那是 fork 在 `Scheduler` 里加的端点），所以面板完全
建立在**这类服务本来就暴露的数据**上：只读、只用标准库、不需要 dev-mode。

| 区块 | 数据来源 |
|---|---|
| `CONFIG` | `/proc/<pid>/cmdline`（启动参数 + `--kv-transfer-config` JSON：`engine_id`、tier 尺寸、策略）与 `vllm:cache_config_info`（`block_size`、池 token 数、dtype、mamba 参数） |
| `STATUS` | `/metrics`：请求数、prefix / external 命中、GPU 池占用、connector 的 store/load 字节与直方图、tiering 的逐 tier 查询/命中/读写/job/失败、fs 配额与淘汰 |
| `CHUNKS` | 直接扫磁盘层目录：chunk 数、字节、rank/group 分布、最新/最旧时间、最近写入的若干条 |
| `Resources` | `nvidia-smi`、`/dev/shm`（含被 unlink、只能从 `/proc/*/maps` 看到的 staging 映射及其进程数）、`MemAvailable` |
| `LOG`（可选 `--log`）| 日志尾部 256 KiB 的错误行计数与最后一条 |

~~~bash
python3 scripts/tools/monitor_kv_offload.py                 # :8000，5s 刷新
python3 scripts/tools/monitor_kv_offload.py --port 8001 -d 2
python3 scripts/tools/monitor_kv_offload.py --once
python3 scripts/tools/monitor_kv_offload.py --json --count 5 # 每 tick 一行 JSON
python3 scripts/tools/monitor_kv_offload.py --append --count 3 # 逐帧完整输出（写日志用）
python3 scripts/tools/monitor_kv_offload.py --self-test       # 校验终端接管与整屏重画
python3 scripts/tools/monitor_kv_offload.py --no-chunks --log 'logs/server_128k_*.log'
~~~

**像 vim 一样接管屏幕，但只读**：终端上进入 alternate screen（`\033[?1049h`，你原来的回滚缓冲
不会被写脏）、隐藏光标（`\033[?25l`）、把 tty 设为 cbreak —— 于是**按键既不回显也不被读取**，
只查看；`Ctrl-C` 是唯一出口，退出时还原光标与终端属性（`TCSAFLUSH` 顺带丢掉这期间敲的键），
`SIGTERM`/`SIGHUP` 也会走同一套还原。`SIGWINCH` 唤醒等待循环，改窗口大小 0.2 s 内重画而不是
等满一个间隔。

**整屏重画（已撤掉差量原位重绘）**：shell 的画面已经被 alternate screen 换走，差量重绘要保护的
东西不复存在，而它的逐行记账正是滚动/改窗口时错位的来源。现在每 tick 就是 `\033[H\033[J` +
整帧、一次 flush 写出；行宽仍设上限（`--width`，默认 `终端列数-1`）以免换行改变帧的实际高度。
stdout 不是终端、或显式 `--append`/`--json` 时，退回"逐帧完整文本"，**完全不含转义序列**，
便于重定向与抓取。

实测（`-d 1 --count 3 --no-chunks`，伪终端下抓字节流）：开头 `\033[?1049h\033[?25l`、结尾
`\033[?25h\033[?1049l`，`\033[H\033[J` 次数 = 帧数（3），相对光标移动 **0** 次；每帧 ≈2.9 KB（整帧，按 5 s 间隔
可以忽略；差量版曾是 554 B / 97 B）。`kill -TERM` 中途打断同样以还原序列结尾；管道输出 41 行、**0 个 ESC**。
`--self-test` 校验进入/退出序列成对且光标恢复、每帧整屏重画、禁用终端时不写任何东西、
非 tty 输出无转义。

计数器显示 `累计 (+本 tick 增量)`；tier 标签直接取自引擎（`0:primary`、`1:fs`…）。`--json` 的每
行含 `config / metrics / gpus / shm / disk / log`，便于脚本化告警。

**与 fork 面板的差别（诚实说明）**：上游没有逐请求清单端点，所以 GPU/CPU 层只能看聚合占用；
磁盘层是按 chunk hash 存放的，因此 `CHUNKS` 列的是**真实落盘的 chunk**（按 rank/group 分布），
而不是 fork 的"逐条会话"。

**实测**（128K profile）：

- 冷启 120K 请求 → `Store GPU→CPU 7.2 GiB in 0.76 s (9.45 GiB/s)`、
  `fs quota 54.2% (9.2 GiB / 17.0 GiB)`、`CHUNKS 178 file(s)`（group `g0/g1/g2` 各 11、`g3` 145）；
- 挤掉 GPU 块后重发同一条 120K → `External queries 363,080 hits 155,200 (42.7%)`、
  `Load CPU→GPU 5.3 GiB in 0.51 s (10.51 GiB/s)`、`tier 1:fs lookups 112 hits 104 (92.9%)`、
  `read 5.4 GiB in 5.41 s`；请求本身 **4.6 s** 返回、命中 118,400/120,000（98.7%）；
- 端口写错 / 服务不可达 → `(unavailable: no api_server process with --port N)` 与
  `server down / metrics unavailable`，**不串台**（pid 只按 `--port` 严格匹配，靠解析
  `/proc/*/cmdline` 而非 `pgrep -f`——后者会匹配到命令行里恰好含该字符串的 shell）。

---

## 6. 已知问题与注意事项

| 问题 | 说明 / 处置 |
|---|---|
| `RLIMIT_MEMLOCK = 8192 KB`（软硬同） | 无法在不提权的情况下调高（`sudo -n` 要密码；`systemd-run --user -p LimitMEMLOCK=infinity` 报 Unknown assignment）。靠 §5.1 的补丁降级运行 |
| 上游 fs 层默认没有回收机制 | 上游只有 `root_dir` / 读写线程数 / `locality`，**无配额、无 TTL、无淘汰删除**（`os.remove` 仅出现在探测文件、写失败的临时文件、短读判定损坏三处），占用随累计 spill 单调增长（实测 **~4.5 GB / 条 120K 链**），只能靠 `VLLM_SSD_CLEAN_START=1` 在启动时回收。本线已补 `max_bytes` 字节预算 + LRU 淘汰（§5.5），128K/16K 两个 profile 默认 64 GiB、且不再在启动时清空目录 |
| 磁盘层目录随模型/配置，不随 `engine_id` | 目录名由模型路径+配置摘要派生（`<root>/<model>_<digest>_r<rank>/`），所以换 `engine_id` 不会留下孤儿；`engine_id` 只决定 `/dev/shm` 的 staging 文件名，每个 profile 各自固定 |
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

1. offload 尚未覆盖：**池接近满时的恢复**（最高优先，唯一可能 stall 的路径）、多轮
   offload/restore 的长时间稳定性、`max-num-seqs > 1` 的并发，以及 pinned / unpinned DMA
   的性能差（后者要 root 或 ≤8 MB 的 staging，实际做不了）。
   磁盘层写满的行为已在 §5.4 实测，字节上限 + LRU 淘汰已在 §5.5 实现并实测。
2. 磁盘层预算的**长稳**：多天运行下 mtime 作为 recency 的退化（例如备份/rsync 改写 mtime）
   尚未验证。
3. **500K 三档的端到端**（§5.6）：内存到位后在 64 GB 主机上各跑一次
   （冷启 → 重发 → 跨会话换出/恢复），并把 `RAMx2` 的 `/dev/shm` remount 纳入装机清单。
