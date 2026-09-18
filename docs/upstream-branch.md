# 上游分支（`2080ti_dual_qwen38-27B`）：开分支 / 移植 / 部署记录

本文记录"在**上游 vLLM main** 上重建 SM75 部署"这条线的工作：分支怎么开的、
移植了什么、编译环境怎么修的、怎么部署、128K 下 offload 实测结果。

---

## 1. 分支结构

| 项 | 值 |
|---|---|
| 源码目录 | `/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/src/vllm-0271` |
| 新分支 | `2080ti_dual_qwen38-27B`（基于上游 `main`） |

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

本机 FlashQLA checkout 取分支 `2080ti_dual_qwen38-27B` 的 HEAD（`4459b70`，已含本地 SM75
改动，即 `patches/flashqla-sm70-sm75-local.patch` 的内容），以 editable 方式装在当前 checkout
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
| `scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh` | 生产档：256K（模型原生上限），无 offload，5.3e9 池 |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k.sh` | 长上下文基础档：500.8K 上下文，无 offload，9.6e9 池 |
| `scripts/run_vllm_qwen38_awq_fp16_225k.sh` | **速度取向档**：225,280 上下文 + **fp16 KV + n=6**，用与 500K 档同样的 9.6e9 预算（fp16 使池降到 252,223 token）；215K 实测比 fp8/n=5 快 **+13%**（§6.14） |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_128k_RAMx1_SSDx4.sh` | 128K 上下文 + 上游两层 offload（RAM 1 条链 staging + 磁盘 4 条链的环），**本文主要验证对象** |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_256k_RAMx1_SSDx4.sh` | 256K + 同样两层 offload（staging 9.15 GB，需 ~32 GB 主机） |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k_RAMx2.sh` | 500K + 纯 RAM offload（CPU 层即 store，2 条链） |
| `scripts/run_vllm_qwen38_awq_fp8e4m3_500k_RAMx1_SSDx4.sh` | 500K + RAM staging（1 条链）+ 磁盘 4 条链的环 |
| `config/vllm*.env.example` | 各档配置模板（无 offload 的 256K/500K 档用 `vllm.env.example`） |

profile 命名规律：`<模型>_<量化>_<上下文>[_RAMx<N>[_SSDx<M>]]` —— 后缀即容量
（`<N>` 个满长上下文常驻 RAM / `<M>` 个在磁盘上成环）；文件名与容量一一对应，
尺寸在脚本内由 `MAX_MODEL_LEN` 推导。

三个 profile 都从 `.env` 取路径（`MODEL_PATH` / `VLLM_PYTHON` / `FLASHQLA_PATH` /
`CHAT_TEMPLATE`），profile 参数在脚本内有默认值、可用环境变量覆盖。

共同参数：`--dtype half`、TP=2、`--device-ids 0,1`、`--kv-cache-dtype fp8_e4m3`、
`--gdn_prefill_backend=flashqla_legacy`、MTP `num_speculative_tokens=5`（§6.2）、
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
停实例用 `scripts/tools/stop_server.sh <port>`（`--port`/位置参数，`--list` 列出所有实例的
pid/端口/engine/model，`--dry-run` 只打印计划不发信号，**默认回收该实例的 staging 文件**，
`--no-clean-shm` 可关闭）：
它按端口找到监听进程（`ss` 优先，退回扫描 `/proc/*/cmdline`，不用会自匹配的 `pgrep -f`），
先 TERM 主进程让其自行收尾，10 s 后升级为整组 TERM，再不行 KILL，最后校验进程与端口都已释放。
**显式 `--port` 优先于 `.env` 的 `PORT`**（否则会把"测试不存在端口"变成真杀生产实例 —— 实测踩过）。

**为什么必须显式删 staging**：本分支**不会** unlink 这个文件（启动日志只有
`Created/Opened existing mmap file`，从来没有 `Unlinked mmap file`，源码里 unlink 只在给了 barrier
的路径上），所以它是真实文件、进程结束后仍然存在，**且继续占着 tmpfs 的页（即内存）**：实测停掉
128K 实例后 `/dev/shm` 仍是 `used=4.3 GiB`，删掉该文件立刻回到 `used=0 / 7.8 GiB`。因此
profile 启动前只清**同名**文件（换 engine id 就清不到），而 `stop_server.sh` 现在默认按该实例
cmdline 里的 `engine_id` 删自己的文件并打印前后用量。

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
要等内存到位**（见 §8）。

### 5.6b 256K offload 档（同一套两层 offload，链更短）

`run_vllm_qwen38_awq_fp8e4m3_256k_RAMx1_SSDx4.sh` 把同一套两层 offload 用在 256K：
`MAX_MODEL_LEN=262144`、池 5.3e9 → 267,842 tokens（n=5 实测；n=3 时 278,253，≈1.06× 一个满请求），链 =
`ceil(262144/1600)` = **164 chunks = 9.15 GB**，磁盘是 4 条链的 **36.6 GB** LRU 环。

staging 是启动前预 fault 的 `/dev/shm` 硬预留，所以这档要 **~10 GB tmpfs（约 32 GB 主机）**：
本机 15 GiB 上 `CHECK_ONLY=1` 直接拒绝——`/dev/shm` 只有 7.7 GiB free 而 staging 要 8.5 GiB，
`MemAvailable` 也低于 staging+4 GiB 的余量要求。128K 档（4.58 GB staging）在本机可跑，两者
就是按这个分工选的。用途与 128K 档相同（长 prompt 的 KV 跨重启可恢复，省掉每次 ~400 s 的
重算），只是上下文翻倍；256K 档的**真实恢复同样待内存到位**。

### 5.7 信息面板 `scripts/tools/monitor_kv_offload.py`

本分支不改 vLLM 的 offload / tiering 层，也没有额外的私有端点，所以面板完全建立在**这类服务
本来就暴露的数据**上：只读、只用标准库、不需要 dev-mode。

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
python3 scripts/tools/monitor_kv_offload.py --no-chunks --log 'logs/server_128k_*.log'
~~~

计数器显示 `累计 (+本 tick 增量)`；tier 标签直接取自引擎（`0:primary`、`1:fs`…）。`--json` 的每
行含 `config / metrics / gpus / shm / disk / log`，便于脚本化告警。

**能看什么、不能看什么（诚实说明）**：上游没有逐请求清单端点，所以 GPU/CPU 层只能看聚合占用；
磁盘层是按 chunk hash 存放的，因此 `CHUNKS` 列的是**真实落盘的 chunk**（按 rank/group 分布），
而不是逐条会话。

**实测**（128K profile）：

- 冷启 120K 请求 → `Store GPU→CPU 7.2 GiB in 0.76 s (9.45 GiB/s)`、
  `fs quota 54.2% (9.2 GiB / 17.0 GiB)`、`CHUNKS 178 file(s)`（group `g0/g1/g2` 各 11、`g3` 145）；
- 挤掉 GPU 块后重发同一条 120K → `External queries 363,080 hits 155,200 (42.7%)`、
  `Load CPU→GPU 5.3 GiB in 0.51 s (10.51 GiB/s)`、`tier 1:fs lookups 112 hits 104 (92.9%)`、
  `read 5.4 GiB in 5.41 s`；请求本身 **4.6 s** 返回、命中 118,400/120,000（98.7%）；
- 端口写错 / 服务不可达 → `(unavailable: no api_server process with --port N)` 与
  `server down / metrics unavailable`，**不串台**（pid 只按 `--port` 严格匹配，靠解析
  `/proc/*/cmdline` 而非 `pgrep -f`——后者会匹配到命令行里恰好含该字符串的 shell）。

#### 浏览器面板 `scripts/tools/monitor_kv_offload_web.py`

终端版打印的是文本帧：`--once`/`--json`/`--append` 与 grep 都合适，但字符网格不适合"看"状态。
浏览器天生有滚轮、缩放与滚动条，卡片和进度条画起来也不花成本；而真正麻烦的**采集**部分是同
一套代码，所以本脚本直接 `import monitor_kv_offload` 复用 `Metrics` / `collect_config` /
`scan_chunks` / `shm_stats` / `gpu_memory` 等，不重复实现。**纯标准库**，页面内联 CSS/JS，
**不引用任何外部资源**（离线主机可用）：

~~~bash
python3 scripts/tools/monitor_kv_offload_web.py                   # http://127.0.0.1:8199/
python3 scripts/tools/monitor_kv_offload_web.py --port 9000 -d 2
python3 scripts/tools/monitor_kv_offload_web.py --vllm-port 8001 --log 'logs/server_128k_*.log'
python3 scripts/tools/monitor_kv_offload_web.py --host 0.0.0.0     # 局域网（无鉴权，慎用）
python3 scripts/tools/monitor_kv_offload_web.py --self-test
~~~

| 端点 | 内容 |
|---|---|
| `GET /` | 面板页：CONFIG / STATUS / CHUNKS ON DISK / LOG 四张卡片，进度条 + 刷新时钟 + 暂停勾选 |
| `GET /api/view` | 页面消费的结构化数据（headline、config 行、bars、tables、chunks、log） |
| `GET /api/snapshot` | 原始采样，形状与终端版 `--json` 一致（指标系列 + proc/disk/shm/gpu） |
| `GET /healthz` | 存活探测 |

- **采样在后端**：后台线程每 `-d` 秒采一次并缓存，所有浏览器共用同一份，所以"本 tick 增量"
  （如 `last 3s: +N hits`）不会因为多开页面互相稀释；页面只定时 `fetch('/api/view')`。
- **吞吐**：页头给 live 解码速率，`Throughput` 表给 `decode/prefill (live, N秒窗口)`（取
  `vllm:generation_tokens_total` / `vllm:prompt_tokens_total` 的窗口增量 ÷ 采样间隔 —— 引擎
  只导出计数器，没有瞬时速率指标）、`decode (finished requests)`（`request_generation_tokens_sum
  ÷ request_decode_time_seconds_sum`，历史均值，可用来判断当前波动）与 `spec decode (MTP)`
  接受率及 `≈N tok/step`。第一帧没有上一个采样点，窗口速率显示 `—` 而不是拿全部历史除以间隔。
  实测（500K 档、单请求）：live **54.0 tok/s**（4 s 窗口 216 tok），独立测量 56.2 tok/s，
  历史均值 46.6 tok/s（16 请求 / 64.4 s），MTP 接受率 68.1% → ≈3.04 tok/step。
- 默认只绑 `127.0.0.1`（payload 含本机路径）；远端用 SSH 隧道
  `ssh -L 8199:127.0.0.1:8199 <host>`。`--host 0.0.0.0` 无鉴权，启动时会打印警告。
- 只读：只对 vLLM 发 GET，只读 `/proc`、`/dev/shm`、`/proc/meminfo`、`nvidia-smi` 与磁盘层文件。
- 实测：`/api/view` 3 条进度条（GPU 0.0% / CPU 0.0% / fs 54.2%）、5 张表、13 行 CONFIG、
  60 行 chunk，`external 174,400/388,058 (44.9%)`、`Store 7.2 GiB in 0.76 s (9.45 GiB/s)`；
  页面 5.3 KB 无外链；`/api/snapshot` 含 81 个指标系列；`--self-test` 用桩数据渲染一遍 view，
  防属性名/结构漂移（开发时正是它抓到 `cache_layout` 写错）。

---

## 6. 单并发解码性能：MTP + cudagraph（SM75 实测）

**口径先行**：单并发稳态 `tok/s = 步频 × (1 + n × MTP 接受率)`。README 表里的 84–101 tok/s 是
「接受率 ~90% 的前 128 token」口径；长生成（接受率 45–65%）稳态只有 ~45 tok/s，两者都对。

### 6.1 定位：CPU-launch-bound，不是 GPU / 功耗 / 带宽

用 torch profiler（`--profiler-config` + `POST /start_profile`）与高频 `nvidia-smi` 采样实测
（32K 上下文、单请求、MTP n=3）：

| 观测 | 实测 |
|---|---|
| 步耗时 | 57 ms |
| 单步 CUDA kernel 工作 | ~35 ms（**1280 次 launch/步**） |
| GPU0 / GPU1 利用率 | 58% / 83% |
| GPU1 的 kernel 时间构成 | **57% 在 `cross_device_reduce_1stage` 里自旋等 GPU0**（324 µs/次 × 138 次/步） |
| 功耗 / SM 频率 | 211–218 W（上限 250 W）/ 1850 MHz → **非功耗受限** |
| Marlin INT4 GEMM | ~416 GB/s（616 峰值 67%）→ **非带宽受限** |

GPU 时间按父 op 归因：`qwen_gdn_attention_core` 24.9%（**490 kernel/步**）、`aten::mm` GEMV
22.0%（11.6 次 × 667 µs）、`aten::copy_` 系列 ~18%、attention 11.3%。两个结构性原因：

1. **48 个 GDN 层每步走 eager custom op**：SM75 没有融合 GDN decode kernel（要求
   `compute capability 8.0+`），日志明写 `Falling back to the Triton GDN decode path` /
   `GDN decode kernel: triton`；该 op 又被 `@eager_break_during_capture` 强制在 eager 段执行。
2. **MTP 下 cudagraph 被降级为 PIECEWISE**：`FlashInferBackend` 只声明
   `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`（`UNIFORM_BATCH` 仅对 SM90+ 的
   TRT-LLM/XQA 开放），于是 `setting cudagraph_mode=PIECEWISE`；而 `speculator.py` 在非 FULL
   解码模式下把草稿多步解码图设成 `NONE`（完全 eager）。

### 6.2 n 扫描（分离"固定开销"与"每 draft 边际成本"）

| 配置 | 步耗时 | tokens/步 | 稳态 |
|---|---|---|---|
| 无 MTP | 23.4 ms | 1.00 | 41.5 tok/s |
| MTP n=1 | ~49 ms | ~1.8 | ~37 |
| MTP n=3 | 57 ms | ~2.7 | ~45 |
| MTP n=5 | ~71 ms | ~3.2 | ~45 |

n=0→n=1 一步就多 ~25 ms，之后每个 draft 只加 2–6 ms：贵的是「开启 spec-decode 的固定开销」，
不是 draft 数量。（上表是 PIECEWISE 时代的数据，§6.4 修好图捕获后同一组对照的斜率见 §6.5。）

> **2026-09-18 更正：上表按「每步耗时」看会得出 n=3 最优，但该看的是「每 token 成本」
> = 每步耗时 / (1 + n × 接受率)。** 同一天的稳态实测（每配置 2–3 次，按接受率归一）：
>
> | n | 31.5K | 128K | 250K |
> |---|---|---|---|
> | 3 | 70.1 tok/s | 54.3 | 45.1 |
> | 4 | 75.6 | 64.4 | — |
> | 5 | **85.1** | **66.7** | **61.7** |
> | 6 | 89.2 | — | 64.6 |
>
> 每 token 成本随 n 单调下降（31.5K：39.5 → 36.6 → 35.1 → 35.1 ms），因为一轮里 draft 只跑
> 1 层（MTP 头）、verify 才跑全部 64 层——draft 越多，verify 摊得越薄。接受率确实随 n 塌
> （31.5K 从 58.8% 掉到 35.5%），但 tokens/轮 仍升到 ~3.1，净收益为正。**因此 profile 默认从
> n=3 改为 n=5**；n=6 再快约 5%（31.5K/250K 实测）。
>
> **2026-09-19 更正"n≥7 启动失败"**：那不是硬限制，而是 KV 池不够——投机解码需要额外槽位，
> 默认 `KV_CACHE_MEMORY_BYTES=5.3e9` 会因"最大长度请求需要 5.0 GiB > 可用 4.91 GiB"而拒绝
> 启动（只差 0.09 GiB），提到 6.2e9（池 302,797 tokens、仍装得下 262,144）即可正常跑 n=7。
> 但**实测 n=7 明显更差**：接受率掉到 22–37%，按同口径归一后每 token ≈56 ms，而 n=5 是 ~35 ms。
> 原因是每多一个 draft 就多一整套代价（一遍 lm_head 1.27 GB/卡 + 一遍 250K 注意力），接受率却不涨。
> 所以 **n 的最优点是 5–6，不存在"越大越好"**；这也从侧面说明每步的固定成本里 lm_head 占比很大
> （见 §6.12）。
>
> 输出影响：n=4/5/6 的 greedy 结果彼此**逐字节相同**，与 n=3 在第 ~90 个 token 处有一次
> 近平分叉（两侧都是合理续写），属不同批形状下的浮点差异，不是逻辑差异。

### 6.3 已实测并排除的杠杆

| 杠杆 | 结果 |
|---|---|
| `VLLM_SM75_SPEC_SYNC_MODE` safe ↔ nosync | 无差别（45 vs 48 tok/s），greedy 输出逐字节相同 |
| `num_speculative_tokens` 1 / 2 / 5（当时未测 4/6） | 旧结论「3 最优」；**2026-09-18 更正为 n=5**（每 token 成本随 n 下降，见 §6.2） |
| 关闭 MTP | 41.5 tok/s（更差） |
| `VLLM_USE_V2_MODEL_RUNNER=0` | 无差别 |
| `--no-async-scheduling` | 略差（~42），且**会改变输出** |
| `--attention-backend TRITON_ATTN` | 启动失败：SM75 不支持 fp8 KV（要求 SM89+） |
| `TRITON_ATTN` + fp16 KV | 启动成功且启用了融合草稿解码，但 attention 慢 5 倍 → **9–13 tok/s** |
| 功耗上限（220 → 250 W） | 无影响（实测只到 211–218 W） |

### 6.4 已采纳：native spec-as-decode → 保住 FULL cudagraph

FlashInfer 的 native fa2 decode wrapper **本来就支持** uniform `q_len_per_req > 1`
（`BatchDecodeWithPagedKVCacheWrapper.plan` 与 `fast_decode_plan` 都有该参数，且
`is_causal = q_len_per_req > 1`，注释写明为投机解码的验证批次设计），但 vLLM 只在 SM90+ 的
TRT-LLM/XQA 路径开放它。放开后（`VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE=1`，四个 profile 默认
开启，`.env` 可覆盖为 0）：

- `get_cudagraph_support` 声明 `UNIFORM_BATCH`；`reorder_batch_threshold = 1 + n`，验证批次留在
  decode 路径；paged-KV 元数据按**请求数**规划；`q_len_per_req` 透传到 `plan` /
  `fast_decode_plan`；cudagraph 的 decode wrapper 以 `(num_reqs, q_len_per_req)` 为键（FlashInfer
  按 wrapper 冻结 q_len）；非 uniform 批次（如 padding 出 0 长度请求）回退 prefill 路径。

| | 之前 | 之后 |
|---|---|---|
| cudagraph | 仅 PIECEWISE | **FULL + PIECEWISE（含 prefill FULL）** |
| 步耗时 | 57 ms | **37.6 ms** |
| 单并发稳态 | ~45 tok/s | **~65–86 tok/s** |

上游提交：`vllm@2080ti_dual_qwen38-27B` `059727bfa`；部署侧 `20dbbe5`（profile/env 示例默认开启）。

### 6.5 已评估、未采纳：融合多步草稿解码

草稿循环每步在主机侧重算 attention 元数据（`plan()` + numpy + H2D），因此只能每步单独捕获
一张图。理论上可以像 `triton_attn` 那样声明 `supports_draft_decode_metadata_update = True`，
用设备侧 kernel 就地刷新元数据，把整段草稿循环捕获成**一张图**（日志出现
`Capturing decode CUDA graphs (FULL)`）。

已实现并实测（补丁留档 `docs/patches/2026-09-18-fused-multi-step-draft-decode.patch`，**不套用**；它以 §6.4 的 `059727bfa` 为基线，`git apply --check` 可干净套用）：

- 新增 `_paged_kv_meta_kernel`：从设备端 `seq_lens` 重算 `paged_kv_indptr` +
  `last_page_len`（逐条对齐主机版 `_compute_flashinfer_kv_metadata`，含
  `seq % page == 0 && seq != 0 → page_size` 特例），再复用已有的
  `_copy_page_indices_kernel` 刷新页索引；`update_draft_decode_metadata` 调用它们（捕获安全）。
- 结果：步耗时 **37.6 → 35.2 ms（+6%）**，稳态同量级；greedy 输出与基线逐字节一致，接受率
  可比；开关置 0 时完全回到原行为。

**结论：不采纳。** 收益仅 ~6%——每个 draft 的主机元数据重建实际只值 ~1.2 ms，而非原先估的
~3 ms（剩下的每 draft ~4 ms 主要是草稿前向 + 采样 + 必要的设备侧更新）——却要多带一个设备侧
kernel 与相应的捕获安全面。此处仅作记录，便于将来需要时复看。

### 6.6 改动注意力路径时的正确性验证方法

用**逐字节对照**，不要只看指标：

1. `temperature=0`、固定 seed 的 greedy 生成，比较**全文**（不只比 hash）；
2. 覆盖短（~40 token）、中（256 token）与**长 prompt（30,351 token）**三种规模——长 prompt 才能
   同时压到 prefill 的 FULL 图；
3. 把开关置 0 跑一次回归，确认完全回到原行为（cudagraph 降级告警、融合回退日志、输出 hash
   均复原）。

### 6.7 长上下文（250K）的归因：瓶颈性质和 31.5K 完全不同

在 256K 档上对 **~250K token 上下文**做解码 profiler（rank0，20.09 s 窗口，1043 token 生成，
tools 见 `/tmp` 同款 `trace_*.py`）：

| 指标 | 250K | 31.5K（§6.1） |
|---|---|---|
| GPU 忙碌（1 ms 桶 kernel 并集） | **99.6%**（无 ≥5 ms 空闲段） | 58% / 83%（GPU0/1） |
| allreduce 占比 | **1.9%** | 占 GPU1 kernel 时间 **57%**（自旋等 GPU0） |
| GDN decode kernel 占比 | **1.5%** | `qwen_gdn_attention_core` 那类 **24.9%** |

GPU 时间构成：**注意力 ≈42%**（`BatchPrefillWithPagedKVCacheKernel` 37.2% + 4.5%）、
**GEMM/GEMV ≈49%**（`marlin::Marlin` 27.9%、维 vocab 248K 的 `gemvx` 15.1%、cutlass f16 6.2%）、
其余 ≈9%（profiler 自身有约 1.38× 放大：窗口内 80.7 ms/步 vs 平时 58.5 ms/步，看比例即可）。

按未开 profiler 的 58.5 ms/步折算：注意力 ≈24.5 ms、GEMM ≈28.7 ms；而一阶带宽下限约为
权重读取 17 ms + KV 6.6 ms ≈ **24 ms/步**（19.57 GiB checkpoint / 2 卡，KV 按 fp8 250K 计）。

两点结论：

1. **31.5K 的解在 250K 上无效**：那里是 launch/latency 受限（allreduce 自旋 57%、GDN 24.9%），
   所以融合 GDN / 降启动开销有意义；250K 是 GPU 满载，GDN 只剩 1.5%，做融合 kernel 最多省
   1.5%。
2. 250K 只能靠**减少 GPU 工作量**：GEMM 已贴近带宽下限，唯一有余量的是注意力（≈24.5 ms vs
   其 KV 带宽下限 ≈6.6 ms）。

**KV dtype 对照**（128K 上下文、同一 prompt、两次运行、接受率归一后的稳态步耗时）：

| KV dtype | 每步 | 相对 |
|---|---|---|
| `fp8_e4m3`（默认） | 47.4 ms | — |
| `float16`（池加倍到 5e9；max-model-len 需降到 126000 才装下） | 46.2 ms | **−2.7%** |

即 **fp8 的反量化开销可以忽略，KV 字节数也不是长上下文瓶颈**——注意力是算力/流水受限，
要再快必须改 kernel（SM75 + 长 KV 的 tile 策略），换 dtype 拿不到。

### 6.8 长上下文注意力的 kernel 天花板：4 warps/SM（2026-09-18 实测）

把 §6.7 的"注意力 ≈24.5 ms"拆到 kernel 参数级别（独立 harness + profiler，绕开服务）：

**实测基准**（kv_len 250,000、q_len 6、24 qo heads / 4 kv heads、head_dim 256、fp8_e4m3 KV、
page 16；单卡）：**1.59 ms/次调用 → 161 GB/s ≈ 峰值 616 GB/s 的 26%**。一次 decode step 有
22 次这样的调用（16 层 verify + 5 次 MTP draft + 1），合计 ≈25.7 ms/58.5 ms。

**参数事实**（trace 里的模板参数）：

```
KernelTraits<MaskMode=causal, CTA_TILE_Q=64, NUM_MMA_Q=1, NUM_MMA_KV=2,
             NUM_MMA_D_QK=16, NUM_MMA_D_VO=16, NUM_WARPS_Q=4, NUM_WARPS_KV=1,
             DTypeKV=__nv_fp8_e4m3>
grid=(68,1,2)  block=(32,4,1)=128 线程  smem=65536(=Turing 每 SM 上限)  regs=255
```

- **KV 已经切了 68 份**（`padded_batch_size` = SM 数 = 68，`gridDim.z` = 2 个 kv head）→
  136 个 CTA，每个只走 ~3.7K 个 KV，并用 `PersistentVariableLengthMergeStatesKernel` 归约。
  所以并行度（chunk 数）不是问题。
- 真正的限制是**每 CTA 用满 64 KB 共享内存 → 每 SM 只能驻留 1 个 CTA**，而 flashinfer 这套
  kernel 把 warp 数**硬编码成 4**（`get_num_warps_q(CTA_TILE_Q)=4`、
  `get_num_warps_kv = 4/get_num_warps_q = 1`）→ 每 SM 只有 128 线程（12.5% 占用率），
  不足以掩盖 DRAM 延迟。
- 这套设计的目标是 Ampere：同样的 64 KB CTA 在 A100（164 KB smem/SM）能放 2–3 个 → 8–12
  warps/SM；**Turing 只有 64 KB 可用，于是退化成 4 warps/SM**。
- CTA_TILE_Q=64 是被 GQA 打包撑起来的：`packed_qo_len = qo_len × group_size = 6×6 = 36`，
  Q tile 占 64×256×2 = 32 KB，正好一半 smem。

**试过并且被堵死的路**（都是实测，不是推测）：

| 尝试 | 结果 |
|---|---|
| page_size 16/32/64/128 | 16 最优（163–165 GB/s），其余 149–161 |
| `use_fp16_qk_reduction=True` | SM75 上编译失败 |
| workspace 调大（64 MB→2 GB） | plan 完全不变，无效果 |
| 改头文件让 `NUM_WARPS_KV=2`（8 warps、CTA_TILE_KV 保持 32、smem 不变） | JIT 重编成功，但 `KTraits::IsInvalid()` 拒绝：fp8 分支要求 `NUM_MMA_KV*2 % NUM_WARPS_Q == 0`，即 fp8 下 `NUM_MMA_KV ≥ NUM_WARPS_Q/2`；要 8 warps 就得 CTA_TILE_KV=64 → smem ~82 KB > 64 KB 上限（改动已还原，基线复测 1.59 ms） |
| 自写 Triton 分段 flash-decoding kernel | **Triton 在 SM75 上完全不支持 `fp8e4nv`**（无法读取 fp8 KV）；且打包 36 行的 M=64 tile 在 Triton 里最少要 72 KB smem，装不进 64 KB |

**结论**：在"fp8 KV + Turing 64 KB smem + 36 行 GQA 打包"三者同时成立时，flashinfer 与 Triton
都到顶了（≈160–220 GB/s）。要再往上必须**手写 CUDA/CUTLASS kernel**，条件：

1. 用位运算做 fp8→fp16 转换（Turing 没有该转换指令，flashinfer 靠 CUTLASS 的软件转换）；
2. 把 smem 压到 ≤32 KB：Q 按 head_dim 分块流式载入，或不把 6 个 GQA head 打包进 M 而改在 CTA
   内循环（需要 ~6 个 [16,256] 的 fp32 累加器，靠 16 warps 摊到寄存器）；
3. 每个 SM 至少 16 warps（多个小 CTA 或一个大 CTA）；
4. KV 分段 + LSE 归约（分段本身已在做）。

预期 **400–500 GB/s → 0.55–0.7 ms/调用 → 注意力 25.7 ms 降到 ~13 ms → 步耗时 58.5 → ~46 ms
（250K 上 +25–30%，61.7 → ~78 tok/s）**。代价是天级的 kernel 开发 + 接入 vLLM 的 FULL
cudagraph 路径（做成 flashinfer backend 里可捕获的自定义 op）。

顺带确认的两个小杠杆（未做）：22 次调用里有 5–6 次是 q_len=1 的 MTP draft，理论上可走
flashinfer 的 decode kernel；lm_head 的 vocab 248K GEMV 占 15%。

### 6.9 手写注意力 kernel 进展：内存模式已证明可行，卡在 codegen（2026-09-18）

§6.8 的结论是"要更快只能手写 kernel"。这一节记录手写工作的进展、已修掉的坑、以及**一个
还没解决但已精确定位的阻塞点**。工具在 `/tmp/opencode/kern/`（`mq_attn.cu` + `bwprobe.py`
等，未入仓库）。

**第一步：先证明这个访问模式能做到多少带宽。** 写了一个只做"读"的探针 kernel（原始 CUDA，
非 flashinfer）：完全相同的分页 fp8 KV 布局、页 gather、软件 e4m3→fp16 转换、`uint4` 向量载入，
256 MB（250K token 的 K+V）实测：

| kernel | 时间 | 有效带宽 |
|---|---|---|
| flashinfer `BatchPrefillWithPagedKVCacheKernel`（§6.8） | 1.59 ms | 161 GB/s |
| 探针，独立小 kernel | **0.425–0.48 ms** | **532–602 GB/s** |
| 本机纯读上限（`torch.sum` 1 GiB） | — | 583 GB/s |

即**同样的数据、同样的页布局，能跑到接近机器纯读上限**——§6.8 的 161 GB/s 是 kernel 的问题，
不是内存模式或硬件的问题。这是整条路线可行的前提。

**第二步：写真正的 kernel**（`mq_attn.cu`，sm_75 + wmma），结构：

- 36 = q_len(6) × group(6) 个 (行, head) 对打包进 48 行 M（3 个 16 行 wmma tile）；
- **双 pass**：pass A 只读 K 求每行 softmax 的 max/sum，pass B 重算 S 并用最终 max 做 P@V。
  多读一遍 K（1.5× 流量）换来的好处是**完全不需要逐元素访问 wmma 累加器**——它的 per-lane
  布局在 wmma API 里是"未定义"的，只能靠 `store/load_matrix_sync`，而在线 softmax 的
  rescale 必须逐元素改累加器；
- 每 tile 恰好一页（16 keys），页索引只需查一次，且所有行都页对齐；
- K/V 以 fp16 落 smem（Turing 无 fp8 转换指令，转换是纯位运算），M 维 48 行（36 有效）；
- KV 按 chunk 切分（68 chunk × 2 kv head = 136 CTA），分段结果用独立的 merge kernel 按 LSE
  合并——**merge 已验证正确，0.016 ms**。

**过程中修掉的三个真实的坑**（都不是算法问题）：

1. `extern __shared__ Smem s;` 被 nvcc **当成静态共享内存**编译（ptxas 报 `45952 bytes smem`），
   于是动态 smem 额度只剩 ~19 KB，`cudaFuncSetAttribute` 在 ≥32768 时直接返回
   `cudaErrorInvalidValue`。改成 `extern __shared__ char raw[]; Smem& s = *reinterpret_cast<Smem*>(raw);`
   后 kernel 的静态 smem 变 0，动态额度恢复。
2. `memcpy(&u, p, 8)`（`p` 是 `uint8_t*`）无法保证位宽 → 显式 `reinterpret_cast<const uint2*>`。
3. launcher 每次调用都做同步的 `cudaFuncSetAttribute`/`cudaDeviceGetAttribute`，被计进计时窗口
   → 改成只配置一次。

**第三步：定位性能差距——已排除的假设（都是实测，不是推测）：**

| 假设 | 实验 | 结果 |
|---|---|---|
| 占用率 / smem carveout | 给探针加上 `__launch_bounds__(512,1)` + 45875 B 动态 smem（与真 kernel 完全同约束） | **518 GB/s，无变化** → 排除 |
| 循环串行依赖（`acc += 载入结果`） | 手写寄存器双缓冲预取 | 3.396 vs 3.510 ms → 排除 |
| 编译器没展开循环 | `#pragma unroll 4` | 3.506 ms → 排除 |
| 固定开销（Q staging / 写回 / launch） | 空 kernel（mode 8） | **0.014 ms** → 排除 |
| 访问模式本身 | 精简只读 kernel，逐字节相同的循环 | **0.425 ms / 602 GB/s** → 模式没问题 |

**真实原因（已定位）**：那个循环**放进大 kernel 里就慢 8 倍**——同一份循环、同一进程、同一张量：

| | 时间 |
|---|---|
| 精简 kernel 里的循环 | 0.425 ms（602 GB/s） |
| 大 kernel 里的同一个循环（mode 6） | 3.359 ms（76 GB/s） |

两者只差在**周围代码**：大 kernel 有 wmma、smem、Q staging，用 **78 个寄存器**（精简版约 30）。
所以是 ptxas 在寄存器紧张的巨型函数里对这段内存循环做了很差的调度（SASS 里探针的循环被自动
展开成 3 组 `LDG.E.128`，大 kernel 里则是大量 32 位 `LDG.E`）。**这是编译器层面的问题，不是
算法或内存模式的问题**——算法侧的前提（602 GB/s）已经用探针证明了。

**当前状态**：完整 kernel 5.3 ms/调用（比 flashinfer 的 1.59 ms 慢 3.3 倍），**还不能用**；
另有 15% 的相对正确性误差（kv_len 512 对比 torch 参考）待查。

**下一步**（按顺序）：① 压缩寄存器/简化巨型函数让 ptxas 能正常调度这段循环（或把 staging 与
计算解耦）；② 修正确性误差；③ 达标后接入 vLLM 的 FULL cudagraph 路径并做 250K A/B。
预期收益仍按 §6.8：注意力 25.7 → ~13 ms/步，250K 上 **+25–30%**。

### 6.10 上述预期的修正：收益被高估，kernel 路线收益有限（2026-09-18 晚实测）

§6.9 的期望建立在"探针能跑 602 GB/s"上。**但那个探针是误导**：它在读入后立刻在寄存器里累加，
既没有 smem staging，也没有 wmma，转换还被 ptxas 折叠掉了一部分。按真实 kernel 需要的步骤逐段
实测（都读 128 MB 的 K）：

| 步骤 | 时间 | 有效带宽 |
|---|---|---|
| 读 K + e4m3→fp16 转换 + 累加到寄存器 | **0.274 ms** | 468 GB/s |
| 读 K + 转换 + 写 smem（staging 的最小形态） | **0.763 ms** | 168 GB/s |

即**转换本身不贵（468 GB/s 可达），真正贵的是 2 字节粒度的 smem 写入（bank conflict，2.8×）**，
换 swizzle 布局可以改善，但那只是其中一项。

把注意力拆成"两个小 kernel"（A: 只读 K 算 S 与行 max；B: 读 S+V 做 P@V）后的实测：

| kernel | 时间 | 有效带宽 | 说明 |
|---|---|---|---|
| A `mq_scores_kernel` | 4.26 ms | ~30 GB/s | 读 K 128 MB + 写 S |
| B `mq_pv_kernel` | 2.85 ms | ~62 GB/s | 读 V 128 MB + S 48 MB |
| merge | 0.019 ms | — | 正确 |
| 合计 | **6.8–7.1 ms** | 52 GB/s | **比 flashinfer 的 1.59 ms 慢 4 倍** |

各段成本拆解：staging 0.76 ms、访存 0.23 ms、wmma ~0.5 ms、掩码/写 S ~0.1 ms ≈ 1.6 ms，**剩下
约 2.5 ms 无法归因**（仍是 §6.9 那个 codegen 损耗）。以下假设均已实测排除：占用率、smem carveout、
`#pragma unroll`、寄存器双缓冲预取、`__noinline__`、`-maxrregcount` 40/48/56/64/80、
`__launch_bounds__(512,2)`、寄存器压力本身。

**修正后的结论**：在 Turing + fp8 KV 上，这条路的**实际上限远低于 §6.9 的预期**。原因有三，
且互相叠加：

1. fp8→fp16 必须软件转换，且 wmma 要求操作数先进 smem → 每读一个元素"转换 + 2 字节 smem 写"是
   固定开销，实测把有效带宽压到 ~168 GB/s（与 flashinfer 的 161 GB/s 同一量级，**不是巧合**）；
2. 在这之上，复杂 kernel 里的循环还会被 ptxas 调度得更差（无法归因的 2.5 ms）；
3. 于是"手写 kernel 大幅超过 flashinfer"没有依据——**flashinfer 的 161 GB/s 很可能已接近
   Turing + fp8 + wmma 这条技术路线的实际上限**。

因此本部署的建议：**停止 kernel 重写**。已落地的 n=5（250K 上 +37%）仍是长上下文解码最实际、
最划算的收益；注意力再做深挖的性价比不足以支撑继续投入（除非换 fp16 KV 并把上下文降到 ~128K，
那时 §6.9 的表可以复用，但代价是丢掉 256K）。

### 6.11 补充实测：fp16 KV 确实更快；staging 的 2 字节写是真实缺陷

**一、fp16 KV 在 Turing 上真的更快**（同 harness、250K、q_len=6、flashinfer）：

| KV dtype | 注意力 kernel | 相对 |
|---|---|---|
| fp8_e4m3 | 1.456 ms | — |
| float16 | **1.170 ms** | **−18%** |

即**在 Turing 上 fp8 KV 对注意力速度是净负收益**：带宽省下的一半，被软件反量化开销吃掉还倒亏
（与 §6.7 在 vLLM 128K 上量到的 −2.7% 步耗时一致，只是这里同 harness 更干净）。
代价是 KV 占用翻倍：**256K 装不下（约能到 200K），128K 及以下完全装得下**。
→ 若某天要压 128K 档的延迟，把它切到 fp16 KV 是一个免费（无需改 kernel）的 2–3% 步耗时收益。

**二、staging 的 2 字节 smem 写是真实缺陷（已定位并修好）**：

| staging 写法（读 128 MB 的 K，fp8） | 时间 | 说明 |
|---|---|---|
| 每线程 16 个 fp8 → 16 次 2 字节 smem 写（原写法） | 0.761 ms | 4-way bank conflict |
| 每线程 8 个 fp8（8 B 载入）→ **一次 16 字节向量写** | **0.274 ms** | **2.8× 快，468 GB/s** |

修好后 fp8 staging 已经比 fp16（0.444 ms，读 2 倍字节）更快，说明"fp8 读得少"的优势在写法正确
时才兑现。把该修复应用到拆分版两个 kernel 后：6.79 → **5.92 ms**（省下的正是这段 staging）。
但相对 flashinfer 的 1.46 ms 仍慢 4 倍 —— §6.10 里那个**无法归因的约 2.5 ms/ kernel 的损耗
依旧存在**（已排除占用率、寄存器数、unroll、预取、noinline、smem carveout、转换指令数），
所以"停止 kernel 重写"的结论不变；但"2 字节粒度 smem 写 + 软件 fp8 反量化在 Turing 上很贵"
这两条教训是通用且可迁移的，已记录在此。

### 6.12 250K 单步成本完整账（n=3 trace 实测 + n=3/5/7 对照）

对已有 250K trace（rank0，`n=3` 时采集）取 5 个连续 step（每步 63.8 ms，含 profiler 约 1.38×
放大）按 kernel 分类，得到可对齐的完整账目：

| 类别 | 次/步 | μs/步 | 占比 | 可动性 |
|---|---|---|---|---|
| 注意力 causal（16 层 verify + MTP 层） | 17 | 19896 | 37.5% | 见 §6.9–6.11，**关闭** |
| 注意力 non-causal（draft 用，见下） | 2 | 2371 | 4.5% | 关闭 |
| Marlin INT4（主干 GEMM） | 255 | 15467 | 29.1% | 已在带宽下限（§6.7） |
| **lm_head（`gemvx`，vocab 124160 的 GEMV）** | 13 | **8035** | **15.1%** | **只能靠量化 head** |
| cutlass f16 GEMM（含 lm_head 大 M 部分） | 55 | 3324 | 6.3% | 同上 |
| 通信（TP allreduce） | 138 | 976 | 1.9% | 关闭（§6.7） |
| GDN decode | 48 | 823 | 1.5% | 关闭（§6.7） |
| 其余（norm/epilogue/sampling 等） | 600+ | ~2200 | 4% | 关闭 |

**lm_head 的定量确认**：n=3 时 `gemvx` 8.0 ms/步 ≈ 4 遍（verify 1 + draft 3）→ 每遍约 2.0 ms，
与"读 1.27 GB/卡 ÷ 583 GB/s ≈ 2.2 ms"一致；交叉验证：实测步耗时 n=3 ≈46 ms → n=5 = 58.5 ms，
每多一个 draft 约 +6.25 ms，其中注意力 1.18 ms、其余 ~5 ms 都是这一步的 head + MTP 层。
**折算到 n=5：lm_head ≈ 6 遍 ≈ 13 ms/步 ≈ 22%**，是除注意力外最大的单项，且只随 draft 数增长
（这正是 n>6 回退的原因，见 §6.2）。

**draft 注意力的 kernel 归属（Step 2）**：step 内时序显示 3 次 draft 注意力分别在
+0.36/+6.00/+11.33 ms，每次 **1.18 ms —— 与 verify 的 16 层完全相同**，说明 q_len=1 的 draft
**也走 flashinfer 的 prefill kernel**，且和 verify 一样是 KV 流量受限（独立 harness 里 q_len=1
是 1.45 ms、q_len=6 是 1.46 ms，**几乎相同**）。
> **2026-09-19 更正**：先前记的"SM75 上没有可用的 decode kernel"是**错的**，那是把
> `BatchDecodeWithPagedKVCacheWrapper.plan()` 的实参名写错（该接口的参数名是
> `indptr/indices/last_page_len/...`，与 prefill 的不同）导致的 `KeyError`。用正确的关键字
> 参数后，**非 tensor-core 的 decode 路径在 SM75 上可以 plan、可以 run**（KV=1024 出结果正确），
> 只是**首次调用耗时约 500 s** —— 几乎可以肯定是 flashinfer 的一次性 JIT 编译（本机编译较慢，
> prefill 模块当时约 65 s，decode 的 dispatch 更大）。
> 但进一步实测表明这条路**实际上不可用**：不是编译、也不是缺 kernel，而是**该非 TC decode
> kernel 在 SM75 + 本形状（page 16、head_dim 256、fp8 KV）上运行病态** —— KV 仅 1024 token 时
> 首次调用就耗 **500 s**（正常应 ~10 μs），换成 250K 后在 GPU 上跑满 100%、超过 6 分钟不返回
> （手动 kill，进程清掉后 GPU 立即恢复空闲，生产实例未受影响）。所以它是**坏的**，不是"能用但慢"。
> 因此 draft 改走 decode 路径**关闭**，但原因与先前记的"没有 kernel"不同。
> 另：`decode.cuh` 里只有 SM90+ 的 arch 条件（FA3 路径），**非 TC decode kernel 本身没有
> SM75 门槛**；真正需要 SM80+ 的是 FA2/tensor-core decode 与 split-KV。

**结论**：250K 单步的 58.5 ms 已完整归因（注意力 22.3 + 主干 GEMM 15.5 + lm_head 13 + 其它 ~7.7），
其中除 **lm_head 量化（改 checkpoint，含精度/接受率风险，见下表）** 外，其余各项都已在带宽下限
或已被实测关闭。decode 侧在本硬件 + 本精度方案下已基本挖尽。

### 6.13 500K 档（无 offload）实测：45 万 token 上下文（2026-09-19）

用 `500k.sh`（`MAX_MODEL_LEN=500800`、`KV_CACHE_MEMORY_BYTES=9.6e9`、不挂 offload）在本机
（双 2080 Ti、**16 GB 主机**）实测，单请求 prompt **448,747 token**、生成 256 token：

| 指标 | 250K（§6.7/§6.2 对照） | **449K（本次）** |
|---|---|---|
| KV 池 | 267,842（n=5 时 5.3e9） | **515,929**（9.6e9；≥ max-model-len ✓ 一个满长度请求装得下） |
| 显存 | 17.5 GB/卡 | **21.7 GB/卡**（util 0.92，紧但正常） |
| prefill | ~620 tok/s | **412.5 tok/s → TTFT 1087.8 s** |
| **稳态 decode** | 61.7 tok/s（n=5，接受率 ~40%） | **35.0 tok/s**（n=5，接受率 **42.6%**） |
| 每 token 成本 | 16.2 ms | **28.6 ms（×1.77）** |
| 整窗 decode（早期口径，偏乐观） | 80.7 | 58.3 |

结论与意义：

1. **>262K 的 YaRN 长上下文路径验证通过**：448,747 token 的 prompt 能正常 prefill/解码，没有形状
   或 rope 相关的报错；而且**稳态接受率 42.6% 与 250K 的 ~40% 持平**——若 rope 缩放有问题，MTP 的
   draft 会立刻失准、接受率塌掉，所以这是长上下文输出未崩的一个有用间接信号。
2. prefill 从 ~620 降到 **412.5 tok/s（0.67×）**，与注意力的 O(n²) 增长吻合；因此 449K 的首字约
   **18 分钟**（1088 s）。
3. decode 每 token 成本 ×1.77，与每次 attention 的 KV 读量 448/250 = 1.79× **几乎完全一致**——再次
   印证长上下文解码是 KV 带宽受限，而不是别的。
4. 未做：offload 三档（`RAMx1_SSDx4` 16.3 GiB / `RAMx2` 32.5 GiB staging，需 64 GB 主机）；
   n 在 449K 下的最优值（接受率略高于 250K，理论上更大 n 更划算，但每换一次 n 需重启 + 18 min
   prefill）。

### 6.14 fp16 KV + n=6：本机最大可跑上下文档的 A/B（2026-09-19）

新建 `scripts/run_vllm_qwen38_awq_fp16_225k.sh`（自 `500k.sh` 复制，只改三处：`--kv-cache-dtype
float16`、`SPEC_NUM_TOKENS=6`、`MAX_MODEL_LEN=225280`，KV 预算仍 9.6e9），在本机实测：

- 容量：池 **252,223 token**（fp16 是 fp8 的 ~2× 每 token 体量：38.06 KB/token），≥ 225,280 且
  余量 12%；每卡显存 21.7 GB（与 500K/fp8 档相同，实测可跑）；
- 本机可跑的**最大上下文**由此确定：`max_model_len` 最高 ≈ **248K**，取 225,280（220K）留裕度。

**同一 prompt（215,026 token）的 A/B**：

| | fp8_e4m3 + n=5（256K 档，基线） | **fp16 + n=6（新档）** | 变化 |
|---|---|---|---|
| TTFT / prefill | 321.6 s / 668.5 tok/s | 318.3 s / **675.6 tok/s** | 持平（fp16 不拖慢 prefill） |
| **稳态 decode** | **37.8 tok/s** | **42.7 tok/s** | **+13.0%** |
| 稳态接受率 | 29.8% | **31.7%** | 略升 |
| 每 token 成本 | 26.5 ms | 23.4 ms | **−11.6%** |
| 每轮 token 数 | 2.49 | **2.90** | +16% |

解读：

1. **+13% 高于 §6.11 预估的 +7~8%**，多出来的部分来自**接受率同时上升**（29.8% → 31.7%）：fp16
   KV 更精确 → draft 采得更准 → 每轮 token 数从 2.49 升到 2.90（+16%），而每轮成本只升 3%
   （n=6 多出的那次 draft，被 fp16 省下的注意力开销抵掉大半）——两个效应同向叠加。
2. **prefill 不受影响**（668.5 → 675.6 tok/s）：fp16 让 KV 读量翻倍，但省掉了软件 fp8 反量化，
   prefill 是算力受限，两者相抵。
3. **代价是上下文**：这个档最高 225K，**不能替代 500K 需求**；把它当作"≤225K 场景下的速度取向档"。
4. **500K 档的 n=6：已实测**（同 profile、同 prompt 448,747 token、同 9.6e9 预算）：

   | 449K | n=5 | **n=6** | 变化 |
   |---|---|---|---|
   | TTFT / prefill | 1087.8 s / 412.5 tok/s | 1079.4 s / 415.7 tok/s | 持平 |
   | **稳态 decode** | 35.0 tok/s | **36.9 tok/s** | **+5.4%（+1.9 tok/s）** |
   | 稳态接受率 | 42.6% | 42.6% | 相同 |
   | KV 池 | 515,929 | 509,877 | 仍 ≥ 500,800 ✓ |

   **零代价**：同样的 KV 预算（不用加内存、不用降上下文）、接受率不变（说明输出质量路径没动）。
   n=6 至此在 31.5K / 215K / 250K / 449K 四处实测都比 n=5 快（+4.7% ~ +13%）。

  ### 6.15 lm_head 量化（改 checkpoint）：每步省 10 ms，但 MTP 接受率下降（2026-09-19）

  §6.12 的成本账把 lm_head 列为除注意力外最大的单项（n=5 时 ~13 ms/步 ≈ 22%），并注明
  "只能靠量化 head，改 checkpoint，含精度/接受率风险"。本节是这条路的实测。

  **做法**：`scripts/tools/quantize_lm_head.py` 把 checkpoint 里的 `lm_head.weight`
  （bf16 [248320, 5120]，2.37 GiB）量化成与主干**完全相同的 compressed-tensors
  pack-quantized 方案**（4-bit int、group 32、非对称、mse observer），产出变体目录
  `models/Qwen3.8-27B-AWQ-INT4-yarn512k-head4bit/`（新 file1 0.68 GiB + 其余符号链接 +
  新 index + 新 config）。工具用 compressed-tensors 自己的
  `pack_to_int32`/`unpack_from_int32` 打包，并做**打包往返断言**（pack→unpack 必须等于
  量化值）与逐块反量化误差统计，保证 on-disk 布局就是 loader 会解出来的布局。

  **两个必须同时改 config 的地方（否则静默不生效）**：

  1. `config_groups.group_0.targets` 必须**加上 head 的真实 vLLM 前缀
     `language_model.lm_head`**（不是 `lm_head`）。本模型跑的是
     `Qwen3_5ForConditionalGeneration`，语言模型挂在 `language_model` 下，实测
     `get_quant_method` 收到的 prefix 就是 `language_model.lm_head`。而
     `find_matched_target` 是先按 layer_name 精确匹配、再按类名匹配，`_match_class`
     只对 `LinearBase` 特判 `"Linear"`，**`ParallelLMHead` 的 MRO 里没有
     `LinearBase`** —— 所以 `targets: ["Linear"]` 永远匹配不到 head。工具同时写
     `lm_head` 与 `language_model.lm_head` 两个目标（纯文本顶层模型用前者）。
  2. 从 `ignore` 里删掉 `"lm_head"`。实测这条其实是**冗余**的
     （`should_ignore_layer("language_model.lm_head", ["lm_head"]) == False`）：
     head 未量化原本只是因为不匹配 `Linear` 目标，与 ignore 无关；删掉只为清楚。

  改完后 vLLM 走 `CompressedTensorsLinearMethod`，与主干同一套 `MarlinLinearKernel`
  （SM75 上已验证的路径），启动日志会出现第二条
  `Using MarlinLinearKernel for CompressedTensorsWNA16`，加载零错误。

  **实测**（同一 profile：500K 档、fp8 KV、n=6；两次都冷启到干净状态，各 3 次 run）：

  | 指标 | 基线（原 checkpoint） | 变体（head int4） | 差 |
  |---|---|---|---|
  | head 体积 | 2.37 GiB (bf16) | **0.68 GiB** | 3.46× |
  | 每卡显存 | 21.6 GB | **20.85 GB** | −1.15 GB |
  | KV 池 / n | 509,877 / 6 | 509,877 / 6 | 不变 |
  | **每步耗时 @31.5K** | 52.1 ms（52.2/51.4/52.8） | **42.4 ms**（42.2/42.1/42.8） | **−9.7 ms（−18.6%）** |
  | **每步耗时 @215K** | 71.0 ms（71.1/70.5/71.4） | **61.0 ms**（61.7/61.0/60.4） | **−10.0 ms（−14.1%）** |
  | MTP 接受率 @31.5K | 36.3% | 28.4% | −7.9 点 |
  | MTP 接受率 @215K | 29.3% | 26.7% | −2.6 点 |
  | 净吞吐 @31.5K | 58.8 tok/s | 63.8 tok/s | +8.5% |
  | 净吞吐 @215K | 38.8 tok/s | 42.6 tok/s | +9.8% |
  | greedy 输出 | `79c93904b4648be3`（542 字符） | `19258efd04a014be`（557 字符） | 第 223 字符起改写 |

  **结论与判断**：

  - **纯速度效果确凿**：每步省 **9.7~10.0 ms**，两种上下文几乎一致（head 成本与上下文
    长度无关，正合预期）；同配置内 σ≈0.5 ms。**评估这个改动要看每步耗时，不要看
    tok/s** —— tok/s 会被接受率波动污染（同配置 3 次 run 的接受率能差 15 点）。
  - **代价是接受率**：head 同时是 MTP draft 的输出层（`mtp` 模块没有自己的 head，
    15 个键里没有任何 head，draft 复用主 `lm_head`），量化后 draft 的 logits 变糙 →
    接受率下降 2.6~7.9 点。greedy 输出从第 223 字符起改写（"for the following
    reasons" → "because"，语义等价），说明 head 量化**确实改变了输出分布**，不是无损。
  - **净吞吐只 +8~10%**，因为接受率的损失吃掉了大半步耗时收益。
  - 量化误差：mean|dW| = 0.00091、相对 8.42%、RMSE 0.00110（int4 group-32 的正常量级）。

  **未采用**：默认 profile 仍指向原 checkpoint（head 为 bf16）。变体目录、工具、以及
  临时 launcher（`~/Temp/opencode/run_500k_head4bit.sh`，sed 注入 MODEL_PATH）都留着，
  切换只需把 `MODEL_PATH` 指到变体目录。

  **下一步候选**：

  1. **int8 head**：只省 ~5 ms/步，但精度损失应小得多 → 接受率几乎不掉，很可能是比
     int4 更优的折中。工具已支持 `--bits 8`（需给 head 单开一个 num_bits=8 的
     config group，并把它的名字从 group_0 的 targets 里去掉）。
  2. int4 的**更细 MSE 搜索**或 GPTQ 式误差补偿，把 8.42% 的相对误差压下去。
  3. head 量化后每卡空出 1.15 GB → KV 预算可再加 ~0.9e9（约 +9.5 万 token 池），
     与量化收益叠加。

  ---

## 7. 已知问题与注意事项

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
均正确报告 `MTP=5 block=1600`；128K profile 经其校验为
"池 3.0e9 → 安全上限 142,400 ≥ 131,072 ✓"。

---

## 8. 待办与未覆盖

1. offload 尚未覆盖：**池接近满时的恢复**（最高优先，唯一可能 stall 的路径）、多轮
   offload/restore 的长时间稳定性、`max-num-seqs > 1` 的并发，以及 pinned / unpinned DMA
   的性能差（后者要 root 或 ≤8 MB 的 staging，实际做不了）。
   磁盘层写满的行为已在 §5.4 实测，字节上限 + LRU 淘汰已在 §5.5 实现并实测。
2. 磁盘层预算的**长稳**：多天运行下 mtime 作为 recency 的退化（例如备份/rsync 改写 mtime）
   尚未验证。
3. **500K 三档的端到端**（§5.6）：内存到位后在 64 GB 主机上各跑一次
   （冷启 → 重发 → 跨会话换出/恢复），并把 `RAMx2` 的 `/dev/shm` remount 纳入装机清单。
