# Qwen3.8-27B：双 RTX 2080 Ti 22GB + NVLink 的 vLLM 部署（上游分支）

> **本分支 `2080ti_dual_qwen38-27B`**：把 SM75 部署改动重新落到**上游 vLLM main** 上。
> 开分支 / 移植范围 / 编译环境修补 / 部署与 128K offload 实测记录见
> [`docs/upstream-branch.md`](docs/upstream-branch.md)。

## 本分支做了什么

先把 9 个文件的 SM75 / Qwen3.8 移植落到上游 vLLM `main`（`49f68ba24`），再在其上做三处改动，
单并发解码因此明显快于 zyYuc 的实现（同一台机器、同一套参数与测量方法，见下文性能验收）：

| 单并发稳态解码（31.5K 上下文） | 每步耗时 | decode |
| :-- | --: | --: |
| zyYuc 的实现（vLLM 0.27.2.dev16） | 59.1 ms | 53.7 tok/s |
| **本分支（`2080ti_dual_qwen38-27B`）** | **37.5 ms** | **81.2 tok/s** |
| 提升 | **1.58×** | **+51%** |

- **MTP 下保住 FULL cudagraph**（`059727bfa`）：SM75 的投机验证留在 FlashInfer native decode
  路径，不再被降级成 PIECEWISE —— 上表提升的主要来源（本分支自身的开关对照：54.1 → 37.5 ms/步）。
- **上游 fs 层 KV 字节预算 + LRU 淘汰**（`56c60a25f`）：上游默认不回收，磁盘占用随 spill 单调
  增长；本分支让 tier 自管预算，超了淘汰最旧。
- **`cudaHostRegister` 粘性错误清理**（`737fea73b`/`bf78fc276`）：低 memlock 主机（本机 8 MB
  硬顶）上 staging 注册失败不再毒化 CUDA context，offload 档能稳定启动。

定位过程（profiler + GPU 采样：CPU-launch-bound，非功耗/带宽受限）与逐档数据见下文
[跑通后的性能验收](#跑通后的性能验收)；完整记录见
[docs/upstream-branch.md](docs/upstream-branch.md) §6。

这是一个独立的开源部署项目，面向 2 张魔改 RTX 2080 Ti 22GB、并且两卡之间已连接双 NVLink 的用户。

目标是把一套正在运行的 Qwen3.8-27B 长上下文配置完整公开：硬件、驱动、加速路径、补丁、Jinja 模板、环境变量、systemd 和完整加载参数都在这里。

适合：单机双卡、单请求优先、256K（生产）/ 500K（三档：无 offload / RAM×2 / RAM×1+SSD×4）/
128K（tiered offload 验证）上下文、个人/小团队 API、长文档与代码任务。

不包含：模型权重、API Key、内网地址、个人目录、SSH 或隧道配置。

## 本分支新增

本分支从**上游**（[zyYuc](https://github.com/zyYuc)）切出，把 SM75 部署改动重新落在上游
vLLM `main` 上（vLLM 侧对应分支 `2080ti_dual_qwen38-27B`）：

- **SM75 / Qwen3.8 移植**：9 个文件（FlashQLA legacy GDN prefill、Qwen3.5 MTP、SM75
  spec-decode 同步、FlashInfer 的 SM75 支持判定等），即
  `patches/vllm-v0.27.1-sm75-qwen3.8.patch` 对应的改动。
- **MTP 下保住 FULL cudagraph**：让 SM75 的投机验证留在 FlashInfer native decode 路径
  （`VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE=1`，各 profile 默认开启），单并发稳态解码
  **54.1 → 37.5 ms/步**（见下节与 docs/upstream-branch.md §6）。
- **上游 fs 层 KV 字节预算 + LRU 淘汰**（vLLM `56c60a25f`，5 文件）：上游 fs 层默认不回收，
  占用随累计 spill 单调增长（约 4.5 GB / 条 120K 链）；本分支让 tier 自管预算
  （`VLLM_SSD_MAX_BYTES`），超了按 LRU 淘汰最旧的整块文件。
- **`cudaHostRegister` 粘性错误清理**（vLLM `737fea73b`/`bf78fc276`）：staging 注册失败会毒化
  CUDA context（表现为 warmup 的 `torch.full` 报 `invalid argument`），清理后低 memlock 主机
  （本机 8 MB 硬顶）上 offload 档也能稳定启动。
- **256K 生产 profile**：`scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh` —— 模型原生上限
  （262,144）、无 offload，池 278,253 tokens（实测），显存 17.5 GB/卡；另有
  `..._256k_RAMx1_SSDx4.sh`（同上下文 + 两层 offload，长 prompt 的 KV 跨重启可恢复，
  需 ~10 GB `/dev/shm`，即 32 GB 级主机）。
- **500K 部署三档**：`..._500k.sh`（无 offload）、`..._500k_RAMx2.sh`（CPU 层当 store）、
  `..._500k_RAMx1_SSDx4.sh`（RAM staging + 磁盘 LRU 环）。
- **128K 验证档**：`scripts/run_vllm_qwen38_awq_fp8e4m3_128k_RAMx1_SSDx4.sh` —— 128K
  上下文 + 上游 tiering offload（RAM 1 条链 staging + 磁盘 4 条链的环），用于验证而非服务。
- **编译环境修补**与 128K 驱逐/恢复实测见 [`docs/upstream-branch.md`](docs/upstream-branch.md)。

## 已验证环境

| 分类 | 参数 |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti 22GB（22,528 MiB / 卡） |
| GPU 架构 | Turing / SM75 / Compute Capability 7.5 |
| GPU 互联 | NV2：每张卡 2 条 NVLink；单条实测约 25.781 GB/s |
| CPU | Intel Xeon E5-2696 v3，18 核 36 线程 @2.30GHz |
| 内存 | 15 GiB（swap 19 GiB） |
| OS / Kernel | Ubuntu 24.04.4 LTS / Linux 7.0.0-31-generic |
| NVIDIA Driver | 580.173.02 |
| CUDA Runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.26.1rc1.dev2278+g49f68ba24（上游 vLLM `main`；vLLM 侧分支 `2080ti_dual_qwen38-27B` @ `059727bfa`） |
| Transformers / Triton | 5.16.1 / 3.7.1 |
| FlashInfer | 0.6.18.post1 |
| NCCL | 2.29.7 |

上表是**本分支路线**（上游 `main` + 移植提交）的实测环境，锁定清单见
[docs/environment-lock.md](docs/environment-lock.md)；**基础路线**（vLLM `v0.27.1` +
`patches/vllm-v0.27.1-sm75-qwen3.8.patch`，FlashInfer 0.6.16.post3）的版本锁定见
[docs/PATCHING.md](docs/PATCHING.md) 与 [docs/ACCELERATION_AND_ATTRIBUTION.md](docs/ACCELERATION_AND_ATTRIBUTION.md)。

## 这套配置的目标

- TP=2：两张卡共同加载 Qwen3.8-27B。
- **生产 profile**（`scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh`）：AWQ-INT4 权重
  （SM75 没有 FP8 Tensor Core，FP8 权重要反量化走 FP16 GEMM，INT4/INT8 才是快路径，见下节）
  + fp8_e4m3 KV Cache；`max-model-len=262144`（模型原生上限，无需外推），启动日志可用
  KV Cache **278,253 tokens**，显存 17.5 GB/卡。
- **长上下文档**：500K 三档 —— `..._500k.sh`（无 offload，池 525,229）、`..._500k_RAMx2.sh`、
  `..._500k_RAMx1_SSDx4.sh`。
- **验证档**：`..._128k_RAMx1_SSDx4.sh` 验证上游 tiering offload（RAM staging + 磁盘环），
  不作为服务 profile。
- max-num-seqs=1：优先长上下文单请求，不按高并发路线配置。
- Prefix Cache + Chunked Prefill：改善固定系统提示词和超长输入。
- MTP=3 + **FULL CUDA Graph**：`VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE=1` 让投机验证留在
  decode 路径，MTP 下不再降级为 PIECEWISE（各 profile 默认开启）。
- flashqla_legacy：为 SM70/SM75 的 Qwen GDN prefill 提供兼容加速路径。
- Qwen3 thinking、XML tool calling、修复版 chat template：全部包含在启动配置中。
- **基础路线**（vLLM `v0.27.1` + `patches/vllm-v0.27.1-sm75-qwen3.8.patch`）保留原 FP8 / 180K
  启动器 `scripts/run_qwen3.8_27b_sm75.sh`，见 [docs/PATCHING.md](docs/PATCHING.md)。

## 目录

~~~text
config/       环境变量样例（基础 + 500K 三档 / 128K offload，共 4 个）
docs/         打补丁、加速组件与上游分支记录（docs/patches/ 存"已评估未采纳"的补丁）
patches/      已验证工作树导出的 vLLM / FlashQLA patch（会被套用）
scripts/      启动 profile（256K 生产 + 500K 三档 + 128K offload 验证 + 基础路线）与硬件检查
scripts/setup/  硬件与依赖准备
scripts/tools/  启动看护 / 精准停止、池容量测算、KV offload 信息面板（终端 + 浏览器）
systemd/      常驻服务模板
templates/    qwen3.8-froggeric-v22.3 Jinja 模板源文件
benchmarks/   上下文梯度 TTFT / decode 基准与原始结果
reports/      2026-09 优化战役报告（整体报告 + 6 条支线，含原始 JSON）
~~~

## 快速开始

### 1. 验证硬件

~~~bash
bash scripts/setup/verify_hardware.sh
~~~

关键拓扑应包含：

~~~text
GPU0  GPU1
GPU0   X   NV2
GPU1  NV2   X
~~~

没有 NV2 也可能运行，但双卡 TP 的通信条件与本配置不同。

### 2. 获取源码

两条路线，按 docs/PATCHING.md 锁定 vLLM、FlashQLA、FlashInfer 版本：

- **本分支路线（`2080ti_dual_qwen38-27B`）**：上游 vLLM `main` + SM75/Qwen3.8 移植提交（9 个文件），
  **不套用** `patches/`；移植范围与编译环境修补见 docs/upstream-branch.md §2/§3。
- **基础路线**：上游 vLLM `v0.27.1` + `patches/vllm-v0.27.1-sm75-qwen3.8.patch`。

### 3. 填写你的路径

~~~bash
cp config/vllm.env.example .env
~~~

至少修改：

~~~bash
MODEL_PATH=/你的/Qwen3.8-27B-AWQ-INT4-yarn512k/模型目录   # 本分支各 profile 共用
# 基础路线（FP8 / 180K）用：MODEL_PATH=/你的/Qwen3.8-27B-FP8/模型目录
VLLM_PYTHON=/你的/venv/bin/python
FLASHQLA_PATH=/你的/FlashQLA-SM70-SM75
~~~

CHAT_TEMPLATE 默认指向本仓库内的 templates/qwen3.8-froggeric-v22.3.jinja。

### 4. 启动

~~~bash
bash scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh     # 生产 profile（AWQ-INT4 / 256K / 无 offload）
# 长上下文：..._500k.sh、..._500k_RAMx2.sh、..._500k_RAMx1_SSDx4.sh
# offload（KV 跨重启可恢复）：..._256k_RAMx1_SSDx4.sh（需 ~10 GB shm）、..._128k_RAMx1_SSDx4.sh
# 基础路线（FP8 / 180K）：scripts/run_qwen3.8_27b_sm75.sh
~~~

引擎要加载权重并编译数分钟，而且**失败也正是发生在这段时间内**（见下一节）。别用固定
`sleep` 去盯，用仓库自带的看门狗——`/health` 变 200 或日志里出现致命模式时**立即返回**：

~~~bash
bash scripts/tools/wait_server.sh logs/server.log 600
# 0 = ready  1 = 出错（并打印命中的那一行）  2 = 超时
# 端口默认取 .env 的 PORT（其次 8000）
~~~

停实例按**端口**精准定位，只动它的进程树（多实例共存时不会误杀）：

~~~bash
bash scripts/tools/stop_server.sh --list              # 列出所有实例：pid / 端口 / engine / model
bash scripts/tools/stop_server.sh 8000                # 停 8000（先 TERM 主进程，必要时升级整组，最后 KILL）
bash scripts/tools/stop_server.sh --port 8001 --dry-run   # 只打印将会杀谁，不发信号
# 默认顺带删掉该实例的 /dev/shm/vllm_offload_<engine_id>.mmap 并打印前后用量（--no-clean-shm 可关）
~~~

起来之后用只读看板盯 KV / offload 的实时状态（不需要 dev-mode，只用标准库）：

~~~bash
python3 scripts/tools/monitor_kv_offload.py            # :8000，5s 刷新
python3 scripts/tools/monitor_kv_offload.py --port 8001 --once
python3 scripts/tools/monitor_kv_offload.py --json --count 5
~~~

同一个面板还有**浏览器版**（纯标准库、内联 CSS/JS、不依赖外网；滚轮/缩放天生可用；页头显示
live 解码速率，含 prefill、历史均值与 MTP 接受率）：
`python3 scripts/tools/monitor_kv_offload_web.py` → `http://127.0.0.1:8199/`。

`CONFIG` 取自 `/proc/<pid>` 与 `vllm:cache_config_info`，`STATUS` 取自 `/metrics`，
`CHUNKS` 取自磁盘层目录（说明见 docs/upstream-branch.md §5.7）。

### 5. 启动排障（本机实测）

三项都不涉及模型本身，但会直接导致"起不来"或"时好时坏"：

**1. `/dev/shm` 必须留够空间（最重要）**

offload 的 staging 区是 `/dev/shm/vllm_offload_<engine_id>.mmap`，大小等于
`CPU_BYTES_TO_USE`（128K profile 为 **4.57 GiB**），加上 vLLM 自身的 POSIX shm 约 0.7 GiB，
**单实例需 ≈5.3 GiB**（本机 tmpfs 共 7.8 GiB）。
进程被 kill/崩溃时该文件不会被删（只有正常退出才 `unlink`）；若 `engine_id` 是每次启动
随机的 UUID，残留文件永远不会被回收 → `/dev/shm`（本机 7.8 GiB）累积几次即满 →
驱动 `cuMemHostRegister_v2` 无法为 staging 落页，报
`Failed to allocate physical memory` 并返回 `CUDA_ERROR_INVALID_VALUE` →
**毒化 CUDA context** → warmup 里 `torch.full` 报 `CUDA error: invalid argument` → 引擎挂死。
表现就是"有时能起、清一下或重启就能起"。

对策（128K profile 已内置）：**固定 `KV_ENGINE_ID`**（每个 profile 一个，互不重复），
并在启动前 `rm -f /dev/shm/vllm_offload_<id>.mmap`，残留不再累积。磁盘二级层的目录由
模型路径派生（与 `engine_id` 无关），所以目录会跨重启复用（这是恢复的前提）；
磁盘占用由 `VLLM_SSD_MAX_BYTES`（默认 64 GiB）自动回收，需要推倒重来时再设
`VLLM_SSD_CLEAN_START=1` 清空 `VLLM_SSD_ROOT`。

> **起不来的第一反应：先清 `/dev/shm`（一键，别先查别的）**
>
> 只要服务是**异常退出或被 `kill`**（不是正常关机），staging 文件就会残留；换了
> `KV_ENGINE_ID` 反复试更是会一次留一个（128K profile 每个 ≈4.57 GiB）。启动前无论 `engine_id`
> 是什么，直接清干净：
>
> ~~~bash
> df -h /dev/shm                       # 看是否接近 100%（本机 tmpfs 7.8 GiB）
> ls -la /dev/shm/*.mmap               # 看残留了哪些
> rm -f /dev/shm/vllm_offload_*.mmap   # 一键清理（通配所有 engine_id）
> ~~~
>
> **判据**：日志里出现 `Insufficient space in /dev/shm`，或 warmup 阶段
> `torch.full` / `cuMemHostRegister_v2` 报 `CUDA error: invalid argument`
> （`qwen_triton_warmup.py` 附近），**几乎可以直接判定是这条**——`/dev/shm` 满或半满，
> 先清 shm 再谈其他，否则会白白怀疑补丁/编译/模型。
>
> **但先别只看 shm**：第 2 条（linger 未开）会给出**同样的** invalid argument 症状。
> 启动失败时**两条都做**：`loginctl enable-linger $USER` + 清 `/dev/shm`；
> 若是 `kill -9` 之后重启，清理完还要**多等约 20 秒**再启（见第 3 条）。

**2. 启动前先开 linger（不开也会让启动失败，不只是"会话结束被杀"）**

`loginctl enable-linger $USER` 是**一次性**设置，务必先做。linger 未开时，除了"SSH
会话一结束，systemd 停掉 `user@<uid>`、移除整个 user slice、**该用户所有进程被杀**"
（日志表现为 worker `died unexpectedly (exit code: None)` 且无 Python 栈）之外，
**实测还会在启动阶段让 `cudaHostRegister` 失败并毒化 CUDA context**——症状与第 1 条
**完全一样**（`Failed to allocate physical memory` →
warmup 的 `torch.full` 报 `CUDA error: invalid argument`），极易误判成 `/dev/shm`
问题。所以每次排查启动失败，先确认 `linger=yes`：

~~~bash
sudo loginctl enable-linger $USER      # 一次性
loginctl show-user $USER -p Linger     # 期望 Linger=yes
~~~

或者从常驻会话 / 系统服务启动（见下文 systemd 一节）。

> 实测记录：连续多次启动失败、按第 1 条清理 `/dev/shm` 也没稳定起，执行
> `loginctl enable-linger aiakos` 后再启动即成功。两条诱因会给出**同样的** invalid
> argument 症状，**先 linger、再 shm**，或两条都做。

**3. `cudaHostRegister` 偶发失败 → 清 shm + 等 20 秒再启**

`/dev/shm` 空间充足时该调用仍偶发返回 `cudaErrorInvalidValue`，且**失败即毒化 CUDA
context**（实测：失败后下一次 CUDA 调用必报 `invalid argument`）。带
`CUDA_LOG_FILE=stderr` 启动可看到驱动给的确切原因：

~~~text
[CUDA][E] Failed to allocate physical memory
[CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuMemHostRegister_v2
~~~

**实测最稳的"重启三连"**（`kill -9` 之后尤其重要）：

~~~bash
pkill -9 -f "[v]llm.entrypoints"; sleep 2
pkill -9 -f "[V]LLM::Worker"; pkill -9 -f "[E]ngineCore"
sleep 20                                  # ← 关键：等驱动回收 pinned 页 / mmap
rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/psm_*
# 然后再启动
~~~

`kill -9` 会打断正在进行的 CUDA 操作、留下未 unregister 的 pinned 内存与残留 mmap，
**立即重启时 `cudaHostRegister` 最容易失败**（且失败即毒化 context，表现为 warmup 的
`torch.full` 报 `invalid argument`）。清理后**多等约 20 秒**让驱动/nvrm 收尾再启动，成功率
明显更高；仍偶发失败时再重试即可。

## 128K 验证档：tiered offload 要点

`scripts/run_vllm_qwen38_awq_fp8e4m3_128k_RAMx1_SSDx4.sh` 是**验证档，不用于服务**：在 128K
上下文上启用上游的 tiering offload（CPU staging 主层 + 磁盘 fs 二级层），在小尺度上把 offload
的约束压出来。实测出的关键约束：

**CPU staging 层必须装得下整条链。** 上游会把促销（promote）回来的 chunk 保留在 CPU 层，
而不是只把它当流式 bounce buffer，所以容量不足时整次恢复作废
（表现为 `vllm:kv_offload_tiering_promotion_allocation_failures` 计数）：

~~~text
CPU_BYTES_TO_USE >= ceil(MAX_MODEL_LEN / 1600) × 55.8 MB
    实测 chunk 几何：55.8 MB/chunk，1 chunk = 1 block = 1600 tokens
    128K -> 82 chunks ≈ 4.58e9
~~~

同一 120K 会话被挤出后重发：

| staging | 是否装得下 | A 重发命中 | 耗时 |
| --- | --- | --- | --- |
| 2.4e9（43 chunks） | 否 | **0**（且已白写 11.24 GB、白读 2.4 GB） | 132 s（=冷启） |
| 4.6e9（82 chunks） | 是 | **118,400 / 120,000（98.7%）** | **5 s** |

即 staging 配小了不是"没效果"，而是**净亏 I/O**；profile 已内置启动自检，装不下一条满长链
时会打印 `[warn]`。

另外两点来自实测：

- **磁盘层有字节上限（`VLLM_SSD_MAX_BYTES`，默认 64 GiB）**：上游 fs 层默认不回收，
  占用随累计 spill 单调增长（约 **4.5 GB / 条 120K 链**）；本线让 tier 自管预算，超了就按
  **LRU** 淘汰最旧（恢复过的块算"用过"）整块文件，磁盘再也不会被写满。上限至少要装得下
  一条满长链，且假设 tier 独占该目录（多实例共享 `root_dir` 时不要设）。
- **不为上限时，写满不会崩、也不会中止请求**：只是 offload 静默失效 + 持续刷
  `Job N block I/O failed`，每个请求退化为全量重算。profile 已把
  `kv_load_failure_policy` 设为 `recompute`（vLLM 默认 `fail` 会中止受影响请求）。

- **500K 部署三档**（64 GB 主机目标）：`..._500k.sh`（无 offload）→ `..._500k_RAMx2.sh`
  （CPU 层当 store，容 2 条满长链 32.5 GiB，需把 /dev/shm remount 到 ~36 GiB）→
  `..._500k_RAMx1_SSDx4.sh`（RAM 只当 staging 16.3 GiB + 磁盘 4 条链的 LRU 环）。
  三档共用同一套 KV/池参数、尺寸由 `MAX_MODEL_LEN` 推导、自带装机自检
  （`/dev/shm` 总量与**剩余**空间、可用内存；不足即拒绝启动，`CHECK_ONLY=1` 只检查）。
  见 [docs/upstream-branch.md](docs/upstream-branch.md) §5.6。
- **256K offload 档**：`..._256k_RAMx1_SSDx4.sh` —— 同两层结构，链 9.15 GB / 磁盘环 36.6 GB，
  需 ~10 GB `/dev/shm`（约 32 GB 主机）；本机 15 GiB 上用 `CHECK_ONLY=1` 会直接拒绝。

> 排查提示：单请求下 A→B→A 的第三次可能被 **GPU 前缀缓存**冒领（实测出现过
> 118,400/120,000、3 s 的"假恢复"）；判断命中来源要看 `tiering_*` 指标，不能只看
> `cached_tokens`。完整记录（含 10K/40K 小尺度数据、`cudaHostRegister` 粘性错误与对应补丁）
见 [`docs/upstream-branch.md`](docs/upstream-branch.md)。

## 跑通后的性能验收

> **数据来源**：本节及下一节（含 `reports/`）的实测数据来自原项目（zyYuc）在其机器
> （AMD Ryzen 7 5700X / 32 GiB）上的运行；本分支的运行机器为 Intel Xeon E5-2696 v3 / 15 GiB，
> 硬件不同，下列数字仅供形态参考，未在本机复测。本分支自己的实测见
> [`docs/upstream-branch.md`](docs/upstream-branch.md)。

这一步是“AI 能否真的帮你跑到相近速度”的关键，而不是只看到服务能启动。这里的首字时间严格按模型流式输出的第一个思考字符或答案字符计算；Qwen 开始输出 think 中第一个字符，就视为首字。

~~~bash
python benchmarks/run_context_ttft.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen-local \
  --word-counts 2700 5400 8100 19000 57000 \
  --runs 3 \
  --max-tokens 512 --steady-from 128 \
  --output my-benchmark-result.json
~~~

`--max-tokens` 默认 512、`--steady-from` 默认 128：脚本除整窗平均 `decode_tok_s` 外，还会用引擎的
`vllm:generation_tokens_total` 计数器给出**尾部稳态** `decode_steady_tok_s` 与同窗口的 MTP 接受率。
短窗口（旧的 128-token 口径）只覆盖接受率最高的那一段，会明显偏高。

测试方法、真实流式原始结果和验收范围都在 benchmarks/README.md。下面是 2026-08-25 在当前线上服务重新跑出的上下文梯度结果；旧的 20K 合成压测表已移除，不再作为首页代表速度。

| 实际输入 | 平均首字时间（TTFT） | 平均 Prefill 速度 | 平均 Decode 速度 |
| :-- | --: | --: | --: |
| 2.84K tokens | 2.59 s | 1,099.8 tok/s | 97.3 tok/s |
| 5.64K tokens | 4.48 s | 1,260.2 tok/s | 94.3 tok/s |
| 8.45K tokens | 6.45 s | 1,311.2 tok/s | 101.1 tok/s |
| 19.77K tokens | 14.78 s | 1,337.6 tok/s | 101.3 tok/s |
| 59.24K tokens | 53.02 s | 1,117.3 tok/s | 84.4 tok/s |

首字的定义：流式 SSE 收到第一个非空 reasoning_content、reasoning 或 content 字符；Qwen 开始输出 think 内的第一个字符即计入首字。DSH 全任务平均首字应使用 DSH 的原始任务集单独复测，不能与本合成上下文梯度混用。

复现时应先锁定 docs/environment-lock.md，再按 benchmarks/README.md 的方法测试。相同硬件的合理验收范围是约 ±10%；显著偏离时按顺序检查：NVLink 是否为 NV2、TP 是否为 2、补丁是否生效、FlashQLA legacy 是否被日志选中、是否使用 FP8 KV、是否有其他 GPU 占用。

### 本分支实测：单并发稳态解码（2026-09-18）

口径与上表不同——这里是**长生成**的稳态（接受率 55–90%），不是前 128 token。同一台机器、
同一次会话、同一测量方法（30,300 词 prompt → 31,479 token，生成 384，seed 96001），每步耗时
按接受率归一（`step = 1000 × (1 + 3 × acc) / tok/s`）以消除采样随机性：

| 配置 | cudagraph | 每步耗时 | 单并发稳态 decode |
| :-- | :-- | --: | --: |
| zyYuc 的实现（vLLM 0.27.2.dev16） | 只有 PIECEWISE | 59.1 ms | 53.7 tok/s |
| 本分支，`VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE=0`（= 优化前行为） | 只有 PIECEWISE | 54.1 ms | 58.6 tok/s |
| 本分支，`=1`（默认） | FULL + PIECEWISE | **37.5 ms** | 81.2 tok/s |

读法：**本分支自身的开关对照是 54.1 → 37.5 ms/步（1.44×）**，这是图模式改动本身的收益；
zyYuc 的 59.1 ms 除图模式外还含路线/版本差异（它基于 vLLM 0.27.2.dev16，本分支基于上游 main）。

`tok/s = 步频 × (1 + n × MTP 接受率)`，所以前面 84–101 的 128-token 口径与这里的 ~50–80 稳态
并不矛盾：接受率 ~90% 时每个 4-token 步能出 3–4 个 token，接受率 ~50% 时只出 ~2.5 个。

定位过程（profiler + GPU 采样：CPU-launch-bound，非功耗/带宽受限）、n 扫描、已排除的杠杆，
以及一项**已评估未采纳**的改动（融合多步草稿解码，实测仅 +6%）见
[`docs/upstream-branch.md`](docs/upstream-branch.md) §6。开关 `VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE`
（各 profile 默认开启）置 0 即可回到原行为。

## 2026-09 性能更新：W8A8 vs FP8（同硬件、同 180K 条件）

2026-09-07 在同一台双 2080 Ti 上，把权重从 FP8 换成 W8A8（imatrix），保持 fp8_e4m3 KV + 180K 上下文不变，与上方 2026-08-25 基线同条件对比。W8A8 两列分别为 MTP3 与 MTP5。

**首字时间 TTFT（越低越好）**

| 实际输入 | FP8 + MTP3（上次基线） | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 2.59 s | **2.09 s（-19%）** | 2.19 s（-15%） |
| 5.64K tokens | 4.48 s | **3.32 s（-26%）** | 3.42 s（-24%） |
| 8.45K tokens | 6.45 s | **4.56 s（-29%）** | 4.75 s（-26%） |
| ~20K tokens | 14.78 s | **10.97 s（-26%）** | 11.35 s（-23%） |
| ~60K tokens | 53.02 s | **41.11 s（-23%）** | 42.24 s（-20%） |

**Prefill 速度（越高越好）**

| 实际输入 | FP8 + MTP3 | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 1,099.8 tok/s | **1,361（+24%）** | 1,298（+18%） |
| 5.64K tokens | 1,260.2 | **1,701（+35%）** | 1,652（+31%） |
| 8.45K tokens | 1,311.2 | **1,855（+41%）** | 1,780（+36%） |
| ~20K tokens | 1,337.6 | **1,900（+42%）** | 1,836（+37%） |
| ~60K tokens | 1,117.3 | **1,517（+36%）** | 1,476（+32%） |

**Decode 速度（128-token 流式口径，越高越好）**

| 实际输入 | FP8 + MTP3 | W8A8 + MTP3 | W8A8 + MTP5 |
| :-- | --: | --: | --: |
| 2.84K tokens | 97.3 tok/s | 67.9（-30%） | 86.9（-11%） |
| 5.64K tokens | 94.3 | 67.7（-28%） | 85.5（-9%） |
| 8.45K tokens | 101.1 | 74.0（-27%） | 77.2（-24%） |
| ~20K tokens | 101.3 | 78.9（-22%） | 70.7（-30%） |
| ~60K tokens | 84.4 | 75.3（-11%） | 85.2（+1%） |

- **W8A8（imatrix）权重量化是本轮最大单项收益**：首字时间降 19~29%，prefill 升 18~42%（SM75 无 FP8 Tensor Core，FP8 权重要反量化走 FP16 GEMM，W8A8 直接走 INT8 Tensor Core）。
- **代价是 decode 略慢**（-11%~-30%，128-token 口径）；MTP5 的 decode 更接近 FP8 基线。长输出优先的场景可看 W4A16（decode 约 2×，TTFT 劣于 W8A8，见 reports）。
- **MTP3 是甜点位**：MTP5 深层位置接收率坍缩（平均 44.8% vs 62.9%），不建议。
- 若能接受 65K 短上下文 + FP16 KV，W8A8+MTP3 的 ~60K 首字时间进一步降到 **35.76 s（-33%）**。
- 机理、全部变体数据、被排除的路线（TRITON_ATTN / FA2 d256 / SDPA / Triton-Turing fork）见 [reports/2026-09-sm75-optimization/](reports/2026-09-sm75-optimization/00-consolidated-report.md)。
- **W8A8 未做业务侧质量回归，切换前请先评测。**

## 完整加载参数与用途

下表是**生产 profile**（`scripts/run_vllm_qwen38_awq_fp8e4m3_256k.sh`）的完整参数；500K 三档只把
`--max-model-len` / `--kv-cache-memory-bytes` 换成 500800 / 9.6e9，其中 RAM×2 与 RAM×1+SSD×4
再叠加 tiering offload（见 [docs/upstream-branch.md](docs/upstream-branch.md) §5.6）。

| 参数 | 当前值 | 用途 |
| --- | --- | --- |
| --dtype | half | 运行时 FP16 计算 dtype。 |
| --tensor-parallel-size | 2 | 两张 GPU 做 Tensor Parallel。 |
| --device-ids | 0,1 | 明确使用 GPU 0、1。 |
| --quantization | （不传） | 由模型 config 自动识别为 AWQ-INT4（`Qwen3.8-27B-AWQ-INT4-yarn512k`）。 |
| --kv-cache-dtype | fp8_e4m3 | 用 FP8 E4M3 存 KV Cache，降低 KV 显存。 |
| --max-model-len | 262144 | 单请求上下文上限（模型原生 `max_position_embeddings`；500K 档为 500800）。 |
| --gpu-memory-utilization | 0.92 | vLLM 目标使用每卡 92% 显存。 |
| --kv-cache-memory-bytes | 5300000000 | 显式限制 KV Cache 显存预算（实测池 278,253 tokens；500K 档为 9.6e9 / 525,229）。 |
| --max-num-seqs | 1 | 单并发、长上下文优先。 |
| --max-num-batched-tokens | 1024 | 限制单轮调度 token，平衡峰值显存与延迟。 |
| --enable-prefix-caching | 开启 | 缓存重复系统提示词和前缀。 |
| --enable-chunked-prefill | 开启 | 长输入分块 prefill。 |
| --enable-prompt-tokens-details | 开启 | 响应里返回 prompt token 明细。 |
| --speculative-config | mtp / 3 | 每步最多预测 3 个 token。 |
| --additional-config | flashqla_legacy | SM75 GDN prefill 后端。 |
| --reasoning-parser | qwen3 | Qwen3 thinking 输出解析。 |
| --tool-call-parser | qwen3_xml | Qwen3 XML tool calling 解析。 |
| --default-chat-template-kwargs | enable_thinking=true | 默认开启 thinking。 |
| --chat-template | qwen3.8-froggeric-v22.3.jinja | 使用本仓库修复模板。 |
| --skip-mm-profiling | 开启 | 跳过启动时对**多模态编码器激活与 embedding cache** 的显存 profiling（只 profile 语言主干），省启动时间。代价是峰值显存要自己兜住——本部署仍是多模态服务（`--limit-mm-per-prompt` 为 20 图 / 1 视频），靠该上限与显存余量保证不 OOM。 |

两点与**基础路线**（`scripts/run_qwen3.8_27b_sm75.sh`，FP8 / 180K）不同：本分支不再传
`--no-async-scheduling`（实测会改变输出且略慢），也不传 `--compilation-config`
（图模式由 §6 的改动自动选择，MTP 下会捕获 FULL + PIECEWISE）。

## 环境变量与加速路径

| 环境变量 | 值 | 作用 |
| --- | --- | --- |
| OMP_NUM_THREADS | 8 | 本机 Xeon E5-2696 v3 为 18 核 36 线程，取 8 以控制线程开销。 |
| VLLM_USE_DEEP_GEMM | 0 | 关闭此配置未使用的 DeepGEMM 路径。 |
| VLLM_USE_FLASHINFER_SAMPLER | 0 | 关闭 FlashInfer top-k/top-p sampler。 |
| VLLM_QWOPUS_MTP_BF16_DRAFT | 1 | Qwen3.5 MTP draft 层兼容设置。 |
| VLLM_SM75_SPEC_SYNC_MODE | safe | SM75 speculative decoding 保守同步模式。 |
| VLLM_FLASHINFER_NATIVE_SPEC_AS_DECODE | 1 | 让 SM75 的 spec 验证留在 native FlashInfer decode 路径，MTP 下保住 FULL cudagraph；置 0 即回到优化前行为（54.1 → 37.5 ms/步）。各 profile 默认开启。 |
| VLLM_ALLOW_LONG_MAX_MODEL_LEN | 1 | 允许 `--max-model-len` 超过模型 config 的 262144（500K 档需要）。 |
| VLLM_SSD_ROOT | 路径 | offload 档的磁盘二级层根目录（仅 500K RAM×1+SSD×4 与 128K 档）。 |
| VLLM_USE_V2_MODEL_RUNNER | 1 | 使用 V2 model runner。 |
| PYTHONPATH | FLASHQLA_PATH | 让 vLLM 能导入 FlashQLA SM75 GDN backend。 |

完整来源、固定 commit、补丁边界和引用见 docs/ACCELERATION_AND_ATTRIBUTION.md。完整环境锁定清单见 docs/environment-lock.md。

## systemd 常驻服务

1. 把仓库放到 /opt/qwen3-8-27b-dual-2080ti-vllm，或修改 unit 中路径。
2. 把 config/vllm.env.example 复制为 /etc/qwen3.8-vllm.env 并填写路径。
3. 修改 systemd/qwen3.8-27b-vllm.service.example 中的 Linux 用户。
4. 安装并启动：

~~~bash
sudo cp systemd/qwen3.8-27b-vllm.service.example /etc/systemd/system/qwen3.8-27b-vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3.8-27b-vllm
sudo systemctl status qwen3.8-27b-vllm --no-pager
~~~

## 重要说明

- 模型权重不在本仓库内。请从拥有相应许可的来源获取模型。
- 这是 SM75 / 2080 Ti 的特化配置，不能把它当作 H100、4090、A100 或无 NVLink 双卡的通用最优参数。
- FP8 权重和 FP8 KV Cache 需要自行做业务精度回归，尤其是长上下文、数学、代码和工具调用。W8A8 / W4A16 量化 checkpoint 同理（见 reports/2026-09-sm75-optimization/）。
- 本仓库只公开部署配置和已导出的本地补丁；上游组件遵循各自许可证。

## 引用与致谢

感谢并请引用：vLLM、PyTorch、Hugging Face Transformers、FlashInfer、FlashQLA-SM70-SM75、NCCL、Triton-Turing（SM75 fork，reports 战役引用）。详细链接、commit 和许可证在 docs/ACCELERATION_AND_ATTRIBUTION.md。

本分支（`2080ti_dual_qwen38-27B`）把 SM75 部署改动重新落到上游 vLLM `main` 上，并新增 128K
offload profile，记录见 [`docs/upstream-branch.md`](docs/upstream-branch.md)，由
[Aiakos1818](https://github.com/Aiakos1818) 贡献，按本仓库 MIT 许可发布。
