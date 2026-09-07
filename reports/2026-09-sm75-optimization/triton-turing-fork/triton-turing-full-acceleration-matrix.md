# Triton-Turing 与双 22GB SM75 vLLM：完整加速试验矩阵

日期：2026-09-06  
目的：不把 2026-09-06 的一轮 TTFT 试验当作最终结论。系统性枚举可能影响结果的编译器、attention、KV cache、混合 SSM/GDN、量化、MTP、调度、并行和系统参数，使每一项都能独立重测、判断和回滚。

## 0. 当前事实、未知项与试验原则

### 0.1 已有事实（不是对新组合的否定）

2026-09-06 的回退记录显示：在 **Qwen3.8 27B FP8、TP=2、180K、4 GiB/卡 FP8 E4M3 KV、MTP=3、max-num-seqs=1** 的单一组合上，把编译器替换为 `triton-turing`、强制 `TRITON_ATTN`，并让 KV 写入复用 vLLM 原生 FP8 writer 后，2.7K/5.4K/8.1K/19K words 的单次 TTFT 分别比历史 FlashInfer 基线慢约 1% / 3% / 5% / 11%。

这只说明下列**耦合组合**没有赢：

`triton-turing + TRITON_ATTN + 原生 FP8 writer + 当时的模型/图捕获/MTP/调度参数`。

它**不能**推出：

- `triton-turing + FlashInfer` 没有收益；
- 标准 Triton 的 `TRITON_ATTN` 与 `triton-turing` 的差异已经被隔离；
- FP16 KV、FP8 E5M2、INT8/INT4 KV、不同 block size 或不同 scale 策略没有收益；
- `TRITON_SM75_BF16_DOT_AS_F16`、Triton launch tile、num-warps、num-stages 不会影响结果；
- GDN/Mamba/MTP 或 TP all-reduce 不是主瓶颈；
- 相同模型的 W4A16 版本、较小模型的单卡实例、不同 drafter 不能带来更大的端到端收益。

### 0.2 本次调研时的可达性限制

本机于 2026-09-06 对 `<user>@<server>:22` 的无密钥只读 SSH 得到 `Permission denied (publickey,password)`；本机到公网 8443 网关也无法建连。因此本文件把回退报告中 **2026-09-06 18:41:09 Asia/Shanghai** 的 `/health=200`、FlashInfer、标准 `triton==3.7.1` 视为最后审计快照，而不是“此刻已实时复核”的状态。开始任何试验前必须重新冻结真实 import、ExecStart、GPU 拓扑和 metrics。

### 0.3 三条不可省略的规则

1. **单变量。** 不能同时换 compiler、attention、KV dtype、模型和 MTP；否则结果无法归因。
2. **一时刻只加载一个 TP=2 实例。** 当前完整 27B 服务占用两张卡，所谓 8001 canary 需要先停 8000，或换成能单卡容纳的候选模型；不能双实例假装并行 A/B。
3. **先功能和数值、后速度。** 所有候选必须通过 `/health`、`/v1/models`、streaming、thinking、XML tool call、长前缀命中、确定性请求和 30 分钟稳定性，才进入性能比较。

## 1. 先冻结基线 B0

以下不是新的优化，而是所有比较的共同基准。若 B0 不可复现，后续数据无效。

| 项目 | B0 应记录的实际值 | 原因 |
|---|---|---|
| 运行时身份 | `python -c` 的 `vllm.__file__`、`triton.__file__`、版本、`pip check` | 避免源码树补丁与 site-packages 混用 |
| 服务配置 | `systemctl cat/show` 与真实 PID 完整 cmdline | unit 文件可能不是实际运行参数 |
| 模型身份 | 目录清单、每个 shard/`config.json`/tokenizer/template SHA256 | 不把不同 checkpoint 误比为 kernel 差异 |
| GPU | UUID、PCI Bus、NVLink `NV2`、温度、功耗、SM/MEM 时钟、driver | 防止热降频或拓扑改变伪造回退 |
| 后端日志 | FlashInfer attention、FlashQLA legacy GDN、FP8/Marlin、FP8 KV、MTP、CUDA graph | 确认预计路径真的命中 |
| 压测输入 | 2.8K、5.6K、8.5K、19.8K、59.2K token 固定语料；128 和 512 completion；冷/热前缀各一组 | 将 TTFT、prefill、decode、prefix cache 分开 |
| 采样 | 温度 0、固定 seed 的确定性组；当前生产采样组 | 前者看正确性，后者看真实体验 |

每个点至少 5 次暖机后重复；保留每一次的原始 SSE 时间戳、prompt/completion token、TTFT、prefill tok/s、decode tok/s、总耗时、GPU memory、MTP draft/accepted/position acceptance、prefix hit、错误和日志。先使用中位数；需要 p95 时每点至少 20 次。

## 2. 第一优先：把 Triton-Turing 的影响拆开

这组是最重要的，因为此前测试同时改变了编译器和 attention backend。

| ID | 编译器 | attention | KV writer / KV dtype | 要回答的问题 | 预期状态 |
|---|---|---|---|---|---|
| B0 | `triton==3.7.1` | FlashInfer | 当前原生 FP8 E4M3 | 当前基线 | 必做 |
| C1 | triton-turing | **仍用 FlashInfer** | 原生 FP8 E4M3 | fork 是否通过其它 Triton kernel（Mamba/GDN/采样/辅助 op）带来收益，而非只测试 TRITON_ATTN | 必做 |
| C2 | 标准 Triton | TRITON_ATTN | FP16 KV | 纯 backend 在没有 FP8 writer 门槛时的速度/正确性 | 必做 |
| C3 | triton-turing | TRITON_ATTN | FP16 KV | 软件流水线对真正 Triton paged prefill/decode 的净贡献 | 必做 |
| C4 | 标准 Triton | TRITON_ATTN | 原生 FP8 writer + E4M3 | 原生 writer 与 attention backend 的组合是否单独可行 | 条件必做 |
| C5 | triton-turing | TRITON_ATTN | 原生 FP8 writer + E4M3 | 已做过一次，但要在固定 B0 与 5 次重复下重测 | 必做 |
| C6 | triton-turing | TRITON_ATTN | Triton 自定义 FP8 quantize/store + E4M3 | 真正完成 Turing paged FP8 链路是否有价值 | 研发项 |
| C7 | triton-turing | FlashInfer | FP16 KV | compiler 对其它 Triton op 的收益是否被 FP8 转换掩盖 | 条件必做 |

### 2.1 C1 是此前漏掉的关键对照

`triton-turing` 是编译器替换，不只影响 `triton_attn.py`。所以 C1 必须保留已验证的 FlashInfer attention，**只**替换 Triton。若 C1 比 B0 快，而 C5 慢，根因就不是“triton-turing 无效”，而是 `TRITON_ATTN` 或 FP8 paged-KV 路径拖慢了整体。

### 2.2 C2/C3 要使用 FP16 KV 的原因

Turing SM75 没有原生 FP8 Tensor Core。FP8 cache 可以明显节省容量，但其读写/反量化可能成为 Triton path 的额外成本。把 KV 暂时改为 FP16 会牺牲可承载长度，却能回答“软件流水线对 attention 本身是否有效”。

在固定 4 GiB KV 预算下，FP16 的可缓存 token 上限大约是 FP8 的一半；这组测试把最大上下文先限制为 B0 实际可承载长度的一半以内，例如 64K/80K，而不是拿 180K OOM 与 B0 对比。

### 2.3 C6 不是“解除 capability gate”

要让 `TRITON_ATTN + FP8 KV` 成为可维护产品，不能只删除 vLLM 的 SM89 gate。真正的 C6 必须实现并验证：

1. 与 vLLM 当前 paged block layout、slot mapping、K/V scale 语义完全一致的 SM75 FP8 quantize-and-store；
2. FP8 E4M3 的 software encode/decode、NaN/Inf 处理和每层/每头 scale；
3. paged prefill、paged decode、chunked prefill、prefix-cache reuse、MTP draft/target cache 的一致性；
4. 与 vLLM 原生 writer 的逐 token 数据比对（FP8 bit pattern 或允许误差内的反量化值）；
5. 以原生 writer 为正确性 oracle，而不是只看 HTTP 200。

如果 C3 相对 C2 都不能提速，C6 不值得继续开发；如果 C3 明显提速而 C5 慢，C6 才有研发价值。

### 2.4 Triton-Turing 内核调参矩阵

若 C3 显示潜力，再只对热点 kernel 做 autotune。不要先改全局 vLLM 逻辑。

| 参数 | 候选 | 观测 | 说明 |
|---|---|---|---|
| `TRITON_SM75_BF16_DOT_AS_F16` | 0 / 1 | JIT diff、数值、tok/s | 当前 `--dtype half` 未必命中 BF16 dot；只有日志/IR 显示命中才保留 |
| `num_warps` | 2 / 4 / 8 | occupancy、register spill、kernel time | 每种 head_dim、block size 分开调 |
| `num_stages` | 1 / 2 / 3 / 4 | shared memory、stall、kernel time | 软件流水线不等于 stage 越大越快 |
| Q tile / KV tile | 16 / 32 / 64 与 32 / 64 / 128 | prefill kernel 时间、SM occupancy | 必须受 64 KiB shared-memory 和实际 head_dim 约束 |
| cache block size | 8 / 16 / 32（以 `--help` 可用项为准） | fragmentation、prefix 命中、decode | block size 会同时影响 paged attention 与缓存管理 |
| compile cache | 独立目录、预热后固定 | 首请求与稳态分离 | 编译时间绝不能计入推理慢 |

最先运行 `nsys` 或 torch.profiler 得到每个候选的 attention、GEMM、GDN、MTP、NCCL 和 CPU gap 占比。只优化累计占比足以影响端到端结果的 kernel；例如 attention 只占总 TTFT 20%，即使它快 30%，端到端理论上限也仅约 6%。

## 3. KV cache：容量、质量和速度分开测试

`--kv-cache-memory-bytes=4G` 存在时，vLLM 会忽略 `--gpu-memory-utilization` 对 KV 大小的推导。因此把 0.93 改为 0.95 不会自动增加当前 KV；只有改/删 4G 的显式上限才会改变它。

| ID | KV 格式 | 目标 | 必测风险 | 对 Triton-Turing 的意义 |
|---|---|---|---|---|
| K0 | `fp8_e4m3` | B0 长上下文/容量基线 | 长上下文精度、SM75 writer | 必须保留 |
| K1 | `float16` | 消除量化与转码，测纯 attention | token 容量约减半 | C2/C3 关键对照 |
| K2 | `fp8_e5m2` | 动态范围与精度/速度权衡 | 需模型/后端实际支持、长文质量 | 独立 A/B |
| K3 | `int8_per_token_head` | 更可控的压缩/量化尺度 | exact version、backend、精度 | 条件测试 |
| K4 | `int4_per_token_head` | 最大 KV 容量 | Turing 上解码反量化很可能吞掉收益 | 条件测试 |
| K5 | TurboQuant K8V4 / K3V4 | K/V 非对称压缩 | 仅 exact vLLM `--help` 出现且 SM75 可启动才测 | 条件测试 |
| K6 | 层级跳过量化 | 将最敏感 attention 类型保留 FP16，其余量化 | 先从模型 config 获得 layer/type 名称 | 精度优先候选 |
| K7 | 增加 KV bytes：4.5 / 5 / 5.5 GiB | 容量/并发，而非单请求 tok/s | CUDA graph、峰值激活、OOM | 长上下文候选 |

每个 Kx 都必须分别测：最大可承载长度、4K/20K/60K TTFT、512-token decode、prefix 冷/热、数学/代码/工具调用质量、MTP acceptance。不要把“能装更长上下文”误报为“每 token 更快”。

### 3.1 混合模型的 cache 变量

当前模型有 GDN/Mamba 类状态路径，除了 attention KV，还应在 `--help` 与 model config 支持时独立测试：

- `--mamba-cache-dtype auto/float16/float32`；
- `--mamba-ssm-cache-dtype auto/float16/float32`；
- `--mamba-cache-mode align/all/none`；
- `--mamba-block-size 8/16/32/64`（必须满足 causal-conv 对齐约束）；
- `--prefix-match-unit` 的实际 block 因子；
- `--use-replayssm` 与 `--replayssm-buffer-len 8/16/32/64`。

`use-replayssm` 只能用于标准、非 speculative decode，因此应单列为 **MTP=off** 对照；不能和当前 MTP=3 同时打开后把启动失败称为性能结论。

## 4. Attention、GDN 与 Mamba backend 的全覆盖

| ID | 全 attention | GDN/SSM | 适合回答的问题 | 前置条件 |
|---|---|---|---|---|
| A0 | FlashInfer | FlashQLA legacy | 当前工作基线 | B0 |
| A1 | FlashInfer | triton-turing 编译后的 Triton/Mamba 路径 | compiler 是否只对混合层有益 | C1 |
| A2 | TRITON_ATTN | 当前 GDN | Triton paged attention 本体 | C2/C3 |
| A3 | TRITON_ATTN | Triton-Turing 特化 GDN kernel | 是否应把研发放在 GDN 而非 full attention | profile 显示 GDN 是热点 |
| A4 | FlashInfer | FlashQLA legacy FP16/FP32 内核变种 | GDN 数值精度和吞吐取舍 | 现有 FlashQLA 可切换时 |
| A5 | vLLM `--mamba-backend` 的各可用项 | 固定 FlashInfer | Mamba backend 而非 attention 是否为长文本瓶颈 | 只测试 `--help` 显示的枚举 |

注意：`TRITON_ATTN`、FlashInfer、FlashAttention、SDPA、FlexAttention 等并不是所有 SM75 + FP8 KV + hybrid model 组合都支持。每个 backend 的“可启动”只是第一关；必须从日志确认没有静默 fallback，且 profile 中真的出现预期 kernel。

## 5. 权重量化、模型大小与并行策略

这是最可能产生量级差异的一层，但也是质量/成本边界最大的层。

### 5.1 不换模型：同一 27B 的量化与并行矩阵

| ID | 权重/并行 | 目的 | 关键判断 |
|---|---|---|---|
| Q0 | 当前 FP8 + TP2 | 基线 | 确认实际 GEMM 路径是预期 Marlin/FP8，而不是 fallback |
| Q1 | 同 checkpoint 的 W4A16 AWQ + TP2 | 减少 weight 带宽、释放显存给 KV/并发 | W4 解量化可能反而慢；用实际 tok/s 决定 |
| Q2 | 同 checkpoint 的 GPTQ/Marlin W4A16 + TP2 | 比较 AWQ 与 GPTQ pack/kernel | 核对 group size、zero point、Marlin 日志 |
| Q3 | W8A8 / INT8 激活量化（若 checkpoint 和 SM75 path 支持） | 利用 Turing INT8 tensor core 的可能性 | 必须做强质量回归和 kernel 实测 |
| Q4 | Q1/Q2 + FP16 KV | 分离 weight 格式与 FP8 KV 转码 | 最大长度减半，不能与 Q0 180K 直接比 |
| Q5 | Q1/Q2 + FP8/INT8/INT4 KV | 用释放的显存提高上下文或并发 | 分别报告容量与速度 |
| Q6 | W4A16 + TP1（单卡） | 去除每层 TP all-reduce | 可能因整模型单卡计算变慢或不装；只看实测 |
| Q7 | W4A16 + PP2（若模型/版本支持） | 用 pipeline 替代频繁 all-reduce | 单请求延迟常不占优，主要测吞吐 |

以前下载记录中出现过 `Qwen3.8-27B-AWQ-INT4`、带 MTP 的 AWQ checkpoint 和 Qwopus AWQ 目录；是否完整、是否与当前 template/工具调用兼容、是否有正确 MTP head，都必须在目标机通过 `config.json`、权重清单和真实加载核验，不能以下载计划代替事实。

### 5.2 允许换模型时的真正上限方案

| 方案 | 可能收益来源 | 代价/要做的验收 |
|---|---|---|
| 14B 级同能力模型单卡 × 2 副本 | 每请求去掉 TP 通信；两张卡可承载两个独立请求 | 质量、长上下文、tool/thinking、模板兼容性必须与 27B A/B |
| 8B/14B + 原生 MTP | 更少 target FLOP + 更高 draft acceptance 的机会 | 不能只看短题；要测代码、长文、中文工具调用 |
| 27B W4 单卡 | 减少通信，保留 27B 能力 | 22GB 下模型、KV、graph 的实际余量可能不足；只作为可启动的候选 |
| 27B TP2 + 外部 draft/EAGLE/DFlash 等 | 更高 speculative acceptance | draft 额外显存、同步和混合模型状态可能抵消收益 |
| 新 GPU 架构（支持原生 FP8/BF16/更强异步 copy） | 消除 SM75 的硬件数据类型与流水线限制 | 需重新做量化/attention/TP 拓扑基准；不是参数调优 |

如果“最大化加速”允许改变服务能力，**更小模型的单卡副本**和**更高 acceptance 的 draft**比单独优化一个 attention kernel 更可能改变总吞吐。但它们绝不应未经业务题集质量 gate 直接替换 27B。

## 6. Speculative decoding：必须完整 sweep，而非固定 MTP=3

当前 MTP=3 已经是有价值的基线，但最优 `num_speculative_tokens` 取决于 acceptance、TP 同步、KV copy、采样和输出长度。试验不应预设 3、5 或任意数字一定最好。

| ID | drafter | `num_speculative_tokens` | 要记录 | 保留门槛 |
|---|---|---:|---|---|
| S0 | off | 0 | 原生 decode tok/s | 诊断基线 |
| S1-S5 | 当前 native MTP | 1 / 2 / 3 / 4 / 5 | 位置 0..n acceptance、draft/accepted、decode tok/s、KV/graph | 端到端 decode 中位数明确提升且 p95 不恶化 |
| S6 | 当前 MTP + Turing compiler | 每个 C1/C3 胜出的 n | 与 S1-S5 同上 | 判断 fork 是否影响 draft 路径 |
| S7 | N-gram / prompt lookup | 只对重复模板/重复代码前缀 | lookup hit、CPU 开销、decode | 仅命中型业务保留 |
| S8 | 外部 draft / EAGLE / DFlash 等 | 1..模型配置上限 | draft 显存、acceptance、同步、质量 | 只在 native MTP 饱和后评估 |

每个 Sx 都要对温度 0、当前生产温度和 tool-call 组分开报告。只看 `accepted/draft` 的总比值不够；必须看每个 speculative position 是否在后段坍缩。若某配置只提高 128-token短输出、却降低 512/2048-token输出或长上下文稳定性，不能算最终加速。

## 7. 调度、批处理、CUDA Graph 与 API 开销

当前 `max-num-seqs=1`、`max-num-batched-tokens=4096`、MTP3、graph size `[4]` 是**长上下文单请求**画像下的合理起点，不是全业务的全局最优。应拆成两个服务画像，而不是强行找一个参数覆盖所有情况。

### 7.1 L（交互/长上下文）画像

| 变量 | sweep | 目标 |
|---|---|---|
| `max-num-seqs` | 1（固定） | 保证长请求 TTFT |
| `max-num-batched-tokens` | 2048 / 4096 / 6144 / 8192 | 测 chunk 大小与峰值显存 |
| `max-num-scheduled-tokens` | 自动 / 小于 batched 的安全值 | speculative 追加 token 下的稳定性 |
| chunked prefill | on / off | 长 prompt TTFT、decode 插队 |
| CUDA graph | eager / piecewise；按真实 decode shape 捕获 | 不将无命中 size 加入 capture |
| streaming interval | 1 / 2 / 4 / 8 | API CPU/网络开销与首 token 平滑度 |

此前 `[4,8]` 对 `max-num-seqs=1 + MTP3` 无收益的结果有效，但只适用于那个 shape。只要 S sweep、并发或模型改变，capture sizes 必须从启动日志和实际 batch shape 重新确定。

### 7.2 T（多请求吞吐）画像

| 变量 | sweep | 目标 |
|---|---|---|
| `max-num-seqs` | 2 / 4 / 8 | aggregate tok/s、排队延迟、preemption |
| `max-num-batched-tokens` | 4096 / 8192 / 16384 / 32768 | GPU 利用率与交互 TTFT 取舍 |
| `max-num-partial-prefills`、long-prefill threshold | exact `--help` 可用值 | 防止长 prefill 独占或过度切片 |
| async scheduling | off / on | 消除 GPU 空隙；同时验证 custom MTP/GDN 兼容 |
| scheduling policy | FCFS / priority | 不改变纯 kernel 速度，优化 p95 和用户体验 |
| KV watermark / reserve-full-isl | 默认 / 受控值 | 降低 cache thrash 与 preemption |

吞吐画像须把最长上下文下调为一个业务可接受档位，例如 32K/64K，再测 1/2/4 并发。拿“180K × 4 并发 OOM”否定吞吐调度没有信息价值。

### 7.3 编译与图捕获

`--compilation-config` 应测试：eager（诊断）、当前 PIECEWISE、可用的 compile mode、按形状的 capture sizes、warm-up 数、cache 目录。每种都要把 JIT/graph 捕获时间与稳态请求分开。

可进一步测试：

- `torch.compile` 的不同 mode（以当前 vLLM `--help`/schema 可接受项为准）；
- 单独缓存 Turing 与标准 Triton 的 compile cache，绝不共用；
- `jit-monitor-mode=warn/error` 在预热后发现新的动态 shape；
- 多模态仅在真实有图像业务时才测 encoder compile/cudagraph；文本 API 不应为此消耗额外 graph/KV 空间。

## 8. Prefix cache、输入与输出层的加速

| 项目 | 变量 | 可能改善 | 风险/边界 |
|---|---|---|---|
| prefix caching | off / on、冷/热 identical prefix | 第二次 TTFT/prefill 大幅改善 | 不能用累计 metrics 代替本次命中 |
| hash | `sha256` / `xxhash`（仅隔离可信流量） | CPU hash 开销 | 非加密 hash 有 collision 和多租户数据风险 |
| prefix match unit | block 因子内更细粒度 | 模板微变时提高命中 | 需兼容 physical block 和 hybrid cache |
| chat template | 固定/压缩工具 schema/短系统提示 | 减少 prefill token | 业务协议变更，非模型加速 |
| thinking budget | 0 / 小/当前 | 降低总响应时间与 token 成本 | 改变功能与质量，不提高单 token kernel 速度 |
| `stream_interval` | 1 / 2 / 4 / 8 | 低 CPU/网络开销，可能提高总 decode | 较大值降低逐 token 流畅性 |
| tokenization | 线程池/CPU affinity（若 CLI 支持） | 超短 prompt TTFT | 长 prefill/GPU bound 时通常不显著 |

## 9. TP/NVLink、CPU 与运行时层

这些不是“只换参数就必然更快”，但会决定 SM75 双卡的上限。

| 层 | 候选 | 测量/验收 |
|---|---|---|
| TP collective | 默认 PYNCCL/NCCL；Ring/Tree；NVLink P2P 参数 | profile 中 all-reduce 时间、错误、两卡拓扑不变 |
| GPU 时钟 | 持久模式、温度/功耗、稳定时钟 | 性能测试全程记录并排除 thermal throttling |
| CPU | governor performance、OMP 4/8/12/16、CPU affinity | 只用端到端 CPU gap/TTFT 判断，不凭 CPU 占用猜测 |
| NCCL/PCIe | 验证 NV2、P2P 可用、无退化为 host staging | `nvidia-smi topo`、NCCL 日志和带宽微基准 |
| FlashInfer | 当前锁定版与一个候选版 | 必须保留 SM75+FP8 KV patch，启动/数值/性能均通过才升级 |
| FlashQLA | legacy kernel 的 FP32/FP16 或 tile 优化 | profile 证明 GDN 有显著占比才投入 |
| CUDA/Torch/Triton | 锁定当前可用组合；新环境做 ABI gate | 不在生产 venv 上 `pip -U` 或覆盖安装 |

## 10. 推荐的实际执行顺序（完整，但不预设结果）

1. **P0：B0 冻结和 5 次完整复测。** 若无法重现，先解决运行时身份、patch、温度、NVLink 或输入口径。
2. **P1：C1。** 只用 triton-turing，attention 仍是 FlashInfer。这是此前最重要的缺口。
3. **P2：C2/C3。** FP16 KV 下分离标准 Triton 与 turing 的 attention 性能；最大上下文降到不会挤爆 KV 的档位。
4. **P3：C4/C5。** 回到 FP8 E4M3、原生 writer，重测稳定 TTFT/prefill/decode，不再用单次历史数据决策。
5. **P4：profile 决策。** C3 有正收益才进入 C6 自定义 FP8 writer；否则把研发投入转向 profile 中最大的 GEMM/GDN/MTP/collective 项。
6. **P5：K、S、L/T 矩阵。** 一个维度一个维度 sweep，先本模型，后新量化/新 drafter。
7. **P6：Q 矩阵。** 同 27B 的 AWQ/GPTQ/FP8，再决定是否尝试单卡或小模型双副本。
8. **P7：2 小时混合负载与三次重启。** 只有赢得中位数、p95、正确性和稳定性的候选才准生产切换。

## 11. 每个候选的 Go/No-Go 表

| 维度 | 必填指标 | 建议保留线 |
|---|---|---|
| 可用性 | import、启动、HTTP、日志后端 | 零 fallback、零未知 patch、零 kernel error |
| 正确性 | temp=0 token 序列、小数值对照、tool/thinking/XML、长文 | 与 B0 一致或事先定义且可接受的差异 |
| 长上下文 | 4K/20K/60K/目标上限 | 无 OOM、无异常 cache/preemption |
| 速度 | TTFT、prefill、decode、E2E，中位数+p95 | 至少一个目标指标明确赢，且关键 p95 不恶化 >10% |
| MTP | acceptance 分位置、draft/accepted | 不低于 B0 的 95%，不静默退化为 no-spec |
| 容量 | 可用 KV tokens、实际并发 | 与该候选声明的工作画像一致 |
| 稳定性 | 30 分钟 smoke / 2 小时混合流量 | 无 Xid、OOM、worker restart、memory leak |
| 可回滚 | 独立 venv、wheel/hash、unit 副本、原 B0 可启动 | 回退命令和验证已演练 |

## 12. 本轮最可能遗漏、最值得先试的项目

按信息价值排序，而不是按我对结果的主观判断：

1. **C1：triton-turing + FlashInfer，不切到 TRITON_ATTN。** 这是把 compiler 与 attention backend 解耦的最小实验。
2. **C2/C3：FP16 KV 的标准 Triton vs turing。** 直接排除 FP8 writer/反量化对结果的干扰。
3. **S0-S5：MTP 0..5 全 sweep。** 最可能影响 decode 的实际端到端变量。
4. **K0-K6：KV dtype 与混合 cache。** 分别报告容量、长文质量和速度；不要只看启动成功。
5. **A3/A4：GDN profile 后的 kernel 优化。** 当前是 hybrid 模型，attention 未必就是最大热点。
6. **Q1/Q2：同 27B W4A16 AWQ/GPTQ Marlin。** 可验证权重带宽、显存和 TP 的真实权衡。
7. **L/T 双画像。** 长上下文单请求和多用户吞吐分开优化；这是参数层面最容易被混淆的目标。
8. **单卡小模型或单卡 W4 27B。** 允许改模型时，去除 TP 通信与双副本的上限值得用真实业务题集验证。

## 13. 资料口径

- 本地实测与回退依据：`tasks/reports/2026-09-06-triton-turing-integration-rollback-summary.md`。
- vLLM v0.28.0 serve CLI：KV cache、scheduler、async scheduling、compilation、Mamba/ReplaySSM、speculative config 的可用参数以目标机 `--help` 和安装源码为最终准则；不同 0.28 build 可能并不具备文档中的所有项。
- triton-turing：仅作为独立环境中的编译器候选。其微基准宣称与本机端到端 Qwen/TP/FP8 KV 组合不是同一工作负载，不能直接外推。

