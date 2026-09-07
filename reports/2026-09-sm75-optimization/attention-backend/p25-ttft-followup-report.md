# P2.5 追加评估:TTFT 口径验证与剩余手段验证(服务器实测)

> 日期:2026-09-07(09:00–09:45 维护窗口)
> 主机:<user>@<server>,2 × RTX 2080 Ti 22GB(SM75)+ NVLink
> vLLM 0.28.0 / torch 2.13.0+cu130 / **官方 Triton 3.7.1**(已核实回退)
> 参照:`inbox/2026-09-07-triton-turing-final-assessment.md`、`notes/2026-09-06-p0-p1-benchmark-results.md`
> 原始数据:`/home/<user>/benchmarks/2026-09-07-ttft-p25/`

---

## 0. 生产服务状态:已恢复并验证 ✅

- 实验全部停止,实验进程清理完毕,双卡显存归零后拉起 `qwen-vllm-qwopus.service`。
- 验证:`/health` OK、`/v1/models` 返回 qwen-local(max_model_len 180000)、真实生成请求正常输出(content:"好",29 completion tokens)。
- systemd 提示 unit 文件在磁盘上有变更,已 `daemon-reload` 处理路径正常启动;如遇异常可先 `sudo systemctl daemon-reload` 再 restart。

## 1. 核心结论(TL;DR)

1. **SDPA 路线正式排除。** GQA 正确口径下 torch 在 SM75 d256 没有任何快速 kernel(flash/efficient/cuDNN 全部 "No available kernel",只剩 math 兜底);把 K/V 物理展开到 24 头后 mem_efficient 实测只有 **8.4–8.8 TFLOPS,只有 FlashInfer(16.7 TF)的一半**。
2. **FlashInfer 确认为 SM75 d256 prefill 当前最优可用实现**(8K–59K 全程稳定 16.5–16.9 TFLOPS,per-GPU 形状 12Q/2KV 同样)。换 attention 后端这条路已穷尽:TRITON_ATTN(-148%)、fork FA2(64KB OOM)、SDPA(无 kernel/慢 2×)全部排除。
3. **B0 XL 档生产基线锚点:52.8–53.4s / 1121 tok/s。** 今日忠实复刻生产配置实测,与 2026-08-25 生产参考(53.02s / 1117 tok/s)完全一致。以此为锚,W1(W8A8)XL=35.62s 对应 ~35% 的 XL TTFT 改善,已由 W8A8 汇总重测确认(XL 52.98s → 35.76s)。
4. **XL TTFT 的构成(估算)**:E1 实测 per-GPU 59K 每层 full-attention 645ms × 16 层 ≈ **10.3s(约 19–29%)**,其余大头在 GEMM/GDN/NCCL。attention 已接近天花板,继续压 TTFT 的空间主要在权重轨道(W8A8)与 GEMM。
5. E2(bt=8192)S/M 档与 4096 持平(2.58s / 6.4s),L 档未完成即中止;后续请求出现一次 HTTP 500(未排查)。**8192 参数问题悬而未决,不算失败也不算通过。**

## 2. 实测数据

### 2.1 E1:FlashInfer vs SDPA,d256 长上下文(单卡,fp16,causal)

FlashInfer(BatchPrefillWithRaggedKVCacheWrapper,GQA):

| 形状 | N=8192 | 16384 | 32768 | 59240 |
|---|---|---|---|---|
| GQA 24Q/4KV | 24.5ms / 16.8TF | 98.6ms / 16.7TF | 390ms / 16.9TF | 1284ms / 16.8TF |
| GQA 12Q/2KV(TP2 per-GPU) | 12.5ms / 16.5TF | 49.6ms / 16.6TF | 199ms / 16.6TF | **645ms / 16.7TF** |

SDPA mem_efficient(K/V 展开 24 头 + kernel 合计):

| 形状 | N=8192 | 16384 | 32768 | 59240 |
|---|---|---|---|---|
| GQA 24Q(展开后) | 48.7ms / 8.5TF | 192ms / 8.6TF | 764ms / 8.6TF | 2489ms / 8.7TF |
| GQA 12Q(TP2,展开后) | 25.2ms / 8.2TF | 98.5ms / 8.4TF | 387ms / 8.5TF | 1255ms / 8.6TF |

SDPA GQA 原生(enable_gqa):**所有后端 "No available kernel"**(FLASH/EFFICIENT/CUDNN 均拒绝,MATH 兜底正确但极慢)。

口径说明:SDPA 只有在 24 个独立 KV 头(非 GQA 口径)下才能跑通 kernel;正确 GQA 口径下无 kernel、展开口径下慢 2×。

### 2.2 E2/A0:生产配置忠实复刻(官方 Triton 3.7.1,FP8,FP8 KV,bt=4096)

| 档位 | TTFT(run1/run2) | prefill tok/s | 2026-08-25 生产参考 |
|---|---|---|---|
| S 2.8K | 2.64 / 2.58s | 1080–1101 | 2.59s ✅ |
| M 8.4K | 6.41 / 6.39s | 1319–1322 | 6.45s ✅ |
| L 19.8K | 14.95 / 15.02s | 1316–1323 | 14.78s ✅ |
| XL 59.2K | **52.81 / 53.42s** | **1109–1122** | **53.02s ✅** |

四档全部与 08-25 生产参考吻合(±2%),证明 A0 是有效基线;B0 XL 行以本测为锚点(52.8–53.4s)。

### 2.3 E2/A1:bt=8192(中止前的不完整数据)

- S:2.64 / 2.58s(1101 tok/s);M:6.46 / 6.44s(1310 tok/s)——与 4096 持平。
- L run1 进行中被中止;bench 尾部出现一次 HTTP 500(server.log 未及排查)。
- 结论:8192 在 S/M 无收益;L/XL 未测,不下结论。

### 2.4 E3:profiler

vLLM 0.28 的 `/start_profile` 路由**仅在 `--profiler-config` 显式给出时挂载**,默认 404。本次未捕获 trace;XL attention 占比 19–29% 为 E1 kernel 时间外推值,非 trace 实测。

### 2.5 未解观察

A1 的 19.7K prefill 期间观察到 GPU0 持续 100% 而 GPU1 0%(连续 10 秒采样),请求本身速度正常。TP2 常态应为双卡交替满载;该现象与 500 错误是否相关未排查,列为遗留问题。

## 3. 修正后的"剩余 TTFT 手段"清单

| 优先级 | 手段 | 依据 | 预期 |
|---|---|---|---|
| 1 | **重测 W1(W8A8)XL 档 vs B0(53s 锚点)** | W1 当年 XL=35.62s 如可复现即是 **~35% XL TTFT 改善** | XL 53→~36s |
| 2 | **prefix caching 运营化**:固定系统提示词/常用长上下文预热 | 已开启,冷热差即全部 prefill 时间;XL 命中时 TTFT 53s→~1s,零代码改动 | 命中即数量级 |
| 3 | W4A16(AWQ)用于长输出场景 | decode 76 vs 37 tok/s(2×),显存 -30% | decode 翻倍,质量需评测 |
| 4 | bt=8192 L/XL 档重测(先排查 500) | vLLM 自身 warning;S/M 已证无收益 | 未知,≤8% |
| 5 | GEMM/NCCL profiler trace(`--profiler-config` 挂载后重做) | attention 仅占 19–29%,剩余大头未定位 | 定位下一个靶点 |
| — | ❌ SDPA / TRITON_ATTN / fork FA2 d256 / Triton-Turing fork 本体 | 本文 E1/E1b/E1c + P1 W6a + P2 | 全部排除 |

长期:SM75 无更快的 d256 attention kernel 可换,GEMM 侧 W4A8/Marlin 调优属于 P3 范畴且前置条件(占比证据)尚缺;硬件天花板内最现实的两大项就是 W8A8 重验证与 prefix caching 运营化。

## 4. 工件索引

```
服务器 /home/<user>/benchmarks/2026-09-07-ttft-p25/
  A-preflight/environment.txt   # 生产 unit/进程/env 快照(停服前)
  e1_d256_longctx_bench.py + E1_sdpa_microbench.json
  e1b_sdpa_gqa_fix.py     + E1b_sdpa_expanded.json
  e1c_backend_probe.py    (输出在会话日志)
  A0/benchmark.raw.json + server.log     # 生产复刻基线(完整)
  A1/bench_partial.log + server.log      # bt=8192(不完整,含 500)
```
