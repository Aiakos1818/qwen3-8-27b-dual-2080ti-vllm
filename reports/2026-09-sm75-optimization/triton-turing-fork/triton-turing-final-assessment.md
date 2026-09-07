# Triton-Turing Fork 评估终结报告

> 日期：2026-09-07
> 服务器：<user>@<server>
> 硬件：2 × RTX 2080 Ti 22GB（SM75/Turing）+ NVLink
> 模型：Qwen3.8-27B（VLM, FP8, GQA 24/4, head_dim=256, 16层全注意力 + 48层线性注意力）
> vLLM 版本：0.28.0
> 评估周期：2026-09-06 ~ 2026-09-07（P0 → P1 → P2）
> Fork 版本：3.7.0+git82007a85（Chennesxu/triton-turing）

---

## TL;DR

**Triton-Turing fork 对我们的 Qwen3.8-27B + 2× RTX 2080 Ti 服务没有实用价值。** 经 P0/P1/P2 三阶段系统测试，端到端收益 ≤0.4%，fork 的核心卖点——FA2 forward 加速——在 d=256 下因 64KB 共享内存限制完全不可用。已回退生产 venv 到官方 Triton 3.7.1。

---

## 1. 评估历程

### P0：基线与编译器替换（2 case）

| ID | 权重 | Triton | Attention | 结果 |
|----|------|--------|-----------|------|
| B0 | FP8 | 官方 3.7.1 | FlashInfer | 基线 |
| C1 | FP8 | fork 3.7.0+git | FlashInfer | ≤0.1% 偏差 → **持平** |

### P1：整数权重双轨道 + TRITON_ATTN（5 case）

| ID | 权重 | Triton | Attention | TTFT (XL) | Decode | Fork 收益 |
|----|------|--------|-----------|-----------|--------|-----------|
| W1 | W8A8 | fork editable | FlashInfer | 35.62s | 37.2 tok/s | — |
| W2 | W8A8 | fork+IN_TREE | FlashInfer | 35.65s | 37.2 tok/s | ≤0.4% → **持平** |
| W3 | W4A16 | fork editable | FlashInfer | 48.18s | 76.3 tok/s | — |
| W4 | W4A16 | fork+IN_TREE | FlashInfer | 48.00s | 78.3 tok/s | ≤0.4% → **持平** |
| W6a | W8A8 | fork+IN_TREE | TRITON_ATTN | 88.41s | ~27 tok/s | **-148%** 严重劣化 |

### P2：d256 FA2 微基准（3 provider）

| Provider | N=1024 | N=4096 | N=16384 | 状态 |
|----------|--------|--------|---------|------|
| Triton-Turing FA2 | OOM | OOM | OOM | ❌ 不可用 |
| FlashInfer | 0.56ms / 12 TF | 6.93ms / 15 TF | 97.5ms / 17 TF | ✅ 可用 |
| PyTorch SDPA | 0.20ms / 33 TF | 0.67ms / 153 TF | — | ✅ 最快 |

---

## 2. 核心发现

### 2.1 fork 端到端收益为零

三个维度对比全部持平：
- **C1 vs B0**（FP8 基线）：≤0.1%
- **W2 vs W1**（W8A8 INT8）：≤0.4%
- **W4 vs W3**（W4A16 AWQ）：≤0.4%

**原因**：fork 是 Triton 编译器的 fork，只影响 `@triton.jit` kernel。而我们的热路径几乎全部走 CUDA kernel：

| 组件 | 占比 | 实现 | fork 受益 |
|------|------|------|-----------|
| 全注意力 prefill+decode (16层) | 大 | FlashInfer CUDA | ❌ |
| GDN prefill (48层) | 大 | FlashQLA CUDA | ❌ |
| 线性 GEMM (64层) | 大 | cuBLAS / CUTLASS / Marlin CUDA | ❌ |
| GDN decode (48层×每token) | 小 | FLA Triton 递推 | ✅ 理论受益 |
| RoPE/RMSNorm/采样等 | 小 | Triton 小 kernel | ✅ 理论受益 |

fork 直接受益的 Triton kernel 在端到端推理中总占比太小，且 decode 已被显存带宽屋顶限制（~45 tok/s raw, MTP=3 放大至 ~100 tok/s），计算延迟优化空间有限。

### 2.2 FA2 在 d=256 完全不可用

fork README 的 FA2 基准只覆盖 d64/d128（+21~26% / +4~8% vs CUTLASS）。我们的模型全注意力层是 d=256，超出实测包络。

**实测结果**：`OutOfResources: shared memory, Required: 69632, Hardware limit: 65536`

根本原因：fork 的 FA2 forward autotune 配置空间 `BM∈[64,128] × BN∈[32,64,128]` 不含 BN=16（唯一可能 fit 64KB 的 tile）。所有配置编译出的 kernel 都超出 Turing 64KB/CTA 共享内存限制。

Pipeline dump 证实：所有配置要么"not pipelined"（pipeline 被关闭），要么直接 OOM。即使手动添加 BN=16 配置让编译通过，tile 过小会导致 MMA 效率极低。

### 2.3 TRITON_ATTN 在 SM75 不可用

W6a 实测：XL 档 TTFT 88.41s（FlashInfer 35.62s），慢 **148%**。vLLM 原生 Triton paged attention 是通用实现，在 SM75 长上下文上性能崩溃。

### 2.4 W8A8 vs W4A16 轨道对比

| 维度 | W8A8 (W1) | W4A16 (W3) | 胜出 |
|------|-----------|------------|------|
| TTFT S 档 | 2.08s | 2.29s | W8A8 |
| TTFT XL 档 | 35.62s | 48.18s | W8A8 (+35%) |
| Prefill tok/s (L) | 2037 | 1467 | W8A8 (+39%) |
| Decode tok/s (D) | 37.2 | 76.3 | **W4A16 (+105%)** |
| 显存/卡 | ~23GB | ~16GB | W4A16 |

**胜出轨道 = W8A8 (W1)**（按 TTFT 优先规则），但 W4A16 在 decode 速度和显存上有显著优势。

### 2.5 意外发现：FlashInfer d=256 SM75 性能低下

P2 微基准揭示：FlashInfer 在 SM75 d=256 上仅 11~17 TFLOPS，而 PyTorch SDPA (mem_efficient/xFormers) 达 33~305 TFLOPS——**快 2.9~18 倍**。这是 XL 档 TTFT 35s 的算子级瓶颈根源。FlashInfer 的 d=256 路径缺少 SM75 tile tuning 优化。

---

## 3. fork 技术评价

### 3.1 做了什么（技术上是真实的）

| 特性 | 技术评价 | 我们的场景是否命中 |
|------|----------|-------------------|
| SM75 软件流水线（无 cp.async） | ✅ Turing 首次实现，概念正确 | ❌ 热路径不在 Triton |
| Turing 专用 autotune | ✅ 针对 64KB 调优 | ❌ 不覆盖 d256 |
| INT4 MMA (m8n8k32) | ✅ 上游未实现，fork 填补 | ❌ 需要 W4A8 路线 |
| FA2 forward+backward (d64/d128) | ✅ 比手写 CUDA 快 4-26% | ❌ 我们的 d=256 不可用 |
| bf16→fp16 Tensor Core 代理 | ✅ 12-17× 加速 | ❌ 我们用 FP16 不命中 |

### 3.2 为什么对我们没有实用价值

**不是 fork 质量问题，是架构不匹配**：

1. **注意力形状不匹配**：Qwen3.8-27B 全注意力层 d=256 超出 fork 实测包络（d64/d128），64KB 共享内存放不下
2. **热路径不在 Triton**：主要 GEMM 走 cuBLAS/CUTLASS/Marlin（CUDA kernel），注意力走 FlashInfer（CUDA kernel），fork 只影响 Triton kernel
3. **decode 已贴带宽屋顶**：fork 优化计算延迟，但 decode 瓶颈是 616 GB/s 显存带宽，不是计算
4. **bf16 代理不命中**：我们用 `--dtype half`（FP16），fork 的 bf16→fp16 代理特性无法生效

### 3.3 README 准确性

fork README 的措辞是**准确的**——它没有声称测试过 d256，所有性能数字都明确标注了 head dim 64/128。Tutorial 源码中 `assert HEAD_DIM_K in {16, 32, 64, 128, 256}` 只在语法上放行 256，但 benchmark 循环 `for HEAD_DIM in [64, 128]` 从未测试 256。这不是欺骗，但"支持"和"可用"之间的差距需要在实际场景中验证——这正是我们做的。

---

## 4. 环境清理

### 4.1 B0 生产 venv 回退

**操作**：将 `vllm-env-0280-qwopus` 从 fork editable install 回退到官方 Triton 3.7.1。

```
# 回退前
triton 3.7.0+git82007a85
file: /home/<user>/triton-turing-src/python/triton/__init__.py

# 回退后
triton 3.7.1
file: /home/<user>/vllm-env-0280-qwopus/lib/python3.12/site-packages/triton/__init__.py
```

命令：
```bash
pip install --force-reinstall --no-deps triton==3.7.1
```

验证：生产服务正常启动，health=200，models API 正常响应。

### 4.2 Canary venv 保留

`vllm-env-0280-triton-turing-canary` 保留 fork 安装不变，供未来可能的 INT4 MMA / W4A8 路线实验使用。

### 4.3 源码保留

`/home/<user>/triton-turing-src/` 目录保留，不删除。

---

## 5. 下一步方向

### 5.1 可探索（不改 fork）

1. **PyTorch SDPA 作为 d256 prefill 替代**——算子级快 FlashInfer 3~18×，如果 XL 档 TTFT 从 35s 降至 ~5-10s，收益巨大。需验证 paged KV cache 兼容性
2. **FlashInfer d=256 SM75 tile tuning**——向 FlashInfer 社区反馈或自查是否有配置选项
3. **W4A16 在"短输入长输出"场景**——decode 2× 快于 W8A8，需质量评测确认 AWQ 对 thinking/tool call 的影响

### 5.2 不推荐

- ❌ 继续在 fork FA2 上调参 d=256——64KB 是硬限制
- ❌ fork FA2 backward——forward 都不可用
- ❌ P3 整数 W4A8——P2 已关闭，前置条件不满足
- ❌ TRITON_ATTN 在 SM75 长上下文——已验证劣化 148%

---

## 6. 文件索引

### 报告
- `tasks/reports/2026-09-06-p0-p1-benchmark-results.md` — P0/P1 完整基准数据
- `tasks/reports/2026-09-07-p2-d256-fa2-microbench.md` — P2 d256 微基准详细报告
- `tasks/reports/2026-09-07-triton-turing-final-assessment.md` — 本报告

### NAS Vault
- `inbox/2026-09-06-triton-turing-merged.md` — 初始调研合并文档
- `inbox/2026-09-06-qwen38-awq-triton-turing-sequential-execution-plan.md` — 执行计划
- `inbox/2026-09-07-p2-d256-fa2-microbench.md` — P2 报告副本
- `inbox/2026-09-07-triton-turing-final-assessment.md` — 本报告副本

### 服务器数据
```
/home/<user>/benchmarks/2026-09-06-awq-tt-fa2/
  B0/ C1/ W1/ W2/ W3/ W4/ W6a/     # P0/P1 基准 JSON + 日志
  P2_d256_microbench_v3.json        # P2 Triton FA2 (全失败)
  P2_d256_flashinfer_sdpa.json      # P2 FlashInfer + SDPA 数据
  run_fork_bench.py                  # fork tutorial 直接调用
  flashinfer_bench_v3.py             # FlashInfer + SDPA 基准
```

### fork 源码
```
/home/<user>/triton-turing-src/      # 保留，不删除
  commit: 82007a85
  remote: https://github.com/Chennesxu/triton-turing.git
```

---

## 7. 环境状态（最终）

| 组件 | B0 venv (qwopus) | Canary venv |
|------|------------------|-------------|
| Triton | **3.7.1** (官方 PyPI) | 3.7.0+git82007a85 (fork) |
| vLLM | 0.28.0 | 0.28.0 |
| FlashInfer | 0.6.16.post3 | 0.6.16.post3 |
| 服务状态 | ✅ 运行中 (port 8000) | 未运行 |
| fork 源码 | 已移除 editable install | 保留 editable install |
