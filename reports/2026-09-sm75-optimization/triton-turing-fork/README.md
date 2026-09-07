# 支线四：Triton-Turing fork — 🔴 OPEN（未取得加速，移交社区）

## 状态

本战役未用该 fork 取得端到端加速（净效应 ≤0.4%），此支线标记 OPEN 移交。
这不是 fork 的质量问题，而是与当前服务路径的架构错配。

## 做过的事

1. 集成 fork（3.7.0+git82007a85，editable）进 vLLM 0.28.0，跑 P0/P1 矩阵，验证后回退。
2. 全量加速矩阵（P0~P3）+ 独立评估 + 终结评估。
3. FA2 d256 微基准：全 OOM。

## 为什么没有加速（接手者难度清单）

- **主 GEMM 不走 Triton**：W8A8 走 CUTLASS INT8 CUDA kernel，W4A16 走 Marlin CUDA kernel，FP8 走 Marlin FP8——fork 的 SM75 软件流水线只能影响 Triton 写的 kernel（GDN decode 递推、辅助小 kernel），占比很小。
- **FA2 d256 超出 64KB 共享内存硬限制**：autotune 配置空间 BM∈[64,128]、BN∈[32,64,128]，不含 BN=16；末次编译需 69,632 B > 65,536 B。
- **bf16 代理不命中**：服务跑 fp16，`TRITON_SM75_BF16_DOT_AS_F16` 只对 bf16 workload 生效。

## 可能的突破方向

- FA2 forward autotune 增加 BN=16（小 tile）配置，让 d256 在 64KB 内编过——但 tile 过小，MMA 效率预计很低。
- W4A8 / INT4 MMA 路径（SM75 m16n8k8 INT4 tensor core）——P3 方向，未启动。
- 自研 d256 定制 FA2（不依赖 fork 的 autotune 空间）。

## 来源与许可

- Fork：[Chennesxu/triton-turing](https://github.com/Chennesxu/triton-turing) commit `82007a85`，MIT。
- 本仓库仅引用其公开源码（报告中标注版本与 commit），未包含其代码；许可跟随上游。

## 文件

- triton-turing-integration-handoff.md — 集成交接记录
- triton-turing-integration-rollback-summary.md — 回退记录
- triton-turing-full-acceleration-matrix.md — 全量加速矩阵
- triton-turing-independent-assessment.md — 独立评估
- triton-turing-final-assessment.md — 终结评估
