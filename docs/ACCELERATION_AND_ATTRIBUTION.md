# 加速组件、固定版本与引用

本仓库提供同硬件可复现的部署配置、补丁和模板，不包含模型权重。

本仓库有两条可复现路线：**本分支路线**（上游 vLLM `main` + SM75/Qwen3.8 移植，当前部署）
与**基础路线**（上游 vLLM `v0.27.1` + 工作树补丁），选择见 [PATCHING.md](PATCHING.md)。

| 组件 | 本分支路线（`2080ti_dual_qwen38-27B`） | 基础路线 | 用途 | 上游 / 引用 |
| --- | --- | --- | --- | --- |
| vLLM | `main` @ `0.26.1rc1.dev2278+g49f68ba24`（vLLM 侧分支 `2080ti_dual_qwen38-27B` @ `059727bfa`） | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` | API、TP、KV Cache、MTP、CUDA Graph | https://github.com/vllm-project/vllm |
| PyTorch | 2.13.0+cu130 | 同上 | CUDA 运行时与张量计算 | https://github.com/pytorch/pytorch |
| Transformers | 5.16.1 | 同上 | 模型与 Tokenizer 配置 | https://github.com/huggingface/transformers |
| FlashInfer | 0.6.18.post1 | 0.6.16.post3 | SM75 attention / decode 后端 | https://github.com/flashinfer-ai/flashinfer |
| FlashQLA | 3ab27d77d8ca01d7a4718903b726add1a8886c0e | 同上 | Qwen GDN/linear-attention 的 SM70/SM75 legacy prefill | https://github.com/weicj/FlashQLA-SM70-SM75 |
| NCCL | 2.29.7 | 同上 | 双卡 Tensor Parallel 通信 | https://github.com/NVIDIA/nccl |
| Triton-Turing fork | 3.7.0+git82007a85（仅引用，未打包） | 同上 | 2026-09 优化战役的 SM75 软件流水线 / FA2 实验（端到端 ≤0.4%，无加速，支线标记 OPEN） | https://github.com/Chennesxu/triton-turing（MIT） |

## 仓库内提供的本地改动

- patches/vllm-v0.27.1-sm75-qwen3.8.patch：**基础路线**的工作树补丁（基于 vLLM v0.27.1），涵盖 flashqla_legacy GDN prefill、Qwen3.5 MTP 兼容、reasoning 预算、SM75 FlashInfer/采样兼容和 GPU runner 调整。**本分支路线不套用它**——同样的 9 处改动直接落在上游 `main` 上，移植清单见 [upstream-branch.md](upstream-branch.md) §2。
- vLLM `2080ti_dual_qwen38-27B` @ `059727bfa`：让 SM75 的投机验证留在 FlashInfer native decode 路径，MTP 下保住 FULL cudagraph（单并发稳态解码 54.1 → 37.5 ms/步，见 [upstream-branch.md](upstream-branch.md) §6）。
- docs/patches/2026-09-18-fused-multi-step-draft-decode.patch：融合多步草稿解码的**已评估未采纳**方案（实测仅 +6%，不值得多带一个设备侧 kernel），留档备查，**不套用**。
- patches/flashqla-sm70-sm75-local.patch：基于固定 FlashQLA commit 的本地导出/SM legacy 调整。
- scripts/setup/apply_gdn_flashqla_legacy.py：当 git apply 因上游小版本差异无法套用时，用于补充 GDN legacy backend 的辅助脚本。

补丁来自已验证服务的工作树，不是 vLLM、FlashInfer 或 FlashQLA 的官方发布包。升级任一上游组件后必须重新验证。

- reports/2026-09-sm75-optimization/ 收录 2026-09 优化战役的全部报告与原始 JSON（W8A8 / MTP / attention / Triton-Turing fork / W4A16 / INT8 KV 六条支线）。其中引用的 Triton-Turing fork 仅按其公开源码（commit 82007a85）标注版本，本仓库未包含其代码，许可跟随上游（MIT）；FlashQLA-SM70-SM75 同为 MIT。

## Chat template

templates/qwen3.8-froggeric-v22.3.jinja 是当前实际使用的修复模板源文件，模板版本字段为 qwen3.8-froggeric-v22.3。

- 用途：Qwen3 消息格式、thinking 开关、XML/JSON 工具调用、多轮 tool response、图像/视频 token 拼装。
- 启动参数：`--chat-template templates/qwen3.8-froggeric-v22.3.jinja`；基础路线另外传
  `--chat-template-content-format string`（本分支的四个 profile 不传）。
- 模型权重、Tokenizer、上游模型许可证仍以模型发布方许可证为准。
