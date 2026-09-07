# Triton-Turing 集成尝试与回退总结

日期：2026-09-06  
范围：Linux 双 RTX 2080 Ti（SM75）上的 vLLM 0.28.0、8000 服务与 FP8 KV cache。

## 结论

本次 Triton-Turing 接入没有取得优于原 FlashInfer 基线的实测性能，因此已完整回退活动集成。8000 已恢复为原 systemd 服务、FlashInfer attention backend、FP8 KV cache 和标准 Triton 3.7.1。

## 尝试内容与结果

1. 编译并安装 `Chennesxu/triton-turing`（commit `82007a85`），以替换标准 Triton 3.7.1。
2. 强制 vLLM 0.28.0 的 `FLASH_ATTN` backend 失败：vLLM 的 CUDA FA2 backend 在 SM75 被 capability gate 拒绝，且 `fp8_e4m3` KV cache 需要更高版本的 FA backend/架构。
3. 强制 `TRITON_ATTN` 时，原始失败点是其 Triton cache writer 不能在 SM75 写入 FP8 KV cache。
4. 验证性替换：让 `TRITON_ATTN` 复用 vLLM 原生 `reshape_and_cache_flash` 写 FP8 cache，保留 Triton paged attention。该版本在原 180K 上下文、4 GiB FP8 KV cache、TP2、MTP 等参数下成功启动，HTTP 与真实生成请求通过。
5. 该验证性版本未提速，因而未保留。

## TTFT 初测

同一流式脚本、128 completion tokens、每个规模 1 次：

| 目标 words | FlashInfer 历史 TTFT | Triton-Turing 验证版 TTFT | 结论 |
|---:|---:|---:|---|
| 2,700 | 2.62 s | 2.645 s | 基本持平，略慢 |
| 5,400 | 4.53 s | 4.648 s | 慢约 2.6% |
| 8,100 | 6.51 s | 6.837 s | 慢约 5.0% |
| 19,000 | 15.23 s | 16.951 s | 慢约 11.3% |

原始测试 JSON：`/home/<user>/bench-triton-turing-nativefp8-20260906.json`。

## 已完成回退

- `triton`：恢复为 `3.7.1`，导入路径为 `/home/<user>/vllm-env-0280-qwopus/lib/python3.12/site-packages/triton/__init__.py`。
- `vllm/v1/attention/backends/triton_attn.py`：恢复为仓库 `HEAD` 版本；本次针对 FP8 writer 的改动与 SM75 放宽逻辑均不再存在。
- `qwen-vllm-qwopus.service`：恢复为 `enabled` 且 `active`。
- 8000：`/health` 于 2026-09-06 18:41:09（Asia/Shanghai）返回 HTTP 200；服务日志确认使用 `FLASHINFER` attention backend。
- 原服务启动参数保持原配置：FP8 权重量化、`fp8_e4m3` KV cache、180,000 上下文、4 GiB KV cache、TP2、MTP3、prefix cache、chunked prefill 和 FlashQLA legacy。

## 保留但未激活的材料

- `/home/<user>/triton-turing-src/`：源码与构建产物保留，未安装到 vLLM 虚拟环境，也未被服务导入。
- `/home/<user>/*triton*turing*.log`、`/home/<user>/bench-triton-turing-nativefp8-20260906.json`：保留作为审计证据。

如需未来再次评估，应单独实现并验证 paged FP8 KV-cache 兼容的 Turing FA2 prefill kernel；不能通过强制 vLLM 的 CUDA `FLASH_ATTN` backend 来达成。
