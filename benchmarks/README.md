# 性能测试结果与复测方式

这里上传的是实际测试结果，不是推测速度。

## 最终配置的实测数据

文件 2026-08-25-real-streaming-context-ttft.json 是当前线上服务在 2026-08-25 重新运行的结果：约 2.8K、5.6K、8.4K 实际输入，各连续串行 3 次。文件 2026-08-25-real-streaming-context-ttft-20k-60k.json 补充约 19.8K 和 59.2K 实际输入，各连续串行 3 次。请求使用唯一合成英文上下文，避免 Prefix Cache 影响；输出固定 128 tokens，流式计时。

> **口径提醒**：下面这些 128-token 的 Decode 是**整窗平均**，不是稳态速度。MTP 的接受率在 prompt 刚结束
> 时最高、随后衰减，所以短窗口会偏高（实测同一请求：整窗 80.7 tok/s vs 尾部稳态 68.6 tok/s）。要稳态
> 数字请用脚本的 `decode_steady_tok_s`（见下）。

### 当前线上服务：上下文梯度重测

| 实际输入约 | 平均 TTFT | 平均 Prefill | 平均 Decode |
| --- | ---: | ---: | ---: |
| 2.84K tokens | 2.59 s | 1,099.8 tok/s | 97.3 tok/s |
| 5.64K tokens | 4.48 s | 1,260.2 tok/s | 94.3 tok/s |
| 8.45K tokens | 6.45 s | 1,311.2 tok/s | 101.1 tok/s |
| 19.77K tokens | 14.78 s | 1,337.6 tok/s | 101.3 tok/s |
| 59.24K tokens | 53.02 s | 1,117.3 tok/s | 84.4 tok/s |

TTFT / 首字时间的严格口径：从 HTTP 请求发出开始，到流式 SSE 第一次收到任一非空 reasoning_content、reasoning 或 content 字符为止。也就是说，模型开始输出 think 内的第一个思考字符就算首字，不等待最终答案正文。

Decode 是从该首个字符到流结束、按 API 返回的 completion_tokens 计算的平均速率。因此 decode 包含 reasoning 与 content 两部分输出。

上表的 Decode 是**整窗平均**。要**稳态**速度用 `decode_steady_tok_s`：脚本在读流的同时采样引擎自己的
`vllm:generation_tokens_total`，取 `--steady-from`（默认 128）之后那段尾部窗口算速率，并同时给出该窗口的
MTP 接受率（`steady_mtp_acceptance_pct`）与窗口大小（`steady_window_tokens`）。用引擎计数器而不是客户端
分块，是因为 MTP 每块带 1–4 个 token。该计数器是**服务级**的，所以稳态字段只在"本基准是唯一在跑的请求"
时才有意义（本仓库的 profile 都是 `--max-num-seqs 1`）。`--max-tokens` 默认已从 128 改为 512，好让稳态窗口
有足够长度；生成不足 `--steady-from` 时该字段为 `null`，并在 `steady_note` 里说明原因。

### 旧压测与 DSH 数据说明

旧的 2026-08-23 20K 合成压测不再作为首页代表速度；本次 2026-08-25 的 19.77K/59.24K 结果是同一真实流式首字脚本重新测得的上下文梯度。旧 P3/P4 或 2026-08-21 A/B 数字也不再进入速度基线，因为它们来自非流式请求脚本，TTFT 是按固定预填充速度反推的估算值，并非首个思考字符实测。

DSH 全任务平均首字是另一套真实业务任务集指标。没有该任务集本体时，不能用合成上下文测试替代，也不能声称这组结果就是 DSH 的 3.3s。

## 启动验收证据

最终服务启动日志记录：

| 项目 | 实测值 |
| --- | --- |
| 每张卡模型加载 | 14.96 GiB |
| 模型加载耗时 | 约 21.9–26.1 s / TP rank |
| GPU KV Cache 容量 | 216,562 tokens |
| 180K 请求最大并发 | 1.20x |
| GDN prefill | FlashQLA legacy SM70/SM75 kernel 已生效 |

## 如何复测

1. 严格使用 README 固定的 GPU、驱动、vLLM commit、补丁、FlashQLA commit、模板与启动参数。
2. 测试前确认无其他 GPU 任务；GPU0 接显示器时数值可能轻微波动。
3. 每档至少串行运行 3 次，使用唯一 prompt，避免 Prefix Cache 把 prefill 成绩虚高。
4. 测试输出用 `--max-tokens 512`（脚本默认值；128 只覆盖最快的那一段），并记录 prompt_tokens、TTFT、
   total latency、prefill tok/s、`decode_tok_s`（整窗平均）与 `decode_steady_tok_s`（`--steady-from` 之后的
   稳态，含 MTP 接受率）；TTFT 必须按首个 reasoning/content 字符记录。
5. 与上表比较时，允许约 ±10% 波动；超过这个范围先检查 flashqla_legacy、TP=2、NVLink=NV2、FP8 KV 和 MTP。
6. 服务开启鉴权时（本机 profile 会导出 `VLLM_API_KEY`）脚本必须带 key，否则一律 401：

~~~bash
python3 benchmarks/run_context_ttft.py --model qwen38-27b \
  --word-counts 2700 5400 8100 --runs 3 --max-tokens 512 --steady-from 128 \
  --api-key "$VLLM_API_KEY" --output out.json
~~~

   也可以只 `export VLLM_API_KEY=...`（脚本按 `--api-key` → `$VLLM_API_KEY` → `$OPENAI_API_KEY`
   顺序取，并在开始时打印用的是哪一个，不打印 key 本身）；遇到 401/403 立即退出（exit 2）并给出提示，
   不会把整个档位矩阵跑成一片失败。`--timeout` 可调单请求读超时（默认 900 s）。

不同模型权重、显示器占用、温度/功耗墙、显卡改装规格、PCIe/NVLink 状态、驱动与 CUDA 小版本都会改变速度。本仓库给出可比的验收区间，不承诺每台机器得到逐位相同数字。
