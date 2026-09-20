# 量化精度损失评测（logit 级，单变量归因）

日期：2026-09-20
范围：只做 **logit 一致性 / 分布距离**（不测任务型基准）。

## 0. 结论摘要

| 变量 | 短上下文（dense 8K） | 长上下文（256K 探针） | 结论 |
|---|---|---|---|
| **INT4 权重** | 4.3%–7.6% 位置改变，KL 0.023–0.050 | KL 0.0143 | 短上下文主导 |
| **YaRN rope**（非量化） | 1.9%–3.6%，KL 0.005–0.016 | **KL 0.0287** | **长上下文反超，可配置** |
| lm_head int8 | 0.4%–0.8%，KL ≤0.002 | KL 0.0003 | 很小 |
| **FP8 KV** | — | 18/18 探针全一致，KL 0.0004–0.0019 | **几乎无损** |

- 生产配置相对准无损基线的**总分布偏移**很小（dense KL 中位 0.001–0.014 nats），量级与文献中 W4 量化一致。
- **退化来源随上下文换位**：短上下文是权重，**长上下文是 YaRN**；**FP8 KV 在两者中都几乎无损**。
- 域差异：**代码最钝**（argmax 一致 95.4%、覆盖率 99.3%）、**中文最敏感**（91.7%、93.2%）。

## 1. 对比对象

| id | 模型 | 权重 | rope | KV | 上下文 | 内容 |
|---|---|---|---|---|---|---|
| **B0**（基线） | `Qwen3.8-27B-FP8` | FP8 e4m3 | default | fp8_e4m3 | 262144 | dense + probes |
| **W4** | `Qwen3.8-27B-AWQ-INT4` | AWQ INT4 | default | fp8_e4m3 | 262144 | dense + probes |
| **W4y** | `Qwen3.8-27B-AWQ-INT4-yarn512k` | AWQ INT4 | yarn factor=4 | fp8_e4m3 | 262144 | dense + probes |
| **H8f** | `…-yarn512k-head8bit` | AWQ INT4 + head int8 | yarn factor=4 | **float16** | 262144 | probes |
| **H8**（当前生产） | `…-yarn512k-head8bit` | AWQ INT4 + head int8 | yarn factor=4 | fp8_e4m3 | 262144 | dense + probes |

- 全部**无 MTP**、T=0、`max-num-seqs=1`、`--language-model-only`。
- **B0 不是"无量化"**：FP8 本身是量化。真正的 BF16 权重约 54 GB，双 2080Ti 共 44 GB 装不下，本机无法运行。故 B0 是 *quasi-lossless baseline*。
- 相邻两行只差一个变量，故 `B0→W4→W4y→H8` 构成单变量归因链；`H8f` 与 `H8` 只差 KV dtype。

## 2. 方法

- **Dense（逐位置）**：把三域语料切成 8192-token 段，用 `prompt_logprobs=100` 取每个位置的 top-100 分布。共 72 段 × 8192 = **590K 位置**。
- **长上下文探针**：32K / 128K / 262K token 的长上下文各 6 条，生成 1 token 并用 `logprobs=100` 取该位置分布（共 18 个探针）。
- 三域：英文散文（wikitext-2 test）、中文散文（wikimedia/wikipedia zh）、代码（vLLM 源码）。
- 指标：top-1 一致率、top-5 重合度、KL(B0‖H8)（top-100 交集重归一化）、top-100 覆盖率、真实 next-token 的 |Δlogprob|。
- k=100（实测 k=1000 慢 5 倍、精度收益递减）；覆盖率一并报告以判断截断偏差。

## 3. 结果

### Dense（590K 位置，每段 8191 有效位置，按域平均）

| 域 | top-1 一致率 | top-5 重合度 | KL 均值 | KL 中位 | 覆盖率(B0/H8) | \|Δlogprob\| 真实token |
|---|---:|---:|---:|---:|---:|---:|
| en（英文） | 92.23% | 89.26% | 0.0563 | 0.01110 | 0.9755 / 0.9761 | 0.176 |
| zh（中文） | 91.69% | 90.22% | 0.0272 | 0.01401 | 0.9316 / 0.9339 | 0.142 |
| code（代码） | 95.36% | 84.13% | 0.0499 | 0.00092 | 0.9935 / 0.9934 | 0.114 |

### 长上下文探针（每档 6 条）

| 上下文 | argmax 一致 | top-5 重合度 | KL | 覆盖率(B0) |
|---|---:|---:|---:|---:|
| 32K | 6/6 | 0.733 | 0.0118 | 0.9867 |
| 128K | 6/6 | 0.833 | 0.0084 | 0.9881 |
| **256K** | **5/6** | 0.767 | **0.0416** | 0.9412 |

## 4. 单变量归因（T2）

在 B0/H8 之外补跑 3 个配置，使每一步只动一个变量（全部 `--language-model-only`）：

| 对比 | 变量 | 域 | top-1 一致 | top-5 重合 | KL 均值 | KL 中位 | \|Δlogprob\| |
|---|---|---|---:|---:|---:|---:|---:|
| **B0 → W4** | INT4 权重 | en | 92.80% | 89.98% | 0.0501 | 0.00953 | 0.162 |
| | | zh | 92.37% | 91.02% | 0.0230 | 0.01164 | 0.130 |
| | | code | 95.68% | 85.32% | 0.0451 | 0.00076 | 0.107 |
| **W4 → W4y** | **YaRN rope** | en | 96.52% | 94.78% | 0.0163 | 0.00195 | 0.078 |
| | | zh | 96.42% | 95.64% | 0.0049 | 0.00232 | 0.059 |
| | | code | 98.10% | 91.83% | 0.0067 | 0.00019 | 0.044 |
| **W4y → H8** | lm_head int8 | en | 99.20% | 98.73% | 0.0017 | 0.00011 | 0.018 |
| | | zh | 99.22% | 99.05% | 0.0002 | 0.00011 | 0.012 |
| | | code | 99.56% | 97.86% | 0.0003 | 0.00001 | 0.010 |
| **H8f → H8** | **FP8 KV**（探针） | 32K | 6/6 | 1.000 | 0.0005 | — | — |
| | | 128K | 6/6 | 1.000 | 0.0004 | — | — |
| | | 256K | 6/6 | 0.967 | 0.0019 | — | — |

配置说明：
- `W4` = AWQ-INT4 + default rope + **fp8 KV**；`W4y` = AWQ-INT4-yarn512k + **fp8 KV**；`H8f` = AWQ-INT4-yarn512k-head8bit + **fp16 KV**。
- `H8f vs H8` 只差 KV dtype（fp16 vs fp8），其余完全相同；用长上下文探针比较（KV 效应只在长上下文显现）。

### 长上下文归因（探针，每档 6 条）

| 对比 | 变量 | 32K top-1 | 128K top-1 | 256K top-1 | 32K KL | 128K KL | 256K KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 → W4 | INT4 权重 | 1.000 | 0.833 | 1.000 | 0.0070 | 0.0099 | 0.0143 |
| W4 → W4y | **YaRN rope** | 1.000 | 1.000 | 0.833 | 0.0038 | 0.0050 | **0.0287** |
| W4y → H8 | lm_head int8 | 1.000 | 0.833 | 1.000 | 0.0002 | 0.0001 | 0.0003 |
| H8f → H8 | FP8 KV | 1.000 | 1.000 | 1.000 | 0.0005 | 0.0004 | 0.0019 |
| **B0 → H8** | **总计** | 1.000 | 1.000 | 0.833 | 0.0118 | 0.0084 | 0.0416 |

**关键发现：贡献随上下文长度换位。**
- **短上下文（32K/128K）**：INT4 权重占主导（KL 0.0070–0.0099），YaRN 次之（0.0038–0.0050）。
- **256K**：**YaRN 反超成为最大单项**（0.0287，约为权重的 2 倍）。
- 加和自洽：256K 下 权重 0.0143 + rope 0.0287 + head 0.0003 + KV 0.0019 ≈ 0.045，与总计 0.0416 接近。
- 因此 256K 那 1/6 的 argmax 翻转，**归因于 YaRN 而非 KV 或 head**。

### 归因结论

1. **INT4 权重在短上下文是最大来源**：dense（8K）改变约 4.3%–7.6% 位置的 argmax，KL 均值 0.023–0.050。
2. **YaRN rope 在长上下文反超为最大来源**：dense 下改变 1.9%–3.6% 位置（KL 0.005–0.016）；到 **256K 时 KL 0.0287，约为权重的 2 倍**。**YaRN 不是量化**，是上下文外推，且**可配置**。
3. **lm_head int8 很小**：改变约 0.4%–0.8% 位置，KL ≤0.002（长上下文探针同样 ≤0.0003）。
4. **FP8 KV 几乎无损**：18/18 个长上下文探针 argmax 全一致，KL 仅 0.0004–0.0019。

**对 256K 退化的结论**：`B0 vs H8` 在 256K 的 KL 0.042 里，**YaRN 贡献最大（0.0287）**，权重次之（0.0143），KV 与 head 可忽略。此前"FP8 KV 随长度累积"的猜测不成立。

**对 >262K 的推论**：模型原生上限 262144，超过就必须 YaRN（默认 rope 会 NaN/越界）。因此：
- 若要"相对 rope 配对的准无损基线"衡量，差值 = 纯量化（权重 + head）；
- 若要衡量"使用 YaRN 本身的代价"，只能在 ≤262144 用 `W4 vs W4y` 配对测——结果显示该代价随长度增长，在 256K 已超过权重。合理外推：**384K 下 YaRN 会是主导退化项**。

## 5. 解读

1. **top-1 有可测的变化**：dense 上约 **4.6%–8.3% 的位置 argmax 改变**（代码最少、中文最多）。即生产配置确实会在相当比例的位置改变下一个 token。
2. **KL 绝对值很小**（中位 0.001–0.014 nats，均值 0.027–0.056），量级与文献中 W4 权重量化（GPTQ/AWQ 报告的 KL ~0.01–0.1）一致。
3. **域差异明显**：代码分布最尖（覆盖率 99.3%、top-1 一致 95.4%、KL 中位 0.0009），量化几乎不改 argmax；中文分布最平（覆盖率 93.2%、top-1 一致 91.7%），最敏感。
4. **长上下文在 256K 出现退化信号**：`B0 vs H8` 的 KL 从 32K/128K 的 0.008–0.012 升到 **0.042**，且 6 条探针中 1 条 argmax 改变。单变量归因显示这**主要不是 FP8 KV**（KV 单独只贡献 ~0.002），而是权重 INT4 与 YaRN 在长上下文下的表现。

## 6. Caveats（口径限制）

- **B0 是 FP8 而非 BF16**，因此这不是"相对无量化"的损失，而是"相对 FP8"的损失。
- `B0 vs H8` 混合了 **权重 INT4 + head int8 + YaRN rope** 三项；其拆解见第 4 节的单变量归因（W4 / W4y / H8f 已跑）。
- **YaRN 不是量化**：它是上下文外推配置。`W4 → W4y` 的差值里含 mscale（常数，全长度生效）与频率插值两部分，报告未再细分。
- KL 是 **top-100 截断**值（交集重归一化），是真实 KL 的近似；覆盖率 93%–99% 说明尾部仍有未计入质量，中文域尤甚。
- 长上下文每档仅 6 个探针位置，统计功效有限，只能看趋势不能给精确差值。
- 探针位置是"长上下文后的第一个生成 token"，不同配置各自贪心，位置 0 上下文相同、可比；未做多位置 teacher forcing。
- 全部为 **T=0 贪心**；生产温度为 1.0，采样随机性大于量化偏移，故这些数字描述的是"分布偏移"，不等于"生产输出变化幅度"。

## 7. 复现

```bash
# 1) 语料（生产 venv 的 tokenizer）
cd repo-2080ti/experiments/accuracy-regression
/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/venv/bin/python tokenize_corpus.py

# 2) 逐配置起服
#    dense 用 LOGIT_EVAL=1（--skip-tokenizer-init，快，logprobs 与模式 2 逐位一致）
#    probes 用 LOGIT_EVAL=2（需 tokenizer 才能返回生成侧 logprobs）
setsid --fork env LOGIT_EVAL=2 KV_OVERRIDE=fp8_e4m3 POOL=4900000000 \
  bash launch.sh B0 </dev/null > logs/B0.log 2>&1
bash ../../scripts/tools/wait_server.sh logs/B0.log 600 8000

# 3) 评测
/home/aiakos/Temp/opencode/accuracy-eval/venv/bin/python run_logit_eval.py --config B0

# 4) 对比（归因链）
/home/aiakos/Temp/opencode/accuracy-eval/venv/bin/python compare.py --base B0  --other W4
/home/aiakos/Temp/opencode/accuracy-eval/venv/bin/python compare.py --base W4  --other W4y
/home/aiakos/Temp/opencode/accuracy-eval/venv/bin/python compare.py --base W4y --other H8
/home/aiakos/Temp/opencode/accuracy-eval/venv/bin/python compare.py --base H8  --other H8f
```

原始数据：`repo-2080ti/experiments/accuracy-regression/raw/{B0,W4,W4y,H8f,H8}/`（每段一个 npz），
汇总：`raw/compare_*.json`。
评测 venv：`/home/aiakos/Temp/opencode/accuracy-eval/venv`（lm-eval 0.4.13、numpy，未装 torch）。
