# experiments/ — 一次性实验脚本与测量数据归档

本目录收纳 `docs/upstream-branch.md` §6–§9 各小节实验所用的 launcher、driver、分析脚本与
原始测量数据。这些**不是 serving profile**（正式 profile 在 `scripts/`），只是复现文档结论
的入口，多为一次性脚本，仅做了去本机路径改造（脚本位置从临时目录搬进仓库）。

## 前提

- 路径 / 模型 / 解释器从仓库根的 `.env` 取（`MODEL_PATH` / `VLLM_PYTHON` / `FLASHQLA_PATH` /
  `CHAT_TEMPLATE`）。
- 需要 `zyYuc-sandbox` 的 vLLM 树与 FlashQLA（见 `docs/PATCHING.md`）。
- 部分实验需额外权重：§6.17 需要 `models/Qwen3.8-27B-DFlash2/`；§6.15/6.16 的
  `run_500k_head4bit.sh` 需要 `${MODEL_PATH}-head4bit` 变体。
- 驱动脚本按端口 8000 启停服务，会占用 GPU；运行前确认没有其他实例。

## 索引（→ 对应 `docs/upstream-branch.md` 小节）

### §6.7 KV dtype A/B
| 文件 | 说明 |
|---|---|
| `run_ab_128k.sh` | 256K profile 形状指向 128K，`KV_CACHE_DTYPE`/`KV_CACHE_MEMORY_BYTES` 可覆盖（**未点名**，支撑 §6.7 KV-dtype 对照） |
| `run_test.sh` | 命令行覆盖优先于 `.env` 的 A/B launcher（**未点名**） |

### §6.12–6.14 profiling / n 对照
| 文件 | 说明 |
|---|---|
| `run_prof_500k.sh` | 500K + torch profiler dir（**未点名**） |
| `run_prof_500k_iter.sh` | 500K + iteration-detail + profiler（**未点名**） |
| `run_prof2.sh` | 500K prefill profiler（**未点名**） |
| `run_prof_256k.sh` | 250K decode attribution（C3，**未点名**） |
| `prof_prefill.py` | prefill 段 trace 采集（**未点名**） |
| `analyze_full.py` / `analyze_shapes.py` / `analyze_trace.py` | trace 解析（**未点名**） |
| `salvage.py` | 截断 trace 抢救（§6.14 · 1126 行点名） |
| `decode_profile.py` | decode 速率随位置采样（`n_*` 的依赖，**未点名**） |
| `n_check.sh` / `n_sweep.sh` / `n_sweep2.sh` | n 值扫描（**未点名**，支撑 §6.12–6.14 的 n 对照） |

### §6.15–6.16 head 量化 / native spec
| 文件 | 说明 |
|---|---|
| `run_500k_head4bit.sh` | head4bit 变体 launcher（§6.15/6.16 · 180/1104 行点名） |
| `run_256k_mtp6_native0.sh` | 关闭 native spec-as-decode（**未点名**） |
| `run_256k_mtp6_fullcg.sh` | FULL cudagraph 档（**未点名**） |

### §6.17 上游 DFlash2 评估
| 文件 | 说明 |
|---|---|
| `run_256k_dflash{,_cg,_eager,_nonative,_try2}.sh` | DFlash2 各档 launcher（1240 行 `run_256k_*.sh` 覆盖） |
| `ab_dflash.json` / `ab_mtp.json` | 同 prompt A/B 数据（1239 行点名） |
| `fix1_30k.json` / `fix2_mtp_30k.json` / `fullcg_30k.json` | 修复后复测数据（1239 行点名） |

### §6.18 model runner V1 vs V2
| 文件 | 说明 |
|---|---|
| `run_256k_mtp6_{v2,v1,v1auto}.sh` | V2/V1/V1+auto 三个 launcher（1291 行点名） |
| `ab_v1_driver.sh` / `ab_v1auto_driver.sh` | 单相 A/B driver（1294 行点名） |
| `ab_runner.sh` | V1/V2 A/B driver（**未点名**） |
| `ab_runner_v1.json` / `ab_runner_v1auto.json` | 测量数据（1294 行点名） |

### §6.19 thinking_token_budget
| 文件 | 说明 |
|---|---|
| `run_256k_mtp6_budget.sh` | budget 档 launcher（1341 行点名） |
| `ab_budget_driver.sh` / `ab_budget_driver2.sh` | budget A/B driver（1341 行隐含） |
| `test_thinking_budget.py` / `test_tool_after_budget.py` | A/B/C 与 D/E 用例（1340 行点名） |

### §9 上游同步 A/B
| 文件 | 说明 |
|---|---|
| `ab_branch_driver.sh` / `ab_rebased_driver.sh` | rebase 前后配对 A/B（**未点名**） |
| `ab_old6.json` / `ab_new6.json` / `ab_rebased.json` | 配对数据（1450 行点名） |
| `bench_old6.out` / `bench_new6.out` | bench 输出（1450 行点名） |

### §6.8–6.11 手写注意力 kernel（已停止路线）
`kern/` 归档了 SM75 手写 paged-attention kernel 实验（§6.8 结论"停止 kernel 重写"）：
- `mq_attn.cu` / `mq_split.cu` — 主 kernel 与拆分版（sm_75 + wmma，fp8→fp16 位运算转换）
- `mqattn.py` / `bwprobe.py` / `bwprobe_lib.py` — Triton 版与带宽探针
- `diag*.py` / `probe*.py` / `sweep.py` / `variants.py` / `test_*.py` / `stageprobe.py` /
  `fp16probe.py` / `bench_split.py` / `decode_bench.py` / `mini.py` — 性能差距定位脚本
- 目录内脚本用 `__file__` 定位同目录的 `.cu`，可整目录搬移。

未纳入的中间物（反汇编 `mq.sass`、`.cu.b1/.b2` 备份、FlashInfer 头文件副本
`prefill.cuh.orig`、`*.txt`/`*.log`/`__pycache__`）不属于源码，未归档。

### §6.20 量化精度损失（logit 级单变量归因）
| 文件 | 说明 |
|---|---|
| `accuracy-regression/launch.sh` | 5 配置统一 launcher（`LOGIT_EVAL=1/2`、`KV_OVERRIDE`、`POOL`） |
| `accuracy-regression/tokenize_corpus.py` | 三域语料切 8K 段 + 长上下文探针（用生产 venv 的 tokenizer） |
| `accuracy-regression/run_logit_eval.py` | 采集 `prompt_logprobs` / 生成侧 `logprobs`（top-100） |
| `accuracy-regression/compare.py` | top-1/top-5 一致率、KL（交集重归一化）、覆盖率 |

报告：[`../reports/2026-09-sm75-optimization/accuracy-regression/README.md`](../reports/2026-09-sm75-optimization/accuracy-regression/README.md)。
语料 `corpus/` 与原始 npz `raw/` **不入库**（体积大，见 `.gitignore`），可由 `tokenize_corpus.py`
与 `run_logit_eval.py` 重建。

### §6.21 decode 侧内核 / 投机 A/B
| 文件 | 说明 |
|---|---|
| `run_kernel_ab.sh` | 参数化 launcher：`LINEAR_BACKEND`（auto/humming/triton）、`SPEC_METHOD`（mtp/ngram_gpu）、`RUNNER_V2`、`ENABLE_THINKING` |
| `kernel_ab_bench.py` | 一次生成、按 `步数=gen−accepted` 折算 ms/verify 步；`--task open/repeat/extract` 或 `--messages-json` 打真实会话 |
| `oc_session_prompt.py` | 从 opencode SQLite 重建真实会话为 chat 请求（token 预算内从末尾填充、角色交替），供 `kernel_ab_bench.py --messages-json` 使用 |

原始 JSON 与日志不入库，见 `~/Temp/opencode/kernel-ab/`。

### §6.22 重训 MTP 头跨量化目标 A/B
| 文件 | 说明 |
|---|---|
| `run_kernel_ab.sh` | 复用 §6.21 launcher：`MODEL_PATH_IN` 覆盖 `.env` 的模型路径、`HEAD8BIT=0` 只换 MTP 而不换 int8 lm_head |
| `kernel_ab_bench.py` | 复用；新增 `--temp 0`（贪心）——换头/换内核的接受率 A/B 必须贪心，否则两腿 target token 流不同、接受率不可比 |

可变体模型目录与下载的重训头为一次性产物，不入库、已清理。

## 数据文件

`*.json` / `*.out` 是各实验的原始测量输出，供 `docs/upstream-branch.md` 引用核对。
运行脚本会就地覆盖同名输出（`ab_*.json`、`bench_*.out` 等），请勿把运行结果与归档数据混淆。
