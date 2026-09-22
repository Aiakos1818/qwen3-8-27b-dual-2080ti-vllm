# ninfer-2080ti 评估记录（2026-09-22）

评估 `Neroued/ninfer` 与 `mr-september/ninfer-2080ti-22g` 是否值得为本机
（双 RTX 2080Ti 22G / SM75）的 vLLM 部署借鉴，重点是 **decode / prefill 速度**。
结论：**不值得移植**。本页记录构件、构建坑、实测数据与对 int8-KV / shortlist 的调研结论。

## 1. 构件与构建

- 上游运行时：`https://github.com/mr-september/ninfer-2080ti-22g`（Neroued/ninfer 的 SM75 移植分支，commit `6460bf0`）。
- 权重构件：ModelScope `wangcanace/Qwen3.8-27B-NInfer-wwt` 的 `qwen3_8_27b.ninfer`
  - 18,210,531,328 B（16.96 GiB），sha256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`（校验通过）
  - `artifact-manifest.json`：`container_version: 2`、`weights_id: groupwise-int`、`target_key: qwen3_8_27b`、转换自 `charlesarcher/ninfer-4090`（RTX 4090 / sm_89 fork）
  - magic 头 `NINFER\x00\x02` 确认 v2 容器，与移植版的 v2 容器契约一致；`tools/artifact/{numeric,layouts}.py` 覆盖其全部格式/布局。
- 环境：单张 2080Ti 22G（`CUDA_VISIBLE_DEVICES=0`）、CUDA 12.8.93、gcc 13.3、venv 内 cmake 4.4.3 + ninja 1.13.2。

### 构建坑：单个 TU 的 ptxas 病态（可复现）

`src/ops/launcher/gqa_attention_decode.cu` 一次性实例化全部 small-T 宽度的
INT8/BF16 consumer kernel（约 192 个巨型实例化），在 sm_75 上单个 ptxas 调用
**> 52 分钟不收敛**（-O1 / -O2 / -O3、`-Xptxas --allow-expensive-optimizations=false`、
`--split-compile[-extended]` 均无效或无效）。

做法：把该 TU 按 T（1..6）拆成 6 个并行编译单元，全 `-O3` 下约 32 分钟编完：

- 新增 `src/ops/launcher/gqa_attention_decode_width.cuh`：承载 split 策略、`launch_tc_partial_{bf16,i8}`、
  `single_row_batch_view` 与 `gqa_small_t_launch_width<Geometry, CacheInput, W>`（原 switch 体）。
- `gqa_attention_decode.cu` 保留 runtime dispatch/capacity/reduce，并对 24 个
  `gqa_small_t_launch_width<...>` 实例做 `extern template`。
- 新增 `gqa_attention_decode_w{1..6}.cu`，各自显式实例化 4 个 `(Geometry × CacheInput)` 组合。
- `src/CMakeLists.txt` 的 `NINFER_OPS_SOURCES` 加入这 6 个源文件。

（上述拆分不改变生成代码，仅恢复构建并行度。）

## 2. 单卡实测（Qwen3.8-27B groupwise-int / int8 group-64 KV / MTP3 + `--lm-head-draft`）

测量口径与 `repo-2080ti/benchmarks/run_context_ttft.py` 相同（同合成 prompt、同脚本）：
TTFT 到首个 reasoning/content 字符；本构件无 `/metrics`，稳态字段不可用。

| prompt tokens | prefill | decode | 备注 |
|---:|---:|---:|---|
| 29,176 | 79.4 tok/s | 15.9 tok/s | 生成 114 token |
| 58,309 | 77.1 tok/s | 14.8 tok/s | 生成 512 token |
| ~85K（82k words） | 未跑完（已中止） | — | — |

冒烟（4K ctx）：prefill 38.3 tok/s、decode 17.05 tok/s，MTP 接受率 48.65%、2.38 tok/round。
启动显存：权重 16.67 GiB，free-after-weights 4.64 GiB，`--max-context` 上限约 **90K**
（spec：`--max-context 90112 --kv-capacity auto`，runtime 3.58 GiB）。

### 与本机 vLLM 对照（同机已发布数据，README/benchmarks）

| 指标 | ninfer-2080ti | 本机 vLLM |
|---|---:|---:|
| prefill（2.8K–59K） | ~77–79 tok/s | ~1100–1338 tok/s |
| decode（128 窗） | 14.8–15.9 tok/s | 84–101 tok/s |
| 稳态 decode | —（无计数器） | ~45–70 tok/s |

**结论**：prefill 约 **1/15**、decode 约 **1/3–1/5**（即便拿 vLLM 稳态口径比）。
ninfer 在本硬件上没有可迁移的净收益。

## 3. 对"值得借鉴"的答复

1. **端到端不值得借鉴**：ninfer 的 int8-KV + int8 张量核 QK + shortlist draft head 等机制在其
   自家 SM75 实现上端到端仍慢于本机 vLLM，不能作为"这些技术在 SM75 上有效"的证据。
2. **只借鉴到一条移植性教训**：上述"按 T 拆 TU"解决 ptxas 病态。
3. **保留意见**：构件来自 4090 fork；移植版 README 声称 MTP3 decode 42–44 tok/s 未复现（只 ~15），
   预填充则与其自公布的 71–135 tok/s 吻合。要最公平需用 HF 官方 `neroued/Qwen3.8-27B-NInfer` 构件复测，
   但 prefill 的巨大差距与构件无关，方向结论不变。

## 4. 本机 vLLM 上 int8-KV / shortlist 的调研结论（未实施）

**int8-KV（判定：死路）**

- 本 fork 支持 `int8_per_token_head`（`vllm/config/cache.py:39-58`），但**仅 `TRITON_ATTN` 声明支持**
  （`vllm/v1/attention/backends/triton_attn.py:294-306`）；FlashInfer 只支持 fp8/fp16/nvfp4
  （`flashinfer.py:415-424`）。本机后端选择为 `FLASHINFER`（候选仅 `FLASHINFER, TRITON_ATTN`）。
- 本机已实测 `TRITON_ATTN + fp16 KV` 慢 5×（`upstream-branch.md` §6.3，9–13 tok/s），
  故 int8-KV 走 triton 更差；FlashInfer 侧无 int8 通道，加它需自写 SM75 kernel。不建议投入。

**shortlist draft head（判定：边际优化，未实施）**

- 基础设施已具备：`Qwen3_5MTP` 已继承 `LocalArgmaxMixin`（`qwen3_5_mtp.py:233`），
  `interfaces.py:1522-1557` 的 `get_top_tokens` 自动套 `d2t`；缩减 head + `draft_id_to_target_id` +
  `compute_logits` scatter 回全词表的范本见 `qwen3_eagle3.py:302-319,347-369` 与
  `qwen3_dspark.py:271-280`；映射工具 `vllm/v1/spec_decode/vocab_mapping.py`；
  checkpoint 侧工可仿 `scripts/tools/quantize_lm_head.py`。
- MTP 现用全词表 head（`qwen3_5_mtp.py:262-273`）；改成缩减词表需改 `qwen3_5_mtp.py` +
  造 shortlist 变体 checkpoint。用 scatter 写法可对 speculator 完全透明（`speculator.py:400-437`）。
- 预期收益：按 §6.12/§6.15，lm_head 占每步 15–22%，draft 占 `n/(n+1)`；131072/248320 行约省
  ~3 ms/步（31.5K 约 6%、215K 约 4.6%）。但 shortlist 会压低 draft 质量，本机长上下文接受率本就
  30–42%，接受率相对下降 >~7% 即净负。判定：只有 31.5K 档短探针值得，未实施。

## 5. 复现要点

```
# 构建（拆分版本见 §1）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build --parallel --target ninfer ninfer-serve

# 单卡服务
CUDA_VISIBLE_DEVICES=0 ./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b --max-context 90112 \
  --kv-capacity auto --max-concurrency 1 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft

# 对照测量
python repo-2080ti/benchmarks/run_context_ttft.py --base-url http://127.0.0.1:8080 \
  --model qwen3.8-27b --word-counts 28000 56000 --runs 1 --max-tokens 512 --output <out>.json
```

临时工作区（源码 / 权重 / 日志 / 结果）位于 `~/Qwen3.8-27B-Ninfer/`，本记录落定后已清理。
