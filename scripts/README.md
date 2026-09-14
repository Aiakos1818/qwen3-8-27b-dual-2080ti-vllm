# scripts/ 目录索引

启动 profile 与共享测试库放在 `scripts/` 根；其余按用途分组。
模型路径、解释器、服务地址等一律取自仓库根 `.env`（`MODEL_PATH`、`VLLM_PYTHON`、
`FLASHQLA_PATH`、`VLLM_BASE_URL` 等），可用环境变量覆盖。机制与实测见
[`docs/kv-optimization/`](../docs/kv-optimization/README.md)。

> 从仓库根运行，例如 `bash scripts/run_vllm_qwen38_awq_fp8e4m3_100k.sh`、
> `python scripts/tools/kv_pool_sizing.py scripts/run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh`。
> 通过软链（如 `deploy/scripts -> repo/scripts`）调用同样可用。

## 启动 profile（根目录）

| 文件 | 用途 |
|---|---|
| `run_qwen3.8_27b_sm75.sh` | 基础部署（FP8 权重，180K 单请求） |
| `run_vllm_qwen38_awq_fp8e4m3_100k.sh` | AWQ-INT4 + fp8_e4m3 KV，100K 池（小尺度实验台） |
| `run_vllm_qwen38_awq_fp8e4m3_256k.sh` | AWQ-INT4，256K 上下文，SSD-only 两层 offload |
| `run_vllm_qwen38_awq_fp8e4m3_435k_ssd.sh` | AWQ-INT4，435K 上下文，SSD-only，分块流式 |
| `run_vllm_qwen38_awq_fp8e4m3_pool9.6e9.sh` | AWQ-INT4，9.6e9 池（~500.8k 安全上限） |
| `run_vllm_qwen38_fp8_fp8e4m3_100k_kv.sh` | FP8 权重变体，100K 池 |
| `revert_lib.py` | 共享测试库（长会话构造/发送；供 `checks/`、`probes/` import） |

## setup/ — 硬件与环境准备

| 文件 | 用途 |
|---|---|
| `verify_hardware.sh` | 校验 GPU / NVLink 拓扑与关键依赖 |
| `check_flashinfer_sm75.sh` | 检查 FlashInfer SM75 prefill 头文件 |
| `apply_gdn_flashqla_legacy.py` | 手动应用被拒的 GDN patch hunk 到 vLLM 0.27.1 源码 |

## tools/ — 运行期观测与容量测算

| 文件 | 用途 |
|---|---|
| `monitor_host_tier.py` | 实时面板：保活 / 锚点 / RAM+SSD 池 / I/O（`/metrics` + `/host_tier_info`） |
| `kv_pool_sizing.py` | 诊断启动 profile 的池/上下文是否匹配，给推荐值（`--feasible` 可实测 OOM） |

## checks/ — 自动化验证与编排

| 文件 | 用途 |
|---|---|
| `correctness_check.py` | 恢复会话的 greedy 输出正确性（sha 比对） |
| `ssd_matrix.py` | 单次启动的 SSD host-tier 功能矩阵（含 NaN 扫描） |
| `ssd_100k_check.py` | 100K 规模两层 SSD（真 NVMe）park/resume |
| `ssd_435k_check.py` | 435K 规模分块 SSD park/resume |
| `ssd_435k_revert_check.py` | 435K 锚点修复后：resident + restore 后深回退 |
| `ssd_512k_anchor_check.py` | 512K 满长上下文 + Mamba 锚点深回退 |
| `ssd_crash_check.py` | SSD 会话 store 的中断原子性 |
| `test_kv_100k.py` | 100K 池 APC / KV 驱逐语义测试客户端 |
| `offload_matrix.py` | host-tier spill/restore 压力矩阵（100K 池 + CPU tier） |
| `mtp_probe.py` | offload resume + logprobs 的 NaN 探测（MTP 调试） |
| `p03_restore_revert.py` | restore × 深回退交互 |
| `p15_concurrent.py` | 并发 spill/restore 压力（`max-num-seqs 4`） |
| `run_cmatrix_one.sh` | 单 cadence 编排：起引擎 → 跑 `check_*` → 停引擎，追加 TSV |
| `run_anch_measure.sh` | 单次锚点显存采样：起引擎 → 跑 resident → 逐秒采 nvidia-smi |

## probes/ — 人工定向探测

| 文件 | 用途 |
|---|---|
| `probe_kv_100k.py` | 驱逐取最旧会话的头还是尾 |
| `probe_kv_evdir.py` | 受控驱逐方向探测 |
| `probe_evict_isolate.py` | 隔离驱逐方向（配合 `VLLM_DEBUG_EVICT=1`） |
| `probe_one_page.py` | 仅驱逐 ~1 页，验证复用是否存活 |
| `probe_interleave.py` | 插入请求（不驱逐）是否破坏 A 的前缀复用 |
| `probe_keepalive.py` | 保活生效验证（默认 16k 阈值） |
| `probe_keepalive_off.py` | 关闭开关回归（`VLLM_PIN_MIN_TOKENS=0`） |
| `revert_matrix.py` | opencode revert+resend 前缀缓存矩阵驱动 |
| `revert_pinprobe.py` | 大历史 truncating revert 的复用/双释放探测 |
| `revert_ckpt.py` | A1 检查点保留验证（100K 池） |
| `revert_cmatrix.py` | cadence 对比（每个新引擎跑一次，输出 TSV） |
| `resident_once.py` | 发送一条完整 ~86k resident 请求 |
| `resident_big.py` | 近满 ~97k resident 请求构造 |

## capture/ — opencode 负载捕获

| 文件 | 用途 |
|---|---|
| `capture_proxy.py` | 只捕获转发的代理（`:8001 -> :8000`，NDJSON 记录请求体） |
| `capture_opencode.json` | opencode 指向捕获代理的配置样例 |
| `oc_fixture.json` | 捕获的 opencode 系统提示词（设 `KV_TEST_FIXTURE` 复现） |
