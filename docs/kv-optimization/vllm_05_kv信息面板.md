# vLLM KV 优化 05：KV 信息面板

> 实时观测保活、锚点、RAM/SSD offload 的指标与面板。数据来自 vLLM 的 Prometheus
> `/metrics`、引擎进程的 `/proc`、启动日志，以及 `nvidia-smi` / `du`。
>
> 相关文档：
> - 保活：[`vllm_01_保活.md`](vllm_01_保活.md)
> - 锚点：[`vllm_02_锚点.md`](vllm_02_锚点.md)
> - RAM offload：[`vllm_03_offload_ram.md`](vllm_03_offload_ram.md)
> - SSD offload：[`vllm_04_offload_ssd.md`](vllm_04_offload_ssd.md)

---

## 1. 实时监控脚本 `scripts/monitor_host_tier.py`

默认每 5 秒清屏重绘，分类显示配置与状态（仅用 Python 标准库）。

```bash
python scripts/monitor_host_tier.py            # 5s 刷新
python scripts/monitor_host_tier.py -d 10      # 10s 刷新
python scripts/monitor_host_tier.py --once     # 打印一次
python scripts/monitor_host_tier.py --json --count 5   # JSON 行（便于脚本化）
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `-d/--interval` | `5` | 刷新秒数 |
| `--url` | `http://localhost:8000` | vLLM 服务地址 |
| `--once` | — | 打印一次退出 |
| `--count N` | `0`（永久） | 打印 N 次退出 |
| `--server-log` | 自动（最新 `logs/server*.log`） | 解析配置用 |
| `--ssd-root` | 配置/env | `du` 统计的 SSD 目录 |
| `--json` | — | 每 tick 输出一行 JSON（不清屏） |
| `--no-clear` | — | 不清屏 |

### 1.1 数据来源

- **实时值**：`/metrics`（保活、锚点、GPU KV、RAM/SSD 池与计数器）。
- **配置**：引擎进程 `/proc/<pid>/{environ,cmdline}` + 启动日志
  （`GPU KV cache size`、`HostTierSSD: staging_slots=.. chunk_slots=.. evict_small=..
  block_size=..`、`HostTierSSDStore: root=.. quota=..`）。
- **资源**：`nvidia-smi` 两卡显存、`du -sb` 的 SSD 目录实际占用。

### 1.2 输出样式（示意）

```
════════════════ Host-Tier Monitor 19:27:43 ════════════════

 CONFIG
   endpoint     http://localhost:8000
   keep-alive   pin_min_tokens   = 16000
   anchors      ckpt_tokens      = 32000        anchors = 3
   eviction     small_tokens     = 32000        (small<32000 first, else oldest)
   GPU KV pool  kv_cache_bytes   = 8.4 GiB       max_model_len = 435200
                block_size       = 1600 tok     max_num_seqs = 1
                capacity         = 489,789 tok   (306 blocks)
   staging      cpu_bytes_to_use = 3.7 GiB       slots = 71   chunk_slots = 35
   SSD          root             = /home/.../ssd_kv
                quota = 64.0 GiB   max_mibps = 800   only = 1   clean_start = 1

 STATUS
   Keep-alive   entries 1   blocks 38   tokens 60.8k   anchors 3 blk / 1 sess
   GPU KV       [█░░░░░░░░░]  11.8%   57,801 / 489,789 tok   (36 / 306 blocks)
   RAM pool     [░░░░░░░░░░]   0.0%   0 / 71 slots      sessions 0
                spills 0 (+0)   restores 0 (+0)   evictions 0 (+0)   drops 0 (+0)
   SSD pool     [░░░░░░░░░░]   0.0%   0 B / 64.0 GiB   sessions 0
                stores 0 (+0)   restores 0 (+0)   evictions 0 (+0)   drops 0 (+0)
                write 0 B (+0 B)   read 0 B (+0 B)
   Resources    GPU0 21747 / 22528 MiB   GPU1 21244 / 22528 MiB   ssd_dir 0 B

 note: vLLM has no agent session; "session" = one request KV chain
       (matched to a later request by prefix hash).
```

> 计数器显示 `累计 (+本区间增量)`；池用量带进度条与百分比；`CONFIG` 每 tick 随清屏重绘。
>
> 实测：单条 52k 保活会话 → `entries 1 / blocks 38 / tokens 60.8k / anchors 3 blk / 1 sess`
> （3 个 Mamba 组各 1 锚点），与 `/metrics` 一致。

---

## 2. Prometheus 指标全集

> 计数器在 `/metrics` 中带 `_total` 后缀；`vllm:kv_cache_usage_perc` 为 0–1 的比例。

### 2.1 保活 / 锚点

```
vllm:keep_alive_entries            # gauge: 保活 pin 条目（请求链）数
vllm:keep_alive_blocks             # gauge: 保活持有的 KV 块数（各链长度之和，共享前缀会重复计）
vllm:keep_alive_tokens             # gauge: 保活持有的缓存 token 数
vllm:keep_alive_anchors            # gauge: 保活链持有的 Mamba 锚点块数
vllm:keep_alive_anchor_sessions    # gauge: 至少含 1 锚点的保活链数
```

### 2.2 GPU KV 池

```
vllm:kv_cache_usage_perc           # gauge: GPU KV 池占用率(0-1)
```

### 2.3 RAM host-tier

```
vllm:host_tier_slots_used          # gauge: staging 槽已用
vllm:host_tier_slots_total         # gauge: staging 槽总量
vllm:host_tier_sessions            # gauge: RAM 中 parked 会话数
vllm:host_tier_spills_total        # counter: 落盘(RAM)会话数
vllm:host_tier_restores_total      # counter: 恢复会话数
vllm:host_tier_evictions_total     # counter: RAM 配额驱逐数
vllm:host_tier_drops_total         # counter: 失败/超容量丢弃数
```

### 2.4 SSD host-tier

```
vllm:host_tier_ssd_sessions        # gauge: 驻留 SSD 的会话数
vllm:host_tier_ssd_bytes_used      # gauge: 已用字节
vllm:host_tier_ssd_quota_bytes     # gauge: 配额
vllm:host_tier_ssd_stores_total    # counter: 成功落盘会话数
vllm:host_tier_ssd_restores_total  # counter: 成功恢复会话数
vllm:host_tier_ssd_evictions_total # counter: 配额 LRU 驱逐数
vllm:host_tier_ssd_drops_total     # counter: 失败/超容量丢弃数
vllm:host_tier_ssd_write_bytes_total  # counter: 写入字节
vllm:host_tier_ssd_read_bytes_total   # counter: 读出字节
```

---

## 3. RAMTRACE 诊断轨迹（`RAMTRACE=1` → `$RAMTRACE_LOG`，默认 `/tmp/vllm_ramtrace.log`）

`diag lookup`(逐组命中)、`sched`(n_new/local/look/nblocks)、`pin g*`(各组块数)、
`spill groups`/`register_restored`(逐组块数)、`mamba cap`(捕获的 GDN 状态块)、
`restore blocks`(还原块 id)、`NAN target`(目标 logits 的 NaN 数)、
`alloc gate`/`mamba defer`(准入/对齐阻塞原因)。

SSD 分块相关：`ssd spill begin/hashes`、`ssd spill chunk copied`、`ssd store done`、
`restore begin/enqueued`、`hold_restored`、`ssd restore chunk done`、`restore done/probe`、
`ssd evict`、`gate grp*`、`relief req=`。

---

## 4. 环境变量总表

### 4.1 保活 / 锚点

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_PIN_MIN_TOKENS` | `0`（关） | 保活阈值；>0 时结束且 tokens≥阈值的请求整链 pin |
| `VLLM_MAMBA_CKPT_TOKENS` | `0`（关） | 锚点节奏（tokens）；须为 mamba block_size 整数倍 |
| `VLLM_MAMBA_CKPT_ANCHORS` | `3` | 每请求锚点上限（`max(1,·)`） |

### 4.2 host-tier（RAM / SSD）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_DISABLE_HOSTTIER` | `False` | 全局 kill-switch（含 SSD） |
| `VLLM_SPILL_NO_MAMBA` | `False` | 诊断：spill/保活时不捕获 Mamba 边界状态块 |
| `VLLM_RESTORE_TRUST` | `False` | 诊断：信任刚 restore 的链，跳过 connector 重对齐 |
| `RAMTRACE` | `False` | 诊断：写 host-tier spill/restore 轨迹 |
| `RAMTRACE_LOG` | `/tmp/vllm_ramtrace.log` | 轨迹输出路径 |
| `VLLM_HOSTTIER_EVICT_SMALL_TOKENS` | `64000` | 驱逐分档阈值（见 03 §9；0 = 单档最旧优先） |

### 4.3 SSD

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_SSD_ROOT` | `""` | SSD 会话仓根目录；空 = 禁用（RAM 回退） |
| `VLLM_SSD_QUOTA_BYTES` | `8 GiB` (8589934592) | 配额；超限按 LRU 删整会话目录 |
| `VLLM_SSD_READ_THREADS` | `8` | 读优先 I/O 线程 |
| `VLLM_SSD_WRITE_THREADS` | `8` | 写优先 I/O 线程 |
| `VLLM_SSD_MAX_MBPS` | `0` | 聚合读写限速（0 = 不限） |
| `VLLM_SSD_CLEAN_START` | `1` | 启动清空本 engine 的会话目录 |
| `VLLM_SSD_CHUNK_SLOTS` | `0` | 每 chunk 的 slot 数；0 = staging 的一半（双缓冲） |
| `VLLM_SSD_ONLY` | `1` | 1 = 必须有 `VLLM_SSD_ROOT`+staging，否则启动失败（禁用 RAM 回退） |

### 4.4 连接器

```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"cpu_bytes_to_use":<bytes>}}'
```

`cpu_bytes_to_use` = staging 池总大小（双 rank）。启用 host-tier 需 connector + 该值 >0
且 `VLLM_DISABLE_HOSTTIER` 未设（见 03 §4 门控表）。
