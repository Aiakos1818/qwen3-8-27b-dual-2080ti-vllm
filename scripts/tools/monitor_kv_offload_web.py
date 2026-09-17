#!/usr/bin/env python3
"""Browser panel for the KV-offload data the terminal monitor collects.

The terminal monitor prints text frames: ideal for --once, --json, --append and
for grepping, but a character grid is a poor surface for *looking* at the state.
A browser brings mouse wheel, resize and scrolling for free, and cards and bars
cost nothing to draw. The genuinely hard part -- collecting the numbers -- is
identical either way, so this imports the collector instead of duplicating it and
serves one live page over HTTP.

Usage:
  monitor_kv_offload_web.py                        # http://127.0.0.1:8199
  monitor_kv_offload_web.py --port 9000 -d 2
  monitor_kv_offload_web.py --vllm-port 8001
  monitor_kv_offload_web.py --host 0.0.0.0         # LAN, no authentication
  curl -s localhost:8199/api/snapshot | python3 -m json.tool

Endpoints:
  GET /               the panel (inline CSS/JS, no external resources)
  GET /api/view       the tables and bars the page renders
  GET /api/snapshot   raw sample, same shape as the terminal --json output
  GET /healthz        liveness

Read-only and dependency-free (stdlib only): it only issues HTTP GETs to the
vLLM server, reads /proc, /dev/shm, /proc/meminfo, ``nvidia-smi`` and the disk
tier's files, and never writes anything. It binds to localhost by default
because the payload contains local paths; ``--host 0.0.0.0`` is opt-in and
unauthenticated, so an SSH tunnel is the better way to reach it remotely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor_kv_offload import (  # noqa: E402
    Metrics,
    collect_config,
    derive_chunk_bytes,
    fetch_metrics,
    find_pid,
    fmt_bytes,
    fmt_int,
    gpu_memory,
    log_health,
    mem_available,
    rate_pct,
    ratio_pct,
    scan_chunks,
    shm_stats,
    throughput,
)

# --------------------------------------------------------------------------
# view: the same numbers the terminal panel shows, as data
# --------------------------------------------------------------------------


def _chunk_count(nbytes: int | None, cfg) -> str:
    if not nbytes or not cfg.chunk_bytes:
        return "?"
    count = nbytes / cfg.chunk_bytes
    if cfg.chain_chunks:
        return f"{count:,.0f} chunks = {count / cfg.chain_chunks:.2f} chain"
    return f"{count:,.0f} chunks"


def build_view(cfg, metrics, prev, gpus, shm, disk, health, url, interval, tick, updated):
    """Everything the page needs, pre-formatted (so the JS stays dumb)."""

    def counter(name, **labels):
        return metrics.counter(name, **labels) if metrics is not None else None

    def delta(name, **labels):
        now = counter(name, **labels) or 0.0
        before = prev.counter(name, **labels) if prev is not None else None
        return now - (before or 0.0)

    def hist(name, **labels):
        return metrics.hist(name, **labels) if metrics is not None else (None, None)

    note = ""
    tables: list[dict] = []
    bars: list[dict] = []
    headline = "decode —"   # stays empty when /metrics is unreachable

    if metrics is None:
        note = f"{url}/metrics 不可达 —— vLLM 实例未运行或端口不对"
    elif metrics.get("vllm:cache_config_info") is None:
        note = "已连上，但还没有 vllm:cache_config_info（引擎可能仍在启动）"

    config = [
        ("pid", cfg.pid),
        ("model", cfg.model),
        (
            "served / dtype",
            f"{cfg.served}   {cfg.dtype} + {cfg.kv_dtype}   tp {cfg.tp} ({cfg.devices})",
        ),
        ("max_model_len", fmt_int(cfg.max_model_len)),
        (
            "chunk",
            f"{fmt_int(cfg.block_size)} tok   ({fmt_bytes(cfg.chunk_bytes)} per chunk,"
            " both ranks)",
        ),
        (
            "GPU KV pool",
            f"{fmt_bytes(cfg.kv_cache_bytes)}   {fmt_int(cfg.pool_tokens)} tok"
            f"   {fmt_int(cfg.gpu_blocks)} blocks",
        ),
        (
            "engine / scheduling",
            f"gpu_memory_utilization {cfg.gpu_mem_util}"
            f"   max_num_seqs {cfg.max_num_seqs}"
            f"   max_num_batched_tokens {cfg.max_num_batched_tokens}",
        ),
        ("engine_id", cfg.engine_id),
        (
            "offload spec",
            f"{cfg.spec_name}   eviction {cfg.eviction}"
            f"   load-failure {cfg.load_failure}",
        ),
        ("CPU tier", f"{fmt_bytes(cfg.cpu_bytes)}   {_chunk_count(cfg.cpu_bytes, cfg)}"),
        ("fs tier root", cfg.fs_root or "(none)"),
        (
            "fs tier budget",
            f"{fmt_bytes(cfg.fs_max_bytes)}   {_chunk_count(cfg.fs_max_bytes, cfg)}",
        ),
        (
            "mamba",
            f"block_size {cfg.mamba_block_size}   kv_cache_layout {cfg.cache_layout}",
        ),
    ]

    if metrics is not None:
        running = metrics.get("vllm:num_requests_running") or 0.0
        waiting = metrics.get("vllm:num_requests_waiting") or 0.0
        reasons = ", ".join(
            f"{reason} {int(metrics.get('vllm:num_requests_waiting_by_reason', reason=reason) or 0)}"
            for reason in metrics.labels_of(
                "vllm:num_requests_waiting_by_reason", "reason"
            )
        )
        seen = metrics.labels_of("vllm:request_success_total", "finished_reason")
        ordered = [r for r in ("stop", "length", "abort", "error") if r in seen] + [
            r for r in seen if r not in ("stop", "length", "abort", "error")
        ]
        finished = "   ".join(
            f"{reason} {int(metrics.get('vllm:request_success_total', finished_reason=reason) or 0)}"
            for reason in ordered
        )

        prefix_hits = counter("vllm:prefix_cache_hits") or 0.0
        prefix_q = counter("vllm:prefix_cache_queries") or 0.0
        ext_hits = counter("vllm:external_prefix_cache_hits") or 0.0
        ext_q = counter("vllm:external_prefix_cache_queries") or 0.0
        prompt_cached = counter("vllm:prompt_tokens_cached") or 0.0

        def transfer(direction: str) -> str:
            nbytes = counter("vllm:kv_offload_total_bytes", transfer_type=direction)
            seconds = counter("vllm:kv_offload_total_time", transfer_type=direction)
            count, total = hist("vllm:kv_offload_size", transfer_type=direction)
            mean = f"   mean {fmt_bytes(total / count)}" if count else ""
            return (
                f"{fmt_bytes(nbytes)} in {seconds or 0:.2f} s"
                f" ({throughput(nbytes, seconds)})   transfers {fmt_int(count)}{mean}"
            )

        cpu_usage = metrics.get("vllm:kv_offload_cpu_cache_usage_perc")
        cpu_write = metrics.get("vllm:kv_offload_cpu_cache_write_usage_perc")
        cpu_read = metrics.get("vllm:kv_offload_cpu_cache_read_usage_perc")

        bars = [
            {
                "label": "GPU KV cache",
                "frac": metrics.get("vllm:kv_cache_usage_perc"),
                "text": (
                    f"{fmt_int((metrics.get('vllm:kv_cache_usage_perc') or 0) * (cfg.pool_tokens or 0))}"
                    f" / {fmt_int(cfg.pool_tokens)} tok   {fmt_int(cfg.gpu_blocks)} blocks"
                    f"   prompt cached {fmt_int(prompt_cached)}"
                ),
            },
            {
                "label": "CPU (primary) tier",
                "frac": cpu_usage,
                "text": (
                    f"{fmt_bytes((cpu_usage or 0) * (cfg.cpu_bytes or 0))}"
                    f" / {fmt_bytes(cfg.cpu_bytes)}   write-hold {ratio_pct(cpu_write)}"
                    f"   read-hold {ratio_pct(cpu_read)}"
                ),
            },
        ]
        fs_used = metrics.get("vllm:kv_offload_tiering_fs_used_bytes")
        if fs_used is not None or cfg.fs_max_bytes:
            bars.append(
                {
                    "label": "fs (disk) tier quota",
                    "frac": (
                        fs_used / cfg.fs_max_bytes
                        if (fs_used and cfg.fs_max_bytes)
                        else None
                    ),
                    "text": (
                        f"{fmt_bytes(fs_used)} / {fmt_bytes(cfg.fs_max_bytes)}"
                        f"   files {disk.count} ({fmt_bytes(disk.bytes)})"
                        f"   evictions {fmt_int(counter('vllm:kv_offload_tiering_fs_evictions') or 0)}"
                        f"   skipped {fmt_bytes(counter('vllm:kv_offload_tiering_fs_skipped_store_bytes') or 0)}"
                    ),
                }
            )

        tier_rows = []
        for tier in metrics.labels_of("vllm:kv_offload_tiering_chunk_queries", "tier"):
            queries = counter("vllm:kv_offload_tiering_chunk_queries", tier=tier) or 0.0
            hits = counter("vllm:kv_offload_tiering_chunk_hits", tier=tier) or 0.0
            parts = [f"lookups {fmt_int(queries)}   hits {fmt_int(hits)} ({rate_pct(hits, queries)})"]
            read_bytes = counter("vllm:kv_offload_tiering_read_bytes", tier=tier)
            write_bytes = counter("vllm:kv_offload_tiering_write_bytes", tier=tier)
            if any(v for v in (read_bytes, write_bytes)):
                parts.append(
                    f"read {fmt_bytes(read_bytes or 0)}   write {fmt_bytes(write_bytes or 0)}"
                )
            lag, lag_sum = hist("vllm:kv_offload_tiering_lookup_sync_delay_seconds", tier=tier)
            if lag:
                parts.append(f"lookup sync {1000 * lag_sum / lag:.2f} ms")
            promotions = metrics.get("vllm:kv_offload_tiering_active_promotion_jobs", tier=tier) or 0
            cascades = metrics.get("vllm:kv_offload_tiering_active_cascade_jobs", tier=tier) or 0
            failures = (
                counter("vllm:kv_offload_tiering_promotion_job_failures", tier=tier) or 0,
                counter("vllm:kv_offload_tiering_cascade_job_failures", tier=tier) or 0,
            )
            alloc = counter("vllm:kv_offload_tiering_promotion_allocation_failures") or 0
            if promotions or cascades or any(failures) or alloc:
                parts.append(
                    f"jobs {int(promotions)}/{int(cascades)}"
                    f"   failures {int(failures[0])}/{int(failures[1])}"
                    f"   alloc-failures {int(alloc)}"
                )
            tier_rows.append((tier, "   ".join(parts)))
        if not tier_rows:
            tier_rows = [("—", "还没有 tiering 指标（尚未发生 offload）")]

        # Throughput. The engine exports token *counters*, not a rate, so the
        # live rate is the counter delta over one sampling interval; the first
        # sample has no previous scrape, and dividing the whole history by the
        # interval would be nonsense, so it is left empty there.
        window_gen = delta("vllm:generation_tokens_total")
        window_prompt = delta("vllm:prompt_tokens_total")
        live = prev is not None
        gen_count, gen_sum = hist("vllm:request_generation_tokens")
        _, decode_sum = hist("vllm:request_decode_time_seconds")
        avg_decode = (gen_sum / decode_sum) if (gen_sum and decode_sum) else None
        window_draft = delta("vllm:spec_decode_num_draft_tokens_total")
        window_accepted = delta("vllm:spec_decode_num_accepted_tokens_total")

        def rate(value: float | None, suffix: str = "tok/s") -> str:
            return f"{value:,.1f} {suffix}" if value is not None else "—"

        throughput_rows = [
            (
                f"decode (live, {interval:g}s window)",
                f"{rate(window_gen / interval if live else None)}"
                f"   {fmt_int(window_gen) if live else '—'} tok in window",
            ),
            (
                f"prefill (live, {interval:g}s window)",
                f"{rate(window_prompt / interval if live else None)}"
                f"   {fmt_int(window_prompt) if live else '—'} tok in window",
            ),
            (
                "decode (finished requests)",
                f"{rate(avg_decode)}"
                f"   {fmt_int(gen_count)} reqs, {fmt_int(gen_sum)} tok"
                f" in {decode_sum:,.1f}s"
                if avg_decode
                else "—",
            ),
        ]
        if window_draft or window_accepted:
            accepted = (
                window_accepted / window_draft if window_draft else None
            )
            steps = max(window_gen - window_accepted, 1e-9)
            per_step = 1 + window_accepted / steps
            throughput_rows.append(
                (
                    "spec decode (MTP)",
                    f"acceptance {ratio_pct(accepted)}"
                    f"   ≈{per_step:.2f} tok/step (draft {fmt_int(window_draft)},"
                    f" accepted {fmt_int(window_accepted)})",
                )
            )
        headline = (
            f"decode {window_gen / interval:,.1f} tok/s"
            if live
            else (f"decode {avg_decode:,.1f} tok/s (avg)" if avg_decode else "decode —")
        )

        shm_used, shm_total, regions = shm
        staging = sum(size for size, _, _ in regions)
        procs = sum(count for _, count, _ in regions)
        resource_rows = [
            ("GPU", "   ".join(
                f"{index}: {used} / {total} MiB" for index, used, total in gpus
            ) or "nvidia-smi 不可用"),
            (
                "/dev/shm",
                f"{fmt_bytes(shm_used)} / {fmt_bytes(shm_total)}"
                f"   staging {fmt_bytes(staging) if regions else '-'}"
                + (f" ({procs} processes)" if procs else ""),
            ),
            ("MemAvailable", fmt_bytes(mem_available())),
            (
                "disk tier dir",
                f"{fmt_bytes(disk.bytes)}   {disk.count} chunk file(s)"
                f"   tick {interval:.1f}s",
            ),
        ]

        tables = [
            {"title": "Throughput", "rows": throughput_rows},
            {
                "title": "Requests",
                "rows": [
                    ("running", f"{int(running)}"),
                    ("waiting", f"{int(waiting)}" + (f"   ({reasons})" if reasons else "")),
                    ("finished", finished or "—"),
                ],
            },
            {
                "title": "Cache",
                "rows": [
                    (
                        "prefix cache",
                        f"{fmt_int(prefix_hits)} / {fmt_int(prefix_q)}"
                        f" ({rate_pct(prefix_hits, prefix_q)})"
                        f"   last {interval:.0f}s: +{fmt_int(delta('vllm:prefix_cache_hits'))} hits",
                    ),
                    (
                        "external (offload)",
                        f"{fmt_int(ext_hits)} / {fmt_int(ext_q)}"
                        f" ({rate_pct(ext_hits, ext_q)})"
                        f"   last {interval:.0f}s: +{fmt_int(delta('vllm:external_prefix_cache_hits'))} hits",
                    ),
                    ("prompt tokens cached", fmt_int(prompt_cached)),
                ],
            },
            {
                "title": "Transfers (connector)",
                "rows": [
                    ("Store GPU→CPU", transfer("GPU_to_CPU")),
                    ("Load CPU→GPU", transfer("CPU_to_GPU")),
                ],
            },
            {"title": "Tiering", "rows": tier_rows},
            {"title": "Resources", "rows": resource_rows},
        ]

    chunk_summary = f"{disk.count} file(s)   {fmt_bytes(disk.bytes)}"
    if cfg.chain_chunks:
        chunk_summary += f"   ≈ {disk.count / cfg.chain_chunks:.2f} chain"
    if disk.newest:
        chunk_summary += (
            f"   newest {time.strftime('%H:%M:%S', time.localtime(disk.newest))}"
            f"   oldest {time.strftime('%H:%M:%S', time.localtime(disk.oldest))}"
        )
    chunks = {
        "root": disk.root or "(none)",
        "summary": chunk_summary,
        "rows": [
            [
                time.strftime("%H:%M:%S", time.localtime(mtime)),
                fmt_bytes(size),
                short,
            ]
            for mtime, size, short in disk.files
        ],
    }

    log_text = "— (未启用 --log)"
    if health is not None:
        log_text = (
            f"errors {health['errors']}   ({health['path']})"
            if health.get("errors") is not None
            else f"读取失败: {health.get('last')}"
        )
        if health.get("errors"):
            log_text += f"\nlast: {health['last']}"

    return {
        "endpoint": url,
        "tick": tick,
        "updated": time.strftime("%H:%M:%S", time.localtime(updated)),
        "interval": interval,
        "note": note,
        "headline": headline,
        "config": [(str(k), str(v)) for k, v in config],
        "bars": bars,
        "tables": tables,
        "chunks": chunks,
        "log": log_text,
        "footer": (
            "read-only · upstream 没有逐请求清单端点，GPU/CPU 层只有聚合值；"
            "磁盘层按 chunk hash 存放，列的是真实落盘 chunk"
        ),
    }


# --------------------------------------------------------------------------
# sampling thread: one collector for every browser, so the deltas mean something
# --------------------------------------------------------------------------


class Sampler:
    def __init__(self, args) -> None:
        self.args = args
        self.url = args.vllm_url or f"http://localhost:{args.vllm_port}"
        self.lock = threading.Lock()
        self.snapshot: dict | None = None
        self.view: dict | None = None
        self.tick = 0
        self.started = time.time()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.prev: Metrics | None = None

    def sample(self) -> None:
        args = self.args
        metrics = fetch_metrics(self.url)
        pid = args.pid or find_pid(args.vllm_port)
        cfg = collect_config(pid, args.vllm_port, metrics)
        disk = scan_chunks(
            args.ssd_root or cfg.fs_root, keep=max(args.chunk_lines, 1)
        )
        cfg.chunk_bytes = derive_chunk_bytes(cfg, disk)
        gpus = gpu_memory()
        shm = shm_stats()
        health = log_health(args.log)
        self.tick += 1
        snapshot = {
            "_tick": self.tick,
            "_elapsed": round(time.time() - self.started, 2),
            "_url": self.url,
            "config": dict(cfg.__dict__),
            "metrics": metrics.interesting() if metrics else None,
            "gpus": gpus,
            "shm": {
                "used": shm[0],
                "total": shm[1],
                "staging": [
                    {"bytes": size, "procs": count, "path": path}
                    for size, count, path in shm[2]
                ],
            },
            "disk": {
                "root": disk.root,
                "count": disk.count,
                "bytes": disk.bytes,
                "by_rank": disk.by_rank,
                "by_group": disk.by_group,
                "newest": disk.newest,
                "oldest": disk.oldest,
            },
            "log": health,
        }
        view = build_view(
            cfg,
            metrics,
            self.prev,
            gpus,
            shm,
            disk,
            health,
            self.url,
            args.interval,
            self.tick,
            time.time(),
        )
        with self.lock:
            self.snapshot, self.view = snapshot, view
        self.prev = metrics

    def _loop(self) -> None:
        while not self.stop_event.wait(self.args.interval):
            try:
                self.sample()
            except Exception as exc:  # keep serving whatever we have
                print(f"[panel] sample failed: {exc}", file=sys.stderr)

    def start(self) -> None:
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def latest(self) -> tuple[dict | None, dict | None]:
        with self.lock:
            return self.snapshot, self.view


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV Offload Monitor</title>
<style>
:root { color-scheme: dark; --bg:#12151a; --card:#191d24; --line:#272d38;
        --fg:#d8dee9; --dim:#8b95a5; --ok:#5fb37a; --warn:#d9a441; --bad:#d96a5f;
        --accent:#5aa9e6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
header { position:sticky; top:0; z-index:2; display:flex; gap:18px;
         align-items:baseline; flex-wrap:wrap; padding:10px 16px;
         background:rgba(18,21,26,.96); border-bottom:1px solid var(--line); }
h1 { margin:0; font-size:15px; letter-spacing:.4px; }
h2 { margin:0 0 8px; font-size:11px; letter-spacing:1px; color:var(--dim);
     text-transform:uppercase; }
.headline { color:var(--accent); font-weight:600; }
.meta { color:var(--dim); }
main { display:grid; gap:14px; padding:14px 16px 8px;
       grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
       align-items:start; }
section { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:12px 14px; }
table { width:100%; border-collapse:collapse; }
td { padding:2px 0; vertical-align:top; }
td.k { color:var(--dim); white-space:nowrap; padding-right:14px; }
td.v { text-align:right; white-space:pre-wrap; word-break:break-word; }
.bar-head { display:flex; justify-content:space-between; color:var(--dim);
            margin-top:8px; }
.bar { height:8px; margin:4px 0; border:1px solid var(--line); border-radius:4px;
       background:#0d1014; overflow:hidden; }
.bar > i { display:block; height:100%; background:var(--accent); width:0; }
.bar.ok > i { background:var(--ok); } .bar.warn > i { background:var(--warn); }
.bar.bad > i { background:var(--bad); }
.bar-sub { color:var(--dim); }
.err { color:var(--bad); } .stale { color:var(--warn); }
.scroll { max-height:46vh; overflow:auto; border-top:1px solid var(--line);
          margin-top:8px; }
.scroll td { font-size:12px; color:var(--dim); }
pre { margin:0; white-space:pre-wrap; word-break:break-all; color:var(--dim); }
footer { padding:6px 16px 24px; color:var(--dim); font-size:12px; }
@media (max-width:720px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>KV Offload Monitor</h1>
  <span class="headline" id="headline">…</span>
  <span class="meta" id="endpoint">…</span>
  <span class="meta" id="clock"></span>
  <label class="meta"><input type="checkbox" id="pause"> 暂停</label>
  <span class="meta" id="note"></span>
</header>
<main>
  <section><h2>Config</h2><table id="config"></table></section>
  <section><h2>Status</h2><div id="bars"></div><div id="status"></div></section>
  <section><h2>Chunks on disk</h2><div class="bar-sub" id="chunks-summary">…</div>
    <div class="scroll"><table id="chunks"></table></div></section>
  <section><h2>Log</h2><pre id="log">…</pre></section>
</main>
<footer id="footer"></footer>
<script>
"use strict";
const el = (id) => document.getElementById(id);
let paused = false, period = 5000;

const esc = (s) => String(s ?? "?").replace(/[&<>]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const pairRows = (pairs) => pairs.map(([k, v]) =>
  `<tr><td class="k">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`).join("");

function barHtml(b) {
  const frac = (b.frac === null || b.frac === undefined) ? null
             : Math.max(0, Math.min(1, b.frac));
  const cls = frac === null ? "" : frac > 0.9 ? "bad" : frac > 0.7 ? "warn" : "ok";
  const width = frac === null ? 0 : (100 * frac).toFixed(2);
  return `<div class="bar-head"><span>${esc(b.label)}</span>` +
         `<span>${frac === null ? "?" : (100 * frac).toFixed(1) + "%"}</span></div>` +
         `<div class="bar ${cls}"><i style="width:${width}%"></i></div>` +
         `<div class="bar-sub">${esc(b.text)}</div>`;
}

async function tick() {
  if (paused) return;
  try {
    const resp = await fetch("/api/view", { cache: "no-store" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const v = await resp.json();
    period = Math.max(1000, (v.interval || 5) * 1000);
    el("headline").textContent = v.headline || "";
    el("endpoint").textContent = v.endpoint;
    el("clock").textContent = `更新 ${v.updated} · tick ${v.tick}`;
    el("note").textContent = v.note || "";
    el("note").className = v.note ? "err" : "meta";
    el("config").innerHTML = pairRows(v.config);
    el("bars").innerHTML = v.bars.map(barHtml).join("");
    el("status").innerHTML = v.tables.map((t) =>
      `<div class="bar-sub" style="margin-top:8px">${esc(t.title)}</div>` +
      `<table>${pairRows(t.rows)}</table>`).join("");
    el("chunks-summary").textContent = v.chunks.summary;
    el("chunks").innerHTML = v.chunks.rows.map((r) =>
      `<tr><td class="k">${esc(r[0])}</td><td class="v">${esc(r[1])}</td>` +
      `<td class="k">${esc(r[2])}</td></tr>`).join("");
    el("log").textContent = v.log;
    el("footer").textContent = v.footer;
  } catch (e) {
    el("note").textContent = "获取失败: " + e.message;
    el("note").className = "err";
  } finally {
    setTimeout(tick, period);
  }
}

el("pause").addEventListener("change", (e) => {
  paused = e.target.checked;
  if (!paused) tick();
});
tick();
</script>
</body>
</html>
"""


SELF_TEST_METRICS = """\
vllm:cache_config_info{block_size="1600",cache_dtype="fp8_e4m3",gpu_memory_utilization="0.9",kv_cache_layout="None",kv_cache_memory_bytes="1000",kv_cache_size_tokens="1000",mamba_block_size="16",num_gpu_blocks="10"} 1.0
vllm:kv_cache_usage_perc 0.5
vllm:num_requests_running 1
vllm:num_requests_waiting 0
vllm:prefix_cache_queries_total 100
vllm:prefix_cache_hits_total 40
vllm:external_prefix_cache_queries_total 100
vllm:external_prefix_cache_hits_total 60
vllm:prompt_tokens_cached_total 7
vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 1024
vllm:kv_offload_total_time_total{transfer_type="GPU_to_CPU"} 1.0
vllm:kv_offload_size_count{transfer_type="GPU_to_CPU"} 2
vllm:kv_offload_size_sum{transfer_type="GPU_to_CPU"} 1024
vllm:kv_offload_cpu_cache_usage_perc 0.25
vllm:kv_offload_tiering_chunk_queries_total{tier="1:fs"} 10
vllm:kv_offload_tiering_chunk_hits_total{tier="1:fs"} 7
vllm:kv_offload_tiering_fs_used_bytes 2048
"""


def self_test() -> int:
    """Render the view from stub data: catches attribute typos and shape drift."""
    from monitor_kv_offload import DiskChunks, ServerConfig

    cfg = ServerConfig(
        pid=1, model="/m", served="s", max_model_len=1600, block_size=1600,
        kv_cache_bytes=1000, pool_tokens=1000, gpu_blocks=10, max_num_seqs="1",
        max_num_batched_tokens="1024", gpu_mem_util="0.9", dtype="half",
        kv_dtype="fp8_e4m3", tp="2", devices="0,1", engine_id="e",
        spec_name="TieringOffloadingSpec", load_failure="recompute", eviction="lru",
        cpu_bytes=1000, fs_root="/tmp/ssd", fs_max_bytes=2048,
        mamba_block_size="16", cache_layout="None",
    )
    cfg.chain_chunks, cfg.chunk_bytes = 1, 1000
    now = time.time()
    disk = DiskChunks(
        root="/tmp/ssd", count=2, bytes=2048, by_rank={"r0": 2},
        by_group={"g0": 2}, newest=now, oldest=now - 60,
        files=[(now, 1024, "a/b_g0/deadbeef")],
    )
    view = build_view(
        cfg, Metrics(SELF_TEST_METRICS), None, [(0, 1, 2)],
        (10, 20, [(5, 1, "/dev/shm/vllm_offload_e.mmap")]),
        disk, {"path": "log", "errors": 0, "last": "-"}, "http://x", 5.0, 1, now,
    )
    empty = build_view(
        cfg, None, None, [], (0, 0, []), DiskChunks(), None, "http://x", 5.0, 1, now
    )
    # Throughput needs two scrapes: 100 -> 300 generation tokens in 5s = 40/s.
    before = Metrics(
        "vllm:generation_tokens_total 100\n"
        "vllm:prompt_tokens_total 10\n"
    )
    after = Metrics(
        "vllm:generation_tokens_total 300\n"
        "vllm:prompt_tokens_total 10\n"
        "vllm:request_generation_tokens_count 2\n"
        "vllm:request_generation_tokens_sum 200\n"
        "vllm:request_decode_time_seconds_count 2\n"
        "vllm:request_decode_time_seconds_sum 4\n"
        "vllm:spec_decode_num_draft_tokens_total 60\n"
        "vllm:spec_decode_num_accepted_tokens_total 30\n"
    )
    rate_view = build_view(
        cfg, after, before, [(0, 1, 2)],
        (10, 20, [(5, 1, "/dev/shm/vllm_offload_e.mmap")]),
        disk, {"path": "log", "errors": 0, "last": "-"}, "http://x", 5.0, 2, now,
    )
    def table_of(view_, title):
        return {t["title"]: t for t in view_["tables"]}[title]

    rates = dict(table_of(rate_view, "Throughput")["rows"])
    live_row = rates["decode (live, 5s window)"]
    first_live = table_of(view, "Throughput")["rows"][0][1]

    checks = [
        ("config_rows", len(view["config"]) >= 12),
        ("throughput_live", "40.0 tok/s" in live_row and "200 tok in window" in live_row),
        ("throughput_avg", "50.0 tok/s" in rates["decode (finished requests)"]),
        ("throughput_mtp", "50.0%" in rates["spec decode (MTP)"]),
        ("headline", rate_view["headline"] == "decode 40.0 tok/s"),
        # With no previous scrape there is no window rate to report.
        ("throughput_first_sample", first_live.startswith("—") and "tok/s" not in first_live),
        ("bars", [b["label"] for b in view["bars"]] == [
            "GPU KV cache", "CPU (primary) tier", "fs (disk) tier quota"]),
        ("gpu_frac", round(view["bars"][0]["frac"], 3) == 0.5),
        ("tables", len(view["tables"]) == 6),
        (
            "external_hits",
            "60 / 100" in dict(table_of(view, "Cache")["rows"])["external (offload)"],
        ),
        ("tier_row", table_of(view, "Tiering")["rows"][0][0] == "1:fs"),
        ("chunk_row", view["chunks"]["rows"][0][1] == "1.0 KiB"),
        ("degraded_note", bool(empty["note"]) and empty["tables"] == []),
        ("page_self_contained", "http://" not in PAGE and "https://" not in PAGE
         and 'fetch("/api/view"' in PAGE),
    ]
    ok = True
    for name, good in checks:
        ok = ok and good
        print(f"  self-test {name}: {'PASS' if good else 'FAIL'}")
    return 0 if ok else 1


class Handler(BaseHTTPRequestHandler):
    server_version = "kv-offload-panel/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict) -> None:
        self._send(
            200,
            "application/json; charset=utf-8",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = urlsplit(self.path).path.rstrip("/") or "/"
        sampler: Sampler = self.server.sampler  # type: ignore[attr-defined]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/view":
            _, view = sampler.latest()
            if view is None:
                self._send(503, "application/json", b'{"error":"no sample yet"}')
            else:
                self._json(view)
        elif path == "/api/snapshot":
            snapshot, _ = sampler.latest()
            if snapshot is None:
                self._send(503, "application/json", b'{"error":"no sample yet"}')
            else:
                self._json(snapshot)
        elif path == "/healthz":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read-only browser panel for KV offload / tiering state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    ap.add_argument("--port", type=int, default=8199, help="serving port (default 8199)")
    ap.add_argument("--vllm-port", type=int, default=8000, help="vLLM port to observe")
    ap.add_argument("--vllm-url", default=None, help="vLLM base url (overrides port)")
    ap.add_argument("--pid", type=int, default=None, help="engine pid (default: by port)")
    ap.add_argument("-d", "--interval", type=float, default=5.0, help="sampling seconds")
    ap.add_argument("--chunk-lines", type=int, default=60, help="disk chunk rows to expose")
    ap.add_argument("--ssd-root", default=None, help="disk tier root (default: from config)")
    ap.add_argument("--log", default=None, help="server log path/glob for an error count")
    ap.add_argument("--verbose", action="store_true", help="log every HTTP request")
    ap.add_argument("--self-test", action="store_true", help="check the view builder")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    sampler = Sampler(args)
    sampler.start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.sampler = sampler  # type: ignore[attr-defined]
    httpd.verbose = args.verbose  # type: ignore[attr-defined]

    shown = "127.0.0.1" if args.host in ("", "0.0.0.0") else args.host
    print(f"[panel] http://{shown}:{args.port}/  observing {sampler.url}"
          f" every {args.interval:g}s (Ctrl-C to stop)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("[panel] WARNING: bound beyond localhost; the page has no"
              " authentication and exposes local paths. Prefer an SSH tunnel.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.shutdown()
        sampler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
