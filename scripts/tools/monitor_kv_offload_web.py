#!/usr/bin/env python3
"""Browser panel for the KV-offload data the terminal monitor collects.

The terminal monitor prints text frames: ideal for --once, --json, --append and
for grepping, but a character grid is a poor surface for *looking* at the state.
A browser brings mouse wheel, resize and scrolling for free, and cards and charts
cost nothing to draw. The genuinely hard part -- collecting the numbers -- is
identical either way, so this imports the collector instead of duplicating it and
serves one live page over HTTP.

Usage:
  monitor_kv_offload_web.py                        # http://127.0.0.1:8100
  monitor_kv_offload_web.py --port 9000 -d 2
  monitor_kv_offload_web.py --vllm-port 8001
  monitor_kv_offload_web.py --host 0.0.0.0         # LAN, no authentication
  monitor_kv_offload_web.py --start                # background; --stop to end it
  monitor_kv_offload_web.py --status               # pid / liveness / /healthz
  monitor_kv_offload_web.py --stop                 # SIGTERM the panel on --port
  curl -s localhost:8100/api/snapshot | python3 -m json.tool

--start forks into the background (``setsid`` plus stdout/stderr redirection),
writes ``~/.cache/kv-offload-panel/panel-<port>.pid`` and appends stdout/stderr
to ``panel-<port>.log`` in the same directory; --status and --stop read that
pidfile, so stopping never has to pattern-match command lines. Without --start
it runs in the foreground and Ctrl-C ends it -- the pidfile is still written so
--stop works either way.

Endpoints:
  GET /               the panel (inline CSS/JS, no external resources)
  GET /api/view       the tables and charts the page renders
  GET /api/snapshot   raw sample, same shape as the terminal --json output
  GET /healthz        liveness

Dependency-free (stdlib only) and read-only apart from its own pidfile/log: it
only issues HTTP GETs to the vLLM server, reads /proc, /dev/shm, /proc/meminfo,
``nvidia-smi`` and the disk tier's files. It binds to localhost by default
because the payload contains local paths; ``--host 0.0.0.0`` is opt-in and
unauthenticated, so an SSH tunnel is the better way to reach it remotely.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import signal
import sys
import threading
import time
import urllib.request
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
)

# --------------------------------------------------------------------------
# process state: pidfile/log per serving port, so --stop never guesses
# --------------------------------------------------------------------------

# The panel serves one port per instance, so state is keyed by that port: two
# panels (8100, 9000) coexist and --stop targets exactly one. Kept outside the
# repository so the daemon leaves no tracked files behind.
STATE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kv-offload-panel")
DEFAULT_PORT = 8100


def _state_paths(port: int) -> tuple[str, str]:
    """Return (pidfile, logfile) for a serving port."""
    return (
        os.path.join(STATE_DIR, f"panel-{port}.pid"),
        os.path.join(STATE_DIR, f"panel-{port}.log"),
    )


def _read_pid(pidfile: str) -> int | None:
    try:
        with open(pidfile) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but not ours; harmless for liveness
        return True
    return True


def _argv_port(argv: list[str]) -> int:
    """The serving port an argv implies, mirroring argparse's own default.

    A panel started without --port gets DEFAULT_PORT, so the guard must accept
    that too -- requiring a literal ``--port`` in argv would make `--stop` fail
    for every default-port instance. Both ``--port N`` and ``--port=N`` count.
    """
    port = DEFAULT_PORT
    for index, token in enumerate(argv):
        value = None
        if token == "--port" and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith("--port="):
            value = token.split("=", 1)[1]
        if value is not None and value.lstrip("-").isdigit():
            port = int(value)
    return port


def _pid_is_ours(pid: int, port: int) -> bool:
    """Refuse a stale pidfile: the pid must be this script serving this port.

    A pidfile can outlive its process (crash, reboot, manual rm of the log) and
    a recycled pid would then be somebody else's process. /proc is the only
    authority on what a pid currently is, so the command line is checked.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return False
    argv = raw.replace(b"\0", b" ").decode("utf-8", "replace").split()
    if not argv or "python" not in os.path.basename(argv[0]):
        return False
    me = os.path.basename(os.path.abspath(__file__))
    if not any(os.path.basename(tok) == me for tok in argv[1:]):
        return False
    return _argv_port(argv) == port


def stop_panel(port: int) -> int:
    """SIGTERM the panel started with --start on this port.

    Exit codes follow stop_server.sh: 0 stopped, 1 found but trouble, 2 nothing
    to stop -- so a wrapper can tell "was not running" from "failed to stop".
    """
    pidfile, logfile = _state_paths(port)
    pid = _read_pid(pidfile)
    if not pid:
        print(f"[panel] nothing to stop: no pidfile at {pidfile}")
        return 2
    if not _pid_alive(pid):
        print(f"[panel] pid {pid} not alive; removing stale pidfile")
        try:
            os.remove(pidfile)
        except OSError:
            pass
        return 2
    if not _pid_is_ours(pid, port):
        print(f"[panel] refusing: pid {pid} is not this panel on port {port}")
        return 1
    print(f"[panel] stopping pid {pid} (port {port})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(50):
        if not _pid_alive(pid):
            try:
                os.remove(pidfile)
            except OSError:
                pass
            print(f"[panel] stopped pid {pid}")
            return 0
        time.sleep(0.2)
    print(f"[panel] pid {pid} still alive after 10s; check {logfile}")
    return 1


def status_panel(port: int, host: str) -> int:
    """Report the pid, whether it is alive, and whether /healthz answers."""
    pidfile, logfile = _state_paths(port)
    pid = _read_pid(pidfile)
    if pid and _pid_alive(pid):
        print(f"[panel] running: pid {pid}  port {port}  log {logfile}")
    elif pid:
        print(f"[panel] stale pidfile: pid {pid} not alive ({pidfile})")
    else:
        print(f"[panel] not running (no pidfile at {pidfile})")
    probe = "127.0.0.1" if host in ("", "0.0.0.0") else host
    url = f"http://{probe}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            print(f"[panel] {url} -> HTTP {resp.status}")
    except Exception as exc:  # noqa: BLE001 (any failure means "not answering")
        print(f"[panel] {url} unreachable: {exc}")
        return 1
    return 0


def _daemonize(port: int) -> None:
    """Detach into the background for --start.

    In the parent this never returns (it exits after the child has published its
    pid); in the child it returns so main() carries on. Called *before* any
    thread exists, so the fork cannot strand a lock in a half-cloned state.
    """
    if not hasattr(os, "fork"):
        print("[panel] --start needs POSIX fork; running in the foreground")
        return
    pidfile, logfile = _state_paths(port)
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.fork() > 0:
        deadline = time.time() + 10
        while time.time() < deadline:
            child = _read_pid(pidfile)
            if child and _pid_alive(child):
                print(f"[panel] started (pid {child})  log {logfile}")
                raise SystemExit(0)
            time.sleep(0.1)
        print(f"[panel] failed to start; check {logfile}", file=sys.stderr)
        raise SystemExit(1)
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    handle = open(logfile, "ab", buffering=0)
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)


# --------------------------------------------------------------------------
# view: the same numbers the terminal panel shows, as data
# --------------------------------------------------------------------------


# How many samples the timeline keeps. At the default 5 s tick this is a 20
# minute window; the browser redraws the whole buffer each refresh, so the cost
# stays flat as history grows.
HISTORY_POINTS = 240


def history_point(metrics, prev, cfg, interval, elapsed):
    """One timeline row, or None while /metrics is unreachable.

    Levels come from the current scrape; rates are counter deltas over one
    interval, so the first sample (no ``prev``) carries levels only. Keeping the
    shape identical to ``build_view`` means the curves and the tables cannot
    disagree about what "decode tok/s" means.
    """
    if metrics is None:
        return None

    def level(name, **labels):
        return metrics.get(name, **labels)

    def rate(name, **labels):
        if prev is None:
            return None
        now = metrics.counter(name, **labels)
        before = prev.counter(name, **labels)
        if now is None or before is None:
            return None
        return max(now - before, 0.0) / interval

    fs_used = level("vllm:kv_offload_tiering_fs_used_bytes")
    return {
        "t": round(elapsed, 1),
        "gen": rate("vllm:generation_tokens_total"),
        "prompt": rate("vllm:prompt_tokens_total"),
        "gpu": level("vllm:kv_cache_usage_perc"),
        "cpu": level("vllm:kv_offload_cpu_cache_usage_perc"),
        "fs": (fs_used / cfg.fs_max_bytes)
        if (fs_used is not None and cfg.fs_max_bytes)
        else None,
        "store": rate("vllm:kv_offload_total_bytes", transfer_type="GPU_to_CPU"),
        "load": rate("vllm:kv_offload_total_bytes", transfer_type="CPU_to_GPU"),
        "running": level("vllm:num_requests_running"),
        "waiting": level("vllm:num_requests_waiting"),
    }


def _chunk_count(nbytes: int | None, cfg) -> str:
    if not nbytes or not cfg.chunk_bytes:
        return "?"
    count = nbytes / cfg.chunk_bytes
    if cfg.chain_chunks:
        return f"{count:,.0f} chunks = {count / cfg.chain_chunks:.2f} chain"
    return f"{count:,.0f} chunks"


def build_view(cfg, metrics, prev, gpus, shm, disk, health, url, interval, tick, updated,
               history=None, started=None):
    """Everything the page needs, pre-formatted (so the JS stays dumb)."""

    def counter(name, **labels):
        return metrics.counter(name, **labels) if metrics is not None else None

    def delta(name, **labels):
        now = counter(name, **labels) or 0.0
        before = prev.counter(name, **labels) if prev is not None else None
        return now - (before or 0.0)

    def hist(name, **labels):
        return metrics.hist(name, **labels) if metrics is not None else (None, None)

    def hold_pct(x: float | None) -> str:
        return "?" if x is None or not math.isfinite(x) else f"{100.0 * x:.1f}%"

    note = ""
    tables: list[dict] = []
    chart_legend: dict = {}
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

        cpu_usage = metrics.get("vllm:kv_offload_cpu_cache_usage_perc")
        cpu_write = metrics.get("vllm:kv_offload_cpu_cache_write_usage_perc")
        cpu_read = metrics.get("vllm:kv_offload_cpu_cache_read_usage_perc")
        fs_used = metrics.get("vllm:kv_offload_tiering_fs_used_bytes")

        # The GPU KV chart draws the percentage; the legend carries the absolute
        # usage a percentage alone does not say. CPU/fs tier usage lives in the
        # Resources table, not on a chart.
        if cfg.pool_tokens:
            chart_legend["gpu"] = (
                f"{fmt_int((metrics.get('vllm:kv_cache_usage_perc') or 0) * cfg.pool_tokens)}"
                f" / {fmt_int(cfg.pool_tokens)} tok"
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
        # live rate is the counter delta over one sampling interval (drawn by
        # the timeline charts); the first sample has no previous scrape, and
        # dividing the whole history by the interval would be nonsense.
        window_gen = delta("vllm:generation_tokens_total")
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
                "CPU tier",
                f"{fmt_bytes((cpu_usage or 0) * (cfg.cpu_bytes or 0))} / {fmt_bytes(cfg.cpu_bytes)}"
                f"   write-hold {hold_pct(cpu_write)}   read-hold {hold_pct(cpu_read)}",
            ),
            (
                "fs tier",
                f"{fmt_bytes(fs_used)} / {fmt_bytes(cfg.fs_max_bytes)}"
                f"   files {disk.count} ({fmt_bytes(disk.bytes)})"
                f"   evictions {fmt_int(counter('vllm:kv_offload_tiering_fs_evictions') or 0)}"
                f"   skipped {fmt_bytes(counter('vllm:kv_offload_tiering_fs_skipped_store_bytes') or 0)}",
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
        "started": started,
        "note": note,
        "headline": headline,
        "config": [(str(k), str(v)) for k, v in config],
        "chart_legend": chart_legend,
        "tables": tables,
        "history": list(history or []),
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
        # Written only by the sampling thread and copied into each view, so the
        # HTTP threads never observe a half-updated buffer.
        self.history: deque = deque(maxlen=HISTORY_POINTS)

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
        point = history_point(
            metrics, self.prev, cfg, args.interval, time.time() - self.started
        )
        if point is not None:
            self.history.append(point)
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
            history=self.history,
            started=self.started,
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
       grid-template-columns:1fr;
       align-items:start; }
section { background:var(--card); border:1px solid var(--line);
          border-radius:8px; padding:12px 14px; }
table { width:100%; border-collapse:collapse; }
td { padding:2px 0; vertical-align:top; }
td.k { color:var(--dim); white-space:nowrap; padding-right:14px; }
td.v { text-align:right; white-space:pre-wrap; word-break:break-word; }
 .bar-sub { color:var(--dim); }
 .err { color:var(--bad); } .stale { color:var(--warn); }
 pre { margin:0; white-space:pre-wrap; word-break:break-all; color:var(--dim); }
 /* Config halves and the Status card's two columns: side by side on desktop,
    stacked on phones. */
 .cfg-2col { display:grid; grid-template-columns:1fr 1fr; gap:0 28px; }
 .status-2col { display:grid; grid-template-columns:1fr 1fr; gap:0 28px; }
 .tbl-title { color:var(--dim); font-size:11px; letter-spacing:1px;
              text-transform:uppercase; margin:12px 0 4px; }
 .status-half > .tbl-title:first-child { margin-top:0; }
 @media (max-width:720px) { .cfg-2col, .status-2col { grid-template-columns:1fr; }
                            .chart-grid { grid-template-columns:1fr; } }
 footer { padding:6px 16px 24px; color:var(--dim); font-size:12px; }
 #charts { grid-column:1/-1; }
 .chart-grid { display:grid; gap:12px;
               grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }
 .chart-head { display:flex; justify-content:space-between; gap:14px;
               color:var(--dim); font-size:13px; align-items:baseline; }
 .chart-head > span:first-child { font-size:14px; color:var(--fg); }
 .chart-live { text-align:right; white-space:nowrap; overflow:hidden; }
 .chart canvas { display:block; width:100%; height:150px; margin-top:6px;
                 background:#0d1014; border:1px solid var(--line); border-radius:4px; }
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
  <section id="cfg"><h2>Config</h2><div id="config" class="cfg-2col"></div></section>
  <section id="status"><h2>Status</h2>
    <div class="status-2col">
      <div class="status-half">
        <div class="tbl-title">Throughput</div><table id="t-throughput"></table>
        <div class="tbl-title">Requests</div><table id="t-requests"></table>
        <div class="tbl-title">Cache</div><table id="t-cache"></table>
      </div>
      <div class="status-half">
        <div class="tbl-title">Tiering</div><table id="t-tiering"></table>
        <div class="tbl-title">Resources</div><table id="t-resources"></table>
      </div>
    </div>
  </section>
  <section id="charts"><h2>Timeline</h2><div class="chart-grid">
    <div class="chart"><div class="chart-head"><span>Decode (tok/s)</span><span class="chart-live" id="legend-decode"></span></div><canvas id="chart-decode"></canvas></div>
    <div class="chart"><div class="chart-head"><span>Prefill (tok/s)</span><span class="chart-live" id="legend-prefill"></span></div><canvas id="chart-prefill"></canvas></div>
    <div class="chart"><div class="chart-head"><span>GPU KV cache</span><span class="chart-live" id="legend-gpu"></span></div><canvas id="chart-gpu"></canvas></div>
  </div></section>
</main>
<footer id="footer"></footer>
<script>
"use strict";
const el = (id) => document.getElementById(id);
let paused = false, period = 5000, lastHistory = [], lastStarted = null, lastDetail = {};

const esc = (s) => String(s ?? "?").replace(/[&<>]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const pairRows = (pairs) => pairs.map(([k, v]) =>
  `<tr><td class="k">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`).join("");

const CHARTS = [
  { id: "chart-decode", legend: "legend-decode", percent: false,
    series: [{ key: "gen", name: "decode", color: "#5aa9e6" }] },
  { id: "chart-prefill", legend: "legend-prefill", percent: false,
    series: [{ key: "prompt", name: "prefill", color: "#5fb37a" }] },
  { id: "chart-gpu", legend: "legend-gpu", percent: true, detail: "gpu",
    series: [{ key: "gpu", name: "GPU KV", color: "#5aa9e6" }] },
];

function valFmt(v, percent) {
  if (v === null || v === undefined) return "—";
  if (percent) return (100 * v).toFixed(0) + "%";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(1) + "G";
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(1);
}

const pad2 = (n) => String(n).padStart(2, "0");
const fmtClock = (ms) => {
  const d = new Date(ms);
  return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
};

// Round the data max up to a clean axis bound (1/2/5 x 10^k).
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

// Hand-drawn sparkline: one canvas per chart, x is sample index, nulls break
// the line so a stalled scrape shows a gap instead of a fake straight edge.
function drawChart(chart, history, started, detail) {
  const cv = el(chart.id);
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = cv.clientHeight || 150;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 48, padR = 10, padT = 8, padB = 22;
  const iw = Math.max(1, w - padL - padR), ih = Math.max(1, h - padT - padB);
  let max = chart.percent ? 1 : 0;
  if (!chart.percent) {
    for (const s of chart.series)
      for (const p of history) {
        const v = p[s.key];
        if (v !== null && v !== undefined) max = Math.max(max, v);
      }
  }
  if (!(max > 0)) max = 1;
  if (!chart.percent) max = niceMax(max);

  ctx.font = "12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";
  ctx.strokeStyle = "#272d38"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8b95a5"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let g = 0; g <= 2; g++) {
    const val = max * (2 - g) / 2;
    const y = Math.round(padT + ih * g / 2) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + iw, y); ctx.stroke();
    ctx.fillText(valFmt(val, chart.percent), padL - 6, y);
  }

  const n = history.length;
  const xAt = (i) => padL + (n <= 1 ? 0 : iw * i / (n - 1));
  const yAt = (v) => padT + ih * (1 - Math.min(Math.max(v, 0), max) / max);
  for (const s of chart.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const v = history[i][s.key];
      if (v === null || v === undefined) { pen = false; continue; }
      const x = xAt(i), y = yAt(v);
      if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
    }
    ctx.stroke();
  }

  // X axis: wall-clock time of the first and last sample (started = panel epoch).
  if (started && n > 1) {
    ctx.fillStyle = "#8b95a5"; ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(fmtClock((started + history[0].t) * 1000), padL, padT + ih + 6);
    ctx.textAlign = "right";
    ctx.fillText(fmtClock((started + history[n - 1].t) * 1000), padL + iw, padT + ih + 6);
  }

  const legend = el(chart.legend);
  if (legend) {
    legend.innerHTML = chart.series.map((s) => {
      let last;
      for (let i = history.length - 1; i >= 0; i--) {
        const v = history[i][s.key];
        if (v !== null && v !== undefined) { last = v; break; }
      }
      return `<span style="color:${s.color}">${esc(s.name)} ` +
             `${esc(valFmt(last, chart.percent))}</span>`;
    }).join("   ");
    if (chart.detail && detail && detail[chart.detail])
      legend.innerHTML += ` <span>· ${esc(detail[chart.detail])}</span>`;
  }
}

function drawCharts(history, started, detail) {
  for (const chart of CHARTS) drawChart(chart, history || [], started, detail || {});
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
    el("config").innerHTML = (() => {
      const rows = v.config, mid = Math.ceil(rows.length / 2);
      return `<table>${pairRows(rows.slice(0, mid))}</table>` +
             `<table>${pairRows(rows.slice(mid))}</table>`;
    })();
    lastHistory = v.history || [];
    lastStarted = v.started ?? null;
    lastDetail = v.chart_legend || {};
    const TABLE_IDS = { "Throughput":"t-throughput", "Requests":"t-requests",
                        "Cache":"t-cache", "Tiering":"t-tiering", "Resources":"t-resources" };
    v.tables.forEach((t) => {
      const id = TABLE_IDS[t.title];
      if (id) el(id).innerHTML = pairRows(t.rows);
    });
    el("footer").textContent = v.footer;
    drawCharts(lastHistory, lastStarted, lastDetail);
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
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => drawCharts(lastHistory, lastStarted, lastDetail), 150);
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
        started=now - 10,
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

    point = history_point(after, before, cfg, 5.0, 12.0)
    rates = dict(table_of(rate_view, "Throughput")["rows"])
    first_row = table_of(view, "Throughput")["rows"][0][1]
    titles = [t["title"] for t in view["tables"]]
    res_rows = dict(table_of(view, "Resources")["rows"])
    req_rows = dict(table_of(view, "Requests")["rows"])

    checks = [
        ("config_rows", len(view["config"]) >= 12),
        ("throughput_avg", "50.0 tok/s" in rates["decode (finished requests)"]),
        ("throughput_mtp", "50.0%" in rates["spec decode (MTP)"]),
        ("headline", rate_view["headline"] == "decode 40.0 tok/s"),
        # With no previous scrape there is no window rate to report.
        ("throughput_first_sample", first_row.startswith("—") and "tok/s" not in first_row),
        # No occupancy bars; only the GPU KV chart keeps a legend. CPU/fs tier
        # usage and request counts live in the tables, not on a chart.
        ("no_bars", "bars" not in view and "chart_legend" in view),
        ("chart_legend_gpu", view["chart_legend"].get("gpu") == "500 / 1,000 tok"),
        ("no_chart_legend_cpu_fs", "cpu" not in view["chart_legend"] and "fs" not in view["chart_legend"]),
        ("req_running", req_rows.get("running") == "1"),
        ("req_waiting", req_rows.get("waiting") == "0"),
        ("cpu_tier_row", res_rows.get("CPU tier", "").startswith("250 B / 1000 B")),
        ("fs_tier_row", res_rows.get("fs tier", "").startswith("2.0 KiB / 2.0 KiB")),
        ("no_transfers_table", "Transfers (connector)" not in titles),
        ("tables", len(view["tables"]) == 5),
        (
            "external_hits",
            "60 / 100" in dict(table_of(view, "Cache")["rows"])["external (offload)"],
        ),
        ("tier_row", table_of(view, "Tiering")["rows"][0][0] == "1:fs"),
        ("chunk_row", view["chunks"]["rows"][0][1] == "1.0 KiB"),
        # Timeline: same delta the Throughput table reports, levels absent from
        # the stub scrape stay null, and no scrape means no point at all.
        ("history_rate", point["gen"] == 40.0),
        ("history_level", point["gpu"] is None and point["store"] is None),
        ("history_none", history_point(None, None, cfg, 5.0, 0.0) is None),
        ("history_default", view["history"] == []),
        ("view_started", view["started"] == now - 10),
        ("degraded_note", bool(empty["note"]) and empty["tables"] == []),
        (
            "page_self_contained",
            "http://" not in PAGE and "https://" not in PAGE
            and 'fetch("/api/view"' in PAGE
            and "chart-decode" in PAGE and "chart-gpu" in PAGE
            and "chart-cpu" not in PAGE and "chart-fs" not in PAGE
            and "chart-requests" not in PAGE and "chart-offload" not in PAGE
            and "drawCharts(lastHistory, lastStarted, lastDetail)" in PAGE,
        ),
        # Status is one card again, its five tables in two internal columns
        # (Throughput/Requests/Cache left, Tiering/Resources right).
        (
            "status_single_card",
            '<h2>Status</h2>' in PAGE and 'id="status"' in PAGE
            and 'class="status-2col"' in PAGE
            and all(f'id="t-{name}"' in PAGE for name in
                    ("throughput", "requests", "cache", "tiering", "resources")),
        ),
        # Every card takes a full row; the Log and Chunks cards are removed
        # for now (the view still carries their data, only the cards are gone).
        ("full_width_rows",
            "grid-template-columns:1fr" in PAGE and 'id="log"' not in PAGE
            and 'id="chunks"' not in PAGE and "chunks-summary" not in PAGE),
        ("state_paths", _state_paths(8100)[0].endswith("panel-8100.pid")
         and _state_paths(8100)[1].endswith("panel-8100.log")),
        ("pidfile_missing", _read_pid(os.path.join(STATE_DIR, "does-not-exist.pid")) is None),
        # --stop must recognise an instance started with the default port, where
        # argv carries no --port at all; both flag spellings count.
        ("argv_port_default", _argv_port(["python3", "monitor_kv_offload_web.py"]) == DEFAULT_PORT),
        ("argv_port_split", _argv_port(["python3", "x.py", "--port", "9000"]) == 9000),
        ("argv_port_equals", _argv_port(["python3", "x.py", "--port=9001"]) == 9001),
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
        description="Browser panel for KV offload / tiering state (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"serving port (default {DEFAULT_PORT})")
    ap.add_argument("--vllm-port", type=int, default=8000, help="vLLM port to observe")
    ap.add_argument("--vllm-url", default=None, help="vLLM base url (overrides port)")
    ap.add_argument("--pid", type=int, default=None, help="engine pid (default: by port)")
    ap.add_argument("-d", "--interval", type=float, default=5.0, help="sampling seconds")
    ap.add_argument("--chunk-lines", type=int, default=60, help="disk chunk rows to expose")
    ap.add_argument("--ssd-root", default=None, help="disk tier root (default: from config)")
    ap.add_argument("--log", default=None, help="server log path/glob for an error count")
    ap.add_argument("--verbose", action="store_true", help="log every HTTP request")
    ap.add_argument("--start", action="store_true",
                    help="run in the background (pidfile/log under ~/.cache/kv-offload-panel)")
    ap.add_argument("--stop", action="store_true",
                    help="stop the --start daemon serving --port")
    ap.add_argument("--status", action="store_true",
                    help="show the --start daemon's pid and /healthz")
    ap.add_argument("--self-test", action="store_true", help="check the view builder")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.stop:
        return stop_panel(args.port)
    if args.status:
        return status_panel(args.port, args.host)

    # Detach *before* the sampler thread or the HTTP server exist, so the fork
    # cannot clone a half-initialised lock. The parent exits inside here.
    if args.start:
        _daemonize(args.port)

    pidfile, _ = _state_paths(args.port)
    os.makedirs(STATE_DIR, exist_ok=True)

    # Bind before publishing the pid: a port clash must not overwrite the
    # pidfile of the instance that already owns the port.
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"[panel] bind {args.host}:{args.port} failed: {exc}", file=sys.stderr)
        return 1
    sampler = Sampler(args)
    httpd.daemon_threads = True
    httpd.sampler = sampler  # type: ignore[attr-defined]
    httpd.verbose = args.verbose  # type: ignore[attr-defined]

    with open(pidfile, "w") as handle:
        handle.write(str(os.getpid()))

    def _shutdown(signum, frame) -> None:
        # shutdown() waits for serve_forever to return, and a signal handler runs
        # *on* the thread that is inside serve_forever -- so ask another thread.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    shown = "127.0.0.1" if args.host in ("", "0.0.0.0") else args.host
    print(f"[panel] http://{shown}:{args.port}/  observing {sampler.url}"
          f" every {args.interval:g}s (pid {os.getpid()}; --stop or Ctrl-C)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("[panel] WARNING: bound beyond localhost; the page has no"
              " authentication and exposes local paths. Prefer an SSH tunnel.")
    try:
        sampler.start()
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print()
    finally:
        # shutdown() is deliberately *not* called here: it blocks until
        # serve_forever returns, which would hang if we never got that far.
        sampler.stop()
        try:
            os.remove(pidfile)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
