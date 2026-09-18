#!/usr/bin/env python3
"""Live KV-offload / tiering monitor for this branch.

This branch keeps vLLM's offloading/tiering layer unmodified, so the panel is
assembled from what an upstream-shaped server actually exposes:

  * ``GET /metrics`` -- the scheduler's prefix / external-cache counters, the
    offloading connector's store / load bytes and histograms, and the tiering
    manager's per-tier series (``tier="0:primary"``, ``tier="1:fs"``, ...),
    including the disk tier's own quota counters.
  * ``/proc/<pid>/{cmdline,environ}`` -- the launch configuration: the profile
    script's ``--kv-transfer-config`` JSON (engine id, tier sizes, policies)
    plus the model / context / pool arguments.
  * ``vllm:cache_config_info`` -- the engine's resolved config (block size, pool
    tokens, cache dtype, mamba knobs).
  * local resources -- ``nvidia-smi``, ``/dev/shm`` (including the staging
    region the engine mmapped, which it unlinks and which is therefore
    invisible to ``ls``), ``/proc/meminfo``, and the disk tier's files.

Not available here: a per-chain SESSIONS block. Upstream has no
per-request inventory endpoint, so only aggregate occupancy exists for the GPU
and CPU tiers. The disk tier *is* introspectable, because it is
hash-addressed: the CHUNKS block lists the real parked chunks by rank/group.

Usage:
  monitor_kv_offload.py                    # :8000, 5s refresh
  monitor_kv_offload.py --port 8001 -d 2
  monitor_kv_offload.py --once
  monitor_kv_offload.py --json --count 5
  monitor_kv_offload.py --log 'logs/server_128k_RAMx1_SSDx4_*.log'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def fmt_int(n: float | None) -> str:
    return "?" if n is None else f"{int(n):,}"


def bar(frac: float | None, width: int = 10) -> str:
    if frac is None or not math.isfinite(frac):
        return "░" * width
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def pct(used: float | None, total: float | None) -> str:
    if not total:
        return "  ?  %"
    return f"{100.0 * used / total:5.1f}%"


def ratio_pct(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "  ?  %"
    return f"{100.0 * x:5.1f}%"


def rate_pct(hits: float | None, queries: float | None) -> str:
    if not queries:
        return "  ?  %"
    return f"{100.0 * hits / queries:5.1f}%"


def ago(epoch: float | None, now: float) -> str:
    if not epoch:
        return "?"
    secs = max(0.0, now - float(epoch))
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    return f"{int(secs // 3600)}h"


def short_chunk(rel: str, width: int = 14) -> str:
    """``hash`` -> ``hash[:width]…`` so a chunk line stays readable."""
    head, _, tail = rel.rpartition("/")
    if len(tail) <= width + 4:
        return rel
    return f"{head}/{tail[:width]}…" if head else f"{tail[:width]}…"


def hhmmss(epoch: float | None) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(epoch)))
    except (TypeError, ValueError):
        return "?"


def delta(cur: float | None, prev: float | None) -> float:
    if cur is None:
        return 0.0
    return cur - (prev or 0.0)


def throughput(nbytes: float | None, secs: float | None) -> str:
    if not nbytes or not secs:
        return "-"
    return f"{nbytes / secs / 2**30:.2f} GiB/s"


# --------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------

_SERIES_RE = re.compile(
    r"^(?P<name>vllm:[A-Za-z0-9_]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)\s*$"
)

INTERESTING_PREFIXES = (
    "vllm:kv_offload",
    "vllm:kv_cache",
    "vllm:prefix_cache",
    "vllm:external_prefix_cache",
    "vllm:prompt_tokens_cached",
    "vllm:num_requests",
    "vllm:request_success",
    "vllm:cache_config_info",
)


class Metrics:
    """The ``vllm:`` series of a ``/metrics`` scrape, queried by label subset."""

    def __init__(self, text: str) -> None:
        self.series: dict[str, list[tuple[dict[str, str], float]]] = {}
        for line in text.splitlines():
            if not line or line[0] == "#":
                continue
            m = _SERIES_RE.match(line)
            if not m:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            labels: dict[str, str] = {}
            for part in (m.group("labels") or "").split(","):
                if "=" not in part:
                    continue
                key, _, raw = part.partition("=")
                labels[key.strip()] = raw.strip().strip('"')
            self.series.setdefault(m.group("name"), []).append((labels, value))

    def get(self, name: str, **labels: str) -> float | None:
        found = False
        total = 0.0
        for series_labels, value in self.series.get(name, ()):
            if all(series_labels.get(k) == str(v) for k, v in labels.items()):
                total += value
                found = True
        return total if found else None

    def counter(self, name: str, **labels: str) -> float | None:
        """Prometheus appends ``_total`` to counters; accept either spelling."""
        for candidate in (name + "_total", name):
            value = self.get(candidate, **labels)
            if value is not None:
                return value
        return None

    def hist(self, name: str, **labels: str) -> tuple[float | None, float | None]:
        return self.get(name + "_count", **labels), self.get(name + "_sum", **labels)

    def labels_of(self, name: str, key: str) -> list[str]:
        values = set()
        for candidate in (name, name + "_total"):
            for labels, _ in self.series.get(candidate, ()):
                if key in labels:
                    values.add(labels[key])
        return sorted(values)

    def interesting(self) -> dict[str, list[dict]]:
        out = {}
        for name, rows in self.series.items():
            if not name.startswith(INTERESTING_PREFIXES):
                continue
            out[name] = [{"labels": labels, "value": value} for labels, value in rows]
        return out


def fetch_metrics(url: str) -> Metrics | None:
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as resp:
            return Metrics(resp.read().decode("utf-8", "ignore"))
    except Exception:
        return None


# --------------------------------------------------------------------------
# engine process / launch configuration
# --------------------------------------------------------------------------


def run(cmd: list[str], timeout: float = 5) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def find_pid(port: int) -> int | None:
    """The api-server process serving ``port`` (its cmdline holds the config).

    /proc is scanned directly rather than through ``pgrep -f``: a pattern match
    also hits any shell or log tail whose command line merely mentions the
    module, and mixing such a process's config with this port's metrics would be
    worse than showing no config at all. Use ``--pid`` for an instance that was
    launched without an explicit ``--port``.
    """
    for entry in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(entry.rsplit("/", 1)[1])
        except ValueError:
            continue
        argv = [a for a in read_proc(pid, "cmdline").split("\0") if a]
        if "vllm.entrypoints.openai.api_server" not in argv:
            continue
        if "python" not in os.path.basename(argv[0]):
            continue
        if "--port" not in argv:
            continue
        index = argv.index("--port")
        if index + 1 < len(argv) and argv[index + 1] == str(port):
            return pid
    return None


def read_proc(pid: int | None, name: str) -> str:
    if not pid:
        return ""
    try:
        with open(f"/proc/{pid}/{name}", "rb") as handle:
            return handle.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def parse_argv(raw: str) -> dict[str, str]:
    """``--flag value`` pairs and bare ``--flag`` switches from /proc cmdline."""
    argv = [a for a in raw.split("\0") if a]
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[token] = argv[i + 1]
                i += 2
                continue
            out[token] = "1"
        i += 1
    return out


def as_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


@dataclass
class ServerConfig:
    pid: int | None = None
    model: str = "?"
    served: str = "?"
    max_model_len: int | None = None
    block_size: int | None = None
    kv_cache_bytes: int | None = None
    pool_tokens: int | None = None
    gpu_blocks: int | None = None
    max_num_seqs: str = "?"
    max_num_batched_tokens: str = "?"
    gpu_mem_util: str = "?"
    dtype: str = "?"
    kv_dtype: str = "?"
    tp: str = "?"
    devices: str = "?"
    engine_id: str = "?"
    spec_name: str = "?"
    load_failure: str = "?"
    eviction: str = "?"
    cpu_bytes: int | None = None
    fs_root: str | None = None
    fs_max_bytes: int | None = None
    mamba_block_size: str = "?"
    cache_layout: str = "?"
    chain_chunks: int | None = None
    chunk_bytes: int | None = None
    extras: dict[str, str] = field(default_factory=dict)


def collect_config(pid: int | None, port: int, metrics: Metrics | None) -> ServerConfig:
    cfg = ServerConfig(pid=pid)
    raw = read_proc(pid, "cmdline")
    argv = parse_argv(raw)
    env = dict(
        line.split("=", 1)
        for line in read_proc(pid, "environ").split("\0")
        if "=" in line
    )

    if not argv and not env:
        cfg.extras["unavailable"] = (
            f"no api_server process with --port {port}; pass --pid if it was "
            "launched differently"
        )
        return cfg

    cfg.model = argv.get("--model", cfg.model)
    cfg.served = argv.get("--served-model-name", cfg.served)
    cfg.max_model_len = as_int(argv.get("--max-model-len"))
    cfg.kv_cache_bytes = as_int(argv.get("--kv-cache-memory-bytes"))
    cfg.max_num_seqs = argv.get("--max-num-seqs", cfg.max_num_seqs)
    cfg.max_num_batched_tokens = argv.get(
        "--max-num-batched-tokens", cfg.max_num_batched_tokens
    )
    cfg.gpu_mem_util = argv.get("--gpu-memory-utilization", cfg.gpu_mem_util)
    cfg.dtype = argv.get("--dtype", cfg.dtype)
    cfg.kv_dtype = argv.get("--kv-cache-dtype", cfg.kv_dtype)
    cfg.tp = argv.get("--tensor-parallel-size", cfg.tp)
    cfg.devices = argv.get("--device-ids", cfg.devices)

    transfer: dict = {}
    raw_transfer = argv.get("--kv-transfer-config")
    if raw_transfer:
        try:
            transfer = json.loads(raw_transfer)
        except ValueError:
            transfer = {}
    cfg.engine_id = transfer.get("engine_id", cfg.engine_id)
    cfg.load_failure = transfer.get("kv_load_failure_policy", cfg.load_failure)
    extra = transfer.get("kv_connector_extra_config") or {}
    cfg.spec_name = extra.get("spec_name", cfg.spec_name)
    cfg.eviction = extra.get("eviction_policy", cfg.eviction)
    cfg.cpu_bytes = as_int(extra.get("cpu_bytes_to_use"))
    tiers = extra.get("secondary_tiers") or []
    for tier in tiers:
        if tier.get("type") == "fs":
            cfg.fs_root = tier.get("root_dir") or cfg.fs_root
            cfg.fs_max_bytes = as_int(tier.get("max_bytes"))
    if not tiers:
        cfg.extras["secondary_tiers"] = "none (primary/CPU tier only)"

    # Resolved engine config: block size and pool size, as vLLM settled them.
    if metrics is not None:
        info = metrics.series.get("vllm:cache_config_info")
        labels = info[0][0] if info else {}
        cfg.block_size = as_int(labels.get("block_size"))
        cfg.pool_tokens = as_int(labels.get("kv_cache_size_tokens"))
        cfg.gpu_blocks = as_int(labels.get("num_gpu_blocks"))
        cfg.kv_cache_bytes = as_int(labels.get("kv_cache_memory_bytes")) or cfg.kv_cache_bytes
        cfg.mamba_block_size = labels.get("mamba_block_size", cfg.mamba_block_size)
        cfg.cache_layout = labels.get("kv_cache_layout", cfg.cache_layout)
        cfg.kv_dtype = labels.get("cache_dtype", cfg.kv_dtype)
        cfg.gpu_mem_util = labels.get("gpu_memory_utilization", cfg.gpu_mem_util)

    if cfg.block_size and cfg.max_model_len:
        cfg.chain_chunks = math.ceil(cfg.max_model_len / cfg.block_size)
    return cfg


# --------------------------------------------------------------------------
# local resources
# --------------------------------------------------------------------------


def gpu_memory() -> list[tuple[int, int, int]]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return rows


def mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def shm_stats() -> tuple[int | None, int | None, list[tuple[int, int, str]]]:
    """(used, total, [(bytes, mapping count, path)]) for /dev/shm.

    A running engine unlinks its staging region once every worker has mapped it,
    so it never shows up in the directory listing; /proc/*/maps still does.
    """
    used = total = None
    try:
        st = os.statvfs("/dev/shm")
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
    except OSError:
        pass
    regions: dict[str, tuple[int, int]] = {}
    for maps in glob.glob("/proc/[0-9]*/maps"):
        try:
            with open(maps, errors="ignore") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        for line in lines:
            if "/dev/shm/vllm_offload_" not in line:
                continue
            parts = line.split()
            try:
                start, end = (int(x, 16) for x in parts[0].split("-"))
            except (ValueError, IndexError):
                continue
            name = parts[5].removesuffix(" (deleted)")
            size, count = regions.get(name, (0, 0))
            regions[name] = (max(size, end - start), count + 1)
    ordered = sorted(
        ((size, count, name) for name, (size, count) in regions.items()),
        key=lambda row: row[0],
        reverse=True,
    )
    return used, total, ordered


@dataclass
class DiskChunks:
    root: str | None = None
    count: int = 0
    bytes: int = 0
    by_rank: dict[str, int] = field(default_factory=dict)
    by_group: dict[str, int] = field(default_factory=dict)
    newest: float | None = None
    oldest: float | None = None
    files: list[tuple[float, int, str]] = field(default_factory=list)


def scan_chunks(root: str | None, keep: int = 8) -> DiskChunks:
    """Walk the disk tier: it stores one ``.bin`` per chunk, keyed by hash."""
    chunks = DiskChunks(root=root)
    if not root or not os.path.isdir(root):
        return chunks
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".bin"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                st = os.stat(path)
            except OSError:
                continue
            chunks.count += 1
            chunks.bytes += st.st_size
            chunks.newest = max(chunks.newest or 0.0, st.st_mtime)
            chunks.oldest = min(chunks.oldest or st.st_mtime, st.st_mtime)
            rel = os.path.relpath(path, root)
            rank = re.search(r"_r(\d+)/", rel)
            if rank:
                key = f"r{rank.group(1)}"
                chunks.by_rank[key] = chunks.by_rank.get(key, 0) + 1
            short = short_chunk("/".join(rel.split("/")[-3:]))
            chunks.files.append((st.st_mtime, st.st_size, short))
            group = re.search(r"_g(\d+)", rel)
            if group:
                key = f"g{group.group(1)}"
                chunks.by_group[key] = chunks.by_group.get(key, 0) + 1
    chunks.files.sort(reverse=True)
    chunks.files = chunks.files[:keep]
    return chunks


def log_health(pattern: str | None, tail_bytes: int = 262_144) -> dict | None:
    if not pattern:
        return None
    paths = sorted(glob.glob(pattern))
    if not paths:
        return {"path": pattern, "errors": None, "last": "no such file"}
    path = paths[-1]
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - tail_bytes))
            text = handle.read().decode("utf-8", "ignore")
    except OSError as exc:
        return {"path": path, "errors": None, "last": str(exc)}
    patterns = (
        "block I/O failed",
        "Insufficient space",
        "Traceback",
        "ERROR",
        "CUDA out of memory",
        "OutOfMemory",
    )
    last = "-"
    errors = 0
    for line in text.splitlines():
        if any(p in line for p in patterns):
            errors += 1
            last = line.strip()[:96]
    return {"path": path, "errors": errors, "last": last}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_config(cfg: ServerConfig) -> list[str]:
    lines = ["", " CONFIG"]
    if "unavailable" in cfg.extras:
        lines.append(f"   (unavailable: {cfg.extras['unavailable']})")
        return lines

    model = cfg.model if len(cfg.model) <= 38 else "…" + cfg.model[-37:]
    chain = cfg.chain_chunks
    chunk_bytes = cfg.chunk_bytes
    lines.append(f"   {'instance':<12} pid {cfg.pid}   model {model}")
    lines.append(
        f"   {'':<12} served {cfg.served}   dtype {cfg.dtype} + {cfg.kv_dtype}"
        f"   tp {cfg.tp} (devices {cfg.devices})"
    )
    lines.append(
        f"   {'context':<12} max_model_len = {fmt_int(cfg.max_model_len)}"
        f"   chunk = {fmt_int(cfg.block_size)} tok"
        f"   (chunk_bytes = {fmt_bytes(chunk_bytes)})"
    )
    lines.append(
        f"   {'GPU KV pool':<12} kv_cache_bytes = {fmt_bytes(cfg.kv_cache_bytes)}"
        f"   tokens = {fmt_int(cfg.pool_tokens)}   blocks = {fmt_int(cfg.gpu_blocks)}"
    )
    lines.append(
        f"   {'':<12} gpu_memory_utilization = {cfg.gpu_mem_util}"
        f"   max_num_seqs = {cfg.max_num_seqs}"
        f"   max_num_batched_tokens = {cfg.max_num_batched_tokens}"
    )
    lines.append(f"   {'offload':<12} engine_id = {cfg.engine_id}")
    lines.append(
        f"   {'':<12} spec = {cfg.spec_name}   load_failure = {cfg.load_failure}"
        f"   eviction = {cfg.eviction}"
    )
    if "secondary_tiers" in cfg.extras:
        lines.append(f"   {'':<12} secondary_tiers = {cfg.extras['secondary_tiers']}")

    def chunks_of(nbytes: int | None) -> str:
        if not nbytes or not chunk_bytes:
            return "?"
        count = nbytes / chunk_bytes
        chains = f" = {count / chain:.2f} chain" if chain else ""
        return f"{count:,.0f} chunks{chains}"

    lines.append(
        f"   {'CPU tier':<12} cpu_bytes_to_use = {fmt_bytes(cfg.cpu_bytes)}"
        f"   {chunks_of(cfg.cpu_bytes)}"
    )
    if cfg.fs_root:
        lines.append(f"   {'fs tier':<12} root = {cfg.fs_root}")
        lines.append(
            f"   {'':<12} max_bytes = {fmt_bytes(cfg.fs_max_bytes)}"
            f"   {chunks_of(cfg.fs_max_bytes)}"
        )
    else:
        lines.append(f"   {'fs tier':<12} (none)")
    lines.append(
        f"   {'mamba':<12} block_size = {cfg.mamba_block_size}"
        f"   kv_cache_layout = {cfg.cache_layout if cfg.cache_layout != 'None' else '-'}"
    )
    return lines


def _row(label: str, text: str) -> list[str]:
    return [f"   {label:<12} {text}"]


def render_status(
    m: Metrics,
    prev: Metrics | None,
    cfg: ServerConfig,
    gpus: list[tuple[int, int, int]],
    shm: tuple[int | None, int | None, list[tuple[int, int, str]]],
    disk: DiskChunks,
    interval: float,
) -> list[str]:
    lines = ["", " STATUS"]

    def g(name: str, **labels: str) -> float | None:
        return m.counter(name, **labels)

    def d(name: str, **labels: str) -> float:
        return delta(g(name, **labels), prev.counter(name, **labels) if prev else None)

    def h(name: str, **labels: str) -> tuple[float | None, float | None]:
        count, total = m.hist(name, **labels)
        if prev is not None:
            pcount, ptotal = prev.hist(name, **labels)
            count = delta(count, pcount)
            total = delta(total, ptotal)
        return count, total

    # -- requests -------------------------------------------------------
    running = m.get("vllm:num_requests_running") or 0.0
    waiting = m.get("vllm:num_requests_waiting") or 0.0
    by_reason = {
        reason: m.get("vllm:num_requests_waiting_by_reason", reason=reason) or 0.0
        for reason in m.labels_of("vllm:num_requests_waiting_by_reason", "reason")
    }
    reasons = ", ".join(f"{k} {int(v)}" for k, v in by_reason.items())
    lines += _row("Requests", f"running {int(running)}   waiting {int(waiting)}"
                             f"{f' ({reasons})' if reasons else ''}")
    reasons_seen = m.labels_of("vllm:request_success_total", "finished_reason")
    canonical = ["stop", "length", "abort", "error"]
    ordered = [r for r in canonical if r in reasons_seen] + [
        r for r in reasons_seen if r not in canonical
    ]
    finished = ", ".join(
        f"{reason} {int(m.get('vllm:request_success_total', finished_reason=reason) or 0)}"
        for reason in ordered
    )
    lines += _row("Finished", finished or "-")

    # -- prefix / external cache ---------------------------------------
    for label, name in (
        ("Prefix cache", "vllm:prefix_cache"),
        ("External", "vllm:external_prefix_cache"),
    ):
        queries = m.counter(f"{name}_queries") or 0.0
        hits = m.counter(f"{name}_hits") or 0.0
        lines += _row(
            label,
            f"queries {fmt_int(queries)}   hits {fmt_int(hits)}"
            f" ({rate_pct(hits, queries)})"
            f"   +{fmt_int(d(f'{name}_queries'))}q/+{fmt_int(d(f'{name}_hits'))}h",
        )

    # -- GPU pool -------------------------------------------------------
    usage = m.get("vllm:kv_cache_usage_perc")
    tokens = cfg.pool_tokens
    used_tok = usage * tokens if (usage is not None and tokens) else None
    block = cfg.block_size or 1
    lines += _row(
        "GPU KV",
        f"[{bar(usage)}] {ratio_pct(usage)}"
        f"   {fmt_int(used_tok)} / {fmt_int(tokens)} tok"
        f"   ({fmt_int((used_tok or 0) / block)} / {fmt_int(cfg.gpu_blocks)} blocks)",
    )
    lines += _row(
        "prompt cached",
        f"{fmt_int(m.counter('vllm:prompt_tokens_cached'))} tok"
        f"   running {int(running)}   waiting {int(waiting)}",
    )

    # -- connector-level transfers -------------------------------------
    for label, direction in (("Store  GPU→CPU", "GPU_to_CPU"), ("Load   CPU→GPU", "CPU_to_GPU")):
        nbytes = m.counter("vllm:kv_offload_total_bytes", transfer_type=direction)
        seconds = m.counter("vllm:kv_offload_total_time", transfer_type=direction)
        count, total = h("vllm:kv_offload_size", transfer_type=direction)
        # The histogram counts connector transfers, whose granularity differs
        # between store and load (a load may carry a whole batch of chunks), so
        # report the count and mean without claiming a chunk count.
        mean = f"   mean {fmt_bytes(total / count)}" if count else ""
        lines += _row(
            label,
            f"{fmt_bytes(nbytes)} in {seconds or 0:.2f} s"
            f" ({throughput(nbytes, seconds)})"
            f"   transfers {fmt_int(count)}{mean}",
        )

    # -- tiers ----------------------------------------------------------
    cpu_usage = m.get("vllm:kv_offload_cpu_cache_usage_perc")
    cpu_write = m.get("vllm:kv_offload_cpu_cache_write_usage_perc")
    cpu_read = m.get("vllm:kv_offload_cpu_cache_read_usage_perc")
    cpu_used = (cpu_usage or 0.0) * (cfg.cpu_bytes or 0)
    lines += _row(
        "CPU tier",
        f"[{bar(cpu_usage)}] {ratio_pct(cpu_usage)}"
        f"   {fmt_bytes(cpu_used)} / {fmt_bytes(cfg.cpu_bytes)}"
        f"   write-hold {ratio_pct(cpu_write)}   read-hold {ratio_pct(cpu_read)}",
    )
    skipped = m.counter("vllm:kv_offload_stores_skipped")
    if skipped:
        lines += _row("", f"stores skipped {fmt_int(skipped)}")

    tier_names = m.labels_of("vllm:kv_offload_tiering_chunk_queries", "tier")
    for tier in tier_names:
        queries = m.counter("vllm:kv_offload_tiering_chunk_queries", tier=tier) or 0.0
        hits = m.counter("vllm:kv_offload_tiering_chunk_hits", tier=tier) or 0.0
        lag, lag_sum = h("vllm:kv_offload_tiering_lookup_sync_delay_seconds", tier=tier)
        alag, alag_sum = h("vllm:kv_offload_tiering_lookup_async_delay_seconds", tier=tier)
        mean_lag = f"   sync mean {1000 * lag_sum / lag:.2f} ms" if lag else ""
        mean_alag = f"   async mean {1000 * alag_sum / alag:.2f} ms" if alag else ""
        lines += _row(
            f"tier {tier}",
            f"lookups {fmt_int(queries)}   hits {fmt_int(hits)}"
            f" ({rate_pct(hits, queries)}){mean_lag}{mean_alag}",
        )
        read_bytes = m.counter("vllm:kv_offload_tiering_read_bytes", tier=tier)
        read_time = m.counter("vllm:kv_offload_tiering_read_time", tier=tier)
        write_bytes = m.counter("vllm:kv_offload_tiering_write_bytes", tier=tier)
        write_time = m.counter("vllm:kv_offload_tiering_write_time", tier=tier)
        if any(v for v in (read_bytes, write_bytes)):
            lines += _row(
                "",
                f"read {fmt_bytes(read_bytes or 0)} in {read_time or 0:.2f} s"
                f"   write {fmt_bytes(write_bytes or 0)} in {write_time or 0:.2f} s"
                f" ({throughput(write_bytes, write_time)})",
            )
        promos = m.get("vllm:kv_offload_tiering_active_promotion_jobs", tier=tier)
        cascades = m.get("vllm:kv_offload_tiering_active_cascade_jobs", tier=tier)
        pfail = m.counter("vllm:kv_offload_tiering_promotion_job_failures", tier=tier) or 0.0
        cfail = m.counter("vllm:kv_offload_tiering_cascade_job_failures", tier=tier) or 0.0
        alloc_fail = m.counter("vllm:kv_offload_tiering_promotion_allocation_failures") or 0.0
        flag = "  ⚠" if (pfail or cfail or alloc_fail) else ""
        lines += _row(
            "",
            f"jobs active promotion {int(promos or 0)} / cascade {int(cascades or 0)}"
            f"   failures promotion {int(pfail)} cascade {int(cfail)}"
            f"   alloc {int(alloc_fail)}{flag}",
        )
        whold = m.get("vllm:kv_offload_tiering_primary_write_usage_perc", tier=tier)
        rhold = m.get("vllm:kv_offload_tiering_primary_read_usage_perc", tier=tier)
        if whold is not None or rhold is not None:
            lines += _row(
                "",
                f"primary hold  write {ratio_pct(whold)}   read {ratio_pct(rhold)}",
            )

    fs_used = m.get("vllm:kv_offload_tiering_fs_used_bytes")
    if fs_used is not None or cfg.fs_max_bytes:
        frac = fs_used / cfg.fs_max_bytes if (fs_used and cfg.fs_max_bytes) else None
        lines += _row(
            "fs quota",
            f"[{bar(frac)}] {pct(fs_used or 0, cfg.fs_max_bytes)}"
            f"   {fmt_bytes(fs_used)} / {fmt_bytes(cfg.fs_max_bytes)}",
        )
        lines += _row(
            "",
            f"evictions {fmt_int(m.counter('vllm:kv_offload_tiering_fs_evictions') or 0)}"
            f" (+{fmt_int(d('vllm:kv_offload_tiering_fs_evictions'))})"
            f"   evicted {fmt_bytes(m.counter('vllm:kv_offload_tiering_fs_evicted_bytes') or 0)}"
            f" (+{fmt_bytes(d('vllm:kv_offload_tiering_fs_evicted_bytes'))})"
            f"   skipped {fmt_bytes(
                m.counter('vllm:kv_offload_tiering_fs_skipped_store_bytes') or 0
            )}"
            f" (+{fmt_bytes(d('vllm:kv_offload_tiering_fs_skipped_store_bytes'))})",
        )

    # -- local resources ------------------------------------------------
    gpu_text = "   ".join(
        f"GPU{index} {used} / {total} MiB" for index, used, total in gpus
    ) or "GPU n/a"
    lines += _row("Resources", f"{gpu_text}   tick {interval:.1f}s")
    shm_used, shm_total, regions = shm
    staging = sum(size for size, _, _ in regions)
    procs = sum(count for _, count, _ in regions)
    lines += _row(
        "",
        f"/dev/shm {fmt_bytes(shm_used)} of {fmt_bytes(shm_total)} used"
        f"   staging {fmt_bytes(staging) if regions else '-'}"
        f"{f' ({procs} procs)' if procs else ''}"
        f"   MemAvailable {fmt_bytes(mem_available())}",
    )
    lines += _row(
        "",
        f"disk tier dir {fmt_bytes(disk.bytes)}"
        f"   {disk.count} chunk file(s)"
        + (
            f"   newest {hhmmss(disk.newest)} ({ago(disk.newest, time.time())})"
            if disk.newest
            else ""
        ),
    )
    return lines


def render_chunks(disk: DiskChunks, cfg: ServerConfig, limit: int) -> list[str]:
    lines = ["", " CHUNKS ON DISK   (fs tier = hash-addressed chunks, not sessions)"]
    if not disk.root or (not disk.count and not os.path.isdir(disk.root)):
        lines.append(f"   (no disk tier: {disk.root or 'root unknown'})")
        return lines
    ranks = " / ".join(f"{key} {value}" for key, value in sorted(disk.by_rank.items()))
    groups = " / ".join(f"{key} {value}" for key, value in sorted(disk.by_group.items()))
    chains = (
        f"   chains ≈ {disk.count / cfg.chain_chunks:.2f}" if cfg.chain_chunks else ""
    )
    lines.append(
        f"   {disk.count} file(s)   {fmt_bytes(disk.bytes)}"
        + (f"   dir {ranks}" if ranks else "")
        + (f"   group {groups}" if groups else "")
        + chains
        + (
            f"   newest {hhmmss(disk.newest)}   oldest {hhmmss(disk.oldest)}"
            if disk.newest
            else ""
        )
    )
    for mtime, size, short in disk.files[:limit]:
        lines.append(f"      {hhmmss(mtime)}  {fmt_bytes(size):>11}  {short}")
    if disk.count > limit:
        lines.append(f"      … {disk.count - limit} more")
    return lines


def render_log(health: dict | None) -> list[str]:
    if health is None:
        return []
    lines = ["", " LOG"]
    if health.get("errors") is None:
        lines += _row("errors", f"? ({health.get('last')})")
    else:
        lines += _row(
            "errors",
            f"{health['errors']} (last 256 KiB of {os.path.basename(health['path'])})",
        )
        if health["errors"]:
            lines += _row("", f"last: {health['last']}")
    return lines


def render(
    m: Metrics | None,
    prev: Metrics | None,
    cfg: ServerConfig,
    gpus: list[tuple[int, int, int]],
    shm: tuple,
    disk: DiskChunks,
    health: dict | None,
    url: str,
    interval: float,
    show_chunks: bool,
    chunk_lines: int,
) -> str:
    width = 68
    now = time.strftime("%H:%M:%S")
    title = " KV Offload Monitor "
    fixed = len(title) + len(now) + 1
    side = max(0, (width - fixed) // 2)
    header = "═" * side + title + now + " " + "═" * max(0, width - fixed - side)

    body = [header, ""]
    body += render_config(cfg)
    if m is None:
        body += ["", " STATUS", "   server down / metrics unavailable"]
    else:
        body += render_status(m, prev, cfg, gpus, shm, disk, interval)
    if show_chunks:
        body += render_chunks(disk, cfg, chunk_lines)
    body += render_log(health)
    body += [
        "",
        " note: upstream has no per-request inventory, so GPU/CPU tiers are",
        "       aggregate-only; the disk tier's chunks are listed for real.",
        "═" * width,
    ]
    return "\n".join(body) + "\n"


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live KV-offload / tiering monitor (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-d", "--interval", type=float, default=5.0, help="refresh seconds")
    ap.add_argument("--port", type=int, default=8000, help="vLLM server port")
    ap.add_argument("--url", default=None, help="base url (overrides --port)")
    ap.add_argument("--pid", type=int, default=None, help="engine pid (default: by port)")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--count", type=int, default=0, help="print N times (0 = forever)")
    ap.add_argument("--json", action="store_true", help="one JSON object per tick")
    ap.add_argument("--no-clear", action="store_true", help="do not clear the screen")
    ap.add_argument("--no-chunks", action="store_true", help="hide the CHUNKS block")
    ap.add_argument(
        "--chunk-lines", type=int, default=6, help="how many disk chunks to list"
    )
    ap.add_argument("--ssd-root", default=None, help="disk tier root (default: from config)")
    ap.add_argument("--log", default=None, help="server log path/glob for an error count")
    args = ap.parse_args()

    url = args.url or f"http://localhost:{args.port}"
    count = 1 if args.once else args.count

    prev: Metrics | None = None
    tick = 0
    start = time.time()
    try:
        while True:
            metrics = fetch_metrics(url)
            pid = args.pid or find_pid(args.port)
            cfg = collect_config(pid, args.port, metrics)
            disk = scan_chunks(args.ssd_root or cfg.fs_root, keep=max(args.chunk_lines, 1))
            cfg.chunk_bytes = derive_chunk_bytes(cfg, disk)
            gpus = gpu_memory()
            shm = shm_stats()
            health = log_health(args.log)

            tick += 1
            if args.json:
                print(
                    json.dumps(
                        {
                            "_tick": tick,
                            "_elapsed": round(time.time() - start, 2),
                            "_url": url,
                            "config": cfg.__dict__,
                            "metrics": metrics.interesting() if metrics else None,
                            "gpus": gpus,
                            "shm": {
                                "used": shm[0],
                                "total": shm[1],
                                "staging": [
                                    {"bytes": s, "procs": c, "path": p} for s, c, p in shm[2]
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
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                if not args.no_clear:
                    sys.stdout.write("\033[H\033[2J")
                sys.stdout.write(
                    render(
                        metrics,
                        prev,
                        cfg,
                        gpus,
                        shm,
                        disk,
                        health,
                        url,
                        args.interval,
                        not args.no_chunks,
                        args.chunk_lines,
                    )
                )
                sys.stdout.flush()

            prev = metrics
            if count and tick >= count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if not args.json:
            print()
    return 0


def derive_chunk_bytes(cfg: ServerConfig, disk: DiskChunks) -> int | None:
    """Per-chunk bytes across both ranks, from the profile's own arithmetic.

    ``cpu_bytes_to_use`` is a whole number of chains and a chain is a whole
    number of chunks, so the division is exact for these profiles; the median
    on-disk chunk covers hand-written launches.
    """
    if cfg.cpu_bytes and cfg.chain_chunks and cfg.cpu_bytes % cfg.chain_chunks == 0:
        return cfg.cpu_bytes // cfg.chain_chunks
    if disk.files:
        sizes = sorted(size for _, size, _ in disk.files)
        return sizes[len(sizes) // 2]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
