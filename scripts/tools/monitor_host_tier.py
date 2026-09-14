#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Live host-tier monitor: keep-alive / anchors / offload pools.

Live values come from two read-only HTTP endpoints on the vLLM server:

- ``GET /metrics``        -- Prometheus counters/gauges (STATUS block)
- ``GET /host_tier_info`` -- host-tier config + per-chain inventory (CONFIG,
  SESSIONS blocks), served by the engine itself.

The instance is selected with ``--port`` (default 8000); ``--url`` overrides
the derived ``http://localhost:<port>``. Nothing is read from ``/proc``,
startup logs or snapshot files.

Usage:
    python scripts/tools/monitor_host_tier.py -d 10
    python scripts/tools/monitor_host_tier.py --port 8001 --once
    python scripts/tools/monitor_host_tier.py --json --count 5
    python scripts/tools/monitor_host_tier.py --no-sessions

Terminology: vLLM has no agent-session entity. A "session" in the metrics is
one request's KV chain (matched to a later request by prefix hash).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

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


def fmt_tok(n: float | None) -> str:
    if n is None:
        return "?"
    n = int(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def fmt_int(n: float | None) -> str:
    return "?" if n is None else f"{int(n):,}"


def bar(frac: float | None, width: int = 10) -> str:
    if frac is None:
        return "░" * width
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def pct(used: float | None, total: float | None) -> str:
    if not total:
        return "  ?  %"
    return f"{100.0 * used / total:5.1f}%"


def _ago(epoch: float | None, now: float) -> str:
    if not epoch:
        return "?"
    secs = max(0.0, now - float(epoch))
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    return f"{int(secs // 3600)}h"


def _sid_short(sid: object, width: int = 22) -> str:
    text = str(sid)
    return text if len(text) <= width else text[: width - 1] + "…"


# --------------------------------------------------------------------------
# HTTP endpoints
# --------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"^(vllm:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN|Inf|\+Inf)",
    re.M,
)

WANTED = [
    "vllm:keep_alive_entries",
    "vllm:keep_alive_blocks",
    "vllm:keep_alive_tokens",
    "vllm:keep_alive_anchors",
    "vllm:keep_alive_anchor_sessions",
    "vllm:kv_cache_usage_perc",
    "vllm:host_tier_slots_used",
    "vllm:host_tier_slots_total",
    "vllm:host_tier_sessions",
    "vllm:host_tier_spills_total",
    "vllm:host_tier_restores_total",
    "vllm:host_tier_evictions_total",
    "vllm:host_tier_drops_total",
    "vllm:host_tier_ssd_sessions",
    "vllm:host_tier_ssd_bytes_used",
    "vllm:host_tier_ssd_quota_bytes",
    "vllm:host_tier_ssd_stores_total",
    "vllm:host_tier_ssd_restores_total",
    "vllm:host_tier_ssd_evictions_total",
    "vllm:host_tier_ssd_drops_total",
    "vllm:host_tier_ssd_write_bytes_total",
    "vllm:host_tier_ssd_read_bytes_total",
]


def _get_json(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def fetch_metrics(url: str) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", "ignore")
    except Exception:
        return None
    out: dict[str, float] = {}
    for m in _METRIC_RE.finditer(text):
        name, raw = m.group(1), m.group(2)
        if name not in WANTED:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


def fetch_info(url: str) -> dict | None:
    return _get_json(f"{url}/host_tier_info", timeout=15.0)


def gpu_memory() -> list[tuple[int, int, int]]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not out:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return rows


def run(cmd: list[str], timeout: float = 5) -> str | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def dir_bytes(path: str | None) -> int | None:
    if not path or not os.path.isdir(path):
        return None
    out = run(["du", "-sb", path], timeout=5)
    if not out:
        return None
    try:
        return int(out.split()[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_config(cfg: dict | None, url: str) -> list[str]:
    def v(key, default="?"):
        val = (cfg or {}).get(key)
        return default if val in (None, "") else str(val)

    lines = ["", " CONFIG", f"   {'endpoint':<12} {url}"]
    if cfg is None:
        lines.append("   (unavailable: /host_tier_info not reachable)")
        return lines

    gpu_total = cfg.get("gpu_total_tokens")
    block = cfg.get("block_size")
    gpu_blocks = f"{gpu_total // block:,}" if gpu_total and block else "?"
    model = str(cfg.get("model") or "?")
    model_short = model if len(model) <= 40 else "…" + model[-39:]

    lines.append(f"   {'instance':<12} pid {v('pid')}   model {model_short}")
    lines.append(
        f"   {'keep-alive':<12} pin_min_tokens   = {v('pin_min_tokens')}"
    )
    lines.append(
        f"   {'anchors':<12} ckpt_tokens      = {v('ckpt_tokens')}"
        f"        anchors = {v('ckpt_anchors')}"
    )
    lines.append(
        f"   {'eviction':<12} small_tokens     = {v('evict_small_tokens')}"
        f"        (small<{v('evict_small_tokens')} first, else oldest)"
    )
    kv_bytes = cfg.get("kv_cache_bytes")
    kv_bytes_txt = fmt_bytes(float(kv_bytes)) if kv_bytes else "?"
    lines.append(
        f"   {'GPU KV pool':<12} kv_cache_bytes   = {kv_bytes_txt}"
        f"       max_model_len = {v('max_model_len')}"
    )
    lines.append(
        f"   {'':<12} block_size       = {v('block_size')} tok"
        f"     max_num_seqs = {v('max_num_seqs')}"
    )
    lines.append(
        f"   {'':<12} capacity         = {fmt_int(gpu_total)} tok"
        f"   ({gpu_blocks} blocks)"
    )
    lines.append(
        f"   {'staging':<12} cpu_bytes_to_use = {fmt_bytes(cfg.get('cpu_bytes'))}"
        f"       slots = {v('staging_slots')}   chunk_slots = {v('chunk_slots')}"
    )
    ssd = cfg.get("ssd") or {}
    lines.append(f"   {'SSD':<12} root             = {ssd.get('root') or '?'}")
    lines.append(
        f"   {'':<12} quota = {fmt_bytes(ssd.get('quota_bytes'))}"
        f"   max_mibps = {ssd.get('max_mbps')}   only = {ssd.get('only')}"
        f"   clean_start = {ssd.get('clean_start')}"
    )
    return lines


def render_status(
    m: dict[str, float],
    cfg: dict | None,
    prev: dict[str, float],
    gpus: list[tuple[int, int, int]],
    ssd_dir: int | None,
    interval: float,
) -> list[str]:
    cfg = cfg or {}

    def g(key, default=0.0):
        return m.get(key, default)

    def d(key):
        return g(key) - prev.get(key, 0.0)

    lines = ["", " STATUS"]

    lines.append(
        f"   {'Keep-alive':<12} entries {int(g('vllm:keep_alive_entries'))}"
        f"   blocks {int(g('vllm:keep_alive_blocks'))}"
        f"   tokens {fmt_tok(g('vllm:keep_alive_tokens'))}"
        f"   anchors {int(g('vllm:keep_alive_anchors'))} blk / "
        f"{int(g('vllm:keep_alive_anchor_sessions'))} sess"
    )

    usage = g("vllm:kv_cache_usage_perc")
    total = cfg.get("gpu_total_tokens")
    block = cfg.get("block_size") or 1
    used_tok = usage * total if total else None
    used_blk = used_tok / block if used_tok else None
    tot_blk = total / block if total else None
    lines.append(
        f"   {'GPU KV':<12} [{bar(usage)}] {pct(used_tok, total)}"
        f"   {fmt_int(used_tok)} / {fmt_int(total)} tok"
        f"   ({fmt_int(used_blk)} / {fmt_int(tot_blk)} blocks)"
    )

    ru, rt = g("vllm:host_tier_slots_used"), g("vllm:host_tier_slots_total")
    lines.append(
        f"   {'RAM pool':<12} [{bar(ru / rt if rt else None)}] {pct(ru, rt)}"
        f"   {int(ru)} / {int(rt)} slots      "
        f"sessions {int(g('vllm:host_tier_sessions'))}"
    )
    lines.append(
        f"   {'':<12} spills {int(g('vllm:host_tier_spills_total'))} "
        f"(+{int(d('vllm:host_tier_spills_total'))})   "
        f"restores {int(g('vllm:host_tier_restores_total'))} "
        f"(+{int(d('vllm:host_tier_restores_total'))})   "
        f"evictions {int(g('vllm:host_tier_evictions_total'))} "
        f"(+{int(d('vllm:host_tier_evictions_total'))})   "
        f"drops {int(g('vllm:host_tier_drops_total'))} "
        f"(+{int(d('vllm:host_tier_drops_total'))})"
    )

    su = float(g("vllm:host_tier_ssd_bytes_used") or 0.0)
    st = float(
        g("vllm:host_tier_ssd_quota_bytes")
        or (cfg.get("ssd") or {}).get("quota_bytes")
        or 0.0
    )
    lines.append(
        f"   {'SSD pool':<12} [{bar(su / st if st else None)}] {pct(su, st)}"
        f"   {fmt_bytes(su)} / {fmt_bytes(st)}    "
        f"sessions {int(g('vllm:host_tier_ssd_sessions'))}"
    )
    lines.append(
        f"   {'':<12} stores {int(g('vllm:host_tier_ssd_stores_total'))} "
        f"(+{int(d('vllm:host_tier_ssd_stores_total'))})   "
        f"restores {int(g('vllm:host_tier_ssd_restores_total'))} "
        f"(+{int(d('vllm:host_tier_ssd_restores_total'))})   "
        f"evictions {int(g('vllm:host_tier_ssd_evictions_total'))} "
        f"(+{int(d('vllm:host_tier_ssd_evictions_total'))})   "
        f"drops {int(g('vllm:host_tier_ssd_drops_total'))} "
        f"(+{int(d('vllm:host_tier_ssd_drops_total'))})"
    )
    lines.append(
        f"   {'':<12} write {fmt_bytes(g('vllm:host_tier_ssd_write_bytes_total'))} "
        f"(+{fmt_bytes(d('vllm:host_tier_ssd_write_bytes_total'))})   "
        f"read {fmt_bytes(g('vllm:host_tier_ssd_read_bytes_total'))} "
        f"(+{fmt_bytes(d('vllm:host_tier_ssd_read_bytes_total'))})"
    )

    gpu_txt = "   ".join(
        f"GPU{i} {used} / {tot} MiB" for i, used, tot in gpus
    ) or "GPU n/a"
    lines.append(
        f"   {'Resources':<12} {gpu_txt}   ssd_dir {fmt_bytes(ssd_dir)}"
        f"   tick {interval:.1f}s"
    )
    return lines


def render_sessions(info: dict | None, now: float) -> list[str]:
    lines = ["", " SESSIONS"]
    if info is None:
        lines.append("   (unavailable: /host_tier_info not reachable)")
        return lines
    data = info.get("sessions") or {}
    groups = [
        ("GPU", data.get("gpu") or []),
        ("RAM", data.get("ram") or []),
        ("SSD", data.get("ssd") or []),
    ]
    summary = ", ".join(f"{name} {len(items)}" for name, items in groups)
    lines.append(f"   ({summary})")
    for name, items in groups:
        for i, sess in enumerate(items, 1):
            ts = sess.get("last_used")
            try:
                when = time.strftime("%H:%M:%S", time.localtime(float(ts)))
            except (TypeError, ValueError):
                when = "?"
            lines.append(
                f"   {name} [{i}]  id={_sid_short(sess.get('id')):<22}"
                f"  {fmt_tok(sess.get('tokens')):>8} tok"
                f"  {fmt_int(sess.get('blocks')):>5} blk"
                f"  {fmt_bytes(sess.get('bytes')):>10}"
                f"  anchors {int(sess.get('anchors', 0) or 0)}"
                f"  last_used {when} ({_ago(ts, now)})"
            )
    return lines


def render(
    m: dict[str, float] | None,
    info: dict | None,
    prev: dict[str, float],
    url: str,
    gpus: list[tuple[int, int, int]],
    ssd_dir: int | None,
    interval: float,
    show_sessions: bool = True,
) -> str:
    width = 60
    now = time.strftime("%H:%M:%S")
    title = " Host-Tier Monitor "
    fixed = len(title) + len(now) + 1
    side = max(0, (width - fixed) // 2)
    header = (
        "═" * side + title + now + " " + "═" * max(0, width - fixed - side)
    )
    body = [header, ""]
    body += render_config((info or {}).get("config"), url)
    if m is None:
        body += ["", " STATUS", "   server down / metrics unavailable"]
    else:
        body += render_status(
            m, (info or {}).get("config"), prev, gpus, ssd_dir, interval
        )
    if show_sessions:
        body += render_sessions(info, time.time())
    body += [
        "",
        ' note: vLLM has no agent session; "session" = one request KV chain',
        "       (matched to a later request by prefix hash).",
        "═" * width,
    ]
    return "\n".join(body) + "\n"


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--interval", type=float, default=5.0,
                    help="refresh seconds (default 5)")
    ap.add_argument("--port", type=int, default=8000,
                    help="vLLM server port to monitor (default 8000)")
    ap.add_argument("--url", default=None,
                    help="vLLM server base url (default http://localhost:PORT)")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--count", type=int, default=0,
                    help="print N times then exit (0 = forever)")
    ap.add_argument("--ssd-root", default=None,
                    help="SSD dir for du (default: from /host_tier_info)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object per tick (no clear)")
    ap.add_argument("--no-clear", action="store_true",
                    help="do not clear the screen between ticks")
    ap.add_argument("--no-sessions", action="store_true",
                    help="hide the per-chain SESSIONS block")
    args = ap.parse_args()

    url = args.url or f"http://localhost:{args.port}"
    count = 1 if args.once else args.count

    prev: dict[str, float] = {}
    last_dir: int | None = None
    start = time.time()
    tick = 0

    try:
        while True:
            m = fetch_metrics(url)
            if m is not None and not prev:
                prev = dict(m)
            info = fetch_info(url)
            cfg = (info or {}).get("config") or {}

            gpus = gpu_memory()
            ssd_root = args.ssd_root or (cfg.get("ssd") or {}).get("root")
            if isinstance(ssd_root, bytes):
                ssd_root = ssd_root.decode("utf-8", "ignore")
            cur_dir = dir_bytes(ssd_root)
            if cur_dir is not None:
                last_dir = cur_dir

            tick += 1
            if args.json:
                rec = dict(m or {})
                rec["_tick"] = tick
                rec["_elapsed"] = round(time.time() - start, 2)
                rec["_gpu_mem"] = gpus
                rec["_ssd_dir_bytes"] = last_dir
                rec["_host_tier_info"] = info
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            else:
                if not args.no_clear:
                    sys.stdout.write("\033[H\033[2J")
                sys.stdout.write(
                    render(m, info, prev, url, gpus, last_dir,
                           args.interval, not args.no_sessions)
                )
                sys.stdout.flush()

            prev = dict(m or {})
            if count and tick >= count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if not args.json:
            print()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
