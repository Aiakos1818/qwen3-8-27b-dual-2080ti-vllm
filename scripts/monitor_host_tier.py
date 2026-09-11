#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Live host-tier monitor: keep-alive / anchors / offload pools.

Reads live values from the vLLM ``/metrics`` endpoint and the engine's
``/proc`` environment/cmdline plus the server startup log, then redraws a
categorized screen every ``-d`` seconds (default 5).

Usage:
    python scripts/monitor_host_tier.py -d 10
    python scripts/monitor_host_tier.py --once
    python scripts/monitor_host_tier.py --json --count 5

Terminology: vLLM has no agent-session entity. A "session" in the metrics is
one request's KV chain (matched to a later request by prefix hash).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# --------------------------------------------------------------------------
# engine discovery / config
# --------------------------------------------------------------------------


def _read_file(path: str, binary: bool = False):
    try:
        if binary:
            with open(path, "rb") as f:
                return f.read()
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def find_engine_pids() -> tuple[int | None, int | None]:
    """Return (api_server_pid, engine_core_pid) discovered via /proc."""
    api = core = None
    for ent in os.listdir("/proc"):
        if not ent.isdigit():
            continue
        raw = _read_file(f"/proc/{ent}/cmdline", binary=True)
        if not raw:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", "ignore")
        if "vllm.entrypoints" in text:
            api = int(ent)
        elif "VLLM::EngineCore" in text:
            core = int(ent)
    return api, core


def read_environ(pid: int | None) -> dict[str, str]:
    if pid is None:
        return dict(os.environ)
    raw = _read_file(f"/proc/{pid}/environ", binary=True)
    if raw is None:
        return dict(os.environ)
    env: dict[str, str] = {}
    for kv in raw.split(b"\x00"):
        if b"=" in kv:
            k, v = kv.split(b"=", 1)
            env[k.decode("utf-8", "ignore")] = v.decode("utf-8", "ignore")
    return env


def read_cmdline_args(pid: int | None) -> list[str]:
    if pid is None:
        return []
    raw = _read_file(f"/proc/{pid}/cmdline", binary=True)
    if not raw:
        return []
    return [a.decode("utf-8", "ignore") for a in raw.split(b"\x00") if a]


def _arg_value(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _read_tail(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def _last_match(pattern: str, text: str) -> re.Match | None:
    if not text:
        return None
    found = None
    for found in re.finditer(pattern, text):
        pass
    return found


def auto_server_log() -> str | None:
    cands = glob.glob(os.path.join(BASE_DIR, "logs", "server*.log"))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def read_config(api_pid: int | None, server_log: str | None) -> dict:
    env = read_environ(api_pid)
    args = read_cmdline_args(api_pid)
    log = _read_tail(server_log) if server_log else ""

    def _last_int(pat: str) -> int | None:
        m = _last_match(pat, log)
        return int(m.group(1)) if m else None

    cfg: dict = {}
    cfg["pin_min_tokens"] = env.get("VLLM_PIN_MIN_TOKENS")
    cfg["ckpt_tokens"] = env.get("VLLM_MAMBA_CKPT_TOKENS")
    cfg["ckpt_anchors"] = env.get("VLLM_MAMBA_CKPT_ANCHORS")
    cfg["evict_small"] = _last_int(r"evict_small=(\d+)") or env.get(
        "VLLM_HOSTTIER_EVICT_SMALL_TOKENS"
    )
    cfg["block_size"] = _last_int(r"block_size=(\d+)")
    cfg["staging_slots"] = _last_int(r"staging_slots=(\d+)")
    cfg["chunk_slots"] = _last_int(r"chunk_slots=(\d+)")
    m_gpu = _last_match(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", log)
    cfg["gpu_total_tokens"] = (
        int(m_gpu.group(1).replace(",", "")) if m_gpu else None
    )

    m = _last_match(r"HostTierSSDStore:\s*root=(\S+)\s+quota=([\d.]+)GiB", log)
    cfg["ssd_root_log"] = m.group(1) if m else None
    cfg["ssd_quota_log"] = float(m.group(2)) * (1 << 30) if m else None

    cfg["ssd_root"] = env.get("VLLM_SSD_ROOT") or cfg["ssd_root_log"]
    cfg["ssd_quota"] = env.get("VLLM_SSD_QUOTA_BYTES") or cfg["ssd_quota_log"]
    cfg["ssd_max_mbps"] = env.get("VLLM_SSD_MAX_MBPS")
    cfg["ssd_only"] = env.get("VLLM_SSD_ONLY")
    cfg["ssd_clean"] = env.get("VLLM_SSD_CLEAN_START")
    cfg["ssd_chunk_env"] = env.get("VLLM_SSD_CHUNK_SLOTS")

    cfg["max_model_len"] = _arg_value(args, "--max-model-len")
    cfg["max_num_seqs"] = _arg_value(args, "--max-num-seqs")
    cfg["kv_cache_bytes"] = _arg_value(args, "--kv-cache-memory-bytes")
    xfer = _arg_value(args, "--kv-transfer-config")
    cfg["cpu_bytes"] = None
    if xfer:
        try:
            cfg["cpu_bytes"] = json.loads(xfer).get(
                "kv_connector_extra_config", {}
            ).get("cpu_bytes_to_use")
        except (ValueError, AttributeError):
            pass
    return cfg


# --------------------------------------------------------------------------
# live metrics
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


def render_config(cfg: dict, url: str) -> list[str]:
    def v(key, default="?"):
        val = cfg.get(key)
        return default if val in (None, "") else str(val)

    gpu_total = cfg.get("gpu_total_tokens")
    block = cfg.get("block_size")
    gpu_blocks = f"{gpu_total // block:,}" if gpu_total and block else "?"

    lines = ["", " CONFIG"]
    lines.append(f"   {'endpoint':<12} {url}")
    lines.append(
        f"   {'keep-alive':<12} pin_min_tokens   = {v('pin_min_tokens')}"
    )
    lines.append(
        f"   {'anchors':<12} ckpt_tokens      = {v('ckpt_tokens')}"
        f"        anchors = {v('ckpt_anchors')}"
    )
    lines.append(
        f"   {'eviction':<12} small_tokens     = {v('evict_small')}"
        f"        (small<{v('evict_small')} first, else oldest)"
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
        f"   {'staging':<12} cpu_bytes_to_use = "
        f"{fmt_bytes(float(cfg['cpu_bytes'])) if cfg.get('cpu_bytes') else '?'}"
        f"       slots = {v('staging_slots')}   chunk_slots = {v('chunk_slots')}"
    )
    lines.append(f"   {'SSD':<12} root             = {v('ssd_root')}")
    lines.append(
        f"   {'':<12} quota = {fmt_bytes(cfg.get('ssd_quota'))}"
        f"   max_mibps = {v('ssd_max_mbps')}   only = {v('ssd_only')}"
        f"   clean_start = {v('ssd_clean')}"
    )
    return lines


def render_status(
    m: dict[str, float],
    cfg: dict,
    prev: dict[str, float],
    gpus: list[tuple[int, int, int]],
    ssd_dir: int | None,
    interval: float,
) -> list[str]:
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
    st = float(g("vllm:host_tier_ssd_quota_bytes") or cfg.get("ssd_quota") or 0.0)
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


def render(
    m: dict[str, float] | None,
    cfg: dict,
    prev: dict[str, float],
    url: str,
    gpus: list[tuple[int, int, int]],
    ssd_dir: int | None,
    interval: float,
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
    body += render_config(cfg, url)
    if m is None:
        body += ["", " STATUS", "   server down / metrics unavailable"]
    else:
        body += render_status(m, cfg, prev, gpus, ssd_dir, interval)
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
    ap.add_argument("--url", default="http://localhost:8000",
                    help="vLLM server base url")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--count", type=int, default=0,
                    help="print N times then exit (0 = forever)")
    ap.add_argument("--server-log", default=None,
                    help="server log to parse config from (default auto)")
    ap.add_argument("--ssd-root", default=None,
                    help="SSD dir for du (default: config/env)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object per tick (no clear)")
    ap.add_argument("--no-clear", action="store_true",
                    help="do not clear the screen between ticks")
    args = ap.parse_args()

    count = 1 if args.once else args.count
    server_log = args.server_log or auto_server_log()

    cfg: dict = {}
    cfg_pid: int | None = None
    prev: dict[str, float] = {}
    last_dir: int | None = None
    start = time.time()
    tick = 0

    try:
        while True:
            api_pid, core_pid = find_engine_pids()
            pid = api_pid or core_pid
            if pid != cfg_pid:
                cfg = read_config(pid, server_log)
                cfg_pid = pid
                prev = {}

            m = fetch_metrics(args.url)
            if m is not None and not prev:
                prev = dict(m)
            gpus = gpu_memory()
            ssd_root = args.ssd_root or cfg.get("ssd_root")
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
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            else:
                if not args.no_clear:
                    sys.stdout.write("\033[H\033[2J")
                sys.stdout.write(
                    render(m, cfg, prev, args.url, gpus, last_dir,
                           args.interval)
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
