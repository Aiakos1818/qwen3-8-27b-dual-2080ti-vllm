#!/usr/bin/env python3
"""Diagnose a vLLM launch profile's KV-cache pool against its context length.

The vLLM startup check sizes the attention group exactly to ``--max-model-len``,
leaving no room for chunked-prefill copy-on-write. A request that reaches that
ceiling then self-preempts and has to be re-admitted. This tool mirrors the
vLLM arithmetic and, given only the launch run script, reports:

  [池检查]      is the profile's pool large enough for its max-model-len?
  [上下文检查]  is the profile's max-model-len within the pool's safe limit?

Everything is read from the run script; no parameter has to be supplied.

Usage:
  kv_pool_sizing.py <run.sh>                     # the two checks
  kv_pool_sizing.py <run.sh> --max-len 500k      # override the context
  kv_pool_sizing.py <run.sh> --pool-bytes 9.6e9  # override the pool
  kv_pool_sizing.py <run.sh> --feasible          # deploy once, measure OOM risk
  kv_pool_sizing.py --self-test

Sources mirrored (vLLM main):
  * block-size auto-selection: ``vllm/platforms/interface.py``
  * grouping heuristic:        ``vllm/v1/core/kv_cache_utils.py``
  * pool requirement:          ``vllm/v1/core/kv_cache_utils.py``
  * mamba align usage:         ``vllm/v1/kv_cache_interface.py``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field

DTYPE_BYTES = {
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
}

# Fallback used when no model config is given (Qwen3.8-27B).
QWEN38_DEFAULTS = {
    "num_hidden_layers": 64,
    "full_attention_interval": 4,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "mtp_num_hidden_layers": 1,
    "mamba_ssm_dtype": "float32",
}

# Runtime headroom heuristic: copy-on-write / transient blocks on top of the
# attention group. 3 blocks is the value the 128K profile was measured with
# (predicted safe cap 142,400 tokens against 144,584 actual).
HEADROOM_BLOCKS = 3
# Non-KV overhead on top of the weights (activations, graphs, NCCL, startup
# transient), used only when measuring via --log instead of --feasible.
NON_KV_BASELINE_GIB = 2.0
# Usable GPU memory after the driver reserve, for a 22528 MiB card.
GPU_USABLE_GIB = 21.5

# --feasible deploy settings.
FEASIBLE_LOG = "/tmp/kv_pool_sizing_feasible.log"
HEALTH_URL = "http://localhost:8000/v1/models"
POLL_SECONDS = 10
READY_TIMEOUT_S = 1200
MAX_ATTEMPTS = 3
_OOM_RE = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError")
_TRANSIENT_RE = re.compile(
    r"CUDA error: invalid argument|torch\.AcceleratorError"
    r"|Engine core initialization failed|EngineDeadError"
)
_LOAD_RE = re.compile(r"Model loading took ([\d.]+) GiB")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _gib(n: float) -> str:
    return f"{n / (1024**3):.3f} GiB"


def _mib(n: float) -> str:
    return f"{n / (1024**2):.3f} MiB"


def parse_size(text: str) -> int:
    """Parse ``512000``, ``500k``, ``9.6e9`` or ``9.6G`` into an int."""
    t = str(text).strip().lower().replace("_", "")
    mult = 1
    for suffix, m in (
        ("gib", 1024**3),
        ("mib", 1024**2),
        ("g", 1024**3),
        ("m", 1024**2),
        ("k", 1000),
    ):
        if t.endswith(suffix):
            mult = m
            t = t[: -len(suffix)]
            break
    return int(float(t) * mult)


# --------------------------------------------------------------------------
# Run-script parsing
# --------------------------------------------------------------------------
_ENV_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*?)\s*$", re.M)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*?)\s*$", re.M)
_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):[=-]([^}]*)\}")
_REF_RE = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?::[=-]([^}]*))?\}|([A-Za-z_][A-Za-z0-9_]*))"
)
_ARG_RE = re.compile(
    r"--([a-z0-9][a-z0-9-]*)(?:=|\s+)"
    r"""("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s\\]+)"""
)
_FLAG_RE = re.compile(r"--([a-z0-9][a-z0-9-]*)(?=\s|$)")


def _repo_env(path: str) -> dict[str, str]:
    """Read the repo-root .env a profile sources (parse, never execute)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    env_path = os.path.join(root, ".env")
    out: dict[str, str] = {}
    if not os.path.isfile(env_path):
        return out
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip().removeprefix("export ").strip()
            out[name] = value.strip().strip("'\"")
    return out


def _resolve(value: str, lookup: dict[str, str]) -> str:
    """Expand $VAR / ${VAR} / ${VAR:-default} against the profile's variables.

    Values may reference one another (a profile sets KV_XFER from KV_ENGINE_ID,
    for instance), so expand to a fixed point before falling back to the shell
    environment for anything the profile never defines.
    """
    for _ in range(5):
        def repl(m: re.Match) -> str:
            name = m.group(1) or m.group(3)
            default = m.group(2)
            if lookup.get(name):
                return lookup[name]
            if default is not None:
                return default
            return os.environ.get(name, "")

        expanded = _REF_RE.sub(repl, value)
        if expanded == value:
            break
        value = expanded
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return re.sub(r"\\(.)", r"\1", value)


@dataclass
class Profile:
    path: str
    args: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)

    @property
    def max_len(self) -> int:
        return int(self.args.get("max-model-len", 0) or 0)

    @property
    def pool_bytes(self) -> int:
        return int(self.args.get("kv-cache-memory-bytes", 0) or 0)


def parse_profile(path: str) -> Profile:
    """Read (never execute) a launch script's CLI args and exports."""
    with open(path) as f:
        raw = f.read()
    text = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    text = text.replace("\\\n", " ")
    # Mirror the profile's own precedence: it sources $REPO_ROOT/.env first,
    # then `: "${VAR:=default}"` fills in what .env left unset, and finally
    # plain / exported assignments win.
    lookup = _repo_env(path)
    for name, default in _DEFAULT_RE.findall(text):
        lookup.setdefault(name, default.strip())
    for name, value in _ENV_RE.findall(text) + _ASSIGN_RE.findall(text):
        lookup[name] = _resolve(value, lookup)
    prof = Profile(path=path, env=dict(lookup))
    for name, value in _ARG_RE.findall(text):
        prof.args[name] = _resolve(value, lookup)
    for name in _FLAG_RE.findall(text):
        prof.flags.add(name)
    return prof


# --------------------------------------------------------------------------
# Model dimensions
# --------------------------------------------------------------------------
@dataclass
class ModelDims:
    attn_layers: int
    linear_layers: int
    num_kv_heads: int
    head_dim: int
    conv_kernel: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    source: str


def _resolve_config(path: str) -> tuple[dict, str]:
    cfg_path = os.path.join(path, "config.json") if os.path.isdir(path) else path
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        raw = json.load(f)
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(raw.get(key), dict):
            return raw[key], cfg_path
    return raw, cfg_path


def _layer_counts(cfg: dict) -> tuple[int, int]:
    mtp = int(cfg.get("mtp_num_hidden_layers", 0) or 0)
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum("full" in str(t) for t in layer_types)
        return full + mtp, len(layer_types) - full
    n = int(cfg["num_hidden_layers"])
    interval = int(cfg.get("full_attention_interval", 1) or 1)
    full = n // interval if interval > 0 else n
    return full + mtp, n - full


def load_model(path: str | None) -> ModelDims:
    if path and os.path.exists(path):
        cfg, source = _resolve_config(path)
    else:
        cfg, source = dict(QWEN38_DEFAULTS), "内置 Qwen3.8-27B 默认"
    d = {**QWEN38_DEFAULTS, **cfg}
    attn, linear = _layer_counts(d)
    return ModelDims(
        attn_layers=attn,
        linear_layers=linear,
        num_kv_heads=d["num_key_value_heads"],
        head_dim=d["head_dim"],
        conv_kernel=d["linear_conv_kernel_dim"],
        linear_num_key_heads=d["linear_num_key_heads"],
        linear_num_value_heads=d["linear_num_value_heads"],
        linear_key_head_dim=d["linear_key_head_dim"],
        linear_value_head_dim=d["linear_value_head_dim"],
        source=source,
    )


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------
def attn_page_1_token(dims: ModelDims, tp: int, kv_dtype: str) -> int:
    return 2 * (dims.num_kv_heads // tp) * dims.head_dim * DTYPE_BYTES[kv_dtype]


def mamba_page(dims: ModelDims, tp: int, num_spec: int) -> int:
    conv_dim = (
        dims.linear_key_head_dim * dims.linear_num_key_heads * 2
        + dims.linear_value_head_dim * dims.linear_num_value_heads
    )
    conv = (dims.conv_kernel - 1 + num_spec) * (conv_dim // tp) * 2
    temporal = (
        (dims.linear_num_value_heads // tp)
        * dims.linear_value_head_dim
        * dims.linear_key_head_dim
        * 4
    )
    return conv + temporal


def auto_block_size(dims: ModelDims, tp: int, kv_dtype: str, num_spec: int) -> int:
    a = attn_page_1_token(dims, tp, kv_dtype)
    m = mamba_page(dims, tp, num_spec)
    return 16 * cdiv(m, 16 * a)


def _group_size(dims: ModelDims) -> int:
    mn = min(dims.attn_layers, dims.linear_layers)
    mx = max(dims.attn_layers, dims.linear_layers)
    return mx if mx < mn * 1.5 else mn


@dataclass
class Sizing:
    dims: ModelDims
    tp: int
    num_spec: int
    kv_dtype: str
    block_size: int
    page_size: int
    group_size: int
    mamba_groups: int
    mamba_blocks: int
    headroom: int
    max_num_seqs: int
    warnings: list[str] = field(default_factory=list)

    @property
    def slot_bytes(self) -> int:
        return self.group_size * self.page_size

    def min_pool(self, max_len: int) -> int:
        return self.slot_bytes * (cdiv(max_len, self.block_size) + self.mamba_blocks)

    def safe_pool(self, max_len: int) -> int:
        conc = (self.max_num_seqs - 1) * (
            cdiv(max_len, self.block_size) + self.mamba_blocks
        )
        return self.slot_bytes * (
            cdiv(max_len, self.block_size) + self.mamba_blocks + self.headroom + conc
        )

    def max_safe_len(self, pool_bytes: int) -> int:
        slots = pool_bytes // self.slot_bytes
        cap_blocks = max(0, slots - self.mamba_blocks)
        return max(0, cap_blocks - self.headroom) * self.block_size


def build_sizing(prof: Profile | None, args: argparse.Namespace) -> Sizing:
    p = prof.args if prof else {}
    flags = prof.flags if prof else set()

    dims = load_model(args.model_config or p.get("model"))
    tp = args.tp or int(p.get("tensor-parallel-size", 2))
    kv_dtype = args.kv_dtype or p.get("kv-cache-dtype", "fp8_e4m3")

    num_spec = args.num_spec
    if num_spec is None:
        num_spec = 0
        spec = p.get("speculative-config")
        if spec:
            try:
                num_spec = int(json.loads(spec).get("num_speculative_tokens", 0))
            except (ValueError, json.JSONDecodeError):
                num_spec = 0

    max_num_seqs = args.max_num_seqs or int(p.get("max-num-seqs", 1))

    block_size = args.block_size or auto_block_size(dims, tp, kv_dtype, num_spec)
    page_size = block_size * attn_page_1_token(dims, tp, kv_dtype)
    gsize = _group_size(dims)
    mgroups = cdiv(dims.linear_layers, gsize)

    align = "enable-prefix-caching" in flags or args.prefix_caching
    mamba_blocks = mgroups * ((2 + num_spec) if align else (1 + num_spec))

    warnings: list[str] = []
    if not align:
        warnings.append("prefix caching 关闭 -> mamba 模式 none")
    if max_num_seqs > 1:
        warnings.append(f"max-num-seqs={max_num_seqs}：已计入并发余量")

    headroom = args.headroom_blocks or HEADROOM_BLOCKS

    return Sizing(
        dims=dims,
        tp=tp,
        num_spec=num_spec,
        kv_dtype=kv_dtype,
        block_size=block_size,
        page_size=page_size,
        group_size=gsize,
        mamba_groups=mgroups,
        mamba_blocks=mamba_blocks,
        headroom=headroom,
        max_num_seqs=max_num_seqs,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _pool_verdict(pool: int, min_pool: int, safe_pool: int) -> str:
    if pool >= safe_pool:
        return "符合"
    if pool >= min_pool:
        return "偏小（可启动，但满池会自我抢占）"
    return "不足（启动校验会报错）"


def _header(s: Sizing, prof: Profile | None) -> list[str]:
    src = s.dims.source
    if src.endswith("config.json"):
        src = os.path.basename(os.path.dirname(src))
    lines = []
    if prof:
        lines.append(f"profile : {prof.path}")
    lines.append(
        f"model   : {src}  tp={s.tp} kv={s.kv_dtype} "
        f"MTP={s.num_spec} block={s.block_size}"
    )
    lines.append(
        f"          page={_mib(s.page_size)} group_size={s.group_size} "
        f"mamba_blocks={s.mamba_blocks} headroom={s.headroom}"
    )
    for w in s.warnings:
        lines.append(f"WARNING : {w}")
    return lines


def cmd_report(s: Sizing, prof: Profile, args: argparse.Namespace) -> int:
    max_len = args.max_len or prof.max_len
    pool = args.pool_bytes or prof.pool_bytes
    out = _header(s, prof) + [
        f"max-model-len         = {max_len:,}" if max_len
        else "max-model-len         = (未定义)",
        f"kv-cache-memory-bytes = {pool:,}" if pool
        else "kv-cache-memory-bytes = (未定义)",
        "",
    ]
    ok = True

    if max_len:
        safe = s.safe_pool(max_len)
        minimum = s.min_pool(max_len)
        state = _pool_verdict(pool, minimum, safe) if pool else "（未定义池）"
        out.append(f"[池检查] 支持 {max_len:,} 需安全池 >= {safe:,}")
        if pool:
            out.append(f"         当前 {pool:,} -> {state}")
            if pool < safe:
                out.append(f"         -> 建议 --kv-cache-memory-bytes {safe}")
                ok = False
        else:
            out.append(f"         -> 建议 --kv-cache-memory-bytes {safe}")
    else:
        out.append("[池检查] 未定义 --max-model-len，跳过")

    out.append("")
    if pool:
        safe_len = s.max_safe_len(pool)
        out.append(f"[上下文检查] 当前池 {pool:,} 的安全上限 = {safe_len:,}")
        if max_len:
            if max_len <= safe_len:
                out.append(f"         当前 {max_len:,} -> 符合")
            else:
                out.append(f"         当前 {max_len:,} -> 超出")
                out.append(f"         -> 建议 --max-model-len <= {safe_len}")
                ok = False
    else:
        out.append("[上下文检查] 未定义 kv-cache-memory-bytes，跳过")

    print("\n".join(out))
    return 0 if ok else 1


def cmd_feasible(s: Sizing, prof: Profile, args: argparse.Namespace) -> int:
    target_len = args.max_len or prof.max_len
    if not target_len:
        print("[可行性] 未定义 --max-model-len，无法判定")
        return 2
    target_pool = s.safe_pool(target_len)

    if args.non_kv_gib:
        non_kv, src = args.non_kv_gib, "给定 --non-kv-gib"
    elif args.log:
        non_kv, src = _non_kv_from_log(args.log)
    else:
        if not _port_free():
            print("[可行性] 8000 端口已被占用（有引擎在跑），请先停止后再试")
            return 2
        measured = _deploy_and_measure(prof)
        if measured is None:
            print(
                f"[可行性] {MAX_ATTEMPTS} 次部署均未成功（池可能过大或环境不稳定）；"
                "可改用 --log / --non-kv-gib"
            )
            return 1
        non_kv, src = measured

    need = non_kv + target_pool / (1024**3)
    verdict = "OK" if need <= args.gpu_gib else "OOM RISK"
    print("\n".join(_header(s, prof)))
    print(f"\n[可行性] 非KV = {non_kv:.2f} GiB ({src})")
    print(f"         安全池({target_len:,}) = {_gib(target_pool)}")
    print(
        f"         合计 {need:.2f} <= 可用 {args.gpu_gib:.2f} GiB -> {verdict}"
    )
    return 0 if verdict == "OK" else 1


# --------------------------------------------------------------------------
# Feasibility helpers
# --------------------------------------------------------------------------
def _non_kv_from_log(log: str) -> tuple[float, str]:
    with open(log) as f:
        for line in f:
            m = _LOAD_RE.search(line)
            if m:
                w = float(m.group(1))
                return w + NON_KV_BASELINE_GIB, (
                    f"权重 {w} GiB + {NON_KV_BASELINE_GIB} 基线（估算）"
                )
    raise SystemExit(f"日志中无 'Model loading took': {log}")


def _port_free() -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", 8000)) != 0


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _clean_shm() -> None:
    for pat in ("/dev/shm/vllm_offload_", "/dev/shm/sem.mp-"):
        for name in os.listdir("/dev/shm"):
            if name.startswith(os.path.basename(pat)):
                try:
                    os.remove(os.path.join("/dev/shm", name))
                except OSError:
                    pass


def _teardown(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
    for pat in ("VLLM::", "vllm.entrypoints", "resource_tracker"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(3)
    _clean_shm()


def _gpu_used_gib() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    vals = [int(x) for x in out.split()]
    return max(vals) / 1024 if vals else 0.0


def _deploy_and_measure(prof: Profile) -> tuple[float, str] | None:
    """Launch the profile, measure non-KV, tear it down. None if never healthy.

    Any startup failure is retried: the warmup transiently OOMs or throws
    ``CUDA error: invalid argument`` on this platform, so a single failure does
    not mean the pool is infeasible.
    """
    cwd = os.path.dirname(os.path.abspath(prof.path)) or "."
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[可行性] 部署尝试 {attempt}/{MAX_ATTEMPTS} ...")
        _clean_shm()
        with open(FEASIBLE_LOG, "w") as logf:
            proc = subprocess.Popen(
                ["bash", os.path.abspath(prof.path)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                start_new_session=True,
            )
        healthy = False
        reason = ""
        for _ in range(READY_TIMEOUT_S // POLL_SECONDS):
            time.sleep(POLL_SECONDS)
            if _healthy():
                healthy = True
                break
            text = _read(FEASIBLE_LOG)
            if _OOM_RE.search(text):
                reason = "OOM"
                break
            if _TRANSIENT_RE.search(text):
                reason = "瞬态错误"
                break
        if healthy:
            used = _gpu_used_gib()
            pool_gib = prof.pool_bytes / (1024**3)
            non_kv = used - pool_gib
            weights = _parse_weights(FEASIBLE_LOG)
            _teardown(proc)
            return non_kv, (
                f"实测 used {used:.2f} GiB - 池 {pool_gib:.2f} GiB"
                f"（权重 {weights} GiB）"
            )
        _teardown(proc)
        print(f"[可行性] 尝试 {attempt} 未健康（{reason or '超时'}），重试")
    return None


def _read(path: str) -> str:
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _parse_weights(log: str) -> str:
    m = _LOAD_RE.search(_read(log))
    return m.group(1) if m else "?"


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def cmd_self_test() -> int:
    fails: list[str] = []

    def check(name: str, got: object, want: object, tol: int = 0) -> None:
        ok = abs(got - want) <= tol if isinstance(want, int) else got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
        if not ok:
            fails.append(name)

    dims = load_model(None)
    s = Sizing(
        dims=dims,
        tp=2,
        num_spec=3,
        kv_dtype="fp8_e4m3",
        block_size=auto_block_size(dims, 2, "fp8_e4m3", 3),
        page_size=auto_block_size(dims, 2, "fp8_e4m3", 3)
        * attn_page_1_token(dims, 2, "fp8_e4m3"),
        group_size=_group_size(dims),
        mamba_groups=cdiv(dims.linear_layers, _group_size(dims)),
        mamba_blocks=cdiv(dims.linear_layers, _group_size(dims)) * (2 + 3),
        headroom=HEADROOM_BLOCKS,
        max_num_seqs=1,
    )
    check("block_size MTP3", s.block_size, 1600)
    check("group_size", s.group_size, 17)
    check("mamba_blocks MTP3", s.mamba_blocks, 15)
    # Ties the model to the deployment: the 128K profile runs a 3.0e9 pool and
    # the engine reported 144,584 tokens of capacity.
    check("min_pool @131072", s.min_pool(131072), 2701721600)
    check("safe_pool @131072", s.safe_pool(131072), 2785280000)
    check("max_safe_len @3.0e9", s.max_safe_len(3000000000), 142400, tol=3000)
    lo, hi = s.min_pool(131072), s.safe_pool(131072)
    short, tight, fits = (
        "不足（启动校验会报错）",
        "偏小（可启动，但满池会自我抢占）",
        "符合",
    )
    check("verdict 不足", _pool_verdict(2_600_000_000, lo, hi), short)
    check("verdict 偏小", _pool_verdict(2_750_000_000, lo, hi), tight)
    check("verdict 符合", _pool_verdict(3_000_000_000, lo, hi), fits)
    s1 = Sizing(
        dims=dims,
        tp=2,
        num_spec=1,
        kv_dtype="fp8_e4m3",
        block_size=auto_block_size(dims, 2, "fp8_e4m3", 1),
        page_size=0,
        group_size=_group_size(dims),
        mamba_groups=cdiv(dims.linear_layers, _group_size(dims)),
        mamba_blocks=0,
        headroom=0,
        max_num_seqs=1,
    )
    check("block_size MTP1", s1.block_size, 1584)
    check("mamba_blocks MTP1", s1.mamba_groups * (2 + 1), 9)
    if fails:
        print(f"\nSELF-TEST FAILED: {', '.join(fails)}")
        return 1
    print("\nSELF-TEST PASSED")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="诊断 vLLM 启动脚本的 KV 池与上下文是否匹配。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_script", nargs="?", help="启动脚本（如 run_....sh）")
    p.add_argument("--profile", help="启动脚本（同位置参数）")
    p.add_argument("--max-len", help="覆盖上下文，如 500k")
    p.add_argument("--pool-bytes", help="覆盖池，如 9.6e9")
    p.add_argument("--feasible", action="store_true", help="实际部署一次测 OOM")
    p.add_argument("--log", help="用已有启动日志估算非KV（不部署）")
    p.add_argument("--non-kv-gib", type=float, help="直接给非KV GiB（不部署）")
    p.add_argument("--gpu-gib", type=float, default=GPU_USABLE_GIB)
    adv = p.add_argument_group("高级（换模型/覆盖）")
    adv.add_argument("--model-config")
    adv.add_argument("--tp", type=int)
    adv.add_argument("--num-spec", type=int)
    adv.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES))
    adv.add_argument("--max-num-seqs", type=int)
    adv.add_argument("--block-size", type=int)
    adv.add_argument("--headroom-blocks", type=int)
    adv.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.self_test:
        return cmd_self_test()

    path = args.profile or args.run_script
    if not path:
        make_parser().print_help()
        return 2
    if not os.path.isfile(path):
        raise SystemExit(f"启动脚本不存在: {path}")

    prof = parse_profile(path)
    if args.max_len:
        args.max_len = parse_size(args.max_len)
    if args.pool_bytes:
        args.pool_bytes = parse_size(args.pool_bytes)

    s = build_sizing(prof, args)
    if args.feasible:
        return cmd_feasible(s, prof, args)
    return cmd_report(s, prof, args)


if __name__ == "__main__":
    sys.exit(main())
