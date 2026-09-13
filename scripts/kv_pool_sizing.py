#!/usr/bin/env python3
"""Recommend a vLLM KV-cache pool size from a launch profile and a context.

The vLLM startup check sizes the attention group exactly to ``--max-model-len``,
leaving no room for the durable Mamba anchors or chunked-prefill copy-on-write.
A request that reaches that ceiling then self-preempts, which releases the
durable window and makes a later deep revert recompute. This tool mirrors the
vLLM arithmetic so a profile can be paired with a pool that keeps a safe margin.

All launch parameters are read from the run script (``--profile``); only the
target context (or pool, for the reverse) has to be supplied.

Sources mirrored (v0.27.1 fork):
  * block-size auto-selection: ``vllm/platforms/interface.py`` (~L904)
  * grouping heuristic:        ``vllm/v1/core/kv_cache_utils.py`` (~L1263)
  * pool requirement:          ``vllm/v1/core/kv_cache_utils.py`` (~L1959)
  * mamba align usage:         ``vllm/v1/kv_cache_interface.py`` (~L735)

Examples:
  kv_pool_sizing.py --profile run_..._512k_kv.sh --max-len 500k
  kv_pool_sizing.py --profile run_..._512k_kv.sh
  kv_pool_sizing.py --profile run_..._512k_kv.sh --pool-bytes 9.6e9
  kv_pool_sizing.py --profile run_..._512k_kv.sh --max-len 500k --feasible \
      --log fp8-kv-work/server_c5.log
  kv_pool_sizing.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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

# Fallback used when no profile/model config is given (Qwen3.8-27B).
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

# Runtime headroom heuristic: copy-on-write / transient blocks plus one pinned
# state block per (mamba group x cadence anchor). Yields 16 blocks for the
# default (3 groups, ANCHORS=3), matching the 435k/512k measurements.
COW_BLOCKS = 7
# Non-KV overhead on top of the weights: activations, CUDA graphs, NCCL and the
# startup transient (GiB). Calibrated so the AWQ-512k profile's known points
# (9.6e9 fits, 9.78e9 OOMs) come out right; it is an estimate (see --non-kv-gib).
NON_KV_BASELINE_GIB = 2.0
# Usable GPU memory after the driver reserve (GiB), for a 22528 MiB card.
GPU_USABLE_GIB = 21.5


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _gib(n: int) -> str:
    return f"{n / (1024**3):.3f} GiB"


def _mib(n: int) -> str:
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
_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_ARG_RE = re.compile(r"--([a-z0-9][a-z0-9-]*)(?:=|\s+)([^\s\\]+)")
_FLAG_RE = re.compile(r"--([a-z0-9][a-z0-9-]*)(?=\s|$)")


def _resolve(value: str) -> str:
    def repl(m: re.Match) -> str:
        default = m.group(2)
        if default is not None:
            return default
        return os.environ.get(m.group(1), "")

    return _BRACE_RE.sub(repl, value).strip().strip("'\"")


@dataclass
class Profile:
    path: str
    args: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)


def parse_profile(path: str) -> Profile:
    """Read (never execute) a launch script's CLI args and exports."""
    with open(path) as f:
        raw = f.read()
    text = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    text = text.replace("\\\n", " ")
    prof = Profile(path=path)
    for name, value in _ENV_RE.findall(text):
        prof.env[name] = _resolve(value)
    for name, value in _ARG_RE.findall(text):
        prof.args[name] = _resolve(value)
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
    mamba_state_dtype: str
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
        cfg, source = dict(QWEN38_DEFAULTS), "built-in Qwen3.8-27B defaults"
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
        mamba_state_dtype=d["mamba_ssm_dtype"],
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
    anchors: int
    max_num_seqs: int
    warnings: list[str] = field(default_factory=list)

    @property
    def slot_bytes(self) -> int:
        return self.group_size * self.page_size

    def forward(self, max_len: int) -> dict:
        attn_blocks = cdiv(max_len, self.block_size)
        base = attn_blocks + self.mamba_blocks
        conc = (self.max_num_seqs - 1) * (attn_blocks + self.mamba_blocks)
        return {
            "max_len": max_len,
            "attn_blocks": attn_blocks,
            "min_pool_bytes": self.slot_bytes * base,
            "safe_pool_bytes": self.slot_bytes
            * (base + self.headroom + conc),
        }

    def reverse(self, pool_bytes: int) -> dict:
        slots = pool_bytes // self.slot_bytes
        cap_blocks = max(0, slots - self.mamba_blocks)
        safe_blocks = max(0, cap_blocks - self.headroom)
        return {
            "pool_bytes": pool_bytes,
            "capacity_tokens": cap_blocks * self.block_size,
            "max_safe_len": safe_blocks * self.block_size,
        }

    def safe_pool_for(self, max_len: int) -> int:
        return self.forward(max_len)["safe_pool_bytes"]


def build_sizing(prof: Profile | None, args: argparse.Namespace) -> Sizing:
    p = prof.args if prof else {}
    e = prof.env if prof else {}
    flags = prof.flags if prof else set()

    model_path = args.model_config or p.get("model")
    dims = load_model(model_path)

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
    anchors = args.anchors
    if anchors is None:
        anchors = int(e.get("VLLM_MAMBA_CKPT_ANCHORS", 3) or 3)
    max_num_seqs = args.max_num_seqs or int(p.get("max-num-seqs", 1))

    block_size = args.block_size or auto_block_size(dims, tp, kv_dtype, num_spec)
    page_size = block_size * attn_page_1_token(dims, tp, kv_dtype)
    gsize = _group_size(dims)
    mgroups = cdiv(dims.linear_layers, gsize)
    align = "enable-prefix-caching" in flags or args.prefix_caching
    if align:
        mamba_blocks = mgroups * (2 + num_spec)
    else:
        mamba_blocks = mgroups * (1 + num_spec)

    warnings: list[str] = []
    if not align:
        warnings.append(
            "prefix caching off -> mamba cache mode 'none'; anchors disabled"
        )
    ckpt = int(e.get("VLLM_MAMBA_CKPT_TOKENS", 0) or 0)
    if align and ckpt == 0 and not args.durable:
        warnings.append("VLLM_MAMBA_CKPT_TOKENS=0 -> durable anchors off")
    if anchors < 3:
        warnings.append(
            f"ANCHORS={anchors} < 3: near-tail coverage ~{anchors} cadences; "
            "deeper reverts recompute"
        )
    if max_num_seqs > 1:
        warnings.append(
            f"max-num-seqs={max_num_seqs}: reserve {(max_num_seqs - 1)}x a full "
            "sequence (counted below)"
        )

    durable = align and ckpt != 0
    if args.headroom_blocks:
        headroom = args.headroom_blocks
    elif durable:
        headroom = COW_BLOCKS + mgroups * max(1, anchors)
    else:
        headroom = 3

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
        anchors=anchors,
        max_num_seqs=max_num_seqs,
        warnings=warnings,
    )


def _group_size(dims: ModelDims) -> int:
    mn = min(dims.attn_layers, dims.linear_layers)
    mx = max(dims.attn_layers, dims.linear_layers)
    return mx if mx < mn * 1.5 else mn


# --------------------------------------------------------------------------
# Feasibility (optional)
# --------------------------------------------------------------------------
_LOAD_RE = re.compile(r"Model loading took ([\d.]+) GiB")


def non_kv_gib(args: argparse.Namespace) -> tuple[float, str]:
    if args.non_kv_gib:
        return args.non_kv_gib, "given"
    if args.log:
        with open(args.log) as f:
            for line in f:
                m = _LOAD_RE.search(line)
                if m:
                    weights = float(m.group(1))
                    return weights + NON_KV_BASELINE_GIB, (
                        f"weights {weights} GiB + {NON_KV_BASELINE_GIB} baseline "
                        "(estimate)"
                    )
        raise SystemExit(f"no 'Model loading took' line in {args.log}")
    raise SystemExit("--feasible needs --log or --non-kv-gib")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def _summary(s: Sizing, prof: Profile | None, pool: int | None) -> list[str]:
    lines = []
    if prof:
        lines.append(f"profile   : {prof.path}")
    lines.append(f"model     : {s.dims.source}")
    lines.append(
        f"resolved  : tp={s.tp} kv={s.kv_dtype} MTP={s.num_spec} "
        f"anchors={s.anchors} max-num-seqs={s.max_num_seqs}"
    )
    lines.append(
        f"block_size: {s.block_size}  page={_mib(s.page_size)}  "
        f"group_size={s.group_size}  mamba_blocks={s.mamba_blocks}  "
        f"headroom={s.headroom}"
    )
    for w in s.warnings:
        lines.append(f"WARNING   : {w}")
    return lines


def _feasibility(s: Sizing, args: argparse.Namespace, safe_pool: int) -> list[str]:
    if not args.feasible:
        return []
    nk, src = non_kv_gib(args)
    total = args.gpu_gib
    need = nk + safe_pool / (1024**3)
    ok = need <= total
    return [
        f"feasible  : non-KV={nk:.2f} GiB ({src}) + "
        f"pool={safe_pool / (1024**3):.2f} GiB"
        f" = {need:.2f} <= {total:.2f} GiB -> {'OK' if ok else 'OOM RISK'}",
    ]


def cmd_forward(s: Sizing, prof: Profile | None, args: argparse.Namespace) -> int:
    r = s.forward(args.max_len)
    out = _summary(s, prof, None) + [
        "",
        f"target    : {_fmt(r['max_len'])} tokens ({r['attn_blocks']} blocks)",
        f"min pool  : {_fmt(r['min_pool_bytes'])}",
        f"safe pool : {_fmt(r['safe_pool_bytes'])}",
        f"recommend : --kv-cache-memory-bytes {r['safe_pool_bytes']}",
    ]
    out += _feasibility(s, args, r["safe_pool_bytes"])
    print("\n".join(out))
    return 0


def cmd_check(s: Sizing, prof: Profile, args: argparse.Namespace) -> int:
    p = prof.args
    max_len = args.max_len or int(p.get("max-model-len", 0))
    pool = args.pool_bytes or int(p.get("kv-cache-memory-bytes", 0) or 0)
    rev = s.reverse(pool) if pool else None
    fwd = s.forward(max_len) if max_len else None
    out = _summary(s, prof, pool) + [""]
    if max_len:
        out.append(f"max-model-len = {_fmt(max_len)}")
    if pool:
        out.append(f"kv-cache-memory-bytes = {_fmt(pool)}")
    if not pool:
        out.append("no kv-cache-memory-bytes in profile (auto profiling?)")
    if rev and max_len:
        ok = rev["max_safe_len"] >= max_len
        out.append(f"safe max  : {_fmt(rev['max_safe_len'])} tokens")
        if ok:
            out.append("verdict   : SAFE")
        else:
            need = fwd["safe_pool_bytes"] if fwd else 0
            out.append(
                f"verdict   : UNSAFE ({_fmt(max_len)} > {_fmt(rev['max_safe_len'])})"
                f" -> lower --max-model-len to <= {_fmt(rev['max_safe_len'])},"
                f" or raise pool to {_fmt(need)}"
            )
        out += _feasibility(s, args, fwd["safe_pool_bytes"] if fwd else 0)
    print("\n".join(out))
    if rev and max_len:
        return 0 if rev["max_safe_len"] >= max_len else 1
    return 0


def cmd_reverse(s: Sizing, prof: Profile | None, args: argparse.Namespace) -> int:
    rev = s.reverse(args.pool_bytes)
    out = _summary(s, prof, args.pool_bytes) + [
        "",
        f"pool      : {_fmt(args.pool_bytes)}",
        f"capacity  : ~{_fmt(rev['capacity_tokens'])} tokens",
        f"safe max  : {_fmt(rev['max_safe_len'])} tokens",
    ]
    print("\n".join(out))
    return 0


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
        mamba_blocks=0,
        headroom=COW_BLOCKS + cdiv(dims.linear_layers, _group_size(dims)) * 3,
        anchors=3,
        max_num_seqs=1,
    )
    s.mamba_blocks = s.mamba_groups * (2 + 3)
    check("block_size MTP3", s.block_size, 1600)
    check("group_size", s.group_size, 17)
    check("mamba_blocks MTP3", s.mamba_blocks, 15)
    check("headroom (anchors=3)", s.headroom, 16)
    f = s.forward(524288)
    check("min pool @524288", f["min_pool_bytes"], 9553510400)
    check("safe pool @524288", f["safe_pool_bytes"], 9999155200)
    rev = s.reverse(9600000000)
    check("max_safe @9.6e9", rev["max_safe_len"], 500800)
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
        anchors=3,
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
        description="Recommend a vLLM KV-cache pool from a launch profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--profile", help="launch run script (params read from it)")
    p.add_argument("--max-len", help="target context, e.g. 500k (forward)")
    p.add_argument("--pool-bytes", help="pool size, e.g. 9.6e9 (reverse)")
    p.add_argument("--feasible", action="store_true", help="check GPU OOM risk")
    p.add_argument("--log", help="engine startup log (for --feasible)")
    p.add_argument("--non-kv-gib", type=float, help="exact non-KV GiB (--feasible)")
    p.add_argument("--gpu-gib", type=float, default=GPU_USABLE_GIB,
                   help="usable GPU GiB")
    adv = p.add_argument_group("advanced (override profile / other models)")
    adv.add_argument("--model-config", help="model dir or config.json")
    adv.add_argument("--tp", type=int, help="tensor parallel size")
    adv.add_argument("--num-spec", type=int, help="MTP spec tokens")
    adv.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES))
    adv.add_argument("--anchors", type=int, help="VLLM_MAMBA_CKPT_ANCHORS")
    adv.add_argument("--max-num-seqs", type=int)
    adv.add_argument("--block-size", type=int, help="override auto block size")
    adv.add_argument("--headroom-blocks", type=int, help="override headroom")
    adv.add_argument("--prefix-caching", action="store_true")
    adv.add_argument("--durable", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.self_test:
        return cmd_self_test()

    prof = parse_profile(args.profile) if args.profile else None
    if args.max_len:
        args.max_len = parse_size(args.max_len)
    if args.pool_bytes:
        args.pool_bytes = parse_size(args.pool_bytes)

    s = build_sizing(prof, args)
    if args.max_len:
        return cmd_forward(s, prof, args)
    if args.pool_bytes:
        return cmd_reverse(s, prof, args)
    if prof:
        return cmd_check(s, prof, args)
    make_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
