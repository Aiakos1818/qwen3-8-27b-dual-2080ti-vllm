#!/usr/bin/env python3
"""Size the vLLM KV-cache pool for a hybrid full+linear-attention model.

Reproduces the arithmetic vLLM performs at startup so a ``--max-model-len``
can be paired with a ``--kv-cache-memory-bytes`` that still leaves headroom.
Without that headroom a request that reaches the pool ceiling self-preempts,
which releases its durable Mamba anchors and makes a later deep revert
recompute the whole prefix.

Sources mirrored (v0.27.1 fork):
  * block-size auto-selection: ``vllm/platforms/interface.py`` (~L904)
  * grouping heuristic:        ``vllm/v1/core/kv_cache_utils.py`` (~L1263)
  * pool requirement:          ``vllm/v1/core/kv_cache_utils.py`` (~L1959)
  * mamba align usage:         ``vllm/v1/kv_cache_interface.py`` (~L735)

Pure stdlib, no vLLM import. Model constants come from a model ``config.json``
(``text_config``) or the built-in Qwen3.8-27B defaults.

Examples:
  # forward: what pool does 512k need?
  kv_pool_sizing.py --max-len 512000 --num-spec 3
  # reverse: what can a 9.6e9 pool safely serve?
  kv_pool_sizing.py --pool-bytes 9600000000 --num-spec 3
  # cross-check against a startup log
  kv_pool_sizing.py --max-len 524288 --verify-log server.log
  # regression against known measurements
  kv_pool_sizing.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

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

# Fallback used when no --model-config is given (Qwen3.8-27B, see
# docs/kv-optimization/GPU_MEMORY_CALCULATION.md appendix B).
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


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(-a // b)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _gib(n: int) -> str:
    return f"{n / (1024**3):.3f} GiB"


def _mib(n: int) -> str:
    return f"{n / (1024**2):.3f} MiB"


@dataclass
class ModelDims:
    """Resolved per-model constants needed for the pool arithmetic."""

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
    """Load the text/LLM sub-config from a model dir or config.json."""
    cfg_path = path
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        raw = json.load(f)
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(raw.get(key), dict):
            return raw[key], cfg_path
    return raw, cfg_path


def load_model(path: str | None) -> ModelDims:
    """Resolve model dimensions from config.json or the built-in defaults."""
    if not path:
        d = dict(QWEN38_DEFAULTS)
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
            source="built-in Qwen3.8-27B defaults",
        )

    cfg, cfg_path = _resolve_config(path)
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
        source=cfg_path,
    )


def _layer_counts(cfg: dict) -> tuple[int, int]:
    """Return (full_attention_layers + mtp, linear_attention_layers)."""
    mtp = int(cfg.get("mtp_num_hidden_layers", 0) or 0)
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum("full" in str(t) for t in layer_types)
        linear = len(layer_types) - full
        return full + mtp, linear
    n = int(cfg["num_hidden_layers"])
    interval = int(cfg.get("full_attention_interval", 1) or 1)
    full = n // interval if interval > 0 else n
    return full + mtp, n - full


def group_size(dims: ModelDims) -> int:
    """Mirror kv_cache_utils grouping: uniform group size across layer types."""
    mn = min(dims.attn_layers, dims.linear_layers)
    mx = max(dims.attn_layers, dims.linear_layers)
    return mx if mx < mn * 1.5 else mn


def attn_page_size_1_token(dims: ModelDims, tp: int, kv_dtype: str) -> int:
    """Bytes per token for one full-attention layer on one rank."""
    return 2 * (dims.num_kv_heads // tp) * dims.head_dim * DTYPE_BYTES[kv_dtype]


def mamba_page_size(
    dims: ModelDims,
    tp: int,
    num_spec: int,
    conv_dtype: str,
    state_dtype: str,
) -> int:
    """Bytes of one mamba state page (conv + temporal), unpadded."""
    conv_dim = (
        dims.linear_key_head_dim * dims.linear_num_key_heads * 2
        + dims.linear_value_head_dim * dims.linear_num_value_heads
    )
    conv = (
        (dims.conv_kernel - 1 + num_spec)
        * (conv_dim // tp)
        * DTYPE_BYTES[conv_dtype]
    )
    temporal = (
        (dims.linear_num_value_heads // tp)
        * dims.linear_value_head_dim
        * dims.linear_key_head_dim
        * DTYPE_BYTES[state_dtype]
    )
    return conv + temporal


def attn_block_size(
    dims: ModelDims,
    tp: int,
    kv_dtype: str,
    num_spec: int,
    conv_dtype: str,
    state_dtype: str,
    alignment: int,
) -> int:
    """Mirror platforms/interface.py: smallest aligned block >= mamba page."""
    a = attn_page_size_1_token(dims, tp, kv_dtype)
    m = mamba_page_size(dims, tp, num_spec, conv_dtype, state_dtype)
    return alignment * cdiv(m, alignment * a)


@dataclass
class Sizing:
    dims: ModelDims
    tp: int
    num_spec: int
    kv_dtype: str
    conv_dtype: str
    state_dtype: str
    block_size: int
    page_size: int
    group_size: int
    mamba_groups: int
    mamba_blocks: int
    headroom_blocks: int

    @property
    def slot_bytes(self) -> int:
        return self.group_size * self.page_size

    def forward(self, max_len: int) -> dict:
        attn_blocks = cdiv(max_len, self.block_size)
        base = attn_blocks + self.mamba_blocks
        return {
            "max_len": max_len,
            "attn_blocks": attn_blocks,
            "mamba_blocks": self.mamba_blocks,
            "min_pool_bytes": self.slot_bytes * base,
            "safe_pool_bytes": self.slot_bytes * (base + self.headroom_blocks),
        }

    def reverse(self, pool_bytes: int) -> dict:
        slots = pool_bytes // self.slot_bytes
        cap_blocks = max(0, slots - self.mamba_blocks)
        safe_blocks = max(0, cap_blocks - self.headroom_blocks)
        return {
            "pool_bytes": pool_bytes,
            "slots": slots,
            "capacity_tokens": cap_blocks * self.block_size,
            "max_safe_len": safe_blocks * self.block_size,
        }


def build_sizing(args: argparse.Namespace) -> Sizing:
    dims = load_model(args.model_config)
    alignment = args.alignment
    block_size = args.block_size or attn_block_size(
        dims,
        args.tp,
        args.kv_dtype,
        args.num_spec,
        args.mamba_conv_dtype,
        args.mamba_state_dtype,
        alignment,
    )
    page_size = block_size * attn_page_size_1_token(dims, args.tp, args.kv_dtype)
    gsize = group_size(dims)
    mgroups = cdiv(dims.linear_layers, gsize)
    mamba_blocks = mgroups * (2 + args.num_spec)
    return Sizing(
        dims=dims,
        tp=args.tp,
        num_spec=args.num_spec,
        kv_dtype=args.kv_dtype,
        conv_dtype=args.mamba_conv_dtype,
        state_dtype=args.mamba_state_dtype,
        block_size=block_size,
        page_size=page_size,
        group_size=gsize,
        mamba_groups=mgroups,
        mamba_blocks=mamba_blocks,
        headroom_blocks=args.headroom_blocks,
    )


def _header(s: Sizing) -> list[str]:
    return [
        f"model        : {s.dims.source}",
        f"layers       : {s.dims.attn_layers} full-attn (incl. MTP)"
        f" + {s.dims.linear_layers} linear",
        f"block_size   : {s.block_size} tokens"
        f"   page_size: {_fmt(s.page_size)} B ({_mib(s.page_size)})",
        f"group_size   : {s.group_size}"
        f"   mamba_groups: {s.mamba_groups}"
        f"   mamba_blocks: {s.mamba_blocks}",
        f"slot_bytes   : {_fmt(s.slot_bytes)} B"
        f" (= group_size x page_size)",
        f"tp={s.tp} num_spec={s.num_spec} kv_dtype={s.kv_dtype}"
        f" conv={s.conv_dtype} state={s.state_dtype}"
        f" headroom={s.headroom_blocks} blocks",
    ]


def cmd_forward(s: Sizing, args: argparse.Namespace) -> int:
    r = s.forward(args.max_len)
    out = _header(s) + [
        "",
        f"max_len      : {_fmt(r['max_len'])} tokens"
        f"  ({r['attn_blocks']} attn blocks)",
        f"min_pool     : {_fmt(r['min_pool_bytes'])} B"
        f" ({_gib(r['min_pool_bytes'])})   [startup check]",
        f"safe_pool    : {_fmt(r['safe_pool_bytes'])} B"
        f" ({_gib(r['safe_pool_bytes'])})   [+{s.headroom_blocks} blocks]",
        "",
        "recommend    : --kv-cache-memory-bytes "
        f"{r['safe_pool_bytes']}",
    ]
    if args.json:
        print(json.dumps({**r, "recommended_pool_bytes": r["safe_pool_bytes"]}))
    else:
        print("\n".join(out))
    return 0


def cmd_reverse(s: Sizing, args: argparse.Namespace) -> int:
    r = s.reverse(args.pool_bytes)
    verdict = "SAFE" if r["max_safe_len"] >= args.check_len else "UNSAFE"
    out = _header(s) + [
        "",
        f"pool_bytes   : {_fmt(r['pool_bytes'])} B"
        f" ({_gib(r['pool_bytes'])})  slots={r['slots']}",
        f"capacity     : ~{_fmt(r['capacity_tokens'])} tokens",
        f"max_safe_len : {_fmt(r['max_safe_len'])} tokens"
        f"   [-{s.headroom_blocks} blocks headroom]",
    ]
    if args.check_len is not None:
        out.append(
            f"check        : max_len {_fmt(args.check_len)}"
            f" -> {verdict}"
        )
    if args.json:
        print(json.dumps({**r, "check_len": args.check_len, "verdict": verdict}))
    else:
        print("\n".join(out))
    return 0 if verdict == "SAFE" else 1


def cmd_verify_log(s: Sizing, args: argparse.Namespace) -> int:
    block_re = re.compile(r"attention block size to (\d+) tokens")
    cap_re = re.compile(r"GPU KV cache size:\s*([\d,]+) tokens")
    block_log = cap_log = None
    with open(args.verify_log) as f:
        for line in f:
            m = block_re.search(line)
            if m and block_log is None:
                block_log = int(m.group(1))
            m = cap_re.search(line)
            if m and cap_log is None:
                cap_log = int(m.group(1).replace(",", ""))
    print("\n".join(_header(s)))
    print("")
    print(f"log block_size    : {block_log}")
    print(f"calc block_size   : {s.block_size}")
    ok = True
    if block_log is not None and block_log != s.block_size:
        ok = False
        print("  -> MISMATCH (pass --block-size to override)")
    if cap_log is not None and args.pool_bytes:
        r = s.reverse(args.pool_bytes)
        print(f"log capacity      : {_fmt(cap_log)} tokens")
        print(f"calc capacity     : ~{_fmt(r['capacity_tokens'])} tokens")
    return 0 if ok else 1


def cmd_self_test() -> int:
    """Regression against measurements in GPU_MEMORY_CALCULATION.md."""
    failures: list[str] = []

    def check(name: str, got: object, want: object, tol: float = 0.0) -> None:
        if isinstance(want, int) and isinstance(got, int):
            ok = got == want if tol == 0 else abs(got - want) <= tol
        else:
            ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
        if not ok:
            failures.append(name)

    dims = load_model(None)
    s3 = Sizing(
        dims=dims,
        tp=2,
        num_spec=3,
        kv_dtype="fp8_e4m3",
        conv_dtype="fp16",
        state_dtype="fp32",
        block_size=attn_block_size(dims, 2, "fp8_e4m3", 3, "fp16", "fp32", 16),
        page_size=0,
        group_size=group_size(dims),
        mamba_groups=cdiv(dims.linear_layers, group_size(dims)),
        mamba_blocks=0,
        headroom_blocks=16,
    )
    s3.page_size = s3.block_size * attn_page_size_1_token(dims, 2, "fp8_e4m3")
    s3.mamba_blocks = s3.mamba_groups * (2 + 3)
    check("block_size MTP3", s3.block_size, 1600)
    check("page_size MTP3", s3.page_size, 1638400)
    check("group_size", s3.group_size, 17)
    check("mamba_groups", s3.mamba_groups, 3)
    check("mamba_blocks MTP3", s3.mamba_blocks, 15)

    f = s3.forward(524288)
    check("attn_blocks @524288", f["attn_blocks"], 328)
    check("min_pool @524288", f["min_pool_bytes"], 9553510400)
    check("safe_pool @524288", f["safe_pool_bytes"], 9999155200)

    rev = s3.reverse(9600000000)
    check("capacity @9.6e9", rev["capacity_tokens"], 525816, tol=2000)
    check("max_safe @9.6e9", rev["max_safe_len"], 500800)

    s1 = Sizing(
        dims=dims,
        tp=2,
        num_spec=1,
        kv_dtype="fp8_e4m3",
        conv_dtype="fp16",
        state_dtype="fp32",
        block_size=attn_block_size(dims, 2, "fp8_e4m3", 1, "fp16", "fp32", 16),
        page_size=0,
        group_size=group_size(dims),
        mamba_groups=cdiv(dims.linear_layers, group_size(dims)),
        mamba_blocks=0,
        headroom_blocks=16,
    )
    s1.page_size = s1.block_size * attn_page_size_1_token(dims, 2, "fp8_e4m3")
    s1.mamba_blocks = s1.mamba_groups * (2 + 1)
    check("block_size MTP1", s1.block_size, 1584)
    check("mamba_blocks MTP1", s1.mamba_blocks, 9)

    # Existing profile pools vs the safe requirement.
    for name, ml in (("435k", 435200), ("256k", 262144), ("100k", 102400)):
        need = s3.forward(ml)["safe_pool_bytes"]
        print(f"  [info] safe_pool for {name} ({ml}) = {_fmt(need)} B")

    if failures:
        print(f"\nSELF-TEST FAILED: {', '.join(failures)}")
        return 1
    print("\nSELF-TEST PASSED")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Size the vLLM KV-cache pool for a hybrid model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-len", type=int, help="context length (forward mode)")
    p.add_argument("--pool-bytes", type=int, help="pool bytes (reverse mode)")
    p.add_argument("--check-len", type=int, help="verdict for this length")
    p.add_argument("--num-spec", type=int, default=3, help="MTP spec tokens")
    p.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    p.add_argument("--kv-dtype", default="fp8_e4m3", choices=sorted(DTYPE_BYTES))
    p.add_argument("--mamba-conv-dtype", default="fp16", choices=sorted(DTYPE_BYTES))
    p.add_argument("--mamba-state-dtype", default="fp32", choices=sorted(DTYPE_BYTES))
    p.add_argument("--block-size", type=int, default=0, help="override (0=auto)")
    p.add_argument("--alignment", type=int, default=16, help="kernel block alignment")
    p.add_argument("--headroom-blocks", type=int, default=16)
    p.add_argument("--model-config", default=None, help="model dir or config.json")
    p.add_argument("--verify-log", default=None, help="server log to cross-check")
    p.add_argument("--json", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.self_test:
        return cmd_self_test()

    s = build_sizing(args)
    if args.verify_log:
        return cmd_verify_log(s, args)
    if args.max_len is not None:
        return cmd_forward(s, args)
    if args.pool_bytes is not None:
        return cmd_reverse(s, args)
    make_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
