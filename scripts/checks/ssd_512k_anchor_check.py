#!/usr/bin/env python
"""512k full-length context + Mamba anchor deep revert (SSD offload).

S (--turns x --a-tok, default 32x16000 ~= 515k) fills the pool to the memory
ceiling. T (default 7x16000 ~= 112k, > the ~21k free) forces S to spill to the
real NVMe SSD. R resumes S with a tail (chunked restore). V reverts S to the
--keep-th turn (default 30 -> ~481k); with the durable window holding the last
K cadences, V must hit the anchor at --anchor (default 480000 = 15 x 32000;
MTP3 selects block_size=1600, cadence 32000).

The offload tier is required: with the pool ~96% full a revert would otherwise
evict the resident session (instead of reusing it) and recompute from 0.

Usage: ssd_512k_anchor_check.py [--turns 32] [--a-tok 16000] [--keep 30]
                                [--anchor 480000] [--t-turns 7]
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, user_msg  # noqa: E402

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=7200)


def build(uid: int, n_turns: int, a_tok: int, keep: int | None = None,
          edited: bool = False, tail: str | None = None):
    qs = [f"用户第{i}轮问题：请继续深入讲解该主题。" for i in range(n_turns + 1)]
    n = n_turns if keep is None else keep
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, a_tok)))
    m.append(user_msg(qs[n] + ("（改写重发）" if edited else "")))
    if tail:
        m.append(user_msg(tail))
    return m


def run(name: str, msgs, max_tokens: int):
    t0 = time.time()
    r = _client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=0
    )
    u = r.usage
    text = r.choices[0].message.content or ""
    rec = {
        "name": name,
        "prompt": u.prompt_tokens,
        "cached": u.prompt_tokens_details.cached_tokens
        if u.prompt_tokens_details
        else None,
        "wall": round(time.time() - t0, 2),
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
    print(rec, flush=True)
    return rec


def nvidia_used() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=32)
    ap.add_argument("--a-tok", type=int, default=16000)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--anchor", type=int, default=480000)
    ap.add_argument("--t-turns", type=int, default=7)
    args = ap.parse_args()

    S = build(1, args.turns, args.a_tok)
    T = build(2, args.t_turns, args.a_tok)
    R = build(1, args.turns, args.a_tok, tail="请只回答：OK")
    V = build(1, args.turns, args.a_tok, keep=args.keep, edited=True)

    print("nvidia@start", nvidia_used(), flush=True)
    s = run("S full-length resident", S, 1)
    t = run("T smaller (forces S spill)", T, 1)
    r = run("R resumed (SSD restore)", R, 32)
    v = run(f"V revert keep={args.keep} (after restore)", V, 1)
    print("nvidia@end", nvidia_used(), flush=True)

    ok_r = bool(r["cached"]) and r["cached"] > s["prompt"] * 0.9
    ok_v = bool(v["cached"]) and v["cached"] >= args.anchor - 3200
    print(f"R cached={r['cached']} vs S prompt={s['prompt']} "
          f"(>=90%: {ok_r})", flush=True)
    print(f"ANCHOR expect~{args.anchor} V={v['cached']} (hit: {ok_v})", flush=True)
    print("SSD-512K-ANCHOR-" + ("DONE" if (ok_r and ok_v) else "WEAK"), flush=True)
    return 0 if (ok_r and ok_v) else 1


if __name__ == "__main__":
    sys.exit(main())
