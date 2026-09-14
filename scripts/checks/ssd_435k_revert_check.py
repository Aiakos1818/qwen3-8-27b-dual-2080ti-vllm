#!/usr/bin/env python
"""435k-scale post-anchor-fix check: resident + post-restore deep revert.

S (--turns x --a-tok, default 24x16000 ~= 384k) runs resident. V0 reverts S to
the --keep-th cadence (~352k) while resident (tests the eagle-drop / head-free
anchor fixes). T (default 10x16000 ~= 160k, much smaller than the 407k in
ssd_435k_check.py) forces S to spill to the real NVMe SSD. R resumes S (chunked
SSD restore). V2 reverts S to the same cadence after the restore (tests the
resumed-session anchor re-claim).

Expected: V0 and V2 cached ~= keep*a_tok (e.g. 350400/352000), not 0; R cached
> 0.9 * S prompt. This is the 435k regression check for the three anchor fixes.

Usage: ssd_435k_revert_check.py [--turns 24] [--a-tok 16000] [--keep 22]
                                [--t-turns 10]
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, user_msg  # noqa: E402

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=3600)
SSD_ROOT = os.environ.get("VLLM_SSD_ROOT", "/tmp/vllm_ssd")


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


def metrics() -> dict[str, float]:
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for m in re.finditer(
        r"^(vllm:host_tier_ssd_\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M
    ):
        out[m.group(1)] = float(m.group(2))
    return out


def du(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


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
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--a-tok", type=int, default=16000)
    ap.add_argument("--keep", type=int, default=22)
    ap.add_argument("--t-turns", type=int, default=10)
    args = ap.parse_args()

    S = build(1, args.turns, args.a_tok)
    V = build(1, args.turns, args.a_tok, keep=args.keep, edited=True)
    T = build(2, args.t_turns, args.a_tok)
    R = build(1, args.turns, args.a_tok, tail="请只回答：OK")
    target = args.keep * args.a_tok

    m0 = metrics()
    print("nvidia@start", nvidia_used(), flush=True)
    s = run("S resident", S, 1)
    v0 = run(f"V0 revert keep={args.keep} (resident)", V, 1)
    t = run("T smaller (forces S spill)", T, 1)
    r = run("R resumed (SSD restore)", R, 32)
    v2 = run(f"V2 revert keep={args.keep} (after restore)", V, 1)
    m1 = metrics()
    print("nvidia@end", nvidia_used(), flush=True)

    print("METRICS", {
        "stores": m1.get("vllm:host_tier_ssd_stores_total", 0)
        - m0.get("vllm:host_tier_ssd_stores_total", 0),
        "restores": m1.get("vllm:host_tier_ssd_restores_total", 0)
        - m0.get("vllm:host_tier_ssd_restores_total", 0),
        "write_bytes": m1.get("vllm:host_tier_ssd_write_bytes_total", 0)
        - m0.get("vllm:host_tier_ssd_write_bytes_total", 0),
        "read_bytes": m1.get("vllm:host_tier_ssd_read_bytes_total", 0)
        - m0.get("vllm:host_tier_ssd_read_bytes_total", 0),
        "sessions": m1.get("vllm:host_tier_ssd_sessions", 0),
    }, flush=True)
    print("DISK_BYTES", du(SSD_ROOT), flush=True)

    ok_r = bool(r["cached"]) and r["cached"] > s["prompt"] * 0.9
    ok_v0 = bool(v0["cached"]) and v0["cached"] > target * 0.95
    ok_v2 = bool(v2["cached"]) and v2["cached"] > target * 0.95
    print(f"R cached={r['cached']} vs S prompt={s['prompt']} "
          f"(>=90% prefix restored: {ok_r})", flush=True)
    print(f"ANCHOR target~{target} V0={v0['cached']} V2={v2['cached']} "
          f"(V0 hit: {ok_v0}, V2 hit: {ok_v2})", flush=True)
    print("SSD-435K-REVERT-" + ("DONE" if (ok_r and ok_v0 and ok_v2) else "WEAK"),
          flush=True)
    return 0 if (ok_r and ok_v0 and ok_v2) else 1


if __name__ == "__main__":
    sys.exit(main())
