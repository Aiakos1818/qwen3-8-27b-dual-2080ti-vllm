#!/usr/bin/env python
"""435k-scale chunked SSD park/resume check on the real NVMe.

S (~--turns*a-tok tokens) runs resident; T (larger) forces S to spill to SSD;
R resumes S with a tail. Validates that a session far larger than the CPU
staging pool is streamed in chunks both ways.

Usage: ssd_435k_check.py [--turns 26] [--a-tok 15000] [--t-turns 27]
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, user_msg  # noqa: E402

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=3600)
SSD_ROOT = os.environ.get("VLLM_SSD_ROOT", "/tmp/vllm_ssd")


def build(uid: int, n_turns: int, a_tok: int, tail: str | None = None):
    qs = [f"用户第{i}轮问题：请继续深入讲解该主题。" for i in range(n_turns + 1)]
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n_turns):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid + i, a_tok)))
    m.append(user_msg(qs[n_turns]))
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
    ap.add_argument("--turns", type=int, default=26)
    ap.add_argument("--a-tok", type=int, default=15000)
    ap.add_argument("--t-turns", type=int, default=27)
    args = ap.parse_args()

    S = build(500, args.turns, args.a_tok)
    T = build(600, args.t_turns, args.a_tok)
    R = build(500, args.turns, args.a_tok, tail="请只回答：OK")

    m0 = metrics()
    print("nvidia@start", nvidia_used(), flush=True)
    s = run("S resident", S, 1)
    t = run("T bigger (forces S spill)", T, 1)
    r = run("R resumed (SSD restore)", R, 32)
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
    ok = bool(r["cached"]) and r["cached"] > s["prompt"] * 0.9
    print(f"R cached={r['cached']} vs S prompt={s['prompt']} "
          f"(>=90% prefix restored: {ok})", flush=True)
    print("SSD-435K-" + ("DONE" if ok else "WEAK"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
