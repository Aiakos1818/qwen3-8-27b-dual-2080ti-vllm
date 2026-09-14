#!/usr/bin/env python
"""P2: 100k-scale two-tier SSD check on the real NVMe.

S (~84k) runs resident, T (~87k) forces S to spill to the SSD store, then S is
resumed with a tail. Reports cached tokens, wall time and the SSD metrics
deltas (write/read bytes) plus the on-disk size.
"""
import hashlib
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, user_msg  # noqa: E402

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=2400)
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


S = build(500, 6, 14000)
T = build(600, 6, 14500)
R = build(500, 6, 14000, tail="请只回答：OK")

m0 = metrics()
run("S resident", S, 1)
run("T bigger (forces S spill)", T, 1)
r = run("R resumed (SSD restore)", R, 32)
m1 = metrics()

print("METRICS", {
    "stores": m1.get("vllm:host_tier_ssd_stores_total", 0)
    - m0.get("vllm:host_tier_ssd_stores_total", 0),
    "restores": m1.get("vllm:host_tier_ssd_restores_total", 0)
    - m0.get("vllm:host_tier_ssd_restores_total", 0),
    "write_bytes": m1.get("vllm:host_tier_ssd_write_bytes_total", 0)
    - m0.get("vllm:host_tier_ssd_write_bytes_total", 0),
    "read_bytes": m1.get("vllm:host_tier_ssd_read_bytes_total", 0)
    - m0.get("vllm:host_tier_ssd_read_bytes_total", 0),
}, flush=True)
print("DISK_BYTES", du(SSD_ROOT), flush=True)
print("SSD-100K-DONE", flush=True)
