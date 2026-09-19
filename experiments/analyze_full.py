#!/usr/bin/env python3
"""Full analysis of the 250K prefill trace: step wall times + kernel breakdown."""

import collections
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
evs = data["traceEvents"]

steps = [
    e
    for e in evs
    if e.get("cat") == "cpu_op" and e.get("name", "").startswith("ProfilerStep")
]
print(f"  events: {len(evs)}   ProfilerStep events: {len(steps)}")
if steps:
    d = sorted(e["dur"] for e in steps)
    print(
        f"  step wall ms: {[round(x / 1e3, 1) for x in d]}   "
        f"mean={sum(d) / len(d) / 1e3:.1f} ms"
    )

kern = collections.Counter()
cnt = collections.Counter()
for e in evs:
    if e.get("cat") == "kernel" and e.get("dur"):
        kern[e["name"]] += e["dur"]
        cnt[e["name"]] += 1
tot = sum(kern.values())
print(f"  GPU busy: {tot / 1e6:.1f} s   kernel events: {sum(cnt.values())}")
print("  --- top 10 ---")
for n, u in kern.most_common(10):
    print(f"  {u / 1e6:9.2f}s {100 * u / tot:5.1f}% x{cnt[n]:<6d} {n[:76]}")

att = sum(u for n, u in kern.items() if "BatchPrefillWithPagedKVCacheKernel" in n)
att_cnt = sum(c for n, c in cnt.items() if "BatchPrefillWithPagedKVCacheKernel" in n)
print(f"  attention: {att / 1e6:.1f} s = {100 * att / tot:.1f}%  (x{att_cnt})")

# The profiler was armed with delay_iterations=240, so the captured steps are the
# last prefill chunks: 1024 tokens each at kv ~= 241..245 * 1024.
if steps:
    n_steps = len(steps)
    kv = [241 * 1024 + i * 1024 for i in range(n_steps)]
    flops = [4 * 1024 * k * 256 * 24 * 16 for k in kv]
    per_att = [att / n_steps] * n_steps
    tflops = [f / (att / n_steps) / 1e12 for f in flops]
    print(
        f"  attention TFLOPS (assuming 1024 q-tokens at kv={kv[0]}..{kv[-1]}): "
        f"{[round(t, 1) for t in tflops]}  mean={sum(tflops) / len(tflops):.1f}"
    )
    print(
        f"  attention share of step wall: "
        f"{100 * (att / n_steps) / (sum(steps) / len(steps)):.1f}%"
    )
