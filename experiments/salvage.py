#!/usr/bin/env python3
"""Salvage kernel stats from a truncated torch profiler trace (streaming parse)."""

import collections
import gzip
import json
import sys

path = sys.argv[1]
raw = b""
with gzip.open(path) as f:
    while True:
        try:
            b = f.read(1 << 20)
        except EOFError:
            break
        if not b:
            break
        raw += b

txt = raw.decode("utf-8", "replace")
i = txt.find('"traceEvents"')
j = txt.find("[", i)
dec = json.JSONDecoder()
objs, k = [], j + 1
while k < len(txt):
    while k < len(txt) and txt[k] in " \n\r\t,":
        k += 1
    if k >= len(txt) or txt[k] == "]":
        break
    try:
        obj, end = dec.raw_decode(txt, k)
    except json.JSONDecodeError:
        break
    objs.append(obj)
    k = end

kern = collections.Counter()
cnt = collections.Counter()
for e in objs:
    if e.get("cat") == "kernel" and e.get("dur"):
        kern[e["name"]] += e["dur"]
        cnt[e["name"]] += 1

steps = [
    e["dur"]
    for e in objs
    if e.get("cat") == "cpu_op" and e.get("name", "").startswith("ProfilerStep")
]
tot = sum(kern.values())
print(
    f"  recovered events: {len(objs)}  kernel events: {sum(cnt.values())}  "
    f"GPU busy: {tot / 1e6:.3f} s"
)
if steps:
    print(
        f"  profiled steps: {len(steps)}  wall: {[f'{x / 1e3:.0f}ms' for x in steps]}  "
        f"mean={sum(steps) / len(steps) / 1e3:.1f} ms"
    )
print("  --- top 22 kernels ---")
for name, us in kern.most_common(22):
    print(f"  {us / 1e6:9.3f}s {100 * us / tot:5.1f}% x{cnt[name]:<5d} {name[:84]}")
