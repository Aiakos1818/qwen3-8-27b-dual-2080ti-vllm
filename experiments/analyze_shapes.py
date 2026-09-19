#!/usr/bin/env python3
"""Breakdown + attention input shapes from the record_shapes trace."""

import collections
import gzip
import json
import sys

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
evs = data["traceEvents"]

kern = collections.Counter()
cnt = collections.Counter()
for e in evs:
    if e.get("cat") == "kernel" and e.get("dur"):
        kern[e["name"]] += e["dur"]
        cnt[e["name"]] += 1
tot = sum(kern.values())
print(f"  events: {len(evs)}  kernels: {sum(cnt.values())}  GPU busy: {tot / 1e6:.1f} s")

groups = collections.Counter()
for n, u in kern.items():
    ln = n.lower()
    if "batchprefill" in ln:
        groups["attention"] += u
    elif "marlin" in ln:
        groups["GEMM(int4 marlin)"] += u
    elif "gdn_forward" in ln or "flashqla" in ln:
        groups["GDN"] += u
    elif "nccl" in ln or "cross_device" in ln:
        groups["comms"] += u
    elif "humming" in ln:
        groups["head(int8)"] += u
    elif "gemvx" in ln or "gemm" in ln or "cutlass" in ln or "s1688" in ln:
        groups["GEMM(other)"] += u
    else:
        groups["other"] += u
print("  --- groups ---")
for g, u in groups.most_common():
    print(f"  {g:20s} {u / 1e6:9.2f}s {100 * u / tot:5.1f}%")

shapes = collections.Counter()
sdur = collections.Counter()
for e in evs:
    if e.get("cat") == "kernel" and "BatchPrefillWithPagedKVCacheKernel" in e.get("name", ""):
        dims = (e.get("args") or {}).get("Input Dims")
        if dims is None:
            continue
        key = tuple(tuple(d) if isinstance(d, list) else str(d) for d in dims)
        shapes[key] += 1
        sdur[key] += e["dur"]
print("  --- attention input shapes (count, total ms) ---")
for k, c in shapes.most_common(6):
    print(f"  x{c:<5d} {sdur[k] / 1e3:9.1f}ms  {str(k)[:200]}")
