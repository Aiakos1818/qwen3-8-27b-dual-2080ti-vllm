#!/usr/bin/env python3
"""Summarize a torch profiler trace: kernel time by name + per-step spans."""
import collections, gzip, json, re, sys

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
evs = data["traceEvents"]

kern = collections.Counter()
kern_cnt = collections.Counter()
for e in evs:
    if e.get("cat") == "kernel" and e.get("dur"):
        kern[e["name"]] += e["dur"]
        kern_cnt[e["name"]] += 1
tot = sum(kern.values())

steps = [(e["name"], e["dur"]) for e in evs
         if e.get("cat") == "cpu_op" and e.get("name", "").startswith("ProfilerStep")]

def cat(name):
    n = name.lower()
    if "batchprefill" in n or "merge_states" in n or "flash_fwd" in n:
        return "attention(flashinfer)"
    if "flashqla" in n or "gdn" in n or "chunk" in n and "fused" in n:
        return "GDN(flashqla)"
    if "marlin" in n:
        return "GEMM(marlin int4)"
    if "gemvx" in n or "nvjet" in n or "gemm" in n or "cutlass" in n or "sm90" in n or "ampere" in n or "s16816" in n:
        return "GEMM(other)"
    if "allreduce" in n or "nccl" in n or "reduce_scatter" in n or "all_gather" in n:
        return "comms"
    if "elementwise" in n or "rms" in n or "norm" in n or "vectorized" in n or "silu" in n or "act" in n:
        return "norm/act/ew"
    return "other"

groups = collections.Counter()
for name, us in kern.items():
    groups[cat(name)] += us

print(f"  kernel events: {sum(kern_cnt.values())}  GPU busy: {tot/1e6:.3f} s")
if steps:
    d = [x[1] for x in steps]
    print(f"  profiled steps: {len(d)}  wall: {[f'{x/1e3:.0f}ms' for x in d]}  mean={sum(d)/len(d)/1e3:.1f} ms")
print(f"  {'category':24s} {'time(s)':>9s} {'share':>7s}")
for g, us in groups.most_common():
    print(f"  {g:24s} {us/1e6:9.3f} {100*us/tot:6.1f}%")
print(f"  {'TOTAL':24s} {tot/1e6:9.3f} {100:6.1f}%")
print("  --- top 18 kernels ---")
for name, us in kern.most_common(18):
    print(f"  {us/1e6:9.3f}s {100*us/tot:5.1f}% x{kern_cnt[name]:<5d} {name[:82]}")
