#!/usr/bin/env python3
"""Compare two configs' stored logit data (baseline vs quantised).

  compare.py --base B0 --other H8

Per domain (dense) and per length (probes) reports:
  top-1 / top-5 agreement, KL(base||other) over the union support with a
  uniform-tail approximation, mean coverage, and mean |dlogprob| on the true
  next token (dense only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VOCAB = 248320


def metrics(a, b, k: int, dense: bool) -> dict:
    ta, tb = a["top_ids"], b["top_ids"]
    la, lb = a["top_lp"], b["top_lp"]
    valid = a["valid"] & b["valid"]
    ta, tb, la, lb = ta[valid], tb[valid], la[valid], lb[valid]
    n = ta.shape[0]
    if n == 0:
        return {}

    top1 = float((ta[:, 0] == tb[:, 0]).mean())
    top5 = float(
        np.mean([len(set(ta[i, :5]) & set(tb[i, :5])) / 5 for i in range(n)])
    )

    pa_rows = [{int(t): float(l) for t, l in zip(ta[i], la[i]) if t >= 0} for i in range(n)]
    pb_rows = [{int(t): float(l) for t, l in zip(tb[i], lb[i]) if t >= 0} for i in range(n)]

    cov_a = np.array([sum(np.exp(v) for v in d.values()) for d in pa_rows])
    cov_b = np.array([sum(np.exp(v) for v in d.values()) for d in pb_rows])

    kls = np.empty(n)
    dlog = np.empty(n) if dense else None
    tokens = a["tokens"] if dense else None
    idxs = np.nonzero(valid)[0]

    for i in range(n):
        pa, pb = pa_rows[i], pb_rows[i]
        S = set(pa) & set(pb)
        if not S:
            kls[i] = np.nan
            if dense:
                dlog[i] = np.nan
            continue
        za = sum(np.exp(pa[t]) for t in S)
        zb = sum(np.exp(pb[t]) for t in S)
        lza, lzb = np.log(za), np.log(zb)
        kl = 0.0
        for t in S:
            pa_n = np.exp(pa[t] - lza)
            kl += pa_n * ((pa[t] - lza) - (pb[t] - lzb))
        kls[i] = kl
        if dense:
            pos = int(idxs[i])
            if pos == 0:
                dlog[i] = np.nan
            else:
                t = int(tokens[pos])
                lpa = pa.get(t)
                lpb = pb.get(t)
                if lpa is None:
                    lpa = float(np.min(la[i][la[i] > -np.inf])) if len(pa) >= k else -20.0
                if lpb is None:
                    lpb = float(np.min(lb[i][lb[i] > -np.inf])) if len(pb) >= k else -20.0
                dlog[i] = abs(lpa - lpb)

    out = {
        "n_pos": int(n),
        "top1_agree": top1,
        "top5_overlap": top5,
        "kl_mean": float(np.nanmean(kls)),
        "kl_median": float(np.nanmedian(kls)),
        "coverage_base": float(cov_a.mean()),
        "coverage_other": float(cov_b.mean()),
    }
    if dense and dlog is not None:
        out["dlogprob_true_token"] = float(np.nanmean(dlog))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="B0")
    ap.add_argument("--other", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_dir = ROOT / "raw" / args.base
    other_dir = ROOT / "raw" / args.other
    result: dict = {"base": args.base, "other": args.other, "k": args.k,
                    "dense": {}, "probes": {}}

    for dom in ["en", "zh", "code"]:
        agg = []
        for p in sorted((base_dir / "dense" / dom).glob("seg_*.npz")):
            rel = p.relative_to(base_dir)
            if not (other_dir / rel).exists():
                continue
            m = metrics(np.load(p), np.load(other_dir / rel), args.k, dense=True)
            if m:
                agg.append(m)
        if agg:
            keys = list(agg[0])
            result["dense"][dom] = {kk: float(np.mean([m[kk] for m in agg])) for kk in keys}

    for p in sorted((base_dir / "probes").glob("probe_*.npz")):
        rel = p.relative_to(base_dir)
        if not (other_dir / rel).exists():
            continue
        m = metrics(np.load(p), np.load(other_dir / rel), args.k, dense=False)
        if m:
            result["probes"].setdefault(p.stem.split("_")[1], []).append(m)
    for length, ms in result["probes"].items():
        keys = list(ms[0])
        agg = {kk: float(np.mean([m[kk] for m in ms])) for kk in keys}
        agg["n_probes"] = len(ms)
        result["probes"][length] = agg

    out = Path(args.out) if args.out else ROOT / "raw" / f"compare_{args.base}_vs_{args.other}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
