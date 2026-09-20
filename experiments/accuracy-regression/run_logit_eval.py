#!/usr/bin/env python3
"""Run the logit-level evaluation against a running vLLM server.

  run_logit_eval.py --config B0 [--out raw/B0]

Dense:  prompt_logprobs=k over 8K-token segments  -> per-position top-k
Probes: long context + 1 generated token, logprobs=k -> single-position top-k

Stores compact npz per item so a later compare step can compute KL / top-1
agreement without keeping the (large) JSON responses around.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"


def post(url: str, payload: dict, timeout: int = 3600) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def dicts_to_arrays(per_pos: list[dict | None], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """per_pos: list of {token_id_str: {"logprob":..}} or None -> arrays."""
    n = len(per_pos)
    ids = np.full((n, k), -1, dtype=np.int32)
    lps = np.full((n, k), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, d in enumerate(per_pos):
        if not d:
            continue
        items = sorted(d.items(), key=lambda kv: kv[1]["logprob"], reverse=True)[:k]
        for j, (tid, lp) in enumerate(items):
            ids[i, j] = int(tid)
            lps[i, j] = lp["logprob"]
        valid[i] = True
    return ids, lps, valid


def run_dense(url: str, model: str, k: int, out_dir: Path, only: list[str] | None) -> None:
    domains = only or ["en", "zh", "code"]
    for dom in domains:
        segs = sorted((CORPUS / "dense" / dom).glob("seg_*.json"))
        for sp in segs:
            tokens = json.loads(sp.read_text())["tokens"]
            dst = out_dir / "dense" / dom / (sp.stem + ".npz")
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            resp = post(
                f"{url}/v1/completions",
                {
                    "model": model,
                    "prompt": tokens,
                    "max_tokens": 1,
                    "temperature": 0,
                    "prompt_logprobs": k,
                },
            )
            plp = resp["choices"][0]["prompt_logprobs"]
            ids, lps, valid = dicts_to_arrays(plp, k)
            np.savez_compressed(
                dst,
                tokens=np.asarray(tokens, dtype=np.int32),
                top_ids=ids,
                top_lp=lps,
                valid=valid,
            )
            print(
                f"[dense {dom} {sp.stem}] {len(tokens)} pos in {time.time()-t0:.1f}s",
                flush=True,
            )


def topk_str_to_arrays(d: dict, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Completion `logprobs` form: {"token_id:N": logprob_float, ...}."""
    items = []
    for key, val in d.items():
        if not key.startswith("token_id:"):
            continue
        items.append((int(key.split(":", 1)[1]), float(val)))
    items.sort(key=lambda x: x[1], reverse=True)
    ids = np.full((1, k), -1, dtype=np.int32)
    lps = np.full((1, k), np.nan, dtype=np.float32)
    for j, (tid, lp) in enumerate(items[:k]):
        ids[0, j] = tid
        lps[0, j] = lp
    return ids, lps, np.array([True])


def run_probes(url: str, model: str, k: int, out_dir: Path) -> None:
    for pp in sorted((CORPUS / "probes").glob("probe_*.json")):
        tokens = json.loads(pp.read_text())["tokens"]
        dst = out_dir / "probes" / (pp.stem + ".npz")
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        resp = post(
            f"{url}/v1/completions",
            {
                "model": model,
                "prompt": tokens,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": k,
                "return_tokens_as_token_ids": True,
            },
        )
        lp = resp["choices"][0]["logprobs"]
        d = lp["top_logprobs"][0] if lp and lp.get("top_logprobs") else {}
        ids, lps, valid = topk_str_to_arrays(d, k)
        np.savez_compressed(
            dst,
            tokens=np.asarray(tokens, dtype=np.int32),
            top_ids=ids,
            top_lp=lps,
            valid=valid,
        )
        print(
            f"[probe {pp.stem}] len={len(tokens)} in {time.time()-t0:.1f}s",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--domains", default=None, help="comma list, default all")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--skip-dense", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "raw" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    only = args.domains.split(",") if args.domains else None

    if not args.skip_dense:
        run_dense(args.url, args.model, args.k, out_dir, only)
    if not args.skip_probes:
        run_probes(args.url, args.model, args.k, out_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
