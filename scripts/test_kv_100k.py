#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""100k-pool APC / KV-eviction semantics test client.

Measures, per chat request:
  usage.prompt_tokens / usage.completion_tokens
  usage.prompt_tokens_details.cached_tokens  (contiguous-from-start cache hits)
  wall time
plus /metrics deltas, to empirically check:
  R1  intact history replayed -> full prefix reuse
  E1  after newer sessions evict only PART of the OLDEST session (its tail,
      per LRU + tail-first free ordering) -> replay still reuses surviving head
  E2  after the oldest session is FULLY evicted -> replay recomputes (~0 hits)

Sessions A/B/C/D are independent, non-repeating token streams (stateless
"replay the same prompt" simulates resuming an agent conversation whose
history bytes are unchanged).
"""
import argparse
import json
import os
import random
import re
import sys
import time

import requests
from openai import OpenAI
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR")
if not MODEL_DIR:
    raise SystemExit("set MODEL_PATH (or MODEL_DIR) to the model directory")
BLOCK_TOL = 4096  # generous tolerance for page-granularity (attention page ~1600)

WORD_SEED_BASE = [i for i in range(100000000)]


def make_text(tokenizer, target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    parts = []
    acc = []
    n = 0
    while True:
        acc.append("%08d" % rng.randrange(0, 100000000))
        if len(acc) >= 512:
            text = " ".join(acc)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            parts.append(text)
            n += len(ids)
            acc = []
            if n >= target_tokens + 64:
                break
    if acc:
        parts.append(" ".join(acc))
    full = " ".join(parts)
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def metric_line(metrics_text: str, name: str):
    for line in metrics_text.splitlines():
        if line.startswith(name):
            return line
    return None


def scrape_metrics(base: str):
    try:
        txt = requests.get(base.replace("/v1", "/metrics"), timeout=10).text
    except Exception as exc:
        return {"error": str(exc)}
    out = {}
    for key in [
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_misses_total",
        "vllm:prefix_cache_misses",
        "vllm:num_preemptions",
        "vllm:num_preemptions_total",
        "vllm:gpu_cache_usage_perc",
        "vllm:num_free_gpu_blocks",
        "vllm:num_gpu_blocks",
    ]:
        ln = metric_line(txt, key)
        if ln:
            out[key] = ln
    return out


class Test:
    def __init__(self, base, model, pool, tokenizer):
        self.client = OpenAI(base_url=base, api_key="EMPTY", timeout=1800)
        self.model = model
        self.pool = pool
        self.tok = tokenizer
        self.rows = []

    def chat(self, content: str, max_tokens: int = 1, note: str = ""):
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
        )
        wall = time.time() - t0
        usage = resp.usage
        cached = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details
            else None
        )
        row = {
            "note": note,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": cached,
            "wall_s": round(wall, 2),
        }
        self.rows.append(row)
        print(
            f"[{note}] prompt={usage.prompt_tokens} out={usage.completion_tokens} "
            f"cached={cached} wall={wall:.1f}s"
        )
        return row

    def run(self):
        P = self.pool
        L = 102400
        HA = int(0.45 * P)
        HB = int(0.35 * P)
        HC = int(0.40 * P)
        D1 = min(int(0.60 * P), L - 3000)
        D2 = D1
        print(f"pool(P)={P}  HA={HA} HB={HB} HC={HC} D1/D2={D1}")

        print("\n--- warmup ---")
        self.chat(make_text(self.tok, 1500, seed=1), note="warmup")

        print("\n--- build A (%.0fk) ---" % (HA / 1000))
        text_a = make_text(self.tok, HA, seed=101)
        m0 = scrape_metrics(self.client.base_url)
        r = self.chat(text_a, note="A0 first (expect cached=0)")
        self.assert_row(r, "A0_first_cached_0", r["cached_tokens"] in (0, None),
                        "cached_tokens should be 0 on first encounter")

        print("\n--- replay A intact (expect full prefix reuse) ---")
        r2 = self.chat(text_a, note="A0 replay intact")
        self.assert_row(r2, "A0_replay_full", r2["cached_tokens"] >= int(0.90 * HA),
                        f"full reuse expected >=~{int(0.90*HA)}, got {r2['cached_tokens']}")

        print("\n--- run B then C to fill pool & partially evict oldest(A) ---")
        self.chat(make_text(self.tok, HB, seed=202), note="B fill")
        r_c = self.chat(make_text(self.tok, HC, seed=303), note="C fill+evict")
        m1 = scrape_metrics(self.client.base_url)

        print("\n--- replay A after partial eviction ---")
        r3 = self.chat(text_a, note="A0 replay post-partial-eviction")
        mid_ok = r3["cached_tokens"] is not None and (
            int(0.08 * P) <= r3["cached_tokens"] <= int(0.92 * HA)
        )
        self.assert_row(
            r3,
            "E1_partial_reuse_discriminator",
            mid_ok,
            f"expect 0<cached<HA i.e. tail evicted&head kept, got {r3['cached_tokens']}",
        )

        print("\n--- run D1,D2 (new big sessions) to fully evict A ---")
        self.chat(make_text(self.tok, D1, seed=404), note="D1 evict")
        self.chat(make_text(self.tok, D2, seed=505), note="D2 evict")

        print("\n--- replay A after full eviction ---")
        r4 = self.chat(text_a, note="A0 replay post-full-eviction")
        full_ok = r4["cached_tokens"] is not None and r4["cached_tokens"] <= 3 * BLOCK_TOL
        self.assert_row(
            r4,
            "E2_full_recompute",
            full_ok,
            f"expect ~0 hits once head evicted, got {r4['cached_tokens']}",
        )

        self.metrics_delta = {"before_E1": m0, "after_E1": m1}
        return 0

    def assert_row(self, row, name, ok, why):
        row["assert"] = name
        row["pass"] = bool(ok)
        row["why"] = why
        print(f"    -> {name}: {'PASS' if ok else 'FAIL'}   ({why})")

    def save(self, out):
        with open(out, "w") as fh:
            json.dump(
                {"pool": self.pool, "rows": self.rows,
                 "metrics": getattr(self, "metrics_delta", {})},
                fh, indent=2,
            )
        print(f"\nresults -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model",
                    default=os.environ.get("SERVED_MODEL_NAME", "qwen38-27b"))
    ap.add_argument("--pool", type=int, required=True,
                    help="pool token capacity, read from server log 'GPU KV cache size'")
    ap.add_argument("--out",
                    default=os.environ.get("KV_TEST_OUT", "/tmp/test_kv_100k_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    t = Test(args.base, args.model, args.pool, tok)
    try:
        rc = t.run()
        t.save(args.out)
        sys.exit(rc)
    except Exception as exc:
        print(f"TEST FAILED with exception: {exc!r}", file=sys.stderr)
        t.save(args.out)
        sys.exit(1)


if __name__ == "__main__":
    main()
