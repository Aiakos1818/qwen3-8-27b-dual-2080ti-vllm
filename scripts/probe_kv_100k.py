#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decisive probe: does KV eviction take the OLDEST session's head or tail?

Sequence (pool P~106288, max_model_len 102400):
  purge (100k new content)        -> wipe cache
  A0   (70k new content)          -> A becomes only resident
  A0 replay                       -> expect ~full hit (sanity)
  B    (20k, pops unused, no evict)
  C    (20k: pops ~16k leftover-unused + ~4k from A  [A is oldest])
  A0 replay  (measure cached_tokens)
      >0  => tail of A was evicted, head survived (tail-first eviction)
      =0  => A's head (block0..) was evicted => whole thing recomputes
  prefix probes to map surviving contiguous prefix
"""
import json
import os
import random
import sys
import time

from openai import OpenAI
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR")
if not MODEL_DIR:
    raise SystemExit("set MODEL_PATH (or MODEL_DIR) to the model directory")
BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen38-27b")
POOL = 106288
OUT = os.environ.get("KV_PROBE_OUT", "/tmp/probe_kv_100k_out.json")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=1800)


def make_ids(target: int, seed: int):
    rng = random.Random(seed)
    acc, out = [], []
    while True:
        acc.append("%08d" % rng.randrange(0, 100000000))
        if len(acc) >= 1024:
            out.extend(tok(" ".join(acc), add_special_tokens=False)["input_ids"])
            acc = []
            if len(out) > target + 128:
                break
    if acc:
        out.extend(tok(" ".join(acc), add_special_tokens=False)["input_ids"])
    return out[:target]


def send(ids, note):
    text = tok.decode(ids, skip_special_tokens=True)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": text}],
        max_tokens=1, temperature=0,
    )
    wall = time.time() - t0
    u = resp.usage
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    print(f"[{note}] prompt={u.prompt_tokens} out={u.completion_tokens} cached={cached} wall={wall:.1f}s")
    sys.stdout.flush()
    return u.prompt_tokens, cached


A = make_ids(70000, seed=101)
PURGE = make_ids(100000, seed=777)
B = make_ids(20000, seed=202)
C = make_ids(20000, seed=303)

send(PURGE, "purge(100k) wipe cache")
send(A, "A0 first (expect cached=0)")
send(A, "A0 replay intact (expect ~full)")
send(B, "B 20k fill-unused")
send(C, "C 20k evict~4k from A")
res = send(A, "A0 replay POST-C (discriminator)")
print(f"DISCRIMINATOR cached={res[1]}")

print("--- prefix probes of A (map surviving contiguous prefix) ---")
for plen in (2000, 8000, 20000, 40000, 60000):
    send(A[:plen], f"A-prefix {plen}")

json.dump({"pool": POOL, "a_len": len(A)}, open(OUT, "w"))
print("done")
