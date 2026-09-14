#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto keep-alive validation (pool ~106k, min-pin=16k default).

Sizes: A=55k (>=threshold -> auto keep-alive), B=25k (fits free, auto-pinned),
C=45k (free with A+B pinned =26k < 45k -> scheduler must release SMALLEST
pinned first = B, keeping A).
Discriminator: without keep-alive, B alone (25k) already evicts A's head and
A replay -> 0. With keep-alive A replay stays ~full after B AND after C.
"""
import os
import random
import time

from openai import OpenAI
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR")
if not MODEL_DIR:
    raise SystemExit("set MODEL_PATH (or MODEL_DIR) to the model directory")
BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen38-27b")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=1800)


def make_ids(target, seed):
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
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": tok.decode(ids, skip_special_tokens=True)}],
        max_tokens=1, temperature=0,
    )
    wall = time.time() - t0
    u = resp.usage
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    print(f"[{note}] prompt={u.prompt_tokens} cached={cached} wall={wall:.1f}s", flush=True)
    return cached


A = make_ids(55000, seed=1)
send(A, "A(55k) first (expect 0)")
send(A, "A replay intact (expect ~52.5k)")
send(make_ids(25000, seed=2), "B(25k) new session")
send(A, "A replay AFTER B  <- pin keeps A (old behavior would be 0)")
send(make_ids(45000, seed=3), "C(45k) forces release smallest(=B)")
send(A, "A replay AFTER C  <- A should still be pinned & hit")
print("done")
