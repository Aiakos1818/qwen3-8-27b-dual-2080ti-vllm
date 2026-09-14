#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isolated eviction-direction probe: only session A is cached, then C evicts a
small slice of A. With VLLM_DEBUG_EVICT=1 the server log records:
  [DBG-FREE]  with_hash_order_ids=...   (A's blocks as returned to the pool,
                                         first-to-evict = list head)
  [DBG-ALLOC] evicted_ids=...           (block ids C's prefill actually evicted)
Correlating the two tells whether A's HEAD or TAIL pages are evicted first.
Then replaying A measures how much of its prefix survives (0 => head gone).
"""
import os
import sys
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
    import random
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
    print(f"[{note}] prompt={u.prompt_tokens} out={u.completion_tokens} cached={cached} wall={wall:.1f}s", flush=True)
    return u.prompt_tokens, cached


A = make_ids(100000, seed=11)
send(A, "A(100k) first")
send(A, "A replay intact")
send(make_ids(9000, seed=99), "C(9k) evict ~small slice of A")
res = send(A, "A replay POST-C  <- cached==0? head gone; >0? head kept")
print(f"DISCRIMINATOR cached={res[1]}", flush=True)
