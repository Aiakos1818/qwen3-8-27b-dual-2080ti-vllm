#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tightly controlled eviction-direction probe (pool ~106288, model len 102400):
  purge(100k) wipe
  A0(40k) -> replay intact (sanity ~full hit)
  B(60k)  pops 60k of the ~66k unused front (no eviction of A yet)
  C(12k)  pops 6k unused + 6k from OLDEST cached session (A)
  replay A0 -> if cached ~=0 : A's HEAD was evicted (whole thing recomputes)
               if cached ~=34k: A's TAIL was evicted, head reused
"""
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
    print(f"[{note}] prompt={u.prompt_tokens} out={u.completion_tokens} cached={cached} wall={wall:.1f}s", flush=True)
    return u.prompt_tokens, cached


A = make_ids(40000, seed=111)
send(make_ids(100000, seed=888), "purge(100k) wipe")
send(A, "A0 40k first")
send(A, "A0 replay intact")
send(make_ids(60000, seed=222), "B 60k fill-unused(60 of 66k)")
send(make_ids(12000, seed=333), "C 12k evict 6k from A")
send(A, "DISCRIMINATOR A0 replay post-C")
send(A[:3000], "A-head probe 3000")
print("done")
