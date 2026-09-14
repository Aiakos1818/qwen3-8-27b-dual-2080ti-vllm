#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evict only ~1 page of A (smallest possible) and check whether reuse survives."""
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


A = make_ids(100000, seed=11)
send(A, "A(100k) first")
send(A, "A replay intact (expect ~97600)")
for csize, tag in [(2500, "C~2k (evict ~1 page?)"), (4500, "C~4.5k (evict ~2-3 pages?)")]:
    send(make_ids(csize, seed=csize), tag)
    c = send(A, f"A replay after {tag}")
    print(f"    -> cached={c}  (>0 means head survived; ==0 means head gone)", flush=True)
