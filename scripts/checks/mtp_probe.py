#!/usr/bin/env python
"""Offload resume with logprobs to detect NaN (MTP debug)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI  # noqa: E402
from correctness_check import TAIL, build  # noqa: E402

client = OpenAI(
    base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="EMPTY", timeout=2400)
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen38-27b")


def show(name, msgs, max_tokens, lp=False):
    kw = {}
    if lp:
        kw = {"extra_body": {"logprobs": True, "top_logprobs": 0}}
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=max_tokens,
            temperature=0, **kw,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        return
    m = r.choices[0].message
    toks = []
    if lp and r.choices[0].logprobs and r.choices[0].logprobs.content:
        toks = [c.token for c in r.choices[0].logprobs.content]
    print(f"[{name}] cached={r.usage.prompt_tokens_details.cached_tokens} "
          f"out={r.usage.completion_tokens} "
          f"content={repr((m.content or '')[:60])} toks={toks[:8]}", flush=True)


S = build(7, 3, 16000)
R = build(7, 3, 16000, tail=TAIL)
T = build(8, 4, 15000)

show("S", S, 1)
show("R-resident", R, 8, lp=False)
show("S2", S, 1)
show("T", T, 1)
show("R-restored", R, 8, lp=False)
show("R2", R, 8, lp=False)
