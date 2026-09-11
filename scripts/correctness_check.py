#!/usr/bin/env python
"""Correctness check: greedy generation for a resumed session.

Modes:
  baseline : run S, then resume R=S+tail with GPU-resident prefix.
  offload  : run S, run T (forces S spill to RAM), then resume R.
Greedy decoding (temperature=0) must produce identical text in both modes if
the restored KV is byte-correct.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI  # noqa: E402
from revert_lib import (  # noqa: E402
    BASE,
    MODEL,
    SYSTEM,
    TOK,
    assistant_msg,
    filler_fast,
    user_msg,
)

client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=2400)


def build(uid, n_turns, a_tok, tail=None):
    qs = [f"用户第{i}轮问题：讨论主题。".replace("第0轮", "第一轮")
          for i in range(n_turns + 1)]
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n_turns):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, a_tok)))
    m.append(user_msg(qs[n_turns]))
    if tail:
        m.append(user_msg(tail))
    return m


def run(name, msgs, max_tokens):
    r = client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=0
    )
    u = r.usage
    text = r.choices[0].message.content or ""
    rec = {
        "name": name,
        "prompt": u.prompt_tokens,
        "cached": u.prompt_tokens_details.cached_tokens
        if u.prompt_tokens_details
        else None,
        "out_tokens": u.completion_tokens,
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "head": text[:160],
    }
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


TAIL = "请只回答：OK"
# Some checkpoints spend the first tokens in the thinking channel, so a small cap
# can leave `content` empty (and the sha meaningless). Override when needed.
MAX_OUT = int(os.environ.get("KV_CHECK_MAX_TOKENS", "32"))


def main():
    mode = sys.argv[1]
    S = build(7, 3, 16000)          # ~50k
    R = build(7, 3, 16000, tail=TAIL)
    if mode == "baseline":
        run("S", S, 1)
        out = run("R-resident", R, MAX_OUT)
    elif mode == "offload":
        T = build(8, 4, 15000)      # ~62k, bigger than free -> spills S
        run("S", S, 1)
        run("T", T, 1)
        out = run("R-restored", R, MAX_OUT)
    else:
        raise SystemExit("mode: baseline|offload")
    print("RESULT", json.dumps({"mode": mode, "sha": out["sha"]}))


if __name__ == "__main__":
    main()
