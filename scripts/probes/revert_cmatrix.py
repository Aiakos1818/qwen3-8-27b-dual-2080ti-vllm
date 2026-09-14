#!/usr/bin/env python
"""Cadence (VLLM_MAMBA_CKPT_TOKENS) comparison on the 100k pool.

Run ONCE per freshly-restarted engine (env VLLM_MAMBA_CKPT_TOKENS set). Builds a
~86k resident chain, then measures prefix reuse (cached_tokens) for identical
replay and for truncating reverts at junctions ~30k / ~72k. Prints TSV rows.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revert_lib import (SYSTEM, assistant_msg, filler_fast, send, user_msg)  # noqa: E402

N_TURNS = 6
A_TOK = 14000
qs = [f"用户第{i}轮问题：请继续深入讲解该主题。".replace("第0轮", "第一轮")
      for i in range(N_TURNS + 1)]
turns = [(qs[i], filler_fast(300 + i, A_TOK)) for i in range(N_TURNS)]


def full():
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(N_TURNS):
        q, a = turns[i]
        m.append(user_msg(q))
        m.append(assistant_msg(a))
    m.append(user_msg(qs[N_TURNS]))
    return m


def revert(keep, edited=True):
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(keep):
        q, a = turns[i]
        m.append(user_msg(q))
        m.append(assistant_msg(a))
    m.append(user_msg(qs[keep] + ("（改写重发）" if edited else "")))
    return m


def probe(msgs, note):
    r = send(msgs, note)
    print(f"RESULT\t{note}\tprompt={r['prompt']}\tcached={r['cached']}\t"
          f"wall={r['wall']}", flush=True)


if __name__ == "__main__":
    probe(full(), "resident")
    probe(full(), "identical_replay")
    probe(revert(5), "revert_J72k")
    probe(revert(4), "revert_J58k")
    probe(revert(2), "revert_J30k")
    probe(full(), "identical_replay_2")
    sys.exit(0)
