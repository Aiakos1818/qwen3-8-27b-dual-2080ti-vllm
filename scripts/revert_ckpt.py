#!/usr/bin/env python
"""A1 checkpoint-retention validation on the 100k pool.

Chain ~90k tokens. Durable mamba snapshots at cadence C (env). Probes:
  identical full replay   -> reuse ~ everything (baseline)
  revert junction ~40k     -> expect cached ~= C (nearest anchor below 40k)
  revert junction ~79k     -> expect cached ~= 2*C (nearest anchor below 79k)
Engine must stay alive throughout.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def revert(keep, edited):
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(keep):
        q, a = turns[i]
        m.append(user_msg(q))
        m.append(assistant_msg(a))
    m.append(user_msg(qs[keep] + ("（改写重发）" if edited else "")))
    return m


send(full(), "R full resident")
send(full(), "R identical replay (baseline)")
send(revert(2, True), "R revert keep=2 junction~40k (expect cached~=C)")
send(revert(5, True), "R revert keep=5 junction~79k (expect cached~=2C)")
send(full(), "R identical replay again (engine alive check)")
