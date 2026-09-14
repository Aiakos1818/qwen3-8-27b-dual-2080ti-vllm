#!/usr/bin/env python
"""Near-full (~97k) single resident request builder (no import side effects)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revert_lib import (SYSTEM, assistant_msg, filler_fast, send, user_msg)  # noqa: E402

N_TURNS = 7
A_TOK = 13500
qs = [f"用户第{i}轮问题：请继续深入讲解该主题。".replace("第0轮", "第一轮")
      for i in range(N_TURNS + 1)]
turns = [(qs[i], filler_fast(400 + i, A_TOK)) for i in range(N_TURNS)]

m = [{"role": "system", "content": SYSTEM}]
for i in range(N_TURNS):
    q, a = turns[i]
    m.append(user_msg(q))
    m.append(assistant_msg(a))
m.append(user_msg(qs[N_TURNS]))
send(m, "resident-big")
