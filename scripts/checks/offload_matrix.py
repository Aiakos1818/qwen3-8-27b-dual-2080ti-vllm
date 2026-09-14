#!/usr/bin/env python
"""Stress matrix for host-tier spill/restore (100k pool, ~50k CPU tier, MTP on).

Each "session" is a long prefill-only conversation (max_tokens small). Resumes
reuse the same history with an appended divergent tail so the shared prefix is
the whole prior conversation.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revert_lib import (SYSTEM, assistant_msg, filler_fast, send, user_msg)  # noqa: E402


def build(uid: int, n_turns: int = 3, a_tok: int = 16000, tail: str | None = None):
    qs = [
        f"用户第{i}轮问题：请继续深入讲解该主题。".replace("第0轮", "第一轮")
        for i in range(n_turns + 1)
    ]
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n_turns):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, a_tok)))
    m.append(user_msg(qs[n_turns]))
    if tail:
        m.append(user_msg(tail))
    return m


def run(name: str, msgs, max_tokens: int = 1):
    t0 = time.time()
    out = send(msgs, name, max_tokens=max_tokens)
    return out


if __name__ == "__main__":
    S1 = build(1)
    S2 = build(2)
    S3 = build(3)
    R1 = build(1, tail="继续刚才的讨论，追加问题：请总结要点。")
    R2 = build(2, tail="继续刚才的讨论，追加问题：请列出清单。")
    R3 = build(3, tail="继续刚才的讨论，追加问题：请给出结论。")
    SMALL = [{"role": "system", "content": SYSTEM},
             user_msg("简短问题：你好。")]

    plan = [
        ("s1", S1, 1),
        ("s2", S2, 1),
        ("resume-s1", R1, 8),
        ("s3", S3, 1),
        ("resume-s2", R2, 1),
        ("resume-s3", R3, 1),
        ("resume-s1b", R1, 1),
        ("small", SMALL, 1),
        ("resume-s2b", R2, 1),
    ]
    for name, msgs, mt in plan:
        run(name, msgs, mt)
    print("MATRIX-DONE", flush=True)
