#!/usr/bin/env python
"""P0-3: restore x deep-revert interaction.

control : A resident, truncated revert (anchor reuse expected ~32k)
test    : A -> B (forces A spill) -> A resume (restore) -> truncated revert

A is sized to fit the CPU tier (~48k -> 33 slots of 34); B is bigger so it
forces A out of GPU into RAM.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revert_lib import SYSTEM, assistant_msg, filler_fast, send, user_msg  # noqa: E402


def build(uid: int, n_turns: int, a_tok: int, keep: int | None = None, edited: bool = False):
    qs = [f"用户第{i}轮问题：请继续深入讲解该主题。" for i in range(n_turns + 1)]
    n = n_turns if keep is None else keep
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, a_tok)))
    m.append(user_msg(qs[n] + ("（改写重发）" if edited else "")))
    return m


A = lambda **kw: build(1, 3, 16000, **kw)  # ~48k -> ~33 slots
B = lambda **kw: build(2, 4, 16000, **kw)  # ~64k -> too big to park

mode = sys.argv[1]
if mode == "control":
    send(A(), "A full resident")
    send(A(keep=2, edited=True), "revert keep=2 (expect anchor ~32k)")
    send(A(keep=3, edited=True), "revert keep=3 (expect ~48k)")
elif mode == "test":
    send(A(), "A full resident")
    send(B(), "B full (forces A spill)")
    send(A(), "A resume (restore)")
    send(A(keep=2, edited=True), "revert keep=2 after restore")
    send(A(keep=3, edited=True), "revert keep=3 after restore")
else:
    raise SystemExit("mode: control|test")
print("P03-DONE", flush=True)
