#!/usr/bin/env python
"""Probe: does a truncating revert on a LARGE opencode-style history reuse its
pinned/prefix chain without triggering the allocator double-free crash?

H = one big request whose messages are the full history (finished -> auto-pinned
when keep-alive is on).  P = revert probe: keep history through assistant of
turn k-1, replace the following user message (edited), drop the rest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revert_lib import (SYSTEM, assistant_msg, filler_fast, revert_request,
                        send, user_msg)  # noqa: E402

RESULTS = os.environ.get("REVERT_RESULTS", "/tmp/revert_results.ndjson")

qs = [f"用户第{i}轮问题：请继续深入讲解该主题。".replace("第0轮", "第一轮")
      for i in range(9)]
a_sizes = [7000] * 8
turns = list(zip(qs[:8], [filler_fast(200 + i, a_sizes[i]) for i in range(8)]))

msgs = [{"role": "system", "content": SYSTEM}]
for i in range(8):
    q, a = turns[i]
    msgs.append(user_msg(q))
    msgs.append(assistant_msg(a))
msgs.append(user_msg(qs[8]))
res = send(msgs, "H full-history resident (finish->pin)")
with open(RESULTS, "a") as f:
    f.write(__import__("json").dumps(res) + "\n")

keep = int(sys.argv[1])  # keep history through assistant of turn keep-1
msgs = revert_request(SYSTEM, turns, keep, qs[keep] + "（改写后重发）")
res = send(msgs, f"P revert probe keep={keep} (truncating reuse)")
with open(RESULTS, "a") as f:
    f.write(__import__("json").dumps(res) + "\n")
