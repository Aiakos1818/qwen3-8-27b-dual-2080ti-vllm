#!/usr/bin/env python
"""opencode revert+resend prefix-cache matrix driver.

Usage: revert_matrix.py {r1_real|deep|squeeze_on|squeeze_off|leak}
Assumes the 100k-pool vLLM server (with the intended VLLM_PIN_MIN_TOKENS) is up.
Appends per-request results (ndjson) to revert_results.ndjson.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revert_lib import (SYSTEM, assistant_msg, chrono_requests, filler_fast,
                        ntok, prefix_tokens_est, revert_request, send,
                        user_msg)  # noqa: E402

RESULTS = os.environ.get("REVERT_RESULTS", "/tmp/revert_results.ndjson")
CAPTURE_LOG = os.environ.get("CAPTURE_LOG", "/tmp/capture_ndjson.log")


def rec(res):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(res) + "\n")


def load_capture():
    return [json.loads(l) for l in open(CAPTURE_LOG)]


def mode_r1_real():
    rows = load_capture()
    # row6 = pre-revert resident chain (ends at tool result); row7 = revert probe.
    pre = json.loads(rows[6]["body"])
    post = json.loads(rows[7]["body"])
    rec(send(pre["messages"], "R1 pre-revert chain resident (row6)"))
    rec(send(post["messages"], "R1 revert probe (row7): expect cached>0 near full"))
    n_pre = ntok(json.dumps(pre["messages"], ensure_ascii=False))
    n_post = ntok(json.dumps(post["messages"], ensure_ascii=False))
    print(f"[info] pre msgs bytes~tokens={n_pre} post={n_post}", flush=True)


def mode_deep():
    # 3 assistant-bearing turns sized to push a near-full 100k pool, then
    # revert probes at increasing depth. Keep-alive ON (default) so the built
    # chain is pinned; resident through assistant a2.
    qs = ["第一轮问题：请描述前缀缓存。",
          "第二轮问题：继续之前的主题。",
          "第三轮问题：再继续延伸讨论。",
          "最后一轮问题：总结。"]
    asz = [20000, 30000, 38000]
    turns = list(zip(qs[:3], [filler_fast(i, asz[i]) for i in range(3)]))
    # build chronological (resident through a2, ending with q3)
    for u in range(4):
        msgs = chrono_requests(SYSTEM, turns, min(u, 3))
        r = send(msgs, f"DEEP build req upto={u}")
        rec(r)
    # revert probes
    for j, est in ((0, 0), (1, 1), (3, 3)):
        msgs = revert_request(SYSTEM, turns, j, qs[j] + "（已改写重发）")
        r = send(msgs, f"DEEP revert j={j} (keep {prefix_tokens_est(SYSTEM, turns, j)} est)")
        rec(r)


def session_build(label, turns, last_q):
    """Build a K-assistant-turn conversation resident (ends at last_q)."""
    for u in range(len(turns) + 1):
        msgs = chrono_requests(SYSTEM, turns, min(u, len(turns)))
        r = send(msgs, f"{label} build upto={u}")
        rec(r)
    # last_q is already the tail of the final chrono request when u==len(turns)
    return msgs


def mode_squeeze(label):
    # A: ~45k conversation (auto-pinned if ON since >16k). B: ~21k other session.
    # C: ~48k arrives -> if pinned, release-smallest frees B and A survives;
    #    if unpinned (OFF), C evicts a few head pages of A -> whole chain dies.
    A_qs = ["A1：介绍主题甲。", "A2：扩展主题甲。", "A3：最后总结甲。"]
    A_a = [20000, 24000]
    A = list(zip(A_qs[:2], [filler_fast(100 + i, A_a[i]) for i in range(2)]))
    session_build("A", A, A_qs[2])

    B = user_msg(filler_fast(50, 18000))  # ~18k user + system ~3.4k
    rec(send([{"role": "system", "content": SYSTEM}, B], "B session (~21k)"))

    C = user_msg(filler_fast(60, 45000))  # prompt ~48.5k
    rec(send([{"role": "system", "content": SYSTEM}, C], "C squeeze (~48.5k)"))

    # revert probe of A at j=2 (keep through a1, ~ all of A's history)
    msgs = revert_request(SYSTEM, A, 2, A_qs[2] + "（重发改写）")
    exp = prefix_tokens_est(SYSTEM, A, 2)
    r = send(msgs, f"squeeze_{label} probe A revert j=2 (exp~{exp})")
    rec(r)


def mode_leak():
    # Emulate an opencode session that keeps getting reverted + resent: every
    # finished chain shares one long prefix (the unchanged history) and diverges
    # only in a small tail. Keep-alive subsumption keyed on the TAIL hash can NOT
    # merge these (the pre-revert tail never reappears), so each revert finishes
    # a request whose chain gets pinned as a NEW entry -> entry-count leak while
    # physical cost stays tiny (tails share the long prefix).
    shared = filler_fast(700, 20000)  # ~20k-token unchanged history prefix
    msgs = [{"role": "system", "content": SYSTEM}, user_msg(shared + "（首版）")]
    rec(send(msgs, "LEAK X0 initial"))
    for k in range(1, 6):
        body = shared + f"【第{k}次 revert 后的新尾部】" + filler_fast(900 + k, 2500)
        msgs = [{"role": "system", "content": SYSTEM}, user_msg(body)]
        rec(send(msgs, f"LEAK divergent chain X{k}"))
    # newest chain probe: should still prefix-hit regardless of entry count.
    body = shared + "【最终仍要继续的内容】"
    msgs = [{"role": "system", "content": SYSTEM}, user_msg(body)]
    rec(send(msgs, "LEAK newest-chain probe"))


if __name__ == "__main__":
    mode = sys.argv[1]
    {"r1_real": mode_r1_real,
     "deep": mode_deep,
     "squeeze_on": lambda: mode_squeeze("on"),
     "squeeze_off": lambda: mode_squeeze("off"),
     "leak": mode_leak}[mode]()
