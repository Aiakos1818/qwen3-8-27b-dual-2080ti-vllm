#!/usr/bin/env python
"""P1-5: concurrent spill/restore stress (max-num-seqs 4).

Runs N long sessions concurrently (cold), then resumes each concurrently with a
distinct tail. Repeats the whole workload twice and compares per-session greedy
output hashes: identical hashes across passes means every cache path (resident,
restored, recomputed) produced the same KV.

A corrupted restore (e.g. host slots clobbered by a concurrent store) shows up
as a hash mismatch and/or NaN/corrupted requests in the engine log.
"""
import concurrent.futures
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, user_msg  # noqa: E402

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=2400)

N = 4
TURNS = 3
A_TOK = 12000
UIDS = list(range(10, 10 + N))


def build(uid: int, tail: str | None = None):
    qs = [f"用户第{i}轮问题：讨论主题。" for i in range(TURNS + 1)]
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(TURNS):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, A_TOK)))
    m.append(user_msg(qs[TURNS]))
    if tail:
        m.append(user_msg(tail))
    return m


def call(msgs, max_tokens):
    r = _client.chat.completions.create(
        model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=0
    )
    u = r.usage
    text = r.choices[0].message.content or ""
    cached = (
        u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    )
    rec = {
        "prompt": u.prompt_tokens,
        "cached": cached,
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "head": text[:120],
    }
    print(f"[call] prompt={rec['prompt']} cached={cached} sha={rec['sha']}", flush=True)
    return rec


def pass_once(tag: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        cold = list(ex.map(lambda u: call(build(u), 1), UIDS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        warm = list(
            ex.map(lambda u: call(build(u, tail=f"请继续 {u} 并给出结论。"), 8), UIDS)
        )
    print(f"PASS {tag} cold_cached={[c['cached'] for c in cold]}", flush=True)
    print(f"PASS {tag} warm_cached={[w['cached'] for w in warm]}", flush=True)
    print(
        f"PASS {tag} warm_sha={[w['sha'] for w in warm]}", flush=True
    )
    return {u: w for u, w in zip(UIDS, warm)}


p1 = pass_once("1")
p2 = pass_once("2")
bad = []
for u in UIDS:
    a, b = p1[u], p2[u]
    if a["sha"] != b["sha"] or "nan" in a["head"].lower():
        bad.append((u, a["sha"], b["sha"]))
print("P15-BAD", bad, flush=True)
print("P15-DONE", flush=True)
