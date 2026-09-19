#!/usr/bin/env python3
"""Probe thinking_token_budget on the running server (stdlib only)."""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen38-27b"

PROMPT = (
    "Think at extreme length before answering. Enumerate the integers from 1 to 400 "
    "one by one, and for each give its prime factorisation plus a sentence of "
    "commentary. Only after all 400 are done, answer: what is 17*23? "
    "Do not stop thinking early."
)


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntok(text):
    if not text:
        return 0
    try:
        return len(post("/tokenize", {"model": MODEL, "prompt": text})["tokens"])
    except Exception as exc:  # noqa: BLE001
        return f"?({exc})"


def ask(note, max_tokens, budget=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if budget is not None:
        body["thinking_token_budget"] = budget
    print(f"  [{note}]  max_tokens={max_tokens}  thinking_token_budget={budget}")
    try:
        d = post("/v1/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}")
        return
    ch = d["choices"][0]
    msg = ch["message"]
    rc = msg.get("reasoning_content") or ""
    ct = msg.get("content") or ""
    tools = msg.get("tool_calls")
    print(f"    finish_reason={ch.get('finish_reason')}  usage={d.get('usage')}")
    print(
        f"    reasoning={ntok(rc)} tok（{len(rc)} 字符）  "
        f"content={ntok(ct)} tok（{len(ct)} 字符）  tool_calls={len(tools) if tools else 0}"
    )
    print(f"    content: {ct[:110]!r}")


print("[probe] A: 不带 budget 字段，靠服务端默认值（应为 512）")
ask("A default", max_tokens=2000)
print("[probe] B: 显式 thinking_token_budget=128（应覆盖默认值）")
ask("B explicit", max_tokens=2000, budget=128)
print("[probe] C: budget(4000) > max_tokens(600) —— 复现被 max_tokens 截断")
ask("C overrun", max_tokens=600, budget=4000)
print("[probe] done")
