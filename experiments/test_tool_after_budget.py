#!/usr/bin/env python3
"""Does a tool call still happen once the thinking budget force-ends reasoning?"""
import json, urllib.error, urllib.request

BASE, MODEL = "http://127.0.0.1:8000", "qwen38-27b"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"],
        },
    },
}]
PROMPT = ("Think at extreme length first: enumerate the integers 1 to 400 with a "
          "commentary sentence each. Then use the write tool to save a short summary "
          "to /tmp/factor-summary.txt. Do not answer in prose instead of calling it.")


def ask(note, max_tokens, budget=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.7, "tools": TOOLS}
    if budget is not None:
        body["thinking_token_budget"] = budget
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"  [{note}] max_tokens={max_tokens} budget={budget}")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}"); return
    ch = d["choices"][0]; m = ch["message"]
    print(f"    finish_reason={ch.get('finish_reason')}  usage={d.get('usage')}")
    print(f"    message keys={sorted(m.keys())}")
    for tc in (m.get("tool_calls") or []):
        fn = tc["function"]
        print(f"    tool_call -> {fn['name']}({fn['arguments'][:180]})")
    if not m.get("tool_calls"):
        print(f"    content[:150]={ (m.get('content') or '')[:150]!r}")


print("[tool-probe] 默认 512 预算 + 3000 max_tokens —— 期望思考 511 后发出 write")
ask("default", max_tokens=3000)
print("[tool-probe] 显式 128 预算 —— 期望更早结束思考、仍发出 write")
ask("explicit", max_tokens=3000, budget=128)
print("[tool-probe] done")
