#!/usr/bin/env python3
"""Profile the *prefill* phase of one request at a chosen prompt size.

Unlike prof_run.py (which profiles steady-state decode), this starts the torch
profiler before the request so the prefill steps land in the trace, then stops it
after a fixed window.

Usage: prof_prefill.py <prompt_words> <profile_seconds> <tag>
Requires the server to have been started with VLLM_TORCH_PROFILER_DIR=<dir>.
"""

import json
import os
import random
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "qwen38-27b"
KEY = os.environ.get("VLLM_API_KEY", "")
WORDS = (
    "system performance optimization architecture memory bandwidth latency "
    "throughput parallel computation kernel buffer cache scheduling allocation "
    "fragmentation synchronization inference quantization compression precision "
    "stability reliability scalability bottleneck utilization"
).split()

words_n, prof_secs, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
seed = 4242 + words_n


def post(path, body=None):
    req = urllib.request.Request(
        URL + path,
        data=(body or b""),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.status, r.read(120).decode("utf-8", "replace").strip()


rng = random.Random(seed)
prompt = (
    "Task ID {}. Read the following technical context and provide a detailed "
    "multi-paragraph analysis.\n\n".format(seed)
) + " ".join(rng.choice(WORDS) for _ in range(words_n))
body = json.dumps(
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
).encode()
req = urllib.request.Request(
    URL + "/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    method="POST",
)

state = {"usage": None, "done": False}


def stream():
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        ev = json.loads(line[6:])
                        if ev.get("usage"):
                            state["usage"] = ev["usage"]
    finally:
        state["done"] = True


print(f"  [{tag}] prompt_words={words_n} start_profile ...", flush=True)
print("  start_profile:", post("/start_profile"), flush=True)
t0 = time.perf_counter()
th = threading.Thread(target=stream, daemon=True)
th.start()
while time.perf_counter() - t0 < prof_secs and not state["done"]:
    time.sleep(0.5)
print(f"  [{tag}] t+{time.perf_counter() - t0:.1f}s stop_profile ...", flush=True)
print("  stop_profile:", post("/stop_profile"), flush=True)
th.join()
print(
    f"  [{tag}] prompt_tokens={(state['usage'] or {}).get('prompt_tokens')} "
    f"total={time.perf_counter() - t0:.1f}s",
    flush=True,
)
