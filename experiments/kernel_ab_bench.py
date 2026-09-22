#!/usr/bin/env python3
"""One long generation, reported as ms per verify step (the MTP-normalised
metric from docs/upstream-branch.md section 6), plus acceptance and tok/s.

ms/step is derived from the authoritative counters, not from chunk timing:
    steps = spec_decode_num_draft_tokens_total / num_spec_tokens
    ms/step = wall_time / steps      tokens/step = 1 + n * acceptance
so the two A/B legs are comparable even when acceptance drifts between them.
"""
import argparse
import json
import random
import threading
import time
import urllib.request

WORDS = (
    "system performance optimization architecture memory bandwidth latency "
    "throughput parallel computation kernel buffer cache scheduling allocation "
    "fragmentation synchronization inference quantization compression precision "
    "stability reliability scalability bottleneck utilization"
).split()


def scrape(url):
    txt = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    out = {"gen": 0.0, "draft": 0.0, "acc": 0.0, "running": 0.0}
    for line in txt.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            out["gen"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["acc"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.rsplit(" ", 1)[1])
    return out


def build_prompt(task, rng, words, seed):
    if task == "open":
        text = " ".join(rng.choice(WORDS) for _ in range(words))
        return (
            "Task ID {}. Read the following technical context and provide a "
            "detailed multi-paragraph analysis.\n\n".format(seed)
        ) + text
    toks = [rng.choice(WORDS) for _ in range(words)]
    if task == "extract":
        for i in range(50, len(toks), 50):
            toks[i] = "ZQX42"
    doc = " ".join(toks)
    if task == "repeat":
        return (
            "Repeat the following document verbatim, word for word, with no "
            "changes and no commentary.\n\n" + doc
        )
    return (
        "From the following document, quote verbatim every sentence that "
        "contains the token ZQX42. Output only those sentences.\n\n" + doc
    )


def run(
    url, model, key, words, max_tokens, nspec, seed, task="open", messages=None, temp=0.6
):
    if messages is None:
        rng = random.Random(seed)
        messages = [{"role": "user", "content": build_prompt(task, rng, words, seed)}]
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.95,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    result = {}
    base = scrape(url)

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
                                result["usage"] = ev["usage"]
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
            body = getattr(exc, "read", lambda: b"")()
            print("  [stream error]", repr(exc), body[:300].decode("utf-8", "replace"))

    th = threading.Thread(target=stream, daemon=True)
    t0 = time.perf_counter()
    th.start()
    samples = []
    while th.is_alive():
        time.sleep(0.2)
        s = scrape(url)
        samples.append(
            (
                time.perf_counter() - t0,
                s["gen"] - base["gen"],
                s["draft"] - base["draft"],
                s["acc"] - base["acc"],
                s["running"],
            )
        )
    th.join()
    while samples and samples[0][1] == 0:  # drop prefill samples
        samples.pop(0)
    elapsed = time.perf_counter() - t0
    prompt_tokens = (result.get("usage") or {}).get("prompt_tokens")
    if not samples:
        return prompt_tokens, 0, None
    t_first = samples[0][0]
    samples = [(t - t_first, g, d, a, r) for (t, g, d, a, r) in samples]
    dt = samples[-1][0] - samples[0][0]
    dgen = samples[-1][1] - samples[0][1]
    ddraft = samples[-1][2] - samples[0][2]
    dacc = samples[-1][3] - samples[0][3]
    # Each spec-decode step emits (1 + accepted) tokens, so steps = gen - accepted.
    # This holds for MTP and for ngram/ngram_gpu (where per-step draft counts vary).
    steps = max(dgen - dacc, 0.0)
    acc = dacc / ddraft if ddraft else 0.0
    return prompt_tokens, dgen, {
        "elapsed_s": elapsed,
        "wall_s": dt,
        "tokens": dgen,
        "draft": ddraft,
        "accepted": dacc,
        "steps": steps,
        "draft_per_step": ddraft / steps if steps else None,
        "tokens_per_step": dgen / steps if steps else None,
        "acceptance": acc,
        "ms_per_step": 1000.0 * dt / steps if steps else None,
        "ms_per_token": 1000.0 * dt / dgen if dgen else None,
        "tok_s": dgen / dt if dt else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--api-key", default="x")
    ap.add_argument("--words", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--nspec", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="0 = greedy. Use greedy for head/kernel A/B: the target token stream "
        "is then identical across legs, so acceptance is comparable.",
    )
    ap.add_argument("--task", default="open", choices=["open", "repeat", "extract"])
    ap.add_argument(
        "--messages-json",
        default="",
        help="JSON file with a real chat request: {\"messages\":[{role,content}...]}",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    messages = None
    task = a.task
    if a.messages_json:
        with open(a.messages_json) as f:
            messages = json.load(f)["messages"]
        task = "session"
    if messages is None and not a.words:
        ap.error("--words is required unless --messages-json is given")
    p, gen, m = run(
        a.url,
        a.model,
        a.api_key,
        a.words,
        a.max_tokens,
        a.nspec,
        a.seed,
        task,
        messages,
        a.temp,
    )
    if m is None:
        print(f"prompt_tokens={p}  generated={gen}  (no decode samples)")
        return
    print(
        f"  [{task}] prompt_tokens={p}  generated={gen}  steps={m['steps']:.0f}  "
        f"tokens/step={m['tokens_per_step']:.2f}  draft/step={m['draft_per_step']:.2f}  "
        f"acceptance={100*m['acceptance']:.1f}%  "
        f"ms/step={m['ms_per_step']:.2f}  ms/token={m['ms_per_token']:.2f}  "
        f"tok/s={m['tok_s']:.1f}"
    )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {"task": task, "prompt_tokens": p, "generated": gen, **m},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
