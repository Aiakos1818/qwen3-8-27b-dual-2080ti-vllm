#!/usr/bin/env python3
"""Decode rate vs position within one long generation (context grows as it runs).

Rate comes from the authoritative counter vllm:generation_tokens_total, sampled
every 0.5s while the request streams, so it works regardless of how many tokens a
stream chunk carries (MTP delivers 1-4).
"""
import argparse, json, random, threading, time, urllib.request

WORDS = ("system performance optimization architecture memory bandwidth latency "
         "throughput parallel computation kernel buffer cache scheduling allocation "
         "fragmentation synchronization inference quantization compression precision "
         "stability reliability scalability bottleneck utilization").split()


def scrape(url):
    txt = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            out["gen"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["acc"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.rsplit(" ", 1)[1])
    out.setdefault("draft", 0.0); out.setdefault("acc", 0.0); out.setdefault("gen", 0.0); out.setdefault("running", 0.0)
    return out


def one_run(url, model, key, words, max_tokens, seed):
    rng = random.Random(seed)
    prompt = ("Task ID {}. Read the following technical context and provide a "
              "detailed multi-paragraph analysis.\n\n").format(seed) + \
             " ".join(rng.choice(WORDS) for _ in range(words))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key}, method="POST")
    result = {}
    base = scrape(url)

    def stream():
        with urllib.request.urlopen(req, timeout=1800) as resp:
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

    th = threading.Thread(target=stream, daemon=True)
    t0 = time.perf_counter()
    th.start()
    samples = []
    while th.is_alive():
        time.sleep(0.2)
        s = scrape(url)
        samples.append((time.perf_counter() - t0, s["gen"] - base["gen"],
                        s["draft"] - base["draft"], s["acc"] - base["acc"], s["running"]))
    th.join()
    total = int(samples[-1][1]) if samples else 0
    while samples and samples[0][1] == 0:   # 丢掉 prefill 阶段的样本
        samples.pop(0)
    if samples:
        t_first = samples[0][0]
        samples = [(t - t_first, g, d, a, r) for (t, g, d, a, r) in samples]
    prompt_tokens = (result.get("usage") or {}).get("prompt_tokens")
    elapsed = time.perf_counter() - t0
    return prompt_tokens, total, samples, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--words", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    p, total, samples, elapsed = one_run(a.url, a.model, a.api_key, a.words,
                                         a.max_tokens, a.seed)
    print(f"  prompt_tokens={p}  generated={total} in {elapsed:.1f}s  "
          f"整体 {total/elapsed:.1f} tok/s")
    print("  位置(生成 token 区间)     上下文≈      decode      MTP接受率  并发")
    marks = [0, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, total]
    marks = sorted({m for m in marks if m <= total})
    for a_i, b_i in zip(marks, marks[1:]):
        seg = [s for s in samples if a_i <= s[1] < b_i]
        if len(seg) < 2:
            continue
        dt = seg[-1][0] - seg[0][0]
        dgen = seg[-1][1] - seg[0][1]
        ddr = seg[-1][2] - seg[0][2]
        dac = seg[-1][3] - seg[0][3]
        acc = f"{100*dac/ddr:.1f}%" if ddr > 0 else "—"
        run = max((x[4] for x in seg), default=0)
        ctx = f"{p + b_i:,}" if p else "?"
        print(f"  {a_i:>5}–{b_i:<5}                {ctx:>9}   "
              f"{dgen/dt if dt else 0:>6.1f} tok/s   {acc:>6}   running={run:.0f}")


if __name__ == "__main__":
    main()
