#!/usr/bin/env python3
"""Measure real streaming first-character latency across context sizes.

If the server checks an API key, pass --api-key or export VLLM_API_KEY (falling
back to OPENAI_API_KEY); a 401/403 stops immediately with exit code 2 instead of
failing every run in the matrix.

TTFT is measured until the first non-empty reasoning_content, reasoning or
content delta. It therefore counts the first streamed thinking character.

Decode speed comes in two flavours, because a whole-generation average is NOT the
steady-state rate with MTP: acceptance is highest right after the prompt and
decays, so a short window reads high. ``decode_tok_s`` is that whole-window
average (what this script has always reported, fine for continuity), while
``decode_steady_tok_s`` is measured over the tail of the generation (from
``--steady-from`` tokens onwards) using the engine's own
``vllm:generation_tokens_total`` counter, plus the MTP acceptance of the same
window. Keep ``--max-tokens`` well above ``--steady-from`` or the steady fields
stay null.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

WORDS = (
    "system performance optimization architecture memory bandwidth latency "
    "throughput parallel computation kernel buffer cache scheduling allocation "
    "fragmentation synchronization inference quantization compression precision "
    "stability reliability scalability bottleneck utilization"
).split()


def build_prompt(target_words, seed):
    rng = random.Random(seed)
    pieces = [rng.choice(WORDS) for _ in range(target_words)]
    return (
        "Task ID {}. Read the following technical context and provide a concise "
        "one-paragraph summary with the key conclusion.\n\n{}"
    ).format(seed, " ".join(pieces))



def api_key_from_env():
    """The server's key: our profiles export VLLM_API_KEY, OpenAI-style tools use
    OPENAI_API_KEY; accept either. An explicit --api-key wins over both."""
    return os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def scrape_counters(base_url, api_key=None):
    """Generated-token and spec-decode counters, straight from /metrics.

    ``vllm:generation_tokens_total`` is the engine's own count of generated
    tokens, so a decode rate can be measured over any window no matter how many
    tokens a stream chunk carries (MTP delivers 1-4). The counters are
    server-wide, so the steady-state fields only mean something while this
    benchmark is the only running request -- the shipped profiles set
    --max-num-seqs 1. Returns None when /metrics is unavailable.
    """
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    req = urllib.request.Request(base_url + "/metrics", headers=headers)
    out = {"gen": None, "draft": 0.0, "accepted": 0.0}
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            out["gen"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.rsplit(" ", 1)[1])
    return out if out["gen"] is not None else None


def steady_window(samples, steady_from):
    """Rate over the tail of the generation (samples past ``steady_from`` tokens).

    ``samples`` are (seconds since request start, generated tokens, drafted
    tokens, accepted tokens) tuples, so this is the engine's own accounting and
    independent of the client's chunking.
    """
    tail = [s for s in samples if s[1] >= steady_from]
    if len(tail) < 2:
        return None
    elapsed = tail[-1][0] - tail[0][0]
    generated = tail[-1][1] - tail[0][1]
    if elapsed <= 0 or generated <= 0:
        return None
    drafted = tail[-1][2] - tail[0][2]
    accepted = tail[-1][3] - tail[0][3]
    return {
        "decode_steady_tok_s": round(generated / elapsed, 1),
        "steady_from_token": steady_from,
        "steady_window_tokens": int(generated),
        "steady_mtp_acceptance_pct": round(100 * accepted / drafted, 1)
        if drafted > 0 else None,
    }


def request_once(base_url, model, prompt, max_tokens, api_key=None, timeout=900,
                 steady_from=128):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first = None
    usage = {}
    buffer = b""
    samples = []
    stop_sampling = threading.Event()
    base_counters = scrape_counters(base_url, api_key)

    def sample_counters():
        while not stop_sampling.is_set():
            counters = scrape_counters(base_url, api_key)
            if counters is not None and base_counters is not None:
                samples.append((
                    time.perf_counter() - started,
                    counters["gen"] - base_counters["gen"],
                    counters["draft"] - base_counters["draft"],
                    counters["accepted"] - base_counters["accepted"],
                ))
            stop_sampling.wait(0.2)

    sampler = None
    if base_counters is not None:
        sampler = threading.Thread(target=sample_counters, daemon=True)
        sampler.start()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        stop_sampling.set()
        if sampler is not None:
            sampler.join(timeout=2)
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
        except Exception:
            pass
        return {
            "status": exc.code,
            "error": detail,
            "hint": (
                "the server checks the API key: pass --api-key, or export "
                "VLLM_API_KEY (fallback OPENAI_API_KEY)"
                if exc.code in (401, 403)
                else ""
            ),
        }
    with response:
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                delta = (event.get("choices") or [{}])[0].get("delta", {})
                first_piece = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("content")
                )
                if first is None and first_piece:
                    first = time.perf_counter()
                if event.get("usage"):
                    usage = event["usage"]
    ended = time.perf_counter()
    stop_sampling.set()
    if sampler is not None:
        sampler.join(timeout=2)
    ttft = first - started if first else None
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    decode_time = ended - first if first else None
    result = {
        "status": 200,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(ended - started, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
        "decode_tok_s": round(completion_tokens / decode_time, 1)
        if completion_tokens and decode_time else None,
    }
    steady = steady_window(samples, steady_from)
    if steady is None:
        result["decode_steady_tok_s"] = None
        if base_counters is None:
            result["steady_note"] = "/metrics unavailable, steady window skipped"
        else:
            result["steady_note"] = (
                "no tail window: {} tokens generated, --steady-from is {}; raise "
                "--max-tokens or lower --steady-from".format(
                    completion_tokens, steady_from)
            )
    else:
        result.update(steady)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--word-counts", type=int, nargs="+", default=[2700, 5400, 8100])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="tokens to generate per run (default 512). Was 128 historically, "
        "which only measures the fastest part of the decode; keep it well above "
        "--steady-from",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token; defaults to $VLLM_API_KEY, then $OPENAI_API_KEY",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="per-request read timeout in seconds (default 900)",
    )
    parser.add_argument(
        "--steady-from",
        type=int,
        default=128,
        help="skip this many generated tokens before measuring the steady-state "
        "decode rate (default 128), since MTP acceptance is highest right after "
        "the prompt",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    api_key = args.api_key or api_key_from_env()
    if args.api_key:
        source = "--api-key"
    elif os.environ.get("VLLM_API_KEY"):
        source = "$VLLM_API_KEY"
    elif os.environ.get("OPENAI_API_KEY"):
        source = "$OPENAI_API_KEY"
    else:
        source = None
    if api_key:
        print(f"[bench] using API key from {source} ({len(api_key)} chars)", flush=True)
    else:
        print("[bench] no API key supplied (only works on a server without auth)",
              flush=True)
    all_results = []
    for word_count in args.word_counts:
        for run in range(1, args.runs + 1):
            result = request_once(
                args.base_url,
                args.model,
                build_prompt(word_count, word_count * 100 + run),
                args.max_tokens,
                api_key,
                args.timeout,
                args.steady_from,
            )
            result["target_words"] = word_count
            result["run"] = run
            all_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("status") != 200:
                # An auth problem repeats for every run, so stop instead of
                # burning the whole matrix on 401s.
                raise SystemExit(2 if result["status"] in (401, 403) else 1)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

