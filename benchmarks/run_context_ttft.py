#!/usr/bin/env python3
"""Measure real streaming first-character latency across context sizes.

If the server checks an API key, pass --api-key or export VLLM_API_KEY (falling
back to OPENAI_API_KEY); a 401/403 stops immediately with exit code 2 instead of
failing every run in the matrix.

TTFT is measured until the first non-empty reasoning_content, reasoning or
content delta. It therefore counts the first streamed thinking character.
"""
import argparse
import json
import os
import random
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


def request_once(base_url, model, prompt, max_tokens, api_key=None, timeout=900):
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
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
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
    ttft = first - started if first else None
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    decode_time = ended - first if first else None
    return {
        "status": 200,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(ended - started, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
        "decode_tok_s": round(completion_tokens / decode_time, 1)
        if completion_tokens and decode_time else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-local")
    parser.add_argument("--word-counts", type=int, nargs="+", default=[2700, 5400, 8100])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
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

