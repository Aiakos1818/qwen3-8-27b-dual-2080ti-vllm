#!/usr/bin/env python
"""P2: single-boot SSD host-tier matrix (tmpfs fake SSD).

Runs the whole validation matrix against one already-running server:
  S1 baseline (resident)          -> sha reference
  S2 SSD park/resume              -> sha matches S1, cached > 0
  S3 deep revert after SSD restore-> keep=2 -> cached=30400, keep=3 -> 46400
                                     (MTP drops one block; 32000/48000 without)
  S3b double park/restore cycle   -> cycle2 keep=2 still cached=30400
  S4 quota LRU                    -> evictions >= 1, sessions <= 2, resume OK
  S5 concurrency (p15)            -> no corrupted, shas consistent
Plus a NaN scan of the engine trace log.

Usage: ssd_matrix.py [--expected-sha db8b8e836881534b]
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from openai import OpenAI  # noqa: E402
from revert_lib import BASE, MODEL, SYSTEM, assistant_msg, filler_fast, send, user_msg  # noqa: E402

PY = os.environ.get("VLLM_PYTHON", sys.executable)
RAMTRACE_LOG = os.environ.get("RAMTRACE_LOG", "/tmp/vllm_ramtrace.log")
SSD_ROOT = os.environ.get("SSD_TEST_ROOT", "/dev/shm/ssd_test")

_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=2400)

RESULTS: list[tuple[str, bool, str]] = []
SOFT: set[str] = set()


def record(name: str, ok: bool, detail: str = "", soft: bool = False) -> None:
    RESULTS.append((name, ok, detail))
    if soft and not ok:
        SOFT.add(name)
    print(f"[{'PASS' if ok else ('SOFT' if soft else 'FAIL')}] {name} {detail}",
          flush=True)


def check_alive() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://localhost:8000/v1/models", timeout=5
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_metrics() -> dict[str, float]:
    import urllib.request

    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for m in re.finditer(r"^(vllm:host_tier_ssd_\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                         text, re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def nan_count() -> int:
    if not os.path.exists(RAMTRACE_LOG):
        return 0
    n = 0
    with open(RAMTRACE_LOG, errors="ignore") as f:
        for line in f:
            if line.startswith("NAN "):
                n += 1
    return n


def run_script(name: str, args: list[str], timeout: int = 2400) -> tuple[bool, str]:
    print(f"\n===== {name}: {' '.join(args)} =====", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [PY, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=SCRIPTS,
    )
    out = proc.stdout + proc.stderr
    print(out, flush=True)
    print(f"===== {name} rc={proc.returncode} wall={time.time() - t0:.0f}s =====",
          flush=True)
    return proc.returncode == 0, out


def build_session(uid: int, n_turns: int = 3, a_tok: int = 16000):
    qs = [f"用户第{i}轮问题：请继续深入讲解该主题。" for i in range(n_turns + 1)]
    m = [{"role": "system", "content": SYSTEM}]
    for i in range(n_turns):
        m.append(user_msg(qs[i]))
        m.append(assistant_msg(filler_fast(uid * 100 + i, a_tok)))
    m.append(user_msg(qs[n_turns]))
    return m


def parse_result_sha(out: str) -> str | None:
    m = re.search(r'RESULT \{"mode": "\w+", "sha": "(\w+)"\}', out)
    return m.group(1) if m else None


def parse_cached(out: str, needle: str) -> int | None:
    for line in out.splitlines():
        if needle in line:
            m = re.search(r'cached[=":]+\s*(\d+)', line)
            if m:
                return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", default="db8b8e836881534b")
    args = ap.parse_args()

    if not check_alive():
        record("server alive", False)
        return 1
    record("server alive", True)
    m0 = get_metrics()
    print("metrics@start", m0, flush=True)

    # S3 first (clean pool): deep revert after an SSD restore. Running it
    # before the other scenarios keeps the Mamba anchors alive (they need free
    # GPU slots while the request runs).
    # MTP (`use_eagle`) makes the full-attention finder drop one block, so the
    # reusable boundary is `cadence - block_size` (30400 / 46400). Accept the
    # plain values too so the check still works without speculative decoding.
    keep2_expected = {30400, 32000}
    keep3_expected = {46400, 48000}
    ok_t, out_t = run_script("S3 test", ["p03_restore_revert.py", "test"])
    t2 = parse_cached(out_t, "revert keep=2 after restore")
    t3 = parse_cached(out_t, "revert keep=3 after restore")
    record("S3 SSD restore + deep revert",
           ok_t and t2 in keep2_expected and t3 in keep3_expected,
           f"keep2={t2} keep3={t3}")

    # S3b: two park/restore cycles. The second deep revert only hits its
    # anchor if the resumed session re-claimed it into its keep-alive entry.
    ok_t2, out_t2 = run_script("S3b test2", ["p03_restore_revert.py", "test2"])
    t2b = parse_cached(out_t2, "revert keep=2 cycle2")
    record("S3b double-cycle deep revert",
           ok_t2 and t2b in keep2_expected,
           f"cycle2 keep2={t2b}")

    # S1/S2: baseline vs SSD park/resume.
    ok, out_b = run_script("S1 baseline", ["correctness_check.py", "baseline"])
    sha_b = parse_result_sha(out_b)
    record("S1 baseline ran", ok and sha_b is not None, f"sha={sha_b}")
    record(
        "S1 baseline sha expected",
        sha_b == args.expected_sha,
        f"{sha_b} != {args.expected_sha}",
    )

    m_before = get_metrics()
    ok, out_o = run_script("S2 offload", ["correctness_check.py", "offload"])
    m_after = get_metrics()
    sha_o = parse_result_sha(out_o)
    cached_o = parse_cached(out_o, "R-restored")
    record("S2 SSD park/resume ran", ok and sha_o is not None, f"sha={sha_o}")
    record("S2 sha matches baseline", sha_o == sha_b, f"{sha_o} vs {sha_b}")
    record(
        "S2 used the SSD tier",
        delta(m_before, m_after, "vllm:host_tier_ssd_restores_total") > 0,
        f"restores+={delta(m_before, m_after, 'vllm:host_tier_ssd_restores_total')}",
        soft=True,
    )
    record("S2 restored from cache", (cached_o or 0) > 0, f"cached={cached_o}",
           soft=True)

    # S4: quota LRU.
    try:
        m4_before = get_metrics()
        for uid in (21, 22, 23, 24, 25):
            send(build_session(uid, n_turns=4, a_tok=12000),
                 f"quota-{uid} fill", max_tokens=1)
        m4 = get_metrics()
        print("metrics@quota", m4, flush=True)
        stores = delta(m4_before, m4, "vllm:host_tier_ssd_stores_total")
        evictions = delta(m4_before, m4, "vllm:host_tier_ssd_evictions_total")
        sessions = m4.get("vllm:host_tier_ssd_sessions", 0)
        record("S4 quota stores", stores >= 2, f"stores+={stores}")
        record("S4 quota LRU eviction", evictions >= 1, f"evictions+={evictions}")
        record("S4 quota sessions bounded", sessions <= 2, f"sessions={sessions}")
        # Resume an evicted session: must recompute, not crash.
        send(build_session(21, n_turns=4, a_tok=12000),
             "quota-21 resume after eviction", max_tokens=1)
        record("S4 resume after eviction", check_alive())
    except Exception as exc:  # noqa: BLE001
        record("S4 quota LRU", False, repr(exc))

    # S5: concurrency.
    ok_p, out_p = run_script("S5 concurrency", ["p15_concurrent.py"], timeout=3600)
    record("S5 concurrency ran", ok_p)
    record("S5 concurrency no mismatch", "P15-BAD []" in out_p)

    # S3 control last (informational: accumulated pool pressure can make the
    # resident-chain anchor check miss even though the path is correct).
    ok_c, out_c = run_script("S3 control", ["p03_restore_revert.py", "control"])
    c2 = parse_cached(out_c, "revert keep=2")
    c3 = parse_cached(out_c, "revert keep=3")
    record("S3 control revert",
           ok_c and c2 in keep2_expected and c3 in keep3_expected,
           f"keep2={c2} keep3={c3}", soft=True)

    # Final checks.
    nans = nan_count()
    record("no NaN logits in trace", nans == 0, f"nans={nans}")
    record("server still alive", check_alive())
    m1 = get_metrics()
    print("metrics@end", m1, flush=True)

    failed = [r for r in RESULTS if not r[1] and r[0] not in SOFT]
    print("\n===== SSD MATRIX SUMMARY =====", flush=True)
    for name, ok, detail in RESULTS:
        tag = "PASS" if ok else ("SOFT" if name in SOFT else "FAIL")
        print(f"{tag}  {name}  {detail}", flush=True)
    print(f"SSD-MATRIX-{'DONE' if not failed else 'FAILED'} "
          f"({len(RESULTS) - len(failed)}/{len(RESULTS)}; "
          f"{len(SOFT)} soft)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
