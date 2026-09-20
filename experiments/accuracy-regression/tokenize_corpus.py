#!/usr/bin/env python3
"""Build the logit-eval corpus: tokenise three domains and chunk.

Run with the deployment venv (needs transformers tokenizer):
  /home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/venv/bin/python tokenize_corpus.py

Outputs (token-id JSON, consumed by run_logit_eval.py):
  corpus/dense/<domain>/seg_XXX.json   {"tokens": [...]}   8K-token segments
  corpus/probes/probe_<len>_<i>.json   {"tokens": [...]}   long-context probes
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
MODEL = "/home/aiakos/Qwen3.8-27B-Deploy/models/Qwen3.8-27B-FP8"
REPO = "/home/aiakos/Qwen3.8-27B-Deploy/zyYuc-sandbox/src/vllm-0271"

SEG = 8192            # dense segment length (tokens)
DENSE_TOKENS = 200_000  # per domain
PROBE_LENS = [32768, 131072, 262144]
PROBE_PER_LEN = 6
CODE_EXT = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
CODE_MAX_BYTES = 40_000_000


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def gather_code() -> str:
    chunks: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(REPO):
        if ".git" in dirpath or "__pycache__" in dirpath:
            continue
        for fn in sorted(filenames):
            if Path(fn).suffix not in CODE_EXT:
                continue
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > 400_000:
                    continue
                txt = read_text(p)
            except OSError:
                continue
            chunks.append(f"\n# ===== {p.relative_to(REPO)} =====\n{txt}")
            total += len(txt)
            if total > CODE_MAX_BYTES:
                return "".join(chunks)
    return "".join(chunks)


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    random.seed(1234)

    domains = {
        "en": read_text(CORPUS / "text" / "en.txt"),
        "zh": read_text(CORPUS / "text" / "zh.txt"),
        "code": gather_code(),
    }

    # ---- dense segments ----
    for name, text in domains.items():
        ids = tok.encode(text, add_special_tokens=False)
        out_dir = CORPUS / "dense" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_seg = min(len(ids) // SEG, DENSE_TOKENS // SEG)
        for i in range(n_seg):
            seg = ids[i * SEG : (i + 1) * SEG]
            (out_dir / f"seg_{i:03d}.json").write_text(json.dumps({"tokens": seg}))
        print(f"dense {name}: {len(ids)} tokens -> {n_seg} segments x {SEG}")

    # ---- long-context probes (from a domain pool, mixed) ----
    pool = tok.encode(
        domains["en"] + "\n" + domains["zh"] + "\n" + domains["code"],
        add_special_tokens=False,
    )
    probe_dir = CORPUS / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for L in PROBE_LENS:
        for i in range(PROBE_PER_LEN):
            start = random.randint(0, max(0, len(pool) - L - 1))
            seg = pool[start : start + L]
            (probe_dir / f"probe_{L}_{i:02d}.json").write_text(
                json.dumps({"tokens": seg})
            )
        print(f"probes len={L}: {PROBE_PER_LEN} contexts")

    print("done")


if __name__ == "__main__":
    main()
