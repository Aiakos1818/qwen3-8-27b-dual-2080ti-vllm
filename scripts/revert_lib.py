"""Shared helpers for opencode revert-resend prefix-cache matrix tests.

Talks DIRECTLY to the vLLM server on :8000 (bypasses the :8001 capture proxy so
the proxy log stays a pure record of the real opencode session).
"""
import os
import time

from openai import OpenAI
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR")
if not MODEL_DIR:
    raise SystemExit("set MODEL_PATH (or MODEL_DIR) to the AWQ-INT4 model directory")
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen38-27b")
BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
_client = OpenAI(base_url=BASE, api_key="EMPTY", timeout=2400)

# Generic system prompt for the long-session harness (no external fixture).
SYSTEM = os.environ.get(
    "KV_TEST_SYSTEM",
    "You are a helpful assistant. Answer precisely and keep replies short.",
)


def ntok(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False))


def user_msg(q: str) -> dict:
    # opencode sends user content as a part-array of text parts.
    return {"role": "user", "content": [{"type": "text", "text": q}]}


def assistant_msg(txt: str) -> dict:
    # captured assistant messages use plain-string content.
    return {"role": "assistant", "content": txt}


def send(messages, note, max_tokens=1):
    t0 = time.time()
    r = _client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0
    )
    u = r.usage
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    out = {"note": note, "prompt": u.prompt_tokens, "cached": cached,
           "wall": round(time.time() - t0, 1)}
    print(f"[{note}] prompt={u.prompt_tokens} cached={cached} wall={out['wall']}s",
          flush=True)
    return out


def filler(uid: int, ntok_target: int, step: int = 200) -> str:
    """Deterministic, unique-per-turn long text of roughly ntok_target tokens."""
    out = []
    acc = 0
    i = 0
    while acc < ntok_target:
        chunk = f"[turn{uid}-chunk{i}] 这是用于构造长会话上下文的确定性历史内容段落，"
        chunk += f"序号 {uid}-{i}，重复以保证字节一致与长度可控。\n"
        out.append(chunk)
        acc += len(TOK.encode("".join(out[-3:]), add_special_tokens=False)) - (
            len(TOK.encode("".join(out[:-3]), add_special_tokens=False)) if len(out) > 3 else 0
        )
        i += 1
        if i > 600:
            break
    # fallback exact-ish correction loop
    txt = "".join(out)
    while ntok(txt) < ntok_target and i < 4000:
        txt += f"[turn{uid}-pad{i}] 填充。"
        i += 1
    return txt


def filler_fast(uid: int, ntok_target: int) -> str:
    unit = f"[{uid}]长会话上下文填充内容用于前缀缓存实验；"
    u = ntok(unit)
    reps = max(1, ntok_target // max(1, u))
    txt = unit * reps
    return txt


def session_turns(user_qs, assistant_targets):
    """Build [(user_q, assistant_text)...]; assistant sizes in tokens."""
    turns = []
    for i, (q, at) in enumerate(zip(user_qs, assistant_targets)):
        turns.append((q, filler_fast(i, at)))
    return turns


def chrono_requests(system, turns, upto):
    """Simulate opencode resending the whole transcript each turn:
    for turn t<=upto: messages = system + turns[0..t-1] fully + current user_q_t."""
    msgs = [{"role": "system", "content": system}]
    for t in range(upto):
        q, a = turns[t]
        msgs.append(user_msg(q))
        msgs.append(assistant_msg(a))
    q, _ = turns[upto]
    msgs.append(user_msg(q))
    return msgs


def revert_request(system, turns, revert_turn, edited_q):
    """Messages as opencode would send after reverting turn `revert_turn` and
    resending an edited user message: keep history through the assistant answer
    of turn (revert_turn-1), drop the rest, append the edited user message."""
    msgs = [{"role": "system", "content": system}]
    for t in range(revert_turn):
        q, a = turns[t]
        msgs.append(user_msg(q))
        msgs.append(assistant_msg(a))
    msgs.append(user_msg(edited_q))
    return msgs


def prefix_tokens_est(system, turns, upto):
    """Estimate tokens of history through assistant answer of turn upto-1
    (i.e., through the assistant of turn upto-1 => helper: inclusive stop)."""
    n = ntok(system)
    for t in range(upto):
        q, a = turns[t]
        n += ntok(q) + ntok(a)
    return n
