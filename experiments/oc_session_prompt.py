#!/usr/bin/env python3
"""Rebuild a real opencode session into an OpenAI-style chat request.

Reads the opencode SQLite DB read-only, turns a session's messages/parts into
user/assistant messages (tool calls and their outputs are inlined as text so the
context keeps the file contents and command outputs the model may repeat), keeps
the tail up to the last user message, and drops the oldest turns until the
chat-template token count fits a budget.

Usage:
  oc_session_prompt.py --list
  oc_session_prompt.py --session <id> --max-tokens 24000 --out prompts.json
"""
import argparse
import datetime
import json
import os
import sqlite3
import sys

DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
URI = f"file:{DB}?mode=ro&immutable=1"


def connect():
    return sqlite3.connect(URI, uri=True)


def list_sessions(con):
    cur = con.cursor()
    cur.execute(
        """select id,title,tokens_input,tokens_output,time_updated,
                  (select count(*) from message m where m.session_id=s.id)
           from session s order by time_updated desc limit 40"""
    )
    for sid, title, ti, to, tu, nm in cur.fetchall():
        ts = datetime.datetime.fromtimestamp(tu / 1000).strftime("%m-%d %H:%M")
        print(f"{ts} msgs={nm:3d} in={ti:8d} {sid}  {title[:70]}")


def reconstruct(con, sid):
    cur = con.cursor()
    cur.execute(
        "select id,data from message where session_id=? order by time_created", (sid,)
    )
    msgs = []
    for mid, data in cur.fetchall():
        role = json.loads(data).get("role")
        cur.execute(
            "select data from part where message_id=? order by time_created", (mid,)
        )
        if role == "user":
            texts = []
            for (pd,) in cur.fetchall():
                p = json.loads(pd)
                if p.get("type") == "text" and p.get("text"):
                    texts.append(p["text"])
            text = "\n".join(texts).strip()
            if text:
                msgs.append({"role": "user", "content": text})
        elif role == "assistant":
            chunks = []
            for (pd,) in cur.fetchall():
                p = json.loads(pd)
                t = p.get("type")
                if t == "text" and p.get("text"):
                    chunks.append(p["text"])
                elif t == "tool":
                    st = p.get("state") or {}
                    inp = json.dumps(st.get("input"), ensure_ascii=False)
                    out = st.get("output") or st.get("error") or ""
                    chunks.append(
                        f"[tool:{p.get('tool')}] {inp}\n<result>\n{out}\n</result>"
                    )
            text = "\n".join(c for c in chunks if c).strip()
            if text:
                msgs.append({"role": "assistant", "content": text})
    # Keep the tail ending on a user message so the model answers a real request.
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=-1)
    if last_user < 0:
        return [], 0
    return msgs[: last_user + 1], last_user


def token_len(tok, msgs):
    # Token count of the raw content; the chat template adds a small fixed
    # overhead (~tens of tokens) which the budget margin absorbs.
    return sum(len(tok.encode(m["content"]).ids) for m in msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--tokenizer")
    ap.add_argument("--out")
    a = ap.parse_args()

    global URI
    URI = f"file:{a.db}?mode=ro&immutable=1"
    con = connect()
    if a.list or not a.session:
        list_sessions(con)
        if not a.session:
            return

    msgs, last_user = reconstruct(con, a.session)
    if not msgs:
        sys.exit("no user message in session")
    total_chars = sum(len(m["content"]) for m in msgs)

    if a.tokenizer:
        import glob

        from tokenizers import Tokenizer

        path = a.tokenizer
        if os.path.isdir(path):
            path = glob.glob(os.path.join(path, "tokenizer.json"))[0]
        tok = Tokenizer.from_file(path)

        # Fill backwards from the real request. Tool outputs here are huge, so a
        # whole-turn drop overshoots badly; instead keep the newest messages and
        # truncate one message's middle if it alone would blow the budget.
        out, used = [], 0
        for msg in reversed(msgs):
            ids = tok.encode(msg["content"]).ids
            if used + len(ids) <= a.max_tokens:
                out.append(msg)
                used += len(ids)
                continue
            remain = a.max_tokens - used
            if remain > 500:
                half = remain // 2
                out.append(
                    {
                        "role": msg["role"],
                        "content": tok.decode(ids[:half])
                        + "\n...[truncated]...\n"
                        + tok.decode(ids[-half:]),
                    }
                )
                used += remain
            break
        out.reverse()
        # The request must open on a user turn; fold any leading assistant/tool
        # content (often the one big message we truncated to fill the budget)
        # into the first user turn instead of dropping it.
        lead = []
        while out and out[0]["role"] != "user":
            lead.append(out.pop(0)["content"])
        if out and lead:
            out[0]["content"] = "\n\n".join(lead + [out[0]["content"]])
        merged = []  # some chat templates reject consecutive same-role turns
        for m in out:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n\n" + m["content"]
            else:
                merged.append(dict(m))
        msgs = merged
        n = used
        print(f"kept {len(msgs)} messages, {n} tokens (budget {a.max_tokens})")
    else:
        n = None
        print(f"{len(msgs)} messages, {total_chars} chars (no tokenizer)")

    print(f"session={a.session} last_user_index={last_user}")
    print("--- first kept message (head 200) ---")
    print(msgs[0]["content"][:200].replace("\n", " "))
    print("--- last message / the request (tail 500) ---")
    print(msgs[-1]["content"][-500:])
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"session": a.session, "messages": msgs}, f, ensure_ascii=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
