#!/usr/bin/env python
"""Capture-only forwarding proxy: :8001 -> :8000, logs each request body (NDJSON)."""
import http.client
import http.server
import json
import os
import sys
import time

LISTEN = ("127.0.0.1", int(os.environ.get("CAPTURE_LISTEN_PORT", "8001")))
UPSTREAM = ("127.0.0.1", int(os.environ.get("CAPTURE_UPSTREAM_PORT", "8000")))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "CAPTURE_LOG", "/tmp/capture_ndjson.log")
DROP = ["host", "connection", "content-length", "accept-encoding"]


def log(path, body):
    with open(LOG, "ab") as f:
        f.write(json.dumps({"t": time.time(), "path": path,
                            "body": body.decode("utf-8", "replace")}).encode("utf-8") + b"\n")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path.startswith("/v1/chat/completions"):
            log(self.path, body)
        try:
            conn = http.client.HTTPConnection(*UPSTREAM, timeout=1800)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in DROP}
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("connection", "content-length"):
                    self.send_header(k, v)
            if resp.getheader("transfer-encoding", "").lower() == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            conn.close()
        except Exception as exc:  # noqa: BLE001
            try:
                self.send_error(502, f"proxy error: {exc}")
            except Exception:  # noqa: BLE001
                pass

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(LISTEN, Handler).serve_forever()
