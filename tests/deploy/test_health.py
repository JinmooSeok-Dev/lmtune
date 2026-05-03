from __future__ import annotations

import http.server
import json
import threading
from contextlib import contextmanager

from lmtune.deploy.health import probe_openai_models, warmup_one_token


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            body = json.dumps({"object": "list", "data": [{"id": "demo"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path.endswith("/v1/chat/completions"):
            body = json.dumps({"choices": [{"message": {"content": "."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


@contextmanager
def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


def test_probe_ok_with_models_endpoint():
    with _server() as url:
        r = probe_openai_models(url)
    assert r.ready is True
    assert "1 model" in r.detail


def test_probe_fails_unreachable():
    r = probe_openai_models("http://127.0.0.1:1/v1")
    assert r.ready is False


def test_warmup_one_token():
    with _server() as url:
        r = warmup_one_token(url, "demo")
    assert r.ready is True
