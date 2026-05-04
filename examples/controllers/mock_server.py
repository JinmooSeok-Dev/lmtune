"""Mock controller HTTP server — minikube swap-test 용 reference.

stdlib http.server 만 사용 (의존성 0). 외부 LLM/agent API 없이도
HTTPController plug-in 자체가 동작함을 입증.

사용:
  python examples/controllers/mock_server.py --port 8090
  → POST http://localhost:8090/ask  → {"params": {...}}
  → POST http://localhost:8090/tell → 204

axis 종류별로 결정론적 first-value 선택 (random 아님):
  - categorical/bool : values[0]
  - int/float       : low

→ test 시 sweep 결과가 reproducible. 진짜 random 이 필요하면
   examples/controllers/random_server.py 사용.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logging.basicConfig(level=logging.INFO, format="[mock-ctrl] %(message)s")
log = logging.getLogger("mock-ctrl")


def _first_value(axis: dict[str, Any]) -> Any:
    kind = axis.get("kind")
    if kind in ("categorical",):
        vs = axis.get("values") or [None]
        return vs[0]
    if kind == "bool":
        return False
    if kind in ("int",):
        return int(axis["low"])
    if kind in ("float", "log_uniform"):
        return float(axis["low"])
    return None


# 서버 전역 상태 — tell 받은 결과 누적 (디버깅·검증용)
_TELL_HISTORY: list[dict[str, Any]] = []


class Handler(BaseHTTPRequestHandler):
    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, code: int, payload: dict | None = None) -> None:
        body = json.dumps(payload or {}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/ask":
            req = self._read_json()
            study_id = req.get("study_id", "?")
            axes = req.get("active_axes", [])
            history_len = len(req.get("history", []))
            params = {a["name"]: _first_value(a) for a in axes}
            log.info(
                "ASK study=%s history=%d → %s", study_id, history_len, params
            )
            self._send_json(200, {"params": params})
            return

        if self.path == "/tell":
            req = self._read_json()
            log.info(
                "TELL study=%s status=%s value=%s",
                req.get("study_id", "?"), req.get("status"), req.get("value"),
            )
            _TELL_HISTORY.append(req)
            self.send_response(204)
            self.end_headers()
            return

        if self.path == "/history":
            self._send_json(200, {"tells": _TELL_HISTORY})
            return

        self._send_json(404, {"error": f"unknown path: {self.path}"})

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default access log
        return


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    log.info("listening on http://%s:%d", args.host, args.port)
    log.info("endpoints: POST /ask  POST /tell  GET /history")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down — %d tells received", len(_TELL_HISTORY))
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
