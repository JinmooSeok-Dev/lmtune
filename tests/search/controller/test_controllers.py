"""Controller plug-in 테스트 — ABC 4 구현체가 같은 contract 를 만족하는지.

진짜 plug-in 이 되려면: Random / Mock / HTTP 가 OptunaController 와 동일 ABC
로 ask/tell 호출 가능해야 함. 본 테스트는 Optuna 호출 경로 없이 검증.

- RandomController, MockController : 단위
- HTTPController : embedded mock_server 기동 후 실제 HTTP round-trip
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lmtune.search.controller import (
    Controller,
    HTTPController,
    MockController,
    RandomController,
)
from lmtune.search.space import Axis


# ---- 공통 fixture: 4-axis test space ---------------------------------------

def _axes() -> list[Axis]:
    return [
        Axis(name="max_num_seqs", kind="categorical", values=[32, 64, 128]),
        Axis(name="enable_prefix_caching", kind="bool"),
        Axis(name="block_size", kind="int", low=16, high=64, step=16),
        Axis(name="gpu_memory_utilization", kind="float", low=0.80, high=0.92),
    ]


# ---- RandomController ------------------------------------------------------


def test_random_ask_returns_all_axes():
    c = RandomController(seed=42)
    p = c.ask(_axes())
    assert set(p.keys()) == {
        "max_num_seqs", "enable_prefix_caching", "block_size", "gpu_memory_utilization"
    }


def test_random_categorical_in_values():
    c = RandomController(seed=42)
    for _ in range(20):
        p = c.ask(_axes())
        assert p["max_num_seqs"] in [32, 64, 128]
        assert isinstance(p["enable_prefix_caching"], bool)
        assert 16 <= p["block_size"] <= 64 and p["block_size"] % 16 == 0
        assert 0.80 <= p["gpu_memory_utilization"] <= 0.92


def test_random_seed_deterministic():
    a = RandomController(seed=7).ask(_axes())
    b = RandomController(seed=7).ask(_axes())
    assert a == b


def test_random_tell_is_noop():
    c = RandomController(seed=1)
    c.tell({"x": 1}, value=10.0, status="completed")
    c.tell({"x": 2}, value=None, status="crash")
    # noop — no exception


def test_random_implements_abc():
    assert isinstance(RandomController(), Controller)


# ---- MockController --------------------------------------------------------


def test_mock_returns_first_value_default():
    c = MockController()
    p = c.ask(_axes())
    assert p["max_num_seqs"] == 32          # values[0]
    assert p["enable_prefix_caching"] is False
    assert p["block_size"] == 16            # low
    assert p["gpu_memory_utilization"] == 0.80


def test_mock_scripted_sequence():
    scripted = [
        {"max_num_seqs": 64, "block_size": 32},
        {"max_num_seqs": 128, "block_size": 48},
    ]
    c = MockController(scripted_params=scripted)
    p1 = c.ask(_axes())
    p2 = c.ask(_axes())
    assert p1["max_num_seqs"] == 64 and p1["block_size"] == 32
    assert p2["max_num_seqs"] == 128 and p2["block_size"] == 48
    # scripted 미지정 axis 는 default
    assert p1["enable_prefix_caching"] is False


def test_mock_exhausted_after_scripted():
    c = MockController(scripted_params=[{"max_num_seqs": 64}])
    assert not c.exhausted
    c.ask(_axes())
    assert c.exhausted


def test_mock_records_tells():
    c = MockController()
    c.tell({"x": 1}, value=10.0, status="completed")
    c.tell({"y": 2}, value=None, status="crash", metadata={"err": "oom"})
    assert len(c.tells) == 2
    assert c.tells[0]["status"] == "completed" and c.tells[0]["value"] == 10.0
    assert c.tells[1]["metadata"]["err"] == "oom"


# ---- HTTPController ── embedded mock server round-trip ---------------------


class _StubHandler(BaseHTTPRequestHandler):
    """Test 용 stub server — RandomController 흉내."""
    request_log: list[dict] = []

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_POST(self):  # noqa: N802
        body = self._read_json()
        self.request_log.append({"path": self.path, "body": body})
        if self.path == "/ask":
            # axis 의 첫 값 반환 (mock_server.py 와 동일 정책)
            params = {}
            for a in body.get("active_axes", []):
                if a["kind"] == "categorical":
                    params[a["name"]] = a["values"][0]
                elif a["kind"] == "bool":
                    params[a["name"]] = False
                elif a["kind"] == "int":
                    params[a["name"]] = a["low"]
                elif a["kind"] in ("float", "log_uniform"):
                    params[a["name"]] = a["low"]
            payload = json.dumps({"params": params}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/tell":
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args, **_kw):
        return


@pytest.fixture
def stub_server():
    _StubHandler.request_log = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)  # 0 = OS-assigned
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)  # ensure listening
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


def test_http_ask_round_trip(stub_server):
    httpx = pytest.importorskip("httpx")  # noqa: F841
    c = HTTPController(url=f"http://127.0.0.1:{stub_server}", study_id="st-test")
    p = c.ask(_axes())
    assert p["max_num_seqs"] == 32     # stub 이 first-value 반환
    assert p["enable_prefix_caching"] is False
    # request body 가 active_axes 와 study_id 를 포함했는지
    req = _StubHandler.request_log[0]["body"]
    assert req["study_id"] == "st-test"
    assert {a["name"] for a in req["active_axes"]} == {
        "max_num_seqs", "enable_prefix_caching", "block_size", "gpu_memory_utilization",
    }


def test_http_tell_persists_history(stub_server):
    pytest.importorskip("httpx")
    c = HTTPController(url=f"http://127.0.0.1:{stub_server}", study_id="st-test")
    c.tell({"max_num_seqs": 64}, value=142.5, status="completed")
    c.tell({"max_num_seqs": 32}, value=None, status="crash", metadata={"e": "oom"})
    assert len(c._history) == 2
    # tell 이 server 에 도달
    paths = [r["path"] for r in _StubHandler.request_log]
    assert paths.count("/tell") == 2


def test_http_implements_abc():
    pytest.importorskip("httpx")
    c = HTTPController(url="http://localhost:9999", study_id="x")
    assert isinstance(c, Controller)


# ---- 폴리모피즘 — 같은 코드가 4 종 controller 모두 처리 -------------------


@pytest.mark.parametrize("ctrl", [
    RandomController(seed=1),
    MockController(),
])
def test_swappable_uniform_interface(ctrl):
    """진짜 plug-in 이려면 호출자가 구현체를 몰라도 동작해야 함."""
    assert isinstance(ctrl, Controller)
    p = ctrl.ask(_axes())
    assert isinstance(p, dict)
    assert all(isinstance(k, str) for k in p)
    ctrl.tell(p, value=42.0, status="completed")
    # name property 검증 (UI / log 용)
    assert isinstance(ctrl.name, str) and len(ctrl.name) > 0
