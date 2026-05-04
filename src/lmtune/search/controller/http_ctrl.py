"""HTTPController — 외부 LLM/agent/RL controller 통합 base.

URL 에 POST 만 하므로 controller 구현 언어·프레임워크 무관 (Python/TypeScript/
Go/Rust/...). Anthropic Claude SDK / OpenAI / 자체 RL agent / LangGraph
워크플로우 / autoresearch 모두 같은 계약으로 plug-in.

계약 (`docs/architecture.md` § Layer 4 Pluggability):
  POST {url}/ask
    body: {study_id, active_axes:[{name,kind,values,low,high,step}], history:[...], context}
    resp: {"params": {axis_name: value, ...}}

  POST {url}/tell
    body: {study_id, params, value, status, metadata}
    resp: 204 No Content (or 200 with optional payload)

reference 구현은 `examples/controllers/`.
"""
from __future__ import annotations

import logging
from typing import Any

from lmtune.search.controller.base import Controller
from lmtune.search.space import Axis

log = logging.getLogger(__name__)


def _axis_to_dict(a: Axis) -> dict[str, Any]:
    """controller 서비스 가 active_axes 를 해석하기 위한 self-describing JSON."""
    d: dict[str, Any] = {"name": a.name, "kind": a.kind}
    if a.values is not None:
        d["values"] = list(a.values)
    if a.low is not None:
        d["low"] = a.low
    if a.high is not None:
        d["high"] = a.high
    if a.step is not None:
        d["step"] = a.step
    return d


class HTTPController(Controller):
    def __init__(
        self,
        url: str,
        study_id: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ):
        # httpx 는 [runners] extra 의 transitive — 여기서 import 늦게.
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HTTPController requires httpx. install with `pip install httpx` "
                "or `pip install lmtune[runners]`"
            ) from e
        self._url = url.rstrip("/")
        self._study_id = study_id
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return f"http({self._url})"

    def ask(
        self, active_axes: list[Axis], *, context: dict | None = None
    ) -> dict[str, Any]:
        import httpx
        body = {
            "study_id": self._study_id,
            "active_axes": [_axis_to_dict(a) for a in active_axes],
            "history": self._history,
            "context": dict(context or {}),
        }
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                r = httpx.post(
                    f"{self._url}/ask", json=body, timeout=self._timeout_s,
                )
                r.raise_for_status()
                data = r.json()
                if "params" not in data:
                    raise RuntimeError(f"controller /ask response missing 'params': {data}")
                return data["params"]
            except (httpx.HTTPError, RuntimeError) as e:
                last_exc = e
                log.warning(
                    "HTTPController ask failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, e,
                )
        # 재시도 모두 실패 → 호출자에게 raise (Study 가 어떻게 처리할지 결정)
        raise RuntimeError(
            f"HTTPController.ask exhausted retries to {self._url}: {last_exc}"
        )

    def tell(
        self,
        params: dict[str, Any],
        *,
        value: float | list[float] | None,
        status: str,
        metadata: dict | None = None,
    ) -> None:
        import httpx
        body = {
            "study_id": self._study_id,
            "params": dict(params),
            "value": value,
            "status": status,
            "metadata": dict(metadata or {}),
        }
        try:
            r = httpx.post(
                f"{self._url}/tell", json=body, timeout=self._timeout_s,
            )
            # 204 / 200 모두 ok. 학습 실패는 controller 의 내부 문제 — 우리 loop 는 진행.
            if r.status_code >= 500:
                log.warning("HTTPController.tell got %d; controller may have lost state", r.status_code)
        except Exception as e:  # noqa: BLE001
            log.warning("HTTPController.tell failed (continuing): %s", e)

        self._history.append({
            "params": dict(params), "value": value, "status": status,
        })
