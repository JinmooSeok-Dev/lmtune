"""MockController — scripted params 순차 반환. 테스트·minikube smoke 용.

핵심 사용:
1. 단위 테스트 — Study.run() 의 ask/tell 흐름 검증, sampler 학습 noise 제거
2. minikube swap-test — 진짜 LLM/agent 호출 없이 controller plug-in 자체가
   동작함을 입증 (e.g. `lmtune search start --controller mock`)
3. 회귀 디버깅 — 특정 (params, status) sequence 를 재생해 사용자 보고 reproduce

scripted_params=None 이면 axis 의 첫 값으로 모두 채운 default params 반복.
"""
from __future__ import annotations

import logging
from typing import Any

from lmtune.search.controller.base import Controller
from lmtune.search.space import Axis

log = logging.getLogger(__name__)


def _axis_default(axis: Axis) -> Any:
    if axis.kind in ("categorical",):
        return (list(axis.values) or [None])[0]
    if axis.kind == "bool":
        return False
    if axis.kind == "int":
        return int(axis.low)
    if axis.kind == "float":
        return float(axis.low)
    if axis.kind == "log_uniform":
        return float(axis.low)
    return None


class MockController(Controller):
    def __init__(self, scripted_params: list[dict] | None = None):
        self._scripted = list(scripted_params or [])
        self._idx = 0
        # 외부에서 검증 가능하도록 tell 받은 결과 보관
        self.tells: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def exhausted(self) -> bool:
        # scripted 가 주어지면 소진 후 exhausted, 아니면 무한.
        return bool(self._scripted) and self._idx >= len(self._scripted)

    def ask(self, active_axes: list[Axis], *, context: dict | None = None) -> dict[str, Any]:
        if self._idx < len(self._scripted):
            scripted = self._scripted[self._idx]
            self._idx += 1
            # active axes 만 선택, 누락된 것은 default 로 채움
            return {a.name: scripted.get(a.name, _axis_default(a)) for a in active_axes}
        # scripted 소진 — 모든 axis 의 default 반복 (사용자가 exhausted 체크 안 했으면)
        return {a.name: _axis_default(a) for a in active_axes}

    def tell(self, params, *, value, status, metadata=None) -> None:
        self.tells.append({
            "params": dict(params), "value": value, "status": status,
            "metadata": dict(metadata or {}),
        })
