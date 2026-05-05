"""Derived metrics — 기본 메트릭과 session/request 필드에서 파생되는 계산식.

YAML 에서 `derived_metrics: [{name, formula}]` 로 선언적으로 추가할 수도 있고,
built-in 이름으로 가져다 쓸 수도 있다. 수식 평가는 ast.literal_eval 기반의
안전한 제한 평가기를 사용한다.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class DerivedSpec:
    name: str
    formula: str
    description: str = ""


_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr: str, context: dict[str, float]) -> float:
    """숫자 계산 + 단순 min/max/abs 만 허용. 변수는 `context` dict 에서 조회."""
    tree = ast.parse(expr, mode="eval")

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"disallowed constant: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in context:
                raise KeyError(f"unknown variable: {node.id}")
            v = context[node.id]
            return float(v) if v is not None else float("nan")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"disallowed op: {type(node.op).__name__}")
            return op(_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"disallowed unary: {type(node.op).__name__}")
            return op(_walk(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("function calls limited to simple names")
            fname = node.func.id
            args = [_walk(a) for a in node.args]
            if fname == "min":
                return min(args)
            if fname == "max":
                return max(args)
            if fname == "abs":
                return abs(args[0])
            raise ValueError(f"disallowed function: {fname}")
        raise ValueError(f"disallowed ast node: {type(node).__name__}")

    return _walk(tree)


# Built-in derived (이름만 사용 — formula 또는 함수 모두 제공)

BUILTIN_FORMULAS: dict[str, str] = {
    "prefix_hit_rate": "cached_tokens / input_tokens",
    "input_output_ratio": "input_tokens / output_tokens",
    "tool_call_ratio": "tool_call_count / turn_count",
    "tokens_per_usd": "(input_tokens + output_tokens) / cost_usd",
    "energy_per_token": "energy_wh / (input_tokens + output_tokens)",
    "cost_per_task": "total_cost_usd",  # sessions.total_cost_usd 에서 바로 조회
    # EuTB: success_rate / total_input_tokens (SWE-Effi)
    "eutb": "success_rate / total_input_tokens",
}


def compute_derived(specs: list[DerivedSpec], context: dict[str, float]) -> dict[str, float]:
    """선언된 derived spec 목록을 context 위에서 평가. 오류는 NaN 으로 기록."""
    out: dict[str, float] = {}
    for spec in specs:
        formula = spec.formula or BUILTIN_FORMULAS.get(spec.name)
        if not formula:
            out[spec.name] = float("nan")
            continue
        try:
            out[spec.name] = safe_eval(formula, context)
        except (ZeroDivisionError, KeyError, ValueError):
            out[spec.name] = float("nan")
    return out


def resolve_builtin(name: str) -> DerivedSpec | None:
    if name in BUILTIN_FORMULAS:
        return DerivedSpec(name=name, formula=BUILTIN_FORMULAS[name])
    return None
