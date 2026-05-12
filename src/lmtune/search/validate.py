"""Search-space pre-flight validator — 4 카테고리 검증.

study start 전에 search-space YAML 이 (1) schema 제대로, (2) 우리가 사용하는 vllm
0.17.1 의 axis allowlist 안에 들어있고, (3) 이미 영속화된 regression 패턴 (R23/R25/
R26/R28 등) 을 트리거하지 않으며, (4) feasibility constraints 가 N=200 sample 시
대부분 infeasible 이어서 study 무의미 한 상태가 아닌지 확인.

사용자가 매번 4단계 source 검증 (CLAUDE.md § PR 게이트) 을 못 따라잡아도, 본
validator 가 study start 전에 hard block 으로 차단해 동일 결함이 재발하는 걸 방지.

Source-of-truth:
- b200/registry/vllm_0.17.1_axes.yaml          — verified axis allowlist
- b200/registry/known_regressions.yaml         — R-id 별 매칭 룰
- b200/docs/regressions.md                     — 사람용 catalog (yaml 의 ref:)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lmtune.models import by_name
from lmtune.search.feasibility import (
    Environment,
    FeasibilityReport,
    load_constraints,
)
from lmtune.search.feasibility import (
    evaluate as evaluate_feasibility,
)
from lmtune.search.space import SearchSpace, load_space

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AXIS_REGISTRY = _REPO_ROOT / "b200" / "registry" / "vllm_0.17.1_axes.yaml"
_DEFAULT_REGRESSION_REGISTRY = _REPO_ROOT / "b200" / "registry" / "known_regressions.yaml"


@dataclass
class Issue:
    """단일 검증 결함."""

    severity: str  # "block" | "warn"
    category: str  # "schema" | "axis_allowlist" | "regression" | "feasibility"
    msg: str
    axis: str | None = None
    ref: str | None = None  # ID (R23, R25 등) 또는 docs anchor


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    feasibility_stats: dict[str, Any] | None = None  # n_sampled, n_infeasible, %, top failures

    @property
    def blocked(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    @property
    def n_block(self) -> int:
        return sum(1 for i in self.issues if i.severity == "block")

    @property
    def n_warn(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")


# ---------------------------------------------------------------------------
# helpers — axis values 의 "sample 가능한 모든 후보" 로 평탄화


def _enumerate_axis_values(axis_spec: Any) -> list[Any]:
    """axis spec → 가능한 값 리스트 (categorical/bool 만 정확, int/float 은 [low, high] 표현)."""
    if not isinstance(axis_spec, dict):
        return []
    t = axis_spec.get("type") or axis_spec.get("kind")
    if t in ("categorical",) and "values" in axis_spec:
        return list(axis_spec["values"])
    if t == "bool":
        return [True, False]
    if t in ("int", "float", "log_uniform"):
        # range axis — 검증 룰의 ">N" 매칭은 high 만 보면 충분
        return [axis_spec.get("low"), axis_spec.get("high")]
    if "values" in axis_spec:  # generic
        return list(axis_spec["values"])
    return []


def _read_axes_block(space_yaml_text: str) -> dict[str, Any]:
    """space yaml 에서 axes 블록만 dict 로 반환 (load_space 가 active_if 필터를 안 거치도록)."""
    try:
        spec = yaml.safe_load(space_yaml_text)
    except Exception:
        return {}
    axes = (spec or {}).get("axes")
    if isinstance(axes, dict):
        return axes
    if isinstance(axes, list):
        out: dict[str, Any] = {}
        for entry in axes:
            if isinstance(entry, dict) and "name" in entry:
                out[entry["name"]] = entry
        return out
    return {}


# ---------------------------------------------------------------------------
# 1) Schema validation — axis name unique, type/values 형식


def validate_schema(space_yaml_text: str) -> list[Issue]:
    issues: list[Issue] = []
    try:
        spec = yaml.safe_load(space_yaml_text)
    except yaml.YAMLError as e:
        return [Issue("block", "schema", f"YAML parse error: {e}")]
    if not isinstance(spec, dict):
        return [Issue("block", "schema", "search-space root 가 dict 아님")]

    if not spec.get("name"):
        issues.append(Issue("warn", "schema", "search-space.name 미지정"))

    axes = spec.get("axes")
    if axes is None:
        return [Issue("block", "schema", "axes 블록 누락")]

    axes_dict = axes if isinstance(axes, dict) else None
    axes_list = axes if isinstance(axes, list) else None
    if axes_dict is None and axes_list is None:
        return [Issue("block", "schema", "axes 가 dict 또는 list 아님")]

    seen: set[str] = set()
    iter_axes = (
        list(axes_dict.items())
        if axes_dict
        else [(e.get("name"), e) for e in (axes_list or []) if isinstance(e, dict)]
    )

    for name, ax in iter_axes:
        if not name:
            issues.append(Issue("block", "schema", "axis name 비어있음"))
            continue
        if name in seen:
            issues.append(Issue("block", "schema", f"axis 중복: {name}", axis=name))
        seen.add(name)
        if not isinstance(ax, dict):
            issues.append(Issue("block", "schema", "axis spec 이 dict 아님", axis=name))
            continue
        t = ax.get("type") or ax.get("kind")
        if not t:
            issues.append(Issue("block", "schema", "axis type 누락", axis=name))
            continue
        if t == "categorical" and not ax.get("values"):
            issues.append(Issue("block", "schema", "categorical 인데 values 없음", axis=name))
        if t in ("int", "float", "log_uniform"):
            if ax.get("low") is None or ax.get("high") is None:
                issues.append(Issue("block", "schema", f"{t} 인데 low/high 누락", axis=name))
            elif ax["low"] > ax["high"]:
                issues.append(Issue("block", "schema", "low > high", axis=name))
    return issues


# ---------------------------------------------------------------------------
# 2) Axis allowlist — vllm 0.17.1 catalog 와 대조


def _load_axis_registry(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or _DEFAULT_AXIS_REGISTRY
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("axes") or {}


# Axis 이름이 vllm CLI 가 아니라 lmtune 자체 메타이거나 active_if helper 인 경우.
# 여기 등재된 axis 는 vllm allowlist 에 없어도 통과.
_LMTUNE_META_AXES = {
    "well_lit_path",
    "model_id",
    "node_split_strategy",
    "engine_backend",
    "serving_stack",
    "ep_strategy",  # simulator-only
    "sequence_parallel",  # vllm 0.17.1 에서 별도 flag 없이 auto
    "intra_node_type",
    "cross_node_type",
}


def validate_axis_allowlist(
    space_axes: dict[str, Any],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    reg = registry if registry is not None else _load_axis_registry()
    if not reg:
        return [
            Issue(
                "warn",
                "axis_allowlist",
                f"axis registry 부재: {_DEFAULT_AXIS_REGISTRY}. allowlist 검증 건너뜀.",
            )
        ]
    for name, ax in space_axes.items():
        if name in _LMTUNE_META_AXES:
            continue
        entry = reg.get(name)
        if entry is None:
            issues.append(
                Issue(
                    "block",
                    "axis_allowlist",
                    f"axis '{name}' 가 vllm 0.17.1 catalog 에 없음. "
                    f"4단계 source 검증 (CLAUDE.md § PR 게이트) 후 "
                    f"b200/registry/vllm_0.17.1_axes.yaml 에 entry 추가하세요.",
                    axis=name,
                )
            )
            continue
        if entry.get("deprecated_or_unsupported"):
            gates = entry.get("gates") or "vllm 0.17.1 미지원."
            issues.append(
                Issue(
                    "block",
                    "axis_allowlist",
                    f"axis '{name}' 는 vllm 0.17.1 에서 미지원/deprecated: {gates}",
                    axis=name,
                )
            )
            continue
        # categorical 의 choices 매칭
        choices = entry.get("choices")
        if choices and isinstance(ax, dict) and ax.get("type") == "categorical":
            for v in ax.get("values") or []:
                if v not in choices:
                    issues.append(
                        Issue(
                            "block",
                            "axis_allowlist",
                            f"axis '{name}' 값 {v!r} 이 vllm 의 choices {choices} 에 없음",
                            axis=name,
                        )
                    )
    return issues


# ---------------------------------------------------------------------------
# 3) Known regressions — yaml 의 매칭 룰 적용


def _load_regression_registry(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _DEFAULT_REGRESSION_REGISTRY
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("regressions") or []


def _match_regression(
    rule: dict[str, Any],
    axes: dict[str, Any],
    model_meta: dict[str, Any] | None,
) -> bool:
    m = rule.get("match") or {}

    if "has_axis" in m and m["has_axis"] not in axes:
        return False

    if "has_any_axis" in m and not any(a in axes for a in (m["has_any_axis"] or [])):
        return False

    if "has_axis_value_gt" in m:
        sub = m["has_axis_value_gt"]
        ax = axes.get(sub["axis"])
        if not ax:
            return False
        vals = _enumerate_axis_values(ax)
        if not any(isinstance(v, (int, float)) and v > sub["gt"] for v in vals):
            return False

    if "has_axis_combo_with_values" in m:
        sub = m["has_axis_combo_with_values"]
        a = sub.get("a", {})
        b = sub.get("b", {})
        if not _axis_value_matches(axes.get(a.get("axis")), a):
            return False
        if not _axis_value_matches(axes.get(b.get("axis")), b):
            return False

    if "model_has_kv_heads_le" in m:
        if not model_meta:
            return False  # model 메타 없으면 보수적으로 매칭 안 함
        kv = model_meta.get("num_kv_heads")
        if kv is None or kv > m["model_has_kv_heads_le"]:
            return False

    return True


def _axis_value_matches(axis_spec: Any, expect: dict[str, Any]) -> bool:
    if not axis_spec:
        return False
    vals = _enumerate_axis_values(axis_spec)
    if "equals" in expect:
        return expect["equals"] in vals
    if "gt" in expect:
        return any(isinstance(v, (int, float)) and v > expect["gt"] for v in vals)
    if "value_not_in" in expect:
        not_in = set(expect["value_not_in"])
        return any(v not in not_in for v in vals)
    return True


def validate_known_regressions(
    space_axes: dict[str, Any],
    model_meta: dict[str, Any] | None = None,
    registry: list[dict[str, Any]] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    rules = registry if registry is not None else _load_regression_registry()
    if not rules:
        return [
            Issue(
                "warn",
                "regression",
                f"regression registry 부재: {_DEFAULT_REGRESSION_REGISTRY}. R-list 매칭 건너뜀.",
            )
        ]
    for rule in rules:
        if _match_regression(rule, space_axes, model_meta):
            issues.append(
                Issue(
                    severity=str(rule.get("severity", "warn")),
                    category="regression",
                    msg=str(rule.get("msg", "")),
                    ref=str(rule.get("id", "")),
                )
            )
    return issues


# ---------------------------------------------------------------------------
# 4) Feasibility coverage — N=200 sample 후 infeasible % 측정


def validate_feasibility_coverage(
    space: SearchSpace,
    space_yaml_path: Path,
    environment: Environment | None,
    model_id: str | None,
    n_samples: int = 200,
    seed: int = 17,
    block_threshold: float = 0.95,
    warn_threshold: float = 0.70,
) -> tuple[list[Issue], dict[str, Any]]:
    """Sample N candidates 하면서 feasibility evaluator 통과 비율 측정.

    feasibility_constraints 가 너무 빡빡해 95%+ infeasible 이면 study 가 sampler
    예산을 다 prune 으로 태움 → block. 70~95% 면 warn (의도적일 수도).
    """
    constraints = load_constraints(space_yaml_path)
    if not constraints:
        return [], {"n_samples": 0, "n_infeasible": 0, "ratio": 0.0, "note": "no constraints"}

    if environment is None:
        return (
            [
                Issue(
                    "warn",
                    "feasibility",
                    "--cluster-env 미지정 → feasibility coverage 검증 건너뜀.",
                )
            ],
            {"n_samples": 0, "n_infeasible": 0, "ratio": 0.0},
        )

    model_spec = None
    if model_id:
        try:
            model_spec = by_name(model_id)
        except Exception:
            model_spec = None
    rng = random.Random(seed)
    n_inf = 0
    failure_counts: dict[str, int] = {}
    for _ in range(n_samples):
        params = _sample_params(space, rng)
        report: FeasibilityReport = evaluate_feasibility(
            params=params, environment=environment, model=model_spec, constraints=constraints
        )
        if not report.feasible:
            n_inf += 1
            for f in report.failures:
                fid = f.get("id") or "?"
                failure_counts[fid] = failure_counts.get(fid, 0) + 1

    ratio = n_inf / n_samples if n_samples else 0.0
    top_fails = sorted(failure_counts.items(), key=lambda kv: -kv[1])[:5]
    stats = {
        "n_samples": n_samples,
        "n_infeasible": n_inf,
        "ratio": ratio,
        "top_failures": top_fails,
    }

    issues: list[Issue] = []
    if ratio >= block_threshold:
        issues.append(
            Issue(
                "block",
                "feasibility",
                f"{ratio * 100:.1f}% trial 이 infeasible (N={n_samples}). "
                f"feasibility_constraints 가 너무 빡빡합니다. top: "
                f"{', '.join(f'{k}({v})' for k, v in top_fails)}",
            )
        )
    elif ratio >= warn_threshold:
        issues.append(
            Issue(
                "warn",
                "feasibility",
                f"{ratio * 100:.1f}% trial 이 infeasible (N={n_samples}). "
                f"sampler 예산이 prune 에 많이 소비됩니다. top: "
                f"{', '.join(f'{k}({v})' for k, v in top_fails)}",
            )
        )
    return issues, stats


def _sample_params(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    """간단 sampler — categorical/bool 만 random pick, range 는 endpoint 둘 중 하나."""
    out: dict[str, Any] = {}
    for ax in space.axes:
        # active_if 는 무시 (full sample) — feasibility 가 알아서 판단
        if ax.kind == "categorical":
            out[ax.name] = rng.choice(list(ax.values or []))
        elif ax.kind == "bool":
            out[ax.name] = rng.choice([True, False])
        elif ax.kind in ("int",):
            lo, hi = int(ax.low or 0), int(ax.high or 0)
            out[ax.name] = rng.randint(lo, hi) if hi >= lo else lo
        elif ax.kind in ("float", "log_uniform"):
            lo, hi = float(ax.low or 0), float(ax.high or 0)
            out[ax.name] = rng.uniform(lo, hi) if hi >= lo else lo
        else:
            if ax.values:
                out[ax.name] = rng.choice(list(ax.values))
    return out


# ---------------------------------------------------------------------------
# Top-level entry point


def validate_search_space(
    space_yaml_path: Path,
    environment: Environment | None = None,
    model_id: str | None = None,
    n_samples: int = 200,
    axis_registry: dict[str, dict[str, Any]] | None = None,
    regression_registry: list[dict[str, Any]] | None = None,
) -> ValidationReport:
    """4 카테고리 검증 → ValidationReport. blocked=True 면 study start 차단."""
    text = space_yaml_path.read_text(encoding="utf-8")
    issues: list[Issue] = []
    issues += validate_schema(text)

    axes_block = _read_axes_block(text)
    issues += validate_axis_allowlist(axes_block, axis_registry)

    model_meta = None
    if model_id:
        try:
            spec = by_name(model_id)
            model_meta = {
                "num_kv_heads": getattr(spec, "num_kv_heads", None),
                "is_moe": getattr(spec, "is_moe", False),
                "has_mla": getattr(spec, "has_mla", False),
            }
        except Exception:
            model_meta = None

    issues += validate_known_regressions(axes_block, model_meta, regression_registry)

    space = load_space(space_yaml_path)
    feas_issues, stats = validate_feasibility_coverage(
        space=space,
        space_yaml_path=space_yaml_path,
        environment=environment,
        model_id=model_id,
        n_samples=n_samples,
    )
    issues += feas_issues

    return ValidationReport(issues=issues, feasibility_stats=stats)
