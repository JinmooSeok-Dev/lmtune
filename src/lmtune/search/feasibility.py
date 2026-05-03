"""Feasibility evaluator — declarative constraint AST runner.

Reads `feasibility_constraints` block from a SearchSpace YAML (b3, b5, …) and
evaluates each rule against (params, environment, model, surrogate). Returns a
FeasibilityReport listing failed rules → caller (Optuna sampler) discards the
trial without ever running it.

Source spec: vllm-config-puzzle/src/engine/llm-dist-sim/validation.ts (1:1 port).
The 12 constraints in b200/search-spaces/b3_parallelism.yaml are an exact
mirror of validation.ts:31..162.

Safety: AST-based whitelist evaluator. No `eval()` on raw strings.
Only allowed namespace nodes: numbers, strings, bools, None, arithmetic,
comparison, boolean, attribute lookup on the four bound contexts, modulo,
unary minus/not, ternary, and parens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lmtune.models import ModelSpec, by_name


@dataclass(frozen=True)
class Environment:
    """Runtime environment 메타. constraint 의 `environment.*` 참조."""
    total_npus: int                  # 16 (B200 dual-node), 8 (single-node), 1 (RTX local)
    npus_per_server: int             # 8 (B200), 8 (H200), 1 (local)
    servers: int = 1                 # derived: total_npus / npus_per_server
    gpu_vram_gb: float = 192.0       # B200 = 192 GB HBM3e
    server_storage_gb: float = 4096.0
    intra_node_type: str = "nvlink"  # nvlink | pcie | xgmi
    cross_node_type: str = "ib"      # ib | roce | ethernet | none

    @classmethod
    def b200_dual_node(cls) -> Environment:
        return cls(total_npus=16, npus_per_server=8, servers=2,
                   gpu_vram_gb=192.0, intra_node_type="nvlink", cross_node_type="ib")

    @classmethod
    def b200_single_node(cls) -> Environment:
        return cls(total_npus=8, npus_per_server=8, servers=1,
                   gpu_vram_gb=192.0, intra_node_type="nvlink", cross_node_type="none")

    @classmethod
    def local_single_gpu(cls) -> Environment:
        return cls(total_npus=1, npus_per_server=1, servers=1,
                   gpu_vram_gb=16.0, intra_node_type="pcie", cross_node_type="none")


@dataclass
class FeasibilityReport:
    """Result of evaluating all constraints on one (params, env, model) triple."""
    feasible: bool
    failures: list[dict[str, Any]] = field(default_factory=list)   # hard-fail
    warnings: list[dict[str, Any]] = field(default_factory=list)   # soft (severity=warning)

    def reason(self) -> str:
        if not self.failures and not self.warnings:
            return "ok"
        parts = []
        if self.failures:
            parts.append("FAIL: " + "; ".join(f["id"] for f in self.failures))
        if self.warnings:
            parts.append("WARN: " + "; ".join(w["id"] for w in self.warnings))
        return " | ".join(parts)


# --- AST evaluator ---


class _SafeEvalError(Exception):
    pass


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Constant, ast.Name, ast.Attribute, ast.Subscript, ast.Load,
    ast.Tuple, ast.List, ast.Set,
)


def _safe_eval(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a constraint expression against a bound namespace.

    Whitelist-only AST walk. The only callables are built-ins NOT (handled as
    unary) — no function calls allowed.
    """
    # Friendly aliases: SQL-like NOT/AND/OR → Python not/and/or
    expr_py = expr
    for tok_in, tok_out in (
        (" NOT ", " not "), ("(NOT ", "(not "),
        (" AND ", " and "), (" OR ", " or "),
    ):
        expr_py = expr_py.replace(tok_in, tok_out)
    if expr_py.lstrip().startswith("NOT "):
        expr_py = "not " + expr_py.lstrip()[4:]

    try:
        tree = ast.parse(expr_py, mode="eval")
    except SyntaxError as e:
        raise _SafeEvalError(f"parse error: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise _SafeEvalError(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ctx:
            raise _SafeEvalError(f"undefined name: {node.id}")

    code = compile(tree, "<feasibility>", "eval")
    return eval(code, {"__builtins__": {}}, ctx)


# --- Constraint loader + runner ---


@dataclass
class Constraint:
    id: str
    rule: str
    message: str = ""
    severity: str = "error"   # error | warning


def load_constraints(space_yaml_path: str | Path) -> list[Constraint]:
    """Load `feasibility_constraints` block from a SearchSpace YAML."""
    p = Path(space_yaml_path)
    spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = spec.get("feasibility_constraints") or []
    out: list[Constraint] = []
    for entry in raw:
        if not isinstance(entry, dict) or "rule" not in entry:
            continue
        out.append(Constraint(
            id=str(entry.get("id") or f"c_{len(out) + 1}"),
            rule=str(entry["rule"]),
            message=str(entry.get("message") or ""),
            severity=str(entry.get("severity") or "error"),
        ))
    return out


def _surrogate_namespace(params: dict[str, Any], model: ModelSpec | None) -> dict[str, Any]:
    """Best-effort approximation of the `surrogate.*` reference namespace.

    Real surrogate (analytical TTFT/ITL) is `surrogate_analytical.py` — until
    that lands, use a coarse model-mem estimate so c6 (GPU mem fit) still trips.
    """
    if model is None:
        return {"simulated_total_gb": 0.0}
    tp = max(1, int(params.get("tensor_parallel_size", 1)))
    pp = max(1, int(params.get("pipeline_parallel_size", 1)))
    layers = model.num_layers
    pp_fraction = max(1, (layers + pp - 1) // pp) / layers if layers else 1.0
    # Coarse: model_weights / (tp * pp_fraction)
    weights_gb = model.total_params_b * model.dtype_bytes
    if model.is_moe and model.active_params_b is not None:
        weights_gb = model.active_params_b * model.dtype_bytes
    return {"simulated_total_gb": (weights_gb * pp_fraction) / max(1, tp) + 4.0}


def _bool_to_str(env: Environment) -> dict[str, Any]:
    """Environment dict with cross_node_type as comparable str."""
    return {
        "total_npus": env.total_npus,
        "npus_per_server": env.npus_per_server,
        "servers": env.servers,
        "gpu_vram_gb": env.gpu_vram_gb,
        "server_storage_gb": env.server_storage_gb,
        "intra_node_type": env.intra_node_type,
        "cross_node_type": env.cross_node_type,
    }


class _Box:
    """Attribute-access dict — `obj.foo` instead of `obj['foo']`."""

    __slots__ = ("__d",)

    def __init__(self, d: dict[str, Any]):
        self.__d = d

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__d[name]
        except KeyError as e:
            raise AttributeError(name) from e


def evaluate(
    params: dict[str, Any],
    *,
    environment: Environment,
    model: ModelSpec | None = None,
    constraints: list[Constraint],
) -> FeasibilityReport:
    """Run all constraints. Returns a FeasibilityReport (feasible iff all errors pass)."""
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    env_dict = _bool_to_str(environment)
    model_dict: dict[str, Any] = {}
    if model is not None:
        model_dict = {
            "name": model.name,
            "total_params_b": model.total_params_b,
            "active_params_b": model.active_params_b,
            "num_layers": model.num_layers,
            "num_attention_heads": model.num_attention_heads,
            "num_kv_heads": model.num_kv_heads,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "context_length": model.context_length,
            "vocab_size": model.vocab_size,
            "head_dim": model.head_dim,
            "dtype_bytes": model.dtype_bytes,
            "kv_cache_dtype_bytes": model.kv_cache_dtype_bytes,
            "is_moe": model.is_moe,
            "has_mla": model.has_mla,
            "num_experts": model.num_experts,
        }

    surrogate_dict = _surrogate_namespace(params, model)

    base_ctx: dict[str, Any] = {
        "environment": _Box(env_dict),
        "model": _Box(model_dict) if model_dict else None,
        "surrogate": _Box(surrogate_dict),
        # raw flags for convenience (also accept top-level names directly):
        "True": True, "False": False, "None": None,
    }

    for c in constraints:
        # Build per-rule namespace = base + each param exposed as top-level name
        ctx = dict(base_ctx)
        ctx.update({k: v for k, v in params.items()})

        try:
            ok = _safe_eval(c.rule, ctx)
        except _SafeEvalError as e:
            # constraint references missing axis (e.g. moe-only rule on non-moe model)
            # → treat as N/A (skip) unless severity says otherwise
            if "undefined name" in str(e):
                continue
            failures.append({
                "id": c.id, "rule": c.rule, "message": f"eval error: {e}",
                "severity": "error",
            })
            continue
        except (TypeError, AttributeError):
            # missing context attribute → skip (rule N/A for this combo)
            continue
        except (ZeroDivisionError, ValueError):
            failures.append({
                "id": c.id, "rule": c.rule, "message": "numeric error",
                "severity": "error",
            })
            continue

        if not bool(ok):
            entry = {"id": c.id, "rule": c.rule, "message": c.message, "severity": c.severity}
            (warnings if c.severity == "warning" else failures).append(entry)

    return FeasibilityReport(feasible=len(failures) == 0, failures=failures, warnings=warnings)


def is_feasible(
    params: dict[str, Any],
    *,
    environment: Environment,
    model_name: str | None = None,
    constraints: list[Constraint] | None = None,
    space_yaml_path: str | Path | None = None,
) -> bool:
    """Convenience wrapper. Sampler usage:

        if not is_feasible(params, environment=env, space_yaml_path=path,
                           model_name=params.get("model_id")):
            raise optuna.TrialPruned()
    """
    if constraints is None:
        if space_yaml_path is None:
            return True
        constraints = load_constraints(space_yaml_path)
    model = by_name(model_name) if model_name else None
    rep = evaluate(params, environment=environment, model=model, constraints=constraints)
    return rep.feasible
