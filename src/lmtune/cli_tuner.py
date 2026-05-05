"""``lmtune tuner`` 서브커맨드 — Sampler / Pruner 의 valid kind 노출 + introspect.

PLUG 패턴이 합류시키는 새 sampler / pruner 가 자동으로 본 명령에 노출되도록
모든 화이트리스트는 ``tuner.factory`` 의 set 들을 단일 진실원으로 사용.

서브커맨드:
  lmtune tuner list-samplers  — 등록된 sampler strategy 목록
  lmtune tuner list-pruners   — 등록된 pruner kind 목록 (Optuna + Native)
  lmtune tuner describe <kind> — 특정 kind 의 __init__ signature + docstring

``lmtune storage list-backends`` 와 동등한 PLUG 노출 패턴.
"""

from __future__ import annotations

import inspect
import json
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="lmtune tuner 도구 — sampler / pruner 메타")
console = Console()


# Optuna 가 실제로 Sampler 으로 받을 수 있는 strategy — search.samplers.make_sampler
# 가 분기하는 항목과 동기. 새 sampler 합류 시 본 set 도 갱신.
_OPTUNA_SAMPLER_STRATEGIES = (
    "random",
    "grid",
    "lhc",
    "tpe",
    "cma_es",
    "nsga2",
    "ucb",
    "botorch",
)


@app.command("list-samplers")
def cmd_list_samplers(
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력"),
    ] = False,
) -> None:
    """등록된 sampler strategy 목록.

    그룹:
    - native: Optuna 위임 없이 stdlib 으로 동작 (`*_native`)
    - optuna: Optuna BaseSampler 어댑터
    - llm: LLM-guided (anthropic SDK 필요)
    """
    from lmtune.tuner.factory import _LLM_STRATEGIES, _NATIVE_STRATEGIES

    groups = {
        "native": sorted(_NATIVE_STRATEGIES),
        "optuna": sorted(_OPTUNA_SAMPLER_STRATEGIES),
        "llm": sorted(_LLM_STRATEGIES),
    }

    if json_out:
        print(json.dumps(groups, separators=(",", ":")))
        return

    console.print("[bold]registered samplers[/bold]")
    for group, items in groups.items():
        console.print(f"  [cyan]{group}[/cyan]")
        for name in items:
            console.print(f"    - {name}")


@app.command("list-pruners")
def cmd_list_pruners(
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력"),
    ] = False,
) -> None:
    """등록된 pruner kind 목록.

    그룹:
    - native: Optuna 위임 없이 stdlib 으로 동작 (`*_native`)
    - optuna: Optuna BasePruner 어댑터 (search.pruners.make_pruner 위임)
    """
    from lmtune.tuner.factory import _NATIVE_PRUNER_KINDS, _OPTUNA_PRUNER_KINDS

    groups = {
        "native": sorted(_NATIVE_PRUNER_KINDS),
        "optuna": sorted(_OPTUNA_PRUNER_KINDS),
    }

    if json_out:
        print(json.dumps(groups, separators=(",", ":")))
        return

    console.print("[bold]registered pruners[/bold]")
    for group, items in groups.items():
        console.print(f"  [cyan]{group}[/cyan]")
        for name in items:
            console.print(f"    - {name}")
    console.print(
        "  [dim]+ 'none' / None — pruning disabled (default)[/dim]",
    )


# ─── describe ────────────────────────────────────────────────────────


def _resolve_kind(kind: str) -> tuple[str, str, type] | None:
    """``kind`` 이름 → ``(axis, group, cls)`` resolution.

    introspect 가능한 (lmtune-defined) 클래스만 매핑. Optuna 빌트인 sampler/
    pruner 는 ``None`` 반환 — describe 가 외부 reference 안내로 fallback.

    Returns:
        (axis, group, cls) — axis ∈ {'sampler', 'pruner'}, group ∈ {'native', 'llm'}.
        None 이면 introspect 불가.
    """
    k = kind.lower()
    # Pruner (native)
    if k == "median_native":
        from lmtune.tuner.median_pruner import NativeMedianPruner

        return ("pruner", "native", NativeMedianPruner)
    if k == "percentile_native":
        from lmtune.tuner.percentile_pruner import NativePercentilePruner

        return ("pruner", "native", NativePercentilePruner)
    # Sampler (native) — search.samplers.native 의 클래스들
    if k in ("random_native", "lhc_native", "tpe_native"):
        from lmtune.search.samplers.native import (
            NativeLHCSampler,
            NativeRandomSampler,
            NativeTPESampler,
        )

        cls = {
            "random_native": NativeRandomSampler,
            "lhc_native": NativeLHCSampler,
            "tpe_native": NativeTPESampler,
        }[k]
        return ("sampler", "native", cls)
    # Sampler (llm)
    if k == "llm_oracle":
        from lmtune.tuner.llm_oracle import LLMOracleSampler

        return ("sampler", "llm", LLMOracleSampler)
    return None


@app.command("describe")
def cmd_describe(
    kind: Annotated[
        str,
        typer.Argument(help="sampler / pruner kind (예: median_native, llm_oracle)"),
    ],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력"),
    ] = False,
) -> None:
    """특정 sampler / pruner ``kind`` 의 hyperparameter 표시.

    introspect 가능한 lmtune 정의 클래스 (native + llm) 만 처리. Optuna 빌트인
    (sh / hyperband / tpe / random / nsga2 등) 은 외부 reference 안내.
    """
    resolved = _resolve_kind(kind)
    if resolved is None:
        # Optuna 빌트인 등 introspect 불가 — kind 가 valid 한지만 확인
        from lmtune.cli_tuner import _OPTUNA_SAMPLER_STRATEGIES
        from lmtune.tuner.factory import _OPTUNA_PRUNER_KINDS

        if kind in _OPTUNA_PRUNER_KINDS:
            ref = "https://optuna.readthedocs.io/en/stable/reference/pruners.html"
            payload = {"kind": kind, "axis": "pruner", "group": "optuna", "reference": ref}
        elif kind in _OPTUNA_SAMPLER_STRATEGIES:
            ref = "https://optuna.readthedocs.io/en/stable/reference/samplers/index.html"
            payload = {"kind": kind, "axis": "sampler", "group": "optuna", "reference": ref}
        else:
            raise typer.BadParameter(
                f"unknown kind: {kind!r}. "
                "use 'lmtune tuner list-samplers' / 'list-pruners' to see valid kinds."
            )
        if json_out:
            print(json.dumps(payload, separators=(",", ":")))
            return
        console.print(f"[bold]{payload['kind']}[/bold]  ({payload['axis']}, {payload['group']})")
        console.print(f"  reference: {payload['reference']}")
        return

    axis, group, cls = resolved
    sig = inspect.signature(cls.__init__)
    # __init__ 의 self 제외 parameter 목록
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        kind_str = str(param.kind).split(".")[-1]
        annot = param.annotation if param.annotation is not inspect.Parameter.empty else None
        default = param.default if param.default is not inspect.Parameter.empty else None
        params.append(
            {
                "name": name,
                "kind": kind_str,
                "annotation": str(annot) if annot is not None else None,
                "default": repr(default) if default is not None else None,
            }
        )

    payload = {
        "kind": kind,
        "axis": axis,
        "group": group,
        "class_name": cls.__name__,
        "module": cls.__module__,
        "doc": (cls.__doc__ or "").strip(),
        "params": params,
    }

    if json_out:
        print(json.dumps(payload, separators=(",", ":")))
        return

    console.print(
        f"[bold]{kind}[/bold]  ({axis}, {group}) — [cyan]{cls.__module__}.{cls.__name__}[/cyan]"
    )
    if payload["doc"]:
        # 첫 줄만 표시 (full docstring 은 --json 에서)
        first_line = payload["doc"].splitlines()[0]
        console.print(f"  [dim]{first_line}[/dim]")
    console.print("  [bold]params[/bold]:")
    for p in params:
        annot = f": {p['annotation']}" if p["annotation"] else ""
        default = f" = {p['default']}" if p["default"] is not None else ""
        console.print(f"    - {p['name']}{annot}{default}")


# ─── make-config ──────────────────────────────────────────────────────


@app.command("make-config")
def cmd_make_config(
    kind: Annotated[
        str,
        typer.Argument(help="sampler / pruner kind (예: median_native, llm_oracle)"),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="출력 형식: yaml | json"),
    ] = "yaml",
    flat: Annotated[
        bool,
        typer.Option(
            "--flat",
            help="블록 wrapper 없이 kwargs 만 출력 (paste 시 직접 사용)",
        ),
    ] = False,
) -> None:
    """``kind`` 의 default kwargs 가 채워진 SearchSpace YAML/JSON 블록 출력.

    SearchSpace YAML 의 ``pruner:`` / ``sampler:`` 슬롯에 직접 paste 가능.
    ``--flat`` 으로 wrapper 제거 시 kwargs dict 만 출력 — 다른 도구가 dict
    deserialize 시 편함.

    Optuna 빌트인 (SH / Hyperband 등) 은 introspect 불가 — describe 와 동일
    fallback 정책 (kind + 안내문만 출력).
    """
    resolved = _resolve_kind(kind)

    if resolved is None:
        # Optuna 빌트인 — kwargs 정확한 default 모름. kind 만 출력.
        from lmtune.cli_tuner import _OPTUNA_SAMPLER_STRATEGIES
        from lmtune.tuner.factory import _OPTUNA_PRUNER_KINDS

        if kind not in _OPTUNA_PRUNER_KINDS and kind not in _OPTUNA_SAMPLER_STRATEGIES:
            raise typer.BadParameter(
                f"unknown kind: {kind!r}. "
                "use 'lmtune tuner list-samplers' / 'list-pruners' to see valid kinds."
            )
        axis = "pruner" if kind in _OPTUNA_PRUNER_KINDS else "sampler"
        kwargs: dict[str, object] = {}
        comment = (
            f"# {kind} ({axis}, optuna) — Optuna 빌트인. "
            "see https://optuna.readthedocs.io for hyperparameters."
        )
    else:
        axis, _group, cls = resolved
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                # 필수 인자 — null 로 표시 (사용자가 채워야 함)
                kwargs[name] = None
            else:
                kwargs[name] = param.default
        comment = ""

    block = {"kind": kind, **kwargs}

    if fmt.lower() == "json":
        payload = block if flat else {axis: block}
        print(json.dumps(payload, separators=(",", ":")))
        return

    if fmt.lower() != "yaml":
        raise typer.BadParameter(f"unknown format: {fmt!r}. valid: yaml | json")

    # yaml 출력 — PyYAML 의존성 회피, 단순 dict 라 직접 렌더
    lines: list[str] = []
    if comment:
        lines.append(comment)
    indent = "" if flat else "  "
    if not flat:
        lines.append(f"{axis}:")
    for k, v in block.items():
        lines.append(f"{indent}{k}: {_yaml_scalar(v)}")
    print("\n".join(lines))


def _yaml_scalar(v: object) -> str:
    """단순 YAML scalar 렌더 — bool/int/float/str/None 만."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # quoting 단순화 — 알파넘 + 일부 기호만 unquoted, 나머지 single-quote
        if v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            return v
        return f"'{v}'"
    return str(v)
