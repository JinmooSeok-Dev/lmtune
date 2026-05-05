"""``lmtune tuner`` 서브커맨드 — Sampler / Pruner 의 valid kind 노출.

PLUG 패턴이 합류시키는 새 sampler / pruner 가 자동으로 본 명령에 노출되도록
모든 화이트리스트는 ``tuner.factory`` 의 set 들을 단일 진실원으로 사용.

서브커맨드:
  lmtune tuner list-samplers  — 등록된 sampler strategy 목록
  lmtune tuner list-pruners   — 등록된 pruner kind 목록 (Optuna + Native)

``lmtune storage list-backends`` 와 동등한 PLUG 노출 패턴.
"""

from __future__ import annotations

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
