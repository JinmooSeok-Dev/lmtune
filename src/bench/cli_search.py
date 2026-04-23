"""`bench search ...` subcommands — Phase S1 inline executor.

    bench search start  --space <yaml> --strategy <grid|random|lhc> ...
    bench search status <study_id>
    bench search resume <study_id>  (S1: logs warning; resume needs backend=k8s, S3)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from bench.search import (
    BenchScoreObjective,
    CallableObjective,
    Study,
    StudyConfig,
    load_space,
)
from bench.search.warmstart import warmstart_from_archive
from bench.storage import DuckDBStore


app = typer.Typer(no_args_is_help=True, help="탐색(search) 실행·조회 커맨드")
console = Console()


def _default_db_path() -> Path:
    return Path(os.environ.get("BENCH_DB", "data/db/bench.duckdb"))


@app.command("start")
def cmd_start(
    space: Annotated[Path, typer.Option(..., "--space", exists=True, readable=True, help="SearchSpace YAML")],
    strategy: Annotated[str, typer.Option("--strategy", help="grid | random | lhc")] = "random",
    endpoint: Annotated[
        Optional[Path],
        typer.Option("--endpoint", "-e", exists=True, readable=True, help="Endpoint YAML (BenchScoreObjective 용)"),
    ] = None,
    profile: Annotated[
        list[Path],
        typer.Option("--profile", "-p", exists=True, readable=True, help="Workload profile YAML (여러개 가능)"),
    ] = [],
    max_trials: Annotated[int, typer.Option("--max-trials", help="최대 trial 수")] = 20,
    name: Annotated[Optional[str], typer.Option("--name")] = None,
    direction: Annotated[str, typer.Option("--direction")] = "maximize",
    metric_name: Annotated[str, typer.Option("--metric-name")] = "total_score",
    seed: Annotated[Optional[int], typer.Option("--seed")] = 42,
    n_samples: Annotated[Optional[int], typer.Option("--n-samples", help="lhc 전용")] = None,
    warmstart_db: Annotated[
        Optional[Path],
        typer.Option("--warmstart-db", exists=True, readable=True, help="과거 DuckDB (archive) 에서 seed 추출"),
    ] = None,
    warmstart_top_k: Annotated[int, typer.Option("--warmstart-top-k")] = 5,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="CallableObjective(constant=0) 로 구조만 검증. vLLM 없어도 동작"),
    ] = False,
):
    sp = load_space(space)
    profile_slugs = [p.stem for p in profile]

    cfg = StudyConfig(
        name=name or sp.name,
        strategy=strategy,
        space=sp,
        metric_name=metric_name,
        direction=direction,
        endpoint_slug=None,
        profile_slugs=profile_slugs,
        seed=seed,
        n_samples=n_samples,
    )

    # Endpoint slug 추출 (BenchScoreObjective 에 쓸 때만 필요)
    if endpoint and not dry_run:
        ep_data = yaml.safe_load(endpoint.read_text(encoding="utf-8"))
        cfg.endpoint_slug = ep_data.get("slug")

    db_path = _default_db_path()
    store = DuckDBStore(db_path)
    study = Study(cfg, store)

    console.print(f"[bold cyan]study_id[/]: {study.study_id}  strategy={strategy}  axes={len(sp.active_axes())}")

    # Warm-start
    if warmstart_db:
        seeds = warmstart_from_archive(
            warmstart_db, sp,
            endpoint_slug=cfg.endpoint_slug,
            profile_slugs=profile_slugs or None,
            top_k=warmstart_top_k,
        )
        if seeds:
            console.print(f"[green]warmstart[/]: seeded {len(seeds)} trial(s) from archive")
            study.enqueue_warmstart(seeds)
        else:
            console.print("[yellow]warmstart[/]: no compatible history found")

    # Objective selection
    if dry_run:
        objective = CallableObjective(lambda params: 0.0)
        console.print("[yellow]dry-run[/]: objective=CallableObjective(constant 0)")
    else:
        if not endpoint or not profile:
            raise typer.BadParameter("endpoint 와 profile 은 --dry-run 이 아닌 경우 필수")
        objective = BenchScoreObjective(
            endpoint_path=endpoint,
            profile_paths=[Path(p) for p in profile],
        )

    trials = study.run(objective, max_trials=max_trials)
    console.print(f"[bold]완료[/]: {len(trials)} trials  study_id={study.study_id}")
    _print_top(store, study.study_id, direction, k=5)


@app.command("status")
def cmd_status(
    study_id: Annotated[str, typer.Argument(help="study_id (bench search start 가 출력)")],
    top: Annotated[int, typer.Option("--top")] = 5,
):
    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        console.print(f"[red]study not found[/]: {study_id}")
        raise typer.Exit(1)

    # 인덱스: study_id, name, strategy, space_yaml, endpoint_slug, profile_slugs,
    #        metric_name, direction, status, created_at, finished_at, notes
    (_, name, strategy, _space_yaml, ep_slug, prof_slugs_json,
     metric_name, direction, status, created_at, finished_at, _notes) = hdr

    console.print(f"[bold]{name}[/]  strategy={strategy}  direction={direction}  status={status}")
    console.print(f"  study_id={study_id}  endpoint={ep_slug}  profiles={prof_slugs_json}")
    console.print(f"  created={created_at}  finished={finished_at}")

    trials = store.list_trials(study_id)
    counts: dict[str, int] = {}
    for t in trials:
        counts[t[3]] = counts.get(t[3], 0) + 1
    console.print(f"  trials: total={len(trials)}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    rows = store.top_trials(study_id, direction=direction, k=top)
    if not rows:
        console.print("[yellow]no completed trials yet[/]")
        return
    table = Table(title=f"Top-{top} trials ({metric_name})")
    table.add_column("seq", justify="right")
    table.add_column("trial_id")
    table.add_column("score", justify="right")
    table.add_column("params", overflow="fold")
    for trial_id, seq, params_json, score, st in rows:
        table.add_row(str(seq), trial_id, f"{score:.2f}" if score is not None else "-", params_json)
    console.print(table)


@app.command("resume")
def cmd_resume(
    study_id: Annotated[str, typer.Argument()],
    max_trials: Annotated[int, typer.Option("--max-trials")] = 10,
):
    """S1: inline Study 는 세션 간 sampler 상태를 유지하지 않으므로 resume 은 제한적.

    추가 trial 을 같은 study_id 에 append 하는 형태로 동작.
    완전한 resume 은 S3 에서 backend=k8s-job 과 함께 도입.
    """
    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        console.print(f"[red]study not found[/]: {study_id}")
        raise typer.Exit(1)
    space_yaml = hdr[3]
    strategy = hdr[2]
    direction = hdr[7]
    metric_name = hdr[6]
    ep_slug = hdr[4]
    prof_slugs_json = hdr[5]

    sp = load_space_from_text(space_yaml)
    cfg = StudyConfig(
        name=hdr[1], strategy=strategy, space=sp,
        metric_name=metric_name, direction=direction,
        endpoint_slug=ep_slug,
        profile_slugs=json.loads(prof_slugs_json) if prof_slugs_json else [],
    )
    study = Study(cfg, store)  # new Optuna study + new study_id → 사용자 경고
    console.print(
        f"[yellow]resume 은 현재 inline 한정: 새 study_id={study.study_id} 로 이어집니다."
        "  (완전한 resume 은 S3 에서 backend=k8s 와 함께 지원)"
    )

    # 과거 trials 를 warmstart 로 복원
    past = store.top_trials(study_id, direction=direction, k=max(max_trials, 10))
    seeds: list[tuple[dict, float]] = []
    for _tid, _seq, params_json, score, _st in past:
        try:
            seeds.append((json.loads(params_json), float(score)))
        except Exception:  # noqa: BLE001
            continue
    if seeds:
        study.enqueue_warmstart(seeds)
        console.print(f"[green]seeded[/]: {len(seeds)} past trials")

    objective = CallableObjective(lambda _p: 0.0)
    console.print("[yellow]objective=dummy. 실제 재개는 start 에서 --warmstart-db 로 DB 지정하는 것을 권장[/]")
    study.run(objective, max_trials=max_trials)


@app.command("prune")
def cmd_prune(
    study_id: Annotated[str, typer.Argument()],
    p_freeze: Annotated[float, typer.Option("--p-freeze")] = 0.01,
    p_drop: Annotated[float, typer.Option("--p-drop")] = 0.05,
    imp_drop: Annotated[float, typer.Option("--imp-drop", help="importance threshold for drop")] = 0.05,
    top_frac: Annotated[float, typer.Option("--top-frac")] = 0.25,
    apply: Annotated[bool, typer.Option("--apply", help="write a narrowed SearchSpace YAML next to the original")] = False,
):
    """Run ANOVA + RF importance + bound-tighten on a study's completed trials."""
    import json as _json
    from bench.search.analysis import anova_per_axis, axis_importance, tighten_bounds
    from bench.search.space import parse_space

    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        console.print(f"[red]study not found[/]: {study_id}")
        raise typer.Exit(1)
    space_yaml = hdr[3]
    sp = parse_space(yaml.safe_load(space_yaml)) if space_yaml else None
    if sp is None:
        console.print("[red]study has no space_yaml; cannot prune[/]")
        raise typer.Exit(1)

    # Collect completed trials as {params, score, status}
    raw_trials = store.list_trials(study_id)
    trials: list[dict] = []
    for trial_id, seq, params_json, status, score, _completed, _backend, _err in raw_trials:
        try:
            p = _json.loads(params_json or "{}")
        except _json.JSONDecodeError:
            p = {}
        trials.append({"params": p, "score": score, "status": status, "seq": seq})

    # Analyses
    anova = anova_per_axis(trials, p_freeze=p_freeze, p_drop=p_drop)
    importance = axis_importance(trials, drop_threshold=imp_drop)
    axes_spec = [
        {"name": a.name, "kind": a.kind, "low": a.low, "high": a.high}
        for a in sp.axes
    ]
    shrink = tighten_bounds(trials, axes_spec, top_frac=top_frac)

    # Emit JSON report
    report = {
        "study_id": study_id,
        "n_trials": len(trials),
        "n_completed": sum(1 for t in trials if t["status"] == "completed"),
        "anova": [
            {
                "axis": a.axis,
                "p_value": a.p_value,
                "f_stat": a.f_stat,
                "recommendation": a.recommendation,
                "best_value": a.best_value,
            } for a in anova
        ],
        "importance": importance,
        "bound_tighten": shrink,
    }
    console.print_json(data=report)

    # Merge recommendations → apply
    if apply:
        narrowed = _apply_recommendations(sp, anova, importance, shrink)
        out = Path(f"{study_id}.narrowed.yaml")
        out.write_text(narrowed.to_yaml(), encoding="utf-8")
        console.print(f"[green]wrote[/]: {out}")


def _apply_recommendations(space, anova_list, importance, shrink) -> object:
    """Produce a narrowed SearchSpace copy applying freeze/drop/shrink."""
    from copy import deepcopy
    from bench.search.space import Axis, SearchSpace

    keep: list[Axis] = []
    imp_drop = {a for a, d in importance.items() if d.get("recommendation") == "drop"}
    anova_by_axis = {a.axis: a for a in anova_list}
    for axis in space.axes:
        a = anova_by_axis.get(axis.name)
        rec = a.recommendation if a else "keep"
        # drop if either ANOVA or importance says drop
        if rec == "drop" or axis.name in imp_drop:
            continue
        if rec == "freeze" and a is not None and a.best_value is not None:
            keep.append(Axis(name=axis.name, kind="categorical", values=[a.best_value]))
            continue
        shrunk = shrink.get(axis.name)
        if shrunk:
            a2 = deepcopy(axis)
            a2.low = float(shrunk["new_low"])
            a2.high = float(shrunk["new_high"])
            keep.append(a2)
            continue
        keep.append(deepcopy(axis))
    return SearchSpace(name=f"{space.name}-narrowed", axes=keep)


@app.command("ls")
def cmd_ls(limit: Annotated[int, typer.Option("--limit")] = 20):
    store = DuckDBStore(_default_db_path())
    rows = store.list_studies(limit=limit)
    if not rows:
        console.print("[yellow]no studies[/]")
        return
    table = Table(title="studies")
    for c in ["study_id", "name", "strategy", "endpoint", "status", "created_at", "finished_at"]:
        table.add_column(c)
    for r in rows:
        table.add_row(*[str(x) if x is not None else "-" for x in r])
    console.print(table)


# helpers --------------------------------------------------------------------

def _print_top(store: DuckDBStore, study_id: str, direction: str, k: int = 5):
    rows = store.top_trials(study_id, direction=direction, k=k)
    if not rows:
        console.print("[yellow]no completed trials[/]")
        return
    table = Table(title=f"Top-{k}")
    table.add_column("seq", justify="right")
    table.add_column("score", justify="right")
    table.add_column("params", overflow="fold")
    for _tid, seq, params_json, score, _st in rows:
        table.add_row(str(seq), f"{score:.2f}" if score is not None else "-", params_json)
    console.print(table)


def load_space_from_text(text: str):
    from bench.search.space import parse_space
    return parse_space(yaml.safe_load(text))
