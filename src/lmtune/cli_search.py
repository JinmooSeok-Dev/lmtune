"""`bench search ...` subcommands — Phase S1 inline executor.

    bench search start  --space <yaml> --strategy <grid|random|lhc> ...
    bench search status <study_id>
    bench search resume <study_id>  (S1: logs warning; resume needs backend=k8s, S3)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from lmtune.search import (
    CallableObjective,
    ScoreObjective,
    Study,
    StudyConfig,
    load_space,
)
from lmtune.search.warmstart import warmstart_from_archive
from lmtune.storage import DuckDBStore

app = typer.Typer(no_args_is_help=True, help="탐색(search) 실행·조회 커맨드")
console = Console()


def _default_db_path() -> Path:
    return Path(os.environ.get("LMTUNE_DB", "data/db/lmtune.duckdb"))


@app.command("start")
def cmd_start(
    space: Annotated[Path, typer.Option(..., "--space", exists=True, readable=True, help="SearchSpace YAML")],
    strategy: Annotated[str, typer.Option("--strategy", help="grid | random | lhc")] = "random",
    endpoint: Annotated[
        Path | None,
        typer.Option("--endpoint", "-e", exists=True, readable=True, help="Endpoint YAML (ScoreObjective 용)"),
    ] = None,
    profile: Annotated[
        list[Path] | None,
        typer.Option("--profile", "-p", exists=True, readable=True, help="Workload profile YAML (여러개 가능)"),
    ] = None,
    max_trials: Annotated[int, typer.Option("--max-trials", help="최대 trial 수")] = 20,
    name: Annotated[str | None, typer.Option("--name")] = None,
    direction: Annotated[str, typer.Option("--direction")] = "maximize",
    metric_name: Annotated[str, typer.Option("--metric-name")] = "total_score",
    seed: Annotated[int | None, typer.Option("--seed")] = 42,
    n_samples: Annotated[int | None, typer.Option("--n-samples", help="lhc 전용")] = None,
    warmstart_db: Annotated[
        Path | None,
        typer.Option("--warmstart-db", exists=True, readable=True, help="과거 DuckDB (archive) 에서 seed 추출"),
    ] = None,
    warmstart_top_k: Annotated[int, typer.Option("--warmstart-top-k")] = 5,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="CallableObjective(constant=0) 로 구조만 검증. vLLM 없어도 동작"),
    ] = False,
    adapter: Annotated[
        str,
        typer.Option("--adapter", help="none | local-vllm | llmd-k8s. none 이면 params 를 endpoint 에 적용 안 함 (S1 호환)"),
    ] = "none",
    backend: Annotated[
        str,
        typer.Option("--backend", help="inline | process-pool (S3 dev, workers=1 권장) | k8s-job (S4)"),
    ] = "inline",
    workers: Annotated[
        int,
        typer.Option("--workers", help="동시 실행 수. process-pool + 단일 GPU/DuckDB 에서는 1 만 안전. "
                                        "실제 병렬은 k8s-job backend(S4) 에서."),
    ] = 1,
    budget_hours: Annotated[
        float | None,
        typer.Option("--budget-hours", help="전체 실험 예산. 경과 시 실행 중 trial 만 마무리"),
    ] = None,
    objectives: Annotated[
        str | None,
        typer.Option(
            "--objectives",
            help="Multi-objective: 쉼표 구분 'metric:workload:direction' 리스트. "
                 "예: 'throughput_tok_avg:short:maximize,ttft_p99:short:minimize'. "
                 "지정 시 NSGA-II/III 등 multi-obj 샘플러 권장.",
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="trial 당 lmtune run 반복 횟수 (variance gate)"),
    ] = 3,
    ttft_slo_ms: Annotated[
        float,
        typer.Option(
            "--ttft-slo-ms",
            help="composite score 의 TTFT penalty denom (=2× 본 값). 클수록 SLO 완화.",
        ),
    ] = 500.0,
):
    sp = load_space(space)
    profile = profile or []
    profile_slugs = [p.stem for p in profile]

    # Parse multi-objective spec if provided.
    obj_keys: list[Any] = []
    directions_list: list[str] | None = None
    if objectives:
        from lmtune.search.objective_pareto import ObjectiveKey
        for spec in objectives.split(","):
            parts = [p.strip() for p in spec.split(":")]
            if len(parts) != 3:
                raise typer.BadParameter(
                    f"bad --objectives entry '{spec}' (expected 'metric:workload:direction')"
                )
            metric, workload, d = parts
            if d not in ("maximize", "minimize"):
                raise typer.BadParameter(f"direction must be maximize|minimize, got '{d}'")
            obj_keys.append(ObjectiveKey(metric, workload or None, d))
        directions_list = [k.direction for k in obj_keys]
        if len(directions_list) < 2:
            raise typer.BadParameter("--objectives needs ≥2 entries for multi-objective")

    # Build adapter early so SearchSpace active_if can be gated by adapter_label.
    adapter_obj = None
    if adapter == "local-vllm":
        from lmtune.deploy import LocalVLLMAdapter
        adapter_obj = LocalVLLMAdapter()
    elif adapter == "llmd-k8s":
        from lmtune.deploy import LLMDK8sAdapter
        # endpoint YAML 의 deployment.helmfile_overrides 블록을 읽어 adapter 구성.
        # bare ctor 는 (selector=name=ms-phase1, env=dev) 같은 peer-repo 디폴트라
        # b200/helmfile/inference-scheduling/helmfile-mini.yaml.gotmpl 등 다른
        # helmfile 을 가리키는 endpoint 에선 release-not-found 로 실패.
        if endpoint is not None:
            ep_data = yaml.safe_load(endpoint.read_text(encoding="utf-8"))
            adapter_obj = LLMDK8sAdapter.from_endpoint(ep_data, dry_run=dry_run)
        else:
            adapter_obj = LLMDK8sAdapter()
    elif adapter != "none":
        raise typer.BadParameter(f"unknown --adapter: {adapter}")

    space_context = adapter_obj.context() if adapter_obj is not None else None

    cfg = StudyConfig(
        name=name or sp.name,
        strategy=strategy,
        space=sp,
        metric_name=metric_name,
        direction=direction,
        directions=directions_list,
        endpoint_slug=None,
        profile_slugs=profile_slugs,
        seed=seed,
        n_samples=n_samples,
        context=space_context,
    )

    # Endpoint slug 추출 (ScoreObjective 에 쓸 때만 필요)
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
        base_objective = ScoreObjective(
            endpoint_path=endpoint,
            profile_paths=[Path(p) for p in profile],
            adapter=adapter_obj,
            repeats=repeats,
            ttft_slo_ms=ttft_slo_ms,
        )
        if obj_keys:
            from lmtune.search.objective_pareto import ParetoObjective
            objective = ParetoObjective(base_objective, obj_keys)
            obj_summary = ", ".join(
                f"{k.metric}|{k.workload or 'agg'}:{k.direction}" for k in obj_keys
            )
            console.print(
                f"[green]multi-objective[/]: {len(obj_keys)} objectives ({obj_summary})"
            )
        else:
            objective = base_objective
        if adapter_obj is not None:
            console.print(f"[green]adapter[/]: {adapter_obj.adapter_label} (params will be applied to endpoint each trial)")

    if backend == "inline" or dry_run:
        trials = study.run(objective, max_trials=max_trials)
    elif backend == "process-pool":
        if dry_run:
            raise typer.BadParameter("process-pool backend 은 --dry-run 과 호환 안 됨")
        if not endpoint or not profile:
            raise typer.BadParameter("process-pool 백엔드는 endpoint 와 profile 필수")
        from lmtune.orchestrate.backend_process_pool import ProcessPoolBackend
        from lmtune.orchestrate.driver import run_distributed
        pool = ProcessPoolBackend(workers=workers)
        trials = run_distributed(
            study, pool,
            endpoint_path=endpoint,
            profile_paths=[Path(p) for p in profile],
            max_trials=max_trials,
            repeats=3,
            budget_seconds=(budget_hours * 3600.0) if budget_hours else None,
        )
    elif backend == "k8s-job":
        raise typer.BadParameter("k8s-job backend 은 Phase S4 에서 활성화됩니다")
    else:
        raise typer.BadParameter(f"unknown backend: {backend}")

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
    table.add_column("trial_id", no_wrap=True, overflow="fold")
    table.add_column("score", justify="right")
    table.add_column("params", overflow="fold")
    for trial_id, seq, params_json, score, _st in rows:
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

    from lmtune.search.analysis import anova_per_axis, axis_importance, tighten_bounds
    from lmtune.search.space import parse_space

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
    for _trial_id, seq, params_json, status, score, _completed, _backend, _err in raw_trials:
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

    from lmtune.search.space import Axis, SearchSpace

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


@app.command("pareto")
def cmd_pareto(
    study_id: Annotated[str, typer.Argument()],
    out: Annotated[Path, typer.Option("--out", help="PNG output path")] = Path("pareto.png"),
):
    """Non-dominated front over (obj1, obj2) stored in trial_metrics.

    Looks up trial_metrics entries with workload='short' (or 'aggregate') for
    'throughput_tok_avg' (maximize) and 'ttft_p99' (minimize) — the canonical
    goodput Pareto. Emits the front as JSON + saves a plot.
    """
    from lmtune.visualization.plots.pareto import non_dominated, plot_pareto

    store = DuckDBStore(_default_db_path())
    rows = store.list_trials(study_id)
    points: list[list[float]] = []
    labels: list[str] = []
    directions = ["maximize", "minimize"]
    for trial_id, seq, _params_json, status, _score, _c, _b, _e in rows:
        if status != "completed":
            continue
        tm = store.get_trial_metrics(trial_id)
        thr = (tm.get("throughput_tok_avg") or {}).get("short")
        if thr is None:
            thr = (tm.get("throughput_tok_avg") or {}).get("aggregate")
        ttft = (tm.get("ttft_p99") or {}).get("short")
        if ttft is None:
            ttft = (tm.get("ttft_p99") or {}).get("aggregate")
        if thr is None or ttft is None:
            continue
        points.append([float(thr), float(ttft)])
        labels.append(f"#{seq}")

    if not points:
        console.print("[yellow]no trial_metrics with both throughput and ttft[/]")
        return

    nd = non_dominated(points, directions)
    front = [{"seq": labels[i], "throughput_tok_avg": points[i][0], "ttft_p99": points[i][1]} for i in nd]
    console.print_json(data={"n_trials": len(points), "pareto_size": len(nd), "front": front})
    plot_pareto(points, directions, labels=labels, out_path=out)
    console.print(f"[green]saved[/]: {out}")


@app.command("sensitivity")
def cmd_sensitivity(
    study_id: Annotated[str, typer.Argument()],
    out: Annotated[Path, typer.Option("--out")] = Path("sobol.png"),
    n_saltelli: Annotated[int, typer.Option("--n-saltelli")] = 1024,
):
    """Global Sobol sensitivity over continuous axes (post-hoc via RF surrogate)."""
    import json as _json

    from lmtune.search.analysis.sobol import sobol_from_history
    from lmtune.search.space import parse_space
    from lmtune.visualization.plots.sobol_bar import plot_sobol

    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        raise typer.BadParameter(f"study not found: {study_id}")
    sp = parse_space(yaml.safe_load(hdr[3])) if hdr[3] else None
    if sp is None:
        raise typer.BadParameter("study has no space_yaml")

    raw = store.list_trials(study_id)
    trials: list[dict] = []
    for _trial_id, _seq, params_json, status, score, _c, _b, _e in raw:
        try:
            p = _json.loads(params_json or "{}")
        except _json.JSONDecodeError:
            p = {}
        trials.append({"params": p, "score": score, "status": status})

    axes_spec = [{"name": a.name, "kind": a.kind, "low": a.low, "high": a.high} for a in sp.axes]
    results = sobol_from_history(trials, axes_spec, n_saltelli=n_saltelli)
    if not results:
        console.print("[yellow]not enough continuous axes / completed trials for Sobol[/]")
        return

    report = [{
        "axis": r.axis, "S1": r.S1, "ST": r.ST,
        "S1_conf": r.S1_conf, "ST_conf": r.ST_conf,
        "interaction_gap": r.interaction_gap,
    } for r in results]
    console.print_json(data={"study_id": study_id, "n_axes": len(results), "sobol": report})
    plot_sobol(results, out_path=out)
    console.print(f"[green]saved[/]: {out}")


@app.command("trace")
def cmd_trace(
    study_id: Annotated[str, typer.Argument()],
    out: Annotated[Path, typer.Option("--out")] = Path("search_trace.png"),
):
    """Running best score over trial sequence — visual 'is the sampler converging?'"""
    from lmtune.visualization.plots.search_trace import plot_search_trace

    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        raise typer.BadParameter(f"study not found: {study_id}")
    direction = hdr[7]
    rows = store.list_trials(study_id)
    seqs = [r[1] for r in rows]
    scores = [r[4] for r in rows]
    plot_search_trace(seqs, scores, direction=direction, out_path=out)
    console.print(f"[green]saved[/]: {out}")


@app.command("ask")
def cmd_ask(
    study_id: Annotated[str, typer.Argument(help="study_id (bench search start 가 출력)")],
    out_json: Annotated[
        Path | None,
        typer.Option("--out", help="JSON 출력 파일 (기본: stdout)"),
    ] = None,
):
    """Phase S6 — 외부 LLM 에이전트(autoresearch) 가 호출.

    study_id 의 spec + 과거 trial 이력으로 sampler 를 재구성하고 다음 trial params 를 추천.
    출력 JSON: `{"study_id", "trial_id", "seq", "params"}`.

    autoresearch.sh 가 이 JSON 의 params 를 endpoint YAML 에 적용한 뒤 측정 → `bench search tell` 호출.
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
    # 새 Study 객체 — sampler 새로 만듦. 외부 study_id 를 강제 사용하기 위해 override.
    study = Study(cfg, store)
    study.study_id = study_id  # 외부 study_id 로 trial 을 INSERT

    # 과거 completed trial 을 sampler 에 주입 (warmstart)
    past = store.top_trials(study_id, direction=direction, k=200)
    seeds: list[tuple[dict, float]] = []
    for _tid, _seq, params_json, score, _st in past:
        if score is None:
            continue
        try:
            seeds.append((json.loads(params_json), float(score)))
        except Exception:  # noqa: BLE001
            continue
    if seeds:
        study.enqueue_warmstart(seeds)

    # 다음 trial 추출 (DuckDB trials 테이블에 status=pending 으로 INSERT 됨)
    trial = study.ask()

    payload = {
        "study_id": study_id,
        "trial_id": trial.trial_id,
        "seq": trial.seq,
        "params": trial.params,
    }
    out_text = json.dumps(payload, sort_keys=False, indent=2)
    if out_json:
        out_json.write_text(out_text)
    else:
        # stdout 에 JSON 한 줄 (autoresearch.sh 가 tail | jq)
        print(out_text)


@app.command("tell")
def cmd_tell(
    study_id: Annotated[str, typer.Argument(help="study_id")],
    trial_id: Annotated[str, typer.Option("--trial", help="ask 가 발급한 trial_id")],
    metrics_json: Annotated[
        Path | None,
        typer.Option("--metrics-json", exists=True, readable=True, help="JSON 파일 (없으면 stdin)"),
    ] = None,
    score: Annotated[float | None, typer.Option("--score", help="명시적 score 값 — metrics 의 'total_score' 키 우선")] = None,
):
    """Phase S6 — 외부 LLM 에이전트가 측정 결과를 study 에 기록.

    metrics-json 형식 (autoresearch.sh 의 METRIC 라인을 변환한 결과):
    ```json
    {
      "total_score": 1906.4,
      "metrics": {
        "throughput_avg_short": 130.4,
        "ttft_p99_short": 192.5,
        "e2e_p99_short": 760.0,
        "slo_pass_all": 1
      },
      "accepted": true
    }
    ```
    """
    store = DuckDBStore(_default_db_path())
    hdr = store.get_study(study_id)
    if not hdr:
        console.print(f"[red]study not found[/]: {study_id}")
        raise typer.Exit(1)

    if metrics_json:
        data = json.loads(metrics_json.read_text())
    else:
        import sys
        data = json.loads(sys.stdin.read())

    final_score = data.get("total_score") or score or 0.0
    metrics_in: dict = data.get("metrics", {})
    accepted = bool(data.get("accepted", True))

    # trials 테이블 직접 업데이트 (study.tell 은 Optuna trial 객체가 필요한데 ask 와 다른 process 라 부재)
    metrics_for_db: dict[tuple[str, str | None], float] = {}
    for k, v in metrics_in.items():
        try:
            metrics_for_db[(k, None)] = float(v)
        except (TypeError, ValueError):
            continue

    # 기존 row 의 seq 와 params 를 살려서 update
    existing = store.conn.execute(
        "SELECT seq, params, backend FROM trials WHERE trial_id = ?",
        [trial_id],
    ).fetchone()
    if not existing:
        console.print(f"[red]trial not found[/]: {trial_id}")
        raise typer.Exit(1)
    seq, params_json, backend = existing
    params = json.loads(params_json)

    status = "completed" if accepted else "pruned"
    store.record_trial(
        trial_id=trial_id, study_id=study_id, seq=seq, params=params,
        status=status, score=final_score, backend=backend or "external",
        completed=True,
    )
    if metrics_for_db:
        store.record_trial_metrics(trial_id, metrics_for_db)

    console.print(
        f"[green]recorded[/]: trial={trial_id} status={status} score={final_score:.2f}"
    )


@app.command("export")
def cmd_export(
    study_id: Annotated[str, typer.Argument(help="study_id (bench search start 가 출력)")],
    out_dir: Annotated[Path, typer.Option("--out", help="결과 디렉토리 (winner/ 가 그 아래 생성)")],
    winner: Annotated[str, typer.Option(
        "--winner",
        help="top-N 또는 top-1 (기본). 향후 'pareto' 추가 가능.",
    )] = "top-1",
    endpoint: Annotated[Path | None, typer.Option(
        "--endpoint", "-e", exists=True, readable=True,
        help="endpoint YAML — adapter 추정 + helmfile_overrides 추출",
    )] = None,
):
    """Export winner trial to a self-contained recipe directory.

    산출:
      <out>/winner/apply.sh             — dry-run | apply 한 줄 실행
      <out>/winner/values-overlay.yaml  — helmfile state-values-file
      <out>/winner/params.json          — raw params dict
      <out>/winner/README.md            — 사람이 읽을 수 있는 recipe + 적용 절차
    """
    from lmtune.search.export_winner import export_winner

    if not winner.startswith("top-"):
        raise typer.BadParameter("--winner must be 'top-N' (e.g., top-1, top-3)")
    try:
        rank = int(winner.split("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise typer.BadParameter(f"invalid --winner: {winner}") from exc

    try:
        result = export_winner(
            study_id,
            db_path=_default_db_path(),
            out_dir=out_dir,
            rank=rank,
            endpoint_yaml_path=endpoint,
        )
    except ValueError as e:
        console.print(f"[red]export failed[/]: {e}")
        raise typer.Exit(1) from e

    console.print(f"[green]exported[/]: {result.out_dir}")
    console.print(f"  trial_id={result.trial_id}  score={result.score}  adapter={result.adapter}")
    for f in result.files:
        console.print(f"  - {f.relative_to(result.out_dir.parent)}")


@app.command("ls")
def cmd_ls(
    limit: Annotated[int, typer.Option("--limit")] = 20,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="copy-paste 친화 plain text (rich table 비활성)"),
    ] = False,
    ids_only: Annotated[
        bool,
        typer.Option("--ids", help="study_id 만 줄별 출력 (스크립트용)"),
    ] = False,
):
    store = DuckDBStore(_default_db_path())
    rows = store.list_studies(limit=limit)
    if not rows:
        console.print("[yellow]no studies[/]")
        return
    if ids_only:
        for r in rows:
            print(r[0])
        return
    if plain:
        # tab-separated, no truncation — copy-paste 친화
        cols = ["study_id", "name", "strategy", "endpoint", "status", "created_at", "finished_at"]
        print("\t".join(cols))
        for r in rows:
            print("\t".join(str(x) if x is not None else "-" for x in r))
        return
    table = Table(title="studies")
    # study_id 는 truncate 금지 — 사용자가 copy 할 수 있어야 함
    table.add_column("study_id", no_wrap=True, overflow="fold")
    for c in ["name", "strategy", "endpoint", "status", "created_at", "finished_at"]:
        table.add_column(c, overflow="fold")
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
    from lmtune.search.space import parse_space
    return parse_space(yaml.safe_load(text))
