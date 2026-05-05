from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from ulid import ULID

from lmtune.analysis import (
    build_nway_table,
    compare_runs,
    nway_to_markdown,
    variance_across_runs,
)
from lmtune.collectors import PrometheusCollector
from lmtune.detectors import run_all_rules
from lmtune.endpoints import load_endpoint
from lmtune.profiles import SLOSpec, discover_profiles, load_profile
from lmtune.runners import get_runner
from lmtune.runners.base import RequestRow
from lmtune.storage import DuckDBStore
from lmtune.visualization import render_run_report

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="LLM endpoint 벤치마크 자동화 CLI"
)
console = Console()

# Phase S1: search subcommand group
from lmtune.cli_search import app as _search_app  # noqa: E402

app.add_typer(_search_app, name="search")

# Phase S4: orchestrate (direct deployment) subcommand group
from lmtune.cli_orchestrate import app as _orchestrate_app  # noqa: E402

app.add_typer(_orchestrate_app, name="orchestrate")

# Phase W: dashboard subcommand group
from lmtune.cli_dashboard import app as _dashboard_app  # noqa: E402

app.add_typer(_dashboard_app, name="dashboard")

# lmtune#SS-rec: contracts subcommand (RecordSpec / QuerySpec schema dump)
from lmtune.cli_contracts import app as _contracts_app  # noqa: E402

app.add_typer(_contracts_app, name="contracts")

# lmtune#WS: workload spec/provider subcommand (WorkloadSpec contract)
from lmtune.cli_workload import app as _workload_app  # noqa: E402

app.add_typer(_workload_app, name="workload")


def _default_db_path() -> Path:
    return Path(os.environ.get("LMTUNE_DB", "data/db/lmtune.duckdb"))


def _default_raw_dir() -> Path:
    return Path(os.environ.get("BENCH_RAW", "data/raw"))


@app.command("run")
def cmd_run(
    profile: Annotated[Path, typer.Option(..., "--profile", "-p", exists=True, readable=True)],
    endpoint: Annotated[Path, typer.Option(..., "--endpoint", "-e", exists=True, readable=True)],
    db: Annotated[Path, typer.Option("--db", help="DuckDB 경로")] = None,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="명령만 조립하고 실행하지 않음")
    ] = False,
    json_summary: Annotated[
        bool, typer.Option("--json-summary", help="stdout 마지막 줄에 machine-readable JSON 요약")
    ] = False,
):
    profile_obj = load_profile(profile)
    endpoint_obj = load_endpoint(endpoint)
    runner = get_runner(profile_obj.runner)
    run_id = str(ULID())
    raw_dir = raw_dir or _default_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        cmd = runner.build_command(profile_obj, endpoint_obj, run_id, raw_dir / run_id)
        cmd = runner._apply_overrides(cmd, profile_obj)
        import shlex

        console.print(shlex.join(cmd))
        raise typer.Exit(0)

    console.print(
        f"[bold]run_id[/bold]={run_id}  profile={profile_obj.slug}  endpoint={endpoint_obj.slug}  runner={runner.kind}"
    )

    endpoint_obj.resolve_api_key()  # env var 누락 시 조기 실패

    env_extra: dict[str, str] = {}
    if endpoint_obj.api_key_env and endpoint_obj.api_key_env != "OPENAI_API_KEY":
        env_extra["OPENAI_API_KEY"] = os.environ[endpoint_obj.api_key_env]

    prom_collector = None
    if endpoint_obj.metrics_url:
        prom_collector = PrometheusCollector(
            url=str(endpoint_obj.metrics_url),
            out_path=raw_dir / run_id / "prom_samples.jsonl",
            interval_sec=5.0,
        )
        prom_collector.start()
    try:
        artifact = runner.run(
            profile_obj, endpoint_obj, run_id, raw_dir, env_extra=env_extra or None
        )
    finally:
        if prom_collector is not None:
            prom_collector.stop()
            console.print(f"  prom samples: {prom_collector.samples_written}")

    store = DuckDBStore(db or _default_db_path())
    profile_yaml_text = profile.read_text(encoding="utf-8")
    git_sha = _git_sha()
    store.record_run(
        artifact,
        profile_obj,
        endpoint_obj,
        profile_yaml_text=profile_yaml_text,
        git_sha=git_sha,
    )
    store.close()

    # R0 contract snapshot — raw_dir/<run_id>/result.json 로 BenchmarkResult 덤프.
    # 후속 PR (OD) 가 ArtifactStore 경유로 적재 시 본 JSON 이 source of truth.
    from lmtune.contracts import to_records
    from lmtune.runners.result_emit import runartifact_to_result
    from lmtune.storage.store import LocalArtifactStore

    result = runartifact_to_result(
        artifact,
        profile_obj,
        endpoint_obj,
        profile_yaml=profile_yaml_text,
        git_sha=git_sha,
        tool_versions={artifact.runner_kind: artifact.tool_version},
    )
    (raw_dir / run_id / "result.json").write_text(
        result.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    # LocalArtifactStore — records 를 kind 별 jsonl 로 raw_dir/<run_id>/records/.
    # DuckDB 미설치 환경 / git archive / S3 sync 시 동일 데이터 보유.
    LocalArtifactStore(raw_dir / run_id / "records").put(to_records(result))

    console.print(f"[bold green]done[/bold green] status={artifact.status}")
    if artifact.metrics:
        _print_metrics(artifact.metrics)
    if json_summary:
        summary = _build_run_summary(run_id, artifact, profile_obj.slo)
        print(json.dumps(summary, separators=(",", ":")))
    if artifact.status != "ok" and artifact.error:
        console.print(f"[red]error:[/red] {artifact.error}")
        raise typer.Exit(1)


_OP_FUNCS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _evaluate_slo(metrics: dict[str, dict[str, float]], slo: SLOSpec) -> tuple[bool, list[dict]]:
    """Return (all_passed, per_check_results). Warning + critical both counted."""
    results: list[dict] = []
    all_pass = True
    for chk in slo.resolved_checks():
        bucket = metrics.get(chk.metric) or {}
        observed = bucket.get(chk.p)
        if observed is None:
            results.append(
                {
                    "metric": chk.metric,
                    "p": chk.p,
                    "op": chk.op,
                    "value": chk.value,
                    "observed": None,
                    "passed": False,
                    "reason": "missing",
                }
            )
            all_pass = False
            continue
        op_fn = _OP_FUNCS.get(chk.op)
        passed = bool(op_fn(observed, chk.value)) if op_fn else False
        if not passed:
            all_pass = False
        results.append(
            {
                "metric": chk.metric,
                "p": chk.p,
                "op": chk.op,
                "value": chk.value,
                "observed": observed,
                "passed": passed,
                "severity": chk.severity,
            }
        )
    return all_pass, results


def _build_run_summary(run_id: str, artifact, slo: SLOSpec) -> dict:
    """One-line JSON summary for --json-summary / autoresearch consumption."""
    flat: dict[str, float] = {}
    for metric_name, bucket in (artifact.metrics or {}).items():
        for stat, val in bucket.items():
            flat[f"{metric_name}.{stat}"] = val
    slo_pass, slo_detail = _evaluate_slo(artifact.metrics or {}, slo)
    return {
        "run_id": run_id,
        "status": artifact.status,
        "slo_pass": slo_pass,
        "slo_checks": slo_detail,
        "metrics": flat,
        "error": artifact.error,
    }


@app.command("sweep")
def cmd_sweep(
    profile_dir: Annotated[Path, typer.Option("--profile-dir", exists=True, file_okay=False)],
    endpoint: Annotated[Path, typer.Option("--endpoint", "-e", exists=True)],
    db: Annotated[Path, typer.Option("--db")] = None,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error/--fail-fast")] = True,
):
    profiles = discover_profiles(profile_dir)
    if not profiles:
        console.print(f"[yellow]no profiles found under[/yellow] {profile_dir}")
        raise typer.Exit(1)
    endpoint_obj = load_endpoint(endpoint)
    endpoint_obj.resolve_api_key()

    store = DuckDBStore(db or _default_db_path())
    raw_dir = raw_dir or _default_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    for p in profiles:
        run_id = str(ULID())
        runner = get_runner(p.runner)
        console.print(f"[bold]running[/bold] {p.slug} run_id={run_id}")
        artifact = runner.run(p, endpoint_obj, run_id, raw_dir)
        store.record_run(
            artifact,
            p,
            endpoint_obj,
            profile_yaml_text=_yaml_for(p, profile_dir),
            git_sha=_git_sha(),
        )
        console.print(f"  → status={artifact.status}")
        if artifact.status == "failed" and not continue_on_error:
            store.close()
            raise typer.Exit(1)
    store.close()


@app.command("ls")
def cmd_ls(
    endpoint_slug: Annotated[str | None, typer.Option("--endpoint", help="endpoint slug")] = None,
    profile_slug: Annotated[str | None, typer.Option("--profile", help="profile slug")] = None,
    last: Annotated[int, typer.Option("--last", "-n", help="최근 N건")] = 20,
    db: Annotated[Path, typer.Option("--db")] = None,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="copy-paste 친화 tab-separated (rich table 비활성)"),
    ] = False,
    ids_only: Annotated[
        bool,
        typer.Option("--ids", help="run_id 만 줄별 출력 (스크립트용)"),
    ] = False,
):
    store = DuckDBStore(db or _default_db_path())
    rows = store.list_runs(endpoint_slug=endpoint_slug, profile_slug=profile_slug, limit=last)
    if ids_only:
        for r in rows:
            print(r[0])
        store.close()
        return
    if plain:
        cols = ["run_id", "profile", "endpoint", "started_at", "status", "runner"]
        print("\t".join(cols))
        for r in rows:
            print("\t".join(str(x) if x is not None else "-" for x in r))
        store.close()
        return
    table = Table()
    table.add_column("run_id", no_wrap=True, overflow="fold")
    for c in ["profile", "endpoint", "started_at", "status", "runner"]:
        table.add_column(c, overflow="fold")
    for r in rows:
        table.add_row(*[str(x) if x is not None else "—" for x in r])
    console.print(table)
    store.close()


@app.command("report")
def cmd_report(
    run_id: str,
    out: Annotated[Path, typer.Option("--out", "-o", help="리포트 출력 디렉토리")] = Path(
        "data/reports"
    ),
    db: Annotated[Path, typer.Option("--db")] = None,
):
    store = DuckDBStore(db or _default_db_path())
    row = store.get_run(run_id)
    if row is None:
        console.print(f"[red]run not found:[/red] {run_id}")
        raise typer.Exit(1)
    metrics = store.get_metrics(run_id)
    req_rows = store.conn.execute(
        """
        SELECT req_id, turn_idx, conversation_id, input_tokens, output_tokens,
               cached_tokens, thinking_tokens, tool_call_count, tool_result_tokens,
               phase, role, energy_wh, cost_usd,
               ttft_ms, itl_mean_ms, e2e_ms,
               NULL AS started_at_epoch, NULL AS completed_at_epoch,
               status, error
        FROM requests WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    rows = [RequestRow(*r) for r in req_rows]

    out_dir = Path(out) / run_id
    path = render_run_report(
        run_id=run_id,
        profile_slug=row[1],
        endpoint_slug=row[2],
        metrics=metrics,
        rows=rows,
        out_dir=out_dir,
    )
    console.print(f"report: [bold]{path}[/bold]")
    store.close()


@app.command("compare")
def cmd_compare(
    baseline: str,
    candidate: str,
    threshold_pct: Annotated[float, typer.Option("--threshold-pct", help="회귀 임계치 %")] = 10.0,
    db: Annotated[Path, typer.Option("--db")] = None,
):
    store = DuckDBStore(db or _default_db_path())
    b = store.get_metrics(baseline)
    c = store.get_metrics(candidate)
    if not b or not c:
        console.print("[red]one or both runs have no metrics[/red]")
        raise typer.Exit(1)
    cmp_ = compare_runs(baseline, candidate, b, c, regression_threshold_pct=threshold_pct)
    console.print(cmp_.to_markdown(regression_threshold_pct=threshold_pct))
    store.close()


@app.command("detect")
def cmd_detect(
    run_id: str,
    profile: Annotated[
        Path | None, typer.Option("--profile", help="SLO 참조용 profile yaml")
    ] = None,
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="회귀 비교용 baseline run_id")
    ] = None,
    threshold_pct: Annotated[float, typer.Option("--threshold-pct")] = 10.0,
    db: Annotated[Path, typer.Option("--db")] = None,
    save: Annotated[bool, typer.Option("--save/--no-save", help="detections 테이블에 기록")] = True,
):
    store = DuckDBStore(db or _default_db_path())
    if store.get_run(run_id) is None:
        console.print(f"[red]run not found:[/red] {run_id}")
        raise typer.Exit(1)

    metrics = store.get_metrics(run_id)
    req_rows = store.conn.execute(
        """
        SELECT req_id, turn_idx, conversation_id, input_tokens, output_tokens,
               cached_tokens, thinking_tokens, tool_call_count, tool_result_tokens,
               phase, role, energy_wh, cost_usd,
               ttft_ms, itl_mean_ms, e2e_ms,
               NULL AS started_at_epoch, NULL AS completed_at_epoch,
               status, error
        FROM requests WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    rows = [RequestRow(*r) for r in req_rows]

    slo = load_profile(profile).slo if profile else SLOSpec()
    baseline_data = None
    if baseline:
        baseline_metrics = store.get_metrics(baseline)
        if not baseline_metrics:
            console.print(
                f"[yellow]baseline run {baseline} has no metrics, skipping regression check[/yellow]"
            )
        else:
            baseline_data = (baseline, baseline_metrics)

    dets = run_all_rules(
        metrics=metrics,
        rows=rows,
        slo=slo,
        baseline=baseline_data,
        candidate_run_id=run_id,
        regression_threshold_pct=threshold_pct,
    )
    if not dets:
        console.print("[green]no detections[/green]")
    else:
        table = Table("severity", "detector", "metric", "observed", "threshold", "message")
        for d in dets:
            table.add_row(
                d.severity,
                d.detector,
                d.metric or "—",
                _fmt(d.observed),
                _fmt(d.threshold),
                d.message,
            )
        console.print(table)
        if save:
            store.record_detections(run_id, [d.to_dict() for d in dets])
            console.print(f"saved {len(dets)} detections to db")
    store.close()


@app.command("repeat")
def cmd_repeat(
    profile: Annotated[Path, typer.Option("--profile", "-p", exists=True)],
    endpoint: Annotated[Path, typer.Option("--endpoint", "-e", exists=True)],
    count: Annotated[
        int, typer.Option("--count", "-n", help="반복 실행 횟수 (variance 측정용)")
    ] = 10,
    db: Annotated[Path, typer.Option("--db")] = None,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = None,
):
    """같은 profile 을 N 번 반복 실행 (run-to-run variance 측정)."""
    profile_obj = load_profile(profile)
    endpoint_obj = load_endpoint(endpoint)
    endpoint_obj.resolve_api_key()
    runner = get_runner(profile_obj.runner)
    raw_dir = raw_dir or _default_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    store = DuckDBStore(db or _default_db_path())

    run_ids: list[str] = []
    for i in range(count):
        rid = str(ULID())
        console.print(f"[bold]run {i + 1}/{count}[/bold] run_id={rid}")
        art = runner.run(profile_obj, endpoint_obj, rid, raw_dir)
        store.record_run(
            art,
            profile_obj,
            endpoint_obj,
            profile_yaml_text=profile.read_text(encoding="utf-8"),
            git_sha=_git_sha(),
        )
        run_ids.append(rid)
        console.print(f"  → status={art.status}")
    store.close()
    console.print(f"[green]repeat done[/green] run_ids=[{', '.join(run_ids)}]")


@app.command("variance")
def cmd_variance(
    profile_slug: str,
    last: Annotated[int, typer.Option("--last", "-n", help="최근 N 개 run")] = 10,
    metric: Annotated[str, typer.Option("--metric")] = "ttft",
    p: Annotated[str, typer.Option("--p")] = "p99",
    db: Annotated[Path, typer.Option("--db")] = None,
):
    """최근 N 개 run 에서 variance 통계 (μ, σ, CV, IQR)."""
    store = DuckDBStore(db or _default_db_path())
    rows = store.list_runs(profile_slug=profile_slug, limit=last)
    run_ids = [r[0] for r in rows]
    if not run_ids:
        console.print(f"[yellow]no runs for profile {profile_slug}[/yellow]")
        raise typer.Exit(1)
    run_metrics = {rid: store.get_metrics(rid) for rid in run_ids}
    stats = variance_across_runs(run_metrics, metric, p)
    console.print(f"[bold]{profile_slug} — {metric}[{p}] across {stats.n} runs[/bold]")
    table = Table("stat", "value")
    for name, v in [
        ("mean", stats.mean),
        ("std", stats.std),
        ("cv", stats.cv),
        ("min", stats.min_),
        ("p50", stats.p50),
        ("max", stats.max_),
        ("iqr", stats.iqr),
        ("iqr/median", stats.iqr_ratio),
    ]:
        table.add_row(name, _fmt(v))
    console.print(table)
    store.close()


@app.command("nway")
def cmd_nway(
    run_ids: Annotated[list[str], typer.Argument(help="비교할 run_id 들 (2+)")],
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    db: Annotated[Path, typer.Option("--db")] = None,
):
    """N 개 run 의 metrics 매트릭스 비교."""
    if len(run_ids) < 2:
        console.print("[red]nway 는 최소 2 개 run 이 필요합니다[/red]")
        raise typer.Exit(1)
    store = DuckDBStore(db or _default_db_path())
    run_metrics = {rid: store.get_metrics(rid) for rid in run_ids}
    table = build_nway_table(run_metrics, metrics=[metric] if metric else None)
    console.print(
        nway_to_markdown(
            table, title=f"N-way: {', '.join(run_ids[:3])}{'...' if len(run_ids) > 3 else ''}"
        )
    )
    store.close()


@app.command("export")
def cmd_export(
    run_id: str,
    fmt: Annotated[str, typer.Option("--format", "-f", help="csv | parquet | json")] = "csv",
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data/exports"),
    db: Annotated[Path, typer.Option("--db")] = None,
):
    """requests 테이블을 원하는 포맷으로 export."""
    store = DuckDBStore(db or _default_db_path())
    req_rows = store.conn.execute("SELECT * FROM requests WHERE run_id = ?", [run_id]).fetchdf()
    metrics_df = store.conn.execute("SELECT * FROM metrics WHERE run_id = ?", [run_id]).fetchdf()
    out = Path(out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        req_rows.to_csv(out / "requests.csv", index=False)
        metrics_df.to_csv(out / "metrics.csv", index=False)
    elif fmt == "parquet":
        req_rows.to_parquet(out / "requests.parquet", index=False)
        metrics_df.to_parquet(out / "metrics.parquet", index=False)
    elif fmt == "json":
        req_rows.to_json(out / "requests.json", orient="records", indent=2)
        metrics_df.to_json(out / "metrics.json", orient="records", indent=2)
    else:
        console.print(f"[red]unknown format: {fmt}[/red]")
        raise typer.Exit(1)
    console.print(
        f"exported {len(req_rows)} requests + {len(metrics_df)} metrics to [bold]{out}[/bold]"
    )
    store.close()


@app.command("show")
def cmd_show(run_id: str, db: Annotated[Path, typer.Option("--db")] = None):
    store = DuckDBStore(db or _default_db_path())
    run = store.get_run(run_id)
    if run is None:
        console.print(f"[red]run not found:[/red] {run_id}")
        raise typer.Exit(1)
    console.print_json(data={"run": list(map(str, run))})
    metrics = store.get_metrics(run_id)
    _print_metrics(metrics)
    store.close()


def _print_metrics(metrics: dict[str, dict[str, float]]):
    if not metrics:
        return
    table = Table("metric", "p50", "p95", "p99", "avg")
    for name in sorted(metrics):
        b = metrics[name]
        table.add_row(
            name, _fmt(b.get("p50")), _fmt(b.get("p95")), _fmt(b.get("p99")), _fmt(b.get("avg"))
        )
    console.print(table)


def _fmt(v):
    return f"{v:.2f}" if v is not None else "—"


def _git_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return (out.stdout or "").strip() or None
    except OSError:
        return None


def _yaml_for(profile, dir_: Path) -> str:
    candidates = list(Path(dir_).rglob(f"{profile.slug}.yaml"))
    return candidates[0].read_text(encoding="utf-8") if candidates else ""


if __name__ == "__main__":
    app()
