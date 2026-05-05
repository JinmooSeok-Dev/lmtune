"""`bench dashboard` — static HTML dashboard build/serve."""

from __future__ import annotations

import http.server
import os
import socketserver
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lmtune.visualization.dashboard import build_dashboard

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Static HTML dashboard")
console = Console()


def _default_db_path() -> Path:
    return Path(os.environ.get("LMTUNE_DB", "data/db/lmtune.duckdb"))


@app.command("build")
def cmd_build(
    out_dir: Annotated[Path, typer.Option("--out", help="출력 디렉토리")] = Path("b200/dashboards"),
    db: Annotated[Path, typer.Option("--db", help="DuckDB 경로")] = None,
    perf_changelog: Annotated[
        Path | None,
        typer.Option("--perf-changelog", help="perf-changelog.yaml 경로 (옵션)"),
    ] = None,
    study_id: Annotated[
        list[str] | None,
        typer.Option("--study", help="특정 study_id 만 포함 (반복 가능)"),
    ] = None,
):
    """DuckDB → 정적 HTML 대시보드를 생성."""
    db_path = db or _default_db_path()
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/red]")
        raise typer.Exit(1)

    if perf_changelog is None:
        candidate = Path("b200/perf-changelog.yaml")
        if candidate.exists():
            perf_changelog = candidate

    written = build_dashboard(
        db_path=db_path,
        out_dir=out_dir,
        perf_changelog=perf_changelog,
        study_ids=list(study_id) if study_id else None,
    )

    table = Table(title=f"dashboard built — {out_dir}")
    table.add_column("artifact", style="cyan")
    table.add_column("path")
    for name, path in sorted(written.items()):
        table.add_row(name, str(path))
    console.print(table)
    console.print(f"\n[green]✓[/green] open: file://{(out_dir / 'index.html').resolve()}")


@app.command("serve")
def cmd_serve(
    out_dir: Annotated[Path, typer.Option("--out", help="대시보드 디렉토리")] = Path(
        "b200/dashboards"
    ),
    port: Annotated[int, typer.Option("--port", "-p")] = 8765,
    bind: Annotated[str, typer.Option("--bind")] = "127.0.0.1",
):
    """`python -m http.server` 래핑 — 대시보드 디렉토리를 로컬에서 서빙."""
    if not out_dir.exists():
        console.print(f"[red]not found: {out_dir} — run `bench dashboard build` first[/red]")
        raise typer.Exit(1)

    def handler(*a, **kw):
        return http.server.SimpleHTTPRequestHandler(*a, directory=str(out_dir.resolve()), **kw)

    with socketserver.TCPServer((bind, port), handler) as httpd:
        url = f"http://{bind}:{port}/"
        console.print(f"[green]serving[/green] {out_dir} at {url}")
        console.print("Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\nstopped")
