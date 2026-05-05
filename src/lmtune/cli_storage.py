"""``lmtune storage`` subcommand — ArtifactStore backend 간 변환/이관 도구.

서브커맨드:
  lmtune storage migrate --src-kind {local,duckdb} --src <path> \
                         --dst-kind {local,duckdb} --dst <path> \
                         [--kinds run,metric,...] [--batch N]
      한 backend → 다른 backend 로 record 일괄 복사. ABC 의 put/query 만 사용
      → backend 추가 시 코드 수정 0.

Use cases:
  - DuckDB (운영) → Local jsonl (git archive, S3 sync, jq 검색)
  - Local jsonl (외부에서 받은 archive) → DuckDB (분석 쿼리)
  - 동일 종류끼리 복사 (백업)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RECORD_KINDS
from lmtune.storage.store import (
    ArtifactStore,
    DuckDBArtifactStore,
    LocalArtifactStore,
    PostgresArtifactStore,
)

app = typer.Typer(no_args_is_help=True, help="lmtune storage 도구")
console = Console()


_BACKENDS = ("local", "duckdb", "postgres")


def _open_store(kind: str, path: Path) -> ArtifactStore:
    if kind == "local":
        return LocalArtifactStore(path)
    if kind == "duckdb":
        return DuckDBArtifactStore(path)
    if kind == "postgres":
        # Postgres backend 는 path 를 dsn (postgres://...) 으로 해석.
        # psycopg 미설치 시 명확한 ImportError → typer.BadParameter 로 변환.
        try:
            return PostgresArtifactStore(str(path))
        except ImportError as e:
            raise typer.BadParameter(str(e)) from None
    raise typer.BadParameter(f"unknown backend: {kind!r}, valid: {_BACKENDS}")


@app.command("migrate")
def cmd_migrate(
    src_kind: Annotated[
        str,
        typer.Option("--src-kind", help=f"source backend: {' | '.join(_BACKENDS)}"),
    ],
    src: Annotated[
        Path,
        typer.Option("--src", help="source path (local: dir, duckdb: file)"),
    ],
    dst_kind: Annotated[
        str,
        typer.Option("--dst-kind", help=f"dest backend: {' | '.join(_BACKENDS)}"),
    ],
    dst: Annotated[
        Path,
        typer.Option("--dst", help="dest path"),
    ],
    kinds: Annotated[
        str | None,
        typer.Option(
            "--kinds",
            help=f"kind 화이트리스트 (콤마구분). 생략 시 전체. valid: {', '.join(RECORD_KINDS)}",
        ),
    ] = None,
) -> None:
    """src ArtifactStore → dst ArtifactStore 일괄 복사.

    각 record kind 마다 src.query(QuerySpec(record_kind=...)) → dst.put(...) 단순 루프.
    backend 무관 — local↔local, duckdb↔duckdb, local↔duckdb 4 조합 모두 동작.
    """
    if src_kind not in _BACKENDS:
        raise typer.BadParameter(f"--src-kind must be one of {_BACKENDS}")
    if dst_kind not in _BACKENDS:
        raise typer.BadParameter(f"--dst-kind must be one of {_BACKENDS}")

    selected: tuple[str, ...]
    if kinds:
        wanted = tuple(k.strip() for k in kinds.split(",") if k.strip())
        unknown = [k for k in wanted if k not in RECORD_KINDS]
        if unknown:
            raise typer.BadParameter(f"unknown kinds: {unknown}, valid: {RECORD_KINDS}")
        selected = wanted
    else:
        selected = RECORD_KINDS

    src_store = _open_store(src_kind, src)
    dst_store = _open_store(dst_kind, dst)

    total_copied = 0
    breakdown: dict[str, int] = {}
    try:
        for kind in selected:
            rows = src_store.query(QuerySpec(record_kind=kind))
            if not rows:
                continue
            n = dst_store.put(rows)
            breakdown[kind] = n
            total_copied += n
    finally:
        src_store.close()
        dst_store.close()

    if not breakdown:
        console.print("[yellow]no records[/yellow]  (src is empty for selected kinds)")
        return

    summary = ", ".join(f"{k}={v}" for k, v in sorted(breakdown.items()))
    console.print(
        f"[bold green]migrated[/bold green]  {total_copied} records  "
        f"({src_kind}://{src} → {dst_kind}://{dst})  [{summary}]"
    )


@app.command("list-backends")
def cmd_list_backends() -> None:
    """등록된 ArtifactStore backend 목록.

    새 backend 가 ABC 를 구현해 plug-in 되면 본 목록에 합류한다 (PLUG 패턴 시연).
    """
    for b in _BACKENDS:
        console.print(f"- {b}")
