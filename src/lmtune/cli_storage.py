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


@app.command("validate")
def cmd_validate(
    kind: Annotated[
        str,
        typer.Option("--kind", help=f"backend: {' | '.join(_BACKENDS)}"),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="store path"),
    ],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력"),
    ] = False,
) -> None:
    """ArtifactStore 의 모든 record 가 RecordSpec schema 를 통과하는지 검증.

    각 record kind 마다 ``store.query(QuerySpec(record_kind=k))`` 시도. backend
    가 jsonl/duckdb 파일을 RecordSpec 으로 deserialize 하는 단계에서 schema 위반
    시 raise. 본 명령은 그 raise 를 잡아 어느 kind 가 깨졌는지 보고.

    Use cases:
    - 외부 archive 디렉토리를 받았을 때 신뢰성 검증
    - DuckDB 마이그레이션 후 round-trip 무결성 확인
    - CI 의 schema drift 가드 (record 추가 시 누락된 필드 자동 검출)
    """
    import json as _json

    store = _open_store(kind, path)
    valid: dict[str, int] = {}
    invalid: dict[str, str] = {}
    try:
        for record_kind in RECORD_KINDS:
            try:
                rows = store.query(QuerySpec(record_kind=record_kind))
                valid[record_kind] = len(rows)
            except Exception as e:  # noqa: BLE001
                invalid[record_kind] = f"{type(e).__name__}: {e}"
    finally:
        store.close()

    total_valid = sum(valid.values())
    is_valid = not invalid

    if json_out:
        payload = {
            "backend": kind,
            "path": str(path),
            "valid": is_valid,
            "valid_counts": valid,
            "invalid": invalid,
        }
        print(_json.dumps(payload, separators=(",", ":")))
        if not is_valid:
            raise typer.Exit(1)
        return

    if is_valid:
        console.print(
            f"[bold green]valid[/bold green]  {kind}://{path}  "
            f"({total_valid} records across {len(valid)} kinds)"
        )
    else:
        console.print(f"[red]invalid[/red]  {kind}://{path}")
        for k, err in invalid.items():
            console.print(f"  ✗ {k}: {err}")
        for k, n in valid.items():
            if n > 0:
                console.print(f"  ✓ {k}: {n}")
        raise typer.Exit(1)


@app.command("info")
def cmd_info(
    kind: Annotated[
        str,
        typer.Option("--kind", help=f"backend: {' | '.join(_BACKENDS)}"),
    ],
    path: Annotated[
        Path,
        typer.Option("--path", help="store path (local: dir, duckdb: file, postgres: dsn)"),
    ],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력 (autotune/monitoring 용)"),
    ] = False,
) -> None:
    """ArtifactStore 의 record kind 별 count 요약.

    backend 무관 — ABC 의 ``count(kind)`` 만 호출. 새 backend 가 합류해도 본 명령
    수정 0. 빈 kind 는 0 으로 표시.
    """
    store = _open_store(kind, path)
    counts: dict[str, int] = {}
    try:
        for record_kind in RECORD_KINDS:
            counts[record_kind] = store.count(record_kind)
    finally:
        store.close()

    total = sum(counts.values())

    if json_out:
        # one-line JSON — autoresearch 가 stat polling 시 grep 가능
        import json

        payload = {"backend": kind, "path": str(path), "total": total, "counts": counts}
        print(json.dumps(payload, separators=(",", ":")))
        return

    console.print(f"[bold]storage info[/bold] {kind}://{path}")
    for k, v in counts.items():
        marker = "  " if v == 0 else "● "
        console.print(f"  {marker}{k:<20} {v}")
    console.print(f"  [bold]total[/bold]               {total}")
