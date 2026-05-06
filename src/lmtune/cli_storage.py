"""``lmtune storage`` subcommand — ArtifactStore backend 운영 도구 6 명령.

서브커맨드:
  lmtune storage list-backends
      등록된 backend 이름 목록.
  lmtune storage describe-backend <name>
      backend 의 class / 의존성 / path 인자 의미 / capability 표시.
  lmtune storage migrate --src-kind ... --dst-kind ... ...
      한 backend → 다른 backend 로 record 일괄 복사 (ABC put/query 만 사용).
  lmtune storage info --kind <backend> --path <p>
      record kind 별 count 보고.
  lmtune storage validate --kind <backend> --path <p>
      모든 record 의 schema validity 검증.
  lmtune storage diff --left-* --right-*
      두 store 의 record 차이 (only_left / only_right / mismatched).

Use cases:
  - DuckDB (운영) → Local jsonl (git archive, S3 sync, jq 검색)
  - Local jsonl (외부에서 받은 archive) → DuckDB (분석 쿼리)
  - 동일 종류끼리 복사 (백업)
"""

from __future__ import annotations

import json
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
def cmd_list_backends(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="기계 친화적 JSON 출력"),
    ] = False,
) -> None:
    """등록된 ArtifactStore backend 목록.

    새 backend 가 ABC 를 구현해 plug-in 되면 본 목록에 합류한다 (PLUG 패턴 시연).
    ``lmtune tuner list-{samplers,pruners}`` / ``lmtune contracts list-records`` 와
    동일한 ``--json`` 표면.
    """
    if json_output:
        print(json.dumps({"backends": list(_BACKENDS)}))
        return
    for b in _BACKENDS:
        console.print(f"- {b}")


# backend 별 메타 — describe-backend 가 노출. CLI 모듈에서 직접 관리하는 이유:
# 외부 SDK 미설치 환경에서도 PostgresArtifactStore 의 docstring 등을 읽을 수 있어야
# 하지만 클래스 자체는 import 시 ImportError 가능. 가벼운 dict 가 실용적.
_BACKEND_META: dict[str, dict[str, object]] = {
    "local": {
        "class_name": "LocalArtifactStore",
        "module": "lmtune.storage.store.local",
        "summary": "JSONL per kind — disk 영속, primary_key dedup, grep 친화",
        "path_kind": "directory",
        "extras": None,
        "transactional": False,
        "concurrent_writers": False,
    },
    "duckdb": {
        "class_name": "DuckDBArtifactStore",
        "module": "lmtune.storage.store.duckdb",
        "summary": "DuckDB single-file — query 성능, ACID",
        "path_kind": "file",
        "extras": None,
        "transactional": True,
        "concurrent_writers": False,
    },
    "postgres": {
        "class_name": "PostgresArtifactStore",
        "module": "lmtune.storage.store.postgres",
        "summary": "Postgres DSN — multi-writer, server-side store (stub, psycopg 필요)",
        "path_kind": "dsn",  # postgres://user:pass@host/db
        "extras": "[postgres]",
        "transactional": True,
        "concurrent_writers": True,
    },
}


@app.command("describe-backend")
def cmd_describe_backend(
    name: Annotated[
        str,
        typer.Argument(help=f"backend name. valid: {', '.join(_BACKENDS)}"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="기계 친화적 JSON 출력"),
    ] = False,
) -> None:
    """특정 backend 의 메타 표시 — class / 의존성 / path 인자 의미 / capability.

    ``lmtune tuner describe`` / ``lmtune contracts describe-record`` 와 동일한
    metadata 표면 패턴. axis 대칭 — Storage / Tuner / Contracts 가 같은
    list / describe / paste-able 4 layer 가시성을 갖는다.
    """
    if name not in _BACKENDS:
        raise typer.BadParameter(
            f"unknown backend: {name!r}. use 'lmtune storage list-backends' to see valid names."
        )

    meta = _BACKEND_META[name]
    payload = {"name": name, **meta}
    if json_output:
        print(json.dumps(payload, separators=(",", ":"), default=str))
        return

    console.print(f"[bold]{name}[/bold]  ({meta['class_name']})")
    console.print(f"  [dim]{meta['summary']}[/dim]")
    console.print(f"  module: [dim]{meta['module']}[/dim]")
    console.print(f"  path arg: [cyan]{meta['path_kind']}[/cyan]")
    if meta["extras"]:
        # rich 가 [postgres] 를 markup 으로 해석하지 않도록 escape
        extras_str = str(meta["extras"]).replace("[", r"\[")
        console.print(f"  install: pip install lmtune{extras_str}")
    transactional = "yes" if meta["transactional"] else "no"
    concurrent = "yes" if meta["concurrent_writers"] else "no"
    console.print(f"  transactional: {transactional}, concurrent writers: {concurrent}")


@app.command("diff")
def cmd_diff(
    left_kind: Annotated[
        str,
        typer.Option("--left-kind", help=f"left backend: {' | '.join(_BACKENDS)}"),
    ],
    left: Annotated[
        Path,
        typer.Option("--left", help="left store path"),
    ],
    right_kind: Annotated[
        str,
        typer.Option("--right-kind", help=f"right backend: {' | '.join(_BACKENDS)}"),
    ],
    right: Annotated[
        Path,
        typer.Option("--right", help="right store path"),
    ],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="machine-readable JSON 출력"),
    ] = False,
) -> None:
    """두 ArtifactStore 의 record 차이 보고 — primary_key 기준.

    각 record kind 마다 양쪽 store 를 query → primary_key 집합 비교.
    only_left, only_right, mismatched (같은 PK 인데 다른 값) 3 카테고리로 분류.

    Use cases:
    - DuckDB 운영본 ↔ Local archive drift 검출
    - 외부에서 받은 archive 와 자기 DB 의 차이 분석
    - migrate 후 round-trip 정합성 (양쪽 should be equal)
    """
    import json as _json

    left_store = _open_store(left_kind, left)
    right_store = _open_store(right_kind, right)

    by_kind: dict[str, dict[str, int]] = {}
    try:
        for record_kind in RECORD_KINDS:
            l_rows = left_store.query(QuerySpec(record_kind=record_kind))
            r_rows = right_store.query(QuerySpec(record_kind=record_kind))
            l_map = {r.primary_key(): r for r in l_rows}  # type: ignore[attr-defined]
            r_map = {r.primary_key(): r for r in r_rows}  # type: ignore[attr-defined]

            l_keys = set(l_map.keys())
            r_keys = set(r_map.keys())
            only_left = l_keys - r_keys
            only_right = r_keys - l_keys
            common = l_keys & r_keys

            mismatched = 0
            for k in common:
                if l_map[k].model_dump() != r_map[k].model_dump():
                    mismatched += 1

            counts = {
                "left": len(l_rows),
                "right": len(r_rows),
                "only_left": len(only_left),
                "only_right": len(only_right),
                "mismatched": mismatched,
            }
            # 0/0/0 인 kind 는 생략 (출력 노이즈 감소)
            if any(v > 0 for v in counts.values()):
                by_kind[record_kind] = counts
    finally:
        left_store.close()
        right_store.close()

    is_equal = all(
        c["only_left"] == 0 and c["only_right"] == 0 and c["mismatched"] == 0
        for c in by_kind.values()
    )

    if json_out:
        payload = {
            "left": f"{left_kind}://{left}",
            "right": f"{right_kind}://{right}",
            "equal": is_equal,
            "by_kind": by_kind,
        }
        print(_json.dumps(payload, separators=(",", ":")))
        return

    if is_equal:
        console.print(
            f"[bold green]equal[/bold green]  {left_kind}://{left}  ==  {right_kind}://{right}"
        )
        return

    console.print(f"[yellow]differs[/yellow]  {left_kind}://{left}  vs  {right_kind}://{right}")
    for k, c in by_kind.items():
        if c["only_left"] == 0 and c["only_right"] == 0 and c["mismatched"] == 0:
            continue
        console.print(
            f"  {k:<20} L={c['left']} R={c['right']}  "
            f"only_L={c['only_left']} only_R={c['only_right']} mismatch={c['mismatched']}"
        )


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
