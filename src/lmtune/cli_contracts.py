"""``lmtune contracts`` subcommand — contract schema dump 도구.

서브커맨드:
  lmtune contracts dump-schema --kind {record,query} [--out file.json]
      RecordSpec 또는 QuerySpec 의 JSON Schema 출력.
  lmtune contracts dump-schema --kind record --record-kind run [--out file.json]
      특정 record kind (run, trial, ...) 만의 schema.
  lmtune contracts validate-record <yaml-or-json>
      RecordSpec 단일 레코드 yaml/json 의 schema validity 검증.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import TypeAdapter
from rich.console import Console

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RECORD_KINDS, RecordSpec, kind_to_class

app = typer.Typer(no_args_is_help=True, help="lmtune contracts 단독 도구")
console = Console()


@app.command("dump-schema")
def cmd_dump_schema(
    kind: Annotated[
        str,
        typer.Option("--kind", help="``record`` | ``query``"),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="출력 파일 (생략 시 stdout)"),
    ] = None,
    record_kind: Annotated[
        str | None,
        typer.Option(
            "--record-kind",
            help=f"--kind=record 일 때 특정 record kind 만. valid: {', '.join(RECORD_KINDS)}",
        ),
    ] = None,
) -> None:
    """RecordSpec / QuerySpec → JSON Schema."""
    if kind == "record":
        if record_kind:
            schema = kind_to_class(record_kind).model_json_schema()
        else:
            schema = TypeAdapter(RecordSpec).json_schema()
    elif kind == "query":
        if record_kind:
            raise typer.BadParameter("--record-kind 는 --kind=record 일 때만 사용")
        schema = QuerySpec.model_json_schema()
    else:
        raise typer.BadParameter(f"unknown kind: {kind!r}, valid: record | query")

    text = json.dumps(schema, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[bold green]wrote[/bold green] {out}")
    else:
        # stdout 직접 (pipe 가능)
        print(text)


@app.command("validate-record")
def cmd_validate_record(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """단일 record yaml/json → RecordSpec 검증."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})

    try:
        rec = TypeAdapter(RecordSpec).validate_python(data)
    except Exception as e:
        console.print(f"[red]invalid:[/red] {e}")
        raise typer.Exit(1) from None
    console.print(
        f"[bold green]ok[/bold green]  kind={rec.kind}  "
        f"primary_key={rec.primary_key()}"
    )
