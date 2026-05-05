"""``lmtune contracts`` subcommand — contract schema dump 도구.

서브커맨드:
  lmtune contracts dump-schema --kind {record,query,result} [--out file.json]
      RecordSpec / QuerySpec / BenchmarkResult 의 JSON Schema 출력.
  lmtune contracts dump-schema --kind record --record-kind run [--out file.json]
      특정 record kind (run, trial, ...) 만의 schema.
  lmtune contracts validate-record <yaml-or-json>
      RecordSpec 단일 레코드 yaml/json 의 schema validity 검증.
  lmtune contracts validate-result <yaml-or-json>
      BenchmarkResult yaml/json 의 schema validity 검증.
  lmtune contracts records-from-result <result.json> --out <dir>
      BenchmarkResult → records (run/metric/...) jsonl 디렉토리.
      to_records() + LocalArtifactStore 결합. archive/migration 도구.
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
from lmtune.contracts.result_spec import BenchmarkResult

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
    """RecordSpec / QuerySpec / BenchmarkResult → JSON Schema."""
    if kind == "record":
        if record_kind:
            schema = kind_to_class(record_kind).model_json_schema()
        else:
            schema = TypeAdapter(RecordSpec).json_schema()
    elif kind == "query":
        if record_kind:
            raise typer.BadParameter("--record-kind 는 --kind=record 일 때만 사용")
        schema = QuerySpec.model_json_schema()
    elif kind == "result":
        if record_kind:
            raise typer.BadParameter("--record-kind 는 --kind=record 일 때만 사용")
        schema = BenchmarkResult.model_json_schema()
    else:
        raise typer.BadParameter(f"unknown kind: {kind!r}, valid: record | query | result")

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
    console.print(f"[bold green]ok[/bold green]  kind={rec.kind}  primary_key={rec.primary_key()}")


@app.command("validate-result")
def cmd_validate_result(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """BenchmarkResult yaml/json → schema validity 검증."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})

    try:
        result = BenchmarkResult.model_validate(data)
    except Exception as e:
        console.print(f"[red]invalid:[/red] {e}")
        raise typer.Exit(1) from None
    console.print(
        f"[bold green]ok[/bold green]  run_id={result.run_id}  "
        f"runner={result.runner_kind}  status={result.status}  "
        f"metrics={len(result.metrics)}  requests={len(result.requests)}"
    )


@app.command("records-from-result")
def cmd_records_from_result(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="LocalArtifactStore root (jsonl 디렉토리)"),
    ],
) -> None:
    """BenchmarkResult yaml/json → ``out/<kind>.jsonl`` 디렉토리.

    ``to_records()`` + ``LocalArtifactStore`` 결합. result.json 한 파일에 들어
    있는 RunRecord/MetricRecord/RequestRecord/SessionRecord/TrajectoryEvent 를
    kind 별 jsonl 로 풀어내어 grep/jq/git 친화 형식 + ArtifactStore.query() 가능.
    """
    from lmtune.contracts import to_records
    from lmtune.storage.store import LocalArtifactStore

    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})

    try:
        result = BenchmarkResult.model_validate(data)
    except Exception as e:
        console.print(f"[red]invalid result:[/red] {e}")
        raise typer.Exit(1) from None

    records = to_records(result)
    store = LocalArtifactStore(out)
    n = store.put(records)

    by_kind: dict[str, int] = {}
    for rec in records:
        by_kind[rec.kind] = by_kind.get(rec.kind, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
    console.print(f"[bold green]wrote[/bold green]  {n} records → {out}  ({breakdown})")
