"""``lmtune contracts`` subcommand — contract schema dump 도구.

서브커맨드:
  lmtune contracts list-records [--json]
      유효한 record kind 목록을 출력. 다른 axis 의 가시성 표면과 대칭
      (``lmtune storage list-backends`` / ``lmtune tuner list-{samplers,pruners}``).
  lmtune contracts describe-record <kind> [--json]
      특정 record kind 의 필드 표시 — name / type / required / default / description.
      Pydantic ``model_fields`` introspect. lmtune tuner describe 와 동일 패턴.
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
  lmtune contracts make-template --record-kind run [--format json|yaml]
      특정 record kind 의 빈 (필수 필드 채워진) 템플릿 출력 — 사용자가
      신규 record 작성 시 paste 가능. lmtune tuner make-config 와 동일 패턴.
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


@app.command("list-records")
def cmd_list_records(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="기계 친화적 JSON 출력"),
    ] = False,
) -> None:
    """유효한 record kind 목록을 출력.

    ``RECORD_KINDS`` (record_spec 의 단일 진실원) 를 그대로 노출. 다른 PLUG axis
    의 가시성 표면 (``lmtune storage list-backends`` / ``lmtune tuner
    list-{samplers,pruners}``) 과 대칭. 새 record kind 추가 시 자동 반영.
    """
    if json_output:
        print(json.dumps({"records": list(RECORD_KINDS)}))
        return
    console.print(f"[bold]record kinds[/bold] ({len(RECORD_KINDS)}):")
    for kind in RECORD_KINDS:
        cls = kind_to_class(kind)
        console.print(f"  - [cyan]{kind}[/cyan]  ({cls.__name__})")


@app.command("describe-record")
def cmd_describe_record(
    record_kind: Annotated[
        str,
        typer.Argument(help=f"record kind (필수). valid: {', '.join(RECORD_KINDS)}"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="기계 친화적 JSON 출력"),
    ] = False,
) -> None:
    """특정 record kind 의 필드 표시 — name / type / required / default / description.

    Pydantic ``model_fields`` introspect. 외부 사용자가 record schema 를 학습할 때
    full JSON Schema (`dump-schema`) 보다 짧고 사람 친화적인 표면. ``lmtune tuner
    describe`` 와 동일 패턴 (metadata 표면).
    """
    if record_kind not in RECORD_KINDS:
        raise typer.BadParameter(
            f"unknown record kind: {record_kind!r}. "
            "use 'lmtune contracts list-records' to see valid kinds."
        )

    cls = kind_to_class(record_kind)
    fields = []
    for name, field in cls.model_fields.items():
        ann = field.annotation
        default = field.default
        fields.append(
            {
                "name": name,
                "annotation": str(ann) if ann is not None else None,
                "required": field.is_required(),
                "default": None if field.is_required() else repr(default),
                "description": field.description,
            }
        )

    payload = {
        "kind": record_kind,
        "class_name": cls.__name__,
        "module": cls.__module__,
        "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        "primary_key_arity": len(cls.model_fields),
        "fields": fields,
    }
    if json_output:
        print(json.dumps(payload, separators=(",", ":"), default=str))
        return

    console.print(f"[bold]{payload['kind']}[/bold]  ({payload['class_name']})")
    if payload["doc"]:
        console.print(f"  [dim]{payload['doc']}[/dim]")
    console.print(f"  module: [dim]{payload['module']}[/dim]")
    console.print(f"  fields ({len(fields)}):")
    for f in fields:
        marker = "[red]*[/red]" if f["required"] else " "
        ann_short = (f["annotation"] or "?").replace("typing.", "")
        line = f"    {marker} [cyan]{f['name']}[/cyan]: {ann_short}"
        if not f["required"]:
            line += f"  [dim](default={f['default']})[/dim]"
        if f["description"]:
            line += f"  — {f['description']}"
        console.print(line)


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


@app.command("make-template")
def cmd_make_template(
    record_kind: Annotated[
        str,
        typer.Option(
            "--record-kind",
            "-k",
            help=f"record kind (필수). valid: {', '.join(RECORD_KINDS)}",
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="출력 형식: json | yaml"),
    ] = "json",
) -> None:
    """``record_kind`` 의 빈 (필수 필드 채워진) 템플릿 출력.

    Pydantic ``model_fields`` introspection — 필수 필드는 placeholder, optional
    필드는 default 값. 사용자가 신규 record 작성 시 paste 후 placeholder 만
    수정. ``lmtune contracts validate-record`` 으로 검증 가능한 형식.

    ``lmtune tuner make-config`` 와 동일 패턴 (paste-able 표면).
    """
    if record_kind not in RECORD_KINDS:
        raise typer.BadParameter(
            f"unknown record kind: {record_kind!r}. valid: {list(RECORD_KINDS)}"
        )

    cls = kind_to_class(record_kind)
    template: dict[str, object] = {}
    for name, field in cls.model_fields.items():
        if name == "kind":
            template[name] = record_kind
            continue
        if field.is_required():
            # placeholder for required fields — type-aware
            ann = field.annotation
            template[name] = _placeholder_for(ann, name)
        else:
            template[name] = field.default if field.default is not None else None

    if fmt.lower() == "json":
        print(json.dumps(template, separators=(",", ":"), default=str))
        return
    if fmt.lower() == "yaml":
        print(yaml.safe_dump(template, sort_keys=False, allow_unicode=True).rstrip())
        return
    raise typer.BadParameter(f"unknown format: {fmt!r}. valid: json | yaml")


def _placeholder_for(ann: object, name: str) -> object:
    """type annotation → 합리적 placeholder.

    int → 0, float → 0.0, str → '<name>', bool → false, dict → {}, list → [].
    Union/Optional 은 첫 non-None branch. 그 외 None.
    """
    s = str(ann)
    if "int" in s and "List" not in s and "list" not in s:
        return 0
    if "float" in s:
        return 0.0
    if "bool" in s:
        return False
    if "dict" in s or "Dict" in s or "Mapping" in s:
        return {}
    if "list" in s or "List" in s:
        return []
    if "datetime" in s:
        # ISO 8601 placeholder
        return "1970-01-01T00:00:00Z"
    if "str" in s:
        return f"<{name}>"
    return None
