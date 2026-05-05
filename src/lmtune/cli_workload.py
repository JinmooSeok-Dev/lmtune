"""``lmtune workload`` subcommand — WorkloadSpec 의 단독 입출력.

본 명령은 cmd_run 과 별개. cmd_run 통합 (--workload-spec / --workload-source flag)
은 후속 PR. 본 PR 은 leaf 영역만 — Provider 호출 + yaml 검증 + JSON Schema dump.

서브커맨드:
  lmtune workload generate --source vllm-log:/path --out ws.yaml
      LMWorkloadsProvider 호출 → WorkloadSpec yaml 작성

  lmtune workload validate ws.yaml
      yaml 파일이 workloads/v1alpha1 schema 와 부합하는지 확인

  lmtune workload dump-schema --out workload.schema.json
      Pydantic → JSON Schema export (다른 언어/도구에서 검증 가능)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="WorkloadSpec 단독 도구")
console = Console()


@app.command("generate")
def cmd_generate(
    source: Annotated[
        str,
        typer.Option("--source", help="``<adapter>:<path>`` (예: vllm-log:/var/log/vllm.ndjson)"),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="출력 yaml 경로"),
    ],
    cluster_id: Annotated[
        str | None,
        typer.Option("--cluster-id", help="다중 cluster 발견 시 명시 선택"),
    ] = None,
    store_path: Annotated[
        Path | None,
        typer.Option("--store-path", help="lm-workloads DuckDB store (영속 누적)"),
    ] = None,
):
    """lm-workloads 호출 → WorkloadSpec yaml 작성."""
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    provider = LMWorkloadsProvider(
        source_uri=source, cluster_id=cluster_id, store_path=store_path
    )
    spec = provider.provide()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"[bold green]wrote[/bold green] {out}  (id={spec.meta.id})")


@app.command("validate")
def cmd_validate(
    yaml_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
):
    """WorkloadSpec yaml 의 schema validity 검증. 종료코드 0=pass, 1=fail."""
    from lmtune.workload.providers.literal import LiteralWorkloadProvider

    try:
        spec = LiteralWorkloadProvider(yaml_path).provide()
    except Exception as e:
        console.print(f"[red]invalid:[/red] {e}")
        raise typer.Exit(1) from None
    console.print(
        f"[bold green]ok[/bold green]  apiVersion={spec.apiVersion}  "
        f"id={spec.meta.id}  category={spec.classification.category}"
    )


@app.command("dump-schema")
def cmd_dump_schema(
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("workload.schema.json"),
):
    """WorkloadSpec Pydantic → JSON Schema (언어 무관)."""
    from lmtune.contracts.workload_spec import WorkloadSpec

    schema = WorkloadSpec.model_json_schema()
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]wrote[/bold green] {out}")
