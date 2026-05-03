"""`bench orchestrate ...` — direct DeploymentAdapter invocations (debugging).

Primary use:
    bench orchestrate deploy --adapter local-vllm \
        --endpoint configs/endpoints/local_vllm_autotune.yaml \
        --params-json '{"max_num_seqs": 64, "enable_prefix_caching": true}'

This is intentionally separate from `bench search`: when a deployment goes
sideways you want to iterate on `apply()` alone without running N trials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="DeploymentAdapter 직접 호출 (디버깅/수동 배포)")
console = Console()


@app.command("deploy")
def cmd_deploy(
    endpoint: Annotated[Path, typer.Option("--endpoint", "-e", exists=True, readable=True)],
    adapter: Annotated[str, typer.Option("--adapter", help="local-vllm | llmd-k8s")] = "local-vllm",
    params_json: Annotated[str, typer.Option("--params-json", help="'{\"key\": val, ...}'")] = "{}",
):
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"invalid --params-json: {e}") from e
    if adapter == "local-vllm":
        from lmtune.deploy import LocalVLLMAdapter
        ad = LocalVLLMAdapter()
    elif adapter == "llmd-k8s":
        from lmtune.deploy import LLMDK8sAdapter
        ad = LLMDK8sAdapter()
    else:
        raise typer.BadParameter(f"unknown adapter: {adapter}")

    console.print(f"[bold]apply[/]: adapter={adapter} endpoint={endpoint} params={params}")
    result = ad.apply(endpoint, params)
    console.print(f"  ok={result.ok}  health.ready={result.health.ready}  "
                  f"latency_ms={result.health.latency_ms:.1f}  detail={result.health.detail}")
    if not result.ok:
        console.print(f"[red]failed[/]: {result.notes}")
        raise typer.Exit(1)
