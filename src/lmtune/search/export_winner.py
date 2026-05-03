"""Phase W — `bench search export <study_id> --winner top-1` → self-contained recipe.

산출 (b200/results/<study_id>/winner/ 디렉토리):
  - apply.sh             : 한 줄 실행 가능 (dry-run | apply)
  - values-overlay.yaml  : llmd-k8s adapter 의 helmfile state-values-file
  - params.json          : trial.params 그대로
  - README.md            : 사람이 읽을 수 있는 recipe + 적용 절차

이 디렉토리는 self-contained: 외부 사용자가 git pull 후 `bash apply.sh --dry-run`
으로 plan 을 검토하고 `bash apply.sh --apply` 한 줄로 자기 클러스터에 재배포 가능.
"""
from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lmtune.deploy.llmd_k8s import render_values_overlay
from lmtune.storage import DuckDBStore


@dataclass(slots=True)
class WinnerExportResult:
    out_dir: Path
    files: list[Path]
    study_id: str
    trial_id: str
    score: float | None
    adapter: str


def _load_template(name: str) -> str:
    """Load a template from src/bench/visualization/dashboard/templates/."""
    here = Path(__file__).resolve().parents[1] / "visualization" / "dashboard" / "templates"
    return (here / name).read_text(encoding="utf-8")


def _render_jinja(template_str: str, ctx: dict[str, Any]) -> str:
    try:
        from jinja2 import BaseLoader, Environment
    except ImportError as e:
        raise RuntimeError("jinja2 required: pip install jinja2") from e
    env = Environment(loader=BaseLoader(), autoescape=False, keep_trailing_newline=True)
    return env.from_string(template_str).render(**ctx)


def export_winner(
    study_id: str,
    *,
    db_path: str | Path,
    out_dir: str | Path,
    rank: int = 1,
    endpoint_yaml_path: Path | None = None,
) -> WinnerExportResult:
    """Export the top-`rank` trial of `study_id` to `out_dir/winner/`.

    Always writes: apply.sh + values-overlay.yaml + params.json + README.md.
    `out_dir` is created if missing.
    """
    store = DuckDBStore(Path(db_path))
    hdr = store.get_study(study_id)
    if not hdr:
        raise ValueError(f"study not found: {study_id}")
    (_, study_name, strategy, space_yaml_text, ep_slug, prof_slugs_json,
     metric_name, direction, status, created_at, finished_at, _notes) = hdr

    rows = store.top_trials(study_id, direction=direction, k=rank)
    if not rows or len(rows) < rank:
        raise ValueError(
            f"study {study_id} has only {len(rows)} completed trials; cannot export rank={rank}"
        )
    trial_id, seq, params_json, score, _status = rows[rank - 1]
    params: dict[str, Any] = json.loads(params_json)

    space_name = "unknown"
    try:
        space_dict = yaml.safe_load(space_yaml_text) or {}
        space_name = space_dict.get("name", "unknown")
    except Exception:
        pass

    # Determine adapter + endpoint path from study notes / endpoint slug heuristic.
    # The CLI passes --adapter on `bench search start`; we recover it via study notes
    # if recorded. Fall back to "local-vllm".
    adapter = "local-vllm"
    if endpoint_yaml_path is not None and endpoint_yaml_path.exists():
        try:
            ep_data = yaml.safe_load(endpoint_yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            ep_data = {}
        if "helmfile_overrides" in (ep_data.get("deployment") or {}):
            adapter = "llmd-k8s"
    else:
        ep_data = {}

    # Build the values overlay.
    helmfile_overrides: dict[str, Any] = {}
    if ep_data:
        # Apply trial params to a synthetic merged-endpoint view (engine_args overlay).
        merged = dict(ep_data)
        deployment = dict(merged.get("deployment") or {})
        engine_args = dict(deployment.get("engine_args") or {})
        engine_args.update({
            k: v for k, v in params.items()
            if k not in ("tp", "pp", "dp", "ep")
        })
        deployment["engine_args"] = engine_args
        merged["deployment"] = deployment
        helmfile_overrides = dict(deployment.get("helmfile_overrides") or {})
        release_names = helmfile_overrides.get("release_names") or [
            helmfile_overrides.get("release_name", "ms-phase1")
        ]
        overlay_dict = render_values_overlay(merged, release_names=release_names)
    else:
        # No endpoint YAML available — emit minimal placeholder overlay.
        merged = {"model": "<unknown>", "deployment": {"engine_args": params}}
        overlay_dict = render_values_overlay(merged, release_name="ms-phase1")

    # Output dir.
    target = Path(out_dir) / "winner"
    target.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    # 1. params.json
    params_path = target / "params.json"
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(params_path)

    # 2. values-overlay.yaml
    overlay_path = target / "values-overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay_dict, sort_keys=False), encoding="utf-8")
    files.append(overlay_path)

    # 3. apply.sh
    apply_ctx = {
        "study_id": study_id,
        "study_name": study_name,
        "trial_id": trial_id,
        "seq": seq,
        "score": f"{score:.2f}" if isinstance(score, (int, float)) else str(score),
        "params_json_pretty": json.dumps(params, indent=2, sort_keys=True),
        "created_at": created_at,
        "strategy": strategy,
        "direction": direction,
        "metric_name": metric_name,
        "endpoint_path": str(endpoint_yaml_path) if endpoint_yaml_path else "<set ENDPOINT env>",
        "adapter": adapter,
        "helmfile_root": helmfile_overrides.get("helmfile_root", ""),
        "helmfile_file": helmfile_overrides.get("helmfile_file", ""),
        "environment": helmfile_overrides.get("environment", "dev"),
        "selector": helmfile_overrides.get("selector", ""),
        "namespace": helmfile_overrides.get("namespace", "default"),
        "deployment_names": helmfile_overrides.get(
            "deployment_names", [helmfile_overrides.get("deployment_name", "ms-phase1")]
        ),
    }
    apply_text = _render_jinja(_load_template("winner_apply.sh.j2"), apply_ctx)
    apply_path = target / "apply.sh"
    apply_path.write_text(apply_text, encoding="utf-8")
    apply_path.chmod(apply_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    files.append(apply_path)

    # 4. README.md  — pull metrics for the winner trial
    metrics_summary: list[dict[str, str]] = []
    if hasattr(store, "get_trial_metrics"):
        metrics_dict = store.get_trial_metrics(trial_id)
        for metric, by_wl in sorted(metrics_dict.items()):
            for wl, val in sorted(by_wl.items(), key=lambda x: str(x[0])):
                metrics_summary.append({
                    "metric": metric,
                    "workload": wl or "-",
                    "value": f"{val:.2f}" if isinstance(val, (int, float)) else str(val),
                })
    readme_ctx = {
        **apply_ctx,
        "finished_at": finished_at,
        "metrics_summary": metrics_summary,
        "total_trials": len(store.list_trials(study_id)),
        "space_name": space_name,
        "winner_dir": str(target),
    }
    readme_text = _render_jinja(_load_template("winner_README.md.j2"), readme_ctx)
    readme_path = target / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    files.append(readme_path)

    return WinnerExportResult(
        out_dir=target,
        files=files,
        study_id=study_id,
        trial_id=trial_id,
        score=score,
        adapter=adapter,
    )
