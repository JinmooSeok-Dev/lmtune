"""Dashboard build — DuckDB → InferenceX-compat JSON × 3 → Jinja2 → static HTML.

산출 (`b200/dashboards/`):
    index.html              ← 모든 study 카드, top score 매트릭스
    studies/<id>.html       ← study 별 상세
    compare.html            ← cross-study 비교
    data/studies_index.json         ← StudiesIndex (InferenceX-app 호환)
    data/throughput_vs_latency.json ← list[ThroughputVsLatency]
    data/perf_history.json          ← PerfHistory

dependency: jinja2 (default dep, package-data 포함). Tailwind/Chart.js는 CDN.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lmtune.storage import DuckDBStore
from lmtune.visualization.dashboard.schemas import (
    PerfHistory,
    PerfHistoryEntry,
    StudiesIndex,
    StudyCard,
    ThroughputVsLatency,
    TrialPoint,
)

SCHEMA_VERSION = "lmtune/dashboard/v1alpha1-inferencex-compat"


def _load_template(name: str) -> str:
    here = Path(__file__).resolve().parent / "templates"
    return (here / name).read_text(encoding="utf-8")


def _render(template_name: str, ctx: dict[str, Any]) -> str:
    try:
        from jinja2 import BaseLoader, Environment
    except ImportError as e:
        raise RuntimeError("jinja2 required: pip install jinja2") from e
    env = Environment(loader=BaseLoader(), autoescape=True, keep_trailing_newline=True)
    return env.from_string(_load_template(template_name)).render(**ctx)


def _infer_endpoint_meta(endpoint_slug: str | None) -> tuple[str | None, str | None, str | None]:
    """Heuristic (model_id, framework, hardware_id) from endpoint slug."""
    s = (endpoint_slug or "").lower()
    model_id: str | None = None
    framework: str | None = None
    hardware_id: str | None = None
    if "qwen25" in s or "qwen2.5" in s or "qwen2_5" in s:
        model_id = "Qwen2.5"
    elif "qwen3" in s:
        model_id = "Qwen3"
    elif "llama" in s:
        model_id = "Llama-3"
    elif "deepseek" in s:
        model_id = "DeepSeek"
    if "vllm" in s:
        framework = "vllm"
    elif "sglang" in s:
        framework = "sglang"
    elif "trt" in s:
        framework = "trt-llm"
    if "minikube" in s:
        hardware_id = "minikube"
    elif "rtx" in s or "local" in s:
        hardware_id = "RTX-local"
    elif "b200" in s:
        hardware_id = "B200"
    elif "h200" in s:
        hardware_id = "H200"
    elif "h100" in s:
        hardware_id = "H100"
    return model_id, framework, hardware_id


def _flatten_metrics(metrics_by_workload: dict[str, dict[str, float]]) -> dict[str, float]:
    """`{metric: {workload: value}}` → `{metric.workload: value}` flat dict."""
    flat: dict[str, float] = {}
    for metric, by_wl in metrics_by_workload.items():
        if not isinstance(by_wl, dict):
            continue
        for wl, v in by_wl.items():
            try:
                flat[f"{metric}.{wl}"] = float(v)
            except (TypeError, ValueError):
                continue
    return flat


def _pick_metric(metrics_flat: dict[str, float], metric: str, prefer_workloads: list[str]) -> float | None:
    """Pick first available `<metric>.<workload>` from preference list, else any.

    Supports two name shapes:
      1. canonical: `<metric>.<workload>` (e.g. `throughput_tok_avg.short`)
      2. suffix: `<metric>_<workload>.<anything>` (e.g. `throughput_avg_short.aggregate`)
    """
    # canonical: metric.workload
    for wl in prefer_workloads:
        v = metrics_flat.get(f"{metric}.{wl}")
        if v is not None:
            return v
    for k, v in metrics_flat.items():
        if k.startswith(f"{metric}."):
            return v
    # suffix: metric_workload.<anything>
    for wl in prefer_workloads:
        for k, v in metrics_flat.items():
            if k.startswith(f"{metric}_{wl}."):
                return v
    return None


def _resolve_throughput(metrics_flat: dict[str, float], prefer_workloads: list[str]) -> float | None:
    """Try canonical names, suffix names, and legacy aliases for throughput."""
    for name in ("throughput_tok_avg", "throughput_avg"):
        v = _pick_metric(metrics_flat, name, prefer_workloads)
        if v is not None:
            return v
    return None


def _annotate_trial(
    trial: TrialPoint,
    running_best: float | None,
    direction: str,
    strategy: str,
    n_startup: int,
) -> dict[str, Any]:
    """Derive a one-liner explainability tag for a trial.

    Pure function of past trial sequence + sampler strategy. No internal sampler
    state needed — annotates from the *outcome*, not the search call.
    """
    seq = trial.seq
    score = trial.score
    is_warmup = seq <= n_startup and strategy in {"tpe", "cma_es", "nsga2"}

    # phase label: warmup vs main loop
    if strategy == "random":
        phase = "random"
    elif strategy == "grid":
        phase = "grid"
    elif strategy == "lhc":
        phase = "LHC"
    elif is_warmup:
        phase = f"{strategy.upper()} warmup"
    else:
        phase = strategy.upper()

    # outcome label: improvement vs gap to best
    if score is None:
        outcome = "no score"
        delta_pct: float | None = None
    elif running_best is None:
        outcome = "first"
        delta_pct = 0.0
    else:
        better = (direction == "maximize" and score > running_best) or (
            direction == "minimize" and score < running_best
        )
        outcome = "🟢 new best" if better else "↳ exploring"
        denom = abs(running_best) if abs(running_best) > 1e-9 else 1.0
        delta_pct = ((score - running_best) / denom) * 100.0

    return {
        "phase": phase,
        "outcome": outcome,
        "delta_pct_vs_best": delta_pct,
    }


def _safe_axis_importance(
    points: list[TrialPoint],
    *,
    drop_threshold: float = 0.05,
) -> list[dict[str, Any]] | None:
    """RandomForest 기반 axis importance. sklearn 미설치/부족한 trial 시 None.

    return: list[{axis, importance, recommendation}] sorted by importance desc.
    """
    rows = [
        {
            "status": "completed" if p.score is not None else "incomplete",
            "score": p.score,
            "params": p.params,
        }
        for p in points
    ]
    try:
        from lmtune.search.analysis.importance import axis_importance as _axis_importance
    except Exception:
        return None
    try:
        out = _axis_importance(rows, drop_threshold=drop_threshold)
    except Exception:
        return None
    if not out:
        return None
    items = [
        {"axis": k, "importance": float(v["importance"]), "recommendation": v["recommendation"]}
        for k, v in out.items()
    ]
    items.sort(key=lambda r: r["importance"], reverse=True)
    return items


def _compute_pareto(
    points: list[TrialPoint],
    *,
    x_metric_candidates: tuple[str, ...] = ("ttft_p99",),
    y_metric_candidates: tuple[str, ...] = ("throughput_tok_avg", "throughput_avg"),
    workloads: tuple[str, ...] = ("short", "aggregate", "medium"),
) -> list[dict[str, Any]]:
    """Non-dominated front on (x↓, y↑). 빈 경우 [] 반환.

    각 dict: {x, y, seq, trial_id}. JS 가 그대로 overlay dataset 으로 쓸 수 있음.
    """
    pts: list[dict[str, Any]] = []
    for p in points:
        if p.score is None:
            continue
        m = p.metrics or {}
        x: float | None = None
        y: float | None = None
        for cand in x_metric_candidates:
            x = _pick_metric(m, cand, list(workloads))
            if x is not None:
                break
        for cand in y_metric_candidates:
            y = _pick_metric(m, cand, list(workloads))
            if y is not None:
                break
        if x is None or y is None:
            continue
        pts.append({"x": float(x), "y": float(y), "seq": p.seq, "trial_id": p.trial_id})
    if not pts:
        return []
    keep: list[bool] = [True] * len(pts)
    for i, a in enumerate(pts):
        if not keep[i]:
            continue
        for j, b in enumerate(pts):
            if i == j:
                continue
            # b dominates a iff b.x ≤ a.x and b.y ≥ a.y, with at least one strict
            if b["x"] <= a["x"] and b["y"] >= a["y"] and (b["x"] < a["x"] or b["y"] > a["y"]):
                keep[i] = False
                break
    front = [p for k, p in zip(keep, pts, strict=False) if k]
    front.sort(key=lambda p: p["x"])
    return front


def _count_axes_in_space_yaml(space_yaml: str | None) -> int | None:
    """Count axes from a SearchSpace YAML snapshot stored in `studies.space_yaml`."""
    if not space_yaml:
        return None
    try:
        spec = yaml.safe_load(space_yaml)
    except Exception:
        return None
    if not isinstance(spec, dict):
        return None
    axes = spec.get("axes")
    if isinstance(axes, dict):
        return len(axes)
    if isinstance(axes, list):
        return len(axes)
    return None


def _compute_axis_diff(
    points: list[TrialPoint],
    direction: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """For top-K trials by score, list which axes differ from the #1 winner.

    Returns rows: [{seq, score, diff: {axis: (winner_val, this_val)}}, ...]
    """
    completed = [p for p in points if p.score is not None]
    if not completed:
        return []
    completed.sort(key=lambda p: p.score, reverse=(direction == "maximize"))
    winner = completed[0]
    rows: list[dict[str, Any]] = []
    for p in completed[: top_k + 1]:
        diff: dict[str, tuple[Any, Any]] = {}
        for k, v in (p.params or {}).items():
            wv = (winner.params or {}).get(k)
            if wv != v:
                diff[k] = (wv, v)
        rows.append({
            "seq": p.seq,
            "trial_id": p.trial_id,
            "score": p.score,
            "is_winner": p.trial_id == winner.trial_id,
            "diff": diff,
        })
    return rows


@dataclass
class _StudyView:
    """Loose render context for Jinja2 templates (not serialized to JSON).

    Augments strict schemas with template-only fields (status per point,
    `throughput_avg` / `ttft_p99` / `e2e_p99` projected from the flat metrics dict,
    plus per-trial explainability annotations and axis-importance / Pareto front).
    """
    card: StudyCard
    tvl: ThroughputVsLatency
    point_status: dict[str, str] = field(default_factory=dict)  # trial_id -> status
    top_params: dict[str, Any] = field(default_factory=dict)
    n_pruned: int = 0
    metric_name: str = "total_score"
    model: str | None = None
    hw: str | None = None
    n_axes: int | None = None
    axis_importance: list[dict[str, Any]] | None = None
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    n_infeasible: int = 0   # pruned-by-feasibility (subset of n_pruned)

    def _annotate_points(self) -> dict[str, dict[str, Any]]:
        """Compute reasoning tags per trial in seq order."""
        # Sort by seq ascending to scan running best.
        ordered = sorted(self.tvl.points, key=lambda p: p.seq)
        # n_startup default for TPE — Optuna uses 10; we mirror that.
        n_startup = 10 if self.card.strategy == "tpe" else 0
        out: dict[str, dict[str, Any]] = {}
        running_best: float | None = None
        for p in ordered:
            tag = _annotate_trial(p, running_best, self.card.direction, self.card.strategy, n_startup)
            out[p.trial_id] = tag
            if p.score is not None and (
                running_best is None
                or (self.card.direction == "maximize" and p.score > running_best)
                or (self.card.direction == "minimize" and p.score < running_best)
            ):
                running_best = p.score
        return out

    def to_dict(self) -> dict[str, Any]:
        d = self.card.model_dump()
        annotations = self._annotate_points()

        def _pt_view(p: TrialPoint) -> dict[str, Any]:
            base = p.model_dump()
            metrics = base.get("metrics") or {}
            base["status"] = self.point_status.get(p.trial_id, "completed")
            wl_pref = ["short", "aggregate", "medium"]
            base["throughput_avg"] = _resolve_throughput(metrics, wl_pref)
            base["ttft_p99"] = _pick_metric(metrics, "ttft_p99", wl_pref)
            base["e2e_p99"] = _pick_metric(metrics, "e2e_p99", wl_pref)
            base["cv"] = _pick_metric(metrics, "cv_throughput", wl_pref)
            base["reasoning"] = annotations.get(p.trial_id, {})
            return base

        d.update(
            top_params=self.top_params,
            n_pruned=self.n_pruned,
            n_infeasible=self.n_infeasible,
            metric_name=self.metric_name,
            model=self.model or self.tvl.model_id,
            hw=self.hw or self.tvl.hardware_id,
            framework=self.tvl.framework,
            workload=self.tvl.workload,
            points=[_pt_view(p) for p in self.tvl.points],
            axis_diff=_compute_axis_diff(self.tvl.points, self.card.direction),
            n_axes=self.n_axes,
            axis_importance=self.axis_importance or [],
            pareto_front=self.pareto_front,
        )
        return d


def _build_study(store: DuckDBStore, study_id: str) -> _StudyView | None:
    hdr = store.get_study(study_id)
    if not hdr:
        return None
    (sid, name, strategy, space_yaml, ep_slug, prof_slugs, metric, direction,
     status, created_at, finished_at, _notes) = hdr
    # DuckDB stores profile_slugs as a JSON string or a native list depending on version
    if isinstance(prof_slugs, str):
        try:
            profile_slugs = list(json.loads(prof_slugs))
        except (TypeError, ValueError, json.JSONDecodeError):
            profile_slugs = []
    elif prof_slugs:
        profile_slugs = list(prof_slugs)
    else:
        profile_slugs = []

    trials = store.list_trials(study_id)
    completed = [t for t in trials if t[3] == "completed"]
    pruned = [t for t in trials if t[3] == "pruned"]
    n_infeasible = 0
    for t in pruned:
        # trials tuple: (trial_id, seq, params_json, status, score, completed_at, backend, err)
        err = t[7] if len(t) > 7 else None
        if err and ("FAIL: c" in str(err) or "infeasible" in str(err).lower()):
            n_infeasible += 1

    points: list[TrialPoint] = []
    point_status: dict[str, str] = {}
    for trial_id, seq, params_json, t_status, score, _comp_at, _backend, _err in trials:
        params = json.loads(params_json) if params_json else {}
        m = store.get_trial_metrics(trial_id)
        flat = _flatten_metrics(m)
        points.append(TrialPoint(
            trial_id=trial_id,
            seq=int(seq),
            score=float(score) if score is not None else None,
            params=params,
            metrics=flat,
        ))
        point_status[trial_id] = str(t_status)

    top = store.top_trials(study_id, direction=direction, k=1)
    top_score = float(top[0][3]) if top else None
    top_params: dict[str, Any] = json.loads(top[0][2]) if top else {}

    model_id, framework, hardware_id = _infer_endpoint_meta(ep_slug)

    card = StudyCard(
        study_id=sid,
        name=name,
        strategy=strategy,
        direction=direction,
        status=status,
        n_trials=len(trials),
        n_completed=len(completed),
        top_score=top_score,
        endpoint_slug=ep_slug,
        profile_slugs=profile_slugs,
        created_at=str(created_at) if created_at else None,
        finished_at=str(finished_at) if finished_at else None,
    )
    tvl = ThroughputVsLatency(
        study_id=sid,
        model_id=model_id,
        framework=framework,
        hardware_id=hardware_id,
        workload=profile_slugs[0] if profile_slugs else None,
        points=points,
    )
    return _StudyView(
        card=card, tvl=tvl,
        point_status=point_status,
        top_params=top_params, n_pruned=len(pruned),
        n_infeasible=n_infeasible,
        metric_name=metric or "total_score",
        n_axes=_count_axes_in_space_yaml(space_yaml),
        axis_importance=_safe_axis_importance(points),
        pareto_front=_compute_pareto(points),
    )


def _load_perf_history(perf_changelog_path: Path) -> PerfHistory:
    if not perf_changelog_path.exists():
        return PerfHistory()
    try:
        raw = yaml.safe_load(perf_changelog_path.read_text(encoding="utf-8"))
    except Exception:
        return PerfHistory()
    # supports both legacy list-of-entries and new {entries: [...]} envelope
    if isinstance(raw, dict) and "entries" in raw:
        items = raw["entries"] or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    entries: list[PerfHistoryEntry] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("landed_at") or entry.get("timestamp")
        entries.append(PerfHistoryEntry(
            config_keys=list(entry.get("config-keys") or entry.get("config_keys") or []),
            description=list(entry.get("description") or []),
            pr_link=entry.get("pr-link") or entry.get("pr_link"),
            evals_only=bool(entry.get("evals-only", entry.get("evals_only", False))),
            landed_at=str(ts) if ts is not None else None,
        ))
    return PerfHistory(entries=entries)


def dump_inferencex_json(
    *,
    studies_index: StudiesIndex,
    throughput_vs_latency: list[ThroughputVsLatency],
    perf_history: PerfHistory,
    out_dir: Path,
) -> dict[str, Path]:
    """Emit the 3 InferenceX-app compatible JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    p = out_dir / "studies_index.json"
    p.write_text(json.dumps(studies_index.model_dump(), indent=2, default=str), encoding="utf-8")
    written["data/studies_index.json"] = p

    p = out_dir / "throughput_vs_latency.json"
    p.write_text(
        json.dumps([t.model_dump() for t in throughput_vs_latency], indent=2, default=str),
        encoding="utf-8",
    )
    written["data/throughput_vs_latency.json"] = p

    p = out_dir / "perf_history.json"
    p.write_text(json.dumps(perf_history.model_dump(), indent=2, default=str), encoding="utf-8")
    written["data/perf_history.json"] = p

    return written


def _summarize_axis(name: str, spec: Any) -> dict[str, Any]:
    """Best-effort summary of one axis entry. spec 은 dict 또는 None."""
    if not isinstance(spec, dict):
        return {"name": name, "type": "?", "values": "?", "active_if": None, "cost_tier": None,
                "apply_via": None, "doc": None}
    kind = spec.get("type") or spec.get("kind") or "?"
    values: Any = "—"
    if kind in ("categorical",) and "values" in spec:
        values = spec["values"]
    elif kind in ("int", "float", "logfloat", "logloguniform"):
        lo = spec.get("low")
        hi = spec.get("high")
        values = f"[{lo}, {hi}]"
    elif kind in ("bool",):
        values = "[true, false]"
    elif "values" in spec:
        values = spec["values"]
    if isinstance(values, list) and len(values) > 6:
        values = values[:6] + ["…"]
    return {
        "name": name,
        "type": kind,
        "values": values,
        "active_if": spec.get("active_if"),
        "cost_tier": spec.get("cost_tier"),
        "apply_via": spec.get("apply_via"),
        "doc": spec.get("doc") or spec.get("description"),
    }


def _summarize_search_space(path: Path) -> dict[str, Any] | None:
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(spec, dict):
        return None
    axes_raw = spec.get("axes") or {}
    axes: list[dict[str, Any]] = []
    if isinstance(axes_raw, dict):
        for n, s in axes_raw.items():
            axes.append(_summarize_axis(n, s))
    elif isinstance(axes_raw, list):
        for entry in axes_raw:
            if isinstance(entry, dict) and "name" in entry:
                axes.append(_summarize_axis(entry["name"], entry))
    sections = spec.get("sections")
    n_constraints = 0
    if isinstance(spec.get("feasibility_constraints"), list):
        n_constraints = len(spec["feasibility_constraints"])
    return {
        "path": str(path),
        "name": spec.get("name") or path.stem,
        "description": spec.get("description"),
        "n_axes": len(axes),
        "n_constraints": n_constraints,
        "sections": sections if isinstance(sections, dict) else None,
        "axes": axes,
        "default_pruner": spec.get("default_pruner"),
        "default_objectives": spec.get("default_objectives"),
        "profile_catalog": spec.get("profile_catalog"),
    }


def _summarize_env_profile(path: Path) -> dict[str, Any] | None:
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(spec, dict):
        return None
    locked = spec.get("env_locked") or {}
    tunable_raw = spec.get("env_tunable") or []
    tunable: list[dict[str, Any]] = []
    if isinstance(tunable_raw, list):
        for entry in tunable_raw:
            if isinstance(entry, dict):
                tunable.append({
                    "name": entry.get("name"),
                    "kind": entry.get("kind"),
                    "values": entry.get("values"),
                    "cost_tier": entry.get("cost_tier"),
                })
    return {
        "path": str(path),
        "name": spec.get("name") or path.stem,
        "priority": spec.get("priority"),
        "description": spec.get("description"),
        "applies_when": spec.get("applies_when"),
        "n_locked": len(locked) if isinstance(locked, dict) else 0,
        "n_tunable": len(tunable),
        "env_locked": locked if isinstance(locked, dict) else {},
        "env_tunable": tunable,
    }


def _build_matrix_view(views: list[_StudyView]) -> dict[str, Any]:
    """Aggregate study views into a (model × hw) matrix for InferenceX-style heatmap.

    Each cell = best (top-score) study for that (model, hw). When multiple
    studies exist per cell, prefer the highest score. Per-framework split kept
    as separate datasets for the front-end.
    """
    rows_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    models_seen: list[str] = []
    hw_seen: list[str] = []
    framework_seen: list[str] = []

    for v in views:
        model = v.model or v.tvl.model_id or "(unknown)"
        hw = v.hw or v.tvl.hardware_id or v.card.endpoint_slug or "(unknown)"
        framework = v.tvl.framework or "(unknown)"
        if model not in models_seen:
            models_seen.append(model)
        if hw not in hw_seen:
            hw_seen.append(hw)
        if framework not in framework_seen:
            framework_seen.append(framework)

        # cell payload
        wl_pref = ["short", "aggregate", "medium"]
        ttft = None
        tput = None
        for p in v.tvl.points:
            if p.score == v.card.top_score and p.metrics:
                ttft = _pick_metric(p.metrics, "ttft_p99", wl_pref)
                tput = _resolve_throughput(p.metrics, wl_pref)
                break
        cell = {
            "study_id": v.card.study_id,
            "study_name": v.card.name,
            "framework": framework,
            "score": v.card.top_score,
            "ttft_p99": ttft,
            "throughput": tput,
            "n_completed": v.card.n_completed,
            "strategy": v.card.strategy,
        }
        existing = rows_by_model.setdefault(model, {}).get(hw)
        if existing is None or (
            v.card.top_score is not None and existing["score"] is not None
            and v.card.top_score > existing["score"]
        ) or (existing["score"] is None and v.card.top_score is not None):
            rows_by_model[model][hw] = cell

    return {
        "models": models_seen,
        "hardware": hw_seen,
        "frameworks": framework_seen,
        "cells": rows_by_model,    # cells[model][hw] = cell
        "n_filled": sum(len(row) for row in rows_by_model.values()),
        "n_total": len(models_seen) * len(hw_seen),
    }


def _build_spaces_page(
    *,
    search_spaces_dir: Path | None,
    env_profiles_dir: Path | None,
) -> dict[str, Any]:
    spaces: list[dict[str, Any]] = []
    if search_spaces_dir and search_spaces_dir.exists():
        for f in sorted(search_spaces_dir.glob("*.yaml")):
            s = _summarize_search_space(f)
            if s:
                spaces.append(s)
    profiles: list[dict[str, Any]] = []
    if env_profiles_dir and env_profiles_dir.exists():
        for f in sorted(env_profiles_dir.glob("*.yaml")):
            p = _summarize_env_profile(f)
            if p:
                profiles.append(p)
    profiles.sort(key=lambda p: (p.get("priority") if p.get("priority") is not None else 99))
    total_axes = sum(s["n_axes"] for s in spaces)
    return {
        "spaces": spaces,
        "profiles": profiles,
        "total_axes": total_axes,
        "n_spaces": len(spaces),
        "n_profiles": len(profiles),
    }


def build_dashboard(
    *,
    db_path: str | Path,
    out_dir: str | Path,
    perf_changelog: str | Path | None = None,
    study_ids: list[str] | None = None,
    search_spaces_dir: str | Path | None = None,
    env_profiles_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate the static dashboard. Returns paths of all written files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "studies").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)

    store = DuckDBStore(Path(db_path))
    if study_ids is None:
        rows = store.list_studies(limit=200)
        study_ids = [r[0] for r in rows]

    views: list[_StudyView] = []
    for sid in study_ids:
        v = _build_study(store, sid)
        if v is not None:
            views.append(v)

    studies_index = StudiesIndex(studies=[v.card for v in views])
    tvl_list = [v.tvl for v in views]
    perf = _load_perf_history(Path(perf_changelog)) if perf_changelog else PerfHistory()

    written: dict[str, Path] = {}

    # 1. JSON × 3 (InferenceX-compat)
    written.update(dump_inferencex_json(
        studies_index=studies_index,
        throughput_vs_latency=tvl_list,
        perf_history=perf,
        out_dir=out / "data",
    ))

    # 2. render context for templates (loose view that includes derived fields)
    studies_view = [v.to_dict() for v in views]
    perf_view = [e.model_dump() for e in perf.entries]

    # spaces.html — search-space + env-profile 카탈로그 (옵션, 둘 다 None 이면 skip)
    auto_repo = Path.cwd()
    ss_dir = Path(search_spaces_dir) if search_spaces_dir else (auto_repo / "b200" / "search-spaces")
    ep_dir = Path(env_profiles_dir) if env_profiles_dir else (auto_repo / "configs" / "autoresearch" / "env_profiles")
    spaces_ctx = _build_spaces_page(search_spaces_dir=ss_dir, env_profiles_dir=ep_dir)
    has_spaces = spaces_ctx["n_spaces"] > 0 or spaces_ctx["n_profiles"] > 0

    # 3. index.html
    index_html = _render("index.html.j2", {
        "schema_version": SCHEMA_VERSION,
        "n_studies": len(views),
        "has_spaces": has_spaces,
        "spaces_summary": {
            "n_spaces": spaces_ctx["n_spaces"],
            "n_profiles": spaces_ctx["n_profiles"],
            "total_axes": spaces_ctx["total_axes"],
        },
        "data": {
            "schema_version": SCHEMA_VERSION,
            "studies": studies_view,
            "perf_history": perf_view,
        },
    })
    p = out / "index.html"
    p.write_text(index_html, encoding="utf-8")
    written["index.html"] = p

    # 4. studies/<id>.html
    for v in views:
        ctx = v.to_dict()
        study_html = _render("study.html.j2", {
            "study": ctx,
            "study_json": json.dumps(ctx, indent=2, default=str),
        })
        sp = out / "studies" / f"{v.card.study_id}.html"
        sp.write_text(study_html, encoding="utf-8")
        written[f"studies/{v.card.study_id}.html"] = sp

    # 5. compare.html
    compare_html = _render("compare.html.j2", {"studies": studies_view, "perf_history": perf_view})
    cp = out / "compare.html"
    cp.write_text(compare_html, encoding="utf-8")
    written["compare.html"] = cp

    # 6. spaces.html (옵션)
    if has_spaces:
        spaces_html = _render("spaces.html.j2", spaces_ctx)
        sp = out / "spaces.html"
        sp.write_text(spaces_html, encoding="utf-8")
        written["spaces.html"] = sp

    # 7. matrix.html — InferenceX-style 모델 × HW heatmap
    matrix_ctx = _build_matrix_view(views)
    matrix_html = _render("matrix.html.j2", {
        "matrix": matrix_ctx,
        "n_studies": len(views),
    })
    mp = out / "matrix.html"
    mp.write_text(matrix_html, encoding="utf-8")
    written["matrix.html"] = mp

    return written
