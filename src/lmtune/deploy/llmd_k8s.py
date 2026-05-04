"""LLMDK8sAdapter — apply a (params, endpoint_yaml) pair to an llm-d helmfile.

Flow:
1. Merge params into endpoint YAML (engine_args + parallelism).
2. Render a Helm values-override file from the merged ParallelismSpec +
   engine_args.  The peer repo at `HELMFILE_ROOT` owns the base templates;
   we only write the overlay values.
3. `helmfile apply -f phase1/helmfile.yaml.gotmpl --state-values-file <overlay>`
4. `kubectl rollout status deployment/<name>` up to `rollout_timeout_s`.
5. HTTP probe on the endpoint URL.

For unit testing without a cluster: `render_values_overlay(...)` returns the
dict; tests snapshot-compare it. Live smoke requires minikube + helmfile
(Phase S0-C prerequisite).
"""

from __future__ import annotations

import logging
import os as _os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lmtune.deploy.base import (
    ApplyResult,
    DeploymentAdapter,
    HealthReport,
    merge_params_into_endpoint,
)
from lmtune.deploy.health import probe_openai_models, warmup_one_token

log = logging.getLogger(__name__)


# lmtune repo root — auto-detected from this file's location (src/lmtune/deploy/).
# 우선순위: LMTUNE_REPO_ROOT env > __file__ 상대 (이게 거의 모든 정상 설치/CI/worktree 에서 동작).
# 호스트 무관 — 이전엔 절대경로 hardcode 였음 (회귀 방지: 어떤 endpoint YAML 도 absolute path 박지 말 것).
DEFAULT_LMTUNE_REPO_ROOT = Path(
    _os.environ.get("LMTUNE_REPO_ROOT") or Path(__file__).resolve().parents[3]
)

# helmfile templates root — b200/helmfile/* 가 lmtune repo 안에 self-contained 이므로
# 디폴트는 lmtune repo root. peer-repo (llm-distributed-inference) 가 필요한 경우만 override.
# 우선순위: 명시 인자 > endpoint YAML helmfile_overrides.helmfile_root > LMTUNE_HELMFILE_ROOT env > 자동
DEFAULT_HELMFILE_ROOT = Path(
    _os.environ.get("LMTUNE_HELMFILE_ROOT") or DEFAULT_LMTUNE_REPO_ROOT
)


# Well-lit-path 디스패치 테이블. b200/helmfile/<key>/ 가 기준 (self-contained).
# README placeholder 만 있는 path 는 본 디스패치에서 제외 — 재시도 시 명확한 에러.
WELL_LIT_PATHS: dict[str, dict[str, str]] = {
    "inference-scheduling": {
        "helmfile_file": "b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl",
        "default_namespace": "b200-infsch",
    },
    "pd-disaggregation": {
        "helmfile_file": "b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl",
        "default_namespace": "b200-pd",
    },
    "wide-ep-lws": {
        "helmfile_file": "b200/helmfile/wide-ep-lws/helmfile.yaml.gotmpl",
        "default_namespace": "b200-wideep",
    },
}


class UnsupportedWellLitPath(ValueError):
    """Sampled `well_lit_path` not yet implemented (only README placeholder)."""


def resolve_well_lit_path(
    path_name: str,
    *,
    bench_repo_root: Path | None = None,
) -> tuple[Path, str]:
    """Map a `well_lit_path` axis value → (helmfile_root, helmfile_file).

    Raises `UnsupportedWellLitPath` if the requested path has no working
    helmfile in `b200/helmfile/<name>/` (only a README placeholder).
    """
    if path_name not in WELL_LIT_PATHS:
        raise UnsupportedWellLitPath(
            f"well_lit_path={path_name!r} is not autotune-driveable yet. "
            f"Available: {sorted(WELL_LIT_PATHS)}. "
            "(tiered-prefix-cache / precise-prefix-cache / "
            "predicted-latency-scheduling / workload-autoscaling 은 helmfile 미작성.)"
        )
    root = Path(bench_repo_root) if bench_repo_root else DEFAULT_LMTUNE_REPO_ROOT
    return root, WELL_LIT_PATHS[path_name]["helmfile_file"]


@dataclass(slots=True)
class _K8sTarget:
    namespace: str = "default"
    release_name: str = "ms-phase1"
    deployment_name: str = "ms-phase1"


def render_values_overlay(
    endpoint_data: Mapping[str, Any],
    *,
    release_name: str | None = None,
    release_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the minimal Helm values overlay from a merged endpoint dict.

    Shape matches the peer repo's `ms-{phase}/values-*.yaml` schema (modelspec
    + vllmArgs). Only the keys we set via trials appear — everything else is
    inherited from the base template.

    For multi-release helmfile (e.g. P/D disaggregation = ms-pd-prefill +
    ms-pd-decode), pass `release_names=[...]` and the same vllmArgs are emitted
    for each. Single release: legacy `release_name="..."` still works.
    """
    deployment = dict(endpoint_data.get("deployment") or {})
    engine_args = dict(deployment.get("engine_args") or {})
    parallelism = dict(deployment.get("parallelism") or {})
    replicas = dict(deployment.get("replicas") or {})

    # Dict-to-CLI-flag shape: values-*.yaml uses kebab-case keys.
    vllm_args: dict[str, Any] = {}
    for k, v in engine_args.items():
        vllm_args[k.replace("_", "-")] = v

    # Parallelism → typical vLLM flags (llm-d maps these onto Deployment replicas
    # and --tensor-parallel-size etc.)
    if "tp" in parallelism:
        vllm_args["tensor-parallel-size"] = int(parallelism["tp"])
    if "pp" in parallelism:
        vllm_args["pipeline-parallel-size"] = int(parallelism["pp"])
    if "dp" in parallelism:
        vllm_args["data-parallel-size"] = int(parallelism["dp"])
    if parallelism.get("ep"):
        vllm_args["enable-expert-parallel"] = True

    # Resolve target releases.
    if release_names:
        targets = list(release_names)
    elif release_name:
        targets = [release_name]
    else:
        targets = ["ms-phase1"]

    payload: dict[str, Any] = {
        "modelspec": {
            "modelArtifactUri": f"hf://{endpoint_data.get('model')}",
        },
        "vllmArgs": vllm_args,
    }
    # P/D disaggregation: replicas.prefill / replicas.decode → helmfile values 의
    # prefill.replicas / decode.replicas 로 분리 emit. base helmfile values yaml 의
    # parallelism.tensor 등 다른 prefill/decode 필드는 그대로 상속.
    if "prefill" in replicas:
        payload["prefill"] = {"replicas": int(replicas["prefill"])}
    if "decode" in replicas:
        payload["decode"] = {"replicas": int(replicas["decode"])}
    return {name: payload for name in targets}


class LLMDK8sAdapter(DeploymentAdapter):
    adapter_label = "llmd-k8s"

    def __init__(
        self,
        *,
        helmfile_root: str | Path | None = None,
        environment: str = "dev",
        selector: str = "name=ms-phase1",
        release_name: str = "ms-phase1",
        release_names: list[str] | None = None,
        namespace: str = "default",
        deployment_name: str = "ms-phase1",
        deployment_names: list[str] | None = None,
        rollout_timeout_s: int = 600,
        helmfile_file: str = "phase1/helmfile.yaml.gotmpl",
        dry_run: bool = False,
    ):
        self._root = Path(helmfile_root) if helmfile_root else DEFAULT_HELMFILE_ROOT
        self._env = environment
        self._selector = selector
        self._helmfile_file = helmfile_file
        # Multi-release P/D 지원: release_names 가 우선, 없으면 release_name 단일.
        self._release_names = list(release_names) if release_names else [release_name]
        self._deployment_names = list(deployment_names) if deployment_names else [deployment_name]
        self._target = _K8sTarget(
            namespace=namespace,
            release_name=release_name,
            deployment_name=deployment_name,
        )
        self._rollout_timeout_s = int(rollout_timeout_s)
        self._dry_run = bool(dry_run)

    @classmethod
    def from_endpoint(cls, endpoint_data: Mapping[str, Any], *, dry_run: bool = False) -> LLMDK8sAdapter:
        """Construct from endpoint YAML's `deployment.helmfile_overrides` block.

        Accepts (all optional):
          deployment.helmfile_overrides:
            helmfile_root: <path>
            helmfile_file: phase2/helmfile.yaml.gotmpl
            environment: dev
            selector: name=ms-pd
            namespace: llm-d-pd-qwen25
            release_names: [ms-pd-prefill, ms-pd-decode]
            deployment_names: [ms-pd-prefill, ms-pd-decode]
            rollout_timeout_s: 600
        """
        deployment = dict(endpoint_data.get("deployment") or {})
        ov = dict(deployment.get("helmfile_overrides") or {})
        return cls(
            helmfile_root=ov.get("helmfile_root"),
            environment=ov.get("environment", "dev"),
            selector=ov.get("selector", "name=ms-phase1"),
            release_name=ov.get("release_name", "ms-phase1"),
            release_names=ov.get("release_names"),
            namespace=ov.get("namespace", "default"),
            deployment_name=ov.get("deployment_name", "ms-phase1"),
            deployment_names=ov.get("deployment_names"),
            rollout_timeout_s=int(ov.get("rollout_timeout_s", 600)),
            helmfile_file=ov.get("helmfile_file", "phase1/helmfile.yaml.gotmpl"),
            dry_run=dry_run,
        )

    # ---- public API ----------------------------------------------------

    def apply(self, endpoint_path: str | Path, params: Mapping[str, Any]) -> ApplyResult:
        ep = Path(endpoint_path)

        # well_lit_path is a meta-axis; must NOT bleed into engine_args. Strip it
        # before merging, then resolve per-trial helmfile routing from it.
        params_clean = {k: v for k, v in dict(params).items() if k != "well_lit_path"}
        sampled_path = params.get("well_lit_path") if isinstance(params, Mapping) else None

        data = merge_params_into_endpoint(ep, params_clean)
        overlay = render_values_overlay(
            data, release_names=self._release_names,
        )

        # Path-aware routing: if a well_lit_path was sampled, override the static
        # (helmfile_root, helmfile_file) to point at b200/helmfile/<path>/.
        if sampled_path:
            try:
                helmfile_root, helmfile_file = resolve_well_lit_path(str(sampled_path))
            except UnsupportedWellLitPath as e:
                return ApplyResult(
                    ok=False,
                    health=HealthReport(ready=False, detail=str(e)),
                    endpoint_path=ep, adapter=self.adapter_label,
                    notes="unsupported well_lit_path",
                )
        else:
            helmfile_root, helmfile_file = self._root, self._helmfile_file

        # Always write the overlay (useful for inspection + dry-run + winner export).
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(overlay, tf, sort_keys=False)
            overlay_path = Path(tf.name)
        log.info("LLMDK8sAdapter overlay: %s", overlay_path)

        if self._dry_run:
            return ApplyResult(
                ok=True,
                health=HealthReport(ready=True, detail=f"dry-run; overlay at {overlay_path}"),
                endpoint_path=ep, adapter=self.adapter_label,
                notes="dry-run skipped helmfile/rollout/probe",
            )

        if not helmfile_root.exists():
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail=f"helmfile root not found: {helmfile_root}"),
                endpoint_path=ep, adapter=self.adapter_label,
                notes="repo unavailable",
            )

        # 1. helmfile apply
        cmd = [
            "helmfile",
            "--environment", self._env,
            "--selector", self._selector,
            "--state-values-file", str(overlay_path),
            "-f", str(helmfile_root / helmfile_file),
            "apply",
        ]
        log.info("LLMDK8sAdapter: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(helmfile_root), capture_output=True, text=True)
        if proc.returncode != 0:
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail=(proc.stderr or proc.stdout)[-800:]),
                endpoint_path=ep, adapter=self.adapter_label,
                notes=f"helmfile apply rc={proc.returncode}",
            )

        # 2. rollout status — for every deployment in the release set
        for dep in self._deployment_names:
            rollout = subprocess.run(
                ["kubectl", "-n", self._target.namespace, "rollout", "status",
                 f"deployment/{dep}",
                 f"--timeout={self._rollout_timeout_s}s"],
                capture_output=True, text=True,
            )
            if rollout.returncode != 0:
                return ApplyResult(
                    ok=False,
                    health=HealthReport(ready=False, detail=f"{dep}: " + rollout.stderr.strip()[-400:]),
                    endpoint_path=ep, adapter=self.adapter_label,
                    notes=f"rollout failed for {dep}",
                )

        # 3. probe + warmup
        # k8s rollout status 가 Ready 라고 해도 vllm 컨테이너 안에서 모델 로딩이
        # 60~120s 더 걸린다. /v1/models 가 떠야 진짜 serving 가능. probe 단발은
        # RemoteDisconnected 로 깨지므로 budget 내 retry/backoff.
        url = (data.get("url") or "").strip()
        if not url:
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail="no url"),
                endpoint_path=ep, adapter=self.adapter_label,
            )
        import time as _time
        probe_budget_s = max(60, self._rollout_timeout_s)
        deadline = _time.time() + probe_budget_s
        health = HealthReport(ready=False, detail="probe not started")
        attempt = 0
        while _time.time() < deadline:
            attempt += 1
            health = probe_openai_models(url)
            if health.ready:
                break
            _time.sleep(min(5.0, max(1.0, 0.5 * attempt)))
        if not health.ready:
            return ApplyResult(
                ok=False,
                health=health,
                endpoint_path=ep, adapter=self.adapter_label,
                notes=f"probe failed after {attempt} attempts within {probe_budget_s}s",
            )
        warmup_one_token(url, data.get("model", ""))
        return ApplyResult(
            ok=True,
            health=health, endpoint_path=ep, adapter=self.adapter_label,
        )

    def teardown(self, endpoint_path: str | Path) -> None:
        # Phase S4 we let `helmfile apply` of the next trial replace state.
        # For explicit teardown (between studies), run helmfile destroy — left
        # as a user-triggered operation to avoid accidentally wiping shared state.
        return None
