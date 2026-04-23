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
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from bench.deploy.base import ApplyResult, DeploymentAdapter, HealthReport, merge_params_into_endpoint
from bench.deploy.health import probe_openai_models, warmup_one_token


log = logging.getLogger(__name__)


# Peer repo with the llm-d helmfile templates; override via env if needed.
DEFAULT_HELMFILE_ROOT = Path("/home/jinmoo/ml_ai/agentic/llm-distributed-inference")


@dataclass(slots=True)
class _K8sTarget:
    namespace: str = "default"
    release_name: str = "ms-phase1"
    deployment_name: str = "ms-phase1"


def render_values_overlay(
    endpoint_data: Mapping[str, Any],
    *,
    release_name: str = "ms-phase1",
) -> dict[str, Any]:
    """Build the minimal Helm values overlay from a merged endpoint dict.

    Shape matches the peer repo's `ms-phase1/values-*.yaml` schema (modelspec
    + vllmArgs). Only the keys we set via trials appear — everything else is
    inherited from the base template.
    """
    deployment = dict((endpoint_data.get("deployment") or {}))
    engine_args = dict(deployment.get("engine_args") or {})
    parallelism = dict(deployment.get("parallelism") or {})

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

    overlay: dict[str, Any] = {
        release_name: {
            "modelspec": {
                "modelArtifactUri": f"hf://{endpoint_data.get('model')}",
            },
            "vllmArgs": vllm_args,
        }
    }
    return overlay


class LLMDK8sAdapter(DeploymentAdapter):
    adapter_label = "llmd-k8s"

    def __init__(
        self,
        *,
        helmfile_root: str | Path | None = None,
        environment: str = "dev",
        selector: str = "name=ms-phase1",
        release_name: str = "ms-phase1",
        namespace: str = "default",
        deployment_name: str = "ms-phase1",
        rollout_timeout_s: int = 600,
        helmfile_file: str = "phase1/helmfile.yaml.gotmpl",
    ):
        self._root = Path(helmfile_root) if helmfile_root else DEFAULT_HELMFILE_ROOT
        self._env = environment
        self._selector = selector
        self._helmfile_file = helmfile_file
        self._target = _K8sTarget(
            namespace=namespace,
            release_name=release_name,
            deployment_name=deployment_name,
        )
        self._rollout_timeout_s = int(rollout_timeout_s)

    # ---- public API ----------------------------------------------------

    def apply(self, endpoint_path: str | Path, params: Mapping[str, Any]) -> ApplyResult:
        ep = Path(endpoint_path)
        data = merge_params_into_endpoint(ep, params)
        overlay = render_values_overlay(data, release_name=self._target.release_name)

        if not self._root.exists():
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail=f"helmfile root not found: {self._root}"),
                endpoint_path=ep, adapter=self.adapter_label,
                notes="peer repo unavailable",
            )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(overlay, tf, sort_keys=False)
            overlay_path = Path(tf.name)

        # 1. helmfile apply
        cmd = [
            "helmfile",
            "--environment", self._env,
            "--selector", self._selector,
            "--state-values-file", str(overlay_path),
            "-f", str(self._root / self._helmfile_file),
            "apply",
        ]
        log.info("LLMDK8sAdapter: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self._root), capture_output=True, text=True)
        if proc.returncode != 0:
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail=(proc.stderr or proc.stdout)[-800:]),
                endpoint_path=ep, adapter=self.adapter_label,
                notes=f"helmfile apply rc={proc.returncode}",
            )

        # 2. rollout status
        rollout = subprocess.run(
            ["kubectl", "-n", self._target.namespace, "rollout", "status",
             f"deployment/{self._target.deployment_name}",
             f"--timeout={self._rollout_timeout_s}s"],
            capture_output=True, text=True,
        )
        if rollout.returncode != 0:
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail=rollout.stderr.strip()[-400:]),
                endpoint_path=ep, adapter=self.adapter_label,
                notes="rollout failed",
            )

        # 3. probe + warmup
        url = (data.get("url") or "").strip()
        health = probe_openai_models(url) if url else HealthReport(ready=False, detail="no url")
        if health.ready:
            warmup_one_token(url, data.get("model", ""))
        return ApplyResult(
            ok=bool(health.ready),
            health=health, endpoint_path=ep, adapter=self.adapter_label,
        )

    def teardown(self, endpoint_path: str | Path) -> None:
        # Phase S4 we let `helmfile apply` of the next trial replace state.
        # For explicit teardown (between studies), run helmfile destroy — left
        # as a user-triggered operation to avoid accidentally wiping shared state.
        return None
