"""K8sJobBackend — one Kubernetes Job per trial.

Submits a `batch/v1 Job` templated from `k8s/trial-job-template.yaml` with
environment variables carrying params / endpoint / profiles. Worker image is
`bench-trial-runner:latest` (built from docker/Dockerfile.trial_runner); its
ENTRYPOINT is `python -m bench.orchestrate.trial_runner`, which prints a
single JSON line on stdout.

poll() reads Job status; when complete it tail-logs the pod and parses the
last stdout line as a TrialResult. The Job is left on-cluster with
`ttlSecondsAfterFinished` until the TTL GC removes it.

Phase S3 ships the plumbing (template + manifest + backend). A full smoke
requires minikube + the built image; wire-up in S4.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from lmtune.orchestrate.backend import (
    TrialBackend,
    TrialHandle,
    TrialPayload,
    TrialResult,
)

log = logging.getLogger(__name__)


_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "k8s" / "trial-job-template.yaml"


def render_job_manifest(
    payload: TrialPayload,
    *,
    image: str,
    namespace: str = "default",
    gpu_count: int = 1,
    ttl_seconds: int = 600,
) -> dict[str, Any]:
    """Return the rendered Job manifest dict (no cluster call).
    Tests call this to snapshot-check params without needing a cluster."""
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(_TEMPLATE_PATH)
    tpl = yaml.safe_load(_TEMPLATE_PATH.read_text(encoding="utf-8"))

    name = f"bench-trial-{payload.trial_id.lower()}"
    tpl["metadata"]["name"] = name
    tpl["metadata"].setdefault("labels", {})
    tpl["metadata"]["labels"].update(
        {
            "bench/study-id": payload.study_id,
            "bench/trial-id": payload.trial_id,
        }
    )
    tpl["metadata"]["namespace"] = namespace
    tpl["spec"]["ttlSecondsAfterFinished"] = int(ttl_seconds)

    container = tpl["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image
    env = {e["name"]: e for e in container.get("env", [])}

    def _set(k, v):
        env[k] = {"name": k, "value": str(v)}

    _set("TRIAL_ID", payload.trial_id)
    _set("STUDY_ID", payload.study_id)
    _set("TRIAL_SEQ", payload.seq)
    _set("PARAMS_JSON", json.dumps(payload.params, sort_keys=True))
    _set("ENDPOINT_PATH", payload.endpoint_path)
    _set("PROFILE_PATHS", ":".join(payload.profile_paths))
    _set("REPEATS", payload.repeats)
    _set("TTFT_SLO_MS", payload.ttft_slo_ms)
    container["env"] = list(env.values())

    resources = container.setdefault("resources", {})
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})
    if gpu_count > 0:
        requests["nvidia.com/gpu"] = str(gpu_count)
        limits["nvidia.com/gpu"] = str(gpu_count)

    return tpl


class K8sJobBackend(TrialBackend):
    """Submit one K8s Job per trial, poll status, parse stdout log for JSON."""

    name = "k8s-job"

    def __init__(
        self,
        *,
        workers: int = 4,
        image: str = "bench-trial-runner:latest",
        namespace: str = "default",
        kubeconfig: str | None = None,
        gpu_count: int = 1,
        ttl_seconds: int = 600,
    ):
        from kubernetes import client, config

        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
        self._batch = client.BatchV1Api()
        self._core = client.CoreV1Api()
        self._workers = int(workers)
        self._image = image
        self._ns = namespace
        self._gpu = int(gpu_count)
        self._ttl = int(ttl_seconds)

    # ---- TrialBackend --------------------------------------------------

    def submit(self, payload: TrialPayload) -> TrialHandle:
        manifest = render_job_manifest(
            payload,
            image=self._image,
            namespace=self._ns,
            gpu_count=self._gpu,
            ttl_seconds=self._ttl,
        )
        job_name = manifest["metadata"]["name"]
        self._batch.create_namespaced_job(body=manifest, namespace=self._ns)
        return TrialHandle(trial_id=payload.trial_id, backend=self.name, ref=job_name)

    def poll(self, handle: TrialHandle, timeout_s: float | None = None) -> TrialResult | None:
        job = self._batch.read_namespaced_job_status(name=handle.ref, namespace=self._ns)
        s = job.status
        done = (s.succeeded or 0) > 0 or (s.failed or 0) > 0
        if not done:
            if timeout_s:
                time.sleep(min(timeout_s, 2.0))
            return None

        # Fetch the last pod logs and parse the final JSON line.
        # The kubernetes-client Python SDK mangles JSON-looking lines in
        # read_namespaced_pod_log (it applies literal_eval-style conversion),
        # so we shell out to `kubectl logs` which streams the bytes verbatim.
        pods = self._core.list_namespaced_pod(
            namespace=self._ns,
            label_selector=f"job-name={handle.ref}",
        )
        log_text = ""
        if pods.items:
            pod_name = pods.items[-1].metadata.name
            kubectl = shutil.which("kubectl")
            if kubectl is not None:
                try:
                    p = subprocess.run(
                        [kubectl, "-n", self._ns, "logs", pod_name, "--tail=200"],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    log_text = p.stdout or ""
                except Exception as e:  # noqa: BLE001
                    log.warning("kubectl logs %s failed: %s", pod_name, e)
            if not log_text:
                try:
                    log_text = (
                        self._core.read_namespaced_pod_log(
                            name=pod_name, namespace=self._ns, tail_lines=200
                        )
                        or ""
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("failed to read pod log %s: %s", pod_name, e)

        parsed: dict[str, Any] | None = None
        for line in reversed(log_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        if parsed is None:
            return TrialResult(
                trial_id=handle.trial_id,
                status="crash",
                score=None,
                error=f"no JSON in pod log for Job {handle.ref}",
                backend=self.name,
            )

        # Decode metrics: {"metric|workload": value} → {(metric, workload|None): value}
        metrics: dict[tuple[str, str | None], float] = {}
        for k, v in (parsed.get("metrics") or {}).items():
            if "|" in k:
                m, w = k.split("|", 1)
                metrics[(m, w or None)] = float(v)
            else:
                metrics[(k, None)] = float(v)
        return TrialResult(
            trial_id=parsed.get("trial_id") or handle.trial_id,
            status=parsed.get("status", "crash"),
            score=parsed.get("score"),
            metrics=metrics,
            error=parsed.get("error"),
            backend=self.name,
            worker_id=parsed.get("worker_id"),
        )

    def cancel(self, handle: TrialHandle) -> None:
        try:
            self._batch.delete_namespaced_job(
                name=handle.ref,
                namespace=self._ns,
                propagation_policy="Background",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to delete job %s: %s", handle.ref, e)
