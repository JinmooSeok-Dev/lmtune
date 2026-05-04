"""Smart rollout waiter — fast-fail on pod crash with classification.

`kubectl rollout status` 가 CrashLoopBackOff pod 를 deadline 까지 기다리는 문제 해결.
직접 pod status 를 polling 해서 crash 즉시 감지하고 container log 분석으로
크래시 종류 분류. lmtune trial 1개당 최대 5분 → 60-90초로 단축.

Crash classification:
  - infeasible : 구성 자체가 호환 안 됨 (mxfp4×float16 등). sampler 가 region 학습.
  - oom        : OutOfMemory. 더 작은 max_num_seqs / gpu_mem_util 로 retry 가능.
  - transient  : NCCL/network 일시적. 1회 retry 권장.
  - hard       : 분류 불가. score=0.
  - startup_timeout : crash 는 아니지만 시간 안에 ready 못 함.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


CRASH_PATTERNS: dict[str, list[str]] = {
    "infeasible": [
        # vllm pydantic VllmConfig 검증 — multiline 메시지 (e.g., mxfp4 × float16) 도
        # 매치하도록 [\s\S] 사용. re.DOTALL 안 쓰는 이유: 다른 패턴이 . 의 newline
        # non-match 에 의존할 수 있어 영향 범위 격리.
        r"ValidationError:[\s\S]{0,300}?not supported for quantization",
        r"unrecognized arguments",
        r"argparse.*invalid",
        r"required argument.*missing",
        r"is not a valid value",
        r"Found duplicate keys",
        # torch.compile / Dynamo 가 모델 코드의 hardcoded assertion 을 못 트레이스해서
        # partial graph 컴파일 거부 — 모델이 특정 axis 조합을 거부하는 신호 (구성 무효).
        # 예: gpt-oss-120b 의 attention.py:408 `assert self.kv_cache_dtype in {"fp8", ...}`
        # → kv_cache_dtype=auto/fp8_e5m2 선택 시 trip.
        r"Data-dependent assertion failed",
        r"cannot compile partial graph",
        r"assert self\.kv_cache_dtype in",
        # vllm V1 engine 의 CPU weight offload 미지원 조합 (vllm-project/vllm#18298):
        # cpu_offload_gb > 0 + multiproc_executor + input batch re-init 시 RuntimeError.
        # axis 조합이 invalid 임을 알리는 explicit message → infeasible.
        r"Cannot re-initialize the input batch when CPU weight offloading",
        r"CPU weight offloading is enabled",
    ],
    "oom": [
        r"OutOfMemoryError",
        r"CUDA out of memory",
        r"torch\.cuda\.OutOfMemoryError",
        r"Free memory.*requested",
        r"unable to allocate.*GPU memory",
    ],
    "transient": [
        r"NCCL.*timeout",
        r"NCCL.*error",
        r"Connection refused",
        r"socket\.gaierror",
        r"All-reduce.*failed",
        r"NIXL.*failed",
    ],
}


@dataclass(slots=True)
class RolloutResult:
    ok: bool
    crash_class: str | None = None
    detail: str = ""
    logs_tail: str = ""
    pods_seen: int = 0
    elapsed_s: float = 0.0


def _kubectl_run(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout, proc.stderr


def _get_deployment_selector(name: str, namespace: str) -> str:
    rc, out, _ = _kubectl_run(["-n", namespace, "get", "deployment", name, "-o", "json"])
    if rc != 0:
        return ""
    spec = (
        json.loads(out).get("spec", {}).get("selector", {}).get("matchLabels", {})
    )
    return ",".join(f"{k}={v}" for k, v in spec.items())


def _get_deployment_desired(name: str, namespace: str) -> int:
    rc, out, _ = _kubectl_run(["-n", namespace, "get", "deployment", name, "-o", "json"])
    if rc != 0:
        return 1
    return int(json.loads(out).get("spec", {}).get("replicas", 1))


def _get_pods(namespace: str, selector: str) -> list[dict]:
    rc, out, _ = _kubectl_run(
        ["-n", namespace, "get", "pods", "-l", selector, "-o", "json"]
    )
    if rc != 0:
        return []
    return json.loads(out).get("items", [])


def _logs_tail(pod_name: str, namespace: str, container: str, lines: int = 200) -> str:
    rc, out, _ = _kubectl_run(
        ["-n", namespace, "logs", pod_name, "-c", container, f"--tail={lines}"]
    )
    return out if rc == 0 else ""


def classify_crash(logs: str) -> str:
    if not logs:
        return "hard"
    for cls, patterns in CRASH_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, logs, re.IGNORECASE | re.MULTILINE):
                return cls
    return "hard"


def _crash_check(pod: dict, container_name: str) -> tuple[str, str] | None:
    """Return (reason_short, pod_name) if this pod is crashing, else None."""
    cs_list = pod.get("status", {}).get("containerStatuses") or []
    pod_name = pod["metadata"]["name"]
    for cs in cs_list:
        if cs.get("name") != container_name:
            continue
        waiting = (cs.get("state") or {}).get("waiting") or {}
        if waiting.get("reason") == "CrashLoopBackOff":
            return (f"CrashLoopBackOff: {waiting.get('message','')}", pod_name)
        # 2회 이상 재시작은 hang/crash
        if cs.get("restartCount", 0) >= 2:
            return (f"restartCount={cs['restartCount']}", pod_name)
        # 마지막 종료가 비정상 + 재시작 1회 = crash 시작
        last = (cs.get("lastState") or {}).get("terminated") or {}
        if last and last.get("exitCode", 0) != 0 and cs.get("restartCount", 0) >= 1:
            return (
                f"exit={last.get('exitCode')} reason={last.get('reason','')}",
                pod_name,
            )
    return None


def _ready_count(pods: list[dict], container_name: str) -> int:
    n = 0
    for p in pods:
        for cs in p.get("status", {}).get("containerStatuses") or []:
            if cs.get("name") == container_name and cs.get("ready"):
                n += 1
                break
    return n


def wait_rollout_smart(
    deployment_name: str,
    namespace: str,
    *,
    container_name: str = "vllm",
    total_timeout_s: int = 600,
    crash_threshold_s: int = 120,
    poll_interval_s: int = 5,
    desired_replicas: int | None = None,
) -> RolloutResult:
    """Watch deployment rollout, fast-fail on pod crash.

    Returns ok=True when desired_replicas pods are Ready (per container_name).
    Returns ok=False with classified crash_class on CrashLoopBackOff or
    restart>=2. Honors total_timeout_s as hard cap.
    """
    selector = _get_deployment_selector(deployment_name, namespace)
    if not selector:
        return RolloutResult(
            ok=False,
            crash_class="hard",
            detail=f"could not resolve selector for deployment/{deployment_name}",
        )
    if desired_replicas is None:
        desired_replicas = _get_deployment_desired(deployment_name, namespace)

    start = time.time()
    deadline = start + total_timeout_s
    last_summary = ""
    last_pods: list[dict] = []

    while time.time() < deadline:
        pods = _get_pods(namespace, selector)
        last_pods = pods
        if not pods:
            time.sleep(poll_interval_s)
            continue

        for pod in pods:
            crash = _crash_check(pod, container_name)
            if crash:
                reason, pod_name = crash
                logs = _logs_tail(pod_name, namespace, container_name, 200)
                cls = classify_crash(logs)
                log.warning(
                    "rollout fast-fail: dep=%s pod=%s class=%s reason=%s",
                    deployment_name, pod_name, cls, reason,
                )
                return RolloutResult(
                    ok=False,
                    crash_class=cls,
                    detail=f"{pod_name}: {reason}",
                    logs_tail=logs[-2000:],
                    pods_seen=len(pods),
                    elapsed_s=time.time() - start,
                )

        ready = _ready_count(pods, container_name)
        wanted = max(desired_replicas, 1)
        if ready >= wanted:
            return RolloutResult(
                ok=True,
                detail=f"{ready}/{wanted} ready",
                pods_seen=len(pods),
                elapsed_s=time.time() - start,
            )
        last_summary = f"{ready}/{wanted} ready, {len(pods)} pods"
        time.sleep(poll_interval_s)

    # timeout
    logs_tail = ""
    if last_pods:
        logs = _logs_tail(
            last_pods[0]["metadata"]["name"], namespace, container_name, 200
        )
        logs_tail = logs[-2000:]
    return RolloutResult(
        ok=False,
        crash_class="startup_timeout",
        detail=f"timeout {total_timeout_s}s ({last_summary})",
        logs_tail=logs_tail,
        pods_seen=len(last_pods),
        elapsed_s=time.time() - start,
    )
