"""b200/helmfile/_postrender/postrender.sh 가 multi-doc YAML 입력에서
Deployment 에만 spec.template.spec.runtimeClassName='nvidia' 를 주입하는지 검증.

llm-d-modelservice v0.4.12 chart 가 runtimeClassName 을 expose 안 해서
NHN k3s (default runtime=runc + RuntimeClass nvidia) 환경에서 GPU pod 가
libcuda 못 찾는 문제의 우회. helm --post-renderer 로 등록된다.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "b200/helmfile/_postrender/postrender.sh"


pytestmark = pytest.mark.skipif(
    not shutil.which("kubectl") or not SCRIPT.exists(),
    reason="kubectl required (kustomize 빌트인) and post-render script must exist",
)


def _run(stdin_yaml: str) -> str:
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        input=stdin_yaml,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_injects_runtime_class_into_deployment():
    """Deployment 의 spec.template.spec 아래에 runtimeClassName: nvidia 주입."""
    out = _run("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foo
  namespace: ns1
spec:
  replicas: 1
  selector: {matchLabels: {app: foo}}
  template:
    metadata: {labels: {app: foo}}
    spec:
      containers: [{name: c, image: nginx}]
""")
    docs = [d for d in yaml.safe_load_all(out) if d]
    deploys = [d for d in docs if d.get("kind") == "Deployment"]
    assert len(deploys) == 1
    assert deploys[0]["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"


def test_leaves_non_deployment_untouched():
    """Service / Gateway / HTTPRoute 같은 non-Deployment 는 손대지 않는다."""
    out = _run("""\
apiVersion: v1
kind: Service
metadata:
  name: svc1
  namespace: ns1
spec:
  ports: [{port: 80}]
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: rt1
  namespace: ns1
spec:
  rules: []
""")
    # 입력 doc 수와 동일 (kustomize 가 추가하지 않음)
    docs = [d for d in yaml.safe_load_all(out) if d]
    assert len(docs) == 2
    # 어느 것도 runtimeClassName 안 가짐
    for d in docs:
        spec = (d.get("spec") or {}).get("template", {}).get("spec", {}) if d else {}
        assert spec.get("runtimeClassName") is None
    # 더 단순한 안전망: 출력에 runtimeClassName 토큰 자체가 없어야
    assert "runtimeClassName" not in out


def test_multi_doc_only_deployments_patched():
    """Deployment + Service 섞인 multi-doc 입력에서 Deployment 만 patch."""
    out = _run("""\
apiVersion: apps/v1
kind: Deployment
metadata: {name: a, namespace: ns}
spec:
  selector: {matchLabels: {app: a}}
  template:
    metadata: {labels: {app: a}}
    spec: {containers: [{name: c, image: x}]}
---
apiVersion: v1
kind: Service
metadata: {name: s, namespace: ns}
spec: {ports: [{port: 80}]}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: b, namespace: ns}
spec:
  selector: {matchLabels: {app: b}}
  template:
    metadata: {labels: {app: b}}
    spec: {containers: [{name: c, image: x}]}
""")
    docs = [d for d in yaml.safe_load_all(out) if d]
    deploys = [d for d in docs if d.get("kind") == "Deployment"]
    services = [d for d in docs if d.get("kind") == "Service"]
    assert len(deploys) == 2 and len(services) == 1
    for d in deploys:
        assert d["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"
    # Service 본문엔 runtimeClassName 키 X
    assert "runtimeClassName" not in yaml.safe_dump(services[0])
