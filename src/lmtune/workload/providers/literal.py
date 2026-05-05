"""LiteralWorkloadProvider — yaml 파일 경로 → WorkloadSpec.

사용자가 hand-write 한 yaml 또는 다른 도구가 emit 한 yaml 을 lmtune 이 그대로
소비하는 경로. 별도 외부 호출 0.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import yaml

from lmtune.workload.providers.base import WorkloadProvider


class LiteralWorkloadProvider(WorkloadProvider):
    """주어진 yaml 파일을 읽어 WorkloadSpec 으로 검증·반환."""

    def __init__(self, yaml_path: str | Path) -> None:
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"workload spec yaml not found: {self.yaml_path}")

    def provide(self, *, refresh: bool = False):  # noqa: ARG002 - refresh N/A here
        # Import 시점에 [workloads] extra 가 없으면 contracts/workload_spec 에서 친절한 에러
        from lmtune.contracts.workload_spec import WorkloadSpec

        text = self.yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return WorkloadSpec.model_validate(data)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(b"LiteralWorkloadProvider\x00")
        h.update(str(self.yaml_path.resolve()).encode("utf-8"))
        h.update(b"\x00")
        with contextlib.suppress(OSError):
            h.update(self.yaml_path.read_bytes())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:
        return f"LiteralWorkloadProvider(yaml_path={self.yaml_path!r})"
