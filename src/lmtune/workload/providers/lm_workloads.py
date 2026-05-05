"""LMWorkloadsProvider — lm-workloads 의 운영 trace 변환 파이프라인을 호출.

본 Provider 는 lmtune 의 ``[workloads]`` extra 가 설치되어 있을 때만 동작.
미설치 환경에서는 import 시점에 친절한 에러 메시지.

Source URI 형식:
  - ``vllm-log:/path/to/access.ndjson``
  - ``vllm-prom:http://prom:9090?model=...``
  - ``prodstack:/path/to/dump.json``

위 source 종류는 lm-workloads 의 ``IngestionAdapter`` 등록 plugin 과 일치한다.
새 source 추가 시 lm-workloads 쪽에 plugin 등록 + 본 Provider 는 변경 0.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from lmtune.workload.providers.base import WorkloadProvider


class LMWorkloadsProvider(WorkloadProvider):
    """lm-workloads 의 ``run_cycle`` 을 호출해 운영 trace → WorkloadSpec 추출.

    Args:
        source_uri: ``<adapter>:<path>`` 형식. adapter 는 lm-workloads 의
            IngestionAdapter plugin 이름 (vllm-log / vllm-prom / prodstack).
        store_path: lm-workloads 가 사용하는 DuckDB store. None 이면 tmpdir.
            (운영에서 누적 분석 원하면 영속 경로 지정.)
        out_dir: lm-workloads 의 export 산출 dir. None 이면 tmpdir.
        cluster_id: 다중 cluster 가 발견되면 어느 것을 선택할지. None 이면 첫번째.
    """

    def __init__(
        self,
        source_uri: str,
        *,
        store_path: str | Path | None = None,
        out_dir: str | Path | None = None,
        cluster_id: str | None = None,
    ) -> None:
        if ":" not in source_uri:
            raise ValueError(f"source URI 형식: <adapter>:<path>. got: {source_uri!r}")
        self.source_uri = source_uri
        self.store_path = Path(store_path) if store_path else None
        self.out_dir = Path(out_dir) if out_dir else None
        self.cluster_id = cluster_id

    def provide(self, *, refresh: bool = False):  # noqa: ARG002
        try:
            from lm_workloads.orchestrate.pipeline import run_cycle
        except ImportError as e:
            raise ImportError(
                "LMWorkloadsProvider requires lm-workloads. "
                "Install with: pip install 'lmtune[workloads]'"
            ) from e

        adapter, _, path = self.source_uri.partition(":")

        with tempfile.TemporaryDirectory(prefix="lmtune-lmworkloads-") as tmpdir:
            store_path = self.store_path or (Path(tmpdir) / "store.duckdb")
            out_dir = self.out_dir or (Path(tmpdir) / "out")

            result = run_cycle(
                source=adapter,
                input_path=path,
                store_path=str(store_path),
                out_dir=str(out_dir),
                formats=("native",),  # WorkloadSpec native 만 — lmtune profile 변환은 후속
            )

        if not result.specs:
            raise RuntimeError(
                f"lm-workloads run_cycle returned no WorkloadSpec for {self.source_uri!r}"
            )

        if self.cluster_id is not None:
            for s in result.specs:
                if s.meta.id == self.cluster_id:
                    return s
            available = [s.meta.id for s in result.specs]
            raise ValueError(f"cluster_id={self.cluster_id!r} not found in {available}")

        if len(result.specs) > 1:
            import warnings

            available = [s.meta.id for s in result.specs]
            warnings.warn(
                f"lm-workloads returned {len(result.specs)} clusters {available}; "
                "using first. Pass cluster_id=... to select explicitly.",
                stacklevel=2,
            )
        return result.specs[0]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(b"LMWorkloadsProvider\x00")
        h.update(self.source_uri.encode("utf-8"))
        h.update(b"\x00")
        if self.cluster_id:
            h.update(f"cluster={self.cluster_id}".encode())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"LMWorkloadsProvider(source_uri={self.source_uri!r}, cluster_id={self.cluster_id!r})"
        )
