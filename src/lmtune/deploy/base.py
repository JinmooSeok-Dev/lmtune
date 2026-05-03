"""Deployment adapter protocol + endpoint YAML merger.

The adapter contract is intentionally small:

    adapter.apply(endpoint_path, params) -> ApplyResult
    adapter.teardown(endpoint_path)      -> None       (idempotent)

`apply()` is responsible for:
  1. mutating the endpoint YAML in place (merging trial params into
     deployment.engine_args / deployment.parallelism as appropriate),
  2. restarting / rolling out the underlying serving stack,
  3. waiting until the endpoint passes a health probe.

`teardown()` releases the resources and is a no-op on the local adapter.

Conditional axis gating — axes declared with `active_if: {adapter: llmd-k8s}`
are only activated under that adapter. Adapters advertise their label via the
class attribute `adapter_label`, and SearchSpace.active_axes(context) reads
`{"adapter": adapter.adapter_label}` to filter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENGINE_ARG_KEYS: set[str] = {
    "max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill",
    "gpu_memory_utilization", "max_model_len", "kv_cache_dtype",
    "block_size", "enforce_eager", "async_scheduling",
}
_PARALLELISM_KEYS: set[str] = {"tp", "pp", "dp", "ep", "rsd"}
# P/D disaggregation 의 release-level replica axis. helmfile values 의 prefill/decode
# 블록의 replicas 키와 1:1 매핑 (e.g., prefill_replicas=2 → prefill.replicas: 2).
_PD_REPLICA_KEYS: set[str] = {"prefill_replicas", "decode_replicas"}


@dataclass(slots=True)
class HealthReport:
    ready: bool
    latency_ms: float = 0.0
    detail: str = ""
    pod_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApplyResult:
    ok: bool
    health: HealthReport
    endpoint_path: Path
    notes: str = ""
    adapter: str = ""


def merge_params_into_endpoint(endpoint_path: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    """Merge trial params into endpoint YAML; write in place; return the merged dict.

    Keys in `_ENGINE_ARG_KEYS` land under `deployment.engine_args`.
    Keys in `_PARALLELISM_KEYS` land under `deployment.parallelism`.
    Unknown keys are written under `deployment.engine_args` (vLLM passes unknowns
    through as CLI flags) so new axes work without code changes.
    """
    p = Path(endpoint_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    deployment = data.setdefault("deployment", {})
    engine_args = deployment.setdefault("engine_args", {})
    parallelism = deployment.setdefault("parallelism", {})
    replicas = deployment.setdefault("replicas", {})

    for k, v in params.items():
        if k in _PARALLELISM_KEYS:
            parallelism[k] = v
        elif k in _PD_REPLICA_KEYS:
            # prefill_replicas → replicas.prefill, decode_replicas → replicas.decode
            replicas[k.removesuffix("_replicas")] = int(v)
        else:
            engine_args[k] = v

    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data


class DeploymentAdapter(ABC):
    """Base class for deployment adapters. Subclasses set `adapter_label`."""

    adapter_label: str = "abstract"

    @abstractmethod
    def apply(
        self,
        endpoint_path: str | Path,
        params: Mapping[str, Any],
    ) -> ApplyResult: ...

    def teardown(self, endpoint_path: str | Path) -> None:  # noqa: ARG002
        return None

    def context(self) -> dict[str, str]:
        """Context dict for SearchSpace.active_axes gating."""
        return {"adapter": self.adapter_label}
