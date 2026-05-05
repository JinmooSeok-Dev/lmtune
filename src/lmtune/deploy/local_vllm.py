"""LocalVLLMAdapter — writes params into endpoint YAML and restarts bare-metal vLLM.

Wraps `scripts/vllm_restart.sh`: that script reads `deployment.engine_args`
and spawns `vllm serve ...` with the matching CLI flags. We do the YAML
merge (`merge_params_into_endpoint`) then delegate the restart + readiness
polling to the shell script (which already does 180s polling on /v1/models).

Constraint: single-GPU bare metal. tp/pp/dp/ep must stay at 1 — we assert.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lmtune.deploy.base import (
    ApplyResult,
    DeploymentAdapter,
    HealthReport,
    merge_params_into_endpoint,
)
from lmtune.deploy.health import probe_openai_models

log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "vllm_restart.sh"


class LocalVLLMAdapter(DeploymentAdapter):
    adapter_label = "local-vllm"

    def __init__(self, *, restart_script: str | Path | None = None, timeout_s: float = 240.0):
        self._script = Path(restart_script) if restart_script else _SCRIPT
        if not self._script.exists():
            raise FileNotFoundError(self._script)
        self._timeout_s = float(timeout_s)

    def apply(self, endpoint_path: str | Path, params: Mapping[str, Any]) -> ApplyResult:
        ep = Path(endpoint_path)
        # 1. Merge params into YAML (engine_args vs parallelism dispatch in base.py).
        data = merge_params_into_endpoint(ep, params)
        parallelism = (data.get("deployment") or {}).get("parallelism") or {}
        for key in ("tp", "pp", "dp"):
            v = parallelism.get(key, 1)
            if int(v or 1) != 1:
                return ApplyResult(
                    ok=False,
                    health=HealthReport(
                        ready=False, detail=f"{key}={v} not supported in LocalVLLMAdapter"
                    ),
                    endpoint_path=ep,
                    adapter=self.adapter_label,
                    notes="LocalVLLMAdapter requires tp=pp=dp=1",
                )

        # 2. Invoke the restart shell script (kills old, starts new, polls /v1/models).
        log.info("LocalVLLMAdapter: restarting via %s", self._script)
        proc = subprocess.run(
            ["bash", str(self._script), str(ep)],
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
            return ApplyResult(
                ok=False,
                health=HealthReport(ready=False, detail="\n".join(tail)),
                endpoint_path=ep,
                adapter=self.adapter_label,
                notes=f"vllm_restart.sh failed rc={proc.returncode}",
            )

        # 3. Independent health probe (vllm_restart.sh already polled, but double-check).
        url = (data.get("url") or "").strip()
        health = (
            probe_openai_models(url)
            if url
            else HealthReport(ready=False, detail="no url in endpoint yaml")
        )
        return ApplyResult(
            ok=bool(health.ready),
            health=health,
            endpoint_path=ep,
            adapter=self.adapter_label,
            notes="",
        )
