"""Objective — wraps a (params → score, metrics) evaluation.

Two built-in implementations:
- CallableObjective: user-supplied Python function (for tests, synthetic studies, and
                     in-process experiments without vLLM).
- BenchScoreObjective: invokes scripts/bench_score.py per workload profile and
                       aggregates into a total_score + per-workload secondary metrics.

Objectives must be deterministic in schema but noisy in value — the Study layer
handles N-repeat/CV gates if the underlying Objective supports it (BenchScoreObjective
delegates to bench_score.py which already implements N=3+CV auto-extend).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(slots=True)
class ObjectiveResult:
    score: float
    metrics: dict[tuple[str, str | None], float] = field(default_factory=dict)
    error: str | None = None
    accepted: bool = True  # False → reproducibility gate failed


class Objective(Protocol):
    def __call__(self, params: dict[str, Any]) -> ObjectiveResult: ...


# ---------------------------------------------------------------------------

class CallableObjective:
    """Wrap a plain Python function (params → float | tuple[float, dict])."""

    def __init__(self, fn: Callable[[dict[str, Any]], Any]):
        self._fn = fn

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        try:
            res = self._fn(params)
        except Exception as e:  # noqa: BLE001 — worker errors become CRASH
            return ObjectiveResult(score=0.0, error=str(e), accepted=False)
        if isinstance(res, ObjectiveResult):
            return res
        if isinstance(res, (int, float)):
            return ObjectiveResult(score=float(res))
        if isinstance(res, tuple) and len(res) == 2:
            score, metrics = res
            out: dict[tuple[str, str | None], float] = {}
            for k, v in (metrics or {}).items():
                if isinstance(k, tuple):
                    out[k] = float(v)
                else:
                    out[(k, None)] = float(v)
            return ObjectiveResult(score=float(score), metrics=out)
        raise TypeError(f"objective returned unexpected type: {type(res)!r}")


# ---------------------------------------------------------------------------

class BenchScoreObjective:
    """Call `scripts/bench_score.py` once per workload profile; sum scores.

    The caller is responsible for having already applied `params` to the endpoint
    YAML and restarted the server (DeploymentAdapter does this in Phase S4; S1
    uses LocalVLLMAdapter via a small shim or a pre-running endpoint).
    """

    def __init__(
        self,
        endpoint_path: str | Path,
        profile_paths: list[str | Path],
        *,
        repeats: int = 3,
        ttft_slo_ms: float = 500.0,
        bench_bin: str | None = None,
        python_bin: str | None = None,
    ):
        self.endpoint_path = Path(endpoint_path)
        self.profile_paths = [Path(p) for p in profile_paths]
        self.repeats = int(repeats)
        self.ttft_slo_ms = float(ttft_slo_ms)
        # Resolve bench CLI: explicit > BENCH_BIN env > venv(sys.executable) > PATH
        if bench_bin is None:
            import os as _os
            env_bin = _os.environ.get("BENCH_BIN")
            venv_bench = Path(sys.executable).parent / "bench"
            if env_bin:
                bench_bin = env_bin
            elif venv_bench.exists():
                bench_bin = str(venv_bench)
        self.bench_bin = bench_bin
        self.python_bin = python_bin or sys.executable or shutil.which("python") or "python3"
        self.script = Path(__file__).resolve().parents[3] / "scripts" / "bench_score.py"
        if not self.script.exists():
            raise FileNotFoundError(self.script)

    def _run_one(self, profile: Path) -> dict:
        import os as _os
        cmd = [
            self.python_bin, str(self.script),
            "-p", str(profile),
            "-e", str(self.endpoint_path),
            "-n", str(self.repeats),
            "--ttft-slo-ms", str(self.ttft_slo_ms),
        ]
        if self.bench_bin:
            cmd += ["--bench-bin", self.bench_bin]
        # Propagate venv bin through PATH so sub-subprocess (bench → guidellm) can resolve.
        env = _os.environ.copy()
        venv_bin = str(Path(self.python_bin).parent)
        env["PATH"] = venv_bin + _os.pathsep + env.get("PATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        last = ""
        for line in reversed(proc.stdout.splitlines()):
            if line.strip():
                last = line.strip()
                break
        if not last:
            return {"error": proc.stderr.strip()[-400:] or f"no output (rc={proc.returncode})"}
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return {"error": f"non-json output: {last[-400:]}"}

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        # NOTE: params application (endpoint YAML edit + restart) is out of scope
        # for this Objective. The caller orchestrates it via DeploymentAdapter.
        # Phase S4 composes them together; S1 assumes the endpoint is already
        # configured for these params upstream.
        del params

        import logging
        _log = logging.getLogger(__name__)

        total = 0.0
        metrics: dict[tuple[str, str | None], float] = {}
        accepted_all = True
        any_fail = False
        first_error: str | None = None
        for prof in self.profile_paths:
            res = self._run_one(prof)
            if "error" in res:
                any_fail = True
                _log.warning("bench_score fail (%s): %s", prof.name, res.get("error"))
                if first_error is None:
                    first_error = f"{prof.stem}: {res['error']}"
                continue
            wl = prof.stem
            score = float(res.get("score") or 0.0)
            total += score
            metrics[("score", wl)] = score
            for k in ("ttft_p99", "throughput_tok_avg", "e2e_p99", "cv_throughput"):
                v = res.get(k)
                if v is not None:
                    metrics[(k, wl)] = float(v)
            if res.get("accepted") is False:
                accepted_all = False
            if res.get("slo_pass") is False:
                any_fail = True
                if first_error is None:
                    first_error = f"{wl}: slo_pass=False"

        metrics[("score", None)] = total
        return ObjectiveResult(
            score=total,
            metrics=metrics,
            accepted=accepted_all and not any_fail,
            error=first_error if any_fail else None,
        )
