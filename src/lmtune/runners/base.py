from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec


class RunnerError(RuntimeError):
    pass


@dataclass
class RequestRow:
    req_id: str
    turn_idx: int | None = None
    conversation_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    thinking_tokens: int | None = None
    tool_call_count: int | None = None
    tool_result_tokens: int | None = None
    phase: str | None = None              # exploration | editing | execution | verification | other
    role: str | None = None               # planner | reasoner | verifier | solo
    energy_wh: float | None = None
    cost_usd: float | None = None
    ttft_ms: float | None = None
    itl_mean_ms: float | None = None
    e2e_ms: float | None = None
    started_at: float | None = None       # epoch seconds
    completed_at: float | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class SessionRow:
    session_id: str
    task_id: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cached_tokens: int | None = None
    turn_count: int | None = None
    tool_call_count: int | None = None
    duration_ms: float | None = None
    success: bool | None = None
    total_cost_usd: float | None = None
    total_energy_wh: float | None = None


@dataclass
class TrajectoryEvent:
    session_id: str
    seq: int
    event_type: str                       # user | assistant | tool_call | tool_result | thinking
    ts: float | None = None
    phase: str | None = None
    tokens: int | None = None
    metadata: dict | None = None


@dataclass
class RunArtifact:
    run_id: str
    runner_kind: str
    command: list[str]
    raw_dir: Path
    stdout_path: Path
    stderr_path: Path
    status: str = "ok"
    error: str | None = None
    tool_version: str | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    requests: list[RequestRow] = field(default_factory=list)
    sessions: list[SessionRow] = field(default_factory=list)
    trajectory: list[TrajectoryEvent] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None


class Runner(ABC):
    kind: str

    @abstractmethod
    def build_command(
        self, profile: ProfileSpec, endpoint: EndpointSpec, run_id: str, raw_dir: Path
    ) -> list[str]: ...

    @abstractmethod
    def parse(self, raw_dir: Path) -> tuple[dict[str, dict[str, float]], list[RequestRow]]: ...

    def tool_version(self) -> str | None:
        return None

    def _apply_overrides(self, cmd: list[str], profile: ProfileSpec) -> list[str]:
        """profile.runner_overrides[self.kind] 의 {flag: value} 를 CLI 뒤에 pass-through.

        값이 True 이면 플래그만 붙이고(boolean flag), None/False 이면 생략, 나머지는 str 변환.
        """
        extras = profile.runner_overrides.get(self.kind) or {}
        for flag, value in extras.items():
            if value is None or value is False:
                continue
            if value is True:
                cmd.append(str(flag))
            else:
                cmd.extend([str(flag), str(value)])
        return cmd

    def run(
        self,
        profile: ProfileSpec,
        endpoint: EndpointSpec,
        run_id: str,
        workdir: Path,
        env_extra: dict[str, str] | None = None,
    ) -> RunArtifact:
        raw_dir = workdir / run_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._apply_overrides(self.build_command(profile, endpoint, run_id, raw_dir), profile)
        stdout_path = raw_dir / "stdout.log"
        stderr_path = raw_dir / "stderr.log"

        artifact = RunArtifact(
            run_id=run_id,
            runner_kind=self.kind,
            command=cmd,
            raw_dir=raw_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            tool_version=self.tool_version(),
        )

        import os
        import time

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        (raw_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        artifact.started_at = time.time()
        try:
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                proc = subprocess.run(cmd, stdout=out, stderr=err, env=env, check=False)
        except FileNotFoundError as e:
            artifact.status = "failed"
            artifact.error = f"runner binary not found: {e}"
            artifact.finished_at = time.time()
            return artifact

        artifact.finished_at = time.time()
        if proc.returncode != 0:
            artifact.status = "failed"
            artifact.error = f"exit={proc.returncode}"

        try:
            metrics, requests = self.parse(raw_dir)
            artifact.metrics = metrics
            artifact.requests = requests
        except Exception as e:  # noqa: BLE001 — 파서 실패 시 raw 만 남기고 partial 로
            artifact.status = "partial" if artifact.status == "ok" else artifact.status
            artifact.error = (artifact.error or "") + f" | parse_error: {e}"

        return artifact
