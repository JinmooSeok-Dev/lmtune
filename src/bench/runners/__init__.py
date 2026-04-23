from bench.runners.aiperf import AIPerfRunner
from bench.runners.base import RunArtifact, Runner, RunnerError
from bench.runners.guidellm import GuideLLMRunner
from bench.runners.vllm_bench import VllmBenchRunner


def get_runner(kind: str) -> Runner:
    match kind:
        case "aiperf":
            return AIPerfRunner()
        case "vllm_bench":
            return VllmBenchRunner()
        case "guidellm":
            return GuideLLMRunner()
        case _:
            raise RunnerError(f"unknown runner kind: {kind}")


__all__ = [
    "AIPerfRunner",
    "GuideLLMRunner",
    "RunArtifact",
    "Runner",
    "RunnerError",
    "VllmBenchRunner",
    "get_runner",
    "get_runner",
]
