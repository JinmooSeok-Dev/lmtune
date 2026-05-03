"""Deployment adapters — apply a (params, endpoint_yaml) pair to a real serving stack.

A trial's params correspond to vLLM/llm-d config flags (engine_args) and, on
K8s, to the ParallelismSpec (tp/pp/dp/ep). The DeploymentAdapter takes the
endpoint YAML, merges the trial params into `deployment.engine_args` /
`deployment.parallelism`, and makes the backing server reflect that config.

S4 ships two concrete adapters:
- LocalVLLMAdapter  → wraps scripts/vllm_restart.sh  (single-host, tp=pp=dp=1)
- LLMDK8sAdapter    → helmfile apply + kubectl rollout  (tp/pp/dp/ep searchable)

Both implement the same interface so Objective / Driver code doesn't branch.
"""

from lmtune.deploy.base import (
    ApplyResult,
    DeploymentAdapter,
    HealthReport,
    merge_params_into_endpoint,
)
from lmtune.deploy.health import probe_openai_models, warmup_one_token
from lmtune.deploy.llmd_k8s import LLMDK8sAdapter
from lmtune.deploy.local_vllm import LocalVLLMAdapter

__all__ = [
    "ApplyResult",
    "DeploymentAdapter",
    "HealthReport",
    "LocalVLLMAdapter",
    "LLMDK8sAdapter",
    "merge_params_into_endpoint",
    "probe_openai_models",
    "warmup_one_token",
]
