"""Orchestration — dispatch trials to local worker pools or K8s Jobs.

Phase S3 introduces:
- TrialBackend (ABC)    — submit / poll / cancel
- ProcessPoolBackend    — single-host multi-worker via concurrent.futures
- K8sJobBackend         — one K8s Job per trial (Phase S4 completes the adapter)
- Driver                — ask/dispatch/poll/tell loop shared by all backends
- DuckDBWriterQueue     — single-writer thread so workers never touch DB directly
- GPU lease             — flock on /tmp so two workers never grab the same GPU
"""

from lmtune.orchestrate.backend import (
    TrialBackend,
    TrialHandle,
    TrialPayload,
    TrialResult,
)
from lmtune.orchestrate.gpu_lease import GPULease, try_acquire_gpu

__all__ = [
    "TrialBackend",
    "TrialHandle",
    "TrialPayload",
    "TrialResult",
    "GPULease",
    "try_acquire_gpu",
]
