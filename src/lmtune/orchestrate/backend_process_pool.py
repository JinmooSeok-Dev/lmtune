"""ProcessPoolBackend — single-host multi-worker via concurrent.futures.

Each worker is a child Python process that imports `bench.orchestrate.trial_runner`
and evaluates one TrialPayload at a time. Fits local dev (one machine, 1-N GPUs).

For K8s-native deployment, see backend_k8s.py. The interface is identical.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor

from lmtune.orchestrate.backend import TrialBackend, TrialHandle, TrialPayload, TrialResult
from lmtune.orchestrate.trial_runner import run_trial_locally


class ProcessPoolBackend(TrialBackend):
    """Local multi-worker pool.

    NOTE: workers > 1 only makes sense with (a) multiple GPUs + GPU lease and
    (b) per-worker DuckDB shards or an out-of-process writer service. On a
    single-GPU dev box with a shared DuckDB file you must use workers=1 —
    otherwise concurrent 'bench run' children collide on the DB lock and
    compete for the same GPU. Use the k8s-job backend (Phase S4) for real
    parallel execution with deployment-isolated endpoints.
    """

    name = "process-pool"

    def __init__(self, workers: int = 1):
        self._pool = ProcessPoolExecutor(max_workers=int(workers))
        self._workers = int(workers)
        self._inflight: dict[str, Future] = {}

    def submit(self, payload: TrialPayload) -> TrialHandle:
        fut = self._pool.submit(run_trial_locally, payload)
        self._inflight[payload.trial_id] = fut
        return TrialHandle(trial_id=payload.trial_id, backend=self.name, ref=fut)

    def poll(self, handle: TrialHandle, timeout_s: float | None = None) -> TrialResult | None:
        fut: Future = handle.ref
        try:
            if timeout_s is None and not fut.done():
                return None
            result = fut.result(timeout=timeout_s)
        except TimeoutError:
            return None
        except Exception as e:  # noqa: BLE001
            result = TrialResult(
                trial_id=handle.trial_id,
                status="crash",
                score=None,
                error=f"worker exception: {e}",
                backend=self.name,
            )
        self._inflight.pop(handle.trial_id, None)
        return result

    def cancel(self, handle: TrialHandle) -> None:
        fut: Future | None = handle.ref
        if fut is not None:
            fut.cancel()
        self._inflight.pop(handle.trial_id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
