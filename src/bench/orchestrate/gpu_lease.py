"""Process-local GPU lease using fcntl.flock on /tmp files.

Each worker grabs a lease on one GPU index before starting a trial. Two workers
cannot lease the same GPU simultaneously (flock on a sentinel file). The lease
is released on context exit or on worker process death (OS cleans the lock).

No-op mode: set BENCH_GPU_LEASE_DISABLE=1 to skip leasing (useful in CI without GPUs).
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


_LEASE_DIR = Path(os.environ.get("BENCH_GPU_LEASE_DIR", "/tmp"))


class GPULease:
    """Context manager that holds a flock on `/tmp/bench_gpu_<N>.lock`."""

    def __init__(self, gpu_id: int):
        self.gpu_id = int(gpu_id)
        self.path = _LEASE_DIR / f"bench_gpu_{self.gpu_id}.lock"
        self._fp = None

    def __enter__(self):
        self._fp = open(self.path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fp.close()
            self._fp = None
            raise
        self._fp.write(str(os.getpid()))
        self._fp.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fp is not None:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            self._fp.close()
            self._fp = None


@contextmanager
def try_acquire_gpu(gpu_ids: list[int]):
    """Try each GPU id in order; yield the first that leases successfully, else
    raise RuntimeError. Releases on exit.
    """
    if os.environ.get("BENCH_GPU_LEASE_DISABLE") == "1":
        yield None
        return
    last_exc: Exception | None = None
    for gid in gpu_ids:
        try:
            lease = GPULease(gid)
            lease.__enter__()
            try:
                yield lease
            finally:
                lease.__exit__(None, None, None)
            return
        except BlockingIOError as e:
            last_exc = e
            continue
    raise RuntimeError(f"no free GPU in {gpu_ids}: {last_exc}")
