from __future__ import annotations

import pytest

from lmtune.orchestrate.gpu_lease import GPULease, try_acquire_gpu


def test_double_lease_on_same_gpu_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_GPU_LEASE_DIR", str(tmp_path))
    # force-reimport module constant
    import lmtune.orchestrate.gpu_lease as gl
    gl._LEASE_DIR = tmp_path

    with GPULease(0):
        # Second lease on same GPU must block → non-blocking → raises
        with pytest.raises(BlockingIOError):
            GPULease(0).__enter__()


def test_different_gpus_can_coexist(tmp_path, monkeypatch):
    import lmtune.orchestrate.gpu_lease as gl
    gl._LEASE_DIR = tmp_path
    with GPULease(0), GPULease(1):
        # both held, no exception
        pass


def test_try_acquire_skips_held(tmp_path, monkeypatch):
    import lmtune.orchestrate.gpu_lease as gl
    gl._LEASE_DIR = tmp_path
    with GPULease(0):
        with try_acquire_gpu([0, 1]) as lease:
            assert lease is not None
            assert lease.gpu_id == 1


def test_disable_via_env(monkeypatch):
    monkeypatch.setenv("BENCH_GPU_LEASE_DISABLE", "1")
    with try_acquire_gpu([0, 1]) as lease:
        assert lease is None
