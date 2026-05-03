"""DuckDBWriterQueue — single-writer thread that serializes DB writes.

Why: DuckDB enforces a single file lock per process. When the Driver spawns
worker processes (or receives async results from K8s Jobs), only the Driver
process owns the DB. This queue lets producers (main thread + backend poll)
enqueue write tasks without direct DB access.

Contract:
- The queue is created with an already-connected DuckDBStore (Driver-owned).
- Callers invoke `enqueue(kind, *args, **kwargs)` where `kind` is a method name
  on DuckDBStore. A background thread dequeues and dispatches.
- `flush()` blocks until all pending tasks are drained.
- `stop()` flushes and joins the thread.

This is a stop-gap for Phase S3 single-host distribution. For multi-process
writer coordination across hosts, switch to DuckDB WAL (experimental) or a
separate writer micro-service.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class DuckDBWriterQueue:
    _SENTINEL = object()

    def __init__(self, store, *, name: str = "duckdb-writer"):
        self._store = store
        self._q: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._stopped = threading.Event()
        self._done = threading.Event()
        self._error: Exception | None = None
        self._thread.start()

    def enqueue(self, method: str, *args, **kwargs) -> None:
        if self._stopped.is_set():
            raise RuntimeError("writer queue is stopped")
        self._q.put((method, args, kwargs))

    def flush(self) -> None:
        """Block until all currently enqueued tasks are processed."""
        marker = threading.Event()
        self._q.put(("__marker__", (marker,), {}))
        marker.wait()

    def stop(self) -> None:
        self._stopped.set()
        self._q.put(self._SENTINEL)
        self._thread.join(timeout=30)
        self._done.set()

    def error(self) -> Exception | None:
        return self._error

    # ---- internal -------------------------------------------------------

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._SENTINEL:
                    break
                method, args, kwargs = item
                if method == "__marker__":
                    args[0].set()
                    continue
                fn: Callable = getattr(self._store, method, None)
                if fn is None:
                    raise AttributeError(f"DuckDBStore has no '{method}'")
                fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                self._error = e
                log.exception("writer queue failed while dispatching %r", item)
            finally:
                self._q.task_done()
