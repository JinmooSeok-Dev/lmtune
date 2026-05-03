from __future__ import annotations

import threading
from pathlib import Path

from lmtune.storage.duckdb_store import DuckDBStore
from lmtune.storage.writer_queue import DuckDBWriterQueue


def test_writer_queue_serializes_concurrent_writes(tmp_path: Path):
    store = DuckDBStore(tmp_path / "q.duckdb")
    store.record_study(
        study_id="st-wq",
        name="wq-demo",
        strategy="random",
        metric_name="score",
        direction="maximize",
    )
    wq = DuckDBWriterQueue(store)

    N_producers = 4
    N_per = 25

    def produce(pid: int):
        for i in range(N_per):
            tid = f"tr-{pid}-{i:02d}"
            wq.enqueue(
                "record_trial", tid, "st-wq", pid * N_per + i, {"i": i},
                status="completed", score=float(i), backend="test",
                completed=True,
            )
            wq.enqueue("record_trial_metrics", tid, {("metric", None): float(i)})

    threads = [threading.Thread(target=produce, args=(p,)) for p in range(N_producers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wq.flush()
    wq.stop()

    assert wq.error() is None

    rows = store.list_trials("st-wq")
    assert len(rows) == N_producers * N_per

    # Every trial must have its single metric recorded.
    for tid, *_ in rows:
        m = store.get_trial_metrics(tid)
        assert "metric" in m
    store.close()


def test_writer_queue_reports_method_errors(tmp_path: Path):
    store = DuckDBStore(tmp_path / "q.duckdb")
    wq = DuckDBWriterQueue(store)
    wq.enqueue("no_such_method", "x")
    wq.flush()
    wq.stop()
    err = wq.error()
    assert err is not None
    assert "no_such_method" in str(err)
    store.close()
