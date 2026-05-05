"""PostgresArtifactStore — PLUG 패턴 stub 검증.

본 stub 의 acceptance:
1. ArtifactStore ABC 의 instance 가 됨 (subclass 관계).
2. ``psycopg`` 미설치 환경에서는 ImportError 가 친절 메시지와 함께 발생.
3. ``psycopg`` 가 있으면 인스턴스 생성 성공 (lazy connection — 첫 put/query 까지
   미연결).
4. put / query 는 NotImplementedError (follow-up PR 에서 채움).
5. cli_storage 의 _BACKENDS 에 합류 → ``lmtune storage migrate --src-kind postgres``
   가 typer 의 valid choice 로 받아짐.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lmtune.cli_storage import _BACKENDS, app
from lmtune.contracts import QuerySpec, RunRecord
from lmtune.storage.store import ArtifactStore, PostgresArtifactStore

_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None


def test_postgres_is_artifact_store():
    """클래스 자체는 항상 ABC subclass — psycopg 미설치라도 type check 가능."""
    assert issubclass(PostgresArtifactStore, ArtifactStore)


@pytest.mark.skipif(_HAS_PSYCOPG, reason="psycopg 설치된 환경에서는 ImportError 분기 없음")
def test_postgres_import_error_message_when_missing():
    """psycopg 미설치 시 친절 ImportError."""
    with pytest.raises(ImportError) as ei:
        PostgresArtifactStore("postgres://localhost/dummy")
    msg = str(ei.value)
    assert "psycopg" in msg
    assert "lmtune[postgres]" in msg


@pytest.mark.skipif(not _HAS_PSYCOPG, reason="psycopg 미설치 — instance 생성 검증 skip")
def test_postgres_lazy_connection():
    """psycopg 있으면 instance 생성 성공 + connection 미수립."""
    store = PostgresArtifactStore("postgres://localhost:1/never-connect")
    assert store._conn is None
    store.close()  # no-op


@pytest.mark.skipif(not _HAS_PSYCOPG, reason="psycopg 미설치")
def test_postgres_put_query_raise_not_implemented():
    store = PostgresArtifactStore("postgres://localhost:1/never-connect")
    with pytest.raises(NotImplementedError):
        store.put(
            [RunRecord(run_id="r", profile_slug="p", endpoint_slug="e", runner="g", status="ok")]
        )
    with pytest.raises(NotImplementedError):
        store.query(QuerySpec(record_kind="run"))
    store.close()


def test_postgres_in_cli_backends_list():
    """``lmtune storage list-backends`` 에 postgres 가 합류."""
    assert "postgres" in _BACKENDS
    runner = CliRunner()
    result = runner.invoke(app, ["list-backends"])
    assert result.exit_code == 0
    assert "postgres" in result.output


def test_postgres_in_cli_migrate_choices(tmp_path: Path):
    """``--src-kind postgres`` 가 typer.BadParameter 의 'unknown backend' 분기로
    빠지지 않음. psycopg 미설치 시엔 ImportError 메시지로 친절 거절.
    """
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "postgres",
            "--src",
            "postgres://localhost:1/never-connect",
            "--dst-kind",
            "local",
            "--dst",
            str(tmp_path / "out"),
        ],
    )
    if not _HAS_PSYCOPG:
        # extra 미설치 → ImportError → typer.BadParameter
        assert result.exit_code != 0
        assert "psycopg" in result.output or "lmtune[postgres]" in result.output
    else:
        # 설치돼있으면 첫 put/query 호출 시 NotImplementedError 로 빠짐.
        # 본 테스트는 그 detail 까진 검증하지 않음 — 허용/거부만.
        # query 단계에서 NotImplementedError → typer 가 처리 (exit != 0 가능)
        pass
