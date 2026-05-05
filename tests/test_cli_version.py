"""``lmtune --version`` flag 검증.

PR pyproject-plug-extras (#60) 의 drift test 와 같은 정신: ``__version__``
의 단일 진실이 CLI / package metadata / pyproject 어디서나 일관되어야 한다.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from lmtune import __version__
from lmtune.cli import app

runner = CliRunner()


def test_version_long_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert result.output.startswith("lmtune ")


def test_version_short_flag():
    """``-V`` 단축 flag 도 동일 동작."""
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_matches_pyproject():
    """__version__ 이 pyproject.toml [project].version 과 일치 — drift 차단."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert cfg["project"]["version"] == __version__, (
        f"__version__={__version__} != pyproject version={cfg['project']['version']}"
    )


def test_version_is_eager_overrides_subcommand():
    """``--version`` 이 subcommand 보다 먼저 처리 (is_eager=True)."""
    # subcommand 가 있는 invocation 에서도 version 만 출력하고 종료
    result = runner.invoke(app, ["--version", "search", "--help"])
    assert result.exit_code == 0
    assert __version__ in result.output
    # search 의 help 는 출력되지 않아야 함
    assert "Usage: lmtune search" not in result.output
