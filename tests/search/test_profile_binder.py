"""Phase W — env profile binder unit tests.

Phase W 의 macro × profile binding 동작 검증. binder 가:
  1. applies_when 으로 매칭 (정확/리스트 멤버)
  2. priority 순서로 env_locked merge
  3. env_tunable 의 unique union
  4. unmatched profile 은 결과에 포함 안 됨
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from lmtune.search.profile_binder import EnvProfileBinder, _matches


def _write(p: Path, content: str) -> None:
    p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_matches_empty_always_true():
    assert _matches({}, {"anything": 1}) is True


def test_matches_exact():
    assert _matches({"a": 1}, {"a": 1, "b": 2}) is True
    assert _matches({"a": 1}, {"a": 2}) is False
    assert _matches({"a": 1, "b": 2}, {"a": 1}) is False  # missing b


def test_matches_list_membership():
    assert _matches({"adapter": ["llmd-k8s", "local-vllm"]}, {"adapter": "llmd-k8s"}) is True
    assert _matches({"adapter": ["llmd-k8s", "local-vllm"]}, {"adapter": "raw-vllm"}) is False


def test_binder_loads_yamls(tmp_path: Path):
    p1 = tmp_path / "p1.yaml"
    _write(
        p1,
        """
        name: p1
        applies_when: {adapter: local-vllm}
        env_locked: {NCCL_IB_DISABLE: "1"}
        env_tunable:
          - name: NCCL_BUFFSIZE
            kind: categorical
            values: [default, 2MB]
    """,
    )
    binder = EnvProfileBinder(tmp_path)
    assert len(binder.all()) == 1
    p = binder.all()[0]
    assert p.name == "p1"
    assert len(p.env_tunable) == 1
    assert p.env_tunable[0].name == "NCCL_BUFFSIZE"


def test_binder_match_and_priority(tmp_path: Path):
    _write(
        tmp_path / "lo.yaml",
        """
        name: lo_priority
        applies_when: {adapter: llmd-k8s}
        env_locked: {NCCL_IB_DISABLE: "1", NIXL_TRANSPORT: tcp}
        priority: 0
        env_tunable:
          - {name: A, kind: categorical, values: [1, 2]}
    """,
    )
    _write(
        tmp_path / "hi.yaml",
        """
        name: hi_priority
        applies_when: {adapter: llmd-k8s, well_lit_path: pd-disaggregation}
        env_locked: {NIXL_TRANSPORT: rdma}    # 후순위가 lo 의 NIXL_TRANSPORT 를 override
        priority: 1
        env_tunable:
          - {name: B, kind: categorical, values: [4, 16]}
    """,
    )
    binder = EnvProfileBinder(tmp_path)

    locked, tunable, matched = binder.bind(
        {
            "adapter": "llmd-k8s",
            "well_lit_path": "pd-disaggregation",
        }
    )
    assert matched == ["lo_priority", "hi_priority"]  # priority 순
    assert locked["NCCL_IB_DISABLE"] == "1"  # lo 만 set
    assert locked["NIXL_TRANSPORT"] == "rdma"  # hi 가 lo override
    assert {a.name for a in tunable} == {"A", "B"}


def test_binder_unmatched_excluded(tmp_path: Path):
    _write(
        tmp_path / "p.yaml",
        """
        name: p
        applies_when: {adapter: llmd-k8s}
        env_locked: {NCCL_IB_DISABLE: "1"}
    """,
    )
    binder = EnvProfileBinder(tmp_path)
    locked, tunable, matched = binder.bind({"adapter": "local-vllm"})
    assert matched == []
    assert locked == {}
    assert tunable == []


def test_binder_empty_dir(tmp_path: Path):
    binder = EnvProfileBinder(tmp_path / "nonexistent")
    assert binder.all() == []
    locked, tunable, matched = binder.bind({"any": "thing"})
    assert (locked, tunable, matched) == ({}, [], [])
