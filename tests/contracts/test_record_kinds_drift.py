"""``RECORD_KINDS`` 화이트리스트 drift 가드.

본 테스트는 ``record_spec.py`` 의 ``_RecordBase`` 자식 클래스와 ``RECORD_KINDS``
튜플이 항상 1:1 동기되도록 영속 검증. 새 record class 가 추가되었지만
``RECORD_KINDS`` 에 등록 누락된 경우 즉시 실패.

PLUG 정신: 새 record kind 합류는 ``record_spec.py`` 한 파일에서 (1) class
정의 + (2) ``RECORD_KINDS`` 추가 + (3) ``kind_to_class`` 매핑 등 세 곳을
동기화해야 함. 본 테스트가 그 세 동기 지점 중 (1)↔(2) 와 (2)↔(3) 을
영속 검증.
"""

from __future__ import annotations

from lmtune.contracts.record_spec import (
    RECORD_KINDS,
    RecordSpec,  # noqa: F401  -- discriminated union; ensure import side-effect
    _RecordBase,
    kind_to_class,
)


def _discovered_kinds() -> set[str]:
    """``_RecordBase`` 자식의 ``kind`` literal default 를 모두 수집."""
    discovered: set[str] = set()
    for cls in _RecordBase.__subclasses__():
        kind_field = cls.model_fields.get("kind")
        if kind_field is not None and kind_field.default is not None:
            discovered.add(str(kind_field.default))
    return discovered


def test_record_kinds_matches_subclasses():
    """``_RecordBase`` 자식 ↔ ``RECORD_KINDS`` 1:1 동기."""
    discovered = _discovered_kinds()
    declared = set(RECORD_KINDS)

    only_subclass = discovered - declared
    only_declared = declared - discovered

    assert not only_subclass, (
        f"_RecordBase 자식인데 RECORD_KINDS 에 없음: {sorted(only_subclass)}. "
        "새 record class 추가 시 RECORD_KINDS 튜플도 갱신 필요."
    )
    assert not only_declared, (
        f"RECORD_KINDS 에 있지만 _RecordBase 자식 아님: {sorted(only_declared)}. "
        "stale entry 제거 필요."
    )


def test_kind_to_class_covers_all_record_kinds():
    """``RECORD_KINDS`` 의 모든 kind 가 ``kind_to_class`` 에 매핑."""
    for kind in RECORD_KINDS:
        cls = kind_to_class(kind)
        assert cls is not None
        assert issubclass(cls, _RecordBase)
        # kind literal default 가 RECORD_KINDS 의 entry 와 일치
        assert cls.model_fields["kind"].default == kind


def test_kind_to_class_round_trip():
    """``kind_to_class(kind)`` 의 결과가 다시 ``kind_to_class`` 의 inverse 와 일치."""
    for kind in RECORD_KINDS:
        cls = kind_to_class(kind)
        # cls.kind == kind
        round_trip_kind = cls.model_fields["kind"].default
        assert round_trip_kind == kind


def test_record_kinds_count_consistent():
    """RECORD_KINDS 의 entry 수와 _RecordBase 자식 수가 동일."""
    assert len(RECORD_KINDS) == len(_RecordBase.__subclasses__())
