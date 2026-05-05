"""QuerySpec — DSL round-trip + frozen + raw_sql escape hatch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lmtune.contracts.query_spec import (
    AggregateSpec,
    FilterCond,
    QuerySpec,
    SortKey,
)


def test_minimal_query():
    q = QuerySpec(record_kind="run")
    assert q.api_version == "lmtune/query/v1alpha1"
    assert q.filters == []
    assert q.limit is None
    assert q.is_raw() is False


def test_filter_op_validation():
    """CompareOp Literal 이 invalid op 을 reject."""
    with pytest.raises(ValidationError):
        FilterCond(column="status", op="bogus", value="ok")  # type: ignore[arg-type]


def test_filter_in_op_with_list():
    f = FilterCond(column="status", op="in", value=["ok", "error"])
    assert f.value == ["ok", "error"]


def test_sort_default_asc():
    s = SortKey(column="created_at")
    assert s.direction == "asc"


def test_aggregate_count_no_column():
    """count 는 column 없어도 OK (count(*))."""
    a = AggregateSpec(group_by=["status"], function="count")
    assert a.column is None
    assert a.alias == "agg_value"


def test_full_query_compose():
    q = QuerySpec(
        record_kind="trial",
        filters=[
            FilterCond(column="study_id", op="==", value="st1"),
            FilterCond(column="status", op="in", value=["completed", "pruned"]),
        ],
        sort=[SortKey(column="score", direction="desc")],
        limit=10,
        select=["trial_id", "score", "params"],
        aggregate=None,
    )
    assert q.record_kind == "trial"
    assert len(q.filters) == 2
    assert q.sort[0].direction == "desc"
    assert q.limit == 10


def test_raw_sql_escape_hatch():
    q = QuerySpec(record_kind="", raw_sql="SELECT 1")
    assert q.is_raw() is True


def test_query_spec_frozen():
    q = QuerySpec(record_kind="run")
    with pytest.raises((ValidationError, TypeError)):
        q.record_kind = "metric"  # type: ignore[misc]


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        QuerySpec(record_kind="run", _bogus=1)  # type: ignore[call-arg]


def test_json_schema_dump():
    schema = QuerySpec.model_json_schema()
    assert schema["title"] == "QuerySpec"
    props = schema.get("properties", {})
    assert "record_kind" in props
    assert "filters" in props
