from pathlib import Path

import pytest
import yaml

from bench.search.space import Axis, SearchSpace, load_space, parse_space


def test_axis_categorical_requires_values():
    with pytest.raises(ValueError, match="requires non-empty 'values'"):
        Axis(name="x", kind="categorical", values=[])


def test_axis_bool_normalized():
    a = Axis(name="p", kind="bool")
    assert a.values == [False, True]


def test_axis_float_requires_low_high():
    with pytest.raises(ValueError, match="requires low and high"):
        Axis(name="lr", kind="float")


def test_axis_log_uniform_positive_low():
    with pytest.raises(ValueError, match="log_uniform requires low > 0"):
        Axis(name="lr", kind="log_uniform", low=0, high=1)


def test_axis_active_if_gate():
    a = Axis(name="tp", kind="int", low=1, high=8, active_if={"adapter": "llmd-k8s"})
    assert a.is_active({"adapter": "llmd-k8s"}) is True
    assert a.is_active({"adapter": "local-vllm"}) is False
    assert a.is_active({}) is False

    b = Axis(name="x", kind="categorical", values=[1, 2])
    assert b.is_active({}) is True


def test_space_grid_size_discrete_only():
    sp = SearchSpace(
        name="s",
        axes=[
            Axis("x", "categorical", values=[1, 2, 4]),
            Axis("p", "bool"),
            Axis("n", "int", low=0, high=4, step=2),
        ],
    )
    # x (3) * p (2) * n (0,2,4) = 3 * 2 * 3 = 18
    assert sp.grid_size() == 18


def test_space_grid_size_rejects_float():
    sp = SearchSpace(
        name="s",
        axes=[Axis("lr", "float", low=0.1, high=0.9)],
    )
    with pytest.raises(ValueError, match="continuous"):
        sp.grid_size()


def test_space_yaml_round_trip(tmp_path: Path):
    sp = SearchSpace(
        name="demo",
        axes=[
            Axis("a", "categorical", values=[1, 2, 3]),
            Axis("b", "float", low=0.1, high=1.0),
            Axis("c", "int", low=1, high=10, step=1, active_if={"env": "dev"}),
        ],
    )
    p = tmp_path / "space.yaml"
    p.write_text(sp.to_yaml(), encoding="utf-8")
    loaded = load_space(p)
    assert loaded.name == sp.name
    assert {a.name for a in loaded.axes} == {"a", "b", "c"}
    c = loaded.axis_by_name("c")
    assert c.active_if == {"env": "dev"}
    # Conditional gate works after round-trip
    assert c.is_active({"env": "dev"}) is True
    assert c.is_active({}) is False


def test_parse_space_rejects_wrong_kind():
    raw = {"apiVersion": "bench/search/v1alpha1", "kind": "Profile", "name": "x", "axes": {}}
    with pytest.raises(ValueError, match="expected kind=SearchSpace"):
        parse_space(raw)
