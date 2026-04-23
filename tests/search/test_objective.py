from bench.search.objective import CallableObjective, ObjectiveResult


def test_callable_objective_scalar_return():
    obj = CallableObjective(lambda p: p["x"] * 2.0)
    r = obj({"x": 3})
    assert isinstance(r, ObjectiveResult)
    assert r.score == 6.0
    assert r.accepted is True


def test_callable_objective_tuple_return_with_dict_metrics():
    def f(p):
        return p["x"] * 2.0, {"err": 0.5, ("extra", "short"): 10.0}

    obj = CallableObjective(f)
    r = obj({"x": 3})
    assert r.score == 6.0
    assert r.metrics[("err", None)] == 0.5
    assert r.metrics[("extra", "short")] == 10.0


def test_callable_objective_catches_exceptions():
    def boom(_p):
        raise RuntimeError("nope")

    obj = CallableObjective(boom)
    r = obj({})
    assert r.accepted is False
    assert r.score == 0.0
    assert "nope" in (r.error or "")
