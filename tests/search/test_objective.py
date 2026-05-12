from lmtune.search.objective import CallableObjective, ObjectiveResult, ScoreObjective


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


def test_score_objective_slo_miss_is_pruned_not_crash(tmp_path, monkeypatch):
    """R27: SLO 미달은 정상 측정 (낮은 score) — error 미set 으로 PRUNED 흐름.
    이전엔 error="...slo_pass=False" 가 set 되어 study.tell 이 CRASH 분류 →
    breaker 가 안정성 fail 카운트 → study 빨리 halt."""

    # Fake script + bench bin: ScoreObjective._run_one 을 monkeypatch 하는 게
    # 더 깨끗.
    ep = tmp_path / "ep.yaml"
    ep.write_text("model: x\nurl: http://x\n")
    profile = tmp_path / "p.yaml"
    profile.write_text("name: x\n")
    fake_script = tmp_path / "lmtune_score.py"
    fake_script.write_text("# stub\n")

    obj = ScoreObjective.__new__(ScoreObjective)
    obj.adapter = None
    obj.endpoint_path = ep
    obj.profile_paths = [profile]
    obj.repeats = 1
    obj.ttft_slo_ms = 500.0
    obj.script = fake_script

    # SLO miss simulation
    def fake_run(_self, _p):
        return {
            "score": 5.0,
            "ttft_p99": 800.0,
            "throughput_tok_avg": 10.0,
            "slo_pass": False,
            "accepted": True,
        }

    monkeypatch.setattr(ScoreObjective, "_run_one", fake_run)
    r = obj({"max_num_seqs": 64})

    # SLO 미달 → accepted=False (sampler 학습 신호) but error 는 None
    # (study.tell 이 CRASH 가 아닌 PRUNED 분류)
    assert r.accepted is False
    assert r.error is None, (
        f"R27: SLO 미달은 error 비어있어야 PRUNED 분류됨. got: {r.error}"
    )
    assert r.score == 5.0  # 측정값 보존
