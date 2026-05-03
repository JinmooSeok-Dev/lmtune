from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricDelta:
    metric: str
    p: str
    baseline: float
    candidate: float
    delta_abs: float
    delta_pct: float


@dataclass
class RunComparison:
    baseline_run_id: str
    candidate_run_id: str
    deltas: list[MetricDelta]
    regressions: list[MetricDelta]  # delta_pct > regression_threshold
    improvements: list[MetricDelta]

    def to_markdown(self, regression_threshold_pct: float = 10.0) -> str:
        lines = [
            f"# Run Comparison",
            f"- baseline: `{self.baseline_run_id}`",
            f"- candidate: `{self.candidate_run_id}`",
            f"- regression threshold: ±{regression_threshold_pct}%",
            "",
            "| metric | p | baseline | candidate | Δabs | Δ% |",
            "|:-------|:-:|---------:|----------:|-----:|---:|",
        ]
        for d in self.deltas:
            lines.append(
                f"| {d.metric} | {d.p} | {d.baseline:.2f} | {d.candidate:.2f} | {d.delta_abs:+.2f} | {d.delta_pct:+.2f}% |"
            )
        if self.regressions:
            lines.append("\n## 🔻 Regressions")
            for d in self.regressions:
                lines.append(f"- {d.metric}[{d.p}]: {d.baseline:.2f} → {d.candidate:.2f} ({d.delta_pct:+.2f}%)")
        if self.improvements:
            lines.append("\n## 🟢 Improvements")
            for d in self.improvements:
                lines.append(f"- {d.metric}[{d.p}]: {d.baseline:.2f} → {d.candidate:.2f} ({d.delta_pct:+.2f}%)")
        return "\n".join(lines) + "\n"


from bench.analysis.registry import direction_of


def compare_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_metrics: dict[str, dict[str, float]],
    candidate_metrics: dict[str, dict[str, float]],
    regression_threshold_pct: float = 10.0,
) -> RunComparison:
    deltas: list[MetricDelta] = []
    for metric, bbucket in baseline_metrics.items():
        cbucket = candidate_metrics.get(metric) or {}
        for p, bval in bbucket.items():
            if p not in cbucket:
                continue
            cval = cbucket[p]
            dabs = cval - bval
            dpct = (dabs / bval * 100.0) if bval else 0.0
            deltas.append(MetricDelta(metric, p, bval, cval, dabs, dpct))

    regressions: list[MetricDelta] = []
    improvements: list[MetricDelta] = []
    for d in deltas:
        direction = _regression_sign(d.metric)
        if direction == 0:
            continue
        if direction * d.delta_pct > regression_threshold_pct:
            regressions.append(d)
        elif direction * d.delta_pct < -regression_threshold_pct:
            improvements.append(d)

    return RunComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        deltas=deltas,
        regressions=regressions,
        improvements=improvements,
    )


def _regression_sign(metric: str) -> int:
    d = direction_of(metric)
    if d == "lower_better":
        return 1
    if d == "higher_better":
        return -1
    return 0
