"""BenchmarkResult → list[RecordSpec] 변환.

driver 가:
    result = runner.emit_result(...)
    records = to_records(result)
    store.put(records)

으로 호출하기 위한 단일 entry point. 한 번의 emit 으로 RunRecord 1 + Metric N
+ Request N + Session N + TrajectoryEvent N 이 한꺼번에 dispatchable.
"""

from __future__ import annotations

from lmtune.contracts.record_spec import (
    MetricRecord,
    RecordSpec,
    RequestRecord,
    RunRecord,
    SessionRecord,
    TrajectoryEventRecord,
)
from lmtune.contracts.result_spec import BenchmarkResult


def to_records(result: BenchmarkResult) -> list[RecordSpec]:
    """BenchmarkResult → 다중 RecordSpec.

    포함:
      - 1 RunRecord
      - 0~N MetricRecord (metric × percentile 의 cross product)
      - 0~N RequestRecord
      - 0~N SessionRecord
      - 0~N TrajectoryEventRecord
    """
    records: list[RecordSpec] = []

    # ── RunRecord ──────────────────────────────────────────────────
    records.append(
        RunRecord(
            run_id=result.run_id,
            profile_slug=result.profile_slug,
            endpoint_slug=result.endpoint_slug,
            runner=result.runner_kind,
            status=result.status,
            started_at=result.started_at,
            finished_at=result.finished_at,
            profile_yaml=result.profile_yaml,
            endpoint_meta=result.endpoint_meta,
            git_sha=result.git_sha,
            tool_versions=result.tool_versions,
            error=result.error,
            trial_id=result.trial_id,
        )
    )

    # ── MetricRecord 들 ────────────────────────────────────────────
    # metrics: {metric_name: {p_or_avg: value}}
    for metric_name, bucket in result.metrics.items():
        for p, v in bucket.items():
            records.append(
                MetricRecord(
                    run_id=result.run_id,
                    metric=metric_name,
                    p=p,
                    value=float(v),
                )
            )

    # ── RequestRecord 들 ──────────────────────────────────────────
    for r in result.requests:
        records.append(
            RequestRecord(
                run_id=result.run_id,
                req_id=r.req_id,
                turn_idx=r.turn_idx,
                conversation_id=r.conversation_id,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                cached_tokens=r.cached_tokens,
                thinking_tokens=r.thinking_tokens,
                tool_call_count=r.tool_call_count,
                tool_result_tokens=r.tool_result_tokens,
                phase=r.phase,
                role=r.role,
                energy_wh=r.energy_wh,
                cost_usd=r.cost_usd,
                ttft_ms=r.ttft_ms,
                itl_mean_ms=r.itl_mean_ms,
                e2e_ms=r.e2e_ms,
                started_at=r.started_at,
                completed_at=r.completed_at,
                status=r.status,
                error=r.error,
            )
        )

    # ── SessionRecord 들 ──────────────────────────────────────────
    for s in result.sessions:
        records.append(
            SessionRecord(
                run_id=result.run_id,
                session_id=s.session_id,
                task_id=s.task_id,
                total_input_tokens=s.total_input_tokens,
                total_output_tokens=s.total_output_tokens,
                total_cached_tokens=s.total_cached_tokens,
                turn_count=s.turn_count,
                tool_call_count=s.tool_call_count,
                duration_ms=s.duration_ms,
                success=s.success,
                total_cost_usd=s.total_cost_usd,
                total_energy_wh=s.total_energy_wh,
            )
        )

    # ── TrajectoryEventRecord 들 ──────────────────────────────────
    for ev in result.trajectory:
        records.append(
            TrajectoryEventRecord(
                run_id=result.run_id,
                session_id=ev.session_id,
                seq=ev.seq,
                event_type=ev.event_type,
                ts=ev.ts,
                phase=ev.phase,
                tokens=ev.tokens,
                metadata=ev.metadata,
            )
        )

    return records
