# Storage RecordSpec + QuerySpec — `lmtune/{record,query}/v1alpha1`

> Master = lmtune. Storage abstraction (`lmtune#SS`) PR 의 prerequisite contract subset.
> `WorkloadStore` / `ArtifactStore` ABC 의 입력/조회 schema 를 정형화 — backend (DuckDB, Postgres, Parquet, S3) 무관 동일 호출.

## 책임 분리

| 영역 | 누가 | 어디서 |
|:---|:---|:---|
| Schema 정의 (Pydantic 모델) | **lmtune** | `src/lmtune/contracts/record_spec.py`, `query_spec.py` |
| Schema bump | lmtune | 본 contract 관리 |
| Storage backend 구현 | **lmtune#SS** PR | `src/lmtune/storage/store/` (후속) |
| RecordSpec 직접 emit (runner 가 result → records) | lmtune#R0 | runner 별 변환 |

## RecordSpec — type-tagged union (10 kinds)

기존 `schema.sql` 의 9 테이블을 1:1 mirror + `trial_metric`. 각 kind 가 `kind` discriminator + 자기 PK 보유.

```python
from lmtune.contracts import RecordSpec, RunRecord, MetricRecord
from pydantic import TypeAdapter

# 어떤 kind 든 RecordSpec union 으로 검증/디스패치
adapter = TypeAdapter(RecordSpec)
rec = adapter.validate_python({"kind": "run", "run_id": "01ABC", ...})
# type(rec) == RunRecord (자동 분기)
```

| Record kind | 대응 테이블 | PK | 용도 |
|:---|:---|:---|:---|
| `run` | runs | (run_id,) | 단일 벤치 실행 메타 |
| `metric` | metrics | (run_id, metric, p) | run 단위 percentile/avg metric |
| `request` | requests | (run_id, req_id) | per-request agent 메타 + latency |
| `session` | sessions | (run_id, session_id) | 다중 turn session 집계 |
| `trajectory_event` | trajectory_events | (run_id, session_id, seq) | agent step (user/assistant/tool/thinking) |
| `prom_sample` | prom_samples | (run_id, ts, metric, labels_json) | Prometheus raw 샘플 |
| `detection` | detections | (run_id, detector, created_at) | 룰 기반 anomaly detection |
| `study` | studies | (study_id,) | search session 메타 |
| `trial` | trials | (trial_id,) | search 의 단일 trial |
| `trial_metric` | trial_metrics | (trial_id, metric, workload) | trial × metric × workload secondary |

**설계 결정**:
- `frozen=True` (불변) — store insert 후 mutation 금지, cache key 로 그대로 사용
- `extra=forbid` — typo 차단, schema drift 방지
- `primary_key()` 메서드 — store backend 의 dedup/upsert 키

## QuerySpec — 단순 query DSL

DuckDB SQL 로 자명 변환 가능한 단순 schema. 복잡 쿼리는 `raw_sql` escape hatch.

```python
from lmtune.contracts import QuerySpec, FilterCond, SortKey

q = QuerySpec(
    record_kind="trial",
    filters=[
        FilterCond(column="study_id", op="==", value="st-2026-04"),
        FilterCond(column="status", op="in", value=["completed"]),
    ],
    sort=[SortKey(column="score", direction="desc")],
    limit=10,
    select=["trial_id", "score", "params"],
)
# store.query(q) → list[TrialRecord] (SS PR 의 인터페이스)
```

**필드**:
- `record_kind`: 어느 RecordKind 를 조회할지 (run, metric, ..., trial_metric)
- `filters`: `[FilterCond]` — 컬럼별 비교 (`==`, `<`, `in`, `like`, `is_null`, ...)
- `sort`: `[SortKey]` — 다중 정렬 키 + 방향
- `limit`, `offset`: pagination
- `select`: 특정 컬럼만 (None = 전체)
- `aggregate`: `AggregateSpec(group_by, function, column)` — GROUP BY + 1 함수
- `raw_sql`: escape hatch — 위 필드 무시하고 직접 SQL

## CLI 표면

```bash
# 전체 RecordSpec union 의 JSON Schema
lmtune contracts dump-schema --kind record --out record.schema.json

# 특정 record kind 만
lmtune contracts dump-schema --kind record --record-kind run --out run.schema.json

# QuerySpec 의 JSON Schema
lmtune contracts dump-schema --kind query --out query.schema.json

# 단일 record yaml 검증
cat > rec.yaml <<EOF
kind: run
run_id: 01ABC
profile_slug: smoke
endpoint_slug: local-vllm
runner: guidellm
status: ok
EOF
lmtune contracts validate-record rec.yaml
# → ok  kind=run  primary_key=('01ABC',)
```

## SS PR 에서의 활용 (요약)

```python
# WorkloadStore / ArtifactStore ABC (lmtune#SS 에서 신설)
class ArtifactStore(ABC):
    @abstractmethod
    def put(self, records: list[RecordSpec]) -> None: ...

    @abstractmethod
    def query(self, spec: QuerySpec) -> list[RecordSpec]: ...

# DuckDBStore(ArtifactStore) — 기본 구현
# InMemoryStore(ArtifactStore) — 테스트용
# PostgresStore(ArtifactStore) — entry_points plug-in
```

→ runner / driver / output sinks 모두 본 ABC 만 의존. backend 교체 시 client 코드 변경 0.

## Acceptance criteria (본 PR)

1. RecordSpec 의 10 kind 가 모두 Pydantic 검증 + frozen + discriminator 동작
2. QuerySpec 의 `==/!=/</...../in/like/is_null` op 지원
3. `lmtune contracts dump-schema --kind record/query` 가 valid JSON Schema 출력
4. `lmtune contracts validate-record <yaml-or-json>` round-trip
5. 단위 테스트 ≥ 30 케이스 (record × kind, query DSL, CLI E2E)
6. 전체 회귀 테스트 PASS (407+ tests)

## Non-goals (본 PR)

- Storage backend 구현 (`DuckDBStore`, `InMemoryStore`) — `lmtune#SS` PR
- Runner 의 RecordSpec emit 변환 — `lmtune#R0` PR
- 기존 `runs/metrics/...` 테이블 직접 변경 — `lmtune#SS` 가 ALTER 없이 wrapper
- `device_perf_samples` (Tier 3 hardware bench) — `lmtune#R0` + `lmtune#OD` 후
