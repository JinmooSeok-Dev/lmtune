# WorkloadSpec — `apiVersion: workloads/v1alpha1`

> Master = [lm-workloads](file:///home/jinmoo/ml_ai/workloads). lmtune 은 mirror (직접 re-export) + Provider 호스팅.

## 책임 분리

| 영역 | 누가 | 어디서 |
|:---|:---|:---|
| Schema 정의 (Pydantic 모델) | **lm-workloads** | `lm_workloads/spec/workload_spec.py` |
| Schema bump (v1alpha1 → v1alpha2) | **lm-workloads** (이 master 변경 시 lmtune mirror 도 같은 PR) | (그쪽 repo) |
| 운영 trace → WorkloadSpec 변환 | **lm-workloads** | `lm_workloads/orchestrate/pipeline.py::run_cycle` |
| WorkloadSpec → lmtune profile 변환 | **lm-workloads** | `lm_workloads/export/lmtune_profile.py::LmtuneProfileExporter` |
| lmtune CLI 진입 + Provider 호스팅 | **lmtune** | `src/lmtune/contracts/workload_spec.py` (re-export) + `src/lmtune/workload/providers/` |
| lmtune 안에서 yaml 검증 | **lmtune** | `lmtune contracts validate workload <path>` |

→ **drift 0**: lmtune 의 `WorkloadSpec` = `from lm_workloads.spec.workload_spec import WorkloadSpec` 직접 re-export. `[workloads]` extra 가 dep 으로 들어옴.

## WorkloadProvider ABC

```python
# src/lmtune/workload/providers/base.py
class WorkloadProvider(ABC):
    """WorkloadSpec 을 어디서 가져오든 동일 인터페이스."""

    @abstractmethod
    def provide(self, *, refresh: bool = False) -> WorkloadSpec: ...

    def fingerprint(self) -> str:
        """Cache key — 같은 입력은 같은 fingerprint."""
        ...
```

### 구현체 — 본 PR 에서 2개

| Provider | 입력 | 동작 | 의존 |
|:---|:---|:---|:---|
| `LiteralWorkloadProvider(yaml_path)` | yaml 파일 경로 | yaml read → Pydantic validation → WorkloadSpec | 없음 |
| `LMWorkloadsProvider(source, options)` | `vllm-log:/path` 같은 URI | lm-workloads `run_cycle()` 호출 → WorkloadSpec | `lm-workloads` (`[workloads]` extra) |

### 미래 확장 (entry_points 로 외부 패키지 등록)

| Provider | 의도 |
|:---|:---|
| `PromWorkloadProvider` | Prometheus query 직접 (lm-workloads 우회) |
| `LokiWorkloadProvider` | Loki 로그 |
| 사내 logger Provider | 운영 환경별 |

## CLI 표면

```bash
# 직접 yaml (BYO) — LiteralWorkloadProvider
lmtune run --workload-spec ws.yaml -p profile.yaml -e endpoint.yaml

# lm-workloads 호출 (Discover) — LMWorkloadsProvider
lmtune run --workload-source vllm-log:/var/log/vllm/access.log \
           -p profile.yaml -e endpoint.yaml

# yaml 만 만들고 끝 (단독 명령, 다음 run 에선 BYO 로 재사용)
lmtune workload generate --source vllm-log:/path --out ws.yaml

# 미지정 시 기존 동작 유지 (profile.yaml 의 inline workload 사용)
lmtune run -p profile.yaml -e endpoint.yaml
```

## 데이터 흐름

```
[원천]                          [Provider]                      [내부]
ws.yaml          ──→ LiteralWorkloadProvider ──┐
사용자 hand-write  ──┘                            ├──→ WorkloadSpec
                                                   │      (Pydantic obj,
vllm-log file    ──→ LMWorkloadsProvider ────────┘       lm-workloads master)
                                                   │
                                                   ▼
                                       lm_workloads.export.LmtuneProfileExporter
                                                   │
                                                   ▼
                                      lmtune ProfileSpec.workload (override)
                                                   │
                                                   ▼
                                          기존 cmd_run 흐름 (runner.run, ...)
```

## Sidecar 인식

lm-workloads 의 `LmtuneProfileExporter` 가 손실 정보를 두 곳에 보존:
- `runner_overrides.lmtune_extensions` — profile yaml 안 (재현 가능)
- `<slug>.warnings.json` — sidecar 파일 (사람 검토용)

lmtune 측은 위 두 영역을 **DuckDB `runs` 테이블의 메타로 저장** (현 PR 에선 보존만, 사용은 후속).

## Acceptance — 본 PR (`lmtune#WS`)

1. ✅ `from lmtune.contracts.workload_spec import WorkloadSpec` import 가능
2. ✅ `[workloads]` extra 미설치 시 import 시점에 친절한 에러 (`pip install lmtune[workloads]`)
3. ✅ `LiteralWorkloadProvider(path).provide()` 가 yaml read → Pydantic validation
4. ✅ `LMWorkloadsProvider(source).provide()` 가 lm-workloads examples fixture 로 e2e 동작
5. ✅ `lmtune run --workload-spec ws.yaml -p profile.yaml -e endpoint.yaml` 정상 종료
6. ✅ `lmtune workload generate --source vllm-log:/path --out ws.yaml` 정상 종료
7. ✅ 기존 96+ 테스트 PASS (cmd_run backward-compat)
8. ✅ 신규 단위 테스트 ≥ 5건 + e2e 테스트 1건

## 변경되는 파일

```
src/lmtune/contracts/__init__.py            (신규)
src/lmtune/contracts/workload_spec.py       (신규 — re-export)
src/lmtune/workload/__init__.py             (신규)
src/lmtune/workload/providers/__init__.py   (신규)
src/lmtune/workload/providers/base.py       (신규 — ABC)
src/lmtune/workload/providers/literal.py    (신규)
src/lmtune/workload/providers/lm_workloads.py (신규 — [workloads] extra)
src/lmtune/workload/cache.py                (신규 — fingerprint + TTL)
src/lmtune/cli_workload.py                  (신규 — `lmtune workload` subcommand)
src/lmtune/cli.py                           (수정 — cmd_run 에 --workload-spec/--workload-source flag)
pyproject.toml                              (수정 — [workloads] extra)
docs/contracts/{README,workload-spec}.md    (신규)
docs/architecture/REFACTOR-PLAN.md          (신규)
tests/workload/__init__.py                  (신규)
tests/workload/test_providers.py            (신규)
```

## 변경되지 않는 영역 (backward-compat)

- `src/lmtune/profiles.py` 의 기존 `Workload Union` (Synthetic/Dataset/Trace) — 그대로 유지. 후속 PR 에서 점진적 정리.
- `cmd_run` 의 기존 흐름 — `--workload-spec` 미지정 시 동작 동일.
- 기존 96+ 테스트 — 변경 0.
