# PLUG 패턴 — 새 backend / sampler 추가하는 법

> 본 문서는 lmtune 의 **PLUG 패턴** (REFACTOR-PLAN 핵심 원칙 #2 "모든 layer 가
> ABC + 구현체") 을 외부 기여자가 1 PR 로 확장할 수 있도록 step-by-step 으로
> 영속화한다. 이미 두 axis (Storage backend, Tuner sampler) 에서 시연됐으며 본
> 문서는 그 형식을 재현 가능한 recipe 로 코드화.

## 두 PLUG axis 의 현황 (2026-05-06 기준)

| 추상 (ABC) | 위치 | 첫 빌트인 | 두 번째 빌트인 | 세 번째 (Native) | PLUG stub | extras |
|:---|:---|:---|:---|:---|:---|:---|
| `ArtifactStore` | `lmtune.storage.store.base` | `DuckDBArtifactStore` | `LocalArtifactStore` | `InMemoryArtifactStore` | `PostgresArtifactStore` (#58) | `[postgres]` |
| `Sampler` | `lmtune.tuner.base` | `OptunaSamplerAdapter` (TPE/NSGA-II/CMA-ES 6종) | `Native{Random,LHC,TPE}` | — | `LLMOracleSampler` (#59) | `[agent]` |

다른 layer 의 ABC (`Pruner`, `TrialBackend`, `DeploymentAdapter`, `Runner`) 도
같은 패턴을 따른다 — 본 문서의 5단계가 그대로 적용.

## 5단계 (대표: 새 ArtifactStore backend 추가)

### 1. ABC 구현체 1 파일 작성

`src/lmtune/storage/store/<backend_name>.py`:

```python
"""<BackendName>ArtifactStore — <한 줄 설명>."""

from __future__ import annotations

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RecordSpec
from lmtune.storage.store.base import ArtifactStore


class FooArtifactStore(ArtifactStore):
    def __init__(self, dsn: str, *, optional_args=...):
        # External SDK optional import — 미설치 시 친절 ImportError
        try:
            import foo_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "foo_sdk is required for FooArtifactStore — "
                "install with: pip install 'lmtune[foo]'"
            ) from e
        self.dsn = dsn

    def put(self, records: list[RecordSpec]) -> int: ...
    def query(self, spec: QuerySpec) -> list[RecordSpec]: ...
    def close(self) -> None: ...
```

**규칙**:
- ABC 의 3 메서드 (`put`, `query`, `close`) 를 모두 구현 (또는 `NotImplementedError`
  로 명시 — stub 단계 OK).
- 외부 SDK 가 필요하면 **`__init__` 안에서 lazy import**. 모듈 import 자체는
  실패하지 않음 → 미설치 환경에서도 type check / isinstance 가능.
- ImportError 메시지에는 `lmtune[<extra>]` 정확히 인용 — drift 테스트 (#60) 가
  검증.

### 2. `store/__init__.py` re-export 추가

```python
from lmtune.storage.store.foo import FooArtifactStore

__all__ = [..., "FooArtifactStore"]
```

### 3. `pyproject.toml` 의 optional extra 등록

```toml
[project.optional-dependencies]
foo = ["foo-sdk>=1.0"]
```

### 4. CLI 디스패치 합류 (`src/lmtune/cli_storage.py`)

```python
_BACKENDS = ("local", "duckdb", "postgres", "foo")  # ← 1줄

def _open_store(kind: str, path: Path) -> ArtifactStore:
    ...
    if kind == "foo":                                 # ← 4줄
        try:
            return FooArtifactStore(str(path))
        except ImportError as e:
            raise typer.BadParameter(str(e)) from None
    ...
```

이 1+4 줄 변경으로 **`lmtune storage migrate --src-kind foo`,
`lmtune storage info --kind foo`, `lmtune storage list-backends`** 모두
즉시 동작. 기존 명령은 변경 0.

### 5. Acceptance test (`tests/storage/test_<backend>_stub.py`)

기존 `test_postgres_store_stub.py` 를 그대로 복사 + backend 이름만 변경:

```python
def test_foo_is_artifact_store():
    assert issubclass(FooArtifactStore, ArtifactStore)

@pytest.mark.skipif(_HAS_FOO, reason="foo_sdk 설치된 환경에서는 ImportError 분기 없음")
def test_foo_import_error_message_when_missing():
    with pytest.raises(ImportError) as ei:
        FooArtifactStore("dummy://x")
    assert "foo_sdk" in str(ei.value)
    assert "lmtune[foo]" in str(ei.value)

def test_foo_in_cli_backends_list():
    assert "foo" in _BACKENDS
    result = runner.invoke(app, ["list-backends"])
    assert "foo" in result.output
```

**최소 case**:
1. ABC subclass 검증 (항상 통과 — type level)
2. SDK 미설치 시 ImportError 메시지가 extras 키 정확 인용 (drift 가드)
3. `_BACKENDS` 합류 + CLI 노출

(실제 구현이 들어오면 put/query round-trip 검증도 추가.)

## Sampler 의 경우 (미세 차이만)

| 단계 | ArtifactStore | Sampler |
|:---|:---|:---|
| 1 | `lmtune/storage/store/<name>.py` | `lmtune/tuner/<name>.py` |
| 2 | `store/__init__.py` re-export | (필수 X — factory 가 직접 import) |
| 3 | pyproject extra | pyproject extra (동일) |
| 4 | `cli_storage._BACKENDS` + `_open_store` | `tuner.factory._LLM_STRATEGIES` + `_make_llm` |
| 5 | `tests/storage/test_<name>_stub.py` | `tests/tuner/test_<name>_stub.py` |

`LLMOracleSampler` (#59) 가 reference impl — 그 파일 + 테스트를 그대로 복제하면 1 시간 안에 새 sampler PLUG 가능.

## 머지 가능한 stub 의 acceptance bar

stub 단계 (실제 구현 없이) 머지 가능:
- `__init__` 의 SDK 가드 (ImportError 메시지)
- ABC 메서드들이 `NotImplementedError("follow-up PR 에서 ...")` (사람이 읽을 수
  있는 hint 포함)
- factory / CLI 합류 + acceptance test 위 5종 통과

stub 머지 후 follow-up 으로 실 구현 단계적 채움. 본 패턴이 의도적으로 두는
stub 단계의 가치는 **새 axis 의 wiring 검증 (driver, CLI, tests 까지 모두 인식)
이 implementation 보다 먼저 검증되는 것**. wiring 이 깨지면 follow-up 의 구현
PR 들이 매번 wire-up 부터 다시 다툼 — 본 패턴이 차단.

## Drift 차단 — 두 곳을 같이 보는 테스트

PLUG 의 본질은 "한 번 틀어지면 install 자체가 깨지는" 형식의 정합성. 본 repo
는 다음을 영속 보증:

| Drift 위험 | 차단 위치 |
|:---|:---|
| ImportError 메시지 ↔ pyproject extras 키 | `tests/test_pyproject_plug_extras.py` (#60) |
| `__version__` ↔ pyproject `[project].version` | `tests/test_cli_version.py` (#64) |
| ABC ↔ 모든 구현체의 method signature | mypy/pyright (현 repo 미운용 — TODO) |
| factory dispatch 매핑 ↔ 실 클래스 | `tests/tuner/test_factory.py`, `tests/storage/test_cli_storage_*.py` |

새 PLUG 추가 시 위 4 영역 중 해당하는 곳에 case 1줄씩 추가.

## 기여 체크리스트

새 PLUG PR 의 description 에 본 체크리스트 체크 후 머지 가능:

- [ ] ABC 구현체 1 파일 (`src/lmtune/<layer>/<name>.py`)
- [ ] 외부 SDK lazy import + ImportError 친절 메시지 (`lmtune[<extra>]` 정확 인용)
- [ ] `<layer>/__init__.py` re-export (필요 시)
- [ ] `pyproject.toml` 의 `[<extra>]` optional-dependencies
- [ ] CLI / factory dispatch 매핑 합류 (1-5줄)
- [ ] Acceptance test 5종 (ABC subclass / ImportError 메시지 / drift / wiring / monitoring)
- [ ] `docs/architecture/REFACTOR-PLAN.md` CHANGELOG entry
- [ ] (옵션) README 의 PLUG 표에 row 추가

## 참고 — 두 reference impl

- **PostgresArtifactStore** (#58): `src/lmtune/storage/store/postgres.py` + `tests/storage/test_postgres_store_stub.py`
- **LLMOracleSampler** (#59): `src/lmtune/tuner/llm_oracle.py` + `tests/tuner/test_llm_oracle_stub.py`

위 두 PR 의 코드 + 테스트가 본 문서 5단계의 살아있는 1:1 reference. 새 PLUG
추가 시 둘 중 가까운 axis 를 그대로 복사 + 이름 / SDK 만 바꾸는 게 가장 빠른
경로.
