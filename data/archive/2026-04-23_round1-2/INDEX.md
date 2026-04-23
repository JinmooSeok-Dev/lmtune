# Archive — Round 1 / Round 2 manual autotune (2026-04-23)

Phase S0 cleanup 시점 직전의 수동 autotune 결과물. Phase S 의 `bench search` 기반 신규 플랫폼 진입 전에 격리·보존함.

## 생성 정보

- **Archive 일자**: 2026-04-23
- **직전 commit (HEAD)**: `ebece34` (`autoresearch.sh: wire BENCH_BIN to venv bench`)
- **이 archive 를 만든 커밋**: S0 cleanup commit (아래 `git log` 의 바로 다음 commit; 본 INDEX.md 가 그 커밋에 포함)
- **HW / SW 조건**: RTX 5060 Ti 16GB (Blackwell sm_120), CUDA 12.8, PyTorch 2.10, vLLM 0.19.1, guidellm 0.6.0, Python 3.12
- **탐색 방식**: 수동 선언한 `hypotheses_round*.json` 을 `scripts/autotune_run.py` 가 순회하는 grid 전용 orchestrator (신규 `bench search` 로 대체 예정)

## 산출물 목록

### hypotheses/
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `hypotheses_round1.json` | `data/autotune/hypotheses_round1.json` | 7 config × engine_args sweep (Qwen2.5-1.5B 고정) |
| `hypotheses_round2.json` | `data/autotune/hypotheses_round2.json` | 8 config × 모델·양자화 sweep (Qwen2.5-1.5B / Qwen3 0.6B-14B + AWQ/FP8) |

### experiments/
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `experiments.jsonl` | `data/autotune/experiments.jsonl` | Round 1 실험 로그 (한 줄 = 한 config) |
| `experiments_round2.jsonl` | `data/autotune/experiments_round2.jsonl` | Round 2 실험 로그 |

### reports/
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `report_round1.md` | `data/autotune/report_round1.md` | Round 1 요약 (autotune_report.py 생성) |
| `report_round2.md` | `data/autotune/report_round2.md` | Round 2 요약 |
| `report_per_workload.md` | `data/autotune/report_per_workload.md` | cross-round per-workload 랭킹 (analyze_per_workload.py 생성) |

### logs/
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `round2_orchestrator.log` | `data/autotune/round2_orchestrator.log` | Round 2 autotune_run.py stdout |

### autoresearch-legacy/
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `goal.md` | `autoresearch/goal.md` | Phase E7 autoresearch-claude-code 초기 goal 스펙 (repo 루트 `autoresearch.md` 에 흡수됨) |
| `search_space.yaml` | `autoresearch/search_space.yaml` | Phase E7 초기 탐색공간 선언 (Phase S `configs/search/spaces/*.yaml` 로 대체) |

### 본 디렉토리 루트
| 파일 | 원 위치 | 설명 |
|:---|:---|:---|
| `FINAL_REPORT.md` | `data/autotune/FINAL_REPORT.md` (원본 유지) | Round 1+2 최종 서사 리포트 사본 |
| `METHOD.md` | `data/autotune/METHOD.md` (원본 유지) | 4-layer 탐색 방법론 사본 |
| `bench_round1-2.duckdb` | `data/db/bench.duckdb` 의 복사본 (원본 유지) | Round 1/2 전체 runs/metrics/requests. Phase S1 warm-start 의 seed |

## DuckDB 건수 (archive 시점 기준)

```
runs:       161
metrics:    3600
requests:   2598
sessions:   0      -- sessions 테이블은 아직 사용 안 함
detections: 0      -- detect 결과는 기록 안 됨
```

## 원본 유지 자원

- **`data/raw/<run_id>/`** — guidellm 원본 artifact. `.gitignore` 로 untracked 상태 유지. DuckDB `runs` 테이블의 `run_id` 로 링크 가능.
- **`data/db/bench.duckdb`** — 운영 DB. Phase S1 이후 새 `studies`, `trials`, `trial_metrics` 테이블이 여기에 추가됨. 과거 `runs` 데이터는 그대로 살아남아 warm-start 소스로 재사용.
- **`data/autotune/FINAL_REPORT.md`, `METHOD.md`** — 문서 가치로 원 위치 유지. archive 사본은 읽기 편의.

## Phase S 에서의 사용처

1. **Warm-start (S1)**: `bench search warmstart` 가 `bench_round1-2.duckdb` 를 읽어 과거 best config 를 Optuna `enqueue_trial()` 로 seed.
2. **Regression 회귀 테스트 (S1 Acceptance)**: grid sampler 가 `hypotheses_round*.json` 과 동일 결과를 재현하는지 비교.
3. **히스토리 기반 공간 축소 (S2)**: `bench search prune` 의 ANOVA / feature importance 입력이 이 archive DB.

## 복원 방법

원래 위치로 되돌리려면:

```bash
# hypotheses / reports / logs / experiments / autoresearch-legacy 원위치로
git mv data/archive/2026-04-23_round1-2/hypotheses/* data/autotune/
git mv data/archive/2026-04-23_round1-2/reports/*   data/autotune/
mv data/archive/2026-04-23_round1-2/experiments/*   data/autotune/
mv data/archive/2026-04-23_round1-2/logs/*          data/autotune/
mkdir -p autoresearch && git mv data/archive/2026-04-23_round1-2/autoresearch-legacy/* autoresearch/
```

DuckDB 는 원본이 이미 제 위치에 있으므로 복원 불필요.
