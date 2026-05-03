# Search Principles — what the autotuner is doing under the hood

Goal: make every trial answer "왜 이 값?" without leaving the dashboard.
Each section is one screen of reading; the dashboard side panel renders the
TL;DR.

## TPE (Tree-structured Parzen Estimator)

매 trial 마다 과거 trial 을 score 분위로 **good (top γ%)** vs **bad (rest)** 로 가른다.
두 그룹에서 **Parzen window** 로 분포 `l(x)` (good) / `g(x)` (bad) 를 추정한 뒤,
ratio `l(x)/g(x)` 가 큰 후보를 다음 trial 로 고른다.

- 직관: "score 가 높았던 영역의 분포를 닮은 점" 을 뽑는다 (exploit), 단
  bad 분포로 정규화하기 때문에 "좋은 곳에만 몰린 axis" 는 자동 페널티.
- 첫 `n_startup_trials` 개는 random — KDE 가 흔들리지 않게 warmup.

대시보드 표기: "TPE — exploit" / "TPE — warmup".

## NSGA-II (Non-dominated Sorting Genetic Algorithm II)

multi-objective 전용. 매 generation 에서 (1) **non-dominated sort** 로 trial 을
front 로 계층화, (2) 같은 front 안에서는 **crowding distance** 로 다양성 우선.
다음 세대는 상위 front 에서 토너먼트 + crossover + mutation.

- 직관: Pareto front 를 따라 점들을 골고루 펴는 게 목표 — "TTFT 낮은 점"
  과 "throughput 높은 점" 사이 빈 구간을 채우려 함.

대시보드 표기: front 위 점에 "Pareto front" 배지, 그 외는 "dominated".

## CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

continuous 축 전용. multivariate Gaussian `N(m, σ² C)` 에서 후보군을 뽑고,
score 상위 점들로 (a) 평균 `m` 갱신, (b) covariance `C` 갱신 (좋은 방향
으로 길게 늘림), (c) step size `σ` 갱신 (탐색 폭 자동 조절).

- 직관: "어느 방향으로 가야 점수가 올라가는지" 를 covariance 가 학습.
  완만한 ridge 형 surface 에 강함.

## Sobol total-order index (sensitivity analysis)

`lmtune search sensitivity <study>` 가 출력. 각 axis 가 단독·상호작용 포함
**전체 출력 분산의 몇 % 를 설명** 하는지 정량화. 0.05 미만이면 axis freeze
권고 (`lmtune search prune`).

- 직관: "이 axis 를 고정해도 score 가 바뀔까?" 의 통계적 답.

## Successive Halving / Hyperband (pruner)

자원이 적을수록 많은 후보를, 자원이 많을수록 살아남은 소수에게 — 의 다단계
배분. 우리는 N=3 → N=5 escalation gate 로 단순화 (CV ≥ 0.10 시 N=5 재실행).

## Cost tier (axis별 적용 비용)

| tier | 의미 | 적용 시간 |
|:---|:---|:---|
| 1 | random seed | <1s |
| 2 | container env | 5s |
| 3 | helmfile values | 1-3min |
| 4 | engine_args (vLLM restart) | 30s-2min |
| 5 | kernel cmdline | reboot |

샘플러에게 cost 를 함께 알려주면 "값이 같으면 cost 낮은 쪽" 을 우선.
