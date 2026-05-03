# 탐색·민감도 분석 도구 스택 (2026-04 기준)

| 메타 | 값 |
|:---|:---|
| 최종 검토일 | 2026-04-23 |
| 적용 범위 | `src/bench/search/**` + `src/bench/visualization/plots/{pareto,sobol_bar,search_trace}.py` |
| 소유자 | jinmoo |
| 유효기간 | 2026-10 이후 재점검 — Optuna 5.x / BoTorch 1.x 릴리스 사이클 기준 |
| 상태 | active (Phase S5 반영) |

## 1. Executive Summary

이 프로젝트는 vLLM + llm-d 엔드포인트 구성을 지속적으로 튜닝하는 autotune 플랫폼이다. Phase S 는 단일 호스트 탐색(S1) → Bayesian + pruning(S2) → 분산 실행(S3) → 실배포 adapter(S4) → 다중목적·민감도·자체 구현(S5) 의 5 단계로 구성된다. 본 문서는 S5 시점(2026-04)에 **어떤 라이브러리를 왜 택했는지**, **후일 어떻게 교체 가능한지** 를 기록한다.

→ _"구체적으로 지금 어떤 버전을 쓰고 있는가?"_

## 2. 현재 스택

| 라이브러리 | 버전 | 역할 | 왜 선택했나 |
|:---|:---|:---|:---|
| `optuna` | 4.8.0 | 샘플러 허브 (Grid/Random/TPE/CMA-ES/NSGA-II/III/MOTPE/QMC) + study 상태 추상화 | 2024-11 v4 릴리스 이후 multivariate TPE 와 conditional axis group(`group=True`) 이 정식. 2026-04 현재 HPO 의 de-facto 중심. |
| `SALib` | 1.5.2 | 전역 민감도 (Sobol / Morris / FAST) | 2026 시점 global sensitivity 표준. Sobol-Saltelli 샘플링의 공식 구현. |
| `scipy` | 1.17.1 | Latin Hypercube (`stats.qmc`), `gaussian_kde` (native TPE), one-way ANOVA | numpy/HPO 생태계의 기반. |
| `scikit-learn` | 1.8.0 | `RandomForestRegressor` (feature importance + Sobol surrogate) | Sobol 을 위한 cheap surrogate 에 가장 간결. |
| `kubernetes` | 35.0.0 | K8s Job 제출 (S3 분산 백엔드) | 공식 Python client. 단, pod log 읽기에 이슈 — 아래 [주의] 참조. |

→ _"왜 BoTorch/Ax 가 아닌가?"_

## 3. 대안과 교체 기준

### 3.1 BoTorch + Ax (Meta)

`Ax 0.6+` / `BoTorch` 는 **Gaussian Process 기반 고차원 연속축** HPO 에 강하다. 벡터화된 acquisition function (qEI, qNEI 등) 으로 동일 budget 대비 탐색 효율이 TPE 보다 우수한 영역이 있으며, Meta·OpenAI 계열 논문이 2024-2025 에 다수 채택.

**우리가 지금 채택하지 않은 이유**:

- **Trial 수 vs cost 비율**: BoTorch 의 이점은 수천 trial 이상 / per-trial 비용이 낮을 때 명확. 우리는 per-trial 3~7 분 × 수십~수백 trial 스케일 → Optuna TPE 로 이미 수렴. GP 적합 비용이 trial 수의 O(N³) 라는 점도 중~대규모에서 부담.
- **혼합 공간 처리**: 우리 공간은 categorical(bool, kv_cache_dtype, max_model_len) + continuous(gpu_memory_utilization) + int(tp/pp/dp) 가 섞여있다. TPE(multivariate + group) 가 깔끔. BoTorch 는 MixedSingleTaskGP 등으로 가능하지만 추가 코드 필요.
- **디펜던시 무게**: `botorch` 가 `gpytorch` + `torch` 를 끌어옴 (~2GB). 경량 runner 이미지(865MB) 에 포함시키면 4~5GB. K8s Job per trial 로 pull 비용이 매 trial 곱해짐.

**교체 지점**: `src/bench/search/samplers/__init__.py::make_sampler()` 의 dispatch. strategy="botorch" 를 추가하고 `optuna-integration` 또는 `botorch.acquisition` 직접 래핑. Phase S5 에서 해당 seam 을 이미 준비 (Phase S5 follow-up #4 "BoTorch backend adapter").

### 3.2 Ray Tune

분산 backend 의 대안. 강력하지만 우리는 **K8s Job-per-trial** 을 이미 S3 에서 구현했고, llm-d 가 K8s 네이티브이므로 Ray 계층을 끼우는 이득이 적다.

### 3.3 Optuna 내 대안

- **QMCSampler**: scipy LatinHypercube 와 기능 중복. 우리는 native LHC 를 직접 구현(교육 목적) + Optuna QMC 은 샘플러 팩토리에 strategy="qmc" 로 언제든 추가 가능.
- **MOTPE**: NSGA-II 의 multi-objective 대안. 우리는 NSGA-II 채택 — 더 해석 용이 + 대규모 population 에 강함. 3+ 목적에선 NSGA-III (이미 `make_nsga3()` 구현).

→ _"Optuna pod log / SDK 사용시 알려진 함정?"_

## 4. [주의] 2026-04 시점 알려진 이슈

### 4.1 `kubernetes` Python SDK (35.0.0) 의 pod log 파싱

`client.CoreV1Api().read_namespaced_pod_log()` 가 JSON 형태 라인을 Python dict repr 로 변환해 반환한다. 예:

```
pod stdout: {"status":"completed","score":99.9}
SDK return: "{'status': 'completed', 'score': 99.9}"
```

busybox `echo '{"ok":true,"x":42}'` 로도 재현. 원인은 SDK 가 structured logging 을 감지해 `ast.literal_eval`-like 변환을 시도하는 것으로 추정.

**우리 워크어라운드**: `K8sJobBackend.poll()` 이 `kubectl logs` CLI 를 subprocess 로 호출한다. SDK 경로는 fallback 으로 유지. 해당 이슈가 SDK 36+ 에서 해소되면 subprocess 경로를 제거.

**관련 파일**: `src/bench/orchestrate/backend_k8s.py::K8sJobBackend.poll()`

### 4.2 Optuna `multivariate=True`, `group=True` ExperimentalWarning

TPESampler 에서 conditional axis (`active_if`) 를 올바르게 처리하려면 `group=True` 필요. 2024-11 v4 이후 안정 동작이지만 `ExperimentalWarning` 여전히 뜬다. 필요 시 `warnings.filterwarnings` 로 억제.

### 4.3 DuckDB 단일 writer

DuckDB 는 한 파일에 단일 프로세스 writer 만 허용. S3 의 `Study.storage.suspend() / resume()` 은 자식 subprocess 가 lmtune run → DuckDB 쓰기를 해야 할 때 부모가 잠시 락을 풀어주는 shim. 분산 스케일에서 writer queue (S3 `writer_queue.py`) 로 대체 예정.

→ _"앞으로 재점검 시 확인할 포인트?"_

## 5. K8s E2E (Phase S5 follow-up #3) — llm-d P/D 분리 배치 검증

| 항목 | 값 |
|:---|:---|
| 환경 | minikube + cri-dockerd, RTX 3060(8 GB) + RTX 5060 Ti(16 GB) |
| 모델 | Qwen/Qwen2.5-1.5B (3 GB) |
| 배포 helmfile | `phase2/pd-qwen25-1.5b/helmfile.yaml.gotmpl` (peer repo `llm-distributed-inference`) |
| 릴리즈 | infra-pd-qwen25 / gaie-pd-qwen25 / ms-pd-qwen25 (decode + prefill) |
| KV 전송 | NIXL TCP (RDMA 없음) |
| 검증 run | `01KQC2BC272W6HN7XNJD550K1J` (`autotune-short`, guidellm runner, 30 req @ concurrency 8) |
| TTFT p99 | 91.2 ms |
| e2e p99 | 1.53 s |
| throughput | 796 tok/s avg |
| SLO | 3/3 pass |

### 5.1 배치 시 부딪힌 함정

- **agentgateway 가 `InferencePool` 미지원**: HTTPRoute backendRef 가 InferencePool 일 때 `InvalidKind` 로 거부됨. 우회는 (a) kgateway 사용 또는 (b) 본 검증처럼 decode pod 의 routing-proxy 컨테이너로 직접 port-forward.
- **decode pod 의 routing-proxy 는 `/v1/chat/completions` 만 P/D 분리**. `/v1/completions` 는 그대로 vLLM 본체로 전달됨.
- **이미지 풀링 ~30 분**: `ghcr.io/llm-d/llm-d-cuda:v0.5.1` 압축 7.3 GB / 풀어진 27 GB. minikube docker driver + cri-dockerd 경로로 풀이 진행되며 containerd `/var/lib/containerd/...content` 에는 진행 흔적이 남지 않는다 — 진행 추적은 `journalctl ... cri-dockerd ... Downloading`.
- **메모리 압박**: helm 차트의 EPP(`gaie-pd-qwen25-epp`) 가 기본 8 GiB 요청. 16 GiB minikube 노드에서 prefill 4 GiB + decode 4 GiB + EPP 8 GiB 가 다른 시스템 pod 와 충돌. EPP 를 `0.5/1 GiB` 로 다운사이즈 후 정상 스케줄.

### 5.2 endpoint config 정본

`configs/endpoints/llmd_pd_qwen25.yaml` 에 deployment 메타로 박제. `bench` 자동화 로직은 이 필드를 읽지 않으나, runs 테이블에 `endpoint_meta` JSON 으로 저장되어 후속 비교 / archive 시 컨텍스트가 보존된다.

→ _"앞으로 재점검 시 확인할 포인트?"_

## 6. 재점검 트리거

아래 조건 하나라도 충족 시 본 문서 업데이트:

- **Optuna 5.x 릴리스** (multi-objective API 변경 가능)
- **BoTorch backend 도입** (sampler adapter seam 으로 추가 시)
- **BoTorch 의 MixedSingleTaskGP 가 K8s runner 이미지에 fit 하도록 경량화**
- **SALib 의 Saltelli 샘플링 기본 변경** (현재 `sobol_from_history()` 가 `calc_second_order=False` 기본)
- **Trial 규모가 5 000+ 로 증가** → BoTorch 이점 우세 구간 진입

## 7. References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). ["Algorithms for Hyper-Parameter Optimization"](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html), NIPS 2011. — TPE.
- Deb, K. et al. (2002). ["A fast and elitist multiobjective genetic algorithm: NSGA-II"](https://ieeexplore.ieee.org/document/996017), IEEE TEC. — NSGA-II.
- Saltelli, A. (2002). ["Making best use of model evaluations to compute sensitivity indices"](https://www.sciencedirect.com/science/article/abs/pii/S0010465502002804), Comput. Phys. Comm. — Sobol-Saltelli.
- [Optuna 4.x release notes](https://github.com/optuna/optuna/releases) (Accessed 2026-04-23).
- [SALib docs](https://salib.readthedocs.io/en/latest/) (Accessed 2026-04-23).
- [BoTorch](https://botorch.org/) + [Ax Platform](https://ax.dev/) — 후일 대규모 trial 스케일 도입 시 참조.
