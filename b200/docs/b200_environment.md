# B200 클러스터 환경 스냅샷

| 메타 | 값 |
|:---|:---|
| 최종 검토일 | _(사용자가 probe.sh 실행 후 갱신)_ |
| 적용 범위 | `b200/**` |
| 소유자 | jinmoo |
| 상태 | initializing |

> 본 문서는 사용자가 직접 B200 클러스터에서 `bash b200/scripts/probe.sh` 를 실행하고 결과로 갱신한다. 작성 시점에는 가정값으로 채워둔다.

## 1. Executive Summary

2 노드 × 8 B200 = 16-GPU k3s 클러스터에서 `lmtune search` autotuning 을 운용하기 위한 환경 정의. 본 문서가 단일 출처(single source of truth) 로 모든 helmfile / search-space / endpoint 가 참조한다.

→ _"실제로 어떤 하드웨어·소프트웨어 위에서 도는가?"_

## 2. 노드 인벤토리 _(probe.sh 실행 후 갱신)_

| 노드 | IP / hostname | GPU 수 | GPU 모델 | 드라이버 | RDMA NIC |
|:---|:---|:---|:---|:---|:---|
| node-1 | _TBD_ | 8 | NVIDIA B200 | _TBD_ | _TBD_ |
| node-2 | _TBD_ | 8 | NVIDIA B200 | _TBD_ | _TBD_ |

## 3. 소프트웨어 스택 _(probe.sh 실행 후 갱신)_

| 항목 | 값 |
|:---|:---|
| K8s | k3s _TBD_ |
| Container runtime | _TBD_ (containerd 권장) |
| CUDA driver | _TBD_ (≥ 12.6 권장 for B200) |
| nvidia-device-plugin | _TBD_ |
| Multus / SR-IOV CNI | _TBD_ |
| helmfile | _TBD_ |
| helm | _TBD_ |
| peer repo SHA | _TBD_ |
| llm-d 이미지 digest (cuda) | ghcr.io/llm-d/llm-d-cuda:v0.5.1 (사전 풀 권장) |
| llm-d 이미지 digest (routing-sidecar) | ghcr.io/llm-d/llm-d-routing-sidecar:v0.5.1 |

## 4. 인터노드 fabric

| 항목 | 값 |
|:---|:---|
| Fabric 종류 | _TBD_ (InfiniBand / RoCE / TCP) |
| 측정 대역폭 | _TBD_ (iperf3) |
| nccl-tests all-reduce 8M-1G | _TBD_ |
| RDMA 사용 가능 여부 | _TBD_ |

## 5. 알려진 제한 / 주의

- **k3s 기본 CNI 가 RDMA·SR-IOV 미지원** — Multus + SR-IOV NetworkAttachmentDefinition 사전 설치 필요
- **agentgateway 가 InferencePool 미지원** (S5f-3 함정 메모) — 가능하면 kgateway 사용 권장
- **B200 sm_100 + NCCL ≥ 2.23 필요** — 호환성 미충족 시 vLLM 부팅이 ptxas 에러로 실패
- **이미지 풀 시간** — `ghcr.io/llm-d/llm-d-cuda:v0.5.1` 약 27 GB. 첫 풀 30분+, 사전 풀 권장

## 6. 환경 변경 시 영향 범위

| 변경 | 영향 | 갱신 필요 파일 |
|:---|:---|:---|
| 모델 추가 | helmfile values, endpoint, search-space 모델군 | `b200/helmfile/<path>/values-<model>.yaml`, `b200/endpoints/`, `b200/search-spaces/b1_baselines.yaml` |
| 노드 수 변경 | parallelism 조합 가용성 | `b200/search-spaces/b3_parallelism.yaml` 의 active_if |
| RDMA 활성화 변경 | NCCL 환경변수, NetworkAttachmentDefinition | `b200/helmfile/base/values-b200-common.yaml.gotmpl` |
| llm-d 이미지 업데이트 | image tag, helmfile chart version | base values + 각 path helmfile 의 `version:` |

## 7. probe.sh 결과 archive

- `b200/studies/B0_smoke/probe.txt` (사람이 읽는 출력)
- `b200/studies/B0_smoke/probe.json` (`probe.sh --json` 결과)

## 8. References

- llm-d v0.5 release notes: <https://github.com/llm-d/llm-d/releases>
- gateway-api-inference-extension: <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- B200 spec: <https://www.nvidia.com/en-us/data-center/b200/>
- 작업 흐름 정의: `(internal dev plan, not in repo)` (Phase B 섹션)
