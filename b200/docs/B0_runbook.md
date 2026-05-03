# B0 RUNBOOK — B200 클러스터 온보딩

> 사용자가 B200 호스트에서 직접 실행하는 명령 시퀀스. 각 단계의 결과를 본 파일 또는 `b200/docs/b200_environment.md` 에 갱신.

## 0. 사전 준비 (B200 호스트)

```bash
# 작업 디렉토리
git clone <this-repo> ~/ml_ai/benchmark    # 또는 git pull
cd ~/ml_ai/benchmark

# python 환경
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,search,distributed]"

# 도구 (빠진 게 있으면 설치)
which kubectl helm helmfile yamllint || echo "missing tools — see b200/docs/b200_environment.md"

# kubeconfig 가 B200 클러스터를 가리키는지 확인
kubectl config current-context
kubectl get nodes -o wide
```

## 1. 환경 진단

```bash
bash b200/scripts/probe.sh | tee b200/studies/B0_smoke/probe.txt
bash b200/scripts/probe.sh --json > b200/studies/B0_smoke/probe.json
```

기대 출력:
- `gpu.total = 16` PASS
- `device_plugin` PASS
- `peer_repo`, `helmfile`, `helm` PASS
- `ghcr.io` PASS
- `rdma` PASS 또는 WARN (TCP 만이면 docs 에 기록)

문제 발생 시:
- WARN: `b200/docs/b200_environment.md` 에 "알려진 제한" 으로 기록 후 진행 가능
- FAIL: 해당 항목을 해결한 뒤 재실행

## 2. (옵션) 인터노드 fabric 측정

### 2a. K8s 기반 (NCCL all-reduce + iperf3)

```bash
kubectl apply -f b200/scripts/fabric_test.yaml

# 두 노드에 분산 스케줄 확인
kubectl get pods -n bench-fabric-test -o wide

# NCCL all-reduce 결과
kubectl logs -n bench-fabric-test job/nccl-test-allreduce | tail -50

# iperf3 — 다른 노드 IP 를 IPERF_SERVER 로 주입
NODE2_IP=$(kubectl get nodes -o jsonpath='{.items[1].status.addresses[?(@.type=="InternalIP")].address}')
kubectl set env -n bench-fabric-test job/iperf3-client IPERF_SERVER=$NODE2_IP
kubectl logs -n bench-fabric-test job/iperf3-client | tail -20

# 정리
kubectl delete -f b200/scripts/fabric_test.yaml
```

### 2b. Host-level RDMA Perftest baseline (권장)

`ib_write_bw` / `ib_read_bw` / `ib_send_bw` 로 호스트 직접 측정. 자세한 절차는 `b200/docs/rdma_perftest_baseline.md`.

```bash
# 노드 A (server) — 백그라운드로 3 test 동시
bash b200/scripts/rdma_bench.sh server &

# 노드 B (client) — server NIC IP 를 인자로
bash b200/scripts/rdma_bench.sh client <NODE_A_RDMA_IP>

# 결과 확인 (client 측에 summary.json 생성)
cat data/raw/rdma/$(ls -1t data/raw/rdma/ | head -1)/summary.json
```

NHN Cloud B200 reference 363.98 Gbps. 측정값을 `b200/docs/b200_environment.md` §4 에 기록.

## 3. HuggingFace 토큰 (gated 모델)

```bash
kubectl create ns b200-infsch || true
kubectl -n b200-infsch create secret generic huggingface-token \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 4. helmfile dry-run

```bash
cd b200/helmfile/inference-scheduling
helmfile -f helmfile.yaml.gotmpl --selector role=smoke apply --skip-deps --dry-run
```

성공 종료코드 0. release 3개 (infra-infsch / gaie-infsch / ms-infsch) 가 deploy 되어야 함을 보고.

## 5. 실제 배포 + 부팅 대기

```bash
helmfile -f helmfile.yaml.gotmpl --selector role=smoke apply

# 모델 다운로드 + 부팅 (Llama-3.1-8B ~ 16 GB) 5-15 분
kubectl get pods -n b200-infsch -w
```

기대: `ms-infsch-llm-d-modelservice-decode-*` 가 `2/2 Running`.

## 6. HTTPRoute 적용 (kgateway 사용 시)

```bash
kubectl apply -f httproute.yaml
```

agentgateway 인 경우는 InferencePool 미지원이라 직접 port-forward 사용 (S5f-3 함정 메모 참조).

## 7. Endpoint 노출

```bash
# 옵션 A: HTTPRoute + LoadBalancer / nodePort (kgateway)
GW_IP=$(kubectl get gateway -n b200-infsch infra-infsch-inference-gateway -o jsonpath='{.status.addresses[0].value}')
echo "endpoint: http://$GW_IP"

# 옵션 B: decode pod 직접 port-forward (agentgateway / smoke)
DECODE_POD=$(kubectl get pods -n b200-infsch -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n b200-infsch pod/$DECODE_POD 8011:8000 &

curl -s --max-time 10 http://127.0.0.1:8011/v1/models | head -50
```

`b200/endpoints/b200_smoke.yaml` 의 `url` 을 실측치로 갱신 (예: `http://<GW_IP>/v1` 또는 `http://127.0.0.1:8011/v1`).

## 8. lmtune run smoke

```bash
source .venv/bin/activate
lmtune run -p configs/profiles/autotune/short.yaml -e b200/endpoints/b200_smoke.yaml --json-summary | tee b200/studies/B0_smoke/run_smoke.log
```

기대:
- `status=ok`
- `slo_pass=true`
- TTFT p99 ≤ 500 ms (B200 + Llama-3.1-8B 에서 충분히 여유)

## 9. lmtune search smoke (4 trial)

```bash
lmtune search start \
  --strategy random \
  --space b200/search-spaces/b0_smoke.yaml \
  --endpoint b200/endpoints/b200_smoke.yaml \
  -p configs/profiles/autotune/short.yaml \
  -p configs/profiles/autotune/medium.yaml \
  --backend k8s-job --workers 2 --max-trials 4 \
  --name B0-smoke \
  | tee b200/studies/B0_smoke/search.log

# study_id 는 search start 출력의 첫 줄 'study_id=st-XXXX' 에서 확인
STUDY_ID=$(grep -oE 'st-[A-Z0-9]+' b200/studies/B0_smoke/search.log | head -1)
lmtune search status "$STUDY_ID"
```

기대:
- 4 trial 완주 (status=completed)
- `lmtune search status` 에 top-3 score 출력
- DuckDB `studies` / `trials` 테이블에 행 적재

## 10. pytest

```bash
pytest -q
```

기대: 기존 96+ 테스트 PASS. Phase B 의 새 신규 테스트가 추가되면 함께 PASS.

## 11. 정리 (smoke 완료 후 다음 phase 진입 전)

```bash
# helmfile destroy 또는 namespace 째 삭제
helmfile -f b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl destroy
kubectl delete ns b200-infsch
```

## 12. 산출물 commit

```bash
# 환경 문서 + probe 결과 + smoke run 로그
git add b200/docs/b200_environment.md b200/studies/B0_smoke/
git commit -m "B0: B200 cluster onboarding probe results + smoke runs"
```

## Acceptance

- [ ] probe.sh PASS (또는 WARN 로 기록 + 진행 동의)
- [ ] helmfile dry-run + apply 성공
- [ ] /v1/models 응답
- [ ] lmtune run smoke status=ok + slo_pass=true
- [ ] lmtune search start 4 trial 완주, DuckDB 적재
- [ ] pytest PASS
- [ ] 환경 문서 갱신 + commit
