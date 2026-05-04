# B200 + llm-d Quickstart — 3 well-lit paths in ~30 분

> 본 문서 = B200 클러스터 접속 후 **복붙으로 끝까지 가는** 명령 시퀀스.
> 더 자세한 진단 절차는 `b200/docs/B0_runbook.md`, 전체 컨텍스트는 `b200/README.md` 참조.
>
> **3 well-lit paths** (현재 자동 디스패치 가능):
> 1. `inference-scheduling` — Llama-3.1-8B single-GPU smoke (B0 검증용)
> 2. `pd-disaggregation` — Llama-3.1-70B Prefill/Decode 분리 + NIXL/UCX
> 3. `wide-ep-lws` — Mixtral-8x22B DP=2 EP=8 (MoE)

---

## 0. 전제 조건 (한 번만)

B200 호스트에 다음이 갖춰져 있어야 합니다:

| 도구 | 검증 명령 |
|:---|:---|
| `kubectl` + 클러스터 컨텍스트 | `kubectl get nodes -o wide` 가 GPU 노드 보여야 함 |
| `helm` v3 | `helm version` |
| `helmfile` | `helmfile version` |
| Python 3.11+ | `python --version` |
| nvidia-device-plugin | `kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'` 가 8 8 (또는 16) |
| HuggingFace 토큰 | gated 모델용. https://huggingface.co/settings/tokens |

부족하면 `b200/docs/B0_runbook.md` §0~§1 따라가면 됩니다.

---

## 1. 클론 + 설치 (5분)

```bash
git clone <this-repo> ~/ml_ai/lmtune
cd ~/ml_ai/lmtune

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,search,distributed]"

lmtune --help                          # 동작 확인
```

---

## 2. 클러스터 진단 (3분)

```bash
mkdir -p b200/studies/B0_smoke
bash b200/scripts/probe.sh | tee b200/studies/B0_smoke/probe.txt
```

**기대**: `gpu.total = 8` 또는 `16`, `device_plugin/peer_repo/helmfile/helm/ghcr.io = PASS`. RDMA 는 PASS 또는 WARN(TCP-only) 둘 다 OK.

FAIL 이 있으면 `b200/docs/B0_runbook.md` 의 해당 섹션 참조 후 재시도.

---

## 3. HuggingFace 토큰 secret 생성 (gated 모델 용)

3 path 의 모든 모델 (Llama-3.1-8B/70B, Mixtral-8x22B) 가 gated 입니다. 각 path 의 namespace 에 동일 secret 을 만들어 둡니다:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx     # 본인 토큰

for ns in b200-infsch b200-pd b200-wideep; do
  kubectl create namespace "$ns" 2>/dev/null || true
  kubectl create secret generic huggingface-token \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    -n "$ns" --dry-run=client -o yaml | kubectl apply -f -
done
```

---

## 4. Path 1 — inference-scheduling (Llama-8B smoke, ~10분)

가장 가볍습니다. 1 GPU 만 쓰므로 다른 워크로드와 GPU 자원만 안 겹치면 OK.

```bash
# 4.1 helmfile apply
NS=b200-infsch helmfile -f b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl apply

# 4.2 rollout wait
kubectl -n b200-infsch get pods -w     # vllm decode pod 가 Running 될 때까지 (모델 풀이 5-10분)
# (Ctrl+C 로 종료)
kubectl -n b200-infsch rollout status deployment --timeout=15m

# 4.3 endpoint 확인
SVC=$(kubectl -n b200-infsch get svc -o jsonpath='{.items[?(@.spec.ports[0].port==80)].metadata.name}')
kubectl -n b200-infsch port-forward svc/$SVC 8011:80 &
PF_PID=$!
sleep 3
curl -s http://localhost:8011/v1/models | jq

# 4.4 smoke autotune (4 trial, ~5분)
lmtune search start \
  --strategy random \
  --space b200/search-spaces/b0_smoke.yaml \
  --endpoint b200/endpoints/b200_smoke.yaml \
  -p configs/profiles/autotune/short.yaml \
  --backend inline --max-trials 4 \
  --name b0-infsch-smoke

# 4.5 결과
lmtune ls --kind study | head
lmtune dashboard build --out b200/dashboards
echo "open: file://$(realpath b200/dashboards/index.html)"

# 4.6 정리
kill $PF_PID 2>/dev/null
# 다음 path 로 넘어가기 전 자원 회수:
helmfile -f b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl destroy
```

---

## 5. Path 2 — pd-disaggregation (Llama-70B, ~20분)

Prefill 4 GPU + Decode 4 GPU = 단일 노드 8 GPU 차지. 인터노드 NIXL 도 가능 (decode 가 다른 노드에 가도 OK, RDMA fabric 활성 시).

```bash
# 5.1 helmfile apply
NS=b200-pd helmfile -f b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl apply

# 5.2 rollout — prefill 먼저, decode 그 다음
kubectl -n b200-pd rollout status deployment --timeout=20m
kubectl -n b200-pd get pods       # ms-pd-decode-* + ms-pd-prefill-* 모두 Running

# 5.3 endpoint
SVC=$(kubectl -n b200-pd get svc -o jsonpath='{.items[?(@.spec.ports[0].port==80)].metadata.name}')
kubectl -n b200-pd port-forward svc/$SVC 8021:80 &
PF_PID=$!
sleep 3
curl -s http://localhost:8021/v1/models | jq

# 5.4 PD endpoint YAML 생성 (port 만 다름)
cat > /tmp/b200-pd-endpoint.yaml <<EOF
apiVersion: lmtune/v1alpha1
slug: b200-pd
name: B200 PD Llama-70B
url: http://localhost:8021/v1
model: meta-llama/Llama-3.1-70B-Instruct
api_type: openai
deployment:
  engine: vllm
  parallelism: {tp: 4, dp: 1, ep: false}
  engine_args:
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_seqs: 128
    max_model_len: 8192
    gpu_memory_utilization: 0.85
notes: PD via NIXL/UCX
tags: [b200, pd, llama70b]
EOF

# 5.5 autotune (medium + long workload, 8 trial, ~15분)
lmtune search start \
  --strategy tpe \
  --space b200/search-spaces/b0_smoke.yaml \
  --endpoint /tmp/b200-pd-endpoint.yaml \
  -p configs/profiles/autotune/medium.yaml \
  -p configs/profiles/autotune/long.yaml \
  --backend inline --max-trials 8 \
  --name b1-pd-llama70b

# 5.6 결과
lmtune dashboard build --out b200/dashboards

# 5.7 정리
kill $PF_PID 2>/dev/null
helmfile -f b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl destroy
```

---

## 6. Path 3 — wide-ep-lws (Mixtral-8x22B, ~25분)

8 GPU × DP=2 EP=8. Mixtral-8x22B 모델 풀 (~280 GB) 이 오래 걸리니 첫 풀은 한 번에 받아두는 게 좋습니다.

```bash
# 6.1 helmfile apply
NS=b200-wideep helmfile -f b200/helmfile/wide-ep-lws/helmfile.yaml.gotmpl apply

# 6.2 rollout (모델 풀 5-15분)
kubectl -n b200-wideep rollout status deployment --timeout=30m
kubectl -n b200-wideep get pods

# 6.3 endpoint
SVC=$(kubectl -n b200-wideep get svc -o jsonpath='{.items[?(@.spec.ports[0].port==80)].metadata.name}')
kubectl -n b200-wideep port-forward svc/$SVC 8031:80 &
PF_PID=$!
sleep 3
curl -s http://localhost:8031/v1/models | jq

# 6.4 endpoint YAML
cat > /tmp/b200-wideep-endpoint.yaml <<EOF
apiVersion: lmtune/v1alpha1
slug: b200-wideep
name: B200 wide-EP Mixtral-8x22B
url: http://localhost:8031/v1
model: mistralai/Mixtral-8x22B-Instruct-v0.1
api_type: openai
deployment:
  engine: vllm
  parallelism: {tp: 1, dp: 2, ep: true}
  engine_args:
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_seqs: 256
    max_model_len: 16384
    gpu_memory_utilization: 0.88
notes: wide-EP MoE; DP=2 EP=8
tags: [b200, wide-ep, moe, mixtral]
EOF

# 6.5 autotune (8 trial, ~15-20분)
lmtune search start \
  --strategy tpe \
  --space b200/search-spaces/b0_smoke.yaml \
  --endpoint /tmp/b200-wideep-endpoint.yaml \
  -p configs/profiles/autotune/short.yaml \
  -p configs/profiles/autotune/medium.yaml \
  --backend inline --max-trials 8 \
  --name b1-wideep-mixtral

# 6.6 결과
lmtune dashboard build --out b200/dashboards

# 6.7 정리
kill $PF_PID 2>/dev/null
helmfile -f b200/helmfile/wide-ep-lws/helmfile.yaml.gotmpl destroy
```

---

## 7. Cross-path 비교 (3 path 모두 끝낸 후)

```bash
# 3 study 의 winning config + Pareto + axis importance 비교
lmtune dashboard build --out b200/dashboards
xdg-open b200/dashboards/index.html        # 또는 firefox / chrome

# study list — `lmtune search ls` 로 study_id 확인 후 export
COLUMNS=200 lmtune search ls --limit 5

# winner export — 위 출력에서 study_id 복사
for sid in $(COLUMNS=200 lmtune search ls --limit 3 | grep -oE "st-[A-Z0-9]{26}"); do
  lmtune search export "$sid" --winner top-1 --out "b200/results/$sid/winner/"
done
ls b200/results/*/winner/
```

각 `winner/` 디렉토리에 `apply.sh`, `values-overlay.yaml`, `params.json`, `README.md` 가 떨어집니다 — 동일 클러스터 (또는 같은 helmfile 을 가진 다른 클러스터) 에서 `bash apply.sh` 한 줄로 재배포.

---

## 8. 본격 multi-trial autotune (선택, ~수 시간)

위 quickstart 는 trial 수가 적습니다. 본격 탐색은 NSGA-II Pareto + B1 search-space:

```bash
# 한 path 만 (예: pd-disaggregation 활성 상태에서)
lmtune search start \
  --strategy nsga2 \
  --space b200/search-spaces/b1_baselines.yaml \
  --endpoint /tmp/b200-pd-endpoint.yaml \
  -p configs/profiles/autotune/short.yaml \
  -p configs/profiles/autotune/medium.yaml \
  -p configs/profiles/autotune/long.yaml \
  --objectives "throughput_tok.avg:short:maximize,ttft.p99:short:minimize" \
  --backend inline --max-trials 40 --budget-hours 4 \
  --name b1-nsga2-pd
```

대시보드의 **"왜 이 winner?"** axis-diff 카드 + per-trial **"why?"** (TPE warmup · 🟢 new best) 라벨 + **📚 Search principles** 패널이 자동 채워집니다.

---

## 9. 트러블슈팅

| 증상 | 점검 |
|:---|:---|
| `helmfile apply` 가 chart download 실패 | `curl -sI https://llm-d-incubation.github.io/llm-d-modelservice/index.yaml` 200 인지 확인. 사내 프록시 환경이면 helm repo 미러 필요 |
| 모델 풀 무한 대기 | `kubectl -n <ns> describe pod` 의 events 확인. HF token secret 누락 가능성 |
| `port-forward` 연결 끊김 | 백그라운드 프로세스가 죽었는지 `jobs` 확인. `&` 대신 `tmux new-session -d "kubectl port-forward ..."` 권장 |
| autotune 첫 trial timeout | `--profile-timeout-s 600` 추가 (대형 모델은 첫 워크업이 길다) |
| `lmtune dashboard build` 가 빈 카드 | `lmtune ls --kind study` 로 study 가 DB 에 들어갔는지 확인. `LMTUNE_DB` 환경변수 옵션 |
| 노드가 1개 (8 GPU) 인데 PD 가 안 뜸 | values-llama-3.1-70b-tp4-pd.yaml 의 prefill+decode = 4+4 = 8 GPU 가 한 노드에서 잡혀야 함. `kubectl describe node` 로 GPU 가용성 확인 |

---

## 10. 다음 단계

3 path quickstart 가 끝나면:

- **wider 탐색**: B-track plan (`(internal dev plan, not in repo)` § Phase B) 의 B1~B6
- **path 조합**: 현재 미구현. 4 layer 분해 (topology / routing / cache / autoscale) 는 plan 의 후속 phase
- **다른 모델**: `b200/helmfile/<path>/values-<model>.yaml` 추가 + `--endpoint` 의 url/model 만 바꾸면 같은 helmfile 위에서 동작
- **외부 공개**: B-IV (B7/B8) — multi-engine + RECIPES.md + 블로그 초고
