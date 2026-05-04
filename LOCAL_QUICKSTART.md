# Local Quickstart — 5분 sanity check 두 트랙

> B200 으로 가기 전에 **로컬에서 동일 흐름을 미리 검증**합니다. 두 트랙 중 본인 환경에 맞는 것 선택:
>
> | 트랙 | 환경 | 시간 | 검증 대상 |
> |:---|:---|:---|:---|
> | **A. 순수 local-vLLM** | host + GPU 1장 (RTX/A 시리즈) | ~5분 | autotune 흐름 + dashboard. K8s 없음 |
> | **B. minikube + llm-d** | minikube + GPU 1장 | ~15분 | B200 와 동일 흐름 (helmfile + LLMDK8sAdapter) |
>
> 두 트랙 모두 `Qwen/Qwen2.5-1.5B-Instruct` (~3GB bf16) 사용 — gated 아님, 토큰 불필요.

---

## 트랙 A — 순수 local-vLLM (5분)

K8s 없이 host 에서 `vllm serve` 직접 실행. autotune 흐름·dashboard·explainability 다 검증됨.

### 1. 설치

```bash
git clone <this-repo> ~/ml_ai/lmtune
cd ~/ml_ai/lmtune

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,search,distributed,runners]"
pip install vllm                       # 호스트에 vLLM 설치 (CUDA 12+ 권장)
```

### 2. autotune 실행 (10 trial, ~30분)

```bash
lmtune search start \
  --strategy tpe \
  --space b200/search-spaces/w_local_minimal.yaml \
  --endpoint configs/endpoints/local_vllm_autotune.yaml \
  -p configs/profiles/autotune/short.yaml \
  --adapter local-vllm \
  --backend inline --max-trials 10 \
  --name local-vllm-mini
```

> `local_vllm_autotune.yaml` 의 model 이 Qwen2.5-1.5B-Instruct, `scripts/vllm_restart.sh` 가 매 trial 마다 vllm 서버 자동 재기동. **첫 trial 만 모델 로드 ~1분**, 이후 trial 은 ~30초.

### 3. 진행 / 결과

```bash
# 별도 터미널에서 실시간 진행
watch -n 5 'lmtune search ls --limit 5'

# 종료 후
lmtune dashboard build --out /tmp/dash-local
xdg-open /tmp/dash-local/index.html

# winner export — search start 출력의 study_id 를 그대로 복사·붙여넣기
# (예: study_id: st-01KQN632MS8GB3FK5V6M15FR3S)
SID=st-01KQN632MS8GB3FK5V6M15FR3S    # 본인 출력으로 교체
lmtune search export "$SID" --winner top-1 --out "/tmp/winner-local/"
cat /tmp/winner-local/README.md

# 또는 자동: COLUMNS=200 으로 rich Table truncation 회피 후 grep
SID=$(COLUMNS=200 lmtune search ls --limit 1 | grep -oE "st-[A-Z0-9]{26}" | head -1)
echo "found: $SID"
```

대시보드의 study detail 에서 확인:
- **"왜 이 winner?"** axis-diff 카드 — top-K trial 들이 어느 axis 에서 갈렸는지
- **per-trial `why?` 컬럼** — "TPE warmup · 🟢 new best (+12.3%)" 라벨
- **📚 Search principles** 패널 — TPE / NSGA-II / Sobol 원리

---

## 트랙 B — minikube + llm-d (15분)

`b200/helmfile/inference-scheduling/helmfile-mini.yaml.gotmpl` 가 작은 values 로 inference-scheduling path 를 띄움. B200 quickstart 와 **명령 구조가 같아** 학습 곡선 0.

### 1. minikube + GPU + 도구

> **중요**: minikube driver 가 `docker` (또는 `none`) 여야 GPU 노출됨. **`kvm2` driver 는 별도 VFIO passthrough 설정 없이는 GPU 미사용**. 기존 minikube 가 kvm2 로 떠있으면 `minikube delete` 후 docker 로 재시작.

```bash
# 0) 기존 minikube driver 확인
minikube profile list

# 만약 driver 가 kvm2 로 표시되거나 GPU 가 안 떠있으면:
# minikube delete
# minikube start --cpus=8 --memory=16384 --gpus=all --driver=docker

# 1) GPU 노출 검증 — 출력이 "1" (또는 그 이상) 이어야 함. 빈 줄이면 GPU 미인식
GPUS=$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}')
echo "minikube allocatable GPUs: '${GPUS:-NONE}'"
[ -n "$GPUS" ] || echo "⚠️  GPU 미노출 — minikube delete 후 --driver=docker --gpus=all 로 재시작 필요"

# 2) 도구
which kubectl helm helmfile && helm version && helmfile version
```

**GPU 미노출 원인 체크리스트**:
- 호스트에 `nvidia-ctk runtime configure --runtime=docker` 적용됐는지 (`docker info | grep nvidia` 가 runtime 보여야)
- minikube 가 docker driver 인지 (`minikube profile list` 의 VM Driver 컬럼)
- nvidia-device-plugin 켜졌는지 (`minikube addons list | grep nvidia`)

### 2. 코드 + 패키지 설치 (트랙 A 와 동일)

```bash
cd ~/ml_ai/lmtune
source .venv/bin/activate
pip install -e ".[dev,search,distributed]"     # K8s 트랙은 [runners] 불필요
```

### 3. CRDs + Gateway controller 사전 설치 (한 번만)

helm chart 들이 다음 CRD/Controller 가 사전에 클러스터에 깔려있다고 가정합니다 (B200 환경엔 이미 있겠지만 fresh minikube 엔 없음). 자체 검증 결과 발견된 사전 조건:

```bash
# (a) Gateway API CRD — llm-d-infra 가 Gateway/HTTPRoute 사용
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml

# (b) Gateway API Inference Extension CRD — GAIE InferencePool 정의
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.1/manifests.yaml

# (c) Gateway controller — 다음 중 택1
#   1) Istio (provider: istio, default) — istioctl install --set profile=minimal
#   2) agentgateway (provider: agentgateway, 본 mini track 의 default) — github.com/agentgateway/agentgateway 의 install manifest 적용
# B200 클러스터엔 보통 이미 깔려있음. minikube 신규 환경은 별도 설치 필요.
```

> **주의**: 사용자 자체 검증에서 발견 — `helmfile-mini.yaml.gotmpl` 의 `provider: agentgateway` 면 `agentgateway.dev/v1alpha1.AgentgatewayParameters` CRD 가 필요합니다. CRD/Controller 없이 helmfile apply 시 `resource mapping not found` 로 막힙니다.

### 4. helmfile apply (Qwen2.5-1.5B mini)

```bash
# (Qwen2.5-1.5B 는 gated 아니지만 helmfile 의 secretRef 가 optional 로 잡혀있어
#  비어있는 secret 만 만들어두면 됨)
kubectl create namespace mini-infsch 2>/dev/null || true
kubectl create secret generic huggingface-token \
  --from-literal=HF_TOKEN="" \
  -n mini-infsch --dry-run=client -o yaml | kubectl apply -f -

# helmfile apply
NS=mini-infsch helmfile \
  -f b200/helmfile/inference-scheduling/helmfile-mini.yaml.gotmpl apply

# rollout 대기 (모델 풀 ~1-2분)
kubectl -n mini-infsch rollout status deployment --timeout=10m
kubectl -n mini-infsch get pods
```

> 첫 풀에서 chart download (`llm-d-infra` / `llm-d-modelservice` / `inferencepool`) 가 일어남. 사내 프록시 환경이면 helm repo 미러 필요.

### 5. port-forward + 동작 검증

```bash
# 첫 service 의 80 포트를 로컬 8011 로
SVC=$(kubectl -n mini-infsch get svc -o jsonpath='{.items[?(@.spec.ports[0].port==80)].metadata.name}')
kubectl -n mini-infsch port-forward "svc/$SVC" 8011:80 &
PF_PID=$!
sleep 3

# OpenAI-compatible /v1/models
curl -s http://localhost:8011/v1/models | jq

# 1 token 추론
curl -s http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' | jq
```

### 6. autotune 실행 (helmfile-baked engine_args, ~10분)

minikube 의 K8s 트랙은 engine_args 가 **helmfile 에 baked** 되어 있어 매 trial 마다 helmfile redeploy 가 필요합니다 (~3분/cycle). Quickstart 는 engine_args 비활성, ask/tell 흐름만 검증:

```bash
# minikube endpoint YAML (port 만 다름)
cat > /tmp/mini-endpoint.yaml <<EOF
apiVersion: lmtune/v1alpha1
slug: mini-infsch
name: minikube inference-scheduling Qwen2.5-1.5B
url: http://localhost:8011/v1
model: Qwen/Qwen2.5-1.5B-Instruct
api_type: openai
deployment:
  engine: vllm
  parallelism: {tp: 1, dp: 1, ep: false}
  engine_args:
    enable_prefix_caching: false
    max_num_seqs: 32
    max_model_len: 4096
    enforce_eager: true
notes: minikube llm-d inference-scheduling, single GPU
tags: [minikube, llmd, qwen25-1.5b]
EOF

# autotune (8 trial, ~10분 — engine_args 를 측정만, 적용 X)
lmtune search start \
  --strategy tpe \
  --space b200/search-spaces/w_minikube_minimal.yaml \
  --endpoint /tmp/mini-endpoint.yaml \
  -p configs/profiles/autotune/short.yaml \
  --adapter llmd-k8s --backend inline --max-trials 8 \
  --name mini-infsch-tpe
```

> 진짜 engine_args 를 매 trial 적용하려면: `--adapter llmd-k8s` + helmfile redeploy hook (Phase B-I 의 `LLMDK8sAdapter.apply()` 가 자동 처리하는 흐름). 본 quickstart 는 sanity check 가 목적이라 측정만.

### 7. 결과 + 정리

```bash
lmtune dashboard build --out /tmp/dash-mini
xdg-open /tmp/dash-mini/index.html

# 정리
kill $PF_PID 2>/dev/null
helmfile -f b200/helmfile/inference-scheduling/helmfile-mini.yaml.gotmpl destroy
# minikube 도 종료하고 싶으면:
# minikube stop  # 또는 minikube delete
```

---

## 트랙 비교 — 무엇을 검증하나

| 검증 항목 | 트랙 A (local-vLLM) | 트랙 B (minikube + llm-d) |
|:---|:---:|:---:|
| `lmtune search start` ask/tell + DB 적재 | ✅ | ✅ |
| `LocalVLLMAdapter` 의 `vllm_restart.sh` 흐름 | ✅ | — |
| `LLMDK8sAdapter.apply()` 의 helmfile + rollout + probe | — | ✅ |
| OCI chart 다운로드 + GAIE EPP + modelservice | — | ✅ |
| Dashboard explainability (axis-diff / TPE warmup / principles) | ✅ | ✅ |
| Winner export (`apply.sh` 재배포) | ✅ | ✅ |
| 매 trial 마다 engine_args 실제 적용 | ✅ | ⚠️ helmfile redeploy 필요 |

→ **B200 가기 전 권장 순서**: 트랙 A → 트랙 B → B200 QUICKSTART. 각 단계 사이 흐름 격차 0 (명령 구조 동일).

---

## 트러블슈팅

| 증상 | 점검 |
|:---|:---|
| `minikube start --gpus=all` 실패 | NVIDIA Container Toolkit 설치 + `sudo nvidia-ctk runtime configure --runtime=docker` |
| `nvidia.com/gpu: 0` | `minikube addons enable nvidia-device-plugin`. 안 되면 `kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml` |
| `no matches for kind "Gateway"` / `"InferencePool"` | 위 §3 의 (a) Gateway API + (b) Inference Extension CRD 가 누락. 두 manifest 모두 `kubectl apply -f ...` |
| `no matches for kind "AgentgatewayParameters"` | agentgateway controller 미설치. (c) Gateway controller 단계 필요. Istio 로 가려면 helmfile values 의 `provider: agentgateway` 를 `istio` 로 바꾸고 istioctl install |
| `no matches for kind "Telemetry"` (telemetry.istio.io) | helmfile values 의 `provider:` 가 `istio` 인데 Istio 미설치. agentgateway 트랙으로 가거나 istioctl install |
| Helm chart 다운로드 timeout | `curl -sI https://llm-d-incubation.github.io/llm-d-modelservice/index.yaml` 200 인지. 사내 프록시면 `helm repo add` 우회 필요 |
| `kubectl port-forward` 끊김 | `tmux new -d "kubectl -n mini-infsch port-forward svc/$SVC 8011:80"` 으로 영속화 |
| 첫 trial 0 score | `kubectl -n mini-infsch logs deploy/<vllm-decode>` 에 모델 로딩 에러 확인. 보통 GPU memory / disk I/O |
| dashboard 빈 카드 | `lmtune search ls --limit 5` 로 study 가 들어갔는지. `LMTUNE_DB` env 가 다른 경로 가리키지 않는지 |

---

## 다음 단계

로컬 검증이 끝나면:

- **B200 16-GPU**: [`b200/QUICKSTART.md`](b200/QUICKSTART.md) — 3 well-lit-paths 동일 흐름
- **본격 NSGA-II**: B200 QUICKSTART §8 — 40 trial multi-obj Pareto + budget-hours
- **계획 전체**: `(internal dev plan, not in repo)` Phase B 섹션
