# Custom Controllers — plug-in 작성 가이드

본 디렉토리는 lmtune 의 Layer 4 Controller 를 외부 서비스로 구현하는 reference. lmtune 은 `--controller http --controller-url http://...` 로 호출 → 서비스가 next params 를 결정 → lmtune 이 측정.

## 계약 (HTTP 2 endpoint)

`docs/architecture.md` § Layer 4 Pluggability 의 정본. 요약:

```http
POST /ask
  body: {study_id, active_axes:[{name,kind,values,low,high,step}], history:[...], context:{}}
  resp: {"params": {axis_name: value, ...}}

POST /tell
  body: {study_id, params, value, status, metadata}
  resp: 204 No Content
```

## reference 구현

| 파일 | 언어 | 용도 |
|:---|:---|:---|
| `mock_server.py` | Python (Flask) | minikube smoke — random params + 결정론적 sequence |
| `random_server.py` | Python (FastAPI) | RandomController 의 HTTP 버전 |
| `claude_server.py` (skeleton) | Python (Anthropic SDK) | Claude 가 next params 를 추론하는 LLM controller |

## 사용 예 (minikube swap-test)

```bash
# 1. Mock controller 띄우기 (별도 터미널)
python examples/controllers/mock_server.py --port 8090

# 2. lmtune 이 mock controller 로 sweep
lmtune search start --strategy http \
  --controller http --controller-url http://localhost:8090 \
  --space b200/search-spaces/b6_interconnect_tier1.yaml \
  --endpoint configs/endpoints/minikube_pd_qwen25.yaml \
  -p configs/profiles/autotune/short.yaml \
  --backend k8s-job --workers 1 --max-trials 4 \
  --name minikube-mock-ctrl

# 3. 결과 확인
duckdb data/db/lmtune.duckdb \
  "SELECT seq, status, score FROM trials WHERE study_id LIKE 'st-%' ORDER BY seq"
```

## 자기만의 controller 작성

3 단계:

1. `/ask` 와 `/tell` 두 endpoint 노출 (위 계약)
2. `active_axes` 의 type 별 (categorical/bool/int/float/log_uniform) 처리
3. `history` 누적해서 학습 (또는 LLM context 로 전달)

언어 무관 — Python (이 디렉토리), TypeScript (LangGraph workflow), Go (자체 RL agent), Rust (high-perf bandit) 모두 가능.
