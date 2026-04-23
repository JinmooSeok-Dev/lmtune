# 코딩 에이전트 워크로드 벤치마크 계획서 v3

> **대상 모델**: Qwen/Qwen3-30B-A3B (MoE, 30B total / 3B active)
> **Inference 엔진**: vLLM + Rebellions NPU (RBLN) — TP4, Expert Parallel
> **벤치마크 도구**: NVIDIA AIPerf v0.5+ (GenAI-Perf 후속)
> **작성일**: 2026-04-01

---

## 1. 현재 Endpoint 구성 및 제약

### 1.1 서버 실행 구성

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-log-requests \
  --enable-log-outputs \
  --disable-uvicorn-access-log \
  --max-num-batched-tokens 128 \
  --enable-chunked-prefill \
  --max-model-len 40960 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --block-size 4096 \
  --num-gpu-blocks-override 30 \
  --trust-remote-code
```

### 1.2 제약 분석

| 항목 | 값 | 실 서비스 비교 | 벤치마크 영향 |
|:-----|:--:|:-------------|:-------------|
| max-model-len | **40,960 tok** | Claude Code 200K, Codex 400K | 단일 요청 input+output 상한. 실 agent의 **1/5~1/25** |
| max-num-seqs | **4** | Cursor 1M+ QPS 환경 | 동시 처리 상한. concurrency ≤ 4 |
| max-num-batched-tokens | **128** | GPU: 수천~수만 | NPU chunked prefill 단위. TTFT ∝ ⌈input/128⌉ |
| KV cache 총량 | **122,880 tok** | GPU: 수백만 tok | 30 blocks × 4,096 tok/block |
| block-size | **4,096 tok** | GPU 기본: 16 tok | 50 tok 요청도 4,096 블록 1개 점유 → KV 낭비 |
| prefix-caching | **ON** | — | shared system prompt 재활용 가능 |

### 1.3 KV Cache 용량 시나리오

```
시나리오               동시 seqs   seq당 context   총 KV 사용    122,880 대비
──────────────────────────────────────────────────────────────────────────
Tab Completion (A)     4           ~2,050 tok      8,200 tok     7%  ✅ 여유
IDE Chat Turn5 (B)     2           ~12,500 tok     25,000 tok    20% ✅ 여유
Agent Turn12 (C)       1           ~19,500 tok     19,500 tok    16% ✅ 여유
Agent Turn15 (C-agg)   1           ~30,400 tok     30,400 tok    25% ✅ 가능
동시 4 × 30K           4           30,000 tok      120,000 tok   98% ⚠️ 한계
```

---

## 2. 워크로드 설계 개요

### 2.1 3가지 대표 워크로드

| Profile | 실 서비스 패턴 | 본 벤치마크 (완화) | 핵심 측정 |
|:--------|:-------------|:-----------------|:---------|
| **A** Tab Completion | Cursor 13K ctx, 1M+ QPS, output 10-50 tok | input 2K, output 50, 1턴, concurrency 4 | TTFT, Throughput |
| **B** IDE Chat | Copilot 64K, 3-6턴, 누적 ~20K | 5턴, 턴당 2K/500, 누적 ~12.5K | Turn별 TTFT 변화, ITL |
| **C** Agent Loop | Claude Code 31K→200K, SWE-bench 150 steps | 12턴, 턴당 1.2K/300, 누적 ~19.5K | Token Snowball, TTFT at large ctx |
| **C-agg** Agent Aggressive | 위 + 40K 한계 탐색 | 15턴, 턴당 1.5K/400, 누적 ~30.4K | max-model-len 벽, 에러 발생 턴 |

### 2.2 실 서비스 → 벤치마크 완화 원칙

| 실 서비스 값 | 완화 후 값 | 이유 |
|:-----------|:---------|:-----|
| Concurrency 수천+ | **≤ 4** | max-num-seqs = 4 |
| Context 200K~1M | **≤ 40K** | max-model-len = 40,960 |
| Agent 턴 수 150 | **12~15** | 40K 내 context 수용 |
| 초기 overhead 31K | **0~3K** | 31K 적용 시 9턴에 40K 초과 |
| 동시 agent 8개 | **1~2** | KV cache 122K 한도 내 |

### 2.3 AIPerf 실행 모드 구분

> ⚠️ AIPerf에는 두 가지 부하 생성 모드가 있으며, **혼용하지 않는 것이 원칙**:

| 모드 | 용도 | 파라미터 | 동시성 결정 |
|:-----|:-----|:---------|:----------|
| **Concurrency 모드** | 1턴 stateless 요청 (Profile A) | `--concurrency` + `--request-count` | concurrency 값이 직접 결정 |
| **User-centric 모드** | Multi-turn 대화 (Profile B, C) | `--num-users` + `--user-centric-rate` + `--conversation-*` | num-users × rate로 간접 결정 |

> ⚠️ **파라미터 이름 참고**: 본 문서는 `--conversation-num`, `--conversation-turn-mean` 등을 사용.
> AIPerf 버전에 따라 `--num-conversations`, `--session-turns-mean` 등으로 이름이 다를 수 있음.
> 실행 전 `aiperf profile --help`로 확인할 것.

---

## 3. Profile A — Tab Completion (Stateless, 고QPS)

### 3.1 워크로드 상세

**시나리오**: 에디터에서 코드 타이핑 중 자동완성 제안. FIM(Fill-in-the-Middle) 패턴으로, 커서 위치의 prefix/suffix "small snippet"을 서버에 전송하고, 수십 토큰의 짧은 완성을 반환.

| 항목 | 실 서비스 | 본 벤치마크 | 완화 이유 |
|:-----|:---------|:----------|:---------|
| Input | 1–4K tok (context 13K의 일부) | **2,000 tok** | 40K 내 충분. 실 서비스 중앙값 |
| Output | 10–50 tok | **50 tok** | 동일 |
| 턴 수 | 1 (stateless) | **1** | 동일 |
| 동시성 | 수천~수만 (1M+ QPS) | **4** | max-num-seqs 제약 |
| I:O ratio | ~30:1 | 40:1 | 유사 |

**왜 이 워크로드인가**: DynamoLLM(HPCA 2025)의 Microsoft fleet 분석에서 코딩 워크로드는 "input long / output short" 특성이 확인됨. OpenRouter 100T 토큰 분석에서 programming이 prompt 토큰 성장의 최대 동인. Cursor autocomplete는 1M+ QPS로 **가장 빈번한 코딩 inference 워크로드**.

### 3.2 측정 지표 및 목표

| 지표 | 왜 중요한가 | 목표 기준 |
|:-----|:----------|:---------|
| **TTFT p50** | 자동완성은 타이핑 흐름을 끊지 않아야 함 | < 500ms (Cursor Fusion p50=260ms 참고) |
| **TTFT p99** | tail latency 높으면 간헐적 "멈춤" 체감 | < 2s |
| **Throughput (req/s)** | max-num-seqs=4 하 처리량 상한 | 측정 (baseline) |
| **Output tok/s** | NPU decode 성능 | 측정 (baseline) |
| **TTFT vs input_len 기울기** | chunked prefill(128 tok/iter) NPU 고유 특성 | ms/chunk 값 도출 |

### 3.3 AIPerf 실행 명령

```bash
# ──────────────────────────────────────────────────────────────
# A-1: Baseline 측정
# → Concurrency 모드 (stateless 1턴 → conversation 파라미터 불필요)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --synthetic-input-tokens-mean 2000 \
  --output-tokens-mean 50 \
  --concurrency 4 \
  --request-count 200 \
  --random-seed 42

# concurrency 4: max-num-seqs=4에 맞춤
# request-count 200: 통계적으로 유의미한 샘플 수
# conversation 파라미터 없음: 1턴 stateless는 대화가 아님

# ──────────────────────────────────────────────────────────────
# A-2: Chunked Prefill 선형성 테스트 (Input 크기별 TTFT sweep)
# → Concurrency 모드, concurrency=1로 격리 측정
# ──────────────────────────────────────────────────────────────
for ISL in 500 1000 2000 4000 8000 16000; do
  echo "=== Testing ISL=$ISL ==="
  aiperf profile \
    --model Qwen/Qwen3-30B-A3B \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --synthetic-input-tokens-mean $ISL \
    --output-tokens-mean 50 \
    --concurrency 1 \
    --request-count 30 \
    --random-seed 42
done

# concurrency 1: 다른 요청 간섭 없이 순수 prefill 시간 측정
# 기대 결과: TTFT vs input_len이 선형
# 기울기 = TTFT 증가분 / (input_len 증가분 / 128) = ms per chunk
```

### 3.4 근거 자료

| # | 자료 | 핵심 내용 | URL |
|:-:|:-----|:---------|:----|
| 1 | Cursor Tab Fusion Blog | context 5.5K→13K, p50 260ms, 10x longer changes | https://cursor.com/blog/tab-update |
| 2 | ByteByteGo — Cursor 아키텍처 | "small snippet" 전송, **1M+ QPS** autocomplete | https://blog.bytebytego.com/p/how-cursor-serves-billions-of-ai |
| 3 | DynamoLLM (HPCA 2025) | MS fleet: Coding = **input long / output short**, peak:valley **34.6×** | https://jovans2.github.io/files/DynamoLLM_HPCA2025.pdf |
| 4 | OpenRouter 100T 분석 | avg prompt 1.5K→6K (4×↑), Programming = 최대 driver | https://arxiv.org/abs/2601.10088 |
| 5 | Fireworks State of Agents | Claude **36% coding**; OpenRouter top 4 = **80% coding agents** | https://fireworks.ai/blog/state-of-agent-environments |

---

## 4. Profile B — IDE Chat / Code Gen (Short Multi-turn)

### 4.1 워크로드 상세

**시나리오**: IDE 채팅창에서 코드 질문 → 답변 확인 → 후속 질문. 대화 히스토리가 context에 점진적으로 누적. 개발자의 "읽고 생각하고 입력"하는 딜레이가 턴 간격에 반영됨.

| 항목 | 실 서비스 | 본 벤치마크 | 완화 이유 |
|:-----|:---------|:----------|:---------|
| Input (턴당 신규) | 2–4K tok | **2,000 tok** | 실 서비스 하한 |
| Output (턴당) | 100–1,500 tok | **500 tok** | 중앙값 |
| 턴 수 | 3–6 | **5** (stddev 1) | 범위 내 |
| 동시 세션 | 수십 | **2** | max-num-seqs 4에서 안전 |
| 누적 context (최종턴) | ~13–20K | **~12,500 tok** (40K의 31%) | 안전 범위 |

**왜 이 워크로드인가**: Copilot Chat 64K window이지만 실 Q&A는 턴당 2~4K. AgentTaxo(ICLR 2025)에서 I:O = 2:1~3:1 일관. llm-d 벤치마크에서 prefix-aware routing이 P95 TTFT 63% 개선 — prefix caching 효과 검증 기준 워크로드.

**턴별 context 누적 추정**:

```
Turn0: 2,000(user) + 500(asst) = 2,500
Turn1: 2,500 + 2,000 + 500     = 5,000
Turn2: 5,000 + 2,500            = 7,500
Turn3: 7,500 + 2,500            = 10,000
Turn4: 10,000 + 2,500           = 12,500  ← 40K의 31%

KV cache: 2 sessions × 12,500 = 25,000 tok (122K의 20%)
Prefill chunks at Turn4: 12,500 / 128 ≈ 98 chunks
```

### 4.2 측정 지표 및 목표

| 지표 | 왜 중요한가 | 목표 기준 |
|:-----|:----------|:---------|
| **TTFT Turn0 vs Turn4** | context 누적(2.5K→12.5K) TTFT 열화율 | Turn4/Turn0 ≈ 5–7× (선형이면 정상) |
| **ITL p50/p99** | 턴 진행 중 decode 속도 안정성 | 턴 간 ITL 편차 < 20% |
| **E2E session latency** | 5턴 전체 세션 완료 시간 | 측정 (baseline) |
| **Prefix caching 효과** | B-1 vs B-2 TTFT 절감 정량화 | 절감률 % |

### 4.3 AIPerf 실행 명령

```bash
# ──────────────────────────────────────────────────────────────
# B-1: 기본 5턴 IDE Chat
# → User-centric 모드 (multi-turn → --concurrency 사용하지 않음)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 10 \
  --conversation-turn-mean 5 \
  --conversation-turn-stddev 1 \
  --synthetic-input-tokens-mean 2000 \
  --output-tokens-mean 500 \
  --num-users 2 \
  --user-centric-rate 0.5 \
  --random-seed 42

# num-users 2 × rate 0.5 = 최대 동시 1 req/s → max-num-seqs=4 내 안전
# user-centric-rate 0.5: 유저당 2초에 1 request (읽고 입력하는 딜레이)
# 10 conversations × ~5 turns = ~50 requests

# ──────────────────────────────────────────────────────────────
# B-2: Prefix Caching 효과 측정 (shared system prompt 3K)
# → User-centric 모드 + sticky session
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 10 \
  --conversation-turn-mean 5 \
  --conversation-turn-stddev 0 \
  --synthetic-input-tokens-mean 2000 \
  --output-tokens-mean 500 \
  --shared-system-prompt-length 3000 \
  --num-users 2 \
  --user-centric-rate 0.5 \
  --connection-reuse-strategy sticky-user-sessions \
  --random-seed 42

# shared-system-prompt-length 3000:
#   Claude Code system prompt 4.3K [참조 #16]의 축소 버전
# sticky-user-sessions:
#   같은 유저 턴 → 같은 replica → KV cache locality 보장
#
# 비교: B-1 vs B-2
#   → Turn0 TTFT 차이 = prefix 3K 캐시 효과
#   → 3000/128 ≈ 24 chunks 만큼 prefill 절감 기대
```

### 4.4 근거 자료

| # | 자료 | 핵심 내용 | URL |
|:-:|:-----|:---------|:----|
| 6 | Copilot Chat 64K context | GPT-4o 기반, VS Code Insiders 128K | https://github.blog/changelog/2024-12-06-copilot-chat-now-has-a-64k-context-window-with-openai-gpt-4o/ |
| 7 | Cursor IDE context | Normal **128K**, Max **200K**, chat ~**20K** default | https://www.qodo.ai/blog/claude-code-vs-cursor/ |
| 8 | AgentTaxo (ICLR 2025) | Multi-agent I:O = **2:1~3:1** 일관 | https://openreview.net/pdf?id=0iLbiYYIpC |
| 9 | llm-d multi-turn routing | prefix-aware → KV hit ~90%, **P95 TTFT 63%↓** | https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d |
| 10 | Pliops vLLM multi-turn | ~90% cache hit 권장, 100%는 비현실적 | https://pliops.com/setting-the-standard-multi-turn-benchmarking-in-vllm/ |
| 11 | Cost of Dynamic Reasoning | prefix caching → KV cache 메모리 **51.7%↓** | https://arxiv.org/abs/2506.04301 |

---

## 5. Profile C — Agent Loop (Unbounded Multi-turn)

### 5.1 워크로드 상세

**시나리오**: SWE-bench 스타일 자율 코딩 에이전트. issue 입력 → 탐색 → 편집 → 테스트 자율 반복. 매 턴 tool call JSON(짧은 output) 생성, tool result(긴 파일 내용)가 context에 누적되는 "Token Snowball Effect".

| 항목 | 실 서비스 | 본 벤치마크 | 완화 이유 |
|:-----|:---------|:----------|:---------|
| Input (턴당 신규) | tool result 수천 tok | **1,200 tok** | 40K 내 12턴 수용 |
| Output (턴당) | tool call JSON 100–500 tok | **300 tok** | 중앙값 |
| 턴 수 | 10–150 (SWE-bench max 150) | **12** (stddev 2) | 40K 내 context 수용 |
| 동시 세션 | 1–8 (Cursor 8 parallel) | **1** | deep session 집중 |
| 초기 overhead | 31K (Claude Code) | **0** | 31K 시 9턴에 40K 초과 |
| 누적 context (최종턴) | 200K+ | **~19,500 tok** (40K의 48%) | 스케일 축소, 패턴 보존 |
| Task당 총 토큰 | 1–3.5M | ~25K | 대폭 축소 |

**왜 이 워크로드인가**: SWE-Effi에서 "Token Snowball Effect" 정의 — agent 매 LLM call마다 prompt 선형 증가, 실패 시 4× 토큰 낭비. OpenHands 분석에서 run간 최대 10× 차이. Iternal Guide(2026.03) 기준 task당 평균 1~3.5M tokens. **가장 비용이 큰 inference 패턴**.

**완화 상세**:

| 실 서비스 → 본 벤치마크 | 이유 |
|:----------------------|:-----|
| 턴 수 150 → **12** | max-model-len 40K 내 context 수용 |
| 초기 overhead 31K → **0** | 31K 적용 시 9턴에 40K 초과 |
| 동시 세션 8 → **1** | KV cache 122K, 깊은 context 추적 |
| 총 토큰 1~3.5M → **~25K** | Token Snowball 패턴 자체는 동일 관찰 |

**턴별 context 누적 추정**:

```
Turn0:  1,200 + 300 = 1,500
Turn3:  1,500 + 3 × 1,500 = 6,000
Turn5:  1,500 + 5 × 1,500 = 9,000
Turn8:  1,500 + 8 × 1,500 = 13,500
Turn10: 1,500 + 10 × 1,500 = 16,500
Turn12: 1,500 + 12 × 1,500 = 19,500  ← 40K의 48%

Prefill chunks at Turn12: 19,500 / 128 ≈ 153 chunks
KV cache: 1 × 19,500 = 19,500 tok (122K의 16%)
```

### 5.2 측정 지표 및 목표

| 지표 | 왜 중요한가 | 목표 기준 |
|:-----|:----------|:---------|
| **TTFT vs Turn# 곡선** | Token Snowball 관찰. chunked prefill에서 선형이 정상 | 선형 = 정상. 급등 = KV eviction |
| **TTFT (Turn12)** | 19.5K prefill: ≈153 chunks | Turn0 대비 ~13× 예상 |
| **ITL 안정성** | 긴 세션 decode 속도 유지 | 전 턴 ITL 편차 < 10% |
| **에러율** | 12턴은 안전 범위 | 0% |

**핵심 그래프 — TTFT vs Turn#**:

```
TTFT(ms)
  │
  │                                          × Turn12 (19.5K)
  │                                     ×
  │                                ×
  │                           ×
  │                      ×
  │                 ×
  │            ×
  │       ×
  │  ×  Turn0 (1.5K)
  └──────────────────────────────────── Turn #
     0   1   2   3   4   5   6   7   8  10  12

  선형 = 정상 (chunked prefill 특성)
  급등 = KV cache eviction 또는 메모리 경쟁
```

### 5.3 AIPerf 실행 명령

```bash
# ──────────────────────────────────────────────────────────────
# C-1: Agent Loop 12턴
# → User-centric 모드 (multi-turn → --concurrency 사용하지 않음)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 5 \
  --conversation-turn-mean 12 \
  --conversation-turn-stddev 2 \
  --synthetic-input-tokens-mean 1200 \
  --output-tokens-mean 300 \
  --num-users 1 \
  --user-centric-rate 2.0 \
  --random-seed 42

# num-users 1: 깊은 단일 세션 (실 agent = sequential tool calls)
# user-centric-rate 2.0: agent는 사람보다 빠르게 요청 (0.5초 간격)
#   → 실 agent: tool result 수신 → 즉시 다음 tool call (사람 개입 없음)
# 5 conversations × ~12 turns = ~60 requests
```

### 5.4 근거 자료

| # | 자료 | 핵심 내용 | URL |
|:-:|:-----|:---------|:----|
| 12 | Claude Code session overhead | system 4.3K + tools 20.3K = **31K/200K (16%)** | https://github.com/anthropics/claude-code/issues/24243 |
| 13 | Anthropic Advanced Tool Use | tool defs **50K+**, 85% 절감, 134K→17K | https://www.anthropic.com/engineering/advanced-tool-use |
| 14 | Claude Context API | **200K** standard, **1M** beta | https://platform.claude.com/docs/en/build-with-claude/context-windows |
| 15 | SWE-bench Harness | max **150 steps**, 50 chat reqs, **1M token** limit | https://www.vals.ai/benchmarks/swebench |
| 16 | SWE-Effi | **Token Snowball Effect** — prompt 선형 증가. 실패 시 **4× 토큰** | https://arxiv.org/abs/2509.09853 |
| 17 | OpenHands 토큰 분석 | run간 **최대 10×**, input tokens 지배적 | https://openreview.net/forum?id=1bUeVB3fov |
| 18 | Agent Framework 비교 | OpenHands **1.26B input / 30.54M output** | https://arxiv.org/abs/2511.00872 |
| 19 | Iternal Guide (2026.03) | task당 **1–3.5M tokens**; MCP metadata 40-50% | https://iternal.ai/token-usage-guide |
| 20 | Tokenomics (MSR 2026) | SDLC 단계별 토큰 분배 | https://arxiv.org/abs/2601.14470 |

---

## 6. Profile C-agg — Agent Aggressive (40K 벽 탐색)

### 6.1 워크로드 상세

**시나리오**: Profile C 확장. max-model-len 40,960의 **실질적 한계점** 탐색.

| 항목 | 값 | 비고 |
|:-----|:--|:----|
| Input (턴당) | **1,500 tok** | C보다 25%↑ |
| Output (턴당) | **400 tok** | C보다 33%↑ |
| 턴 수 | **15** (고정, stddev=0) | 한계점 탐색용 |
| 동시 세션 | 1 | — |
| 누적 context (Turn15) | **~30,400 tok** (40K의 74%) | — |

**왜 이 워크로드인가**: 현 40K 한계는 실 agent(200K~1M)와 5~25배 괴리. agent 세션이 몇 턴까지 유지 가능한지 upper bound 확인. Codex CLI는 context 초과 시 compaction(대화 압축)으로 대응 — vLLM에서의 동작 관찰.

**턴별 context 누적**:

```
Turn0:  1,900           Turn10: 20,900
Turn5:  11,400          Turn13: 26,600
Turn15: 30,400  ← 40K의 74%

output 편차로 600 tok 시: Turn15 = 33,600 (82%)
Turn ~19에서 40K 초과 가능
```

### 6.2 측정 지표 및 목표

| 지표 | 왜 중요한가 |
|:-----|:----------|
| **TTFT at 30K context** | 30,400/128 ≈ 238 chunks. NPU 대형 prefill 성능 |
| **에러 발생 턴** | 40K 근접 시 vLLM 동작 (rejection? truncation?) |
| **TTFT 급등 지점** | 선형 이탈 턴 = 실질적 성능 한계 |

### 6.3 AIPerf 실행 명령

```bash
# ──────────────────────────────────────────────────────────────
# C-agg: Agent Aggressive 15턴
# → User-centric 모드 (--concurrency 사용하지 않음)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 3 \
  --conversation-turn-mean 15 \
  --conversation-turn-stddev 0 \
  --synthetic-input-tokens-mean 1500 \
  --output-tokens-mean 400 \
  --num-users 1 \
  --user-centric-rate 3.0 \
  --random-seed 42

# conversation-turn-stddev 0: 정확히 15턴 고정 (한계점 탐색)
# user-centric-rate 3.0: agent 최고 속도 ~0.33초 간격
# conversation-num 3: 결과 편차 확인용
```

### 6.4 근거 자료

| # | 자료 | 핵심 내용 | URL |
|:-:|:-----|:---------|:----|
| 21 | Claude Code vs Codex | Claude 200K~1M vs Codex 400K + compaction | https://labs.adaline.ai/p/claude-code-vs-openai-codex |
| 22 | Codex CLI 아키텍처 | compaction — context 초과 시 대화 압축 | https://newsletter.pragmaticengineer.com/p/how-codex-is-built |
| 23 | AGENTS.md 효과 | 존재 시 **runtime 28.64%↓, output 16.58%↓** | https://arxiv.org/abs/2601.20404 |
| 24 | Devin 성능 | junior task ~7.8min; migration 3-4hr | https://cognition.ai/blog/devin-annual-performance-review-2025 |

---

## 7. Goodput (SLO 기반) 측정

### 7.1 워크로드 상세

**시나리오**: Profile B, C에 SLO 적용. SLO 이내 응답만 유효 처리량(goodput)으로 카운트.

**왜 이 워크로드인가**: AIPerf 가이드에서 "같은 throughput이어도 SLO 달성률이 2배 차이 가능" 지적. 코딩 에이전트 TTFT SLO: IDE Chat(사람 대기) ≤ 3초, Agent(자동화) ≤ 5초가 합리적.

### 7.2 측정 지표 및 목표

| 지표 | SLO 기준 | 의미 |
|:-----|:---------|:-----|
| **Goodput (IDE Chat)** | TTFT ≤ 3s, E2E ≤ 30s | SLO 충족 req/s |
| **Goodput (Agent)** | TTFT ≤ 5s, E2E ≤ 60s | SLO 충족 req/s (더 관대) |
| **SLO 달성률** | — | 전체 요청 중 SLO 이내 비율 (%) |

### 7.3 AIPerf 실행 명령

```bash
# ──────────────────────────────────────────────────────────────
# G-1: IDE Chat + SLO
# → User-centric 모드 (B-1과 동일 워크로드 + goodput 추가)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 10 \
  --conversation-turn-mean 5 \
  --conversation-turn-stddev 1 \
  --synthetic-input-tokens-mean 2000 \
  --output-tokens-mean 500 \
  --num-users 2 \
  --user-centric-rate 0.5 \
  --random-seed 42 \
  --goodput "time_to_first_token:3000 request_latency:30000"

# ──────────────────────────────────────────────────────────────
# G-2: Agent Loop + SLO
# → User-centric 모드 (C-1과 동일 워크로드 + goodput 추가)
# ──────────────────────────────────────────────────────────────
aiperf profile \
  --model Qwen/Qwen3-30B-A3B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --conversation-num 5 \
  --conversation-turn-mean 12 \
  --conversation-turn-stddev 2 \
  --synthetic-input-tokens-mean 1200 \
  --output-tokens-mean 300 \
  --num-users 1 \
  --user-centric-rate 2.0 \
  --random-seed 42 \
  --goodput "time_to_first_token:5000 request_latency:60000"
```

### 7.4 근거 자료

| # | 자료 | 핵심 내용 | URL |
|:-:|:-----|:---------|:----|
| 25 | AIPerf Goodput 가이드 | "SLO 충족 req/s = 사용자 체감 품질" | https://github.com/ai-dynamo/aiperf/blob/main/docs/comprehensive-llm-benchmarking.md |
| 26 | SWE-Pruner | pruning으로 **23–38% 절감**, rounds **18–26%↓** | https://arxiv.org/abs/2601.16746 |
| 27 | SWE-Skills-Bench | skill injection으로 -56%~+451% 토큰 변동 | https://arxiv.org/abs/2603.15401 |

---

## 8. 실행 순서

```
Phase 1: Baseline (Day 1)
  ├── A-1: Tab Completion baseline (concurrency 모드)
  │     → TTFT p50/p99, throughput (req/s), output tok/s
  ├── A-2: Input 크기별 TTFT sweep (concurrency 모드, conc=1)
  │     → TTFT vs input_len 그래프, ms/chunk 기울기
  └── 산출물: NPU chunked prefill 성능 기준선

Phase 2: Multi-turn (Day 2-3)
  ├── B-1: IDE Chat 5턴 (user-centric 모드)
  │     → Turn별 TTFT, ITL 안정성, E2E latency
  ├── B-2: IDE Chat 5턴 + prefix 3K (user-centric + sticky)
  │     → B-1 대비 prefix caching TTFT 절감률
  ├── C-1: Agent Loop 12턴 (user-centric 모드)
  │     → TTFT vs Turn# 곡선 (Token Snowball)
  └── 산출물: Turn별 TTFT 그래프, prefix caching 효과

Phase 3: Stress & SLO (Day 4)
  ├── C-agg: Agent 15턴 40K 벽 (user-centric 모드)
  │     → 에러 발생 턴, TTFT 급등 지점
  ├── G-1: IDE Chat + Goodput TTFT≤3s (user-centric + goodput)
  ├── G-2: Agent + Goodput TTFT≤5s (user-centric + goodput)
  └── 산출물: max context 한계, SLO 달성률

Analysis & Report (Day 5)
  ├── 전 Profile 결과 종합 표/그래프
  ├── NPU 고유 특성 도출
  │     - chunked prefill ms/chunk 기울기
  │     - block-size 4096 KV 낭비율
  │     - prefix caching 절감률
  └── 보고서 작성
```

---

## 부록 A: 전체 실행 명령 Quick Reference

| ID | Profile | AIPerf 모드 | 핵심 파라미터 | 요청 수 |
|:--:|:--------|:-----------|:------------|:------:|
| A-1 | Tab Baseline | **Concurrency** | ISL=2K, OSL=50, conc=4, req=200 | 200 |
| A-2 | Tab Sweep | **Concurrency** | ISL=500~16K, OSL=50, conc=1, req=30×6 | 180 |
| B-1 | Chat 기본 | **User-centric** | ISL=2K, OSL=500, 5턴, users=2, rate=0.5 | ~50 |
| B-2 | Chat Prefix | **User-centric** | B-1 + prefix=3K, sticky | ~50 |
| C-1 | Agent 12턴 | **User-centric** | ISL=1.2K, OSL=300, 12턴, users=1, rate=2.0 | ~60 |
| C-agg | Agent 15턴 | **User-centric** | ISL=1.5K, OSL=400, 15턴, users=1, rate=3.0 | 45 |
| G-1 | Chat+SLO | **User-centric** | B-1 + goodput TTFT≤3s | ~50 |
| G-2 | Agent+SLO | **User-centric** | C-1 + goodput TTFT≤5s | ~60 |

---

## 부록 B: 참조 자료 전체 목록

### 코딩 에이전트 토큰 소비 분석

| # | 자료 | URL |
|:-:|:-----|:----|
| 1 | OpenHands 토큰 분석 — run간 최대 10×, input 지배적 | https://openreview.net/forum?id=1bUeVB3fov |
| 2 | Agent Framework 비교 — 7개 framework 토큰 분배 | https://arxiv.org/abs/2511.00872 |
| 3 | SWE-Effi — Token Snowball Effect, EuTB 메트릭 | https://arxiv.org/abs/2509.09853 |
| 4 | SE Agent Trajectories — 120 trajectory, 2,822 interactions | https://arxiv.org/abs/2506.18824 |
| 5 | Reducing Token Usage (TU Wien) | https://repositum.tuwien.at/bitstream/20.500.12708/224666/1/Hrubec%20Nicolas%20-%202025%20-%20Reducing%20Token%20Usage%20of%20Software%20Engineering%20Agents.pdf |
| 6 | SWE-Pruner — 23–38% 절감 | https://arxiv.org/abs/2601.16746 |
| 7 | SWE-Skills-Bench — skill injection 영향 | https://arxiv.org/abs/2603.15401 |
| 8 | GitTaskBench — repo 크기 vs 토큰 | https://arxiv.org/abs/2508.18993 |

### Multi-Agent 토큰 분배

| # | 자료 | URL |
|:-:|:-----|:----|
| 9 | AgentTaxo — I:O 2:1~3:1 | https://openreview.net/pdf?id=0iLbiYYIpC |
| 10 | AgentDropout — 동적 agent 제거 | https://aclanthology.org/2025.acl-long.1170.pdf |
| 11 | Codified Prompting — input 67.8% 절감 | https://arxiv.org/abs/2507.03254 |

### 프로덕션 워크로드 트레이스

| # | 자료 | URL |
|:-:|:-----|:----|
| 12 | DynamoLLM — Coding vs Conversation fleet | https://jovans2.github.io/files/DynamoLLM_HPCA2025.pdf |
| 13 | OpenRouter 100T — prompt 4×↑, Programming driver | https://arxiv.org/abs/2601.10088 |
| 14 | Fireworks — 36% coding, 80% top traffic | https://fireworks.ai/blog/state-of-agent-environments |
| 15 | BurstGPT — 10.31M traces, Zipf 분포 | https://arxiv.org/abs/2401.17644 |

### Claude Code / Codex CLI

| # | 자료 | URL |
|:-:|:-----|:----|
| 16 | Claude Code 31K overhead | https://github.com/anthropics/claude-code/issues/24243 |
| 17 | Anthropic Tool Use 50K+ | https://www.anthropic.com/engineering/advanced-tool-use |
| 18 | Claude Context API 200K/1M | https://platform.claude.com/docs/en/build-with-claude/context-windows |
| 19 | Codex CLI — compaction 아키텍처 | https://newsletter.pragmaticengineer.com/p/how-codex-is-built |
| 20 | Claude Code vs Codex 비교 | https://labs.adaline.ai/p/claude-code-vs-openai-codex |

### 2026 Q1 최신

| # | 자료 | URL |
|:-:|:-----|:----|
| 21 | Tokenomics (MSR 2026) — SDLC 토큰 분배 | https://arxiv.org/abs/2601.14470 |
| 22 | AGENTS.md — runtime 28.64%↓ | https://arxiv.org/abs/2601.20404 |
| 23 | Dynamic Reasoning — prefix caching 51.7%↓ | https://arxiv.org/abs/2506.04301 |
| 24 | Iternal Token Guide — task당 1–3.5M | https://iternal.ai/token-usage-guide |
| 25 | Token Tracking Tools — ccusage 등 30+ 도구 | https://blog.starmorph.com/blog/ai-token-throughput-tracking-tools |
| 26 | Energy of Coding Agents — Claude Code 세션 에너지 | https://www.simonpcouch.com/blog/2026-01-20-cc-impact/ |

### IDE/서비스 Context Window

| # | 자료 | URL |
|:-:|:-----|:----|
| 27 | Cursor Tab Fusion — 13K context, p50 260ms | https://cursor.com/blog/tab-update |
| 28 | Cursor 1M+ QPS autocomplete | https://blog.bytebytego.com/p/how-cursor-serves-billions-of-ai |
| 29 | Copilot Chat 64K/128K | https://github.blog/changelog/2024-12-06-copilot-chat-now-has-a-64k-context-window-with-openai-gpt-4o/ |
| 30 | Cursor IDE Normal 128K / Max 200K | https://www.qodo.ai/blog/claude-code-vs-cursor/ |
| 31 | SWE-bench max 150 steps, 1M token | https://www.vals.ai/benchmarks/swebench |
| 32 | Devin — 7.8min avg, 67% PR merge | https://cognition.ai/blog/devin-annual-performance-review-2025 |

### 벤치마크 도구

| # | 자료 | URL |
|:-:|:-----|:----|
| 33 | AIPerf (ai-dynamo) | https://github.com/ai-dynamo/aiperf |
| 34 | AIPerf Comprehensive Guide | https://github.com/ai-dynamo/aiperf/blob/main/docs/comprehensive-llm-benchmarking.md |
| 35 | vLLM multi-turn bench RFC | https://github.com/vllm-project/vllm/issues/20265 |
| 36 | Pliops multi-turn 가이드 | https://pliops.com/setting-the-standard-multi-turn-benchmarking-in-vllm/ |
| 37 | llm-d multi-turn routing | https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d |
