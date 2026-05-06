# vllm 0.17.1 CLI args catalog (Source-verified)

> 본 catalog 는 vllm v0.17.1 tag 의 source code (arg_utils.py + parallel.py + vllm_config.py) 직접 fetch 로 작성. trial-and-error 로 결함 발견하지 않도록 search-space 작성 / 검증의 1차 reference.

> 마지막 검토: 2026-05-07 / 검토 주기: vllm 메이저 업데이트 시 / 소유자: B200 lmtune track / 상태: living

## 1. 출처

- `https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/engine/arg_utils.py` (2269 lines)
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/config/parallel.py` (839 lines)
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.17.1/vllm/config/vllm.py` (1787 lines)

## 2. ParallelConfig — Wide-EP / DBO / EPLB / PCP / DCP 관련

| CLI flag | short | type | default | source | 비고 |
|:--|:--|:--|:--|:--|:--|
| `--tensor-parallel-size` | `-tp` | int | 1 | arg_utils:816 | chart auto-inject (lmtune adapter 가 vllmArgs 안 거치고 decode.parallelism.tensor 로 emit) |
| `--data-parallel-size` | `-dp` | int | 1 | arg_utils:838 | chart auto-inject |
| `--pipeline-parallel-size` | `-pp` | int | 1 | (별도 grep 필요) | |
| **`--prefill-context-parallel-size`** | `-pcp` | int | 1 | arg_utils:834 | **valid CLI flag** — R11 `_SIMULATOR_ONLY_KEYS` 에서 제거 가능 (chart wiring 검증 후) |
| **`--decode-context-parallel-size`** | `-dcp` | int | 1 | arg_utils:820 | 동일. 제약: `tp % dcp == 0` (parallel.py:389) |
| `--enable-expert-parallel` | `-ep` | bool | False | arg_utils:889 | wide-EP 의 ON 스위치 |
| **`--all2all-backend`** | (none) | Literal | `"allgather_reducescatter"` | arg_utils:894 | **7 valid choice** — § 2.1 |
| `--enable-dbo` | (none) | bool | False | arg_utils:895 | Dual Batch Overlap. **deepep_* 만** § 2.2 |
| `--ubatch-size` | (none) | int | 0 | arg_utils:896 | DBO microbatch 크기 |
| **`--dbo-decode-token-threshold`** | (none) | int | 32 | arg_utils:907 | decode 시 DBO 활성 임계값 |
| **`--dbo-prefill-token-threshold`** | (none) | int | 512 | arg_utils:911 | prefill 시 DBO 활성 임계값 |
| `--enable-eplb` | (none) | bool | False | arg_utils:915 | Expert Load Balancing. **ep=true 필수** § 2.3 |
| **`--eplb-config`** | (none) | JSON dict | `{}` | arg_utils:916 | **window_size / step_interval / num_redundant_experts 가 이 안에**. 별개 flag 아님 (R17a/b) |
| `--expert-placement-strategy` | (none) | Literal | `"linear"` | arg_utils:917 | choices: `linear`, `round_robin` |
| `--enable-elastic-ep` | (none) | bool | False | arg_utils:903 | Stateless NCCL groups for DP/EP |
| `--data-parallel-backend` | `-dpb` | Literal | `"mp"` | arg_utils:870 | choices: `mp`, `ray` |
| `--data-parallel-hybrid-lb` | `-dph` | bool | False | arg_utils:879 | |
| `--data-parallel-external-lb` | `-dpe` | bool | False | arg_utils:884 | |
| `--disable-nccl-for-dp-synchronization` | (none) | bool | None | arg_utils:912 | async-scheduling 활성 시 default True |
| `--max-parallel-loading-workers` | (none) | int | None | arg_utils:923 | |
| `--ray-workers-use-nsight` | (none) | bool | False | arg_utils:927 | |
| `--disable-custom-all-reduce` | (none) | bool | False | arg_utils:930 | |
| `--worker-cls` | (none) | str | (default) | arg_utils:933 | |
| `--worker-extension-cls` | (none) | str | (default) | arg_utils:934 | |

### 2.1 `--all2all-backend` valid choices (parallel.py:39-49)

```python
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "allgather_reducescatter",
    "flashinfer_all2allv",
]
```

→ **7개**. 그 외 값은 vllm argparse 에서 `invalid choice` 거부 (R14a 의 `nixl_ep` 가 이 패턴).

### 2.2 DBO (microbatching) 호환성 — vllm_config.py:1128-1134

```python
if a2a_backend not in (
    "deepep_low_latency",
    "deepep_high_throughput",
):
    raise ValueError(
        "Microbatching currently only supports the deepep_low_latency and "
        f"deepep_high_throughput all2all backend. {a2a_backend} is not "
        "supported. To fix use --all2all-backend=deepep_low_latency or "
        "--all2all-backend=deepep_high_throughput and install the DeepEP"
    )
```

→ **`--enable-dbo` ⇒ `--all2all-backend ∈ {deepep_low_latency, deepep_high_throughput}`** (R16)

### 2.3 EPLB 호환성 — parallel.py 의 `_validate_*`

`enable_eplb=True` 시 반드시 `enable_expert_parallel=True` 필요. 미충족 시 pydantic ValidationError:

```
enable_expert_parallel must be True to use EPLB.
```

→ **`--enable-eplb` ⇒ `--enable-expert-parallel`** (R15)

### 2.4 `--eplb-config` JSON 구조 (parallel.py:54-86)

```python
class EPLBConfig:
    window_size: int = 1000
    step_interval: int = 3000
    num_redundant_experts: int = 0
    log_balancedness: bool = False
    log_balancedness_interval: int = 1
    use_async: bool = False
    policy: Literal["default"] = "default"
```

CLI: `--eplb-config '{"window_size": 1000, "step_interval": 3000}'` 형태로 JSON 통째 전달. 별개 flag 가 아니라 (R17a/b 의 root cause).

→ lmtune adapter 가 search-space 의 `eplb_window_size`, `eplb_step_interval` axis 를 JSON 합치는 wiring 추가가 필요한 영역. 본 PR 외.

### 2.5 DCP 제약 — parallel.py:389

```python
if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
    raise ValueError(
        f"tp_size={self.tensor_parallel_size} must be divisible by "
        f"dcp_size={self.decode_context_parallel_size}."
    )
```

→ DCP 활성 시 `tp % dcp == 0` (vllm-config-puzzle simulator 와 동일)

## 3. SchedulerConfig — Batching / Prefill (arg_utils:1154-1212)

| CLI flag | type | default | 비고 |
|:--|:--|:--|:--|
| `--max-num-batched-tokens` | int | None | chunked-prefill token budget |
| `--max-num-seqs` | int | None | concurrent sequence 수 |
| `--max-num-partial-prefills` | int | (chart default) | |
| `--max-long-partial-prefills` | int | (chart default) | |
| `--long-prefill-token-threshold` | int | (chart default) | |
| `--scheduling-policy` | Literal | (chart default) | |
| `--enable-chunked-prefill` | bool | None | |
| `--disable-chunked-mm-input` | bool | False | |
| `--scheduler-cls` | str | (default) | |
| `--disable-hybrid-kv-cache-manager` | bool | False | |
| `--async-scheduling` | bool | False | DP synchronization 영향 (parallel.py:185) |
| `--stream-interval` | int | (default) | |

## 4. CacheConfig — KV / Memory (arg_utils:944-)

| CLI flag | type | default | 비고 |
|:--|:--|:--|:--|
| `--block-size` | Literal | (chart default) | choices: 16, 32, 64, 128 (확인 필요) |
| `--gpu-memory-utilization` | float | 0.9 | |
| `--kv-cache-memory-bytes` | int | None | |
| `--kv-cache-dtype` | Literal | "auto" | choices: auto, fp8, fp8_e4m3, fp8_e5m2 (별도 grep 필요) |
| `--enable-prefix-caching` | bool | None | |
| `--prefix-caching-hash-algo` | Literal | "builtin" | choices: builtin, sha256 |
| ... | | | |

## 5. ModelConfig — Model / dtype (arg_utils:670-)

| CLI flag | type | default | 비고 |
|:--|:--|:--|:--|
| `--dtype` | Literal | "auto" | choices: auto, half, float16, bfloat16, float, float32. **MXFP4 native 모델은 미지정 권장** (R13) |
| `--max-model-len` | int | None | model config 의 native 한도가 상한 |
| `--quantization` | Literal | None | choices: 다수 (별도 catalog 필요 시 추가) |
| ... | | | |

## 6. Cross-flag 제약 요약 (search-space 작성 시 의무 검증)

| 제약 | 출처 | 처리 방법 |
|:--|:--|:--|
| `--enable-eplb` ⇒ `--enable-expert-parallel` | parallel.py validator | search-space 에서 ep axis 제거 + values gotmpl 에서 강제 ON (R15) |
| `--enable-dbo` ⇒ `--all2all-backend ∈ {deepep_low_latency, deepep_high_throughput}` | vllm_config.py:1128 | search-space 의 all2all_backend 를 deepep 2개로 (R16) 또는 enable_dbo active_if |
| DCP > 1 ⇒ `tp % dcp == 0` | parallel.py:389 | feasibility constraint |
| `prefill_context_parallel_size * tp` 가 동시에 너무 크면 NCCL ring 경합 | (실측) | 제약 룰 미정 |
| `--data-parallel-rank`/external-lb 명시 ⇒ implicit `data_parallel_external_lb=True` | parallel.py | 동시 사용 주의 |

## 7. lmtune adapter ↔ vllm flag 매핑

`src/lmtune/deploy/llmd_k8s.py::render_values_overlay` 가 다음 변환:

| search-space axis (snake) | vllmArgs (kebab) | 챠트 emit | 비고 |
|:--|:--|:--|:--|
| `tp` | (parallelism.tp) | `decode.parallelism.tensor` (chart auto-inject) | 별도 vllmArgs 안 거침 |
| `dp` | (parallelism.dp) | `decode.parallelism.data` (chart auto-inject) | |
| `pp` | `pipeline-parallel-size` | vllmArgs | |
| `ep` (제거됨) | `enable-expert-parallel` | values gotmpl `$defaults` 에서 항상 True (R15) | |
| `all2all_backend` | `all2all-backend` | vllmArgs | (R16 으로 deepep 만) |
| `enable_dbo` | `enable-dbo` | vllmArgs (bool true 시 emit) | |
| `enable_eplb` | `enable-eplb` | vllmArgs | |
| `dbo_decode_token_threshold` | `dbo-decode-token-threshold` | vllmArgs | (R17c — 이전 wrong name dbo_token_threshold) |
| `max_num_seqs` 등 | `max-num-seqs` 등 | vllmArgs | adapter 가 snake → kebab 변환 |

## 8. 본 catalog 신규 결함 검출 시 절차

1. vllm release notes / source 확인 (`https://github.com/vllm-project/vllm/releases/tag/v<VERSION>`)
2. 본 catalog 에 변경된 flag / valid choice / 제약 추가
3. `b200/search-spaces/*.yaml` 의 모든 axis 가 catalog 에 있는지 1:1 검증
4. 신규 결함은 `b200/docs/regressions.md` 에 R<n> entry
5. PR 에 catalog + search-space + regressions 동시 업데이트

## 9. 향후 — 자동 validator (별도 PR)

`b200/scripts/validate_search_space.py` 신설 예정. 목표:
- 모든 search-space YAML 의 axis name 이 catalog 에 있는 CLI flag 인지 검증
- categorical axis 의 values 가 catalog 의 Literal choices 안에 있는지
- 본 catalog § 6 의 cross-flag 제약 위반 (`active_if` 또는 `feasibility_constraints` 로 표현됐는지)
- pytest mark `@pytest.mark.search_space_validity` 로 자동 회귀
