# lmtune Contracts

> 본 디렉토리 = lmtune 의 **input/output spec 정본**. 외부 프로젝트나 사용자가 본 spec 만 만족하면 lmtune 위에서 바로 동작.

## Contract 6종

| # | Contract | apiVersion | Master | 문서 |
|:--|:---|:---|:---|:---|
| 1 | WorkloadSpec | `workloads/v1alpha1` | **lm-workloads** | [workload-spec.md](workload-spec.md) |
| 2 | ClusterSpec | `ariadne/cluster/v1alpha1` | **ariadne** | [cluster-spec.md](cluster-spec.md) |
| 3 | EndpointSpec | `lmtune/endpoint/v1alpha1` | lmtune | (후속 PR) |
| 4 | ProfileSpec | `lmtune/profile/v1alpha1` | lmtune | (후속 PR) |
| 5 | SearchSpace | `lmtune/search/v1alpha1` | lmtune | (후속 PR) |
| 6 | BenchmarkResult | `lmtune/result/v1alpha1` | lmtune | (lmtune#R0) |

→ 진행 상황은 [`docs/architecture/REFACTOR-PLAN.md`](../architecture/REFACTOR-PLAN.md) 참조.

## 외부 사용자가 lmtune 에 input 주는 4 방법

각 contract 는 **하나의 ABC + 다수 구현체** 패턴. 사용자는 4 방법 중 자기 환경에 맞는 걸 선택.

### 1. yaml 파일 직접 작성 (BYO)
```bash
lmtune run --workload-spec my-workload.yaml \
           --cluster-spec my-cluster.yaml \
           -p profile.yaml -e endpoint.yaml
```
JSON Schema 로 자기 환경에서 validate 가능: `lmtune contracts validate workload my-workload.yaml`

### 2. 기본 Provider (외부 master 호출)
```bash
# lm-workloads 가 운영 trace → WorkloadSpec
lmtune run --workload-source vllm-log:/var/log/vllm/access.log ...

# ariadne 가 host 토폴로지 → ClusterSpec
lmtune run --cluster-discover --inventory hosts.txt ...
```

### 3. profile yaml inline (현 방식 유지)
```yaml
# profile.yaml
workload:
  source: synthetic
  synthetic_input_tokens_mean: 1024
  output_tokens_mean: 256
```

### 4. 사용자 자기 Provider 추가 (entry_points)
```toml
# 외부 패키지의 pyproject.toml
[project.entry-points."lmtune.workload_providers"]
my_provider = "my_package.providers:MyProvider"
```

## Schema 게시

- **Pydantic model** (Python): `from lmtune.contracts.workload_spec import WorkloadSpec`
- **JSON Schema** (언어 무관): `lmtune contracts dump --kind workload --out workload.schema.json`
- **CHANGELOG**: 본 디렉토리의 `CHANGELOG.md` (각 contract 의 bump 이력)

## 변경 거버넌스

| Contract | 변경 권한 | bump 정책 |
|:---|:---|:---|
| WorkloadSpec | lm-workloads | master 가 v 올리면 lmtune mirror 도 같은 PR |
| ClusterSpec | ariadne | master 가 v 올리면 lmtune mirror 도 같은 PR |
| Endpoint/Profile/SearchSpace/Result | lmtune | 단독 관리. 추가 only, drop 은 v 분리 |
