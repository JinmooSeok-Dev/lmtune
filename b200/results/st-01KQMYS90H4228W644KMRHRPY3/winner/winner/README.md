# Winner Recipe — w-local-mvp

> Auto-generated from `lmtune search export st-01KQMYS90H4228W644KMRHRPY3 --winner top-1`
> Study created: 2026-05-03 03:21:33.078685 · finished: 2026-05-03 03:47:52.501217

## Identifiers

| Field | Value |
|:---|:---|
| Study ID | `st-01KQMYS90H4228W644KMRHRPY3` |
| Study Name | w-local-mvp |
| Strategy | `tpe` |
| Metric | `total_score` (direction: maximize) |
| Trial ID | `tr-01KQMZVTTM28VNZYE9T0GYV7EN` |
| Trial seq | 8 |
| Score | **736.24** |
| Adapter | `local-vllm` |
| Endpoint | `<set ENDPOINT env>` |

## Winning Parameters

```json
{
  "enable_chunked_prefill": false,
  "enable_prefix_caching": true,
  "gpu_memory_utilization": 0.8139042871430157,
  "max_num_seqs": 64
}
```

## How to Apply

### Dry run (no changes)
```bash
bash apply.sh --dry-run
```
Renders the values overlay and shows what would happen without touching the cluster.

### Live apply
```bash
bash apply.sh --apply
```
- For `local-vllm`: writes engine_args to the endpoint YAML, runs `scripts/vllm_restart.sh`.
- For `llmd-k8s`: runs `helmfile -f <peer> --state-values-file values-overlay.yaml apply`,
  waits for deployment rollout, probes `/v1/models`.

## Reproducibility

Re-run this exact configuration on a different cluster:

```bash
git clone <this-repo>
cd <repo>
pip install -e ".[search,distributed]"
ENDPOINT=<set ENDPOINT env> bash b200/results/st-01KQMYS90H4228W644KMRHRPY3/winner/winner/apply.sh --apply
```

## Related artifacts

- `apply.sh` — the apply script (this directory)
- `values-overlay.yaml` — Helm values overlay (consumed by helmfile)
- `params.json` — raw params dict (machine readable)


## Workload-level Metrics (top-1 measured)

| metric | workload | value |
|:---|:---|---:|

| cv_throughput | short | 0.02 |

| e2e_p99 | short | 1.34 |

| score | aggregate | 736.24 |

| score | short | 736.24 |

| throughput_tok_avg | short | 812.16 |

| ttft_p99 | short | 93.48 |



## Provenance

This recipe was selected from 10 trials in study `st-01KQMYS90H4228W644KMRHRPY3`,
which was started with search space `w-local-minimal`. To inspect the full study:

```bash
lmtune search status st-01KQMYS90H4228W644KMRHRPY3
lmtune dashboard build --out b200/dashboards
```
