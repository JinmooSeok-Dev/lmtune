# Autotune Summary — experiments.jsonl

- Total experiments: 7
- SLO-passing: 6
- SLO-failing: 1
- vLLM start failures: 0

## Top results (by total_score across 3 workloads)

| rank | label | total_score | short | medium | long | max_num_seqs | prefix_cache | chunked_prefill | gpu_mem |
|-----:|:------|-----------:|------:|-------:|-----:|:-------------|:-------------|:----------------|:--------|
| 1 | `chunked-prefill-on` | 1395.0 | 740.9 | 654.0 | 0.0 | 128 | True | True | 0.85 |
| 2 | `gpu-mem-util-0.80` | 1392.2 | 739.0 | 653.2 | 0.0 | 128 | True | False | 0.8 |
| 3 | `gpu-mem-util-0.90` | 1390.5 | 735.5 | 655.0 | 0.0 | 128 | True | False | 0.9 |
| 4 | `max-num-seqs-64` | 1382.7 | 734.0 | 648.7 | 0.0 | 64 | True | False | 0.85 |
| 5 | `max-num-seqs-256` | 1380.9 | 727.5 | 653.4 | 0.0 | 256 | True | False | 0.85 |

## SLO-failing experiments

| label | total_score | TTFT/E2E fail detail |
|:------|-----------:|:---------------------|
| `baseline-default` | 653.0 | short: ttft_p99=713ms |

## Winner

`chunked-prefill-on`  score=1395.0
engine_args: `{"enable_chunked_prefill": true, "enable_prefix_caching": true, "gpu_memory_utilization": 0.85, "max_model_len": 4096, "max_num_seqs": 128}`
