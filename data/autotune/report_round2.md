# Autotune Summary — experiments_round2.jsonl

- Total experiments: 8
- SLO-passing: 2
- SLO-failing: 6
- vLLM start failures: 0

## Top results (by total_score across 3 workloads)

| rank | label | model | total | short | medium | long | mnseq | pfx | chunk | gpumem |
|-----:|:------|:------|-----:|------:|-------:|-----:|:------|:----|:------|:-------|
| 1 | `qwen25-1.5b-winner` | `Qwen2.5-1.5B-Instruct` | 1906.4 | 731.6 | 662.1 | 512.7 | 128 | True | True | 0.85 |
| 2 | `qwen3-1.7b-winner` | `Qwen3-1.7B` | 1685.7 | 717.6 | 598.6 | 369.6 | 128 | True | True | 0.85 |

## Per-workload winners

- **short**: `qwen25-1.5b-winner` (Qwen2.5-1.5B-Instruct) score=731.6 throughput=814.7tok/s ttft_p99=102ms
- **medium**: `qwen25-1.5b-winner` (Qwen2.5-1.5B-Instruct) score=662.1 throughput=711.7tok/s ttft_p99=70ms
- **long**: `qwen25-1.5b-winner` (Qwen2.5-1.5B-Instruct) score=512.7 throughput=592.9tok/s ttft_p99=135ms

## SLO-failing experiments

| label | total_score | TTFT/E2E fail detail |
|:------|-----------:|:---------------------|
| `qwen3-0.6b-winner` | 1302.0 | short: ttft_p99=1161ms |
| `qwen3-4b-winner` | 727.4 | long: ttft_p99=20556ms |
| `qwen3-8b-awq-winner` | 0.0 | short: ttft_p99=4465ms; medium: ttft_p99=584ms; long: ttft_p99=2662ms |
| `qwen3-14b-awq-winner` | 0.0 | short: ttft_p99=745ms; medium: ttft_p99=5369ms; long: ttft_p99=53746ms |
| `qwen3-4b-fp8-winner` | 1085.6 | long: ttft_p99=252ms |
| `qwen3-8b-fp8-winner` | 842.0 | long: ttft_p99=18259ms |

## Winner

`qwen25-1.5b-winner`  score=1906.4
engine_args: `{"enable_chunked_prefill": true, "enable_prefix_caching": true, "gpu_memory_utilization": 0.85, "max_model_len": 4096, "max_num_seqs": 128}`
