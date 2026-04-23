# Per-Workload Goodput Ranking (Round 1 + Round 2)

- SLO: ttft_p99 ≤ 500ms, e2e_p99 ≤ 30000ms
- score = throughput_tok × max(0, 1 − ttft_p99/(2·500ms))
- per-workload `slo` field is the workload-level pass (NOT experiment aggregate)

## Workload: `short`

| rank | label | model | score | throughput | ttft_p99 | e2e_p99 | slo | src |
|-----:|:------|:------|-----:|-----------:|---------:|--------:|:---:|:----|
| 1 | `chunked-prefill-on` | `Qwen2.5-1.5B-Instruct` | 740.9 | 810.3 | 86ms | 1359ms | ✅ | experiments.jsonl |
| 2 | `gpu-mem-util-0.80` | `Qwen2.5-1.5B-Instruct` | 739.0 | 811.1 | 89ms | 1381ms | ✅ | experiments.jsonl |
| 3 | `gpu-mem-util-0.90` | `Qwen2.5-1.5B-Instruct` | 735.5 | 811.9 | 94ms | 1387ms | ✅ | experiments.jsonl |
| 4 | `max-num-seqs-64` | `Qwen2.5-1.5B-Instruct` | 734.0 | 811.8 | 96ms | 1389ms | ✅ | experiments.jsonl |
| 5 | `qwen25-1.5b-winner` | `Qwen2.5-1.5B-Instruct` | 731.6 | 814.7 | 102ms | 1345ms | ✅ | experiments_round2.jsonl |

**🏆 SLO-passing winner**: `chunked-prefill-on` on `Qwen2.5-1.5B-Instruct` — score=740.9, throughput=810.3tok/s, TTFT p99=86ms

## Workload: `medium`

| rank | label | model | score | throughput | ttft_p99 | e2e_p99 | slo | src |
|-----:|:------|:------|-----:|-----------:|---------:|--------:|:---:|:----|
| 1 | `qwen3-0.6b-winner` | `Qwen3-0.6B` | 785.6 | 822.7 | 45ms | 1844ms | ✅ | experiments_round2.jsonl |
| 2 | `qwen25-1.5b-winner` | `Qwen2.5-1.5B-Instruct` | 662.1 | 711.7 | 70ms | 2970ms | ✅ | experiments_round2.jsonl |
| 3 | `gpu-mem-util-0.90` | `Qwen2.5-1.5B-Instruct` | 655.0 | 710.0 | 77ms | 3153ms | ✅ | experiments.jsonl |
| 4 | `chunked-prefill-on` | `Qwen2.5-1.5B-Instruct` | 654.0 | 709.8 | 79ms | 3144ms | ✅ | experiments.jsonl |
| 5 | `prefix-cache-off` | `Qwen2.5-1.5B-Instruct` | 653.5 | 710.0 | 80ms | 3149ms | ✅ | experiments.jsonl |

**🏆 SLO-passing winner**: `qwen3-0.6b-winner` on `Qwen3-0.6B` — score=785.6, throughput=822.7tok/s, TTFT p99=45ms

## Workload: `long`

| rank | label | model | score | throughput | ttft_p99 | e2e_p99 | slo | src |
|-----:|:------|:------|-----:|-----------:|---------:|--------:|:---:|:----|
| 1 | `qwen3-0.6b-winner` | `Qwen3-0.6B` | 516.4 | 574.1 | 100ms | 7385ms | ✅ | experiments_round2.jsonl |
| 2 | `qwen25-1.5b-winner` | `Qwen2.5-1.5B-Instruct` | 512.7 | 592.9 | 135ms | 7022ms | ✅ | experiments_round2.jsonl |
| 3 | `qwen3-1.7b-winner` | `Qwen3-1.7B` | 369.6 | 442.3 | 165ms | 11705ms | ✅ | experiments_round2.jsonl |
| 4 | `baseline-default` | `Qwen2.5-1.5B-Instruct` | 0.0 | 0.0 | 0ms | 0ms | ✅ | experiments.jsonl |
| 5 | `max-num-seqs-256` | `Qwen2.5-1.5B-Instruct` | 0.0 | 0.0 | 0ms | 0ms | ✅ | experiments.jsonl |

**🏆 SLO-passing winner**: `qwen3-0.6b-winner` on `Qwen3-0.6B` — score=516.4, throughput=574.1tok/s, TTFT p99=100ms

