# Goal — vLLM engine_args autotune for Qwen2.5-1.5B-Instruct on RTX 5060 Ti 16GB

## Objective
Maximize the composite score:

```
penalty = max(0, 1 - ttft.p99 / 1000)     # SLO 경계(500ms)에서 0.5, 1000ms에서 0
score   = throughput_tok.avg × penalty
```

Scores are 0 if any SLO constraint fails.

## Hard constraints (SLO, any failure → score = 0)
- `ttft.p99 ≤ 500ms`
- `e2e.p99 ≤ 30000ms`
- `failure_rate ≤ 1%`

## Files you may modify
Only the `deployment.engine_args` block of this single YAML:

- `configs/endpoints/local_vllm_smoke.yaml`

Do NOT touch: `model`, `tokenizer`, `url`, `parallelism.tp|pp|dp|ep` (fixed at 1).

## Search space — see `autoresearch/search_space.yaml`

Tier 1 axes (recommended to explore first):
- `max_num_seqs` ∈ {32, 64, 128, 256}
- `enable_prefix_caching` ∈ {true, false}
- `enable_chunked_prefill` ∈ {true, false}
- `gpu_memory_utilization` ∈ {0.80, 0.85, 0.90}

Keep pinned:
- `enforce_eager: false`
- `max_model_len: 4096`

## Benchmark script

For each experiment, run against all 3 workloads (short, medium, long) and
sum scores. The driver script:

```bash
./scripts/vllm_restart.sh configs/endpoints/local_vllm_smoke.yaml
TOTAL=0
for W in short medium long; do
  OUT=$(./scripts/bench_score.py \
          -p configs/profiles/autotune/${W}.yaml \
          -e configs/endpoints/local_vllm_smoke.yaml)
  S=$(echo "$OUT" | jq .score)
  echo "[$W] $OUT"
  TOTAL=$(echo "$TOTAL + $S" | bc -l)
done
echo "TOTAL_SCORE=$TOTAL"
```

Emit `TOTAL_SCORE` as the objective autoresearch reads.

## Notes
- Reproducibility gate is built into `bench_score.py`: 3 repeats; if
  `CV(throughput_tok) ≥ 0.10` it auto-extends to 5 and sets
  `accepted=false` if still noisy.
- The full `bench` CLI (`bench variance`, `bench nway`) is available for
  post-hoc analysis between experiments.
- `bench ls` shows all runs; each is persisted to DuckDB at `data/db/bench.duckdb`.
- If vLLM fails to start on some engine_args combination, record the failure
  and skip to the next hypothesis.
