#!/usr/bin/env bash
# 야간 sweep: 모든 profile을 지정 endpoint 에 대해 순차 실행하고 결과 diff 를 Markdown 으로 생성.
#
# 사용: ENDPOINT=configs/endpoints/llmd_k8s.yaml ./scripts/nightly_sweep.sh
#
# 필요 env var:
#   ENDPOINT              (필수)  configs/endpoints/*.yaml 경로
#   PROFILE_DIR           기본    configs/profiles
#   BENCH_DB              기본    data/db/bench.duckdb
#   BENCH_RAW             기본    data/raw
#   REGRESSION_THRESHOLD  기본    10.0
#   BASELINE_RUN          선택    지정 시 candidate 대비 compare 리포트 생성

set -euo pipefail

ENDPOINT="${ENDPOINT:-configs/endpoints/local_vllm.yaml}"
PROFILE_DIR="${PROFILE_DIR:-configs/profiles}"
BENCH_DB="${BENCH_DB:-data/db/bench.duckdb}"
BENCH_RAW="${BENCH_RAW:-data/raw}"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-10.0}"
OUT_DIR="data/reports/sweep_$(date +%Y%m%d_%H%M%S)"

export BENCH_DB
export BENCH_RAW

mkdir -p "$OUT_DIR"
echo "[sweep] endpoint=$ENDPOINT profile_dir=$PROFILE_DIR out=$OUT_DIR"
bench sweep --profile-dir "$PROFILE_DIR" --endpoint "$ENDPOINT" --continue-on-error \
    2>&1 | tee "$OUT_DIR/sweep.log"

# 이번 sweep 의 run_id 들 수집 (모든 profile 에 대해 가장 최근 20건)
RUN_IDS=$(bench ls -n 20 | awk 'NR>2 {print $2}' | tail -n +1 || true)

# BASELINE 이 있으면 각 신규 run 과 diff 리포트 생성
if [[ -n "${BASELINE_RUN:-}" ]]; then
    while read -r rid; do
        [[ -z "$rid" || "$rid" == "run_id" ]] && continue
        bench compare "$BASELINE_RUN" "$rid" --threshold-pct "$REGRESSION_THRESHOLD" \
            > "$OUT_DIR/diff_${rid}.md" 2>/dev/null || true
    done <<< "$RUN_IDS"
fi

echo "[sweep] done. reports under $OUT_DIR"
