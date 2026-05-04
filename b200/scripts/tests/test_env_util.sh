#!/usr/bin/env bash
# env util 회귀.
set -euo pipefail
IFS=$'\n\t'

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source b200/scripts/util/env.sh
# shellcheck disable=SC1091
source b200/scripts/tests/fakes/setup.sh
trap 'rm -rf "$FAKE_BIN"' EXIT
PATH="$FAKE_BIN:$PATH"

set -u

# bench_env::values_for_model — 매핑 카탈로그
[[ "$(bench_env::values_for_model openai/gpt-oss-120b)" == "values-gpt-oss-120b.yaml.gotmpl" ]]
[[ "$(bench_env::values_for_model openai/gpt-oss-20b)"  == "values-gpt-oss-20b.yaml.gotmpl" ]]
[[ "$(bench_env::values_for_model meta-llama/Llama-3.1-8B-Instruct)" == "values-llama-3.1-8b-smoke.yaml.gotmpl" ]]
bench_env::values_for_model UnknownVendor/Model 2>/dev/null && {
  echo "expected fail on unknown model"; exit 1;
} || true

# bench_env::model_from_endpoint — endpoint YAML 의 model 추출
got=$(bench_env::model_from_endpoint b200/endpoints/b200_gpt-oss-120b.yaml)
[[ "$got" == "openai/gpt-oss-120b" ]] || { echo "expected gpt-oss-120b, got: $got"; exit 1; }

# bench_env::cluster_check — R2 회귀 (set -u + 한 줄 local)
FAKE_K8S=unreachable bench_env::cluster_check infsch 2>/dev/null && {
  echo "expected fail (unreachable)"; exit 1;
} || true
FAKE_K8S=no_ns bench_env::cluster_check infsch 2>/dev/null && {
  echo "expected fail (no ns)"; exit 1;
} || true
FAKE_K8S=has_decode bench_env::cluster_check infsch >/dev/null

# bench_env::require_model_values — 3 case
unset B200_MODEL_VALUES
bench_env::require_model_values "values-x.yaml" 2>/dev/null && { echo "expected fail unset"; exit 1; } || true

export B200_MODEL_VALUES=values-llama.yaml
bench_env::require_model_values "values-x.yaml" 2>/dev/null && { echo "expected mismatch fail"; exit 1; } || true

export B200_MODEL_VALUES=values-x.yaml
bench_env::require_model_values "values-x.yaml" >/dev/null
