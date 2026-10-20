#!/bin/bash
# E4 in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/e4_stream/run.sh [RUN_ID]
#
# 1. e4_stream/build.sh   P1's pinned weights and llama.cpp, plus patch_stream.py, built into one function.zip
# 2. deploy_e4.py up      one function per arm of deploy_e4.ARMS_SETS[E4_ARMS] (r2 default; r1 = the first run):
#                         same zip and config, different E4_* / STREAM_ORDER variables
# 3. bench_e4.py         P1's interleaved cold / restore / warm harness (baseline/bench.py) over the three arms, with a
#                         fresh Lambda client per wave (bench_e4.py says why)
# 4. summarize_e4.py      lkai-result/1, correctness against P1's reference/outputs.json
# 5. baseline/redact.py   results/<RUN_ID>/ is what gets committed
# Env knobs as baseline/run.sh: REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-e4
# TEARDOWN=versions|all|none, E4_ARMS=r1|r2. LKAI_BUILD (default baseline/build) is P1's build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(cd "$HERE/../baseline" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
PREFIX=${PREFIX:-lkai-e4}
B=${E4_BUILD:-$HERE/build}
RAW=$B/runs/$RUN_ID
PY=${PYTHON:-python3}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }
# P1's pushed baseline HEAD that this branch starts from (feat/lambda-kleidiai-baseline on the fork).
BASELINE_COMMIT=${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}

ARMS=$("$PY" -c "import sys; sys.argv=['x']; sys.path.insert(0, '$HERE'); import deploy_e4; print(','.join(deploy_e4.ARMS_ENV))" 2>/dev/null) || { echo "deploy_e4 import failed"; exit 1; }
[ -f "$B/function.zip" ] && [ -z "${REBUILD:-}" ] || "$HERE/build.sh"
"$PY" "$HERE/deploy_e4.py" up --build "$B" --region "$REGION" --prefix "$PREFIX"
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy_e4.py" teardown --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    all) "$PY" "$HERE/deploy_e4.py" teardown --all --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    none) ;;
    *) echo "TEARDOWN must be versions, all, or none" >&2; return 2 ;;
  esac
}
trap cleanup_versions EXIT
"$PY" "$HERE/bench_e4.py" --build "$B" --arms "$ARMS" --region "$REGION" \
  --rounds "${ROUNDS:-5}" --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
"$PY" "$HERE/summarize_e4.py" "$RAW" --reference "$BASE/reference/outputs.json" --meta "$HERE/meta.json" \
  --experiment-commit "$(git -C "$HERE" rev-parse HEAD)" --baseline-commit "$BASELINE_COMMIT"
cp "$B/manifest.json" "$RAW/build-manifest.json"
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$B/deployment.json")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
"$PY" "$BASE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
P2=$HERE/../results/publish_result.py
[ ! -f "$P2" ] || "$PY" "$P2" --validate-only "$HERE/results/$RUN_ID/results.json"
cleanup_versions
trap - EXIT
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
