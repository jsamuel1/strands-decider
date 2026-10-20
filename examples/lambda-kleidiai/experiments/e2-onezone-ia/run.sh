#!/bin/bash
# E2 in one command, from any host with Python 3.9+, boto3 and AWS credentials for the target account (no build: it
# reuses P1's pinned zip and GGUF, which P1's deploy.py upload put in SOURCE_BUCKET):
#
#   AWS_PROFILE=<profile> SOURCE_BUCKET=<P1 bucket> examples/lambda-kleidiai/experiments/e2-onezone-ia/run.sh [RUN_ID]
#
# deploy.py up -> baseline/bench.py --arms std,ozia -> summarize.py -> baseline/redact.py. Env knobs (defaults are
# P1's published baseline's): REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-e2
# TEARDOWN=versions|all|none. Raw output stays in the git-ignored build/runs/<RUN_ID>/; results/<RUN_ID>/ is the
# redacted copy that gets committed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(cd "$HERE/../../baseline" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
RAW=$HERE/build/runs/$RUN_ID
PY=${PYTHON:-python3}
: "${SOURCE_BUCKET:?set SOURCE_BUCKET to the bucket that the P1 deploy.py upload wrote (pinned zip and GGUF)}"
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }
# P1's pushed baseline HEAD this experiment is measured against (baseline/ is unchanged from it).
BASELINE_COMMIT=${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}
EXPERIMENT_COMMIT=$(git -C "$HERE" rev-parse HEAD)

"$PY" "$HERE/deploy.py" up --region "$REGION" --prefix "${PREFIX:-lkai-e2}" --source-bucket "$SOURCE_BUCKET"
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy.py" teardown --region "$REGION" ;;
    all) "$PY" "$HERE/deploy.py" teardown --all --region "$REGION" ;;
    none) ;;
    *) echo "TEARDOWN must be versions, all, or none" >&2; return 2 ;;
  esac
}
trap cleanup_versions EXIT
"$PY" "$BASE/bench.py" --build "$HERE/build" --arms std,ozia --region "$REGION" --rounds "${ROUNDS:-5}" \
  --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
DIRTY=()
[ -z "$(git -C "$HERE" status --porcelain -- "$HERE" "$BASE")" ] || DIRTY=(--experiment-dirty)
"$PY" "$HERE/summarize.py" "$RAW" --baseline-commit "$BASELINE_COMMIT" --meta "$HERE/meta.json" \
  --deployment "$HERE/build/deployment.json" --reference "$BASE/reference/outputs.json" \
  --experiment-commit "$EXPERIMENT_COMMIT" "${DIRTY[@]}"
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$HERE/build/deployment.json")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
"$PY" "$BASE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
P2=$HERE/../../results/publish_result.py
[ ! -f "$P2" ] || "$PY" "$P2" --validate-only "$HERE/results/$RUN_ID/results.json"
cleanup_versions
trap - EXIT
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
