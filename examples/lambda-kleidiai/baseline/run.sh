#!/bin/bash
# The baseline in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh [RUN_ID]
#
# build.sh (pinned weights + zip) -> deploy.py up -> bench.py -> summarize.py -> redact.py. Env knobs (defaults are the
# published baseline's): REGION=us-east-2 MEMORY=10240 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-p1
# TEARDOWN=versions|all|none. ROUNDS must be >= 2, CONCURRENCY and WARM >= 1 (bench.py rejects others). MEMORY (MB,
# 128..10240) sets both functions' memory, is checked on the deployed functions before the benchmark starts, and is
# recorded in conditions.json and results.json. REGION must have a pinned price table in summarize.PRICES_BY_REGION
# (only us-east-2 today); deploy.py refuses any other before creating anything.
# Raw output stays in the git-ignored build/runs/<RUN_ID>/; results/<RUN_ID>/ is the same with the bucket name and
# account id replaced, which is what gets committed. The first run's answers become reference/outputs.json, but only
# if every invocation succeeded with an answer and matched its expected start class (summarize.py exits 3 otherwise:
# the result is still written and redacted, the reference is not, and run.sh exits non-zero).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
MEMORY=${MEMORY:-10240}
RAW=$HERE/build/runs/$RUN_ID
PY=${PYTHON:-python3}
case "${TEARDOWN:-versions}" in versions | all | none) ;; *)
  echo "TEARDOWN must be versions, all, or none" >&2
  exit 2
  ;;
esac
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }
# deploy.py and bench.py validate it again (deploy.memory_mb); failing here costs nothing.
if ! [[ "$MEMORY" =~ ^[0-9]+$ ]] || ((10#$MEMORY < 128 || 10#$MEMORY > 10240)); then
  echo "MEMORY must be an integer 128..10240 MB, got '$MEMORY'" >&2
  exit 2
fi

"$HERE/build.sh"
"$PY" "$HERE/deploy.py" up --region "$REGION" --prefix "${PREFIX:-lkai-p1}" --memory "$MEMORY"
# Published versions incur SnapStart cache charges even if a later benchmark or summary step fails.
# Arm cleanup immediately after deploy; explicit TEARDOWN=none is the only opt-out. deploy.py teardown attempts every
# version and exits non-zero naming each one it could not delete.
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy.py" teardown --region "$REGION" ;;
    all) "$PY" "$HERE/deploy.py" teardown --all --region "$REGION" ;;
    none) ;;
  esac
}
# The run's own exit status survives the cleanup; a cleanup failure turns a successful run into a failed one.
on_exit() {
  local rc=$?
  trap - EXIT
  if ! cleanup_versions; then
    echo "CLEANUP FAILED: published versions may still exist and bill; run deploy.py teardown --region $REGION" >&2
    [ "$rc" -ne 0 ] || rc=1
  fi
  exit "$rc"
}
trap on_exit EXIT
"$PY" "$HERE/bench.py" --region "$REGION" --memory "$MEMORY" --rounds "${ROUNDS:-5}" \
  --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
REF=$HERE/reference/outputs.json
SUM=0
if [ -f "$REF" ]; then
  "$PY" "$HERE/summarize.py" "$RAW" --reference "$REF" --meta "$HERE/meta.json" || SUM=$?
else
  mkdir -p "$HERE/reference"
  rm -f "$HERE/build/outputs.json"
  "$PY" "$HERE/summarize.py" "$RAW" --write-reference "$HERE/build/outputs.json" --meta "$HERE/meta.json" || SUM=$?
fi
[ "$SUM" -eq 0 ] || [ "$SUM" -eq 3 ] || exit "$SUM"
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$HERE/build/deployment.json")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
"$PY" "$HERE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
if [ ! -f "$REF" ] && [ -f "$HERE/build/outputs.json" ]; then
  "$PY" "$HERE/redact.py" "$HERE/build/outputs.json" "$REF" --bucket "$BUCKET" --account "$ACCOUNT"
fi
cp "$HERE/build/manifest.json" "$HERE/results/$RUN_ID/build-manifest.json"
# P2's validator (examples/lambda-kleidiai/results/, PR #2) checks the file against lkai-result/1 once it has landed.
P2=$HERE/../results/publish_result.py
[ ! -f "$P2" ] || "$PY" "$P2" --validate-only "$HERE/results/$RUN_ID/results.json"
[ "$SUM" -eq 0 ] || { echo "results/$RUN_ID is not a valid baseline (see summarize.py above)" >&2; exit "$SUM"; }
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
