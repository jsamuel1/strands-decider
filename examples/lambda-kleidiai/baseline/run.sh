#!/bin/bash
# The baseline in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh [RUN_ID]
#
# build.sh (pinned weights + zip) -> deploy.py up -> bench.py -> summarize.py -> redact.py. AWS_PROFILE is required:
# no step uses the default credential chain. Before anything is built or spent, run.sh resolves the profile's account
# (sts get-caller-identity) and refuses a working tree whose build or harness inputs differ from HEAD (buildcache.py
# dirty: tracked edits, untracked and deleted files), since the commit it records could not reproduce them. Every AWS
# step then uses that profile and is checked against that account. Env knobs (defaults are the
# published baseline's): REGION=us-east-2 MEMORY=10240 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-p1
# TEARDOWN=versions|all|none. ROUNDS must be >= 2, CONCURRENCY >= 1, WARM >= the number of fixed requests (5), so
# every request in requests.json is answered on both arms (bench.py rejects others). MEMORY (MB,
# 128..10240) sets both functions' memory, is checked on the deployed functions before the benchmark starts, and is
# recorded in conditions.json and results.json. REGION must have a pinned price table in summarize.PRICES_BY_REGION
# (only us-east-2 today); deploy.py refuses any other before creating anything.
# Raw output stays in the git-ignored build/runs/<RUN_ID>/; results/<RUN_ID>/ is the same with the bucket name and
# account id replaced, which is what gets committed. The first run's answers become reference/outputs.json, but only
# if every invocation succeeded with an answer and matched its expected start class, every fixed request was answered
# on both arms (inst and rep), and the build is clean (summarize.py exits 3 otherwise: the result is still written and
# redacted as status "failed", the reference is not, and run.sh exits non-zero).
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
if [ -z "${AWS_PROFILE:-}" ]; then
  echo "AWS_PROFILE is required: run.sh never uses the default credential chain" >&2
  exit 2
fi
export AWS_PROFILE
# Inputs HEAD cannot reproduce (stage inputs and the harness): refuse now, not after an hour of building.
if ! DIRTY=$("$PY" "$HERE/buildcache.py" dirty --repo "$(cd "$HERE/../../.." && pwd)" --check); then
  echo "refusing: these build/harness inputs differ from HEAD; commit or remove them first:" >&2
  echo "$DIRTY" >&2
  exit 2
fi
# Which account this profile reaches, before any build or spend. deploy.py and bench.py refuse any other.
ACCOUNT=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text) || {
  echo "could not resolve AWS_PROFILE=$AWS_PROFILE (sts get-caller-identity)" >&2
  exit 2
}
[[ "$ACCOUNT" =~ ^[0-9]{12}$ ]] || { echo "unexpected account id from sts: '$ACCOUNT'" >&2; exit 2; }
echo "AWS_PROFILE=$AWS_PROFILE account ...${ACCOUNT: -4} region $REGION"
AWS=(--profile "$AWS_PROFILE" --expect-account "$ACCOUNT")

"$HERE/build.sh"
"$PY" "$HERE/deploy.py" up --region "$REGION" --prefix "${PREFIX:-lkai-p1}" --memory "$MEMORY" "${AWS[@]}"
# Published versions incur SnapStart cache charges even if a later benchmark or summary step fails.
# Arm cleanup immediately after deploy; explicit TEARDOWN=none is the only opt-out. deploy.py teardown attempts every
# version and exits non-zero naming each one it could not delete.
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy.py" teardown --region "$REGION" "${AWS[@]}" ;;
    all) "$PY" "$HERE/deploy.py" teardown --all --region "$REGION" "${AWS[@]}" ;;
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
"$PY" "$HERE/bench.py" --region "$REGION" --profile "$AWS_PROFILE" --memory "$MEMORY" --rounds "${ROUNDS:-5}" \
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
# ACCOUNT is the preflight's: the same profile deployed and benchmarked, so the same id is redacted.
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
