#!/bin/bash
# E1 in one command (any host with AWS credentials; no build: it reuses P1's function.zip and GGUF):
#
#   AWS_PROFILE=<profile> SOURCE=s3://<p1-bucket> examples/lambda-kleidiai/e1-s3-express/run_e1.sh [RUN_ID]
#
# deploy_e1.py up -> baseline/bench.py (4 arms) -> summarize_e1.py -> redaction -> P2 validation.
# SOURCE is read only. Or FROM_BUILD=<baseline/build> to upload a local P1 build. Knobs (P1's pinned defaults):
# REGION=us-east-2 AZ_ID=use2-az1 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-e1 TEARDOWN=versions|all|none.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$HERE/../baseline
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
RAW=$HERE/build/runs/$RUN_ID
PY=${PYTHON:-python3}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }
SRC_ARGS=()
if [ -n "${SOURCE:-}" ]; then SRC_ARGS=(--source "$SOURCE"); elif [ -n "${FROM_BUILD:-}" ]; then
  SRC_ARGS=(--from-build "$FROM_BUILD"); else echo "set SOURCE=s3://<p1-bucket> or FROM_BUILD=<dir>"; exit 1; fi

"$PY" "$HERE/deploy_e1.py" up --region "$REGION" --az-id "${AZ_ID:-use2-az1}" --prefix "${PREFIX:-lkai-e1}" \
  --build "$HERE/build" "${SRC_ARGS[@]}"
cleanup() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy_e1.py" teardown --region "$REGION" --build "$HERE/build" ;;
    all) "$PY" "$HERE/deploy_e1.py" teardown --all --region "$REGION" --build "$HERE/build" ;;
    none) ;;
    *) echo "TEARDOWN must be versions, all, or none" >&2; return 2 ;;
  esac
}
trap cleanup EXIT
"$PY" "$BASE/bench.py" --build "$HERE/build" --arms gp,xz,gpv,xzv --region "$REGION" \
  --rounds "${ROUNDS:-5}" --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
cp "$HERE/build/deployment.json" "$RAW/deployment.json"
"$PY" "$HERE/summarize_e1.py" "$RAW" --reference "$BASE/reference/outputs.json" --meta "$HERE/meta.json" \
  --baseline-commit "${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
read -r GP XZ < <("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['bucket'], d['xz_bucket'])" \
  "$HERE/build/deployment.json")
TMP=$HERE/build/redact-$RUN_ID
"$PY" "$BASE/redact.py" "$RAW" "$TMP" --bucket "$XZ" --account "$ACCOUNT"
"$PY" "$BASE/redact.py" "$TMP" "$HERE/results/$RUN_ID" --bucket "$GP"
rm -rf "$TMP"
# deployment.json is private state (role, subnet, SG ids); the redacted copy keeps only what the result needs.
"$PY" - "$HERE/results/$RUN_ID/deployment.json" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
keep = {k: d.get(k) for k in ("prefix", "region", "az_id", "zip_sha256", "runtime_version_arn", "model_key",
                               "staged_sha256", "runtime_sdk")}
keep["vpc"] = {k: (d.get("vpc") or {}).get(k) for k in ("subnet_cidr",)} | {"endpoints": sorted((d.get("vpc") or {}).get("endpoints", {}))}
keep["functions"] = {a: {k: f[k] for k in ("name", "code_sha256_b64", "bucket_kind", "vpc")} for a, f in d["functions"].items()}
json.dump(keep, open(p, "w"), indent=1)
EOF
P2=$HERE/../results/publish_result.py
[ ! -f "$P2" ] || "$PY" "$P2" --validate-only "$HERE/results/$RUN_ID/results.json"
cleanup
trap - EXIT
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
