#!/bin/bash
# E5 in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/e5/run.sh [RUN_ID]
#
# 1. baseline/build.sh                (P1's pinned weights + function.zip; reuses an existing build/)
# 2. convert_no_vocab.py              (the one variable: the GGUF without llama.cpp's text vocabulary)
# 3. deploy_e5.py up                  (lkai-e5-base and lkai-e5-e5: same zip, same config, different MODEL_KEY)
# 4. baseline/bench.py --arms base,e5 (P1's interleaved cold / restore / warm harness, unchanged)
# 5. baseline/summarize.py            (lkai-result/1, correctness against P1's reference/outputs.json)
# 6. baseline/redact.py               (results/<RUN_ID>/ is what gets committed)
# Env knobs as baseline/run.sh: REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-e5
# TEARDOWN=versions|all|none. LKAI_BUILD (default baseline/build) is where P1's build lives.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(cd "$HERE/../baseline" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
PREFIX=${PREFIX:-lkai-e5}
PB=${LKAI_BUILD:-$BASE/build}
B=$HERE/build
RAW=$B/runs/$RUN_ID
PY=${PYTHON:-python3}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY (pip install boto3)"; exit 1; }
PY_IMAGE=$(sed -n 's/^PY_IMAGE=//p' "$BASE/build.sh")
# P1's pushed baseline HEAD that this branch starts from (feat/lambda-kleidiai-baseline on the fork).
BASELINE_COMMIT=${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}

LKAI_BUILD=$PB "$BASE/build.sh"
mkdir -p "$B/model"
ln -sf "$PB/function.zip" "$B/function.zip"
ln -sf "$PB/manifest.json" "$B/manifest.json"
ln -sf "$PB/model/s2b-h512-Q8_0.gguf" "$B/model/s2b-h512-Q8_0.gguf"
if [ ! -f "$B/model/s2b-h512-Q8_0-novocab.gguf" ]; then
  docker run --rm --platform linux/arm64 -v "$PB:/b:ro" -v "$HERE:/e5:ro" -v "$B/model:/o" "$PY_IMAGE" \
    bash /e5/container_convert.sh /b/model/s2b-h512-Q8_0.gguf /o/s2b-h512-Q8_0-novocab.gguf /o/convert.json
fi
"$PY" "$HERE/manifest_e5.py" "$B"

"$PY" "$HERE/deploy_e5.py" up --build "$B" --region "$REGION" --prefix "$PREFIX"
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy_e5.py" teardown --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    all) "$PY" "$HERE/deploy_e5.py" teardown --all --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    none) ;;
    *) echo "TEARDOWN must be versions, all, or none" >&2; return 2 ;;
  esac
}
trap cleanup_versions EXIT
"$PY" "$BASE/bench.py" --build "$B" --arms base,e5 --region "$REGION" --rounds "${ROUNDS:-5}" \
  --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
"$PY" "$HERE/summarize_e5.py" "$RAW" "$B" --reference "$BASE/reference/outputs.json" --meta "$HERE/meta.json" \
  --experiment-commit "$(git -C "$HERE" rev-parse HEAD)" --baseline-commit "$BASELINE_COMMIT"
cp "$B/e5-manifest.json" "$RAW/e5-manifest.json"
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$B/deployment.json")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
"$PY" "$BASE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
P2=$HERE/../results/publish_result.py
[ ! -f "$P2" ] || "$PY" "$P2" --validate-only "$HERE/results/$RUN_ID/results.json"
cleanup_versions
trap - EXIT
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
