#!/bin/bash
# E5 in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/e5/run.sh [RUN_ID]
#
# 1. baseline/build.sh                (P1's pinned weights + function.zip; skipped when P1's build/ already holds
#                                     the zip and GGUF P1's published run used, so the deployed code is that run's)
# 2. convert_no_vocab.py              (the one variable: the GGUF without llama.cpp's text vocabulary)
# 3. deploy_e5.py up                  (lkai-e5-base and lkai-e5-e5: same zip, same config, different MODEL_KEY)
# 4. baseline/bench.py --arms base,e5 (P1's alternating-order cold / restore / warm harness, unchanged)
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
EXPECTED_PY_IMAGE=python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b
[ "$PY_IMAGE" = "$EXPECTED_PY_IMAGE" ] || {
  echo "baseline/build.sh PY_IMAGE $PY_IMAGE is not E5's recorded converter image $EXPECTED_PY_IMAGE" >&2
  exit 1
}
# P1's pushed baseline HEAD that this branch starts from (feat/lambda-kleidiai-baseline on the fork).
BASELINE_COMMIT=${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}

PUB=$BASE/results/20261020T080835Z/build-manifest.json
have() {  # P1's build/ holds the zip and GGUF of P1's published run
  [ -f "$PB/function.zip" ] && [ -f "$PB/model/s2b-h512-Q8_0.gguf" ] && "$PY" - "$PUB" "$PB" <<'EOF'
import hashlib, json, sys
p, b = json.load(open(sys.argv[1])), sys.argv[2]
def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()
ok = sha(b + "/function.zip") == p["function_zip"]["sha256"] and sha(b + "/model/s2b-h512-Q8_0.gguf") == p["model"]["gguf_sha256"]
sys.exit(0 if ok else 1)
EOF
}
have || LKAI_BUILD=$PB "$BASE/build.sh"
mkdir -p "$B/model"
ln -sf "$PB/function.zip" "$B/function.zip"
ln -sf "$PB/model/s2b-h512-Q8_0.gguf" "$B/model/s2b-h512-Q8_0.gguf"
CONVERTED=$B/model/s2b-h512-Q8_0-novocab.gguf
CONVERT_RECORD=$B/model/convert.json
conversion_current() {
  [ -f "$CONVERTED" ] && [ -f "$CONVERT_RECORD" ] && "$PY" - "$CONVERTED" "$CONVERT_RECORD" <<'EOF'
import hashlib, json, os, sys
model, record = sys.argv[1:]
try:
    rec = json.load(open(record))
    h = hashlib.sha256()
    with open(model, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    ok = rec.get("converted_sha256") == h.hexdigest()
except (OSError, ValueError, TypeError):
    ok = False
sys.exit(0 if ok else 1)
EOF
}
if ! conversion_current; then
  rm -f "$CONVERTED" "$CONVERT_RECORD"
  docker run --rm --platform linux/arm64 -e LKAI_E5_CONVERTER_IMAGE="$PY_IMAGE" -v "$PB:/b:ro" \
    -v "$HERE:/e5:ro" -v "$B/model:/o" "$PY_IMAGE" \
    bash /e5/container_convert.sh /b/model/s2b-h512-Q8_0.gguf /o/s2b-h512-Q8_0-novocab.gguf /o/convert.json
fi
"$PY" "$HERE/manifest_e5.py" "$B" "$PB/manifest.json" "$BASE/results/20261020T080835Z/build-manifest.json"

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
