#!/bin/bash
# E9 on Lambda, in one command, on an arm64 Linux host with Docker and AWS credentials for the target account:
#
#   AWS_PROFILE=<profile> examples/lambda-kleidiai/e9/run.sh [RUN_ID]
#
# 1. e9/build.sh                       (E9's zip and prepacked GGUF from P1's build)
# 2. deploy_e9.py up                   (lkai-f1ee-base: P1's zip + GGUF; lkai-f1ee-e9: E9's zip + GGUF)
# 3. baseline/bench.py --arms base,e9  (P1's alternating-order cold / restore / warm harness; fresh invoke client per
#                                      wave, I1's fix, taken from 5764b9e; per-arm zip check)
# 4. summarize_e9.py                   (lkai-result/1, per-arm correctness, e9_vs_base, load paths, host variability)
# 5. baseline/redact.py                (results/<RUN_ID>/ is what gets committed)
# Knobs as baseline/run.sh: REGION=us-east-2 ROUNDS=5 CONCURRENCY=10 WARM=10 PREFIX=lkai-f1ee
# TEARDOWN=versions|all|none. LKAI_BUILD (default baseline/build) is P1's build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(cd "$HERE/../baseline" && pwd)
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
REGION=${REGION:-us-east-2}
PREFIX=${PREFIX:-lkai-f1ee}
PB=${LKAI_BUILD:-$BASE/build}
B=$HERE/build
RAW=$B/runs/$RUN_ID
PY=${PYTHON:-python3}
BASELINE_COMMIT=${BASELINE_COMMIT:-aac59d45555ebded9357c23f6f8e4f939edf25d7}
"$PY" -c "import boto3" 2>/dev/null || { echo "needs boto3 for $PY"; exit 1; }

PUB=$BASE/results/20261020T080835Z/build-manifest.json
"$PY" - "$PUB" "$PB" <<'EOF' || { echo "P1's build at $PB is not P1's published zip + GGUF"; exit 1; }
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
LKAI_BUILD=$PB "$HERE/build.sh"
ln -sf "$PB/function.zip" "$B/base-function.zip"
ln -sf "$PB/model/s2b-h512-Q8_0.gguf" "$B/model/s2b-h512-Q8_0.gguf"
cp "$PUB" "$B/manifest.json"  # P1's published manifest: the base arm is provably the code and weights P1 measured

"$PY" "$HERE/deploy_e9.py" up --build "$B" --region "$REGION" --prefix "$PREFIX"
cleanup_versions() {
  case "${TEARDOWN:-versions}" in
    versions) "$PY" "$HERE/deploy_e9.py" teardown --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    all) "$PY" "$HERE/deploy_e9.py" teardown --all --build "$B" --region "$REGION" --prefix "$PREFIX" ;;
    none) ;;
    *) echo "TEARDOWN must be versions, all, or none" >&2; return 2 ;;
  esac
}
trap cleanup_versions EXIT
"$PY" "$BASE/bench.py" --build "$B" --arms base,e9 --region "$REGION" --rounds "${ROUNDS:-5}" \
  --concurrency "${CONCURRENCY:-10}" --warm "${WARM:-10}" --out "$RAW"
"$PY" "$HERE/summarize_e9.py" "$RAW" "$B" --reference "$BASE/reference/outputs.json" --meta "$HERE/meta.json" \
  --experiment-commit "$(git -C "$HERE" rev-parse HEAD)" --baseline-commit "$BASELINE_COMMIT"
cp "$B/model/pack-kr8.json" "$RAW/pack-kr8.json"
BUCKET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$B/deployment.json")
ACCOUNT=$("$PY" -c "import boto3; print(boto3.client('sts').get_caller_identity()['Account'])")
"$PY" "$BASE/redact.py" "$RAW" "$HERE/results/$RUN_ID" --bucket "$BUCKET" --account "$ACCOUNT"
cleanup_versions
trap - EXIT
echo "RUN_DONE $HERE/results/$RUN_ID/results.json"
