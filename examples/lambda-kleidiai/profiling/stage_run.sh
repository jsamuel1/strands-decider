#!/bin/bash
# Assemble the redacted, committed copy of a P3 run and its results.json. Not Lambda-specific; standard tools only.
#
#   profiling/stage_run.sh <run-dir> <dest-dir> <bucket> <account> <experiment-commit> <prof-manifest.json> [<meta.json>]
#
# Extra summarize_prof.py flags (e.g. --superseded-by <run-id>) go in $SUMMARIZE_EXTRA. <meta.json> defaults to
# profiling/meta.json; a superseded run keeps its own per-run meta copy.
#
# 1. summarize_prof.py over <run-dir> (writes <run-dir>/results.json)
# 2. redact.py <run-dir> -> <dest-dir> (bucket and account replaced; fails if either survives)
# 3. gzip the large JSONL files and point results.json's artifact links at the .gz names
# 4. copy the run-independent evidence beside it: the matrix, the offline summary, the README
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$1 DEST=$2 BUCKET=$3 ACCOUNT=$4 COMMIT=$5 PM=$6 META=${7:-$HERE/meta.json}
python3 "$HERE/summarize_prof.py" "$RUN" --reference "$HERE/../baseline/reference/outputs.json" \
  --baseline-commit aac59d45555ebded9357c23f6f8e4f939edf25d7 --meta "$META" \
  --applicability "$HERE/applicability.json" --offline "$HERE/results/offline" \
  --probe "$HERE/results/lambda-probe/q1.json" --experiment-commit "$COMMIT" --prof-manifest "$PM" ${SUMMARIZE_EXTRA:-}
python3 "$HERE/../baseline/redact.py" "$RUN" "$DEST" --bucket "$BUCKET" --account "$ACCOUNT"
[ -f "$DEST/xray.jsonl" ] && gzip -9n "$DEST/xray.jsonl"
gzip -9n "$DEST/invocations.jsonl"
python3 - "$DEST/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
r["artifacts"]["invocations"] = "invocations.jsonl.gz"
json.dump(r, open(sys.argv[1], "w"), indent=1)
EOF
cp "$HERE/applicability.json" "$HERE/results/offline/offline-summary.json" "$HERE/README.md" "$DEST/"
cp "$META" "$DEST/meta.json"
cp "$PM" "$DEST/prof-manifest.json"
if grep -rlF -e "$BUCKET" -e "$ACCOUNT" "$DEST" || zgrep -lF -e "$BUCKET" -e "$ACCOUNT" "$DEST"/*.gz "$DEST"/logs/*.gz; then
  echo "redaction failed"; exit 1
fi
echo "STAGED $DEST"
