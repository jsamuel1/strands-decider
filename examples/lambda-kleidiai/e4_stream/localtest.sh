#!/bin/bash
# Run e4_stream/localtest.py for each arm in the pinned Lambda image, on the build host. No AWS.
#   examples/lambda-kleidiai/e4_stream/localtest.sh [OUT_DIR]     (MIB_S=80 THREADS=6 by default: P1's Lambda numbers)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
B=${E4_BUILD:-$HERE/build}
OUT=${1:-$B/localtest}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$RECIPE/baseline/build.sh")
rm -rf "$B/task" && mkdir -p "$B/task" "$OUT" && (cd "$B/task" && unzip -q ../function.zip)
for arm in base stream streamfile base stream streamfile; do
  case $arm in
    base) env=(-e E4_STREAM=0) ;;
    stream) env=(-e E4_STREAM=1 -e STREAM_ORDER=arrival) ;;
    streamfile) env=(-e E4_STREAM=1 -e STREAM_ORDER=file) ;;
  esac
  n=$(ls "$OUT" | grep -c "^$arm-" || true)
  docker run --rm --platform linux/arm64 --entrypoint python3.12 "${env[@]}" \
    -e E4_FAKE_MIB_S="${MIB_S:-80}" -e THREADS="${THREADS:-6}" -e PYTHONPATH=/task -e LD_LIBRARY_PATH=/task/lib \
    -e E4_REQUESTS=/recipe/baseline/requests.json -v "$B/task:/task:ro" -v "$B/model:/model:ro" \
    -v "$RECIPE:/recipe:ro" -v "$OUT:/out" -w /task "$LAMBDA_IMAGE" \
    /recipe/e4_stream/localtest.py /model/s2b-h512-Q8_0.gguf "/out/$arm-$n.json" 2>&1 | grep -v "^20[0-9][0-9]-" | tail -3
done
python3 - "$OUT" "$RECIPE/baseline/reference/outputs.json" <<'EOF'
import json, os, sys
out, ref = sys.argv[1], json.load(open(sys.argv[2]))["outputs"]
for f in sorted(os.listdir(out)):
    r = json.load(open(os.path.join(out, f)))
    cpu = r["init"].get("cpu_part") or r["counters"].get("cpu_part")
    same = {rid: a == ref.get(rid, {}).get("0xd40") for rid, a in r["answers"].items()}
    i = r["init"]
    print(f, "dl+load %.2f" % i.get("download_and_load", -1), "dl %.2f" % i.get("weight_download", -1),
          "load %.2f" % i.get("model_load_mmap", -1), "blocked %.2f" % i.get("stream_load_blocked", 0),
          "warmup %.3f" % i.get("warmup_inference", -1), "bit-identical vs P1 0xd40:", sum(same.values()), "/", len(same))
EOF
