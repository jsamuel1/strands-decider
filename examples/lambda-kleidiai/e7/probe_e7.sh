#!/bin/bash
# E7 offline check on an arm64 host with Docker (NOT Lambda: a fixed host, for mechanism and correctness only).
#
#   examples/lambda-kleidiai/e7/probe_e7.sh [PAIRS] [OUT_DIR]
#
# Unzips e7/build/function.zip into P1's pinned Lambda image and runs probe_e7.py once per load, alternating the arms
# (ctl: LKAI_SKIP_TIED_OUTPUT unset; skip: =1) and which arm goes first in each pair. Each load is a new container,
# with 6 threads and the container limited to 6 CPUs (--cpus 6), the Lambda function's vCPU count. Every load
# copies P1's GGUF into a memfd as the function does. Writes OUT_DIR/<pair>-<arm>.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
BASE=$RECIPE/baseline
PB=${LKAI_BUILD:-$BASE/build}
B=${E7_BUILD:-$HERE/build}
PAIRS=${1:-6}
OUT=${2:-$B/probe}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
mkdir -p "$OUT"
for i in $(seq 0 $((PAIRS - 1))); do
  if [ $((i % 2)) -eq 0 ]; then ARMS="ctl skip"; else ARMS="skip ctl"; fi
  for arm in $ARMS; do
    if [ "$arm" = skip ]; then SKIP=(-e LKAI_SKIP_TIED_OUTPUT=1); else SKIP=(); fi
    docker run --rm --platform linux/arm64 --cpus 6 --entrypoint bash "${SKIP[@]}" -e THREADS=6 \
      -v "$B:/b:ro" -v "$PB/model:/m:ro" -v "$HERE:/e7:ro" -v "$BASE:/base:ro" -v "$OUT:/out" "$LAMBDA_IMAGE" -c "
set -euo pipefail
dnf install -y unzip >/dev/null 2>&1
mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /b/function.zip && chmod -R a+rX /tmp/pkg && cd /tmp
LD_LIBRARY_PATH=/tmp/pkg/lib python3.12 /e7/probe_e7.py /tmp/pkg /m/s2b-h512-Q8_0.gguf /base/requests.json /out/$i-$arm.json"
  done
done
echo PROBE_DONE "$OUT"
