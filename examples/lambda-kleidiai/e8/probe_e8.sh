#!/bin/bash
# E8 offline answers check (NOT Lambda: a fixed host, for correctness only).
#
#   examples/lambda-kleidiai/e8/probe_e8.sh <P1 GGUF> [PAIRS] [OUT_DIR]
#
# Unzips base.zip and off.zip into P1's pinned Lambda image and runs probe_e8.py once per load for the three arms
# (base; off with the switch unset; par = off with GGML_KLEIDIAI_REPACK_THREADS=6), rotating the order each pair.
# Each load is a new container pinned with --cpuset-cpus to CPUSET (default 0-5) with 6 engine threads.
# Writes OUT_DIR/<pair>-<arm>.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
BASE=$RECIPE/baseline
B=${E8_BUILD:-$HERE/build}
GGUF=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
PAIRS=${2:-2}
OUT=$(mkdir -p "${3:-$B/probe}" && cd "${3:-$B/probe}" && pwd)
CPUSET=${CPUSET:-0-5}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
ARMS=(base off par)
for i in $(seq 0 $((PAIRS - 1))); do
  for j in 0 1 2; do
    arm=${ARMS[$(((j + i) % 3))]}
    case $arm in
      base) Z=base ENV=() ;;
      off)  Z=off ENV=() ;;
      par)  Z=off ENV=(-e GGML_KLEIDIAI_REPACK_THREADS=6) ;;
    esac
    docker run --rm --platform linux/arm64 --cpuset-cpus "$CPUSET" --entrypoint bash "${ENV[@]}" -e THREADS=6 \
      -v "$B:/b:ro" -v "$GGUF:/m.gguf:ro" -v "$HERE:/e8:ro" -v "$BASE:/base:ro" -v "$OUT:/out" "$LAMBDA_IMAGE" -c "
set -euo pipefail
dnf install -y unzip >/dev/null 2>&1
mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /b/$Z.zip && chmod -R a+rX /tmp/pkg && cd /tmp
LD_LIBRARY_PATH=/tmp/pkg/lib python3.12 /e8/probe_e8.py /tmp/pkg /m.gguf /base/requests.json \
  /base/reference/outputs.json /out/$i-$arm.json"
  done
done
echo PROBE_DONE "$OUT"
