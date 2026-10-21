#!/bin/bash
# E12 offline check on an arm64 host with Docker (NOT Lambda): three arms over the same request orders.
#   p1   P1's function.zip as deployed (its own libdecshim.so)
#   dor  P1's zip with E12's libdecshim.so, DEC_CLEAR_DATA unset (the dormant switch: must equal p1)
#   e12  the same, DEC_CLEAR_DATA=0 (metadata-only clear)
#
#   examples/lambda-kleidiai/e12/offline_run.sh <p1-function.zip> <gguf> <e12 libdecshim.so> <out dir>
#
# Every arm runs in the pinned Lambda python3.12 image with P1's engine environment (DEC_KV=f32, DEC_UBATCH=512,
# N_SEQ=17), 6 threads pinned to CPUs 0-5 (Lambda's 10,240 MB functions report 6 CPUs), arms alternating p1, dor,
# e12, e12, dor, p1 so neither arm always runs first.
set -euo pipefail
ZIP=$1 GGUF=$2 SHIM=$3 OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
ORDERS=${ORDERS:-6} REPEATS=${REPEATS:-3} SEED=${SEED:-12}
mkdir -p "$OUT"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/p1" "$W/e12"
(cd "$W/p1" && unzip -q "$ZIP")
cp -a "$W/p1/." "$W/e12/"
cp "$SHIM" "$W/e12/lib/libdecshim.so"
cp "$HERE/offline_check.py" "$W/"
cp "$HERE/requests.json" "$W/e12-requests.json"
cp "$HERE/../baseline/requests.json" "$W/p1-requests.json"
sha256sum "$ZIP" "$GGUF" "$SHIM" "$W/p1/lib/libdecshim.so" > "$OUT/inputs.sha256"
run() {  # arm pkg clear_env tag
  docker run --rm --platform linux/arm64 --cpuset-cpus 0-5 --entrypoint python3.12 \
    -e THREADS=6 -e OPENBLAS_NUM_THREADS=1 -e DEC_KV=f32 -e DEC_UBATCH=512 -e N_SEQ=17 \
    -e DEC_GGML_DIR="/task/lib" -e LD_LIBRARY_PATH=/task/lib $3 \
    -v "$W/$2:/task:ro" -v "$GGUF:/model.gguf:ro" -v "$W:/w:ro" -v "$OUT:/out" -w /task "$IMAGE" \
    /w/offline_check.py --task /task --model /model.gguf --requests /w/p1-requests.json /w/e12-requests.json \
    --orders "$ORDERS" --repeats "$REPEATS" --seed "$SEED" --out "/out/$1-$4.jsonl" > "$OUT/$1-$4.log" 2>&1
  tail -1 "$OUT/$1-$4.log"
}
i=0
for arm in p1 dor e12 e12 dor p1; do
  case $arm in
    p1) run p1 p1 "" "$i" ;;
    dor) run dor e12 "" "$i" ;;
    e12) run e12 e12 "-e DEC_CLEAR_DATA=0" "$i" ;;
  esac
  i=$((i + 1))
done
{ uname -a; grep -m1 "CPU part" /proc/cpuinfo; nproc; docker --version; } > "$OUT/host.txt"
echo OFFLINE_RUN_DONE
