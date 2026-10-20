#!/bin/bash
# Offline profile of the deployed baseline on an arm64 (Graviton) host: NOT Lambda evidence.
#
#   examples/lambda-kleidiai/profiling/graviton_profile.sh <P1 function.zip> <s2b-h512-Q8_0.gguf> <out-dir>
#
# Runs in P1's pinned Lambda python3.12 image (same AL2023 userland, same glibc, same libraries as Lambda) on the
# host's kernel, with the deployed function.zip unzipped and the same GGUF. Lambda gives a function 6 vCPUs at
# 10,240 MB, so the container is limited to 6 CPUs and the engine to 6 threads. Every output says OFFLINE.
#   perf-stat    perf stat -e task-clock,cycles,instructions,cache-misses,page-faults over one full run
#   perf-record  perf record -F 999 -g (frame-pointer call graph), then perf report; perf script keeps per-sample times
#   sampler      lkaisamp (the in-Lambda sampler) on the same run, to show it agrees with perf
#   cprofile     cProfile of the same run
#   pyspy        py-spy record --native (allowed here, denied in Lambda)
# perf needs perf_event_paranoid <= 2 for user-space sampling of one's own processes, which the DevSpace allows.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ZIP=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
GGUF=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
OUT=$(mkdir -p "$3" && cd "$3" && pwd)
LAMBDA_IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
# unzip, not python -m zipfile: the zip stores lib/*.so.N as symlinks (zip -y), which zipfile writes out as text files.
(cd "$W" && unzip -q "$ZIP" -d pkg)
python3 -m pip download -q --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12 \
  py-spy==0.4.2 -d "$W/whl"
(cd "$W/whl" && python3 -m zipfile -e py_spy-*.whl x && cp x/py_spy-0.4.2.data/scripts/py-spy "$W/py-spy")
chmod -R a+rX "$W" && chmod 755 "$W/py-spy"
docker run --rm --platform linux/arm64 --cap-add SYS_PTRACE --cap-add PERFMON --security-opt seccomp=unconfined \
  --entrypoint bash -e HOST_LABEL="$(grep -m1 'CPU part' /proc/cpuinfo | awk '{print $NF}') $(hostname -s)" \
  -v "$HERE:/p:ro" -v "$HERE/../baseline/requests.json:/w/requests.json:ro" -v "$W/pkg:/pkg:ro" \
  -v "$GGUF:/w/model.gguf:ro" -v "$W/py-spy:/w/py-spy:ro" -v "$OUT:/out" \
  "$LAMBDA_IMAGE" /p/graviton_container.sh
echo "GRAVITON_PROFILE_DONE $OUT"
