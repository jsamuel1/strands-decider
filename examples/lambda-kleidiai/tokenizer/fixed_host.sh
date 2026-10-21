#!/bin/bash
# T0 fixed-host driver: the deployed tokenizer, in P1's pinned Lambda image, on an arm64 host. NOT Lambda evidence.
#
#   examples/lambda-kleidiai/tokenizer/fixed_host.sh <P1 function.zip> <out-dir> <phase> [run-id]
#     phase golden   corpus.jsonl, golden.jsonl.gz, parity-check.json
#     phase bench    bench-<run-id>.json (needs the committed corpus.jsonl)
#     phase profile  perf stat / perf record / folded stacks
#
# The zip must be P1's deployed one (sha256 26eb2651...): the tokenizer, its .so and tokenizer.json are its bytes.
# The work is pinned to 6 CPUs (taskset -c ${T0_CPUS:-0-5} inside the container), the vCPUs Lambda gives a 10,240 MB
# function. On a shared host choose 6 CPUs no other job is pinned to (T0_CPUS); the host-*.txt record lists every
# other process's CPU affinity and the load average so contention is visible in the results.
# docker --cpus/--cpuset-cpus are not used: on the DevSpace's threaded cgroup v2 they refuse to start the container
# ("cannot enter cgroupv2 ... in threaded mode"). perf needs PERFMON and perf_event_paranoid <= 2 on the host.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ZIP=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
PHASE=$3
RUN=${4:-r1}
LAMBDA_IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
EXPECT_ZIP=26eb2651e8915f8ad09eae899feff583a448de6b8d2bc4943bccabb66cae189b
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
[ "$(sha256sum "$ZIP" | cut -d' ' -f1)" = "$EXPECT_ZIP" ] || [ "${T0_ANY_ZIP:-}" = 1 ] \
  || { echo "function.zip is not P1's deployed zip ($EXPECT_ZIP); T0_ANY_ZIP=1 to override"; exit 1; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
(cd "$W" && unzip -q "$ZIP" -d pkg)
cp "$HERE/../baseline/requests.json" "$W/requests.json"
chmod -R a+rX "$W"
{
  echo "label: FIXED HOST, NOT LAMBDA"
  echo "host: $(hostname -s)"; uname -a
  grep -m5 -E 'CPU (implementer|part|variant|revision)|Features' /proc/cpuinfo
  echo "nproc: $(nproc)"; grep -E 'MemTotal' /proc/meminfo; cat /proc/loadavg
  echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "zip_sha256: $(sha256sum "$ZIP" | cut -d' ' -f1)"
  echo "t0_cpus: ${T0_CPUS:-0-5}"
  echo "other_python_affinity:"
  for p in $(pgrep -f 'python3.12 /' || true); do
    echo "  $p cpus=$(taskset -pc "$p" 2>/dev/null | cut -d: -f2 | tr -d ' ') $(ps -o args= -p "$p" | cut -c1-80)"
  done
  echo "image: $LAMBDA_IMAGE"
  echo "date_utc: $(date -u +%FT%TZ)"
} > "$OUT/host-$PHASE-$RUN.txt"
docker run --rm --platform linux/arm64 --cap-add PERFMON --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --entrypoint bash -e PHASE="$PHASE" -e RUN="$RUN" \
  -e HOST_LABEL="$(grep -m1 'CPU part' /proc/cpuinfo | awk '{print $NF}') $(hostname -s)" \
  -e ROUNDS="${ROUNDS:-5}" -e INIT_REPS="${INIT_REPS:-10}" -e ITERS="${ITERS:-200}" -e T0_CPUS="${T0_CPUS:-0-5}" \
  -v "$HERE:/t:ro" -v "$W/requests.json:/req/requests.json:ro" -v "$W/pkg:/pkg:ro" -v "$OUT:/out" \
  "$LAMBDA_IMAGE" /t/tok_container.sh
echo "FIXED_HOST_DONE $PHASE $OUT"
