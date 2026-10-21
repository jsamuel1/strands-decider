#!/bin/bash
# E8 offline byte-identity and load-time check (NOT Lambda: a fixed host, for mechanism and correctness only).
#
#   examples/lambda-kleidiai/e8/hash_check.sh <P1 GGUF> [REPEATS] [OUT_DIR]
#
# In P1's pinned Lambda image, kaihash loads P1's GGUF and hashes every loaded weight (see kaihash.cpp) under:
#   p1      P1's own shipped libggml-cpu variants (from the P1 zip), switch unset
#   off     E8's patched variants, switch unset (the dormant path)
#   t1 t2 t3 t6   E8's patched variants, GGML_KLEIDIAI_REPACK_THREADS=1, 2, 3, 6
# Each configuration runs REPEATS times (default 3) in a new container, configurations interleaved and their order
# rotated every repeat. Each container is pinned with --cpuset-cpus to CPUSET (default 0-5: 6 CPUs, the Lambda
# function's vCPU count at 10,240 MB). Other work on the host is not controlled, so model_load_s is a fixed-host proxy,
# not Lambda latency. The GGUF is mmapped from local disk with a warm page cache (not a memfd filled from S3).
# Writes OUT_DIR/<repeat>-<config>.txt and OUT_DIR/host.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
B=${E8_BUILD:-$HERE/build}
GGUF=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
REPEATS=${2:-3}
OUT=$(mkdir -p "${3:-$B/hash}" && cd "${3:-$B/hash}" && pwd)
CPUSET=${CPUSET:-0-5}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$RECIPE/baseline/build.sh")
CONFIGS=(p1 off t1 t2 t3 t6)
{ uname -a; grep -m1 'CPU implementer' /proc/cpuinfo; grep -m1 'CPU part' /proc/cpuinfo; grep -m1 'CPU variant' /proc/cpuinfo;
  grep -m1 'CPU revision' /proc/cpuinfo; grep -m1 Features /proc/cpuinfo; nproc; echo "cpuset $CPUSET";
  sha256sum "$GGUF"; uptime; } > "$OUT/host.txt" 2>&1
for r in $(seq 0 $((REPEATS - 1))); do
  n=${#CONFIGS[@]}
  for j in $(seq 0 $((n - 1))); do
    c=${CONFIGS[$(((j + r) % n))]}
    case $c in
      p1)  DIR=/p1lib ENV=() ;;
      off) DIR=/e8lib ENV=() ;;
      t*)  DIR=/e8lib ENV=(-e "GGML_KLEIDIAI_REPACK_THREADS=${c#t}") ;;
    esac
    docker run --rm --platform linux/arm64 --cpuset-cpus "$CPUSET" --entrypoint bash "${ENV[@]}" \
      -v "$B/p1pkg/lib:/p1lib:ro" -v "$B/libs:/e8lib:ro" -v "$B/kaihash:/kaihash:ro" -v "$GGUF:/m.gguf:ro" \
      "$LAMBDA_IMAGE" -c "LD_LIBRARY_PATH=/p1lib /kaihash $DIR /m.gguf" > "$OUT/$r-$c.txt" 2> "$OUT/$r-$c.err"
    echo "$r $c $(tail -1 "$OUT/$r-$c.txt")"
  done
done
echo "$(uptime)" >> "$OUT/host.txt"
echo HASH_DONE "$OUT"
