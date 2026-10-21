#!/bin/bash
# E9 offline check on an arm64 host with Docker. Runs probe_e9.py in P1's pinned Lambda image:
#
#   examples/lambda-kleidiai/e9/offline.sh <P1 build/> <E9 build/> <out dir> [timing pairs]
#
# 1. parity: the E9 zip loading P1's GGUF (runtime repack) and the E9 GGUF (adopt) both dump every packed payload;
#    the hashes must match. Repeated with E9_DIAG_NO_I8MM=1 (dot-product kernel, kr=4: the convert path).
# 2. inertness: the E9 zip on P1's GGUF must answer exactly as P1's zip does (the patch changes nothing without a
#    trailer).
# 3. timing: P1 zip + P1 GGUF vs E9 zip + E9 GGUF, alternating A/B then B/A per pair, 3 loads per process.
#    Fixed-host numbers: they isolate the load's CPU work and are never Lambda latency.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(cd "$HERE/../baseline" && pwd)
PB=$(cd "$1" && pwd) EB=$(cd "$2" && pwd)
mkdir -p "$3" && OUT=$(cd "$3" && pwd)
PAIRS=${4:-3}
IMG=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
for z in p1 e9; do
  rm -rf "$OUT/pkg-$z" && mkdir -p "$OUT/pkg-$z"
done
unzip -q "$PB/function.zip" -d "$OUT/pkg-p1"  # unzip, not zipfile: the zip stores libllama.so.0 etc. as symlinks
unzip -q "$EB/function.zip" -d "$OUT/pkg-e9"
chmod -R a+rX "$OUT"
mkdir -p "$OUT/dump" && chmod 777 "$OUT/dump" "$OUT"
probe() {  # probe <name> <pkg> <gguf path under /m> <repeat> [extra docker args]
  local name=$1 pkg=$2 g=$3 rep=$4; shift 4
  docker run --rm --platform linux/arm64 --entrypoint python3.12 "$@" \
    -e THREADS=6 -e PYTHONPATH=/pkg -e LD_LIBRARY_PATH=/pkg/lib \
    -v "$OUT/pkg-$pkg:/pkg:ro" -v "$PB/model:/m/p1:ro" -v "$EB/model:/m/e9:ro" -v "$HERE:/e9:ro" \
    -v "$BASE:/base:ro" -v "$OUT:/out" -w /tmp "$IMG" \
    /e9/probe_e9.py /pkg "$g" /base/requests.json "/out/$name.json" --repeat "$rep" 2>"$OUT/$name.stderr" | tail -2
  grep -h '^E9KAI' "$OUT/$name.stderr" || true
}
D=(-e E9_DUMP_DIR=/out/dump)
N=(-e E9_DIAG_NO_I8MM=1)
probe parity-repack-kr8 e9 /m/p1/s2b-h512-Q8_0.gguf 1 "${D[@]}"
probe parity-adopt-kr8 e9 /m/e9/s2b-h512-Q8_0-e9kr8.gguf 1 "${D[@]}"
probe parity-repack-kr4 e9 /m/p1/s2b-h512-Q8_0.gguf 1 "${D[@]}" "${N[@]}"
probe parity-convert-kr4 e9 /m/e9/s2b-h512-Q8_0-e9kr8.gguf 1 "${D[@]}" "${N[@]}"
probe inert-p1zip p1 /m/p1/s2b-h512-Q8_0.gguf 1
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) = 1 ]; then
    probe "time-$i-a-base" p1 /m/p1/s2b-h512-Q8_0.gguf 3
    probe "time-$i-b-e9" e9 /m/e9/s2b-h512-Q8_0-e9kr8.gguf 3
  else
    probe "time-$i-b-e9" e9 /m/e9/s2b-h512-Q8_0-e9kr8.gguf 3
    probe "time-$i-a-base" p1 /m/p1/s2b-h512-Q8_0.gguf 3
  fi
done
{ uname -r; nproc; grep -m1 'CPU part' /proc/cpuinfo; grep -m1 Features /proc/cpuinfo; uptime; } > "$OUT/host.txt"
echo OFFLINE_DONE
