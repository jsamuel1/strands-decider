#!/bin/bash
# E15 Phase 2 offline probe, OFFLINE, NOT LAMBDA: candidates A and B against P1's libraries on one arm64 host.
#
#   examples/lambda-kleidiai/e15/probe.sh <P1 build dir> <out dir> [rounds=3]
#
# Builds libggml-cpu-*.so with e15/patch_candidates.py (both candidates, each off unless its switch is set) using
# P1's flags and image, swaps them into a copy of the unzipped function.zip, then runs offline_drive.py (10 passes)
# per arm, interleaved per round in a rotating order:
#   ctl   P1's libraries
#   off   candidate libraries, no switch (must match ctl: the patch is inert when off)
#   A     LKAI_E15_A=1   llamafile F32 sgemm in the SVE/i8mm variants
#   B     LKAI_E15_B=1   row-blocked GATED_DELTA_NET (LKAI_GDN_ROWS, default 16)
# Timings come from offline_drive's own per-request timers; the node profiler is not built in.
# E15_CPUS (default 0-5) picks the 6 CPUs; the DevSpace is shared with other cards' offline work, which pins 0-5.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
ROUNDS=${3:-3}
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=${E15_WORK:-$(mktemp -d)}
mkdir -p "$W"
rm -rf "$W/pkg" "$W/pkgc" && (cd "$W" && unzip -q "$B/function.zip" -d pkg) && cp -a "$W/pkg" "$W/pkgc"
if [ ! -s "$W/cand-lib/toolchain.txt" ]; then
  mkdir -p "$W/cand-lib"
  docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e15:ro" \
    -v "$W/cand-lib:/out" "$IMAGE" -c '
      set -euo pipefail
      dnf install -y gcc gcc-c++ cmake make binutils findutils python3 >/dev/null
      rm -rf /b && cp -a /llama /b && cd /b && python3 /e15/patch_candidates.py /b
      COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
      cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CPU_KLEIDIAI=ON >/dev/null
      cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
      cp -P build/bin/libggml-cpu-*.so /out/
      nm /out/libggml-cpu-armv8.6_1.so > /tmp/nm.txt && grep -q lkai_gdn_rowblock /tmp/nm.txt
      gcc --version | head -1 > /out/toolchain.txt'
fi
cp "$W"/cand-lib/libggml-cpu-*.so "$W/pkgc/lib/"
sha256sum "$W"/pkg/lib/libggml-cpu-*.so "$W"/pkgc/lib/libggml-cpu-*.so > "$OUT/libs.sha256"
cp "$HERE/../profiling/offline_drive.py" "$W/offline_drive.py"
docker run --rm --platform linux/arm64 --entrypoint bash -e ROUNDS="$ROUNDS" -e GDN_ROWS="${LKAI_GDN_ROWS:-16}" \
  -e CPUS="${E15_CPUS:-0-5}" \
  -e HOST_LABEL="$(grep -m1 'CPU part' /proc/cpuinfo | awk '{print $NF}') $(hostname -s)" \
  -v "$W/offline_drive.py:/p/offline_drive.py:ro" -v "$HERE/../baseline/requests.json:/w/requests.json:ro" \
  -v "$W/pkg:/pkg:ro" -v "$W/pkgc:/pkgc:ro" -v "$B/model/s2b-h512-Q8_0.gguf:/w/model.gguf:ro" -v "$OUT:/out" \
  "$IMAGE" -c '
    set -euo pipefail
    dnf install -y util-linux >/dev/null 2>&1
    export OPENBLAS_NUM_THREADS=1
    { uname -a; grep -m4 -E "CPU (implementer|part|variant|revision)" /proc/cpuinfo; grep -m1 Features /proc/cpuinfo;
      nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; python3.12 --version; } > /out/host.txt 2>&1
    run() {  # name pkg
      local n=$1 p=$2
      # Other work on the host shares memory bandwidth and L3 even on other cores: record what else ran.
      { date -u +%FT%TZ; cat /proc/loadavg; } > /out/$n.load
      LD_LIBRARY_PATH=$p/lib PYTHONPATH=$p taskset -c $CPUS python3.12 /p/offline_drive.py --pkg $p --gguf /w/model.gguf \
        --threads 6 --requests /w/requests.json --passes 10 --host-label "$HOST_LABEL (offline, Lambda image, CPUs $CPUS) $n" \
        --out /out/$n.json >/dev/null 2>/out/$n.stderr
      { date -u +%FT%TZ; cat /proc/loadavg; } >> /out/$n.load
    }
    arms=(ctl off A B)
    for r in $(seq 1 $ROUNDS); do
      for i in 0 1 2 3; do
        a=${arms[$(( (i + r - 1) % 4 ))]}
        echo ">> round $r $a"
        case $a in
          ctl) run ctl$r /pkg ;;
          off) run off$r /pkgc ;;
          A)   LKAI_E15_A=1 run A$r /pkgc ;;
          B)   LKAI_E15_B=1 LKAI_GDN_ROWS=$GDN_ROWS run B$r /pkgc ;;
        esac
      done
    done
    chmod -R a+rX /out
    echo E15_PROBE_DONE'
echo "E15_PROBE_DONE $OUT"
