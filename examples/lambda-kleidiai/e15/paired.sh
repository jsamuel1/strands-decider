#!/bin/bash
# E15 Phase 2 paired probe, OFFLINE, NOT LAMBDA. Candidates A and B against ctl, toggled per request in one process.
#
#   examples/lambda-kleidiai/e15/paired.sh <P1 build dir> <out dir> [passes=10]
#
# Builds libggml-cpu-*.so with e15/patch_candidates.py (both switches off by default) using P1's flags and image,
# and swaps them into a copy of the unzipped function.zip. It then runs, in P1's Lambda image:
#   bitcheck   offline_drive.py with P1's libraries and with the candidate libraries, switches off: answers must match
#   paired     paired_drive.py, arms ctl,A,B per request in rotating order
# E15_CPUS (default 0-5) picks the 6 CPUs. The DevSpace is shared with other cards; paired arms cancel slow drift.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
PASSES=${3:-10}
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=${E15_WORK:-$(mktemp -d)}
mkdir -p "$W"
rm -rf "$W/pkg" "$W/pkgc" && (cd "$W" && unzip -q "$B/function.zip" -d pkg) && cp -a "$W/pkg" "$W/pkgc"
LIBS="$W/cand-lib-$(sha256sum "$HERE/patch_candidates.py" | cut -c1-12)"
if [ ! -s "$LIBS/toolchain.txt" ]; then
  mkdir -p "$LIBS"
  docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e15:ro" \
    -v "$LIBS:/out" "$IMAGE" -c '
      set -euo pipefail
      dnf install -y gcc gcc-c++ cmake make binutils findutils python3 >/dev/null
      rm -rf /b && cp -a /llama /b && cd /b && python3 /e15/patch_candidates.py /b
      COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
      cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CPU_KLEIDIAI=ON >/dev/null
      cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
      cp -P build/bin/libggml-cpu-*.so /out/
      nm -D /out/libggml-cpu-armv8.6_1.so > /tmp/nm.txt && grep -q lkai_e15_set_a /tmp/nm.txt && grep -q lkai_e15_set_b /tmp/nm.txt
      gcc --version | head -1 > /out/toolchain.txt'
fi
cp "$LIBS"/libggml-cpu-*.so "$W/pkgc/lib/"
cp "$LIBS/toolchain.txt" "$OUT/"
sha256sum "$W"/pkg/lib/libggml-cpu-*.so "$W"/pkgc/lib/libggml-cpu-*.so "$HERE/patch_candidates.py" > "$OUT/libs.sha256"
cp "$HERE/../profiling/offline_drive.py" "$HERE/paired_drive.py" "$W/"
docker run --rm --platform linux/arm64 --entrypoint bash -e PASSES="$PASSES" -e CPUS="${E15_CPUS:-0-5}" \
  -e GDN_ROWS="${LKAI_GDN_ROWS:-16}" -e HOST_LABEL="$(grep -m1 'CPU part' /proc/cpuinfo | awk '{print $NF}') $(hostname -s)" \
  -v "$W/offline_drive.py:/p/offline_drive.py:ro" -v "$W/paired_drive.py:/p/paired_drive.py:ro" \
  -v "$HERE/../baseline/requests.json:/w/requests.json:ro" -v "$W/pkg:/pkg:ro" -v "$W/pkgc:/pkgc:ro" \
  -v "$B/model/s2b-h512-Q8_0.gguf:/w/model.gguf:ro" -v "$OUT:/out" "$IMAGE" -c '
    set -euo pipefail
    dnf install -y util-linux >/dev/null 2>&1
    export OPENBLAS_NUM_THREADS=1
    { uname -a; grep -m4 -E "CPU (implementer|part|variant|revision)" /proc/cpuinfo; grep -m1 Features /proc/cpuinfo;
      nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; python3.12 --version; echo "cpus $CPUS"; cat /proc/loadavg; } \
      > /out/host.txt 2>&1
    for n in p1 cand; do
      p=/pkg; [ $n = cand ] && p=/pkgc
      LD_LIBRARY_PATH=$p/lib PYTHONPATH=$p taskset -c $CPUS python3.12 /p/offline_drive.py --pkg $p --gguf /w/model.gguf \
        --threads 6 --requests /w/requests.json --passes 1 --host-label "$HOST_LABEL bitcheck $n" \
        --out /out/bitcheck-$n.json >/dev/null 2>/out/bitcheck-$n.stderr
    done
    LD_LIBRARY_PATH=/pkgc/lib PYTHONPATH=/pkgc taskset -c $CPUS python3.12 /p/paired_drive.py --pkg /pkgc \
      --gguf /w/model.gguf --variant /pkgc/lib/libggml-cpu-armv8.6_1.so --requests /w/requests.json \
      --passes $PASSES --threads 6 --arms ctl,A,B --gdn-rows $GDN_ROWS \
      --host-label "$HOST_LABEL (offline, Lambda image, CPUs $CPUS) paired" --out /out/paired.json \
      > /out/paired.stdout 2>/out/paired.stderr
    chmod -R a+rX /out
    echo E15_PAIRED_DONE'
echo "E15_PAIRED_DONE $OUT"
