#!/bin/bash
# E15 Phase 1, OFFLINE, NOT LAMBDA: per-node wall time of the warm forward on an arm64 host.
#
#   examples/lambda-kleidiai/e15/nodeprof.sh <P1 build dir> <out dir>
#
# <P1 build dir> is P1's baseline/build (function.zip, model/s2b-h512-Q8_0.gguf, llama.cpp with patch_llama.py applied).
# 1. In P1's pinned Lambda build image, copies that llama.cpp tree, applies e15/patch_nodeprof.py and builds the
#    ggml-cpu variants with P1's flags (baseline/container_build.sh). Only libggml-cpu-*.so are replaced, in a copy of
#    the unzipped function.zip; every other file is P1's.
# 2. In P1's pinned Lambda python3.12 image, 6 CPUs, 6 threads (as graviton_profile.sh), runs offline_drive.py:
#      ctl     P1's libraries                                   10 passes  answers + phase timers
#      off     patched libraries, LKAI_NODEPROF=0               10 passes  the patch is inert when off
#      on      patched libraries, LKAI_NODEPROF=1               10 passes  per-node table (nodeprof.tsv)
#      perf    P1's libraries, perf record -g --call-graph dwarf 3 passes  who calls ggml_vec_dot_f32
#    ctl/off/on/ctl2/off2/on2 are interleaved so host drift shows up as ctl vs ctl2.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
BUILD_IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=${E15_WORK:-$(mktemp -d)}
mkdir -p "$W"
rm -rf "$W/pkg" "$W/pkgnp" && (cd "$W" && unzip -q "$B/function.zip" -d pkg) && cp -a "$W/pkg" "$W/pkgnp"
if [ ! -s "$W/np-lib/toolchain.txt" ]; then
  mkdir -p "$W/np-lib"
  docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e15:ro" \
    -v "$W/np-lib:/out" "$BUILD_IMAGE" -c '
      set -euo pipefail
      dnf install -y gcc gcc-c++ cmake make binutils findutils python3 >/dev/null
      rm -rf /b && cp -a /llama /b && cd /b && python3 /e15/patch_nodeprof.py /b
      COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
      cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CPU_KLEIDIAI=ON >/dev/null
      cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
      cp -P build/bin/libggml-cpu-*.so /out/
      nm -D /out/libggml-cpu-armv8.6_1.so > /tmp/nm.txt && grep -q lkai_nodeprof_dump /tmp/nm.txt  # no grep -q in a pipe: SIGPIPE under pipefail
      gcc --version | head -1 > /out/toolchain.txt'
fi
cp "$W"/np-lib/libggml-cpu-*.so "$W/pkgnp/lib/"
sha256sum "$W"/pkg/lib/libggml-cpu-*.so "$W"/pkgnp/lib/libggml-cpu-*.so > "$OUT/libs.sha256"
cp "$HERE/../profiling/offline_drive.py" "$W/offline_drive.py"
docker run --rm --platform linux/arm64 --cap-add SYS_PTRACE --cap-add PERFMON --security-opt seccomp=unconfined \
  --entrypoint bash -e HOST_LABEL="$(grep -m1 'CPU part' /proc/cpuinfo | awk '{print $NF}') $(hostname -s)" \
  -v "$W/offline_drive.py:/p/offline_drive.py:ro" -v "$HERE/../baseline/requests.json:/w/requests.json:ro" \
  -v "$W/pkg:/pkg:ro" -v "$W/pkgnp:/pkgnp:ro" -v "$B/model/s2b-h512-Q8_0.gguf:/w/model.gguf:ro" -v "$OUT:/out" \
  "$BUILD_IMAGE" -c '
    set -euo pipefail
    dnf install -y perf util-linux >/dev/null 2>&1
    export OPENBLAS_NUM_THREADS=1
    { uname -a; grep -m4 -E "CPU (implementer|part|variant|revision)" /proc/cpuinfo; grep -m1 Features /proc/cpuinfo;
      nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; perf --version; python3.12 --version; } > /out/host.txt 2>&1
    run() {  # name pkg passes [extra args]
      local n=$1 p=$2 k=$3; shift 3
      LD_LIBRARY_PATH=$p/lib PYTHONPATH=$p taskset -c 0-5 python3.12 /p/offline_drive.py --pkg $p --gguf /w/model.gguf \
        --threads 6 --requests /w/requests.json --passes $k --host-label "$HOST_LABEL (offline, Lambda image, 6 CPUs) $n" \
        --out /out/$n.json "$@" >/dev/null 2>/out/$n.stderr
    }
    for r in 1 2; do
      echo ">> round $r"
      run ctl$r /pkg 10
      LKAI_NODEPROF=0 run off$r /pkgnp 10
      LKAI_NODEPROF=1 run on$r /pkgnp 10 --nodeprof /pkgnp/lib/libggml-cpu-armv8.6_1.so --nodeprof-out /out/nodeprof$r.tsv
    done
    echo ">> perf (dwarf call graph, P1 libraries)"
    LD_LIBRARY_PATH=/pkg/lib PYTHONPATH=/pkg perf record -F 499 --call-graph dwarf,16384 -o /tmp/perf.data -- \
      taskset -c 0-5 python3.12 /p/offline_drive.py --pkg /pkg --gguf /w/model.gguf --threads 6 \
      --requests /w/requests.json --passes 3 --host-label "$HOST_LABEL perf" --out /out/perf.json >/dev/null 2>/out/perf.stderr
    perf report -i /tmp/perf.data --no-children --sort sym --stdio -G --symbol-filter=ggml_vec_dot_f32 \
      --percent-limit 0.2 2>/dev/null | head -400 > /out/perf-vec_dot_f32-callers.txt || true
    perf report -i /tmp/perf.data --no-children --sort dso,sym --stdio -g none --percent-limit 0.3 2>/dev/null \
      | head -80 > /out/perf-self.txt || true
    chmod -R a+rX /out
    echo E15_NODEPROF_DONE'
echo "E15_NODEPROF_DONE $OUT"
