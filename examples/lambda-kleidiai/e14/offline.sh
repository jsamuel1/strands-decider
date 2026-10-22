#!/bin/bash
# E14 OFFLINE, NOT LAMBDA: the f32 CONCAT on an arm64 host. What it copies, how long it takes, and the one-change fix.
#
#   [E14_CPUS=0-5] examples/lambda-kleidiai/e14/offline.sh <P1 build dir> <out dir> [rounds=3] [passes=10]
#
# E14_CPUS picks the six host CPUs the container is pinned to (default 0-5). On a shared host pick idle ones: every
# run records /proc/loadavg before and after it (loadavg.txt), and a busy host is a stated limitation, not hidden.
#
# <P1 build dir> is P1's baseline/build (function.zip, model/s2b-h512-Q8_0.gguf, llama.cpp with patch_llama.py applied).
# 1. In P1's pinned Lambda build image (baseline/container_build.sh's flags), builds the ggml-cpu variants twice from
#    copies of that llama.cpp tree:  rows = + e14/patch_concat.py (the change), probe = + e14/patch_concat_probe.py.
#    Only libggml-cpu-*.so differ between the three packages; every other file is P1's unzipped function.zip.
# 2. In P1's pinned Lambda python3.12 image, six CPUs (E14_CPUS), 6 threads (as P3's graviton_profile.sh), runs offline_e14.py
#    interleaved per round: ctl (P1's libs), rows (the change), probe (P1 kernel + probe, LKAI_E14_PROBE=1).
#    Every run hashes every raw hidden-state row, so ctl vs rows is a bit-identity check, not a rounded-answer check.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
ROUNDS=${3:-3}
PASSES=${4:-10}
CPUS=${E14_CPUS:-0-5}
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=${E14_WORK:-$(mktemp -d)}
mkdir -p "$W"
rm -rf "$W/pkg" "$W/pkg-rows" "$W/pkg-probe"
(cd "$W" && unzip -q "$B/function.zip" -d pkg)
cp -a "$W/pkg" "$W/pkg-rows" && cp -a "$W/pkg" "$W/pkg-probe"
for v in rows probe; do
  if [ ! -s "$W/lib-$v/toolchain.txt" ]; then
    mkdir -p "$W/lib-$v"
    script=patch_concat.py; [ "$v" = probe ] && script=patch_concat_probe.py
    docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e14:ro" \
      -v "$W/lib-$v:/out" -e SCRIPT="$script" "$IMAGE" -c '
        set -euo pipefail
        dnf install -y gcc gcc-c++ cmake make binutils findutils python3 >/dev/null
        rm -rf /b && cp -a /llama /b && cd /b && python3 /e14/$SCRIPT /b
        COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
        cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
          -DGGML_CPU_KLEIDIAI=ON >/dev/null
        cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
        cp -P build/bin/libggml-cpu-*.so /out/
        gcc --version | head -1 > /out/toolchain.txt'
  fi
  cp "$W"/lib-$v/libggml-cpu-*.so "$W/pkg-$v/lib/"
done
sha256sum "$W"/pkg/lib/libggml-cpu-*.so "$W"/pkg-rows/lib/libggml-cpu-*.so "$W"/pkg-probe/lib/libggml-cpu-*.so \
  | sed "s#$W/##" > "$OUT/libs.sha256"
sha256sum "$B/function.zip" "$B/model/s2b-h512-Q8_0.gguf" | sed "s#$B/##" > "$OUT/inputs.sha256"
cat "$W/lib-rows/toolchain.txt" > "$OUT/toolchain.txt"
docker run --rm --platform linux/arm64 --entrypoint bash \
  -v "$HERE/offline_e14.py:/p/offline_e14.py:ro" -v "$HERE/../baseline/requests.json:/w/requests.json:ro" \
  -v "$W/pkg:/pkg:ro" -v "$W/pkg-rows:/pkg-rows:ro" -v "$W/pkg-probe:/pkg-probe:ro" \
  -v "$B/model/s2b-h512-Q8_0.gguf:/w/model.gguf:ro" -v "$OUT:/out" -e ROUNDS="$ROUNDS" -e PASSES="$PASSES" \
  --cpuset-cpus "$CPUS" "$IMAGE" -c '
    set -euo pipefail
    dnf install -y util-linux >/dev/null 2>&1
    { uname -a; grep -m4 -E "CPU (implementer|part|variant|revision)" /proc/cpuinfo; grep -m1 Features /proc/cpuinfo;
      nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "cpu.max unknown"; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "memory.max unknown";
      python3.12 --version; } > /out/host.txt 2>&1
    run() {  # name pkg [extra]
      local n=$1 p=$2; shift 2
      echo "$n before $(cat /proc/loadavg)" >> /out/loadavg.txt
      LD_LIBRARY_PATH=$p/lib PYTHONPATH=$p python3.12 /p/offline_e14.py --pkg $p --gguf /w/model.gguf \
        --threads 6 --requests /w/requests.json --passes $PASSES --label "$n" --out /out/$n.json "$@" 2>/out/$n.stderr
      echo "$n after $(cat /proc/loadavg)" >> /out/loadavg.txt
    }
    for r in $(seq 1 $ROUNDS); do
      if [ $((r % 2)) = 1 ]; then order="ctl rows probe"; else order="probe rows ctl"; fi
      for a in $order; do
        echo ">> round $r $a"
        case $a in
          ctl)   run ctl$r /pkg ;;
          rows)  run rows$r /pkg-rows ;;
          probe) LKAI_E14_PROBE=1 run probe$r /pkg-probe --probe /pkg-probe/lib/libggml-cpu-armv8.6_1.so ;;
        esac
      done
    done
    chmod -R a+rX /out
    echo E14_OFFLINE_DONE'
echo "E14_OFFLINE_DONE $OUT"
