#!/bin/bash
# Build E15's deployed candidate libraries: llama.cpp f74e2fa6a + llama/patch_llama.py (P1's tree) + e15/patch_a.py,
# compiled in P1's pinned build image with P1's flags (baseline/container_build.sh). Only libggml-cpu-*.so are built.
#
#   examples/lambda-kleidiai/e15/build_cand.sh <P1 build dir> <out dir>
#
# <P1 build dir>/llama.cpp must be P1's patched tree; the script refuses one that already carries any E15 patch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$1" && pwd)
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
if grep -q "lkai E15" "$B/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c" "$B/llama.cpp/ggml/src/ggml-cpu/ops.cpp"; then
  echo "P1's llama.cpp tree already carries an E15 patch"; exit 1
fi
docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e15:ro" \
  -v "$OUT:/out" "$IMAGE" -c '
    set -euo pipefail
    dnf install -y gcc gcc-c++ cmake make binutils findutils python3 >/dev/null
    rm -rf /b && cp -a /llama /b && cd /b && python3 /e15/patch_a.py /b
    COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
    cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DGGML_CPU_KLEIDIAI=ON >/dev/null
    cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
    cp -L build/bin/libggml-cpu-*.so /out/
    nm -D /out/libggml-cpu-armv8.6_1.so > /tmp/nm.txt
    grep -q lkai_e15_set_a /tmp/nm.txt   # not in a pipe: SIGPIPE under pipefail
    gcc --version | head -1 > /out/toolchain.txt
    sha256sum /e15/patch_a.py /e15/patch_candidates.py > /out/patch_a.sha256'
ls -l "$OUT"
