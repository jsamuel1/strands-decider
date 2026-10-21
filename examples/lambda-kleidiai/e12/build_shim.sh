#!/bin/bash
# Build E12's libdecshim.so on an arm64 host with Docker, exactly as P1's container_build.sh builds the shim: gcc -O2
# in P1's pinned Lambda python3.12 image, against P1's own llama.cpp f74e2fa6a (+ patch_llama.py) headers and the
# libllama/libggml shipped in P1's function.zip.
#
#   examples/lambda-kleidiai/e12/build_shim.sh <P1 build dir (llama.cpp/, function.zip)> <out dir>
#
# Writes <out>/libdecshim.so and <out>/shim-build.json (inputs and toolchain, read by build_zip.py --shim-build).
set -euo pipefail
B=$1 OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
LLAMA_REV=f74e2fa6a31cc33ab4ee4ae516641cfe62df3924
[ "$(git -C "$B/llama.cpp" rev-parse HEAD)" = "$LLAMA_REV" ] || { echo "llama.cpp is not $LLAMA_REV"; exit 1; }
mkdir -p "$OUT"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/pkg" && (cd "$W/pkg" && unzip -q "$B/function.zip" 'lib/*')
cp "$HERE/../shim/decshim.c" "$W/"
docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/b:ro" -v "$W:/w" "$IMAGE" -c '
set -euo pipefail
dnf install -y gcc binutils >/dev/null 2>&1
gcc -O2 -shared -fPIC -o /w/libdecshim.so /w/decshim.c -I/b/include -I/b/ggml/include -L/w/pkg/lib -lllama -lggml \
  -Wl,-rpath,"\$ORIGIN"
nm -D /w/libdecshim.so | grep -q " T dec_clear_data" || { echo "FATAL: no dec_clear_data"; exit 1; }
gcc --version | head -1 > /w/gcc.txt
chmod a+r /w/libdecshim.so /w/gcc.txt'
cp "$W/libdecshim.so" "$OUT/"
python3 - "$OUT" "$W" "$B" "$IMAGE" "$LLAMA_REV" <<'EOF'
import hashlib, json, sys
out, w, b, image, rev = sys.argv[1:]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"image": image, "gcc": open(f"{w}/gcc.txt").read().strip(), "llama_cpp": rev,
           "command": "gcc -O2 -shared -fPIC -o libdecshim.so decshim.c -I/b/include -I/b/ggml/include "
                      "-L<P1 zip lib/> -lllama -lggml -Wl,-rpath,$ORIGIN",
           "decshim_c_sha256": h(f"{w}/decshim.c"), "p1_function_zip_sha256": h(f"{b}/function.zip"),
           "libdecshim_so_sha256": h(f"{out}/libdecshim.so")}, open(f"{out}/shim-build.json", "w"), indent=1)
EOF
echo "libdecshim.so $(sha256sum "$OUT/libdecshim.so" | cut -c1-16)"
