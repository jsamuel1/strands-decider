#!/bin/bash
# Build x86_64 packages of P1 and of each rung's llama.cpp patch, on an x86_64 Linux host with Docker. FIXED HOST.
#
#   general/x86_build.sh <p1.zip> <out dir>
#
# P1's recipe is arm64-only: baseline/build.sh refuses a non-aarch64 host, and llama.cpp f74e2fa6a's CMake stops with
# "GGML_CPU_KLEIDIAI requires a ... AArch64/arm64 target" when GGML_CPU_KLEIDIAI=ON on x86. So the x86_64 package is
# P1's baseline/container_build.sh with exactly two forced changes, applied to a copy and asserted:
#   1. -DGGML_CPU_KLEIDIAI=ON -> OFF (the architecture forbids it; on x86 Q8_0 has no repack path at this revision,
#      ggml-cpu/repack.cpp only repacks Q8_0 for NEON i8mm/dotprod and RISC-V V);
#   2. the llama-quantize step is dropped (its armv8.2 -march does not exist on x86; no arm needs a quantizer).
# Everything else is P1's: the pinned Lambda python3.12 image (its linux/amd64 manifest), GGML_BACKEND_DL,
# GGML_CPU_ALL_VARIANTS (the x86 variant set), P1's llama/patch_llama.py, decshim, PR #77's handler and the same
# pip pins. Each rung's package adds that rung's own patch from its pinned commit:
#   p1   P1's patches               e8  + e8/patch_parallel_repack.py @ bef9c89
#   e9   + e9/patch_e9.py @ 5dce29d e7  + e7/patch_skip_output.py @ 0af73df
# The x86 pip install needs network access (amd64 wheels); the arm64 package's wheels are arm64-only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
REPO=$(cd "$RECIPE/../.." && pwd)
P1_ZIP=$(readlink -f "$1")
mkdir -p "$2" && O=$(cd "$2" && pwd)
IMG=$(sed -n 's/^LAMBDA_IMAGE=//p' "$RECIPE/baseline/build.sh")
# The amd64 entry of that manifest list, by its own digest (docker refuses to pull a list digest for a second
# platform once one is local: "cannot overwrite digest"). x86_run.sh uses the same image.
IMG_AMD64=${IMG%@*}@sha256:586cea700429caaab11d50f7b96258a77bd4899d253e9559acfb647fe383f970
IMG_LIST=$(docker manifest inspect "$IMG" | python3 -c "import json,sys; print(' '.join(m['digest'] for m in json.load(sys.stdin)['manifests'] if m['platform']['architecture'] == 'amd64'))")
[ "$IMG_LIST" = "${IMG_AMD64#*@}" ] || { echo "FATAL: $IMG's amd64 entry is $IMG_LIST"; exit 1; }
IMG=$IMG_AMD64
LLAMA_REV=$(sed -n 's/^LLAMA_REV=//p' "$RECIPE/baseline/build.sh")
[ "$(uname -m)" = x86_64 ] || { echo "x86_build.sh needs an x86_64 host (got $(uname -m))"; exit 1; }
RUNGS=${RUNGS:-p1 e8 e9 e7}

# P1's container_build.sh with the two forced x86 changes; refuse if either anchor moved.
CB=$O/container_build_x86.sh
sed -e 's/-DGGML_CPU_KLEIDIAI=ON/-DGGML_CPU_KLEIDIAI=OFF/' \
    -e '/^echo "=== llama-quantize/,/^cp -L \/usr\/lib64\/libgomp.so.1 .*\/out\/quant-lib\//d' \
    "$RECIPE/baseline/container_build.sh" > "$CB"
grep -q -- '-DGGML_CPU_KLEIDIAI=OFF' "$CB" && ! grep -q 'target llama-quantize' "$CB" \
  || { echo "FATAL: container_build.sh anchors moved"; exit 1; }
diff "$RECIPE/baseline/container_build.sh" "$CB" > "$O/container_build_x86.diff" || true

rm -rf "$O/assets" && mkdir -p "$O/assets"
(cd "$O" && unzip -q -j "$P1_ZIP" 'assets/*' -d assets)

if [ ! -d "$O/llama.cpp.pristine" ]; then
  mkdir -p "$O/llama.cpp.pristine" && git -C "$O/llama.cpp.pristine" init -q
  git -C "$O/llama.cpp.pristine" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_REV"
  git -C "$O/llama.cpp.pristine" checkout -q FETCH_HEAD
fi
[ "$(git -C "$O/llama.cpp.pristine" rev-parse HEAD)" = "$LLAMA_REV" ] || { echo "FATAL: llama.cpp is not $LLAMA_REV"; exit 1; }

for r in $RUNGS; do
  echo ">> x86_64 package: $r"
  T=$O/src-$r
  rm -rf "$T" && cp -a "$O/llama.cpp.pristine" "$T" && git -C "$T" checkout -q -- .
  git -C "$REPO" show 70eab7b:examples/lambda-kleidiai/llama/patch_llama.py > "$O/patch_llama.py"
  python3 "$O/patch_llama.py" "$T"
  case $r in
    e8) git -C "$REPO" show bef9c89:examples/lambda-kleidiai/e8/patch_parallel_repack.py > "$O/patch-$r.py" ;;
    e9) git -C "$REPO" show 5dce29d:examples/lambda-kleidiai/e9/patch_e9.py > "$O/patch-$r.py" ;;
    e7) git -C "$REPO" show 0af73df:examples/lambda-kleidiai/e7/patch_skip_output.py > "$O/patch-$r.py" ;;
    p1) rm -f "$O/patch-$r.py" ;;
  esac
  [ -f "$O/patch-$r.py" ] && python3 "$O/patch-$r.py" "$T"
  git -C "$T" diff > "$O/llama-$r.diff"
  rm -rf "$O/out-$r" && mkdir -p "$O/out-$r"
  docker run --rm --platform linux/amd64 --entrypoint bash \
    -v "$T:/llama:ro" -v "$RECIPE:/recipe:ro" -v "$REPO/examples/lambda:/lambda:ro" -v "$CB:/cb.sh:ro" \
    -v "$REPO/src/strands_decider:/pkgsrc:ro" -v "$O/assets:/assets:ro" -v "$O/out-$r:/out" \
    "$IMG" /cb.sh > "$O/build-$r.log" 2>&1 || { tail -30 "$O/build-$r.log"; exit 1; }
  rm -rf "$O/pkg-$r" && mkdir -p "$O/pkg-$r" && (cd "$O/pkg-$r" && unzip -q "$O/out-$r/function.zip")
  chmod -R a+rX "$O/pkg-$r"
  sha256sum "$O/out-$r/function.zip" "$O/pkg-$r"/lib/*.so* | sed "s#$O/##" > "$O/sha-$r.txt"
  tail -2 "$O/build-$r.log"
done
echo X86_BUILD_DONE
