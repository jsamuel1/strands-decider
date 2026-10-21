#!/bin/bash
# E9 build, on an arm64 Linux host with Docker:
#   build/function.zip                       P1's package with E9's loader patch (patch_e9.py) in libllama/libggml-cpu-*
#   build/model/s2b-h512-Q8_0-e9kr8.gguf     P1's GGUF with every KleidiAI-bound Q8_0 weight prepacked (pack_e9.py)
#   build/model/pack.json                    per-tensor payload sizes and slot hashes, tool versions
#
#   LKAI_BUILD=<P1 build/> examples/lambda-kleidiai/e9/build.sh
#
# Reuses P1's build (weights, assets, patched llama.cpp f74e2fa6a); only libllama/libggml-cpu-* and the GGUF change.
# The zip is built by P1's own baseline/container_build.sh in P1's pinned Lambda image, with /llama pointing at a copy
# of P1's patched tree plus patch_e9.py. Rerunning skips finished steps (delete build/ to start over).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
REPO=$(cd "$RECIPE/../.." && pwd)
BASE=$RECIPE/baseline
PB=${LKAI_BUILD:-$BASE/build}
B=${E9_BUILD:-$HERE/build}
KR=${E9_KR:-8}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
PY_IMAGE=$(sed -n 's/^PY_IMAGE=//p' "$BASE/build.sh")
[ "$(uname -m)" = aarch64 ] || { echo "build.sh needs an arm64 host (got $(uname -m))"; exit 1; }
[ -f "$PB/model/s2b-h512-Q8_0.gguf" ] && [ -d "$PB/llama.cpp" ] || { echo "no P1 build at $PB (run baseline/build.sh)"; exit 1; }
mkdir -p "$B/model"

LLAMA_REV=$(sed -n 's/^LLAMA_REV=//p' "$BASE/build.sh")
if [ "$(git -C "$B/llama.cpp" rev-parse HEAD 2>/dev/null)" != "$LLAMA_REV" ] || [ ! -f "$B/llama.cpp/.e9-patched" ]; then
  rm -rf "$B/llama.cpp" && mkdir -p "$B/llama.cpp"
  git -C "$B/llama.cpp" init -q
  git -C "$B/llama.cpp" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_REV"
  git -C "$B/llama.cpp" checkout -q FETCH_HEAD
  python3 "$RECIPE/llama/patch_llama.py" "$B/llama.cpp"
  python3 "$HERE/patch_e9.py" "$B/llama.cpp"
  touch "$B/llama.cpp/.e9-patched"
fi
git -C "$B/llama.cpp" diff > "$B/llama-e9.diff"

ZIP_INPUTS="examples/lambda-kleidiai/shim examples/lambda-kleidiai/snapstart examples/lambda-kleidiai/llama
  examples/lambda-kleidiai/baseline/lkai_entry.py examples/lambda-kleidiai/baseline/phases.py
  examples/lambda-kleidiai/baseline/container_build.sh examples/lambda/app src/strands_decider
  examples/lambda-kleidiai/e9/patch_e9.py"
# shellcheck disable=SC2086
ZIP_KEY=$( (cd "$REPO" && git ls-files -s -- $ZIP_INPUTS && git diff HEAD -- $ZIP_INPUTS) | sha256sum | cut -c1-16)
if [ -f "$B/function.zip" ] && [ "$(cat "$B/.zip-inputs" 2>/dev/null)" = "$ZIP_KEY" ]; then
  echo "   function.zip already built from these inputs ($ZIP_KEY)"
else
  docker run --rm --platform linux/arm64 --entrypoint bash \
    -v "$B/llama.cpp:/llama:ro" -v "$RECIPE:/recipe:ro" -v "$REPO/examples/lambda:/lambda:ro" \
    -v "$REPO/src/strands_decider:/pkgsrc:ro" -v "$PB/model/assets:/assets:ro" -v "$B:/out" \
    "$LAMBDA_IMAGE" /recipe/baseline/container_build.sh
  printf '%s\n' "$ZIP_KEY" > "$B/.zip-inputs"
  git -C "$REPO" rev-parse HEAD > "$B/.zip-repo-commit"
fi

G=$B/model/s2b-h512-Q8_0-e9kr$KR.gguf
if [ ! -f "$G" ] || [ ! -f "$B/model/pack-kr$KR.json" ]; then
  rm -f "$G" "$B/model/pack-kr$KR.json"
  docker run --rm --platform linux/arm64 -e E9_PACK_IMAGE="$PY_IMAGE" -e KR="$KR" \
    -v "$PB:/b:ro" -v "$HERE:/e9:ro" -v "$B/model:/o" "$PY_IMAGE" \
    bash /e9/container_pack.sh /b/model/s2b-h512-Q8_0.gguf "/o/s2b-h512-Q8_0-e9kr$KR.gguf" "/o/pack-kr$KR.json"
fi
echo BUILD_E9_DONE
