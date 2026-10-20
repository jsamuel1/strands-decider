#!/bin/bash
# Build every baseline artifact from pinned sources on an arm64 Linux host with Docker:
#   build/model/s2b-h512-Q8_0.gguf   the deployed weights (H512 Hadamard fold, Q8_0)
#   build/function.zip                the Lambda package (handler lkai_entry.handler)
#   build/manifest.json               every pin and sha256, read by deploy.py and copied into results
#
#   examples/lambda-kleidiai/baseline/build.sh            # about 1 h on 16 Graviton3 vCPUs, ~60 GB disk
#
# Rerunning skips finished steps (delete build/ to start over). Needs ~32 GB RAM for the fp32 merge.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
REPO=$(cd "$RECIPE/../.." && pwd)
B=${LKAI_BUILD:-$HERE/build}
CKPT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
CKPT_REV=216d1ccb46a9e9b036eb9187b248db98480f77d4
LLAMA_REV=f74e2fa6a31cc33ab4ee4ae516641cfe62df3924
# Immutable image digests from the published baseline build. Tags here would make a later one-command run
# silently use a different compiler, runtime image or weight toolchain.
LAMBDA_IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
PY_IMAGE=python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b
[ "$(uname -m)" = aarch64 ] || { echo "build.sh needs an arm64 host (got $(uname -m))"; exit 1; }
mkdir -p "$B/hf" "$B/model"

if [ "$(git -C "$B/llama.cpp" rev-parse HEAD 2>/dev/null)" != "$LLAMA_REV" ]; then
  rm -rf "$B/llama.cpp" && mkdir -p "$B/llama.cpp"
  git -C "$B/llama.cpp" init -q
  git -C "$B/llama.cpp" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_REV"
  git -C "$B/llama.cpp" checkout -q FETCH_HEAD
  python3 "$RECIPE/llama/patch_llama.py" "$B/llama.cpp"
fi

docker pull -q --platform linux/arm64 "$LAMBDA_IMAGE" >/dev/null
docker pull -q --platform linux/arm64 "$PY_IMAGE" >/dev/null

echo ">> weights: merge, H512 fold, f32 GGUF, assets"
docker run --rm --platform linux/arm64 -e CKPT=$CKPT -e CKPT_REV=$CKPT_REV \
  -v "$REPO:/repo:ro" -v "$B/llama.cpp:/llama:ro" -v "$B:/work" -v "$B/hf:/hf" \
  "$PY_IMAGE" bash /repo/examples/lambda-kleidiai/baseline/weights.sh

echo ">> libraries, shim and function.zip in the Lambda python3.12 image"
# The zip's inputs: a harness-only change (bench.py, summarize.py, docs) keeps the same zip and deployed code.
ZIP_INPUTS="examples/lambda-kleidiai/shim examples/lambda-kleidiai/snapstart examples/lambda-kleidiai/llama
  examples/lambda-kleidiai/baseline/lkai_entry.py examples/lambda-kleidiai/baseline/phases.py
  examples/lambda-kleidiai/baseline/container_build.sh examples/lambda/app src/strands_decider"
# shellcheck disable=SC2086
ZIP_KEY=$( (cd "$REPO" && git ls-files -s -- $ZIP_INPUTS && git diff HEAD -- $ZIP_INPUTS) | sha256sum | cut -c1-16)
if [ -f "$B/function.zip" ] && [ "$(cat "$B/.zip-inputs" 2>/dev/null)" = "$ZIP_KEY" ]; then
  echo "   function.zip already built from these inputs ($ZIP_KEY)"
else
docker run --rm --platform linux/arm64 --entrypoint bash \
  -v "$B/llama.cpp:/llama:ro" -v "$RECIPE:/recipe:ro" -v "$REPO/examples/lambda:/lambda:ro" \
  -v "$REPO/src/strands_decider:/pkgsrc:ro" -v "$B/model/assets:/assets:ro" -v "$B:/out" \
  "$LAMBDA_IMAGE" /recipe/baseline/container_build.sh
printf '%s\n' "$ZIP_KEY" > "$B/.zip-inputs"
git -C "$REPO" rev-parse HEAD > "$B/.zip-repo-commit"
fi

G=$B/model/s2b-h512-Q8_0.gguf
if [ ! -f "$G" ]; then
  echo ">> quantize Q8_0"
  docker run --rm --platform linux/arm64 --entrypoint bash -v "$B:/out" "$LAMBDA_IMAGE" -c \
    "LD_LIBRARY_PATH=/out/quant-lib /out/bin/llama-quantize /out/weights/s2b-h512-f32.gguf /out/model/s2b-h512-Q8_0.gguf.tmp Q8_0 8 | tail -2 && mv /out/model/s2b-h512-Q8_0.gguf.tmp /out/model/s2b-h512-Q8_0.gguf && chmod a+r /out/model/s2b-h512-Q8_0.gguf"
fi
# Earlier runs of weights.sh may predate its chmod: the manifest step below reads these as the host user.
docker run --rm --platform linux/arm64 --entrypoint bash -v "$B:/out" "$LAMBDA_IMAGE" -c \
  "chmod -R a+rX /out/model /out/merge_report.json /out/function.zip /out/toolchain.txt"

echo ">> manifest"
python3 - "$B" "$REPO" <<EOF
import hashlib, json, os, subprocess, sys
b, repo = sys.argv[1:3]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()
def img(ref):
    return subprocess.run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", ref],
                          capture_output=True, text=True).stdout.strip()
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
g = os.path.join(b, "model", "s2b-h512-Q8_0.gguf")
z = os.path.join(b, "function.zip")
m = {
    "schema": "lkai-build-manifest/1",
    "repo_commit": git("rev-parse", "HEAD"),
    "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "model": {"id": "$CKPT", "revision": "$CKPT_REV", "format": "GGUF Q8_0",
              "transform": "LoRA merged fp32 (weights/merge_export.py), H512 Hadamard fold (weights/fold_hadamard.py --block 512)",
              "merge_report": json.load(open(os.path.join(b, "merge_report.json"))),
              "gguf_bytes": os.path.getsize(g), "gguf_sha256": sha(g)},
    "llama_cpp": {"commit": "$LLAMA_REV", "patches": "llama/patch_llama.py (prefill,gdn,kskip,sme)",
                  "flags": "GGML_NATIVE=OFF GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON GGML_CPU_KLEIDIAI=ON"},
    "function_zip": {"bytes": os.path.getsize(z), "sha256": sha(z), "handler": "lkai_entry.handler",
                     "source_commit": open(os.path.join(b, ".zip-repo-commit")).read().strip(),
                     "inputs_sha256": open(os.path.join(b, ".zip-inputs")).read().strip()},
    "assets_sha256": {f: sha(os.path.join(b, "model", "assets", f)) for f in sorted(os.listdir(os.path.join(b, "model", "assets")))},
    "images": {"lambda_build": img("$LAMBDA_IMAGE"), "weights": img("$PY_IMAGE")},
    "toolchain": open(os.path.join(b, "toolchain.txt")).read().split("\n")[:2],
}
json.dump(m, open(os.path.join(b, "manifest.json"), "w"), indent=1)
print(json.dumps({k: m[k] for k in ("repo_commit", "repo_dirty")}), m["model"]["gguf_sha256"], m["function_zip"]["sha256"])
EOF
echo BUILD_DONE
