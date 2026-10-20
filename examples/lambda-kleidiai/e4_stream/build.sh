#!/bin/bash
# Build E4's function.zip on an arm64 Linux host with Docker:
#
#   examples/lambda-kleidiai/e4_stream/build.sh
#
# 1. P1's baseline/build.sh, unchanged, provides the pinned weights, assets and patched llama.cpp checkout
#    (skipped when LKAI_BUILD already holds the GGUF and llama.cpp P1's published run used).
# 2. A copy of that llama.cpp gets e4_stream/patch_stream.py (the dormant load-stream hooks).
# 3. P1's container_build.sh builds the libraries and the package from the patched copy, in the same pinned image;
#    then e4_entry.py and stream_loader.py are added and the zip is re-made with P1's fixed mtimes and order.
# Outputs in e4_stream/build/: function.zip, manifest.json (P1's manifest shape, with the E4 patch and zip
# recorded), model/s2b-h512-Q8_0.gguf (a hard link or copy of P1's GGUF; the same bytes, checked by sha256).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
BASE=$RECIPE/baseline
REPO=$(cd "$RECIPE/../.." && pwd)
PB=${LKAI_BUILD:-$BASE/build}
B=${E4_BUILD:-$HERE/build}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
LLAMA_REV=$(sed -n 's/^LLAMA_REV=//p' "$BASE/build.sh")
PUB=$BASE/results/20261020T080835Z/build-manifest.json
GGUF_SHA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model']['gguf_sha256'])" "$PUB")
[ "$(uname -m)" = aarch64 ] || { echo "build.sh needs an arm64 host (got $(uname -m))"; exit 1; }

if [ ! -f "$PB/model/s2b-h512-Q8_0.gguf" ] || [ "$(git -C "$PB/llama.cpp" rev-parse HEAD 2>/dev/null)" != "$LLAMA_REV" ]; then
  LKAI_BUILD=$PB "$BASE/build.sh"
fi
mkdir -p "$B/model"
if [ ! -f "$B/model/s2b-h512-Q8_0.gguf" ]; then
  ln "$PB/model/s2b-h512-Q8_0.gguf" "$B/model/s2b-h512-Q8_0.gguf" 2>/dev/null || cp "$PB/model/s2b-h512-Q8_0.gguf" "$B/model/"
fi
[ "$(sha256sum "$B/model/s2b-h512-Q8_0.gguf" | cut -c1-64)" = "$GGUF_SHA" ] || { echo "GGUF is not P1's ($GGUF_SHA)"; exit 1; }

rm -rf "$B/llama.cpp" && cp -a "$PB/llama.cpp" "$B/llama.cpp"
python3 "$HERE/patch_stream.py" "$B/llama.cpp"

docker pull -q --platform linux/arm64 "$LAMBDA_IMAGE" >/dev/null
docker run --rm --platform linux/arm64 --entrypoint bash \
  -v "$B/llama.cpp:/llama:ro" -v "$RECIPE:/recipe:ro" -v "$REPO/examples/lambda:/lambda:ro" \
  -v "$REPO/src/strands_decider:/pkgsrc:ro" -v "$PB/model/assets:/assets:ro" -v "$B:/out" \
  "$LAMBDA_IMAGE" -c '
set -euo pipefail
bash /recipe/baseline/container_build.sh
P=/pkg
cp /recipe/e4_stream/e4_entry.py /recipe/e4_stream/stream_loader.py $P/
(cd /tmp && PYTHONPATH=$P LD_LIBRARY_PATH=$P/lib python3.12 -c "
import ast, ctypes
for f in (\"e4_entry.py\", \"stream_loader.py\"):
    ast.parse(open(\"$P/\" + f).read())
ctypes.CDLL(\"$P/lib/libllama.so\").llama_set_load_stream
ctypes.CDLL(\"$P/lib/libggml-base.so\").gguf_set_read_wait
print(\"E4 hooks exported\")")
chmod -R u=rwX,go=rX $P
find $P -exec touch -h -d "2026-01-01 00:00:00" {} +
rm -f /out/function.zip
(cd $P && find . -mindepth 1 -printf "%P\n" | LC_ALL=C sort | zip -q -X -y -D /out/function.zip -@)
chmod a+r /out/function.zip
echo "E4 function.zip $(stat -c %s /out/function.zip) bytes"'

python3 - "$B" "$PB" "$REPO" "$HERE" <<'EOF'
import hashlib, json, os, subprocess, sys
b, pb, repo, here = sys.argv[1:5]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
m = json.load(open(os.path.join(pb, "manifest.json")))
m["repo_commit"] = git("rev-parse", "HEAD")
m["repo_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
m["llama_cpp"]["patches"] += " + e4_stream/patch_stream.py (dormant load-stream hooks)"
m["llama_cpp"]["e4_patch_sha256"] = sha(os.path.join(here, "patch_stream.py"))
z = os.path.join(b, "function.zip")
m["function_zip"] = {"bytes": os.path.getsize(z), "sha256": sha(z), "handler": "e4_entry.handler",
                     "source_commit": m["repo_commit"], "inputs_sha256": None}
m["model"]["gguf_sha256_checked"] = sha(os.path.join(b, "model", "s2b-h512-Q8_0.gguf"))
assert m["model"]["gguf_sha256_checked"] == m["model"]["gguf_sha256"]
json.dump(m, open(os.path.join(b, "manifest.json"), "w"), indent=1)
print("manifest", m["repo_commit"], "dirty" if m["repo_dirty"] else "clean", m["function_zip"]["sha256"])
EOF
echo E4_BUILD_DONE
