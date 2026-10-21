#!/bin/bash
# Build E7's function.zip on an arm64 Linux host with Docker:
#
#   examples/lambda-kleidiai/e7/build.sh
#
# 1. P1's baseline/build.sh provides the pinned weights, assets and patched llama.cpp checkout. It is skipped when
#    LKAI_BUILD already holds P1's published GGUF (sha256 from the published build manifest) and llama.cpp at LLAMA_REV.
# 2. A copy of that llama.cpp gets e7/patch_skip_output.py (the dormant LKAI_SKIP_TIED_OUTPUT switch).
# 3. P1's container_build.sh builds the libraries and the package from the patched copy, in P1's pinned image. Then
#    e7_entry.py and e7_host.py are added and the zip is re-made with P1's fixed mtimes, modes and sorted order.
# Outputs in e7/build/: function.zip and manifest.json (P1's manifest shape, with the E7 patch and zip recorded).
# The GGUF is not copied: E7 deploys P1's weights object as is.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
BASE=$RECIPE/baseline
REPO=$(cd "$RECIPE/../.." && pwd)
PB=${LKAI_BUILD:-$BASE/build}
B=${E7_BUILD:-$HERE/build}
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
LLAMA_REV=$(sed -n 's/^LLAMA_REV=//p' "$BASE/build.sh")
PUB=$BASE/results/20261020T080835Z/build-manifest.json
GGUF_SHA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model']['gguf_sha256'])" "$PUB")
[ "$(uname -m)" = aarch64 ] || { echo "build.sh needs an arm64 host (got $(uname -m))"; exit 1; }
# The manifest names a commit: refuse a tree whose E7 or baseline inputs differ from it (E7_ALLOW_DIRTY=1 for
# iteration). P1's buildcache.dirty covers P1's keyed and harness inputs; E7_SOURCES adds this directory's files.
E7_SOURCES="examples/lambda-kleidiai/e7/patch_skip_output.py examples/lambda-kleidiai/e7/e7_entry.py
examples/lambda-kleidiai/e7/e7_host.py examples/lambda-kleidiai/e7/build.sh examples/lambda-kleidiai/e7/deploy_e7.py
examples/lambda-kleidiai/e7/bench_e7.py examples/lambda-kleidiai/e7/summarize_e7.py"
# P1's buildcache needs Python 3.10+ and git; the build host may have neither, so it runs in P1's pinned weights
# image (PY_IMAGE) against a read-only mount of the repository.
PY_IMAGE=$(sed -n 's/^PY_IMAGE=//p' "$BASE/build.sh")
DIRTY=$(docker run -i --rm --platform linux/arm64 -v "$REPO:/repo:ro" -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' --entrypoint python3 "$PY_IMAGE" - /repo \
  /repo/examples/lambda-kleidiai/baseline $E7_SOURCES <<'EOF'
import sys
sys.path.insert(0, sys.argv[2])
import buildcache
for line in buildcache.dirty(sys.argv[1], extra=buildcache.UNKEYED_SOURCES + sys.argv[3:]):
    print(line)
EOF
) || { echo "dirty-input check failed"; exit 1; }
if [ -n "$DIRTY" ] && [ "${E7_ALLOW_DIRTY:-}" != 1 ]; then
  echo "E7 build inputs differ from HEAD (commit them, or E7_ALLOW_DIRTY=1 for a non-deployable build):" >&2
  echo "$DIRTY" >&2
  exit 2
fi

if [ ! -f "$PB/model/s2b-h512-Q8_0.gguf" ] || [ "$(git -C "$PB/llama.cpp" rev-parse HEAD 2>/dev/null)" != "$LLAMA_REV" ]; then
  LKAI_BUILD=$PB "$BASE/build.sh"
fi
[ "$(sha256sum "$PB/model/s2b-h512-Q8_0.gguf" | cut -c1-64)" = "$GGUF_SHA" ] || { echo "GGUF is not P1's ($GGUF_SHA)"; exit 1; }
mkdir -p "$B"

rm -rf "$B/llama.cpp" && cp -a "$PB/llama.cpp" "$B/llama.cpp"
python3 "$HERE/patch_skip_output.py" "$B/llama.cpp"

docker pull -q --platform linux/arm64 "$LAMBDA_IMAGE" >/dev/null
docker run --rm --platform linux/arm64 --entrypoint bash \
  -v "$B/llama.cpp:/llama:ro" -v "$RECIPE:/recipe:ro" -v "$REPO/examples/lambda:/lambda:ro" \
  -v "$REPO/src/strands_decider:/pkgsrc:ro" -v "$PB/model/assets:/assets:ro" -v "$B:/out" \
  "$LAMBDA_IMAGE" -c '
set -euo pipefail
bash /recipe/baseline/container_build.sh
P=/pkg
cp /recipe/e7/e7_entry.py /recipe/e7/e7_host.py $P/
(cd /tmp && PYTHONPATH=$P LD_LIBRARY_PATH=$P/lib python3.12 -c "
import ast, ctypes, e7_host
ast.parse(open(\"$P/e7_entry.py\").read())
lib = ctypes.CDLL(\"$P/lib/libllama.so\")
lib.llama_model_lkai_tied_output_skipped
lib.llama_model_lkai_buft_bytes
print(\"E7 queries exported\")")
chmod -R u=rwX,go=rX $P
find $P -exec touch -h -d "2026-01-01 00:00:00" {} +
rm -f /out/function.zip
(cd $P && find . -mindepth 1 -printf "%P\n" | LC_ALL=C sort | zip -q -X -y -D /out/function.zip -@)
chmod a+r /out/function.zip
echo "E7 function.zip $(stat -c %s /out/function.zip) bytes"'

python3 - "$B" "$PB" "$REPO" "$HERE" "$DIRTY" <<'EOF'
import hashlib, json, os, subprocess, sys
b, pb, repo, here, dirty = sys.argv[1:6]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
m = json.load(open(os.path.join(pb, "manifest.json")))
m["schema"] = "lkai-build-manifest/2"  # P1's shape and dirty-input check, so P1's deploy/bench gates apply
m["e7_manifest"] = "lkai-e7-build-manifest/1"
m["p1_build_repo_commit"] = m.get("repo_commit")
m["repo_commit"] = git("rev-parse", "HEAD")
m["dirty_inputs"] = [l for l in dirty.splitlines() if l]
m["repo_dirty"] = bool(m["dirty_inputs"])
m["llama_cpp"]["patches"] += " + e7/patch_skip_output.py (dormant LKAI_SKIP_TIED_OUTPUT switch)"
m["llama_cpp"]["e7_patch_sha256"] = sha(os.path.join(here, "patch_skip_output.py"))
z = os.path.join(b, "function.zip")
m["function_zip"] = {"bytes": os.path.getsize(z), "sha256": sha(z), "handler": "e7_entry.handler",
                     "source_commit": m["repo_commit"], "inputs_sha256": None}
m["e7_sources_sha256"] = {f: sha(os.path.join(here, f)) for f in ("e7_entry.py", "e7_host.py", "patch_skip_output.py")}
json.dump(m, open(os.path.join(b, "manifest.json"), "w"), indent=1)
print("manifest", m["repo_commit"], "dirty" if m["repo_dirty"] else "clean", m["function_zip"]["sha256"])
EOF
echo E7_BUILD_DONE
