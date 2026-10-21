#!/bin/bash
# Build E8's libraries and zips on an arm64 Linux host with Docker:
#
#   LKAI_BUILD=<P1 build dir> examples/lambda-kleidiai/e8/build.sh <P1 function.zip> <P1 build manifest.json>
#
# 1. A pristine copy of P1's pinned llama.cpp (LLAMA_REV, from LKAI_BUILD/llama.cpp) gets P1's llama/patch_llama.py
#    and then e8/patch_parallel_repack.py (the dormant GGML_KLEIDIAI_REPACK_THREADS switch).
# 2. In P1's pinned Lambda image (same compiler, flags and KleidiAI release as baseline/container_build.sh), only the
#    libggml-cpu-* variant targets are built: KleidiAI's repack is compiled into those, and nothing else changes.
#    kaihash (the offline identity check) is compiled in the same image against P1's own shipped libraries.
# 3. make_zips.py writes base.zip (P1's zip + the E8 wrapper) and off.zip (the same, six variants replaced).
# Outputs in e8/build/: libs/, kaihash, base.zip, off.zip, zips.json, manifest.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(dirname "$HERE")
BASE=$RECIPE/baseline
REPO=$(cd "$RECIPE/../.." && pwd)
PB=${LKAI_BUILD:-$BASE/build}
B=${E8_BUILD:-$HERE/build}
P1_ZIP=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
P1_MANIFEST=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
LAMBDA_IMAGE=$(sed -n 's/^LAMBDA_IMAGE=//p' "$BASE/build.sh")
LLAMA_REV=$(sed -n 's/^LLAMA_REV=//p' "$BASE/build.sh")
[ "$(uname -m)" = aarch64 ] || { echo "build.sh needs an arm64 host (got $(uname -m))"; exit 1; }
[ "$(git -C "$PB/llama.cpp" rev-parse HEAD)" = "$LLAMA_REV" ] || { echo "$PB/llama.cpp is not $LLAMA_REV"; exit 1; }
DIRTY=$(git -C "$REPO" status --porcelain -- examples/lambda-kleidiai/e8 examples/lambda-kleidiai/llama \
  examples/lambda-kleidiai/baseline examples/lambda-kleidiai/shim | grep -v '^?? examples/lambda-kleidiai/e8/build' || true)
if [ -n "$DIRTY" ] && [ "${E8_ALLOW_DIRTY:-}" != 1 ]; then
  echo "E8 build inputs differ from HEAD (commit them, or E8_ALLOW_DIRTY=1 for a non-deployable build):" >&2
  echo "$DIRTY" >&2
  exit 2
fi
mkdir -p "$B"
rm -rf "$B/llama.cpp" && cp -a "$PB/llama.cpp" "$B/llama.cpp"
git -C "$B/llama.cpp" checkout -q -- .   # pristine LLAMA_REV, then exactly P1's patches plus E8's
python3 "$RECIPE/llama/patch_llama.py" "$B/llama.cpp"
python3 "$HERE/patch_parallel_repack.py" "$B/llama.cpp"

rm -rf "$B/libs" "$B/p1pkg" && mkdir -p "$B/libs" "$B/p1pkg"
(cd "$B/p1pkg" && unzip -q "$P1_ZIP" 'lib/*')
docker pull -q --platform linux/arm64 "$LAMBDA_IMAGE" >/dev/null
docker run --rm --platform linux/arm64 --entrypoint bash -v "$B/llama.cpp:/llama:ro" -v "$HERE:/e8:ro" \
  -v "$B/p1pkg/lib:/p1lib:ro" -v "$B:/out" "$LAMBDA_IMAGE" -c '
set -euo pipefail
dnf install -y gcc gcc-c++ cmake make binutils >/dev/null
rm -rf /b && cp -a /llama /b && cd /b
COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_CPU_KLEIDIAI=ON >/dev/null
cmake --build build -j "$(nproc)" --target $(cmake --build build --target help | awk "{print \$2}" | grep "^ggml-cpu" | sort -u) >/dev/null
cp build/bin/libggml-cpu-*.so /out/libs/
# grep -c on the file itself: no pipe, so pipefail cannot turn an early grep -q exit (SIGPIPE in nm) into a failure
for f in /out/libs/*.so; do [ "$(grep -c "kleidiai repack: threads" "$f")" -ge 1 ] || { echo "FATAL: $f lacks the E8 patch"; exit 1; }; done
g++ -O2 -std=c++17 -o /out/kaihash /e8/kaihash.cpp -I/b/include -I/b/ggml/include -L/p1lib -lllama -lggml -lggml-base \
  -Wl,-rpath,/p1lib
gcc --version | head -1 > /out/toolchain.txt
chmod -R a+rX /out/libs /out/kaihash
echo E8_LIBS_DONE'
python3 "$HERE/make_zips.py" --p1-zip "$P1_ZIP" --p1-manifest "$P1_MANIFEST" --libs "$B/libs" --out "$B"
python3 - "$B" "$P1_MANIFEST" "$REPO" "$HERE" "$DIRTY" <<'EOF'
import hashlib, json, os, subprocess, sys
b, p1m, repo, here, dirty = sys.argv[1:6]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
m = json.load(open(p1m))
z = json.load(open(os.path.join(b, "zips.json")))
m["schema"] = "lkai-build-manifest/2"
m["e8_manifest"] = "lkai-e8-build-manifest/1"
m["p1_build_repo_commit"] = m.get("repo_commit")
m["repo_commit"] = git("rev-parse", "HEAD")
m["dirty_inputs"] = [l for l in dirty.splitlines() if l]
m["repo_dirty"] = bool(m["dirty_inputs"])
m["llama_cpp"]["patches"] += " + e8/patch_parallel_repack.py (dormant GGML_KLEIDIAI_REPACK_THREADS switch)"
m["llama_cpp"]["e8_patch_sha256"] = sha(os.path.join(here, "patch_parallel_repack.py"))
m["llama_cpp"]["e8_toolchain"] = open(os.path.join(b, "toolchain.txt")).read().strip()
m["zips"] = {k: z[f"{k}_zip"] for k in ("base", "off")}
m["zips_detail"] = z
m["function_zip"] = None  # two zips: see zips; deploy_e8.py checks each arm against its own
m["e8_sources_sha256"] = {f: sha(os.path.join(here, f)) for f in sorted(os.listdir(here))
                          if f.endswith((".py", ".cpp", ".sh"))}
json.dump(m, open(os.path.join(b, "manifest.json"), "w"), indent=1)
print("manifest", m["repo_commit"], "dirty" if m["repo_dirty"] else "clean", {k: v["sha256"][:12] for k, v in m["zips"].items()})
EOF
echo E8_BUILD_DONE
