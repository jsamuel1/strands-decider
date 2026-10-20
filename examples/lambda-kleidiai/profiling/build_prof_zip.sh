#!/bin/bash
# Build the P3 profiler package from the deployed P1 function.zip, adding files only, on an arm64 host with Docker.
#
#   examples/lambda-kleidiai/profiling/build_prof_zip.sh <p1-function.zip> <out-dir>
#
# Writes <out-dir>/prof.zip:          P1's zip, unchanged, plus lkai_prof.py, lkai_probe.py, lkai_symbolise.py,
#                                     lib/liblkaisamp.so (built in the same Lambda python3.12 image P1 pins) and
#                                     bin/py-spy (py-spy 0.4.2's static aarch64 binary, from its PyPI wheel)
#        <out-dir>/telemetry-layer.zip: extensions/lkai-telemetry (the Telemetry API extension), for the telemetry arm
#        <out-dir>/prof-manifest.json:  sha256 of every input and output, and the list of added files
# Every P1 file keeps its bytes, so each arm runs the baseline's code; only the added handler wraps it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
OUT=$(mkdir -p "$2" && cd "$2" && pwd)
LAMBDA_IMAGE=public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc
PYSPY_WHL=py_spy-0.4.2-py2.py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
PYSPY_SHA256=${PYSPY_SHA256:-}
[ "$(uname -m)" = aarch64 ] || { echo "needs an arm64 host"; exit 1; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/add/lib" "$W/add/bin" "$W/layer/extensions"
python3 -m pip download -q --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.12 \
  py-spy==0.4.2 -d "$W/whl"
[ -z "$PYSPY_SHA256" ] || [ "$(sha256sum "$W/whl/$PYSPY_WHL" | cut -d' ' -f1)" = "$PYSPY_SHA256" ] \
  || { echo "py-spy wheel sha256 mismatch"; exit 1; }
(cd "$W/whl" && python3 -m zipfile -e "$PYSPY_WHL" x)
cp "$W/whl/x/py_spy-0.4.2.data/scripts/py-spy" "$W/add/bin/py-spy"
docker run --rm --platform linux/arm64 --entrypoint bash -v "$HERE:/p:ro" -v "$W/add/lib:/o" "$LAMBDA_IMAGE" -c \
  "dnf install -y gcc >/dev/null && gcc -O2 -g -shared -fPIC -o /o/liblkaisamp.so /p/lkaisamp.c"
cp "$HERE/lkai_prof.py" "$HERE/lkai_probe.py" "$HERE/lkai_symbolise.py" "$W/add/"
cp "$HERE/extensions/lkai-telemetry" "$W/layer/extensions/"
chmod -R u=rwX,go=rX "$W/add" "$W/layer"
chmod 755 "$W/add/bin/py-spy" "$W/layer/extensions/lkai-telemetry"
find "$W/add" "$W/layer" -exec touch -h -d '2026-01-01 00:00:00' {} +
cp "$SRC" "$OUT/prof.zip"
(cd "$W/add" && find . -mindepth 1 -type f -printf '%P\n' | LC_ALL=C sort | zip -q -X -y "$OUT/prof.zip" -@)
rm -f "$OUT/telemetry-layer.zip"
(cd "$W/layer" && find . -mindepth 1 -printf '%P\n' | LC_ALL=C sort | zip -q -X -y "$OUT/telemetry-layer.zip" -@)
python3 - "$SRC" "$OUT" "$W/add" "$W/whl/$PYSPY_WHL" <<'EOF'
import hashlib, json, os, sys
src, out, add, whl = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
added = sorted(os.path.relpath(os.path.join(r, f), add) for r, _, fs in os.walk(add) for f in fs)
m = {"schema": "lkai-prof-build/1", "base_zip_sha256": sha(src), "prof_zip_sha256": sha(f"{out}/prof.zip"),
     "telemetry_layer_sha256": sha(f"{out}/telemetry-layer.zip"), "pyspy_wheel_sha256": sha(whl),
     "added": {f: sha(os.path.join(add, f)) for f in added}}
json.dump(m, open(f"{out}/prof-manifest.json", "w"), indent=1)
print(json.dumps(m, indent=1))
EOF
echo PROF_ZIP_DONE
