#!/bin/bash
# E6's variable on an x86_64 package: every .py gets py_compile UNCHECKED_HASH bytecode, compiled by the image's own
# CPython 3.12 with dfile /var/task/<path>, exactly as e6/pyc_zip.py (lkai/fada-hotspot e989e39) compiles P1's zip.
# pyc_zip.py itself refuses any zip but P1's arm64 one (it checks P1's manifest sha256), so this applies its
# compile step (general/pyc_tree.py) to an unzipped package instead. Nothing else in the package changes.
#
#   general/x86_e6.sh <x86 pkg-p1 dir> <out pkg-e6 dir>
#
# Writes <out pkg-e6 dir>.manifest.json (interpreter, magic, compiled count, every failure, tree sha256) and exits
# nonzero, without printing X86_E6_DONE, when any .py failed to compile.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$1" && pwd)
rm -rf "$2" && cp -a "$SRC" "$2" && OUT=$(cd "$2" && pwd)
MAN=$OUT.manifest.json
rm -f "$MAN"
IMG=${X86_E6_IMAGE:-public.ecr.aws/lambda/python@sha256:586cea700429caaab11d50f7b96258a77bd4899d253e9559acfb647fe383f970}
PY=${X86_E6_PYTHON:-}
if [ -n "$PY" ]; then  # tests: run the compile step with a local interpreter instead of the image
  "$PY" "$HERE/pyc_tree.py" "$OUT" "$MAN"
else
  touch "$MAN"
  docker run --rm --platform linux/amd64 --network none --user "$(id -u):$(id -g)" --entrypoint python3.12 \
    -v "$OUT:/var/task" -v "$HERE/pyc_tree.py:/g/pyc_tree.py:ro" -v "$MAN:/m/manifest.json" "$IMG" \
    /g/pyc_tree.py /var/task /m/manifest.json
fi
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(1 if m['failed'] or not m['compiled'] else 0)" "$MAN" \
  || { echo "FATAL: $MAN records failed or zero compiles"; exit 1; }
chmod -R a+rX "$OUT"
echo X86_E6_DONE
