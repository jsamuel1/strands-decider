#!/bin/bash
# E6's variable on an x86_64 package: every .py gets py_compile UNCHECKED_HASH bytecode, compiled by the image's own
# CPython 3.12 with dfile /var/task/<path>, exactly as e6/pyc_zip.py (lkai/fada-hotspot e989e39) compiles P1's zip.
# pyc_zip.py itself refuses any zip but P1's arm64 one (it checks P1's manifest sha256), so this applies its
# compile step to an unzipped package instead. Nothing else in the package changes.
#
#   general/x86_e6.sh <x86 pkg-p1 dir> <out pkg-e6 dir>
set -euo pipefail
SRC=$(cd "$1" && pwd)
rm -rf "$2" && cp -a "$SRC" "$2" && OUT=$(cd "$2" && pwd)
IMG=public.ecr.aws/lambda/python@sha256:586cea700429caaab11d50f7b96258a77bd4899d253e9559acfb647fe383f970
docker run --rm --platform linux/amd64 --network none --user "$(id -u):$(id -g)" --entrypoint python3.12 \
  -v "$OUT:/var/task" "$IMG" -c '
import os, py_compile
ok = bad = 0
for d, _, fs in os.walk("/var/task"):
    for f in fs:
        if not f.endswith(".py"):
            continue
        p = os.path.join(d, f)
        try:
            py_compile.compile(p, dfile=p, doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            ok += 1
        except py_compile.PyCompileError:
            bad += 1
print("compiled", ok, "failed", bad)'
chmod -R a+rX "$OUT"
echo X86_E6_DONE
