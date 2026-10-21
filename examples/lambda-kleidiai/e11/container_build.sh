#!/bin/bash
# Runs INSIDE the pinned Lambda python3.12 image (linux/arm64), like baseline/container_build.sh, with:
#   /llama   P1's patched llama.cpp (build/llama.cpp from baseline/build.sh) (ro)
#   /recipe  examples/lambda-kleidiai (ro)      /lambda  examples/lambda (ro)
#   /pkgsrc  src/strands_decider (ro)           /assets  P1's build/model/assets (ro)
#   /out     e11/build (rw)
# Applies e11/patch_memfdrel.py to a copy of /llama, then runs P1's container_build.sh unchanged on that copy, and adds
# E11's entry point (e11_entry.py, hostmeta.py, memstat.py) to the package before re-zipping it with P1's flags.
# One zip serves both arms: the release is dormant unless LKAI_RELEASE_MEMFD=1.
set -euo pipefail
rm -rf /llama-e11 && cp -a /llama /llama-e11
python3.12 /recipe/e11/patch_memfdrel.py /llama-e11
# P1's build script reads /llama; point it at the patched copy without editing it.
sed 's#cp -a /llama /b#cp -a /llama-e11 /b#' /recipe/baseline/container_build.sh > /tmp/container_build_e11.sh
grep -q 'cp -a /llama-e11 /b' /tmp/container_build_e11.sh || { echo "FATAL: P1 container_build.sh anchor moved"; exit 1; }
bash /tmp/container_build_e11.sh

P=/pkg
# Capture first: `nm | grep -q` under pipefail fails when grep exits early and nm takes SIGPIPE.
SYMS=$(nm -D $P/lib/libllama.so)
grep -q lkai_memfd_release_stats <<<"$SYMS" || { echo "FATAL: libllama lacks lkai_memfd_release_stats"; exit 1; }
cp /recipe/e11/e11_entry.py /recipe/e11/hostmeta.py /recipe/e11/memstat.py $P/
(cd /tmp && PYTHONPATH=$P python3.12 -c "import ast, hostmeta, memstat; ast.parse(open('$P/e11_entry.py').read()); print('e11 OK')")
rm -f /out/function.zip
chmod -R u=rwX,go=rX $P
find $P -exec touch -h -d '2026-01-01 00:00:00' {} +
(cd $P && find . -mindepth 1 -printf '%P\n' | LC_ALL=C sort | zip -q -X -y -D /out/function.zip -@)
echo "function.zip $(stat -c %s /out/function.zip) bytes"
echo E11_CONTAINER_BUILD_DONE
