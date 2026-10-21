#!/bin/bash
# Runs INSIDE the pinned Lambda python3.12 image (arm64). Mounts: /zip/function.zip (ro), /model (ro), /e11 (ro),
# /requests.json (ro, optional), /out (rw). $1 = output json name. Unzips the package and runs premise_probe.py on
# the P1 GGUF. LKAI_RELEASE_MEMFD passes through (E11's zip only acts on it).
set -euo pipefail
dnf install -y unzip >/dev/null 2>&1
rm -rf /tmp/pkg && mkdir -p /tmp/pkg && cd /tmp/pkg && unzip -q /zip/function.zip
cd /tmp
R=""
[ -f /requests.json ] && R=/requests.json
LD_LIBRARY_PATH=/tmp/pkg/lib THREADS="${THREADS:-6}" N_SEQ=17 DEC_KV=f32 DEC_UBATCH=512 python3.12 /e11/premise_probe.py \
  /tmp/pkg /model/s2b-h512-Q8_0.gguf "/out/$1" $R
