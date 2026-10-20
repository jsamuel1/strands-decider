#!/bin/bash
# Runs INSIDE python:3.12-bookworm (linux/arm64; run.sh pins the digest baseline/build.sh pins). Called by run.sh:
#   container_convert.sh <in.gguf> <out.gguf> <record.json>
# /b is P1's build/ (ro: llama.cpp f74e2fa6a with its gguf-py), /e5 is this directory (ro), the outputs are rw.
set -euo pipefail
in=$1 out=$2 rec=$3
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
python -m venv /venv && . /venv/bin/activate
pip install -q "numpy==2.5.4" "pyyaml==6.0.3" "tqdm==4.67.1" /b/llama.cpp/gguf-py
python /e5/convert_no_vocab.py "$in" "$out.tmp"
mv "$out.tmp" "$out"
python /e5/tensor_identity.py "$in" "$out" > /tmp/id.json
python - "$rec" <<'EOF'
import json, platform, subprocess, sys
idj = json.load(open("/tmp/id.json"))
free = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.split()
llama = subprocess.run(["git", "-C", "/b/llama.cpp", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump(idj | {"image": "python:3.12-bookworm (digest pinned in baseline/build.sh PY_IMAGE)",
                 "tools": {"python": platform.python_version(), "llama_cpp": llama,
                           "pip_freeze": [p for p in free if not p.startswith("gguf")] + ["gguf (llama.cpp gguf-py)"]}},
          open(sys.argv[1], "w"), indent=1)
EOF
chmod a+r "$out" "$rec"
echo CONVERT_DONE
