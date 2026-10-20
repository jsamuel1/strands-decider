#!/bin/bash
# Runs INSIDE python:3.12-bookworm (linux/arm64; run.sh pins the digest baseline/build.sh pins). Called by run.sh:
#   container_convert.sh <in.gguf> <out.gguf> <record.json>
# /b is P1's build/ (ro: llama.cpp f74e2fa6a with its gguf-py), /e5 is this directory (ro), the outputs are rw.
set -euo pipefail
in=$1 out=$2 rec=$3
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
EXPECTED_IMAGE=python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b
[ "${LKAI_E5_CONVERTER_IMAGE:-}" = "$EXPECTED_IMAGE" ] || {
  echo "container identity is not pinned: expected LKAI_E5_CONVERTER_IMAGE=$EXPECTED_IMAGE" >&2
  exit 1
}
python -m venv /venv && . /venv/bin/activate
pip install -q "numpy==2.5.4" "pyyaml==6.0.3" "tqdm==4.67.1" /b/llama.cpp/gguf-py
python /e5/convert_no_vocab.py "$in" "$out.tmp"
mv "$out.tmp" "$out"
python /e5/tensor_identity.py "$in" "$out" > /tmp/id.json
python - "$rec" <<'EOF'
import importlib.metadata
import json, platform, subprocess, sys
idj = json.load(open("/tmp/id.json"))
free = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True).stdout.splitlines()
llama = subprocess.run(
    ["git", "-C", "/b/llama.cpp", "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
json.dump(idj | {"image": __import__("os").environ["LKAI_E5_CONVERTER_IMAGE"],
                 "tools": {"python": platform.python_version(), "llama_cpp": llama,
                           "gguf_py": importlib.metadata.version("gguf"),
                           "pip_freeze": free}},
          open(sys.argv[1], "w"), indent=1)
EOF
chmod a+r "$out" "$rec"
echo CONVERT_DONE
