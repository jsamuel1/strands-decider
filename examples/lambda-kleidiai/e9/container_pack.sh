#!/bin/bash
# Runs INSIDE python:3.12-bookworm (linux/arm64, the digest baseline/build.sh pins). Called by build.sh:
#   container_pack.sh <in.gguf> <out.gguf> <record.json>      (KR=8 or 4)
# /b is P1's build/ (ro, llama.cpp f74e2fa6a with its gguf-py), /e9 is this directory (ro), the outputs are rw.
set -euo pipefail
in=$1 out=$2 rec=$3
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
python -m venv /venv && . /venv/bin/activate
pip install -q "numpy==2.5.4" "pyyaml==6.0.3" "tqdm==4.67.1" /b/llama.cpp/gguf-py
python /e9/pack_e9.py "$in" "$out.tmp" --kr "${KR:-8}" --record /tmp/pack.json
mv "$out.tmp" "$out"
python - "$rec" "$in" "$out" <<'EOF'
import hashlib, importlib.metadata, json, os, platform, subprocess, sys
rec, src, dst = sys.argv[1:]
def sha(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
llama = subprocess.run(["git", "-c", "safe.directory=/b/llama.cpp", "-C", "/b/llama.cpp", "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
p = json.load(open("/tmp/pack.json"))
p |= {"image": os.environ.get("E9_PACK_IMAGE"), "source": {"file": os.path.basename(src), "bytes": os.path.getsize(src), "sha256": sha(src)},
      "packed": {"file": os.path.basename(dst), "bytes": os.path.getsize(dst), "sha256": sha(dst)},
      "tools": {"python": platform.python_version(), "llama_cpp": llama, "gguf_py": importlib.metadata.version("gguf"),
                "numpy": importlib.metadata.version("numpy")}}
json.dump(p, open(rec, "w"), indent=1, sort_keys=True)
print(json.dumps({k: p[k] for k in ("source", "packed", "packed_count")}))
EOF
chmod a+r "$out" "$rec"
echo PACK_DONE
