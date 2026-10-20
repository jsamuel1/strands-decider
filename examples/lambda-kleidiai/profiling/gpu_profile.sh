#!/bin/bash
# Profile the baseline engine on a GPU host (NVIDIA driver + nvidia-container-toolkit + docker), outside Lambda.
#
#   GGUF_URL=<presigned https> ZIP_URL=<presigned https> GGUF_SHA256=<hex> REPO_URL=... REPO_REF=<commit> \
#     bash gpu_profile.sh /opt/lkai-p3
#
# The URLs are short-lived presigned GETs of the baseline's own GGUF and function.zip (`aws s3 presign`, see
# profiling/README.md), so the host needs no AWS permission. Works in the given directory only, and changes nothing else on the host.
set -euo pipefail
W=${1:-/opt/lkai-p3}
mkdir -p "$W/in" "$W/out"
cd "$W"
if [ ! -f in/model.gguf ] || [ "$(sha256sum in/model.gguf | cut -d' ' -f1)" != "$GGUF_SHA256" ]; then
  curl -sSf -o in/model.gguf.tmp "$GGUF_URL"
  [ "$(sha256sum in/model.gguf.tmp | cut -d' ' -f1)" = "$GGUF_SHA256" ] || { echo "GGUF sha256 mismatch"; exit 1; }
  mv in/model.gguf.tmp in/model.gguf
fi
curl -sSf -o in/function.zip "$ZIP_URL"
rm -rf sd && git clone -q "$REPO_URL" sd && git -C sd checkout -q "$REPO_REF"
nvidia-smi -L > out/nvidia-smi-L.txt
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$W/sd:/w/sd:ro" -v "$W/in:/w/in:ro" -v "$W/out:/w/out" \
  -v "$W/cache:/b" nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /w/sd/examples/lambda-kleidiai/profiling/gpu_container.sh
tar -C "$W" -czf "$W/gpu-out.tgz" out
echo "GPU_PROFILE_DONE $W/gpu-out.tgz"
