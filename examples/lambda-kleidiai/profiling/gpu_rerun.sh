#!/bin/bash
# Re-run the GPU host's container step on inputs gpu_profile.sh already fetched (no new presigned URLs needed).
#   bash gpu_rerun.sh /opt/lkai-p3 <commit>
set -euo pipefail
W=$1
REF=$2
cd "$W"
rm -rf sd && git clone -q https://github.com/jsamuel1/strands-decider.git sd && git -C sd checkout -q "$REF"
rm -rf out && mkdir -p out
nvidia-smi -L > out/nvidia-smi-L.txt
docker run --rm --gpus all --cap-add SYS_ADMIN -v "$W/sd:/w/sd:ro" -v "$W/in:/w/in:ro" -v "$W/out:/w/out" \
  -v "$W/cache:/b" nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /w/sd/examples/lambda-kleidiai/profiling/gpu_container.sh
tar -C "$W" -czf "$W/gpu-out.tgz" out
echo "GPU_PROFILE_DONE $W/gpu-out.tgz"
