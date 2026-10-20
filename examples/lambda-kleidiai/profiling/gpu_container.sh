#!/bin/bash
# Runs INSIDE nvidia/cuda:12.8.1-devel-ubuntu24.04 on the GPU host (docker run --gpus all), called by gpu_profile.sh:
#   /w/sd      this repository at the profiled commit (ro)
#   /w/in      model.gguf and function.zip (the baseline's deployed zip: only its assets/ and .py files are used)
#   /w/out     results (rw)
# Builds the SAME llama.cpp commit and patches as the Lambda zip, for x86_64 with the CUDA backend and the portable CPU
# variants, and decshim.c with one change: DEC_NGL sets n_gpu_layers (the Lambda shim hard-codes 0). Then:
#   cpu   offline_drive.py on the x86_64 CPU backend, CUDA hidden (CUDA_VISIBLE_DEVICES=), 6 threads
#   gpu   offline_drive.py with every layer offloaded (DEC_NGL=99), nvidia-smi sampled every 100 ms beside it
#   nsys  the gpu run again under Nsight Systems (cuda, nvtx, osrt), 3 passes
#   ncu   Nsight Compute on the first 40 kernels of one forward (needs the profiling capability; recorded if denied)
# None of this is Lambda evidence. Every output is labelled with its host.
set -euo pipefail
LLAMA_REV=f74e2fa6a31cc33ab4ee4ae516641cfe62df3924
R=/w/sd/examples/lambda-kleidiai
O=/w/out
mkdir -p "$O" /b
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null
apt-get install -y -q --no-install-recommends git cmake build-essential python3 python3-venv python3-pip unzip \
  ca-certificates curl >/dev/null
apt-get install -y -q --no-install-recommends cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 >/dev/null 2>&1 \
  || echo "nsight packages not installable" > "$O/nsight-install.txt"
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1 || true)
NCU=$(ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1 || true)

if [ ! -f /b/llama/.done ]; then
  rm -rf /b/llama && mkdir -p /b/llama && cd /b/llama
  git init -q && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_REV" && git checkout -q FETCH_HEAD
  python3 "$R/llama/patch_llama.py" /b/llama
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF \
    -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null
  cmake --build build -j "$(nproc)" >/b/build.log 2>&1 || { tail -40 /b/build.log; exit 1; }
  touch /b/llama/.done
fi

P=/b/pkg
rm -rf $P && mkdir -p $P/lib $P/strands_decider
cp -P /b/llama/build/bin/lib*.so* $P/lib/
# decshim.c with DEC_NGL: the only source change, and only in this GPU build.
sed 's|    mp.n_gpu_layers = 0;|    { const char * ngl = getenv("DEC_NGL"); mp.n_gpu_layers = ngl ? atoi(ngl) : 0; }|' \
  "$R/shim/decshim.c" > /b/decshim_gpu.c
grep -q 'getenv("DEC_NGL")' /b/decshim_gpu.c
gcc -O2 -g -shared -fPIC -o $P/lib/libdecshim.so /b/decshim_gpu.c -I/b/llama/include -I/b/llama/ggml/include \
  -L$P/lib -lllama -lggml -Wl,-rpath,'$ORIGIN'
gcc -O2 -g -shared -fPIC -o /b/lkaisamp.so "$R/profiling/lkaisamp.c"
cp "$R/snapstart/engine.py" $P/
(cd $P && unzip -q -o /w/in/function.zip 'assets/*')
cp /w/sd/src/strands_decider/{__init__,schema,prompting,hidden_engine}.py $P/strands_decider/
python3 -m venv /b/venv
/b/venv/bin/pip install -q pydantic==2.14.0 numpy==2.5.3 tokenizers==0.23.2 safetensors==0.8.0 py-spy
PY=/b/venv/bin/python3
{ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader; uname -a;
  grep -m1 'model name' /proc/cpuinfo; nproc; $PY --version; ls $P/lib; echo "nsys=$NSYS ncu=$NCU"; } > "$O/host.txt"

DRIVE="$PY $R/profiling/offline_drive.py --pkg $P --gguf /w/in/model.gguf --threads 6"
echo ">> x86_64 CPU (CUDA hidden)"
CUDA_VISIBLE_DEVICES= $DRIVE --passes 10 --host-label "g6.4xlarge AMD EPYC 7R13 CPU" --out "$O/x86-cpu.json" \
  >/dev/null 2>"$O/x86-cpu.stderr"
echo ">> x86_64 CPU with lkaisamp (1 kHz) and cProfile"
CUDA_VISIBLE_DEVICES= $DRIVE --passes 10 --sampler /b/lkaisamp.so --samples "$O/x86-cpu.samp" \
  --cprofile "$O/x86-cpu.prof" --host-label "g6.4xlarge AMD EPYC 7R13 CPU" --out "$O/x86-cpu-sampled.json" \
  >/dev/null 2>"$O/x86-cpu-sampled.stderr"
echo ">> x86_64 CPU under py-spy (100 Hz, native)"
CUDA_VISIBLE_DEVICES= /b/venv/bin/py-spy record --rate 100 --native --threads --format raw -o "$O/x86-cpu.pyspy.txt" \
  -- $DRIVE --passes 3 --host-label "g6.4xlarge CPU (py-spy)" --out "$O/x86-cpu-pyspy.json" >/dev/null \
  2>"$O/x86-cpu-pyspy.stderr" || echo "py-spy failed rc=$?" >> "$O/x86-cpu-pyspy.stderr"

echo ">> GPU (all layers offloaded), nvidia-smi every 100 ms"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,pstate \
  --format=csv -lms 100 > "$O/gpu.nvidia-smi.csv" &
SMI=$!
$DRIVE --ngl 99 --passes 10 --host-label "g6.4xlarge NVIDIA L4" --out "$O/gpu.json" >/dev/null 2>"$O/gpu.stderr"
kill $SMI || true
if [ -n "$NSYS" ]; then
  echo ">> Nsight Systems"
  "$NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o "$O/gpu-nsys" --force-overwrite true \
    $DRIVE --ngl 99 --passes 3 --host-label "g6.4xlarge NVIDIA L4 (nsys)" --out "$O/gpu-nsys.json" >/dev/null \
    2>"$O/gpu-nsys.stderr" || echo "nsys rc=$?" >> "$O/gpu-nsys.stderr"
  "$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
    -o "$O/gpu-nsys" "$O/gpu-nsys.nsys-rep" >/dev/null 2>>"$O/gpu-nsys.stderr" || true
fi
if [ -n "$NCU" ]; then
  echo ">> Nsight Compute (first 40 kernels)"
  "$NCU" --launch-count 40 --set basic --csv --page raw $DRIVE --ngl 99 --passes 1 --out "$O/gpu-ncu.json" \
    > "$O/gpu-ncu.csv" 2>"$O/gpu-ncu.stderr" || echo "ncu rc=$?" >> "$O/gpu-ncu.stderr"
fi
chmod -R a+rX "$O"
echo GPU_CONTAINER_DONE
