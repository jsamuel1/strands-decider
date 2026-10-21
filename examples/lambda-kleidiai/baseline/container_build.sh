#!/bin/bash
# Runs INSIDE public.ecr.aws/lambda/python:3.12 (linux/arm64, AL2023, GCC 11). Called by build.sh with:
#   /llama   llama.cpp f74e2fa6a with llama/patch_llama.py applied (ro)
#   /recipe  examples/lambda-kleidiai (ro)      /lambda  examples/lambda (ro)
#   /pkgsrc  src/strands_decider (ro)           /assets  build/model/assets (ro)
#   /out     build/ (rw): writes lib/, function.zip and bin/llama-quantize
# The package is PR #77's zip layout (examples/lambda/container_build.sh) with the snapstart/ engine and app and the
# portable KleidiAI libraries (snapstart/build_ours.sh), plus baseline/lkai_entry.py and phases.py. Same flags as the
# measured recipe; nothing else differs.
set -euo pipefail
dnf install -y gcc gcc-c++ cmake make binutils zip findutils >/dev/null

rm -rf /b && cp -a /llama /b && cd /b
COMMON="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"
echo "=== llama.cpp: portable variants + KleidiAI (snapstart/build_ours.sh) ==="
cmake -B build $COMMON -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_CPU_KLEIDIAI=ON >/dev/null
cmake --build build -j "$(nproc)" --target llama \
  $(cmake --build build --target help | awk '{print $2}' | grep '^ggml-cpu' | sort -u) >/dev/null
echo "=== llama-quantize (plain armv8.2 build; Q8_0 rounding is the same on every path) ==="
cmake -B build-q $COMMON -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod \
  -DGGML_BACKEND_DL=OFF >/dev/null
cmake --build build-q -j "$(nproc)" --target llama-quantize >/dev/null
mkdir -p /out/bin /out/quant-lib && cp -P build-q/bin/llama-quantize /out/bin/ && cp -P build-q/bin/lib*.so* /out/quant-lib/
cp -L /usr/lib64/libgomp.so.1 /usr/lib64/libstdc++.so.6 /usr/lib64/libgcc_s.so.1 /out/quant-lib/  # gcc not in the run image

P=/pkg
rm -rf $P && mkdir -p $P/lib $P/assets $P/strands_decider
cp -P build/bin/lib*.so* $P/lib/
cp -L /usr/lib64/libgomp.so.1 /usr/lib64/libstdc++.so.6 /usr/lib64/libgcc_s.so.1 $P/lib/
gcc -O2 -shared -fPIC -o $P/lib/libdecshim.so /recipe/shim/decshim.c -I/b/include -I/b/ggml/include -L$P/lib -lllama \
  -lggml -Wl,-rpath,'$ORIGIN'
nm -D $P/lib/libdecshim.so | grep -q dec_load_phases_ns || { echo "FATAL: decshim lacks dec_load_phases_ns"; exit 1; }

# PR #77's handler and loader, the snapstart engine/app/eval mode, the baseline entry point.
cp /lambda/app/handler.py /lambda/app/loader.py $P/
cp /recipe/snapstart/engine.py /recipe/snapstart/app.py /recipe/snapstart/eval_np.py $P/
cp /recipe/baseline/lkai_entry.py /recipe/baseline/phases.py /recipe/baseline/hostinfo.py $P/
cp /assets/tokenizer.json /assets/head.safetensors /assets/head_config.json $P/assets/
cp /pkgsrc/__init__.py /pkgsrc/schema.py /pkgsrc/prompting.py /pkgsrc/hidden_engine.py $P/strands_decider/
python3.12 -m pip install -q --root-user-action=ignore --no-compile --only-binary=:all: --target $P \
  pydantic==2.14.0 numpy==2.5.3 tokenizers==0.23.2 safetensors==0.8.0
find $P -name __pycache__ -prune -exec rm -rf {} +

for so in $P/lib/*.so*; do
  [ -L "$so" ] && continue
  if LD_LIBRARY_PATH=$P/lib ldd "$so" | grep -q "not found"; then
    echo "FATAL: unresolved libraries in $so"; LD_LIBRARY_PATH=$P/lib ldd "$so"; exit 1
  fi
done
# Importing the handler would load the model: check imports and syntax instead.
(cd /tmp && PYTHONPATH=$P LD_LIBRARY_PATH=$P/lib python3.12 -c "
import ast, ctypes, engine, loader, phases, hostinfo, pydantic, numpy, tokenizers, strands_decider.hidden_engine
for f in ('handler.py', 'app.py', 'lkai_entry.py'):
    ast.parse(open('$P/' + f).read())
ctypes.CDLL('$P/lib/libdecshim.so').dec_load_phases_ns
print('imports OK', numpy.__version__, pydantic.__version__, tokenizers.__version__)")
echo "unzipped $(du -sb $P | cut -f1) bytes (Lambda limit 262144000)"

rm -f /out/function.zip
# Fixed mtimes, modes and sorted names keep the zip stable; the shared libraries are not bit-reproducible across
# builds, so build/manifest.json's sha256 identifies the deployed code rather than proving a rebuild matches it.
# The Lambda sandbox user is not root, so every file must be world-readable (head.safetensors is written 0600).
chmod -R u=rwX,go=rX $P
find $P -exec touch -h -d '2026-01-01 00:00:00' {} +
(cd $P && find . -mindepth 1 -printf '%P\n' | LC_ALL=C sort | zip -q -X -y -D /out/function.zip -@)
gcc --version | head -1 > /out/toolchain.txt
python3.12 --version >> /out/toolchain.txt
echo "function.zip $(stat -c %s /out/function.zip) bytes"
echo CONTAINER_BUILD_DONE
