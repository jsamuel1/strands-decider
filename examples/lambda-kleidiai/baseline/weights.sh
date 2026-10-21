#!/bin/bash
# Runs INSIDE python:3.12-bookworm (linux/arm64). Called by build.sh with:
#   /repo    the repository checkout (ro)          /llama  llama.cpp f74e2fa6a with patch_llama.py (ro)
#   /work    build/ (rw)                            /hf     Hugging Face cache (rw)
# The measured recipe's weight pipeline (README "Layout", weights/), at pinned revisions:
#   1. snapshot the checkpoint at $CKPT_REV (its provenance.json pins the base model's revision)
#   2. weights/merge_export.py   fold the LoRA in fp32
#   3. weights/fold_hadamard.py  --block 512 (H512, the deployed weights)
#   4. convert_hf_to_gguf.py     --outtype f32 (reads hadamard_packing.json -> prism.hadamard metadata)
#   5. examples/lambda/prepare_model.py writes the head, tokenizer and head_config.json (assets/)
# build.sh then quantizes the f32 GGUF to Q8_0 with the llama-quantize container_build.sh builds.
set -euo pipefail
: "${CKPT:?}" "${CKPT_REV:?}"
export HF_HOME=/hf PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
python -m venv /venv && . /venv/bin/activate
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
# /repo is read-only and setuptools writes egg-info beside the sources: install from a copy of what the wheel needs.
rm -rf /tmp/pkg && mkdir -p /tmp/pkg && cp -a /repo/pyproject.toml /repo/README.md /repo/LICENSE /repo/src /tmp/pkg/
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -q /tmp/pkg sentencepiece "protobuf>=4.21,<5"
pip install -q /llama/gguf-py
pip freeze > /work/weights-pip-freeze.txt

CK=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$CKPT', revision='$CKPT_REV'))")
echo "checkpoint $CKPT@$CKPT_REV -> $CK"
W=/work/weights
mkdir -p $W /work/model
# Each step writes beside its output and renames it into place when complete, so an existing output is a finished
# one. build.sh deletes all of them when the stage's key changes (buildcache.py); within one key, a rerun resumes.
if [ ! -d $W/merged-2b ]; then
  rm -rf $W/merged-2b.tmp
  python /repo/examples/lambda-kleidiai/weights/merge_export.py --ckpt "$CK" --out $W/merged-2b.tmp
  mv $W/merged-2b.tmp $W/merged-2b
fi
if [ ! -d $W/merged-2b-h512 ]; then
  rm -rf $W/merged-2b-h512.tmp
  python /repo/examples/lambda-kleidiai/weights/fold_hadamard.py --src $W/merged-2b --out $W/merged-2b-h512.tmp --block 512
  mv $W/merged-2b-h512.tmp $W/merged-2b-h512
fi
if [ ! -f $W/s2b-h512-f32.gguf ]; then
  python /llama/convert_hf_to_gguf.py $W/merged-2b-h512 --outtype f32 --outfile $W/s2b-h512-f32.gguf.tmp 2>&1 | tail -5
  mv $W/s2b-h512-f32.gguf.tmp $W/s2b-h512-f32.gguf
fi
if [ ! -d /work/model/assets ]; then
  rm -rf /work/model-prep.tmp
  python /repo/examples/lambda/prepare_model.py "$CK" --out /work/model-prep.tmp
  mv /work/model-prep.tmp/assets /work/model/assets  # prepare_model's fp16 torso (merged-hf) is not used
  rm -rf /work/model-prep.tmp
fi
cp $W/merged-2b/merge_report.json /work/merge_report.json
chmod -R a+rX /work/model /work/merge_report.json /work/weights  # root writes them; the host user hashes them
ls -la $W/s2b-h512-f32.gguf /work/model/assets
echo WEIGHTS_DONE
