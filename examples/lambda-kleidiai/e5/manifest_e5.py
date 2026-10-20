"""Write <e5-build>/manifest.json and <e5-build>/e5-manifest.json.

  python manifest_e5.py <e5-build> <p1-build-manifest> [<p1-published-build-manifest>]

manifest.json is the build manifest deploy.py and bench.py read. When P1's published run's manifest (results/<run>/
build-manifest.json) names the same function.zip and GGUF sha256 as the local build, that one is used, so E5's
baseline arm is provably the code and weights P1 measured; otherwise the local build's manifest is.
e5-manifest.json records both weight files, the conversion and the tensor-identity proof, read from
<e5-build>/model/convert.json (container_convert.sh's record).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main(b: str, local_manifest: str, published: str | None = None) -> None:
    m = json.load(open(local_manifest))
    zip_sha = sha(os.path.join(b, "function.zip"))
    if zip_sha != m["function_zip"]["sha256"]:
        sys.exit(f"{b}/function.zip is not the one {local_manifest} names")
    if published and os.path.exists(published):
        p = json.load(open(published))
        if (p["function_zip"]["sha256"], p["model"]["gguf_sha256"]) == (
            m["function_zip"]["sha256"],
            m["model"]["gguf_sha256"],
        ):
            m = p | {"source": os.path.relpath(published, os.path.dirname(os.path.dirname(b)))}
    if os.path.lexists(os.path.join(b, "manifest.json")):
        os.remove(os.path.join(b, "manifest.json"))
    json.dump(m, open(os.path.join(b, "manifest.json"), "w"), indent=1)
    conv = json.load(open(os.path.join(b, "model", "convert.json")))
    base = os.path.join(b, "model", "s2b-h512-Q8_0.gguf")
    new = os.path.join(b, "model", "s2b-h512-Q8_0-novocab.gguf")
    base_sha = sha(base)
    if base_sha != m["model"]["gguf_sha256"]:
        sys.exit(f"{base} is not the baseline build's GGUF")
    if not conv["tensors"]["identical"]:
        sys.exit("converted GGUF's tensors differ from the baseline's")
    out = {
        "schema": "lkai-e5-manifest/1",
        "baseline": {
            "file": os.path.basename(base),
            "format": "GGUF v3 Q8_0, tokenizer.ggml.model=gpt2 (BPE vocab and merges embedded)",
            "bytes": os.path.getsize(base),
            "sha256": base_sha,
            "data_offset": conv["tensors"]["data_offset"][0],
        },
        "converted": {
            "file": os.path.basename(new),
            "format": "GGUF v3 Q8_0, tokenizer.ggml.model=none "
            "(LLAMA_VOCAB_TYPE_NONE), qwen35.vocab_size=248320",
            "bytes": os.path.getsize(new),
            "sha256": sha(new),
            "data_offset": conv["tensors"]["data_offset"][1],
        },
        "conversion": {
            "command": "python examples/lambda-kleidiai/e5/convert_no_vocab.py s2b-h512-Q8_0.gguf "
            "s2b-h512-Q8_0-novocab.gguf",
            "container": conv["image"],
            "tools": conv["tools"],
            "removed_keys": conv["removed_keys"],
            "added_keys": conv["added_keys"],
        },
        "tensors": conv["tensors"],
    }
    out["size_delta_bytes"] = out["converted"]["bytes"] - out["baseline"]["bytes"]
    json.dump(out, open(os.path.join(b, "e5-manifest.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("baseline", "converted", "size_delta_bytes")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
