"""E5 conversion: rewrite the baseline GGUF without its text tokenizer, leaving every tensor byte unchanged.

  python convert_no_vocab.py <in.gguf> <out.gguf>

The deployed engine tokenizes in Python (assets/tokenizer.json) and passes token ids to decshim; llama.cpp's own
vocabulary is never used for tokenization, and with the prefill patch it never computes logits over it. The
baseline GGUF still carries the gpt2/qwen35 BPE vocabulary (248,320 tokens, 247,587 merges, ~11 MB of metadata),
which llama_model_load_from_file parses into hash maps before it maps a single tensor.

This writes tokenizer.ggml.model = "none" (llama.cpp's LLAMA_VOCAB_TYPE_NONE) and qwen35.vocab_size = 248320 (the
loader's n_vocab for the token_embd shape), drops the other tokenizer.* keys, and copies the remaining metadata,
the tensor infos (same names, shapes, types and order) and the tensor data byte for byte at the same alignment.
Uses gguf-py from the pinned llama.cpp checkout (f74e2fa6a, gguf-py 0.19.0), the same as gguf_new_metadata.py.
"""

from __future__ import annotations

import sys

import gguf


def main(src: str, dst: str) -> None:
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    align = r.fields.get("general.alignment")
    if align is not None:
        w.data_alignment = align.contents()
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        if f.name.startswith("tokenizer."):
            continue
        t = f.types[0]
        sub = f.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), t, sub_type=sub)
    w.add_key_value("tokenizer.ggml.model", "none", gguf.GGUFValueType.STRING)
    w.add_key_value(f"{arch}.vocab_size", n_vocab, gguf.GGUFValueType.UINT32)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(
        f"wrote {dst}: {len(r.tensors)} tensors, vocab_size {n_vocab}, alignment {w.data_alignment}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
