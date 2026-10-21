"""Summarise folded perf stacks (stackcollapse.py output) into hot-spot shares for T0.

    python perf_summary.py perf-steady.folded perf-init.folded --out perf-summary.json

Shares are of ALL samples in the file (the full denominator). `leaf` is self time per symbol; `self_class` groups
leaf samples: allocator (glibc malloc/free and their helpers, memcpy/memmove), tokenizers (Rust code in
tokenizers.abi3.so), python (libpython), other. `inclusive` counts each stack at most once per pattern, so a pattern
matched by several frames of one stack is not double counted.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

ALLOC = (
    "malloc",
    "_int_free",
    "__free",
    "unlink_chunk",
    "realloc",
    "calloc",
    "cfree",
    "free_perturb",
    "tcache",
    "memcpy",
    "memmove",
    "memset",
    "malloc_consolidate",
    "sysmalloc",
    "brk",
    "mmap",
    "munmap",
)
INCLUSIVE = {
    "python_encode": "__pymethod_encode__",
    "python_encode_batch": "__pymethod_encode_batch__",
    "python_decode": "__pymethod_decode__",
    "from_file": "from_file",
    "normalizer_nfc": "Recompositions",
    "normalize_any": "NormalizedString>::transform",
    "pre_tokenizer_split_regex": "onig",
    "byte_level": "ByteLevel",
    "bpe_tokenize_with_cache": "tokenize_with_cache",
    "bpe_merge_word": "merge_word",
    "offsets_bytes_to_char": "BytesToCharOffsetConverter",
    "into_encoding": "into_encoding",
    "encoding_merge": "Encoding>::merge",
    "rayon": "rayon",
    "serde_json": "serde_json",
    "hashmap_insert_or_rehash": "hashbrown",
    "python_interpreter": "_PyEval_EvalFrameDefault",
}


def classify_leaf(frame: str) -> str:
    sym, _, dso = frame.rpartition(" (")
    dso = dso.rstrip(")")
    if any(k in sym for k in ALLOC):
        return "allocator"
    if dso.startswith("tokenizers"):
        return "tokenizers"
    if dso.startswith("libpython"):
        return "python"
    if dso == "inlined":
        return "inlined-other"
    return "other"


def summarise(path: Path, top: int = 30) -> dict[str, Any]:
    leaf: collections.Counter[str] = collections.Counter()
    cls: collections.Counter[str] = collections.Counter()
    inc: collections.Counter[str] = collections.Counter()
    tot = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            stack, _, n_s = line.rstrip("\n").rpartition(" ")
            n = int(n_s)
            frames = stack.split(";")
            tot += n
            leaf[frames[-1]] += n
            cls[classify_leaf(frames[-1])] += n
            for name, pat in INCLUSIVE.items():
                if any(pat in fr for fr in frames):
                    inc[name] += n
    return {
        "samples": tot,
        "self_class": {k: {"samples": v, "share": v / tot} for k, v in cls.most_common()},
        "leaf": [
            {"frame": k[:240], "samples": v, "share": v / tot} for k, v in leaf.most_common(top)
        ],
        "inclusive": {k: {"samples": v, "share": v / tot} for k, v in inc.most_common()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folded", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {p.stem: summarise(p) for p in a.folded}
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, r in res.items():
        print(f"== {name}: {r['samples']} samples")
        for k, v in r["self_class"].items():
            print(f"  self {k:14s} {v['share'] * 100:6.2f}%")
        for k, v in r["inclusive"].items():
            print(f"  incl {k:26s} {v['share'] * 100:6.2f}%")


if __name__ == "__main__":
    main()
