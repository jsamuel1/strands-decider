"""Compare two GGUFs tensor by tensor (names, order, types, shapes, sha256 of each tensor's bytes) and list the
metadata keys that differ. Prints a JSON record; exit 1 if any tensor differs.

  python tensor_identity.py <a.gguf> <b.gguf>
"""

from __future__ import annotations

import hashlib
import json
import sys

import gguf


def table(p: str) -> tuple[gguf.GGUFReader, dict, str]:
    r = gguf.GGUFReader(p)
    out, h_all = {}, hashlib.sha256()
    for t in r.tensors:
        h = hashlib.sha256(t.data.tobytes()).hexdigest()
        h_all.update(t.name.encode() + b"\0" + h.encode())
        out[t.name] = [t.tensor_type.name, [int(x) for x in t.shape], h]
    return r, out, h_all.hexdigest()


def main(pa: str, pb: str) -> int:
    ra, a, ha = table(pa)
    rb, b, hb = table(pb)
    ka, kb = set(ra.fields), set(rb.fields)
    rec = {
        "tensors": {
            "count": [len(a), len(b)],
            "order_same": list(a) == list(b),
            "identical": a == b and list(a) == list(b),
            "differing": [k for k in a if a[k] != b.get(k)][:10],
            "digest": [ha, hb],
            "digest_method": "sha256 over name NUL sha256(tensor bytes), in file order",
            "data_offset": [int(ra.data_offset), int(rb.data_offset)],
            "alignment": [int(ra.alignment), int(rb.alignment)],
        },
        "removed_keys": sorted(ka - kb),
        "added_keys": {k: rb.fields[k].contents() for k in sorted(kb - ka)},
        "changed_keys": sorted(
            k
            for k in ka & kb
            if not k.startswith("GGUF.") and ra.fields[k].contents() != rb.fields[k].contents()
        ),
    }
    print(json.dumps(rec, indent=1))
    return 0 if rec["tensors"]["identical"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
