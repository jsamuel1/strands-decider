"""Build E8's two zips from P1's deployed function.zip and the six patched libggml-cpu variants.

  python make_zips.py --p1-zip <P1 function.zip> --p1-manifest <its build manifest> --libs <dir> --out <dir>

- base.zip: every entry of P1's zip copied byte for byte with its metadata, plus e8_entry.py and e8_host.py (the
  metadata wrapper). The `base` arm runs P1's deployed libraries.
- off.zip: base.zip with exactly the lib/libggml-cpu-*.so entries replaced by the files in --libs, built from the same
  llama.cpp checkout plus e8/patch_parallel_repack.py. Each replaced entry keeps P1's ZipInfo (name, time, mode,
  compression). Nothing else differs. The `off` and `par` arms both run off.zip; only `par` sets the switch.

KleidiAI's repack is compiled into each libggml-cpu-* variant (ggml/src/ggml-cpu/kleidiai/kleidiai.cpp), so these six
files are the only ones the E8 patch changes. libllama, libggml and libggml-base stay P1's.

Writes base.zip, off.zip and zips.json (sha256 of both zips, P1's zip, and every replaced entry before and after).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = ("e8_entry.py", "e8_host.py")
STAMP = (2026, 1, 1, 0, 0, 0)  # P1's container_build.sh normalises every file to this time
VARIANT_PREFIX = "lib/libggml-cpu-"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _entry(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.external_attr = (0o100000 | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def variant_entries(z: zipfile.ZipFile) -> list[str]:
    """The regular-file libggml-cpu-*.so entries (symlinks excluded: P1's zip stores lib links with zip -y)."""
    out = []
    for i in z.infolist():
        if not i.filename.startswith(VARIANT_PREFIX) or not i.filename.endswith(".so"):
            continue
        if ((i.external_attr >> 16) & 0o170000) == 0o120000:
            continue
        out.append(i.filename)
    return sorted(out)


def build(p1_zip: str, p1_manifest: str, libs: str, out: str, wrapper_dir: str = HERE) -> dict:
    m1 = json.load(open(p1_manifest))
    want = m1["function_zip"]["sha256"]
    if sha256(p1_zip) != want:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes ({want})")
    os.makedirs(out, exist_ok=True)
    wrappers = {n: open(os.path.join(wrapper_dir, n), "rb").read() for n in WRAPPER}
    record: dict = {"p1_zip_sha256": want, "replaced": {}, "added": {}}
    with zipfile.ZipFile(p1_zip) as src:
        names = set(src.namelist())
        if clash := sorted(names & set(WRAPPER)):
            sys.exit(f"{clash} already in the P1 zip")
        variants = variant_entries(src)
        have = sorted(f for f in os.listdir(libs) if f.startswith("libggml-cpu-") and f.endswith(".so"))
        if [os.path.basename(v) for v in variants] != have:
            sys.exit(f"--libs must hold exactly P1's variants {variants}, has {have}")
        for arm in ("base", "off"):
            path = os.path.join(out, f"{arm}.zip")
            with zipfile.ZipFile(path + ".tmp", "w") as dst:
                for info in src.infolist():
                    data = src.read(info)
                    if arm == "off" and info.filename in variants:
                        new = open(os.path.join(libs, os.path.basename(info.filename)), "rb").read()
                        record["replaced"][info.filename] = {
                            "p1_sha256": sha256_bytes(data),
                            "e8_sha256": sha256_bytes(new),
                            "e8_bytes": len(new),
                        }
                        data = new
                    dst.writestr(copy.copy(info), data)
                for n, b in wrappers.items():
                    dst.writestr(_entry(n), b)
                    record["added"][n] = sha256_bytes(b)
            os.replace(path + ".tmp", path)
            record[f"{arm}_zip"] = {"sha256": sha256(path), "bytes": os.path.getsize(path)}
    if any(r["p1_sha256"] == r["e8_sha256"] for r in record["replaced"].values()):
        sys.exit("a patched variant is byte-identical to P1's: the patch did not reach it")
    json.dump(record, open(os.path.join(out, "zips.json"), "w"), indent=1)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--libs", required=True, help="directory with the six patched libggml-cpu-*.so")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = build(a.p1_zip, a.p1_manifest, a.libs, a.out)
    print("base", r["base_zip"]["sha256"][:12], "off", r["off_zip"]["sha256"][:12], len(r["replaced"]), "replaced")


if __name__ == "__main__":
    main()
