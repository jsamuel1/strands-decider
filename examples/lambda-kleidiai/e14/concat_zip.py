"""Build E14's two Lambda zips from P1's function.zip. Standard library only.

  python concat_zip.py --p1-zip <P1 function.zip> --rows-lib <dir of libggml-cpu-*.so built with patch_concat.py> \
      --entry e14_entry.py --out build/

ctl.zip  = P1's zip, every entry byte for byte with its zip metadata, plus e14_entry.py (the host-metadata wrapper).
rows.zip = ctl.zip with only the six lib/libggml-cpu-<variant>.so entries replaced by the rows build. Every other
           entry, symlink and attribute is copied from P1's zip unchanged.
build/manifest.json records both zips' SHA-256, P1's zip SHA-256, and every replaced library's old and new SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile

LIB = re.compile(r"^lib/libggml-cpu-[a-z0-9._]+\.so$")
ENTRY = "e14_entry.py"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _copy(
    zin: zipfile.ZipFile, zout: zipfile.ZipFile, info: zipfile.ZipInfo, data: bytes | None = None
) -> None:
    out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    out.compress_type = info.compress_type
    out.external_attr = info.external_attr
    out.create_system = info.create_system
    zout.writestr(out, zin.read(info) if data is None else data)


def build(p1_zip: str, rows_lib: str, entry: str, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    entry_bytes = open(entry, "rb").read()
    replaced: dict[str, dict] = {}
    with zipfile.ZipFile(p1_zip) as zin:
        names = [i.filename for i in zin.infolist()]
        if ENTRY in names:
            raise SystemExit(f"{ENTRY} is already in {p1_zip}")
        libs = [n for n in names if LIB.match(n)]
        have = sorted(
            f for f in os.listdir(rows_lib) if f.startswith("libggml-cpu-") and f.endswith(".so")
        )
        if sorted(os.path.basename(n) for n in libs) != have:
            raise SystemExit(f"variant set differs: zip {sorted(libs)} vs {rows_lib} {have}")
        for arm in ("ctl", "rows"):
            path = os.path.join(out, f"{arm}.zip")
            with zipfile.ZipFile(path, "w") as zout:
                for info in zin.infolist():
                    if arm == "rows" and LIB.match(info.filename):
                        new = open(
                            os.path.join(rows_lib, os.path.basename(info.filename)), "rb"
                        ).read()
                        replaced[info.filename] = {
                            "p1_sha256": hashlib.sha256(zin.read(info)).hexdigest(),
                            "rows_sha256": hashlib.sha256(new).hexdigest(),
                        }
                        _copy(zin, zout, info, new)
                    else:
                        _copy(zin, zout, info)
                e = zipfile.ZipInfo(ENTRY, date_time=(2026, 1, 1, 0, 0, 0))
                e.compress_type = zipfile.ZIP_DEFLATED
                e.external_attr = 0o100644 << 16
                e.create_system = 3
                zout.writestr(e, entry_bytes)
    m = {
        "schema": "lkai-e14-zips/1",
        "p1_zip_sha256": sha256(p1_zip),
        "entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
        "zips": {
            arm: {"sha256": sha256(os.path.join(out, f"{arm}.zip"))} for arm in ("ctl", "rows")
        },
        "replaced_in_rows": replaced,
    }
    with open(os.path.join(out, "zips.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--rows-lib", required=True)
    ap.add_argument(
        "--entry", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ENTRY)
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.p1_zip, a.rows_lib, a.entry, a.out)["zips"]))


if __name__ == "__main__":
    main()
