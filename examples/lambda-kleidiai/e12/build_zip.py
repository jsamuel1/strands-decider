"""Build E12's function zip from P1's function.zip. Both E12 arms deploy this one zip.

  python build_zip.py --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --shim <E12 libdecshim.so> \
      --out build/

Writes build/e12.zip and build/manifest.json. e12.zip is every entry of P1's zip copied byte for byte with its
metadata, except lib/libdecshim.so, which is replaced by E12's shim (shim/decshim.c, compiled with P1's command in
P1's pinned build image against P1's own llama.cpp build), plus e12_entry.py (the metadata wrapper). The arms differ
only in the environment: `dor` leaves DEC_CLEAR_DATA unset (P1's full clear, the dormant switch), `e12` sets
DEC_CLEAR_DATA=0 (metadata-only clear).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = "e12_entry.py"
SHIM = "lib/libdecshim.so"
STAMP = (2026, 1, 1, 0, 0, 0)  # P1's container_build.sh normalises every file to this time
SCHEMA = "lkai-e12-build-manifest/1"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def _entry(name: str, like: zipfile.ZipInfo | None = None, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.external_attr = like.external_attr if like is not None else mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


# Every file the e12 zip or its arms' behaviour depends on. A difference from the commit makes the build
# unreproducible from that commit, so it is listed in the manifest (P1's dirty_inputs contract) and refused.
INPUTS = (
    "examples/lambda-kleidiai/shim/decshim.c",
    "examples/lambda-kleidiai/e12/e12_entry.py",
    "examples/lambda-kleidiai/e12/build_zip.py",
    "examples/lambda-kleidiai/e12/build_shim.sh",
    "examples/lambda-kleidiai/e12/e12deploy.py",
)


def dirty_inputs() -> list[str]:
    root = git("rev-parse", "--show-toplevel")
    out = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *[os.path.join(root, p) for p in INPUTS],
    )
    return sorted(line[3:] for line in out.splitlines() if line.strip())


def build_zip(p1_zip: str, shim: bytes, wrapper: bytes, out_path: str) -> dict:
    """Copy P1's zip, swapping the shim and adding the wrapper. Returns what changed."""
    with zipfile.ZipFile(p1_zip) as src, zipfile.ZipFile(out_path + ".tmp", "w") as dst:
        names = src.namelist()
        if SHIM not in names:
            raise SystemExit(f"{SHIM} not in {p1_zip}")
        if WRAPPER in names:
            raise SystemExit(f"{WRAPPER} already in {p1_zip}")
        old = None
        for info in src.infolist():
            if info.filename == SHIM:
                old = src.read(info)
                dst.writestr(_entry(SHIM, like=info), shim)
                continue
            # writestr stamps the destination offset onto the ZipInfo it is given: copy it.
            dst.writestr(copy.copy(info), src.read(info))
        dst.writestr(_entry(WRAPPER), wrapper)
    os.replace(out_path + ".tmp", out_path)
    return {
        "replaced": {SHIM: {"p1_sha256": sha256_bytes(old), "e12_sha256": sha256_bytes(shim)}},
        "added": {WRAPPER: sha256_bytes(wrapper)},
        "entries": len(names) + 1,
    }


def unchanged_entries(p1_zip: str, e12_zip: str) -> int:
    """Count of entries identical (bytes and metadata) in both zips; raises if anything else differs."""
    with zipfile.ZipFile(p1_zip) as a, zipfile.ZipFile(e12_zip) as b:
        ia = {i.filename: i for i in a.infolist()}
        ib = {i.filename: i for i in b.infolist()}
        extra = set(ib) - set(ia)
        if extra != {WRAPPER} or set(ia) - set(ib):
            raise SystemExit(f"entry sets differ: +{sorted(extra)} -{sorted(set(ia) - set(ib))}")
        same = 0
        for n, x in ia.items():
            y = ib[n]
            if n == SHIM:
                continue
            if (
                a.read(x) != b.read(y)
                or x.external_attr != y.external_attr
                or x.date_time != y.date_time
            ):
                raise SystemExit(f"{n} differs between the zips")
            same += 1
        return same


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument(
        "--shim-build", help="JSON with the shim's compiler and build inputs, recorded verbatim"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m1 = json.load(open(a.p1_manifest))
    if sha256(a.p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{a.p1_zip} is not the zip {a.p1_manifest} describes")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "e12.zip")
    shim = open(a.shim, "rb").read()
    changes = build_zip(a.p1_zip, shim, open(os.path.join(HERE, WRAPPER), "rb").read(), out)
    same = unchanged_entries(a.p1_zip, out)
    src = os.path.join(os.path.dirname(HERE), "shim", "decshim.c")
    dirty = dirty_inputs()
    m = {
        "schema": SCHEMA,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(dirty),
        "dirty_inputs": dirty,
        "p1_manifest": m1,
        "model": m1["model"],
        "llama_cpp": m1["llama_cpp"],
        "images": m1["images"],
        "toolchain": m1.get("toolchain"),
        "assets_sha256": m1.get("assets_sha256"),
        "base": {"p1_function_zip": m1["function_zip"]},
        "shim": {
            "source": "examples/lambda-kleidiai/shim/decshim.c",
            "source_sha256": sha256(src),
            "so_sha256": sha256_bytes(shim),
            "build": json.load(open(a.shim_build)) if a.shim_build else None,
        },
        "changes": changes,
        "unchanged_entries": same,
        "zips": {
            "e12": {
                "bytes": os.path.getsize(out),
                "sha256": sha256(out),
                "handler": "e12_entry.handler",
            }
        },
        # P1's summarize.py reads the deployed zip from function_zip.
        "function_zip": {
            "bytes": os.path.getsize(out),
            "sha256": sha256(out),
            "handler": "e12_entry.handler",
            "source_commit": git("rev-parse", "HEAD"),
        },
        "arms": {
            "dor": {"DEC_CLEAR_DATA": None, "meaning": "P1's full clear (dormant switch)"},
            "e12": {"DEC_CLEAR_DATA": "0", "meaning": "metadata-only clear"},
        },
    }
    json.dump(m, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    if dirty:
        print("WARNING: build inputs differ from", m["repo_commit"], dirty)
    print(
        "e12.zip",
        m["zips"]["e12"]["sha256"][:12],
        m["zips"]["e12"]["bytes"],
        "bytes;",
        same,
        "entries unchanged",
    )


if __name__ == "__main__":
    main()
