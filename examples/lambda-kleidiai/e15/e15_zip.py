"""Build E15's two zips from P1's published function.zip.

  python e15_zip.py --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --cand-libs <dir> --out build/

- ctl.zip  : P1's zip byte for byte (every entry and its metadata), plus e15_entry.py.
- cand.zip : ctl.zip with the six lib/libggml-cpu-*.so replaced by the candidate-A build in --cand-libs
             (llama.cpp f74e2fa6a + llama/patch_llama.py + e15/patch_a.py, P1's image and flags; e15/build_cand.sh).
             The replaced entries keep P1's ZipInfo (name, time, mode, compression); only the bytes differ.

Arms (e15deploy.ARMS): ctl runs ctl.zip; off runs cand.zip with LKAI_E15_A unset; a runs the same cand.zip with
LKAI_E15_A=1. off - ctl isolates the dormant-build effect, a - off the switch. Writes build/{ctl,cand}.zip and
build/manifest.json with every changed entry's before/after SHA-256.
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
WRAPPER = "e15_entry.py"
STAMP = (2026, 1, 1, 0, 0, 0)  # P1's container_build.sh normalises every file to this time
VARIANTS = tuple(
    f"lib/libggml-cpu-{v}.so"
    for v in ("armv8.0_1", "armv8.2_1", "armv8.2_2", "armv8.2_3", "armv8.6_1", "armv8.6_2")
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def _write(src: zipfile.ZipFile, path: str, wrapper: bytes, replace: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path + ".tmp", "w") as dst:
        for info in src.infolist():
            data = replace.get(info.filename, None)
            # writestr stamps the destination offset onto the ZipInfo it is given, so copy it.
            dst.writestr(copy.copy(info), src.read(info) if data is None else data)
        info = zipfile.ZipInfo(WRAPPER, date_time=STAMP)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info, wrapper)
    os.replace(path + ".tmp", path)


def build(p1_zip: str, p1_manifest: str, cand_libs: str, out: str) -> dict:
    m1 = json.load(open(p1_manifest))
    if sha256(p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes")
    os.makedirs(out, exist_ok=True)
    wrapper = open(os.path.join(HERE, WRAPPER), "rb").read()
    replace = {}
    for name in VARIANTS:
        p = os.path.join(cand_libs, os.path.basename(name))
        if not os.path.isfile(p):
            sys.exit(f"--cand-libs has no {os.path.basename(name)}")
        replace[name] = open(p, "rb").read()
    if b"lkai_e15_set_a" not in replace["lib/libggml-cpu-armv8.6_1.so"]:
        sys.exit("candidate armv8.6_1 lacks candidate A (lkai_e15_set_a)")
    for b in replace.values():
        if b"lkai_np_" in b or b"lkai_gdn_rowblock" in b:
            sys.exit(
                "candidate libraries carry the node profiler or candidate B: build with patch_a.py only"
            )
    changed = {}
    with zipfile.ZipFile(p1_zip) as src:
        names = src.namelist()
        if WRAPPER in names:
            sys.exit(f"{WRAPPER} already in the P1 zip")
        if missing := [n for n in VARIANTS if n not in names]:
            sys.exit(f"P1 zip lacks {missing}")
        for n in VARIANTS:
            before = hashlib.sha256(src.read(n)).hexdigest()
            after = hashlib.sha256(replace[n]).hexdigest()
            changed[n] = {"p1_sha256": before, "cand_sha256": after}
        _write(src, os.path.join(out, "ctl.zip"), wrapper, {})
        _write(src, os.path.join(out, "cand.zip"), wrapper, replace)
    zips = {
        z: {
            "bytes": os.path.getsize(os.path.join(out, f"{z}.zip")),
            "sha256": sha256(os.path.join(out, f"{z}.zip")),
            "handler": "e15_entry.handler",
        }
        for z in ("ctl", "cand")
    }
    cand_meta = {}
    for f in ("toolchain.txt", "patch_a.sha256"):
        p = os.path.join(cand_libs, f)
        if os.path.isfile(p):
            cand_meta[f] = open(p).read().strip()
    m = {
        "schema": "lkai-e15-build-manifest/1",
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", HERE)),
        "p1_manifest": m1,
        "model": m1["model"],
        "llama_cpp": m1["llama_cpp"]
        | {"e15_patch": "e15/patch_a.py (candidate A, LKAI_E15_A) on cand.zip only"},
        "images": m1["images"],
        "toolchain": m1["toolchain"],
        "assets_sha256": m1["assets_sha256"],
        "wrapper_sha256": hashlib.sha256(wrapper).hexdigest(),
        "base": "P1 function.zip " + m1["function_zip"]["sha256"],
        "zips": zips,
        "changed_entries": changed,
        "cand_build": cand_meta,
    }
    # summarize.py's cost and identity fields read build_manifest["function_zip"] (ctl's zip; cand differs only in
    # the six CPU variant libraries listed in changed_entries)
    m["zip"] = zips["ctl"]
    m["function_zip"] = zips["ctl"] | {"base": m["base"]}
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps(zips, indent=1))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--cand-libs", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "build"))
    a = ap.parse_args()
    build(a.p1_zip, a.p1_manifest, a.cand_libs, a.out)


if __name__ == "__main__":
    main()
