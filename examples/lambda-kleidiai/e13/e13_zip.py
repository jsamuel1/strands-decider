"""Build E13's one zip from P1's published function.zip: P1's zip byte for byte, plus e13_entry.py.

  python e13_zip.py --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --out build/

Every E13 arm deploys this same zip. The arms differ only in the OpenMP wait variables in the function environment
(e13deploy.ARMS), so the zip is not a variable. Writes build/e13.zip and build/manifest.json.
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
WRAPPER = "e13_entry.py"
STAMP = (2026, 1, 1, 0, 0, 0)  # P1's container_build.sh normalises every file to this time


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def build(p1_zip: str, p1_manifest: str, out: str) -> dict:
    m1 = json.load(open(p1_manifest))
    if sha256(p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes")
    os.makedirs(out, exist_ok=True)
    wrapper = open(os.path.join(HERE, WRAPPER), "rb").read()
    path = os.path.join(out, "e13.zip")
    with zipfile.ZipFile(p1_zip) as src, zipfile.ZipFile(path + ".tmp", "w") as dst:
        if WRAPPER in src.namelist():
            sys.exit(f"{WRAPPER} already in the P1 zip")
        for info in src.infolist():
            # writestr stamps the destination offset onto the ZipInfo it is given, so copy it.
            dst.writestr(copy.copy(info), src.read(info))
        info = zipfile.ZipInfo(WRAPPER, date_time=STAMP)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info, wrapper)
    os.replace(path + ".tmp", path)
    m = {
        "schema": "lkai-e13-build-manifest/1",
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", HERE)),
        "p1_manifest": m1,
        "model": m1["model"],
        "llama_cpp": m1["llama_cpp"],
        "images": m1["images"],
        "toolchain": m1["toolchain"],
        "assets_sha256": m1["assets_sha256"],
        "wrapper_sha256": hashlib.sha256(wrapper).hexdigest(),
        "base": "P1 function.zip " + m1["function_zip"]["sha256"],
        "zip": {
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
            "handler": "e13_entry.handler",
        },
    }
    # summarize.py's cost and identity fields read build_manifest["function_zip"]
    m["function_zip"] = m["zip"] | {"base": m["base"]}
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps(m["zip"], indent=1))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "build"))
    a = ap.parse_args()
    build(a.p1_zip, a.p1_manifest, a.out)


if __name__ == "__main__":
    main()
