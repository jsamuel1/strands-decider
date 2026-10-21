"""Build E10's one zip: P1's deployed function.zip, every entry byte for byte with its metadata, plus e10_entry.py.

  python make_zip.py --p1-zip p1.zip --p1-manifest p1-manifest.json --out build

Every E10 arm deploys this same zip; the arms differ only in $E10_N_CTX. Writes build/function.zip and
build/manifest.json (lkai-e10-build/1: P1's zip and manifest digests, the wrapper's digest, the output digest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = "e10_entry.py"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(p1_zip: str, wrapper: str, out_zip: str) -> None:
    """Copy every P1 entry (raw bytes, ZipInfo metadata) and append the wrapper with a fixed timestamp."""
    with zipfile.ZipFile(p1_zip) as src:
        names = src.namelist()
        if WRAPPER in names:
            raise SystemExit(f"{p1_zip} already contains {WRAPPER}")
        with zipfile.ZipFile(out_zip, "w") as dst:
            for info in src.infolist():
                dst.writestr(info, src.read(info), compress_type=info.compress_type)
            zi = zipfile.ZipInfo(WRAPPER, date_time=(2026, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(zi, open(wrapper, "rb").read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-zip", required=True)
    ap.add_argument("--p1-manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out_zip = os.path.join(a.out, "function.zip")
    wrapper = os.path.join(HERE, WRAPPER)
    build(a.p1_zip, wrapper, out_zip)
    p1m = json.load(open(a.p1_manifest))
    try:
        commit = subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", HERE, "status", "--porcelain", "--", "."], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", ["git unavailable"]
    m = {
        "schema": "lkai-e10-build/1",
        "repo_commit": commit,
        "repo_dirty": bool(dirty),
        "dirty_inputs": dirty,
        "base": {
            "p1_zip_sha256": sha256(a.p1_zip),
            "p1_manifest_sha256": sha256(a.p1_manifest),
            "p1_repo_commit": p1m.get("repo_commit"),
        },
        "wrapper_sha256": sha256(wrapper),
        "zips": {"e10": {"sha256": sha256(out_zip), "bytes": os.path.getsize(out_zip)}},
        "function_zip": {
            "sha256": sha256(out_zip),
            "bytes": os.path.getsize(out_zip),
            "handler": "e10_entry.handler",
        },
        "model": p1m.get("model"),
        "images": p1m.get("images"),
        "toolchain": p1m.get("toolchain"),
        "assets_sha256": p1m.get("assets_sha256"),
        "llama_cpp": p1m.get("llama_cpp"),
    }
    json.dump(m, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("function.zip", m["zips"]["e10"]["sha256"][:12], m["zips"]["e10"]["bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
