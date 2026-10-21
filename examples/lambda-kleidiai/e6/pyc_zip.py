"""Build E6's two zips from P1's function.zip. The only difference between them is the shipped bytecode.

  python pyc_zip.py --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --out build/

Writes build/ctl.zip, build/pyc.zip and build/manifest.json.

- ctl.zip: every entry of P1's zip copied byte for byte with its metadata, plus e6_entry.py (the metadata wrapper,
  no bytecode). P1's 149 .pyc stay as P1 ships them. Their source-mtime headers name the build time, but the zip
  stamps every .py with 2026-01-01, so Python treats each one as stale. Most .py files have no .pyc at all.
- pyc.zip: ctl.zip, with every P1 .pyc replaced and a .pyc added for every .py. The bytecode is compiled by
  py_compile with PycInvalidationMode.UNCHECKED_HASH: the loader uses it without reading or hashing the source, so
  the read-only, mtime-normalised /var/task cannot make it stale. e6_entry.py is identical in both zips and gets no
  bytecode in either.

The compiling interpreter must be CPython 3.12 (the same bytecode magic number as Lambda's python3.12). A .py that
does not compile is left without bytecode and listed in the manifest. That matches what Lambda does with it today.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = "e6_entry.py"
TAG = sys.implementation.cache_tag  # cpython-312
STAMP = (2026, 1, 1, 0, 0, 0)  # P1's container_build.sh normalises every file to this time


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pyc_name(py: str) -> str:
    """The __pycache__ path Python 3.12 looks up for a source path inside the package root."""
    d, f = os.path.split(py)
    return os.path.join(d, "__pycache__", f"{f[:-3]}.{TAG}.pyc").lstrip("/")


def _entry(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def compile_tree(src: zipfile.ZipFile, workdir: str) -> tuple[dict[str, bytes], list[str]]:
    """Unchecked-hash bytecode for every .py in the zip, keyed by its __pycache__ path; and the .py that failed."""
    out: dict[str, bytes] = {}
    failed: list[str] = []
    for info in src.infolist():
        name = info.filename
        if info.is_dir() or not name.endswith(".py"):
            continue
        path = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(src.read(info))
        cfile = os.path.join(workdir, "__out__", pyc_name(name))
        try:
            py_compile.compile(
                path,
                cfile=cfile,
                dfile="/var/task/" + name,  # the co_filename a compile on Lambda would record
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
        except py_compile.PyCompileError:
            failed.append(name)
            continue
        with open(cfile, "rb") as f:
            out[pyc_name(name)] = f.read()
    return out, sorted(failed)


def stale_pyc(src: zipfile.ZipFile) -> dict[str, int]:
    """P1's own .pyc, classified the way the 3.12 loader sees them on Lambda (source path, size and mtime)."""
    names = {i.filename: i for i in src.infolist()}
    counts = {"pyc": 0, "valid": 0, "stale": 0, "no_source": 0, "py": 0, "py_without_pyc": 0}
    covered = set()
    for name, info in names.items():
        if name.endswith(".py"):
            counts["py"] += 1
        if not name.endswith(".pyc") or "/__pycache__/" not in "/" + name:
            continue
        counts["pyc"] += 1
        d, _, f = name.rpartition("__pycache__/")
        py = d + f.split(".", 1)[0] + ".py"
        covered.add(py)
        if py not in names:
            counts["no_source"] += 1
            continue
        head = src.read(info)[:16]
        flags = int.from_bytes(head[4:8], "little")
        if flags & 1:  # hash-based: valid regardless of mtime
            counts["valid"] += 1
            continue
        mtime = int.from_bytes(head[8:12], "little")
        size = int.from_bytes(head[12:16], "little")
        s = names[py]
        zip_mtime = calendar.timegm(
            (*s.date_time, 0, 0, -1)
        )  # Lambda extracts with the zip's (UTC) time
        ok = size == (s.file_size & 0xFFFFFFFF) and abs(mtime - zip_mtime) < 2
        counts["valid" if ok else "stale"] += 1
    counts["py_without_pyc"] = sum(1 for n in names if n.endswith(".py") and n not in covered)
    return counts


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def build(p1_zip: str, p1_manifest: str, out: str) -> dict:
    m1 = json.load(open(p1_manifest))
    if sha256(p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes")
    if sys.version_info[:2] != (3, 12):
        sys.exit(f"needs CPython 3.12 (Lambda python3.12's bytecode), not {sys.version.split()[0]}")
    os.makedirs(out, exist_ok=True)
    wrapper = open(os.path.join(HERE, WRAPPER), "rb").read()
    with zipfile.ZipFile(p1_zip) as src, tempfile.TemporaryDirectory() as tmp:
        if WRAPPER in src.namelist():
            sys.exit(f"{WRAPPER} already in the P1 zip")
        before = stale_pyc(src)
        compiled, failed = compile_tree(src, tmp)
        for arm in ("ctl", "pyc"):
            path = os.path.join(out, f"{arm}.zip")
            with zipfile.ZipFile(path + ".tmp", "w") as dst:
                seen = set()
                for info in src.infolist():
                    if arm == "pyc" and info.filename in compiled:
                        continue  # replaced below
                    # writestr stamps the destination offset onto the ZipInfo it is given, so copy it: the
                    # source's own infolist must stay intact for the next zip.
                    dst.writestr(copy.copy(info), src.read(info))
                    seen.add(info.filename)
                dst.writestr(_entry(WRAPPER), wrapper)
                if arm == "pyc":
                    for name in sorted(compiled):
                        d = name.rsplit("/", 1)[0] + "/" if "/" in name else None
                        if d and d not in seen:
                            dst.writestr(_entry(d, 0o40755), b"")
                            seen.add(d)
                        dst.writestr(_entry(name), compiled[name])
            os.replace(path + ".tmp", path)
        with zipfile.ZipFile(os.path.join(out, "pyc.zip")) as z:
            after = stale_pyc(z)
            unzipped = sum(i.file_size for i in z.infolist())
    m = {
        "schema": "lkai-e6-build-manifest/1",
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", HERE)),
        "p1_manifest": m1,
        "model": m1["model"],
        "llama_cpp": m1["llama_cpp"],
        "images": m1["images"],
        "toolchain": m1["toolchain"],
        "assets_sha256": m1["assets_sha256"],
        "compiler": {
            "python": sys.version.split()[0],
            "magic": importlib.util.MAGIC_NUMBER.hex(),
            "invalidation_mode": "unchecked-hash",
        },
        "bytecode": {"p1": before, "e6_pyc": after, "compiled": len(compiled), "failed": failed},
        "wrapper_sha256": hashlib.sha256(wrapper).hexdigest(),
        "zips": {
            arm: {
                "bytes": os.path.getsize(os.path.join(out, f"{arm}.zip")),
                "sha256": sha256(os.path.join(out, f"{arm}.zip")),
                "handler": "e6_entry.handler",
            }
            for arm in ("ctl", "pyc")
        },
        "pyc_unzipped_bytes": unzipped,
        "base": "P1 function.zip " + m1["function_zip"]["sha256"],
    }
    # summarize.py's cost and identity fields read build_manifest["function_zip"]
    m["function_zip"] = m["zips"]["pyc"] | {"base": m["base"]}
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: m[k] for k in ("bytecode", "zips", "pyc_unzipped_bytes")}, indent=1))
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
