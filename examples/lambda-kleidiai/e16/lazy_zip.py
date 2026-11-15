"""Build E16's two zips from P1's function.zip. The only difference between them is where the model stack is imported.

  python3.12 lazy_zip.py --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --out build/

Writes build/e6.zip, build/lazy.zip and build/manifest.json.

Both zips are P1's zip, entry for entry, with the same changes in each:
- the harness: baseline/lkai_entry.py at this commit (it patches engine on engine's first import instead of importing
  engine itself, so it leaves import placement to the code it measures), H1's baseline/hostinfo.py, and the wrapper
  e16_entry.py;
- E6's bytecode change: py_compile UNCHECKED_HASH bytecode for every .py, replacing P1's stale .pyc. E16's baseline
  is E6's result, so both arms have it.
Then lazy.zip, and only lazy.zip, takes this commit's examples/lambda/app/handler.py and
examples/lambda-kleidiai/snapstart/app.py (engine imported inside the load, beside the download). e6.zip keeps P1's
handler.py and app.py (engine, and with it numpy, pydantic, tokenizers and strands_decider, imported at module level
on the init's main thread, before the load starts). The tests pin that every other entry is identical between the two.

The compiling interpreter must be CPython 3.12 (Lambda python3.12's bytecode magic number).
"""

# ruff: noqa: SLF001
# (Reuses E6's zip-entry helper pyc_zip._entry, so both experiments stamp entries identically.)
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
KLEIDIAI = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(KLEIDIAI))
sys.path.insert(0, os.path.join(KLEIDIAI, "e6"))
import pyc_zip  # noqa: E402  (E6's compile_tree, stale_pyc, sha256, _entry)

ARMS = ("e6", "lazy")
WRAPPER = "e16_entry.py"
# zip entry -> repo source. HARNESS goes into both zips; PLACEMENT (the variable) into lazy.zip only.
HARNESS = {
    "lkai_entry.py": "examples/lambda-kleidiai/baseline/lkai_entry.py",
    "hostinfo.py": "examples/lambda-kleidiai/baseline/hostinfo.py",
    WRAPPER: "examples/lambda-kleidiai/e16/e16_entry.py",
}
PLACEMENT = {
    "handler.py": "examples/lambda/app/handler.py",
    "app.py": "examples/lambda-kleidiai/snapstart/app.py",
}


def read(rel: str) -> bytes:
    with open(os.path.join(REPO, rel), "rb") as f:
        return f.read()


def entry_sha256(zip_path: str, name: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        return hashlib.sha256(z.read(name)).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def arm_sources(src: zipfile.ZipFile, arm: str) -> dict[str, bytes]:
    """Every file entry of one arm's zip (before bytecode), keyed by name."""
    files = {i.filename: src.read(i) for i in src.infolist() if not i.is_dir()}
    if arm == "e6" and any(files.get(n) is None for n in PLACEMENT):
        sys.exit(f"P1 zip lacks one of {sorted(PLACEMENT)}")
    for name, rel in HARNESS.items():
        files[name] = read(rel)
    if arm == "lazy":
        for name, rel in PLACEMENT.items():
            files[name] = read(rel)
    return files


def build(p1_zip: str, p1_manifest: str, out: str) -> dict:
    m1 = json.load(open(p1_manifest))
    if pyc_zip.sha256(p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes")
    if sys.version_info[:2] != (3, 12):
        sys.exit(f"needs CPython 3.12 (Lambda python3.12's bytecode), not {sys.version.split()[0]}")
    os.makedirs(out, exist_ok=True)
    zips: dict = {}
    with zipfile.ZipFile(p1_zip) as src:
        if WRAPPER in src.namelist() or "hostinfo.py" in src.namelist():
            sys.exit("the P1 zip already holds an E16 harness file")
        p1_bytecode = pyc_zip.stale_pyc(src)
        p1_infos = {i.filename: i for i in src.infolist()}
        for arm in ARMS:
            files = arm_sources(src, arm)
            with tempfile.TemporaryDirectory() as tmp:
                # Compile from a zip holding exactly this arm's sources, so each .pyc is of the bytes it ships.
                staged = os.path.join(tmp, "src.zip")
                with zipfile.ZipFile(staged, "w") as z:
                    for name in sorted(files):
                        if name.endswith(".py"):
                            z.writestr(pyc_zip._entry(name), files[name])
                with zipfile.ZipFile(staged) as z:
                    compiled, failed = pyc_zip.compile_tree(z, os.path.join(tmp, "work"))
            path = os.path.join(out, f"{arm}.zip")
            with zipfile.ZipFile(path + ".tmp", "w") as dst:
                seen = set()
                for name, info in p1_infos.items():
                    if name in compiled:
                        seen.add(name)  # P1's stale .pyc, replaced below
                        continue
                    if info.is_dir() or files.get(name) == src.read(info):
                        dst.writestr(copy.copy(info), src.read(info))  # P1's entry, byte for byte
                    else:
                        dst.writestr(
                            pyc_zip._entry(name), files[name]
                        )  # a harness or placement file
                    seen.add(name)
                for name in sorted(set(files) - seen):
                    dst.writestr(pyc_zip._entry(name), files[name])  # hostinfo.py, e16_entry.py
                    seen.add(name)
                for name in sorted(compiled):
                    d = name.rsplit("/", 1)[0] + "/" if "/" in name else None
                    if d and d not in seen:
                        dst.writestr(pyc_zip._entry(d, 0o40755), b"")
                        seen.add(d)
                    dst.writestr(pyc_zip._entry(name), compiled[name])
            os.replace(path + ".tmp", path)
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                if len(names) != len(set(names)):
                    sys.exit(f"{arm}.zip has duplicate entries")
                zips[arm] = {
                    "bytes": os.path.getsize(path),
                    "sha256": pyc_zip.sha256(path),
                    "handler": "e16_entry.handler",
                    "unzipped_bytes": sum(i.file_size for i in z.infolist()),
                    "bytecode": pyc_zip.stale_pyc(z)
                    | {"compiled": len(compiled), "failed": failed},
                }
    m = {
        "schema": "lkai-e16-build-manifest/1",
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                *HARNESS.values(),
                *PLACEMENT.values(),
            )
        ),
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
        "p1_bytecode": p1_bytecode,
        "harness_sha256": {n: hashlib.sha256(read(r)).hexdigest() for n, r in HARNESS.items()},
        "placement_sha256": {
            "e6": {n: entry_sha256(p1_zip, n) for n in PLACEMENT},
            "lazy": {n: hashlib.sha256(read(r)).hexdigest() for n, r in PLACEMENT.items()},
        },
        "zips": zips,
        "base": "P1 function.zip " + m1["function_zip"]["sha256"],
    }
    # summarize.py's cost and identity fields read build_manifest["function_zip"]
    m["function_zip"] = m["zips"]["lazy"] | {"base": m["base"]}
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: m[k] for k in ("zips", "placement_sha256")}, indent=1))
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
