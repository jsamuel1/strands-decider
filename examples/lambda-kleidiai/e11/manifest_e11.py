"""Write e11/build/manifest.json: what the E11 zip is, and how it relates to P1's published build.

  python e11/manifest_e11.py --p1-build baseline/build --build e11/build

Records the zip's sha256 and size, the commit and dirty state of every E11 and P1 input (buildcache.dirty plus the
e11/ files), the sha256 of patch_memfdrel.py, and P1's manifest identity (zip sha256, GGUF sha256 and llama.cpp
commit), which E11 reuses unchanged: same weights object, same pinned llama.cpp tree with P1's patches, plus one
dormant patch. Refuses to write a manifest whose function.zip does not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(RECIPE))
E11_INPUTS = (
    "patch_memfdrel.py",
    "container_build.sh",
    "e11_entry.py",
    "hostmeta.py",
    "memstat.py",
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def e11_dirty() -> list[str]:
    """E11 inputs whose bytes differ from HEAD, or that HEAD lacks."""
    out = []
    for f in E11_INPUTS:
        rel = f"examples/lambda-kleidiai/e11/{f}"
        try:
            head = subprocess.run(
                ["git", "-C", REPO, "rev-parse", f"HEAD:{rel}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            out.append(f"untracked {rel}")
            continue
        now = subprocess.run(
            ["git", "-C", REPO, "hash-object", "--no-filters", rel],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if now != head:
            out.append(f"modified {rel}")
    return out


def build_manifest(
    p1_manifest: dict, zip_path: str, dirty_p1: list[str], dirty_e11: list[str], commit: str
) -> dict:
    return {
        "schema": "lkai-e11-manifest/1",
        "repo_commit": commit,
        "dirty_inputs": sorted(dirty_p1 + dirty_e11),
        "repo_dirty": bool(dirty_p1 or dirty_e11),
        "function_zip": {
            "bytes": os.path.getsize(zip_path),
            "sha256": sha256(zip_path),
            "handler": "e11_entry.handler",
            "source_commit": commit,
            "inputs_sha256": None,
        },
        "patch_memfdrel_sha256": sha256(os.path.join(HERE, "patch_memfdrel.py")),
        "model": p1_manifest["model"],
        "llama_cpp": dict(
            p1_manifest["llama_cpp"],
            patches=p1_manifest["llama_cpp"]["patches"] + " + e11/patch_memfdrel.py (dormant: "
            "LKAI_RELEASE_MEMFD=1)",
        ),
        "p1": {
            "repo_commit": p1_manifest.get("repo_commit"),
            "schema": p1_manifest.get("schema"),
            "function_zip_sha256": p1_manifest["function_zip"]["sha256"],
            "gguf_sha256": p1_manifest["model"]["gguf_sha256"],
        },
        "images": p1_manifest.get("images"),
        "toolchain": p1_manifest.get("toolchain"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-build", default=os.path.join(RECIPE, "baseline", "build"))
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    a = ap.parse_args()
    z = os.path.join(a.build, "function.zip")
    if not os.path.exists(z):
        sys.exit(f"{z} does not exist; run e11/container_build.sh first")
    sys.path.insert(0, os.path.join(RECIPE, "baseline"))
    import buildcache

    p1m = json.load(open(os.path.join(a.p1_build, "manifest.json")))
    m = build_manifest(p1m, z, buildcache.dirty(REPO), e11_dirty(), git("rev-parse", "HEAD"))
    tmp = os.path.join(a.build, "manifest.json.tmp")
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.build, "manifest.json"))
    print(json.dumps({k: m[k] for k in ("repo_commit", "repo_dirty")}), m["function_zip"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
