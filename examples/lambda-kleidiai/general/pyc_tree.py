#!/usr/bin/env python3
"""E6's compile rule applied to an unzipped package tree, in place (x86_e6.sh runs it in P1's amd64 image).

  python3.12 pyc_tree.py <package dir> <manifest.json>

Every .py gets py_compile UNCHECKED_HASH bytecode with dfile = its own path (the package is mounted at /var/task, so
co_filename is /var/task/<path>), exactly as e6/pyc_zip.py (lkai/fada-hotspot e989e39) compiles P1's zip. Writes the
manifest (interpreter, magic, counts, every failure) and exits 1 when ANY file failed to compile: a partial bytecode
tree is not E6's "complete bytecode", so it must never reach a benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import sys


def compile_tree(root: str) -> tuple[list[str], list[str]]:
    ok, bad = [], []
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            try:
                py_compile.compile(
                    p,
                    dfile=p,
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
                ok.append(os.path.relpath(p, root))
            except py_compile.PyCompileError:
                bad.append(os.path.relpath(p, root))
    return ok, bad


def tree_sha256(root: str) -> str:
    """sha256 over every file's relative path and bytes, in path order (symlinks by their target text)."""
    h = hashlib.sha256()
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        for f in sorted(fs):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            if os.path.islink(p):
                h.update(b"L" + os.readlink(p).encode())
            else:
                with open(p, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root, manifest = argv
    ok, bad = compile_tree(root)
    m = {
        "schema": "lkai-general-pyc-tree/1",
        "compiler": {
            "python": sys.version.split()[0],
            "magic": importlib.util.MAGIC_NUMBER.hex(),
            "invalidation_mode": "unchecked-hash",
        },
        "compiled": len(ok),
        "failed": bad,
        "tree_sha256": tree_sha256(root),
    }
    with open(manifest, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    print("compiled", len(ok), "failed", len(bad))
    if bad:
        print("FATAL: bytecode incomplete, failed:", *bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
