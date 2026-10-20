"""Content-addressed cache keys for build.sh's stages. Standard library only.

  python buildcache.py key   <stage> --repo R --build B [--pin NAME=VALUE ...]   # print the stage's key
  python buildcache.py check <stage> <key> --build B [--remove]                  # 0 fresh, 10 run the stage
  python buildcache.py begin <stage> <key> --build B                             # stale outputs are gone
  python buildcache.py stamp <stage> <key> --build B                             # the stage finished
  python buildcache.py keys  --build B                                           # every stamped key, as JSON

A stage is reused only when its stamped key equals the key of its current inputs. A key hashes the content of
every input file (tracked or not, committed or not), the pinned revisions and image digests, and the keys of the
stages it consumes, so a change anywhere upstream changes every dependent key:

  llama    llama.cpp at LLAMA_REV with llama/patch_llama.py applied            <- LLAMA_REV, patch_llama.py
  weights  merged, H512-folded f32 GGUF and assets/                            <- CKPT, CKPT_REV, PY_IMAGE, llama,
           weights.sh, weights/, examples/lambda/prepare_model.py, src/strands_decider, pyproject.toml
  zip      function.zip, bin/llama-quantize, quant-lib/, toolchain.txt         <- LAMBDA_IMAGE, llama, the files the
           zip copies, the assets' content
  quant    model/s2b-h512-Q8_0.gguf                                            <- QUANT_TYPE, weights, zip

`check` with a mismatched key clears the stamps of the stage and every stage downstream of it and lists their
outputs; build.sh deletes those (through the build container, which writes them as root) and only then calls
`begin`, which records the key as pending. A pending key that matches is not wiped again, so an interrupted stage
resumes (weights.sh's steps each move their output into place only when complete). Because `begin` follows the
delete, a failed delete leaves no pending key, and the next run lists the same outputs again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

ORDER = ("llama", "weights", "zip", "quant")
OUTPUTS = {
    "llama": ["llama.cpp"],
    "weights": ["weights", "model", "merge_report.json", "weights-pip-freeze.txt"],
    "zip": [
        "function.zip",
        "bin",
        "quant-lib",
        "toolchain.txt",
        ".zip-repo-commit",
    ],
    "quant": ["model/s2b-h512-Q8_0.gguf"],
}
# Who consumes whom; a stale stage invalidates every stage reachable from it.
CONSUMERS = {
    "llama": ["weights", "zip"],
    "weights": ["zip", "quant"],
    "zip": ["quant"],
    "quant": [],
}
INPUTS = {
    "llama": ["examples/lambda-kleidiai/llama/patch_llama.py"],
    "weights": [
        "examples/lambda-kleidiai/baseline/weights.sh",
        "examples/lambda-kleidiai/weights",
        "examples/lambda/prepare_model.py",
        "src/strands_decider",
        "pyproject.toml",
    ],
    # What container_build.sh copies or compiles (the shim, snapstart engine, PR #77 app, the package sources).
    "zip": [
        "examples/lambda-kleidiai/shim",
        "examples/lambda-kleidiai/snapstart",
        "examples/lambda-kleidiai/baseline/lkai_entry.py",
        "examples/lambda-kleidiai/baseline/phases.py",
        "examples/lambda-kleidiai/baseline/container_build.sh",
        "examples/lambda/app",
        "src/strands_decider",
    ],
    "quant": [],
}
PINS = {
    "llama": ["LLAMA_REV"],
    "weights": ["CKPT", "CKPT_REV", "PY_IMAGE"],
    "zip": ["LAMBDA_IMAGE"],
    "quant": ["QUANT_TYPE"],
}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
# Files a stage's container reads that cannot change its output: weights.sh copies the root README and LICENSE only
# because the package's metadata names them. Keying on them would rebuild the weights (about an hour) for a doc edit.
NOT_KEYED = {"weights": ["README.md", "LICENSE"]}


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tree_sha(root: str, paths: list[str]) -> str:
    """sha256 over (relative path, exec bit, content sha) of every file under `paths`, sorted. A missing input is
    part of the key (as missing) rather than an error, so deleting an input file also changes the key."""
    entries = []
    for p in paths:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            files = [full]
        elif os.path.isdir(full):
            files = []
            for d, dirs, names in os.walk(full):
                dirs[:] = sorted(
                    x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")
                )
                files += [os.path.join(d, n) for n in names if not n.endswith(".pyc")]
        else:
            entries.append(f"{p}\tmissing")
            continue
        for f in sorted(files):
            rel = os.path.relpath(f, root)
            ex = "x" if os.access(f, os.X_OK) else "-"
            entries.append(f"{rel}\t{ex}\t{_file_sha(f)}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def stage_key(
    stage: str, repo: str, build: str, pins: dict[str, str], upstream: dict[str, str]
) -> str:
    """The key of `stage`'s inputs. `upstream` holds the current keys of the stages it consumes."""
    missing = [p for p in PINS[stage] if not pins.get(p)]
    if missing:
        raise ValueError(f"{stage}: pins {missing} not given")
    needs = [s for s in ORDER if stage in CONSUMERS[s]]
    absent = [s for s in needs if not upstream.get(s)]
    if absent:
        raise ValueError(f"{stage}: upstream keys {absent} not given")
    doc = {
        "stage": stage,
        "pins": {p: pins[p] for p in PINS[stage]},
        "upstream": {s: upstream[s] for s in needs},
        "inputs": tree_sha(repo, INPUTS[stage]),
    }
    # The zip ships assets/ from the weights stage: key it on their bytes, not only on their provenance.
    if stage == "zip":
        doc["assets"] = tree_sha(os.path.join(build, "model"), ["assets"])
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def downstream(stage: str) -> list[str]:
    seen, todo = [], [stage]
    while todo:
        s = todo.pop(0)
        if s not in seen:
            seen.append(s)
            todo += CONSUMERS[s]
    return [s for s in ORDER if s in seen]


def _keyfile(build: str, stage: str, kind: str = "key") -> str:
    return os.path.join(build, ".keys", f"{stage}.{kind}")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _write(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(value + "\n")
    os.replace(path + ".tmp", path)


def _outputs_present(build: str, stage: str) -> bool:
    return all(os.path.exists(os.path.join(build, o)) for o in OUTPUTS[stage])


def check(build: str, stage: str, key: str) -> tuple[bool, list[str]]:
    """(fresh, paths to delete). Fresh: the stamped key is `key` and the outputs exist. Otherwise, unless `begin`
    already recorded this key as pending (an interrupted run of these exact inputs, whose stale outputs are gone),
    every output of the stage and of its downstream stages is stale and their stamps are cleared. The caller deletes
    the paths, then calls begin()."""
    if _read(_keyfile(build, stage)) == key and _outputs_present(build, stage):
        return True, []
    if _read(_keyfile(build, stage, "pending")) == key:
        return False, []
    stale: list[str] = []
    for s in downstream(stage):
        for kind in ("key", "pending"):
            try:
                os.remove(_keyfile(build, s, kind))
            except FileNotFoundError:
                pass
        stale += [o for o in OUTPUTS[s] if os.path.lexists(os.path.join(build, o))]
    return False, stale


def begin(build: str, stage: str, key: str) -> None:
    """The stale outputs check() listed are deleted: record `key` as pending. Refuses if any still exists."""
    if _read(_keyfile(build, stage, "pending")) == key:
        return
    left = [
        o for s in downstream(stage) for o in OUTPUTS[s] if os.path.lexists(os.path.join(build, o))
    ]
    if left:
        sys.exit(f"{stage}: stale outputs still present: {left}")
    _write(_keyfile(build, stage, "pending"), key)


def stamp(build: str, stage: str, key: str) -> None:
    if _read(_keyfile(build, stage, "pending")) != key:
        sys.exit(f"{stage}: {key} was not begun; rerun check and begin")
    if not _outputs_present(build, stage):
        sys.exit(f"{stage}: outputs {OUTPUTS[stage]} missing; not stamping")
    _write(_keyfile(build, stage), key)
    try:
        os.remove(_keyfile(build, stage, "pending"))
    except FileNotFoundError:
        pass


def stamped(build: str) -> dict[str, str | None]:
    return {s: _read(_keyfile(build, s)) for s in ORDER}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["key", "check", "begin", "stamp", "keys"])
    ap.add_argument("stage", nargs="?", choices=ORDER)
    ap.add_argument("value", nargs="?")
    ap.add_argument("--build", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--pin", action="append", default=[], help="NAME=VALUE")
    ap.add_argument("--upstream", action="append", default=[], help="STAGE=KEY")
    ap.add_argument("--remove", action="store_true", help="delete stale outputs here (tests)")
    a = ap.parse_args(argv)
    if a.cmd == "keys":
        print(json.dumps(stamped(a.build), indent=1))
        return 0
    if not a.stage:
        ap.error("stage required")
    if a.cmd == "key":
        pins = dict(p.split("=", 1) for p in a.pin)
        up = dict(u.split("=", 1) for u in a.upstream)
        print(stage_key(a.stage, a.repo, a.build, pins, up))
        return 0
    if not a.value:
        ap.error("key required")
    if a.cmd == "stamp":
        stamp(a.build, a.stage, a.value)
        return 0
    if a.cmd == "begin":
        begin(a.build, a.stage, a.value)
        return 0
    fresh, stale = check(a.build, a.stage, a.value)
    for p in stale:
        print(p)
        if a.remove:
            full = os.path.join(a.build, p)
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
    if a.remove and not fresh:
        begin(a.build, a.stage, a.value)
    return 0 if fresh else 10


if __name__ == "__main__":
    sys.exit(main())
