"""Lambda entry point of both E16 functions: handler "e16_entry.handler". Identical in the e6 and lazy zips.

It reads the environment's host record with H1's baseline/hostinfo.py, before lkai_entry starts its clock, then
imports P1's instrumented entry point (baseline/lkai_entry.py, which patches the engine when engine is first imported,
wherever that happens). At the end of the init it records which of the model stack's packages the main thread had
imported (the mechanism evidence for E16's one variable, import placement); that check is a dict scan of
sys.modules and its own duration is recorded.

On the first invocation of each environment, AFTER lkai_entry's handler has returned (so it is inside Duration on both
arms equally, and never inside Init or a phase timer), it adds: the bytecode check (which task-root modules were
loaded from usable unchecked-hash bytecode; E6's check), and, after a SnapStart restore, a fresh host record, since a
restored copy can run on a different host from the one that built its snapshot. The record rides on that response as
`lkai.e16` and on an `E16 {json}` log line. A field that cannot be read is None or "unknown", never guessed.
"""

from __future__ import annotations

import json
import os
import sys
import time

import hostinfo

UNKNOWN = "unknown"
STACK = (
    "numpy",
    "pydantic",
    "pydantic_core",
    "tokenizers",
    "safetensors",
    "strands_decider",
    "engine",
)
HOST = hostinfo.snapshot()

import lkai_entry  # noqa: E402 - P1's instrumented entry: its clock starts at its own import


def stack_loaded(modules: dict) -> dict[str, int]:
    """How many modules of each model-stack package are in sys.modules (0: not imported yet)."""
    out = dict.fromkeys(STACK, 0)
    for name in list(modules):
        top = name.split(".", 1)[0]
        if top in out:
            out[top] += 1
    return out


def bytecode_use(root: str, modules: dict) -> dict:
    """E6's check: every module loaded from a .py under root, and whether the 3.12 loader could use its .pyc."""
    counts = {
        "task_root_modules": 0,
        "pyc_present": 0,
        "usable_unchecked_hash": 0,
        "usable_timestamp": 0,
    }
    for m in list(modules.values()):
        spec = getattr(m, "__spec__", None)
        origin = getattr(spec, "origin", None) or ""
        if not origin.startswith(root) or not origin.endswith(".py"):
            continue
        counts["task_root_modules"] += 1
        cached = getattr(spec, "cached", None)
        try:
            with open(cached, "rb") as f:
                head = f.read(16)
        except (OSError, TypeError):
            continue
        counts["pyc_present"] += 1
        if len(head) < 16:
            continue
        flags = int.from_bytes(head[4:8], "little")
        if flags & 1:
            if not flags & 2:
                counts["usable_unchecked_hash"] += 1
            continue
        st = os.stat(origin)
        if int.from_bytes(head[8:12], "little") == int(st.st_mtime) & 0xFFFFFFFF and int.from_bytes(
            head[12:16], "little"
        ) == (st.st_size & 0xFFFFFFFF):
            counts["usable_timestamp"] += 1
    counts["recompiled_from_source"] = counts["task_root_modules"] - (
        counts["usable_unchecked_hash"] + counts["usable_timestamp"]
    )
    return counts


_t = time.monotonic_ns()
META = {
    "schema": "lkai-e16/1",
    "arm": os.environ.get("E16_ARM", UNKNOWN),
    "init_type": os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE", UNKNOWN),
    "main_thread_stack_at_init_end": stack_loaded(sys.modules),
}
META["stack_check_s"] = round((time.monotonic_ns() - _t) / 1e9, 6)
_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        t = time.monotonic_ns()
        rec = dict(META)
        rec["stack_at_first_invoke"] = stack_loaded(sys.modules)
        rec["bytecode"] = bytecode_use(
            os.environ.get("LAMBDA_TASK_ROOT", "/var/task") + "/", sys.modules
        )
        restored = bool(getattr(getattr(lkai_entry, "REC", None), "restored", False))
        rec["restored"] = restored
        rec["host"] = hostinfo.snapshot() if restored else HOST
        rec["host_read"] = "after-restore" if restored else "init"
        rec["log_stream"] = getattr(context, "log_stream_name", UNKNOWN)
        rec["first_invoke_check_s"] = round((time.monotonic_ns() - t) / 1e9, 6)
        line = {k: v for k, v in rec.items() if k != "host"} | {
            "host_fingerprint": rec["host"]["fingerprint"]
        }
        sys.stdout.write("E16 " + json.dumps(line, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e16"] = rec
    return out
