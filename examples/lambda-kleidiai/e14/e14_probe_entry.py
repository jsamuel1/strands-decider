"""Lambda entry point of E14's DIAGNOSTIC probe function: handler "e14_probe_entry.handler". Not a measured arm.

lkai-19b2-probe ships P1's zip plus e14_entry.py and this file, with the six libggml-cpu variants built with
e14/patch_concat_probe.py (P1's concat kernel, unchanged, timed on thread 0). It sets LKAI_E14_PROBE=1 in its
environment. Every invocation after the first engine load:
  1. resets the probe table,
  2. runs e14_entry.handler (P1's instrumented handler and E14's host metadata) unchanged,
  3. reads the table and logs `E14_PROBE {json}`: per (concat shape, node), calls, thread-0 ns and dst bytes.
Step 3 runs after P1's timers stop, so P1's phases are unaffected; it does add to this arm's REPORT Duration. The
probe arm's latency is never compared with ctl or rows. Its use is the thread-0 concat time per request IN Lambda,
to set against the offline probe on the same build.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile

import e14_entry

_state: dict = {"n": 0, "lib": None, "path": None}


def _find_lib():
    """The mapped libggml-cpu variant that exports the probe (the one serving forwards), else None."""
    for line in open("/proc/self/maps"):
        parts = line.split()
        path = parts[-1] if parts else ""
        name = os.path.basename(path)
        if name.startswith("libggml-cpu-") and "armv8.0_1" not in name:
            try:
                lib = ctypes.CDLL(path)
                lib.lkai_e14_probe_dump.restype = ctypes.c_int
                return lib, path
            except (OSError, AttributeError):
                continue
    return None, None


def _table(lib) -> dict:
    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        lib.lkai_e14_probe_dump(path.encode())
        out = {}
        for line in open(path):
            k, n, ns, b = line.rstrip("\n").split("\t")
            out[k] = [int(n), int(ns), int(b)]
        return out
    finally:
        os.unlink(path)


def handler(event, context):
    lib = _state["lib"]
    if lib is not None:
        lib.lkai_e14_probe_reset()
    out = e14_entry.handler(event, context)
    _state["n"] += 1
    if lib is None:
        _state["lib"], _state["path"] = _find_lib()
        if _state["lib"] is not None:
            _state[
                "lib"
            ].lkai_e14_probe_reset()  # the first invocation's table includes the model load
        return out
    rec = {
        "n": _state["n"],
        "lib": os.path.basename(_state["path"]),
        "request_id": getattr(context, "aws_request_id", "unknown"),
        "log_stream": getattr(context, "log_stream_name", "unknown"),
        "concat": _table(lib),
    }
    if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
        rec["lkai_request_id"] = out["lkai"].get("request_id")
        rec["cpu_part"] = (out["lkai"].get("inv") or {}).get("cpu_part")
    sys.stdout.write("E14_PROBE " + json.dumps(rec, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return out
