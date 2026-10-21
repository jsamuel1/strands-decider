"""Lambda entry point of both E11 functions: handler "e11_entry.handler". The same zip serves both arms.

The arms differ only in LKAI_RELEASE_MEMFD: unset (`off`, P1's behaviour) or 1 (`rel`, the loader releases the weight
memfd's pages that no tensor aliases, e11/patch_memfdrel.py). This module reads the host metadata the HOST
VARIABILITY rules ask for (hostmeta.py, standard library) before P1's instrumented entry point starts its clock, then
imports baseline/lkai_entry.py unchanged, so P1's timers time `imports` and every later phase. After the load it
records:

- memory: /proc/self/status (VmRSS, RssAnon, RssShmem, ...), smaps_rollup and the cgroup's memory.current/stat,
- the memfd's own resident bytes (mincore over a temporary view of /proc/self/fd/<n>),
- the loader's lkai_memfd_release_stats (released and kept bytes, microseconds, errno, ranges, files).

They go out as an `E11 {json}` line in the before_snapshot hook (the snapshot init, no invocation sees it) and at the
first invocation of every environment (also in the response as `lkai.e11`), taken after P1's handler returns so the
engine latency P1 records does not include it. A field that cannot be read is "unknown".
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import hostmeta
import memstat

UNKNOWN = "unknown"
META = hostmeta.collect() | {
    "e11_arm": os.environ.get("E11_ARM", UNKNOWN),
    "release_flag": os.environ.get("LKAI_RELEASE_MEMFD", ""),
}

import lkai_entry  # noqa: E402  (P1's instrumented entry: its clock starts at its own import)
import loader  # noqa: E402  (PR #77's loader: it keeps the memfd in loader._fds)

_STAT_KEYS = ("released_bytes", "kept_bytes", "release_us", "errno", "ranges", "files")


def release_stats() -> dict | str:
    task = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
    try:
        lib = ctypes.CDLL(os.path.join(task, "lib", "libllama.so"))
        arr = (ctypes.c_int64 * len(_STAT_KEYS)).in_dll(lib, "lkai_memfd_release_stats")
    except (OSError, ValueError):
        return UNKNOWN
    return dict(zip(_STAT_KEYS, list(arr), strict=True))


def memfd_state() -> dict:
    out = {}
    for fd in getattr(loader, "_fds", []):
        try:
            size = os.fstat(fd).st_size
            out[str(fd)] = {"size": size, "resident_bytes": memstat.fd_resident_bytes(fd, size)}
        except OSError as e:
            out[str(fd)] = {"error": f"{type(e).__name__}: {e}"}
    return out


def record(stage: str) -> dict:
    t = time.monotonic_ns()
    rec = {
        "stage": stage,
        "mem": memstat.snapshot(),
        "memfd": memfd_state(),
        "release": release_stats(),
    }
    rec["collect_s"] = round((time.monotonic_ns() - t) / 1e9, 6)
    return rec


def _emit(rec: dict) -> None:
    sys.stdout.write("E11 " + json.dumps(rec, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") != "on-demand":
    # The load ran inside lkai_entry's import. This is the state the snapshot captures.
    _AFTER_LOAD = record("after_load")
else:
    _AFTER_LOAD = (
        None  # on demand the load runs in a background thread; the first invocation records it
    )

try:
    from snapshot_restore_py import register_before_snapshot

    register_before_snapshot(
        lambda: _emit({"kind": "before_snapshot", **META, **(record("before_snapshot"))})
    )
except ImportError:  # not on Lambda
    pass

_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        rec = {
            "kind": "first_invoke",
            **META,
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
            "env_id": lkai_entry._state["env_id"] if hasattr(lkai_entry, "_state") else UNKNOWN,  # noqa: SLF001
            **record("first_invoke"),
        }
        if _AFTER_LOAD is not None:
            rec["after_load"] = _AFTER_LOAD
        _emit(rec)
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e11"] = rec
    return out
