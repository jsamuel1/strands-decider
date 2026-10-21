"""Lambda entry point of both E6 functions: handler "e6_entry.handler". Identical in the ctl and pyc zips.

It reads the host metadata the HOST VARIABILITY rules ask for, then imports P1's instrumented entry point
(baseline/lkai_entry.py) unchanged. P1's own timers therefore time `imports` and every later phase. The metadata is
read from /proc, /sys and the auxiliary vector with the standard library only, before lkai_entry starts its clock,
so none of it lands in P1's phases. After lkai_entry's import returns, the wrapper checks which task-root modules
were loaded from a usable bytecode file. That check is the mechanism evidence for E6's one variable, and its own
duration is recorded. A field that cannot be read is "unknown", never guessed.

The first invocation of each environment carries the record as `lkai.e6` in its response and as an `E6 {json}` log
line. Later invocations carry only `cpu_part`, through P1's own record.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys
import time

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26


def _read(path: str, limit: int = 1 << 16) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def cpuinfo(text: str | None) -> dict:
    """The first processor block's identity fields, plus the processor count. Missing fields are 'unknown'."""
    keys = {
        "CPU implementer": "implementer",
        "CPU architecture": "architecture",
        "CPU variant": "variant",
        "CPU part": "part",
        "CPU revision": "revision",
        "Features": "features",
    }
    out = dict.fromkeys(keys.values(), UNKNOWN)
    if not text:
        return out | {"processors": UNKNOWN}
    n = 0
    for line in text.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "processor":
            n += 1
        if k in keys and out[keys[k]] == UNKNOWN:
            out[keys[k]] = v
    return out | {"processors": n or UNKNOWN}


def auxv(raw: bytes | None) -> dict:
    """AT_HWCAP / AT_HWCAP2 from /proc/self/auxv (64-bit little-endian pairs), as hex strings."""
    out = {"hwcap": UNKNOWN, "hwcap2": UNKNOWN}
    if not raw or len(raw) % 16:
        return out
    for key, val in struct.iter_unpack("<QQ", raw):
        if key == AT_HWCAP:
            out["hwcap"] = hex(val)
        elif key == AT_HWCAP2:
            out["hwcap2"] = hex(val)
    return out


def cgroup() -> dict:
    """cgroup v2 CPU and memory limits, where the sandbox exposes them."""
    out = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        v = _read(f"/sys/fs/cgroup/{name}", 256)
        out[name] = v.strip() if v is not None else UNKNOWN
    return out


def host_metadata() -> dict:
    try:
        with open("/proc/self/auxv", "rb") as f:
            raw: bytes | None = f.read()
    except OSError:
        raw = None
    try:
        affinity: int | str = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = UNKNOWN
    return {
        "cpu": cpuinfo(_read("/proc/cpuinfo")),
        **auxv(raw),
        "os_cpu_count": os.cpu_count() or UNKNOWN,
        "sched_affinity": affinity,
        "cgroup": cgroup(),
        "kernel": platform.release() or UNKNOWN,
        "python": platform.python_version(),
        "execution_env": os.environ.get("AWS_EXECUTION_ENV", UNKNOWN),
        "init_type": os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE", UNKNOWN),
        "memory_mb": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", UNKNOWN),
        "e6_arm": os.environ.get("E6_ARM", UNKNOWN),
    }


def bytecode_use(root: str, modules: dict) -> dict:
    """For every module loaded from a .py under root: whether its cached .pyc exists and the 3.12 loader could use it.

    Usable = hash-based unchecked, or timestamp-based with the source's current mtime and size. Anything else is
    recompiled from source on every import, since a read-only root cannot cache the result."""
    counts = {"task_root_modules": 0, "pyc_present": 0, "usable_unchecked_hash": 0, "usable_timestamp": 0}
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


META = host_metadata()

import lkai_entry  # noqa: E402  P1's instrumented entry: its clock starts at its own import

_t = time.monotonic_ns()
META["bytecode"] = bytecode_use(os.environ.get("LAMBDA_TASK_ROOT", "/var/task") + "/", sys.modules)
META["bytecode_check_s"] = round((time.monotonic_ns() - _t) / 1e9, 6)
_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        rec = META | {"log_stream": getattr(context, "log_stream_name", UNKNOWN)}
        sys.stdout.write("E6 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e6"] = rec
    return out
