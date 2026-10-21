"""Lambda entry point of both E12 functions: handler "e12_entry.handler". The same file in the same zip for both arms.

It wraps P1's instrumented entry point (baseline/lkai_entry.py), imported unchanged, so P1's own timers time every
phase. The two functions differ in one environment variable, DEC_CLEAR_DATA, which E12's libdecshim.so reads once
at load: unset is P1's per-request clear (llama_memory_clear(mem, true): reset the cache metadata and zero every K/V
and recurrent buffer), "0" resets only the metadata (llama_memory_clear(mem, false)).

The first invocation of each execution environment also reads the host metadata the HOST VARIABILITY rules ask for,
AFTER P1's handler has returned (so P1's phase timers never include it; Lambda's Duration does, on both arms
alike, and the read's own duration is recorded). It is read per environment rather than at import: a SnapStart
environment imports this module on the snapshot host, not on the host it restores on. Standard library only. A
field the sandbox does not expose is "unknown", never guessed. The record goes into the response as `lkai.e12` and
into the log as an `E12 {json}` line. It also carries the shim's own report of its clear mode (dec_clear_data), so
the arm is checked from inside the function rather than taken from its name.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys
import time

import lkai_entry  # P1's instrumented entry: its clock starts at its own import

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


def shim_clear_data() -> int | str:
    """The loaded shim's clear mode: 1 zeroes the buffers (P1), 0 resets metadata only. 'unknown' if not loaded."""
    try:
        import engine  # snapstart/engine.py, already imported by P1's entry

        s = engine.SHIM
        if s is None or not hasattr(s.lib, "dec_clear_data"):
            return UNKNOWN
        import ctypes

        s.lib.dec_clear_data.restype = ctypes.c_int
        s.lib.dec_clear_data.argtypes = [ctypes.c_void_p]
        return int(s.lib.dec_clear_data(s.h))
    except Exception:
        return UNKNOWN


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
        "region": os.environ.get("AWS_REGION", UNKNOWN),
        "e12_arm": os.environ.get("E12_ARM", UNKNOWN),
        "dec_clear_data_env": os.environ.get("DEC_CLEAR_DATA", UNKNOWN),
    }


_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        t = time.monotonic_ns()
        rec = host_metadata() | {
            "shim_clear_data": shim_clear_data(),
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
        }
        rec["meta_read_s"] = round((time.monotonic_ns() - t) / 1e9, 6)
        sys.stdout.write("E12 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e12"] = rec
    return out
