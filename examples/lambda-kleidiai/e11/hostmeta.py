"""Host metadata for the HOST VARIABILITY rules, standard library only. A field that cannot be read is "unknown".

Same fields as E6's e6_entry.host_metadata (cpuinfo identity, HWCAPs, CPU count, affinity, cgroup limits, kernel,
runtime), read from /proc, /sys and the auxiliary vector. No processor generation is inferred from any of them.
"""

from __future__ import annotations

import os
import platform
import struct

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26


def _read(path: str, limit: int = 1 << 16) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def cpuinfo(text: str | None) -> dict:
    """The first processor block's identity fields, plus the processor count."""
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
    out = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        v = _read(f"/sys/fs/cgroup/{name}", 256)
        out[name] = v.strip() if v is not None else UNKNOWN
    return out


def collect() -> dict:
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
    }
