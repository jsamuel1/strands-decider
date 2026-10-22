"""Lambda entry point of both E14 functions: handler "e14_entry.handler". Identical in the ctl and rows zips.

It reads the host metadata the initiative's HOST VARIABILITY rules ask for (the same fields and method as E6's
e6_entry.py), then imports P1's instrumented entry point (baseline/lkai_entry.py) unchanged, so P1's own timers time
every phase. The metadata is read from /proc, /sys and the auxiliary vector with the standard library only, before
lkai_entry starts its clock. It also records the SHA-256 of the libggml-cpu variant this environment mapped, so each
invocation shows which kernel build ran. A field that cannot be read is "unknown", never guessed.

The first invocation of each environment carries the record as `lkai.e14` in its response and as an `E14 {json}`
log line: `at_init` (read at import, which for a restored environment is the snapshot host) and `at_first_invoke`
(this environment's host).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import sys

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
    out = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        v = _read(f"/sys/fs/cgroup/{name}", 256)
        out[name] = v.strip() if v is not None else UNKNOWN
    return out


def ggml_variants(maps: str | None) -> dict:
    """{library file name: sha256} for every libggml-cpu-*.so this process has mapped."""
    out: dict[str, str] = {}
    for line in (maps or "").splitlines():
        path = line.split()[-1] if line.split() else ""
        name = os.path.basename(path)
        if name.startswith("libggml-cpu-") and name not in out:
            try:
                with open(path, "rb") as f:
                    out[name] = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                out[name] = UNKNOWN
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
        "e14_arm": os.environ.get("E14_ARM", UNKNOWN),
    }


META = host_metadata()

# P1's instrumented entry: its clock starts at its own import, after META is read.
import lkai_entry  # noqa: E402

_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        # Read again here: after a SnapStart restore, META (read at import) describes the host that built the
        # snapshot, not this one. The engine has loaded by now, so the variant it chose is mapped. Both arms pay this
        # once per environment, after P1's timers, inside REPORT Duration of the first invocation only.
        rec = {
            "at_init": META,
            "at_first_invoke": host_metadata(),
            "ggml_cpu": ggml_variants(_read("/proc/self/maps", 1 << 22)),
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
        }
        sys.stdout.write("E14 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e14"] = rec
    return out
