"""Lambda entry point of every E15 function: handler "e15_entry.handler". The same file on every arm.

E15 changes one variable: candidate A, llamafile's F32 sgemm for F32 x F32 MUL_MAT (attention KQV) in the SVE / i8mm
CPU variants, enabled by LKAI_E15_A=1 in the candidate libraries (e15/patch_a.py). This wrapper adds only
observation, the same as E13's (lkai/a088-omp-wait 0dc2eed):

- Before P1's instrumented entry point (baseline/lkai_entry.py) starts its clock, it reads the host metadata the
  HOST VARIABILITY rules ask for: /proc/cpuinfo identity and features, AT_HWCAP/AT_HWCAP2, CPU count and affinity,
  cgroup v1/v2 CPU and memory files where the sandbox exposes them, kernel, runtime. It reads the OpenMP variables
  this environment sets, and the E15 switch. None of this is inside P1's phases.
- Around each invocation it reads getrusage(RUSAGE_SELF) for the whole process (every thread): user and system CPU
  seconds and voluntary / involuntary context switches. The two syscalls sit outside lkai_entry's handler timers and
  REPORT Duration includes them on every arm alike.
- On the first invocation it records which libggml-cpu variant and libgomp are mapped (the variant decides whether
  candidate A can apply: only the SVE / i8mm variants are changed) and the process's thread count.

A field the sandbox does not expose is "unknown", never guessed. No CPU generation is inferred.

The first invocation of each environment carries the full record as `lkai.e15` in its response and as an
`E15 {json}` log line. Every invocation carries `lkai.e15_inv` (the rusage deltas).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import struct
import sys
import time

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26
ENV_VARS = (
    "LKAI_E15_A",
    "LKAI_E15_B",
    "OMP_WAIT_POLICY",
    "GOMP_SPINCOUNT",
    "OMP_NUM_THREADS",
    "THREADS",
)
CGROUP_FILES = (
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/cpu.stat",
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/cpuset.cpus.effective",
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    "/proc/self/cgroup",
)


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
    """cgroup v1 and v2 CPU and memory files, where the sandbox exposes them; 'unknown' otherwise."""
    out = {}
    for path in CGROUP_FILES:
        v = _read(path, 1024)
        out[path] = v.strip() if v is not None else UNKNOWN
    return out


def proc_threads() -> int | str:
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith("Threads:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                break
    return UNKNOWN


def mapped(substr: str) -> list[str] | str:
    maps = _read("/proc/self/maps", 1 << 22)
    if maps is None:
        return UNKNOWN
    return sorted({ln.split()[-1].rsplit("/", 1)[-1] for ln in maps.splitlines() if substr in ln})


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
        "e15_arm": os.environ.get("E15_ARM", UNKNOWN),
        "switch_env": {k: os.environ.get(k, "unset") for k in ENV_VARS},
    }


def rusage() -> dict:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"utime": r.ru_utime, "stime": r.ru_stime, "nvcsw": r.ru_nvcsw, "nivcsw": r.ru_nivcsw}


def rusage_delta(a: dict, b: dict, wall_s: float) -> dict:
    """CPU seconds (user, system), context switches and CPU per wall second between two rusage() reads."""
    u, s = b["utime"] - a["utime"], b["stime"] - a["stime"]
    return {
        "wall_s": round(wall_s, 6),
        "utime_s": round(u, 6),
        "stime_s": round(s, 6),
        "cpu_per_wall": round((u + s) / wall_s, 4) if wall_s > 0 else None,
        "nvcsw": b["nvcsw"] - a["nvcsw"],
        "nivcsw": b["nivcsw"] - a["nivcsw"],
    }


META = host_metadata()

# P1's instrumented entry: its clock starts at its own import, after the metadata read above.
import lkai_entry  # noqa: E402

_n = {"n": 0}


def handler(event, context):
    a, t = rusage(), time.monotonic()
    out = lkai_entry.handler(event, context)
    inv = rusage_delta(a, rusage(), time.monotonic() - t)
    _n["n"] += 1
    if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
        out["lkai"]["e15_inv"] = inv
    if _n["n"] == 1:
        rec = META | {
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
            "libgomp_mapped": mapped("libgomp"),
            "ggml_cpu_mapped": mapped("libggml-cpu"),
            "threads_after_first_invoke": proc_threads(),
        }
        sys.stdout.write("E15 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e15"] = rec
    return out
