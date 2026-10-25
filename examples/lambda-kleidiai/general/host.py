"""Host metadata for the general ladder (G). Standard library only, read the same way on arm64 and x86_64.

A field that cannot be read is "unknown", never guessed. No processor generation is inferred from any field: the
arm64 identity is the /proc/cpuinfo tuple (implementer, part, variant, revision) exactly as the kernel prints it.
The readers follow E8's e8_host.py (lkai/5a61-parallel-repack), extended with x86 identity, cpufreq governor and
load average, which a fixed host can expose and a Lambda sandbox does not.
"""

from __future__ import annotations

import os
import platform
import struct

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26


def read_text(path: str, limit: int = 1 << 16) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def cpuinfo(text: str | None) -> dict:
    """First processor block's identity fields (arm64 and x86 keys), plus the processor count."""
    keys = {
        "CPU implementer": "implementer",
        "CPU architecture": "architecture",
        "CPU variant": "variant",
        "CPU part": "part",
        "CPU revision": "revision",
        "Features": "features",
        "vendor_id": "vendor",
        "model name": "model_name",
        "cpu family": "family",
        "model": "model",
        "stepping": "stepping",
        "flags": "flags",
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


def identity(cpu: dict) -> str:
    """The stratum key: arm64 'part' as printed, or x86 'vendor family/model/stepping'. 'unknown' if neither."""
    if cpu.get("part", UNKNOWN) != UNKNOWN:
        return cpu["part"]
    if cpu.get("vendor", UNKNOWN) != UNKNOWN:
        return f"{cpu['vendor']} {cpu.get('family')}/{cpu.get('model')}/{cpu.get('stepping')}"
    return UNKNOWN


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
        v = read_text(f"/sys/fs/cgroup/{name}", 256)
        out[name] = v.strip() if v is not None else UNKNOWN
    return out


def governor() -> str:
    v = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", 64)
    return v.strip() if v else UNKNOWN


def loadavg() -> list[float] | str:
    v = read_text("/proc/loadavg", 256)
    try:
        return [float(x) for x in (v or "").split()[:3]] or UNKNOWN
    except ValueError:
        return UNKNOWN


def proc_status_kb(
    text: str | None, fields: tuple[str, ...] = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem")
) -> dict:
    """Memory fields from /proc/self/status, in kB ('unknown' when absent)."""
    out: dict = dict.fromkeys(fields, UNKNOWN)
    for line in (text or "").splitlines():
        k, _, v = line.partition(":")
        if k in out and v.strip().endswith("kB"):
            out[k] = int(v.split()[0])
    return out


def host_metadata() -> dict:
    try:
        with open("/proc/self/auxv", "rb") as f:
            raw: bytes | None = f.read()
    except OSError:
        raw = None
    try:
        affinity: list[int] | str = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = UNKNOWN
    cpu = cpuinfo(read_text("/proc/cpuinfo"))
    return {
        "machine": platform.machine() or UNKNOWN,
        "identity": identity(cpu),
        "cpu": cpu,
        **auxv(raw),
        "os_cpu_count": os.cpu_count() or UNKNOWN,
        "sched_affinity": affinity,
        "cgroup": cgroup(),
        "governor": governor(),
        "loadavg": loadavg(),
        "kernel": platform.release() or UNKNOWN,
        "python": platform.python_version(),
    }
