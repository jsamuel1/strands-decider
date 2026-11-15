"""What host is this execution environment on? A standard-library snapshot for the H1 host-variability record.

lkai_entry calls snapshot() once per execution environment: at init (on demand, or the snapshot-building init of a
SnapStart version) and again after every SnapStart restore, because a restored copy can run on a different host from
the one that built its snapshot. The result rides on that environment's first `invoke` record (and on
`before_snapshot` / `after_restore`), so bench.py stores it in the invocation row as `lkai.host` with no change to
bench.py or summarize.py.

Facts only. Every field is either what the environment exposed or None, and every None is listed in `unknown` with
the reason. Nothing names a processor generation or microarchitecture: the CPU part and MIDR are reported as the hex
values the kernel shows, and the HWCAP words as the integers getauxval returned.

Fields:
  cpu        /proc/cpuinfo: every distinct (implementer, architecture, variant, part, revision) tuple with its
             processor count, and every distinct Features line (`features_sets`); a single-tuple host also gets
             the tuple's fields at top level. midr_el1 per CPU from sysfs where the kernel exposes it.
  hwcap      getauxval(AT_HWCAP) and getauxval(AT_HWCAP2), as integers.
  cores      os.cpu_count(), len(os.sched_getaffinity(0)), and /sys/devices/system/cpu/{online,possible}.
  cgroup     cgroup v2 cpu.max / memory.max / cpuset.cpus.effective, or the v1 equivalents.
  memory     MemTotal from /proc/meminfo (kB).
  kernel     uname release/version/machine, platform.python_version().
  lambda     AWS_REGION, AWS_EXECUTION_ENV, AWS_LAMBDA_FUNCTION_MEMORY_SIZE, AWS_LAMBDA_FUNCTION_VERSION,
             AWS_LAMBDA_INITIALIZATION_TYPE (values; none of them is a credential). The log stream name is taken
             from the invocation context by the caller, not from the environment, because environment values
             captured in a snapshot may describe the environment that built it.
"""

from __future__ import annotations

import ctypes
import os
import platform
import time
from typing import Any

SCHEMA = "lkai-host/1"
AT_HWCAP, AT_HWCAP2 = 16, 26  # <elf.h>; the same on aarch64 and x86_64
CPU_KEYS = (
    ("implementer", "CPU implementer"),
    ("architecture", "CPU architecture"),
    ("variant", "CPU variant"),
    ("part", "CPU part"),
    ("revision", "CPU revision"),
)
LAMBDA_ENV = (
    "AWS_REGION",
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_MEMORY_SIZE",
    "AWS_LAMBDA_FUNCTION_VERSION",
    "AWS_LAMBDA_INITIALIZATION_TYPE",
)
CGROUP_FILES = {
    "v2": {
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "memory_max": "/sys/fs/cgroup/memory.max",
        "cpuset_effective": "/sys/fs/cgroup/cpuset.cpus.effective",
    },
    "v1": {
        "cpu_cfs_quota_us": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "cpu_cfs_period_us": "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "memory_limit_in_bytes": "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    },
}


def _read(path: str, limit: int = 1 << 20) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Distinct CPU identity tuples and Features lines of a /proc/cpuinfo text (arm64 layout; one block per CPU)."""
    blocks, cur = [], {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
            cur = {}
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            cur[k.strip()] = v.strip()
    if cur:
        blocks.append(cur)
    blocks = [b for b in blocks if "processor" in b]
    ids: dict[tuple, int] = {}
    feats: dict[str, int] = {}
    for b in blocks:
        t = tuple(b.get(src) for _, src in CPU_KEYS)
        ids[t] = ids.get(t, 0) + 1
        if "Features" in b:
            feats[b["Features"]] = feats.get(b["Features"], 0) + 1
    out: dict[str, Any] = {
        "processors": len(blocks),
        "identities": [
            {**{k: v for (k, _), v in zip(CPU_KEYS, t, strict=True)}, "count": n}
            for t, n in sorted(ids.items(), key=lambda kv: [str(x) for x in kv[0]])
        ],
        "features_sets": [{"features": f, "count": n} for f, n in sorted(feats.items())],
    }
    if len(ids) == 1:
        out.update(out["identities"][0])
        out.pop("count", None)
    if len(feats) == 1:
        out["features"] = next(iter(feats)).split()
    return out


def _midr() -> list[str] | None:
    base = "/sys/devices/system/cpu"
    try:
        cpus = sorted(
            (d for d in os.listdir(base) if d.startswith("cpu") and d[3:].isdigit()),
            key=lambda d: int(d[3:]),
        )
    except OSError:
        return None
    vals = [_read(f"{base}/{c}/regs/identification/midr_el1", 64) for c in cpus]
    vals = [v.strip() for v in vals if v]
    return sorted(set(vals)) or None


def _hwcaps() -> dict[str, int] | None:
    try:
        libc = ctypes.CDLL(None)
        fn = libc.getauxval
    except (OSError, AttributeError):
        return None
    fn.restype, fn.argtypes = ctypes.c_ulong, [ctypes.c_ulong]
    return {"hwcap": int(fn(AT_HWCAP)), "hwcap2": int(fn(AT_HWCAP2))}


def _cgroup() -> dict[str, Any] | None:
    for ver, files in CGROUP_FILES.items():
        vals = {k: (_read(p, 256) or "").strip() or None for k, p in files.items()}
        if any(v is not None for v in vals.values()):
            return {"version": ver, **vals}
    return None


def _meminfo_kb() -> int | None:
    for line in (_read("/proc/meminfo", 8192) or "").splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def snapshot(env: dict[str, str] | None = None) -> dict[str, Any]:
    """One host record. Never raises: a field that cannot be read is None and named in `unknown`."""
    t0 = time.perf_counter()
    env = os.environ if env is None else env
    unknown: dict[str, str] = {}
    rec: dict[str, Any] = {"schema": SCHEMA}

    text = _read("/proc/cpuinfo")
    if text is None:
        rec["cpu"] = None
        unknown["cpu"] = "/proc/cpuinfo not readable"
    else:
        cpu = parse_cpuinfo(text)
        rec["cpu"] = cpu
        for k, _ in CPU_KEYS:
            if cpu.get(k) is None and len(cpu["identities"]) <= 1:
                unknown[f"cpu.{k}"] = "not present in /proc/cpuinfo"
        if not cpu["features_sets"]:
            unknown["cpu.features"] = "no Features line in /proc/cpuinfo"
    rec["midr_el1"] = _midr()
    if rec["midr_el1"] is None:
        unknown["midr_el1"] = "sysfs regs/identification/midr_el1 not exposed"
    rec["hwcap"] = _hwcaps()
    if rec["hwcap"] is None:
        unknown["hwcap"] = "getauxval not available"
    try:
        affinity: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
        unknown["cores.affinity"] = "sched_getaffinity not available"
    rec["cores"] = {
        "cpu_count": os.cpu_count(),
        "affinity": affinity,
        "online": (_read("/sys/devices/system/cpu/online", 256) or "").strip() or None,
        "possible": (_read("/sys/devices/system/cpu/possible", 256) or "").strip() or None,
    }
    rec["cgroup"] = _cgroup()
    if rec["cgroup"] is None:
        unknown["cgroup"] = "no cgroup v1/v2 cpu or memory limit file readable"
    rec["mem_total_kb"] = _meminfo_kb()
    if rec["mem_total_kb"] is None:
        unknown["mem_total_kb"] = "/proc/meminfo MemTotal not readable"
    u = platform.uname()
    rec["kernel"] = {"release": u.release, "version": u.version, "machine": u.machine}
    rec["python"] = platform.python_version()
    rec["lambda"] = {k: env.get(k) for k in LAMBDA_ENV}
    for k, v in rec["lambda"].items():
        if v is None:
            unknown[f"lambda.{k}"] = "not set in this environment"
    unknown["availability_zone"] = "not exposed inside a Lambda function"
    unknown["physical_host"] = "not exposed inside a Lambda function"
    unknown["processor_generation"] = "never inferred: only the identifiers above are recorded"
    rec["unknown"] = unknown
    rec["fingerprint"] = fingerprint(rec)
    rec["probe_s"] = round(time.perf_counter() - t0, 6)
    return rec


IDENTITY_FIELDS = ("cpu", "midr_el1", "hwcap", "cores", "cgroup", "mem_total_kb", "kernel")


def fingerprint(rec: dict[str, Any]) -> str:
    """16 hex digits of SHA-256 over the host-identity fields (not timings, not Lambda settings): two environments
    with the same fingerprint exposed the same CPU identity, features, HWCAPs, cores, limits and kernel."""
    import hashlib
    import json

    body = json.dumps({k: rec.get(k) for k in IDENTITY_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]
