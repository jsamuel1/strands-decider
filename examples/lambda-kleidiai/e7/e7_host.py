"""Host metadata and mechanism readers for E7, standard library only. Imported by e7_entry.py (the Lambda handler)
and probe_e7.py (offline), so both read the same fields the same way. Nothing here loads a model or starts a timer.
A field that cannot be read is "unknown", never guessed. No Graviton generation is inferred from any field.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26
BUFT_NAMES = ("CPU_KLEIDIAI", "CPU", "CPU_REPACK")


def read_text(path: str, limit: int = 1 << 16) -> str | None:
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
        v = read_text(f"/sys/fs/cgroup/{name}", 256)
        out[name] = v.strip() if v is not None else UNKNOWN
    return out


def proc_status_kb(text: str | None, fields: tuple[str, ...] = ("VmRSS", "VmHWM")) -> dict:
    """VmRSS / VmHWM from /proc/self/status, in kB."""
    out = dict.fromkeys(fields, UNKNOWN)
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
        affinity: int | str = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = UNKNOWN
    return {
        "cpu": cpuinfo(read_text("/proc/cpuinfo")),
        **auxv(raw),
        "os_cpu_count": os.cpu_count() or UNKNOWN,
        "sched_affinity": affinity,
        "cgroup": cgroup(),
        "kernel": platform.release() or UNKNOWN,
        "python": platform.python_version(),
        "execution_env": os.environ.get("AWS_EXECUTION_ENV", UNKNOWN),
        "init_type": os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE", UNKNOWN),
        "memory_mb": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", UNKNOWN),
        "skip_env": os.environ.get("LKAI_SKIP_TIED_OUTPUT", "unset"),
    }


def mechanism(shim) -> dict:
    """What the loaded model holds: the patched loader's skip flag and the bytes per weight buffer type.

    `shim.h` is decshim's dec_t *, whose first member is the llama_model *. libllama.so sits beside libdecshim.so
    (engine.Shim sets DEC_GGML_DIR to that directory) and is already loaded as its dependency, so CDLL returns the
    same library."""
    out: dict = {"tied_output_skipped": UNKNOWN, "buft_bytes": dict.fromkeys(BUFT_NAMES, UNKNOWN)}
    if shim is None or not getattr(shim, "h", None) or not os.environ.get("DEC_GGML_DIR"):
        return out
    try:
        lib = ctypes.CDLL(os.path.join(os.environ["DEC_GGML_DIR"], "libllama.so"))
        model = ctypes.c_void_p.from_address(shim.h).value
        f = lib.llama_model_lkai_tied_output_skipped
        f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_int32
        g = lib.llama_model_lkai_buft_bytes
        g.argtypes, g.restype = [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_uint64
    except (AttributeError, OSError, TypeError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["tied_output_skipped"] = bool(f(model))
    out["buft_bytes"] = {name: int(g(model, name.encode())) for name in BUFT_NAMES}
    return out
