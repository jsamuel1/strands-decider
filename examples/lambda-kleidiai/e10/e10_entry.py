"""Lambda entry point of every E10 function: handler "e10_entry.handler". One zip for all arms.

E10's one variable is the llama.cpp context size, n_ctx. P1 hard-codes it in snapstart/engine.py's load_engine
(`n_ctx=4224 if n_seq == 1 else 16384`), which passes it to engine.Shim and from there to decshim's dec_load_seq.
This wrapper replaces that argument with $E10_N_CTX. It does so with an import hook on
the module `engine`: when P1's lkai_entry imports engine, the hook wraps Shim before lkai_entry subclasses it.
Shim keeps the value as self.n_ctx, so its fork grouping and decshim's bounds use the same capacity. The one other
addition is a refusal: a request whose state plus longest question exceeds n_ctx gets a ValueError (HTTP 422) with
its token counts, before any decode. P1 would raise RuntimeError from decshim's rc=-2 there (an HTTP 500); with
P1's n_ctx=16384 above the 4,096-token window the refusal can never fire, so the control arm runs P1's exact path.

It also records the host metadata the HOST VARIABILITY rules ask for, read with the standard library before
lkai_entry starts its clock, and the process's memory high-water marks (VmHWM/VmRSS, read after the first
response is built, outside every phase timer). A field that cannot be read is "unknown", never guessed.

The first invocation of each environment carries the record as `lkai.e10` in its response and as an `E10 {json}`
log line. Later invocations carry only P1's own record.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import platform
import struct
import sys

UNKNOWN = "unknown"
AT_HWCAP, AT_HWCAP2 = 16, 26
N_CTX_ENV = "E10_N_CTX"


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


def proc_memory(text: str | None) -> dict:
    """VmHWM and VmRSS (KiB) from /proc/self/status."""
    out = {"vm_hwm_kib": UNKNOWN, "vm_rss_kib": UNKNOWN}
    names = {"VmHWM": "vm_hwm_kib", "VmRSS": "vm_rss_kib"}
    for line in (text or "").splitlines():
        k, _, v = line.partition(":")
        if k in names and v.strip().endswith("kB"):
            out[names[k]] = int(v.split()[0])
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
        "e10_arm": os.environ.get("E10_ARM", UNKNOWN),
    }


def n_ctx_override(env: dict | None = None) -> int:
    """$E10_N_CTX as a positive int. A missing or invalid value fails the init: an arm must never silently run
    P1's default and be reported as another size."""
    raw = (os.environ if env is None else env).get(N_CTX_ENV)
    try:
        v = int(raw) if raw is not None else 0
    except ValueError:
        v = 0
    if v <= 0:
        raise RuntimeError(f"{N_CTX_ENV}={raw!r}: set a positive context size")
    return v


SEEN: dict = {"requested": [], "applied": []}


class ContextTooSmall(ValueError):
    """A request whose state plus longest question exceeds this deployment's n_ctx. A ValueError, so PR #77's HTTP
    handler answers 422 and a direct invocation fails with this message; raised before any token is decoded, so
    nothing is truncated."""


def check_fits(n_state: int, q_lens: list[int], n_ctx: int) -> None:
    """Shim.forward_shared needs one forward to hold the state plus at least one whole question (fork groups are
    cut to fit). The window fit (max_length 4096) has already trimmed the state, so this only refuses when
    n_ctx < max_length."""
    need = n_state + max(q_lens, default=0)
    if need > n_ctx:
        raise ContextTooSmall(
            f"request needs {need} tokens of context (state {n_state} + longest question {max(q_lens)}); this "
            f"deployment's context is {n_ctx} tokens (n_ctx). Shorten the state by at least {need - n_ctx} tokens."
        )


def wrap_shim(shim_cls: type, n_ctx: int) -> None:
    """Replace the n_ctx argument of Shim.__init__(self, lib_path, model_path, threads, n_ctx, n_seq), and refuse
    an over-context request in forward_shared before it reaches decshim."""
    orig = shim_cls.__init__
    orig_fwd = shim_cls.forward_shared

    def __init__(self, lib_path, model_path, threads, *args, **kwargs):
        if args:
            SEEN["requested"].append(args[0])
            args = (n_ctx, *args[1:])
        else:
            SEEN["requested"].append(kwargs.get("n_ctx"))
            kwargs["n_ctx"] = n_ctx
        SEEN["applied"].append(n_ctx)
        orig(self, lib_path, model_path, threads, *args, **kwargs)

    def forward_shared(self, state_ids, q_ids, rows=None):
        check_fits(len(state_ids), [len(q) for q in q_ids], self.n_ctx)
        return orig_fwd(self, state_ids, q_ids, rows)

    shim_cls.__init__ = __init__
    shim_cls.forward_shared = forward_shared


class _EngineHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds the top-level module `engine` with the normal path finder, then wraps Shim after it executes."""

    def __init__(self, n_ctx: int):
        self.n_ctx = n_ctx

    def find_spec(self, name, path=None, target=None):
        if name != "engine":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        self.inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        wrap_shim(module.Shim, self.n_ctx)
        sys.meta_path.remove(self)


META = host_metadata()
N_CTX = n_ctx_override()
sys.meta_path.insert(0, _EngineHook(N_CTX))

# P1's instrumented entry: its clock starts at its own import, after the hook is installed.
import lkai_entry  # noqa: E402

_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)
    _n["n"] += 1
    if _n["n"] == 1:
        rec = (
            META
            | {"n_ctx": {"env": N_CTX, **SEEN}}
            | proc_memory(_read("/proc/self/status"))
            | {"log_stream": getattr(context, "log_stream_name", UNKNOWN)}
        )
        sys.stdout.write("E10 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e10"] = rec
    return out
