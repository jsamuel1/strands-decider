"""What can be profiled here? A standard-library probe of the host it runs on, for the P3 applicability matrix.

Runs inside a Lambda function (handler "lkai_probe.handler", or the {"mode": "lkai_probe"} event of
lkai_prof.handler) and on any Linux host (`python3 lkai_probe.py`). It reports facts, not conclusions:

  hardware     uname, /proc/cpuinfo (CPU part, features), memory, PCI vendors, /dev/nvidia*, /dev/dri
  kernel knobs perf_event_paranoid, yama ptrace_scope, kptr_restrict, unprivileged_bpf_disabled, PMU devices
  process      capabilities, seccomp mode, NoNewPrivs, uid
  syscalls     perf_event_open (hardware cycles, software cpu-clock, task-clock; user space only), bpf(MAP_CREATE),
               ptrace attach/seize of our own child, process_vm_readv of self, child and parent, setitimer(ITIMER_PROF)
  tools        perf, nvidia-smi, nsys, ncu, gdb, strace on PATH; libcuda / libnvidia-ml loadable; py-spy if shipped
  py-spy       `py-spy dump` of a child python and of this process (with and without PR_SET_PTRACER)

Every attempt records its errno name. Nothing here needs privileges, nothing is left attached or stopped: ptrace is
only tried on a child this probe started and kills, and the parent is only read with process_vm_readv. Environment
variables are reported by name for an allow-list only, never by value for anything that could hold a credential.
"""

from __future__ import annotations

import ctypes
import errno
import glob
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long

# Linux syscall numbers. aarch64 uses the generic table; x86_64 its own.
_NR = {
    "aarch64": {"perf_event_open": 241, "bpf": 280, "process_vm_readv": 270, "ptrace": 117},
    "x86_64": {"perf_event_open": 298, "bpf": 321, "process_vm_readv": 310, "ptrace": 101},
}.get(platform.machine(), {})

PERF_TYPE_HARDWARE, PERF_TYPE_SOFTWARE = 0, 1
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_SW_CPU_CLOCK, PERF_COUNT_SW_TASK_CLOCK = 0, 1
PTRACE_ATTACH, PTRACE_DETACH, PTRACE_SEIZE = 16, 17, 0x4206
PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1).value
ENV_NAMES = (
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_MEMORY_SIZE",
    "AWS_LAMBDA_INITIALIZATION_TYPE",
    "AWS_REGION",
    "AWS_LAMBDA_RUNTIME_API",
    "_X_AMZN_TRACE_ID",
    "LAMBDA_TASK_ROOT",
    "LKAI_PROF",
)


def _sys(name: str, *args) -> int:
    """syscall(2) with every argument widened to a C long: ctypes passes a bare Python int to a variadic function as
    a 32-bit int, which leaves the upper half of the argument register undefined (process_vm_readv then sees nonzero
    flags and fails with EINVAL)."""
    conv = [ctypes.c_long(a) if isinstance(a, int) else a for a in args]
    return _libc.syscall(ctypes.c_long(_NR[name]), *conv)


def _err(e: int) -> str:
    return errno.errorcode.get(e, str(e))


def _read(path: str, limit: int = 4096) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read(limit)
    except OSError as e:
        return f"<{_err(e.errno)}>"


class _PerfAttr(ctypes.Structure):
    # struct perf_event_attr, PERF_ATTR_SIZE_VER5 (112 bytes) is enough for the fields used here.
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("_reserved", ctypes.c_uint16),
    ]


def _busy(seconds: float) -> int:
    n, end = 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        n += 1
    return n


def perf_event_open(kind: str, ptype: int, config: int) -> dict:
    """Open a user-space-only counter on this thread, spin 50 ms, read it."""
    if "perf_event_open" not in _NR:
        return {"event": kind, "result": "unsupported-arch"}
    a = _PerfAttr()
    a.type, a.size, a.config = ptype, ctypes.sizeof(_PerfAttr), config
    a.flags = (1 << 0) | (1 << 5) | (1 << 6)  # disabled, exclude_kernel, exclude_hv
    fd = _sys("perf_event_open", ctypes.byref(a), 0, -1, -1, 0)
    if fd < 0:
        return {"event": kind, "result": "denied", "errno": _err(ctypes.get_errno())}
    try:
        import fcntl

        fcntl.ioctl(fd, 0x2400)  # PERF_EVENT_IOC_ENABLE
        _busy(0.05)
        fcntl.ioctl(fd, 0x2401)  # PERF_EVENT_IOC_DISABLE
        value = struct.unpack("Q", os.read(fd, 8))[0]
        return {"event": kind, "result": "opened", "value_after_50ms_spin": value}
    except OSError as e:
        return {"event": kind, "result": "opened-but-read-failed", "errno": _err(e.errno)}
    finally:
        os.close(fd)


def bpf_map_create() -> dict:
    if "bpf" not in _NR:
        return {"result": "unsupported-arch"}
    # union bpf_attr for BPF_MAP_CREATE: map_type=BPF_MAP_TYPE_ARRAY(2), key_size=4, value_size=4, max_entries=1
    attr = ctypes.create_string_buffer(struct.pack("IIII", 2, 4, 4, 1) + b"\0" * 112)
    fd = _sys("bpf", 0, attr, len(attr.raw))
    if fd < 0:
        return {"result": "denied", "errno": _err(ctypes.get_errno())}
    os.close(fd)
    return {"result": "allowed"}


class _IoVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def _first_readable_addr(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                lo, perms = line.split()[0].split("-")[0], line.split()[1]
                if perms.startswith("r"):
                    return int(lo, 16)
    except OSError:
        return None
    return None


def process_vm_readv(pid: int) -> dict:
    if "process_vm_readv" not in _NR:
        return {"result": "unsupported-arch"}
    addr = _first_readable_addr(pid)
    if addr is None:
        return {"result": "maps-unreadable", "errno": _err(errno.EACCES)}
    buf = ctypes.create_string_buffer(16)
    local, remote = _IoVec(ctypes.addressof(buf), 16), _IoVec(addr, 16)
    n = _sys("process_vm_readv", pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
    if n < 0:
        return {"result": "denied", "errno": _err(ctypes.get_errno())}
    return {"result": "allowed", "bytes": n}


def _ptrace(req: int, pid: int) -> int:
    return _sys("ptrace", req, pid, 0, 0)


def ptrace_child() -> dict:
    """PTRACE_SEIZE and PTRACE_ATTACH, each on a fresh child we started; every child is killed afterwards."""
    if "ptrace" not in _NR:
        return {"result": "unsupported-arch"}
    out: dict = {}
    for name, req in (("seize", PTRACE_SEIZE), ("attach", PTRACE_ATTACH)):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            time.sleep(0.2)
            if name == "seize":
                out["child_vm_readv"] = process_vm_readv(child.pid)
            rc = _ptrace(req, child.pid)
            out[name] = (
                {"result": "denied", "errno": _err(ctypes.get_errno())} if rc < 0 else {"result": "allowed"}
            )
        finally:
            child.kill()  # a traced child dies too; nothing stays attached
            child.wait()
    return out


def itimer_prof() -> dict:
    """SIGPROF from setitimer(ITIMER_PROF) at 1 kHz of CPU time, counted over a 0.2 s spin."""
    hits = [0]

    def on(_s, _f):
        hits[0] += 1

    old = signal.signal(signal.SIGPROF, on)
    try:
        signal.setitimer(signal.ITIMER_PROF, 0.001, 0.001)
        _busy(0.2)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0, 0)
        signal.signal(signal.SIGPROF, old)
    return {"result": "delivered" if hits[0] else "none", "signals_in_0.2s_cpu": hits[0]}


def py_spy(binary: str | None) -> dict:
    """py-spy dump of a child python, then of this process, without and with PR_SET_PTRACER_ANY."""
    if not binary:
        return {"result": "not-shipped"}
    out: dict = {"binary": binary}
    try:
        out["version"] = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"result": "not-runnable", "detail": str(e)[:200]}
    child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"])
    try:
        time.sleep(0.3)
        out["dump_child"] = _spy(binary, child.pid)
    finally:
        child.kill()
        child.wait()
    out["dump_self"] = _spy(binary, os.getpid())
    # --native unwinds C frames, which needs the target stopped (ptrace); without --nonblocking py-spy also pauses it.
    child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"])
    try:
        time.sleep(0.3)
        out["dump_child_native"] = _spy(binary, child.pid, ["--native"])
        out["dump_child_blocking"] = _spy(binary, child.pid, [])
    finally:
        child.kill()
        child.wait()
    rc = _libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    out["pr_set_ptracer_any"] = "ok" if rc == 0 else _err(ctypes.get_errno())
    out["dump_self_after_pr_set_ptracer"] = _spy(binary, os.getpid())
    return out


def _spy(binary: str, pid: int, flags: list[str] | None = None) -> dict:
    flags = ["--nonblocking"] if flags is None else flags
    try:
        r = subprocess.run([binary, "dump", *flags, "--pid", str(pid)], capture_output=True, text=True,
                           timeout=20)
    except subprocess.TimeoutExpired:
        return {"result": "timeout"}
    ok = r.returncode == 0 and "Thread" in r.stdout
    return {"result": "works" if ok else "failed", "rc": r.returncode,
            "stdout_head": r.stdout[:300], "stderr_head": r.stderr[:400]}


def _status_fields() -> dict:
    keep = ("Uid", "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs", "Seccomp", "Seccomp_filters")
    out = {}
    for line in (_read("/proc/self/status", 8192) or "").splitlines():
        k, _, v = line.partition(":")
        if k in keep:
            out[k] = v.strip()
    return out


def _cpuinfo() -> dict:
    text = _read("/proc/cpuinfo", 65536) or ""
    first = text.split("\n\n", 1)[0]
    fields = dict(
        (k.strip(), v.strip()) for k, _, v in (ln.partition(":") for ln in first.splitlines()) if k.strip()
    )
    return {"processors": text.count("processor"), "first": fields}


def _pci_vendors() -> dict:
    out: dict[str, int] = {}
    for p in glob.glob("/sys/bus/pci/devices/*/vendor"):
        v = (_read(p, 16) or "").strip()
        out[v] = out.get(v, 0) + 1
    return out


def _loadable(name: str) -> str:
    try:
        ctypes.CDLL(name)
        return "loaded"
    except OSError as e:
        return f"not-loadable: {str(e)[:120]}"


def probe(py_spy_path: str | None = None) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    if py_spy_path is None:
        cand = os.path.join(here, "bin", "py-spy")
        py_spy_path = cand if os.access(cand, os.X_OK) else shutil.which("py-spy")
    u = platform.uname()
    t0 = time.perf_counter()
    res = {
        "schema": "lkai-probe/1",
        "uname": {"system": u.system, "release": u.release, "version": u.version, "machine": u.machine},
        "python": sys.version.split()[0],
        "cpuinfo": _cpuinfo(),
        "nproc": os.cpu_count(),
        "meminfo_head": (_read("/proc/meminfo", 400) or "").splitlines()[:3],
        "gpu": {
            "dev_nvidia": sorted(glob.glob("/dev/nvidia*")),
            "dev_dri": sorted(glob.glob("/dev/dri/*")),
            "pci_vendors": _pci_vendors(),
            "nvidia_vendor_0x10de_present": "0x10de" in _pci_vendors(),
            "proc_driver_nvidia": os.path.exists("/proc/driver/nvidia"),
            "libcuda": _loadable("libcuda.so.1"),
            "libnvidia_ml": _loadable("libnvidia-ml.so.1"),
        },
        "kernel": {
            k: (_read(p, 64) or "").strip()
            for k, p in {
                "perf_event_paranoid": "/proc/sys/kernel/perf_event_paranoid",
                "yama_ptrace_scope": "/proc/sys/kernel/yama/ptrace_scope",
                "kptr_restrict": "/proc/sys/kernel/kptr_restrict",
                "unprivileged_bpf_disabled": "/proc/sys/kernel/unprivileged_bpf_disabled",
            }.items()
        },
        "pmu_devices": sorted(os.path.basename(p) for p in glob.glob("/sys/bus/event_source/devices/*")),
        "paths": {
            p: os.path.exists(p)
            for p in ("/sys/kernel/debug", "/sys/kernel/tracing", "/sys/fs/bpf", "/proc/kallsyms", "/opt/extensions")
        },
        "process": _status_fields(),
        "tools_on_path": {t: shutil.which(t) for t in ("perf", "nvidia-smi", "nsys", "ncu", "gdb", "strace", "ltrace")},
        "env_present": {k: (k in os.environ) for k in ENV_NAMES},
        "env_values": {k: os.environ.get(k) for k in ("AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_MEMORY_SIZE",
                                                     "AWS_LAMBDA_INITIALIZATION_TYPE", "AWS_REGION", "LKAI_PROF")},
        "clock": {c: str(time.get_clock_info(c)) for c in ("monotonic", "perf_counter", "process_time")},
        "syscalls": {
            "perf_event_open": [
                perf_event_open("hw-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
                perf_event_open("sw-cpu-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK),
                perf_event_open("sw-task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK),
            ],
            "bpf_map_create": bpf_map_create(),
            "process_vm_readv_self": process_vm_readv(os.getpid()),
            "process_vm_readv_parent": process_vm_readv(os.getppid()),
            "ptrace_child": ptrace_child(),
            "itimer_prof_sigprof": itimer_prof(),
        },
        "py_spy": py_spy(py_spy_path),
    }
    res["probe_s"] = round(time.perf_counter() - t0, 3)
    return res


def handler(event, context):  # Lambda entry point
    res = probe()
    res["request_id"] = getattr(context, "aws_request_id", None)
    print("LKAI_PROBE " + json.dumps(res, separators=(",", ":")), flush=True)
    return res


if __name__ == "__main__":
    print(json.dumps(probe(), indent=1))
