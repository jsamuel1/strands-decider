"""Memory accounting for E11, standard library only (runs in the Lambda function and in the offline probe).

snapshot()     /proc/self/status memory fields, /proc/meminfo (MemTotal, Shmem, ...) and /proc/self/smaps_rollup, in
               kB, plus the cgroup's memory.current
               and memory.stat shmem/anon/file where the sandbox exposes them. Unreadable values are "unknown".
mappings_of()  the /proc/self/smaps entries whose path matches (the memfd's mapping): count, Size and Rss in kB.
fd_resident_bytes() resident bytes of a file (the memfd), by mincore(2) over a temporary view.
"""

from __future__ import annotations

import ctypes
import mmap

UNKNOWN = "unknown"
# lkai_memfd_release_stats (e11/patch_memfdrel.py), in slot order.
RELEASE_STAT_KEYS = (
    "released_bytes",
    "kept_bytes",
    "release_us",
    "errno",
    "ranges",
    "files",
    "ranges_via_madvise_remove",
)


def release_stats(libllama_path: str) -> dict | str:
    """The loader's release record, or 'absent' for a libllama without the patch."""
    try:
        lib = ctypes.CDLL(libllama_path)
        arr = (ctypes.c_int64 * len(RELEASE_STAT_KEYS)).in_dll(lib, "lkai_memfd_release_stats")
    except (OSError, ValueError):
        return "absent"
    return dict(zip(RELEASE_STAT_KEYS, list(arr), strict=True))


STATUS_KEYS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap")
MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Shmem", "Cached", "AnonPages")


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def parse_kb_fields(text: str | None, keys: tuple[str, ...] | None = None) -> dict:
    """'Name:   123 kB' lines -> {Name: 123}. With keys, only those (missing ones 'unknown')."""
    out: dict = {}
    for line in (text or "").splitlines():
        k, sep, v = line.partition(":")
        parts = v.split()
        if not sep or not parts or not parts[0].isdigit():
            continue
        out[k.strip()] = int(parts[0])
    if keys is None:
        return out
    return {k: out.get(k, UNKNOWN) for k in keys}


def parse_memory_stat(text: str | None) -> dict:
    """cgroup v2 memory.stat 'key value' lines -> the anon/file/shmem byte counters."""
    want = ("anon", "file", "shmem", "file_mapped")
    vals = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(" ")
        if k in want and v.strip().isdigit():
            vals[k] = int(v)
    return {k: vals.get(k, UNKNOWN) for k in want}


def snapshot() -> dict:
    cur = _read("/sys/fs/cgroup/memory.current")
    return {
        "status_kb": parse_kb_fields(_read("/proc/self/status"), STATUS_KEYS),
        # /proc/meminfo inside the Lambda sandbox is the sandbox's own: Shmem counts the memfd's pages whether or not
        # any process maps them, which RssShmem (mapped pages only) does not.
        "meminfo_kb": parse_kb_fields(_read("/proc/meminfo"), MEMINFO_KEYS),
        "smaps_rollup_kb": parse_kb_fields(_read("/proc/self/smaps_rollup")) or UNKNOWN,
        "cgroup_memory_current": int(cur) if cur and cur.strip().isdigit() else UNKNOWN,
        "cgroup_memory_stat": parse_memory_stat(_read("/sys/fs/cgroup/memory.stat")),
    }


def parse_smaps(text: str | None, needle: str) -> dict:
    """Sum the Size/Rss/Shared of every smaps entry whose header line contains needle."""
    entries, cur = [], None
    for line in (text or "").splitlines():
        head = line.split()
        if head and "-" in head[0] and len(head) >= 5 and not head[0].endswith(":"):
            cur = {"path": " ".join(head[5:]), "range": head[0]} if needle in line else None
            if cur is not None:
                entries.append(cur)
            continue
        if cur is not None:
            k, sep, v = line.partition(":")
            parts = v.split()
            if sep and parts and parts[0].isdigit():
                cur[k.strip()] = int(parts[0])
    keys = ("Size", "Rss", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty")
    summary = {
        "entries": len(entries),
        **{k + "_kb": sum(e.get(k, 0) for e in entries) for k in keys},
    }
    return {"summary": summary, "entries": entries[:64]}


def mappings_of(needle: str) -> dict:
    return parse_smaps(_read("/proc/self/smaps"), needle)


_libc = None


def _lib():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        _libc.mmap.restype = ctypes.c_void_p
        _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return _libc


def fd_resident_bytes(fd: int, size: int) -> int:
    """Resident bytes of a file's first `size` bytes: mincore(2) over a temporary read-only shared view. Mapping a
    shmem file faults nothing in, and mincore reports page-cache residency, so this is the memfd's own footprint."""
    if size <= 0:
        return 0
    lib = _lib()
    addr = lib.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap")
    try:
        page = mmap.PAGESIZE
        vec = (ctypes.c_ubyte * ((size + page - 1) // page))()
        if lib.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        return sum(b & 1 for b in vec) * page
    finally:
        lib.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
