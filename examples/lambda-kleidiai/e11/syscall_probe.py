"""E11 Lambda syscall probe (handler syscall_probe.handler): which ways of freeing a memfd's pages the sandbox allows.

On a 64 MiB memfd, filled, it tries in turn and reports errno and the cgroup/status shmem change of each:
  fallocate(FALLOC_FL_PUNCH_HOLE | KEEP_SIZE)      the card's mechanism
  madvise(MADV_REMOVE) on a writable shared view   frees the backing store of the range
  ftruncate(fd, smaller)                           frees everything past the new size
plus mincore. Deployed only for this check, then deleted.
"""

import ctypes
import mmap
import os

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.mmap.restype = ctypes.c_void_p
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
MB = 1 << 20
MADV_REMOVE = 9


def shmem_kb():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("RssShmem", "VmRSS")):
            k, v = line.split(":")
            out[k] = int(v.split()[0])
    try:
        for line in open("/sys/fs/cgroup/memory.stat"):
            if line.startswith("shmem "):
                out["cg_shmem_kb"] = int(line.split()[1]) // 1024
    except OSError:
        out["cg_shmem_kb"] = "unknown"
    try:
        out["meminfo_shmem_kb"] = next(
            int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("Shmem:")
        )
    except (OSError, StopIteration):
        out["meminfo_shmem_kb"] = "unknown"
    return out


def filled(size):
    fd = os.memfd_create("probe")
    os.ftruncate(fd, size)
    for off in range(0, size, MB):
        os.pwrite(fd, b"x" * MB, off)
    return fd


def handler(event, context):
    size = 64 * MB
    res = {"baseline": shmem_kb()}
    fd = filled(size)
    res["filled"] = shmem_kb()
    r = libc.fallocate(fd, 0x02 | 0x01, 0, 32 * MB)
    res["fallocate_punch"] = {"rc": r, "errno": ctypes.get_errno() if r else 0, "after": shmem_kb()}
    addr = libc.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        res["mmap_rw"] = {"errno": ctypes.get_errno()}
    else:
        r = libc.madvise(ctypes.c_void_p(addr), 32 * MB, MADV_REMOVE)
        res["madvise_remove"] = {
            "rc": r,
            "errno": ctypes.get_errno() if r else 0,
            "after": shmem_kb(),
        }
        vec = (ctypes.c_ubyte * (size // mmap.PAGESIZE))()
        r = libc.mincore(ctypes.c_void_p(addr), size, vec)
        res["mincore"] = {
            "rc": r,
            "errno": ctypes.get_errno() if r else 0,
            "resident_mb": sum(b & 1 for b in vec) * mmap.PAGESIZE / MB if r == 0 else None,
        }
    try:
        os.ftruncate(fd, 16 * MB)
        res["ftruncate"] = {"rc": 0, "after": shmem_kb()}
    except OSError as e:
        res["ftruncate"] = {"errno": e.errno}
    return res
