"""E11 premise check: after a load done the way the Lambda does it, how much resident memory is the weight memfd?

  python premise_probe.py <pkg_dir> <gguf> <out.json> [<requests.json>]

Runs inside the pinned Lambda python3.12 image (arm64) with P1's unzipped function.zip at <pkg_dir>. It copies the
GGUF into a memfd (as loader.download_to_memfd does, minus S3), loads it through the packaged engine (decshim,
llama.cpp mmap of /proc/self/fd/N, KleidiAI repack), runs the handler's warm-up request, and records:

- /proc/self/status (VmRSS, RssAnon, RssFile, RssShmem, VmHWM) and /proc/self/smaps_rollup,
- the smaps entries of the memfd mapping (what llama.cpp still maps after its own post-load munmap),
- the memfd's resident size from its own point of view (mincore over a fresh private view of the whole file),
- the CPU identity and environment metadata observable here (cpuinfo, HWCAPs, nproc, cgroup, kernel),
- with E11's zip, the loader's lkai_memfd_release_stats (LKAI_RELEASE_MEMFD=1 releases; unset is P1's behaviour),
- with <requests.json>, the answers to every fixed request after the load (for the bit-identity check).

It changes nothing in the package; it only measures.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time

pkg, gguf, out = sys.argv[1:4]
reqs_path = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, pkg)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostmeta  # noqa: E402
import memstat  # noqa: E402

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
libc.memfd_create.restype = ctypes.c_int


def to_memfd(path: str) -> int:
    fd = libc.memfd_create(os.path.basename(path).encode(), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    size = os.path.getsize(path)
    os.ftruncate(fd, size)
    with open(path, "rb") as f:
        pos = 0
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            os.pwrite(fd, b, pos)
            pos += len(b)
    return fd


def memfd_resident(fd: int) -> int:
    """Bytes of the memfd's pages that are resident (mincore over a temporary shared view)."""
    return memstat.fd_resident_bytes(fd, os.fstat(fd).st_size)


res: dict = {
    "gguf": os.path.basename(gguf),
    "gguf_bytes": os.path.getsize(gguf),
    "host": hostmeta.collect(),
}
res["mem_start"] = memstat.snapshot()
t = time.monotonic()
fd = to_memfd(gguf)
res["memfd_copy_s"] = round(time.monotonic() - t, 3)
res["mem_after_memfd"] = memstat.snapshot()
res["memfd_resident_after_copy"] = memfd_resident(fd)

import engine  # noqa: E402

threads = int(os.environ.get("THREADS", "6"))
t = time.monotonic()
eng, shim = engine.load_engine(
    os.path.join(pkg, "lib", "libdecider_hidden.so"),
    f"/proc/self/fd/{fd}",
    os.path.join(pkg, "assets"),
    threads,
)
res["load_engine_s"] = round(time.monotonic() - t, 3)
eng.evaluate(
    {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
)
res["mem_after_load"] = memstat.snapshot()
res["memfd_mappings_after_load"] = memstat.mappings_of(f"/memfd:{os.path.basename(gguf)}")
res["memfd_resident_after_load"] = memfd_resident(fd)
res["init_info"] = shim.init_info
res["release_flag"] = os.environ.get("LKAI_RELEASE_MEMFD", "")
try:
    arr = (ctypes.c_int64 * 6).in_dll(
        ctypes.CDLL(os.path.join(pkg, "lib", "libllama.so")), "lkai_memfd_release_stats"
    )
    res["release"] = dict(
        zip(
            ("released_bytes", "kept_bytes", "release_us", "errno", "ranges", "files"),
            list(arr),
            strict=True,
        )
    )
except ValueError:
    res["release"] = "absent (P1 zip)"
if reqs_path:
    answers, lat = {}, {}
    for r in json.load(open(reqs_path))["requests"]:
        t = time.monotonic()
        answers[r["id"]] = eng.evaluate(r["request"]).model_dump(mode="json")["answers"]
        lat[r["id"]] = round(time.monotonic() - t, 4)
    res["answers"], res["latency_s"] = answers, lat
    res["mem_after_requests"] = memstat.snapshot()
res["platform"] = platform.platform()
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
a, b = res["mem_after_load"]["status_kb"], res["memfd_resident_after_load"]
print(
    json.dumps(
        {
            "VmRSS_kb": a.get("VmRSS"),
            "RssAnon_kb": a.get("RssAnon"),
            "RssShmem_kb": a.get("RssShmem"),
            "memfd_resident_MiB": round(b / 2**20, 1),
            "mapped": res["memfd_mappings_after_load"]["summary"],
            "release": res["release"],
            "load_engine_s": res["load_engine_s"],
        }
    )
)
