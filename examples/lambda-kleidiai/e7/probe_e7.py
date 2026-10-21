"""E7 offline probe (NOT Lambda): load P1's GGUF the way the function does and report one arm's load phases, memory,
mechanism readout and answers. One process per load, so every load starts from a fresh process and reads the switch
at load time.

  python probe_e7.py <pkg_dir> <gguf> <requests.json> <out.json>

Runs inside P1's pinned Lambda image (public.ecr.aws/lambda/python:3.12, arm64) with the unzipped E7 function.zip at
<pkg_dir>. The arm is LKAI_SKIP_TIED_OUTPUT in the environment (probe_e7.sh sets it). Every other variable defaults to
P1's function environment (DEC_KV=f32, DEC_UBATCH=512, N_SEQ=17, OPENBLAS_NUM_THREADS=1). The GGUF is copied into a
memfd
and mapped as /proc/self/fd/N, as PR #77's loader does. The phases are decshim's own timers (P1's), the memory is
VmRSS / VmHWM from /proc/self/status, and the answers are the complete answer objects for P1's fixed requests.
"""

from __future__ import annotations

import ctypes
import json
import os
import resource
import sys
import time

pkg, gguf, reqs, out = sys.argv[1:5]
sys.path.insert(0, pkg)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# P1's function environment (baseline/deploy.py); the answers depend on DEC_KV and DEC_UBATCH.
os.environ.setdefault("DEC_KV", "f32")
os.environ.setdefault("DEC_UBATCH", "512")
os.environ.setdefault("N_SEQ", "17")
threads = int(os.environ.get("THREADS", "6"))

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
libc.memfd_create.restype = ctypes.c_int


def status_kb() -> dict:
    d = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
            d[k] = int(v.split()[0])
    return d


def to_memfd(path: str) -> str:
    fd = libc.memfd_create(b"model", 0)
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            os.write(fd, b)
    return f"/proc/self/fd/{fd}"


# the function's own metadata and mechanism readers
import e7_host  # noqa: E402
import engine  # noqa: E402

PH = ("ggml_backend_load", "model_load_mmap", "context_init", "dec_load_total")
res: dict = {
    "arm_env": os.environ.get("LKAI_SKIP_TIED_OUTPUT", "unset"),
    "threads": threads,
    "env": {k: os.environ[k] for k in ("DEC_KV", "DEC_UBATCH", "N_SEQ", "OPENBLAS_NUM_THREADS")},
    "host": e7_host.host_metadata(),
}
t = time.monotonic()
mpath = to_memfd(gguf)
res["memfd_copy_s"] = round(time.monotonic() - t, 3)
res["status_after_memfd_kb"] = status_kb()
t = time.monotonic()
eng, shim = engine.load_engine(
    os.path.join(pkg, "lib", "libdecshim.so"), mpath, os.path.join(pkg, "assets"), threads
)
res["load_engine_s"] = round(time.monotonic() - t, 4)
buf = (ctypes.c_int64 * 4)()
shim.lib.dec_load_phases_ns(buf, 4)
res["phases_s"] = {k: round(v / 1e9, 4) for k, v in zip(PH, buf, strict=True)}
res["status_after_load_kb"] = status_kb()
res["mechanism"] = e7_host.mechanism(shim)
res["cpu_part"] = shim.init_info["cpu_part"]
res["ggml_cpu"] = shim.init_info["ggml_cpu"]
d = json.load(open(reqs))
t = time.monotonic()
eng.evaluate(
    {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
)
res["first_inference_s"] = round(time.monotonic() - t, 4)
answers, lat = {}, {}
for r in d["requests"]:
    t = time.monotonic()
    a = eng.evaluate(r["request"]).model_dump(mode="json")
    lat[r["id"]] = round(time.monotonic() - t, 4)
    answers[r["id"]] = a["answers"]
res["answers"], res["latency_s"] = answers, lat
res["status_after_requests_kb"] = status_kb()
res["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in res.items() if k not in ("answers", "host")}, sort_keys=True))
