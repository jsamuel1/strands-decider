"""E8 offline answers probe (NOT Lambda): load P1's GGUF as the function does, with one zip's libraries and the switch
as set in the environment, then answer P1's fixed requests and compare them with P1's reference for this CPU part.

  python probe_e8.py <pkg_dir> <gguf> <requests.json> <reference outputs.json> <out.json>

Runs inside P1's pinned Lambda image with an unzipped base.zip or off.zip at <pkg_dir> (probe_e8.sh). The GGUF is
copied into a memfd and mapped as /proc/self/fd/N, as PR #77's loader does. Phases are decshim's own timers (P1's).
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

pkg, gguf, reqs, ref_path, out = sys.argv[1:6]
sys.path.insert(0, pkg)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
threads = int(os.environ.get("THREADS", "6"))

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
libc.memfd_create.restype = ctypes.c_int


def to_memfd(path: str) -> str:
    fd = libc.memfd_create(b"model", 0)
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            os.write(fd, b)
    return f"/proc/self/fd/{fd}"


import e8_host  # noqa: E402
import engine  # noqa: E402

PH = ("ggml_backend_load", "model_load_mmap", "context_init", "dec_load_total")
res: dict = {
    "evidence": "OFFLINE, NOT LAMBDA",
    "arm_env": os.environ.get(e8_host.SWITCH, "unset"),
    "threads": threads,
    "host": e8_host.host_metadata(),
}
mpath = to_memfd(gguf)
t = time.monotonic()
eng, shim = engine.load_engine(
    os.path.join(pkg, "lib", "libdecshim.so"), mpath, os.path.join(pkg, "assets"), threads
)
res["load_engine_s"] = round(time.monotonic() - t, 4)
buf = (ctypes.c_int64 * 4)()
shim.lib.dec_load_phases_ns(buf, 4)
res["phases_s"] = {k: round(v / 1e9, 4) for k, v in zip(PH, buf, strict=True)}
res["cpu_part"] = part = shim.init_info["cpu_part"]
res["ggml_cpu"] = shim.init_info["ggml_cpu"]
res["status_kb"] = e8_host.proc_status_kb(e8_host.read_text("/proc/self/status"))
eng.evaluate(
    {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
)
ref = json.load(open(ref_path))["outputs"]
answers, cmp = {}, {}
for r in json.load(open(reqs))["requests"]:
    a = eng.evaluate(r["request"]).model_dump(mode="json")["answers"]
    answers[r["id"]] = a
    want = (ref.get(r["id"]) or {}).get(part)
    cmp[r["id"]] = (
        "no-reference-for-part" if want is None else ("identical" if a == want else "DIFFERS")
    )
res["answers"], res["vs_reference"] = answers, cmp
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print(
    json.dumps(
        {k: res[k] for k in ("arm_env", "cpu_part", "phases_s", "vs_reference")}, sort_keys=True
    )
)
