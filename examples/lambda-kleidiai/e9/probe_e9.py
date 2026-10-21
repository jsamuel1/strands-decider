"""E9 offline probe: load a GGUF as the Lambda does and report load phases, RSS, answers and payload hashes.

  python probe_e9.py <pkg_dir> <gguf> <requests.json> <out.json> [--repeat N]

Runs inside the pinned public.ecr.aws/lambda/python:3.12 image (arm64) with an unzipped function.zip at <pkg_dir>.
Like the Lambda, the GGUF is copied into a memfd and mmapped through /proc/self/fd/N by decshim. With E9_DUMP_DIR
set, the E9-patched libraries write every Q8_0 tensor's packed payload there; this probe hashes them (sha256 per
tensor and over all of them in name order) and deletes the files, so baseline and E9 loads can be compared.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import resource
import sys
import time


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return -1.0


def hash_dump(d: str) -> dict:
    per = {}
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            per[f[: -len(".bin")]] = hashlib.file_digest(fh, "sha256").hexdigest()
        os.remove(p)
    whole = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"count": len(per), "digest": whole, "per_tensor": per}


def main() -> None:
    pkg, gguf, reqs, out = sys.argv[1:5]
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    sys.path.insert(0, pkg)
    for k, v in (("OPENBLAS_NUM_THREADS", "1"), ("DEC_KV", "f32"), ("DEC_UBATCH", "512"), ("N_SEQ", "17")):
        os.environ.setdefault(k, v)
    threads = int(os.environ.get("THREADS", "6"))
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    import engine

    fd = libc.memfd_create(b"model", 0)
    t = time.monotonic()
    with open(gguf, "rb") as f:
        while b := f.read(64 << 20):
            os.write(fd, b)
    mpath = f"/proc/self/fd/{fd}"
    res = {"gguf": os.path.basename(gguf), "bytes": os.path.getsize(gguf), "threads": threads,
           "memfd_copy_s": round(time.monotonic() - t, 3), "env": {k: os.environ.get(k) for k in
                                                                   ("E9_DIAG_NO_I8MM", "E9_DUMP_DIR")}, "loads": []}
    dump = os.environ.get("E9_DUMP_DIR")
    ph = ("ggml_backend_load", "model_load_mmap", "context_init", "dec_load_total")
    for i in range(repeat):
        t = time.monotonic()
        eng, shim = engine.load_engine(os.path.join(pkg, "lib", "libdecshim.so"), mpath, os.path.join(pkg, "assets"),
                                       threads)
        wall = time.monotonic() - t
        buf = (ctypes.c_int64 * 4)()
        shim.lib.dec_load_phases_ns(buf, 4)
        rec = {k: round(v / 1e9, 4) for k, v in zip(ph, buf, strict=True)}
        rec["load_engine_s"], rec["rss_mb"] = round(wall, 4), rss_mb()
        if i == 0:
            if dump:
                res["payloads"] = hash_dump(dump)
            d = json.load(open(reqs))
            answers, lat = {}, {}
            t = time.monotonic()
            eng.evaluate({"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}})
            rec["first_inference_s"] = round(time.monotonic() - t, 4)
            for r in d["requests"]:
                t = time.monotonic()
                answers[r["id"]] = eng.evaluate(r["request"]).model_dump(mode="json")["answers"]
                lat[r["id"]] = round(time.monotonic() - t, 4)
            res["answers"], res["latency_s"] = answers, lat
            res["cpu_part"], res["ggml_cpu"] = shim.init_info["cpu_part"], shim.init_info["ggml_cpu"]
        elif dump:
            hash_dump(dump)
        res["loads"].append(rec)
        shim.lib.dec_free.argtypes = [ctypes.c_void_p]
        shim.lib.dec_free(shim.h)
    res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("answers", "payloads")}, sort_keys=True))
    if "payloads" in res:
        print("payloads", res["payloads"]["count"], res["payloads"]["digest"])


if __name__ == "__main__":
    main()
