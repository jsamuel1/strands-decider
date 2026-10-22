"""E14 offline driver, NOT LAMBDA: warm forward with P1's engine, a hash of every raw hidden-state row it returns.

  python offline_e14.py --pkg <unzipped function.zip, lib/ replaced or not> --gguf <s2b-h512-Q8_0.gguf> \
      --passes 10 --out run.json [--probe <libggml-cpu-*.so built with patch_concat_probe>]

Loads the engine exactly as P1's handler does (DEC_KV=f32, DEC_UBATCH=512, N_SEQ=17, 6 threads, OPENBLAS 1 thread,
max_request_tokens 24576), runs PR #77's warm-up, then every request in requests.json `passes` times in order. For each
request it records the wall time of engine.evaluate and a SHA-256 over the float32 bytes of every hidden-state array
the shim returned (before the pointer head), so two library builds can be compared bit for bit, not only by rounded
answers. It also records process CPU seconds per request (getrusage) beside wall time. --probe reads and resets
the concat probe's table after each steady-state request, outside that request's timed span.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time


def cpu_fields() -> dict:
    out: dict = {}
    try:
        for line in open("/proc/cpuinfo"):
            k, _, v = line.partition(":")
            k = k.strip()
            if (
                k in ("CPU implementer", "CPU part", "CPU variant", "CPU revision", "Features")
                and k not in out
            ):
                out[k] = v.strip()
    except OSError:
        pass
    return out


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def read_probe(probe, path: str) -> dict[str, list[int]]:
    """The probe table since its last reset: key -> [calls, ns, bytes]; then reset it."""
    probe.lkai_e14_probe_dump(path.encode())
    probe.lkai_e14_probe_reset()
    out: dict[str, list[int]] = {}
    for line in open(path):
        k, n, ns, b = line.rstrip("\n").split("\t")
        out[k] = [int(n), int(ns), int(b)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--label", default="")
    ap.add_argument("--probe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    for k, v in (
        ("OPENBLAS_NUM_THREADS", "1"),
        ("DEC_KV", "f32"),
        ("DEC_UBATCH", "512"),
        ("N_SEQ", "17"),
    ):
        os.environ.setdefault(k, v)
    sys.path.insert(0, a.pkg)
    t0 = time.monotonic()
    import engine as eng_mod  # the package's snapstart/engine.py, importable only after the path insert

    lib = os.path.join(a.pkg, "lib", "libdecider_hidden.so")
    engine, shim = eng_mod.load_engine(
        lib, a.gguf, os.path.join(a.pkg, "assets"), a.threads, max_request_tokens=24576
    )
    load_s = time.monotonic() - t0
    digests: list[hashlib._Hash] = []
    inner = engine.forward_shared

    def hashing_forward(*args, **kw):
        out = inner(*args, **kw)
        h = digests[-1] if digests else None
        if h is not None:
            for arr in out:
                h.update(memoryview(arr.astype("<f4", copy=False)).cast("B"))
        return out

    engine.forward_shared = hashing_forward
    warm = {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
    engine.evaluate(warm)
    probe = ctypes.CDLL(a.probe) if a.probe else None
    tmp = tempfile.mkstemp(suffix=".tsv")[1]
    if probe is not None:
        probe.lkai_e14_probe_reset()
    reqs = json.load(open(a.requests))["requests"]
    lat: dict[str, list[float]] = {r["id"]: [] for r in reqs}
    cpu: dict[str, list[float]] = {r["id"]: [] for r in reqs}
    hidden_sha: dict[str, list[str]] = {r["id"]: [] for r in reqs}
    concat: dict[str, dict[str, list[int]]] = {r["id"]: {} for r in reqs}
    answers: dict[str, dict] = {}
    t_ss = time.monotonic()
    c_ss = cpu_s()
    for _ in range(a.passes):
        for r in reqs:
            digests.append(hashlib.sha256())
            c = cpu_s()
            t = time.monotonic()
            out = engine.evaluate(r["request"])
            lat[r["id"]].append(round(time.monotonic() - t, 6))
            cpu[r["id"]].append(round(cpu_s() - c, 6))
            hidden_sha[r["id"]].append(digests[-1].hexdigest())
            answers.setdefault(r["id"], out.model_dump()["answers"])
            if probe is not None:  # outside the timed span
                for k, v in read_probe(probe, tmp).items():
                    acc = concat[r["id"]].setdefault(k, [0, 0, 0])
                    for i in range(3):
                        acc[i] += v[i]
    ss = time.monotonic() - t_ss
    ss_cpu = cpu_s() - c_ss
    os.unlink(tmp)
    res = {
        "schema": "lkai-e14-offline/1",
        "evidence": "OFFLINE, NOT LAMBDA: P1's function.zip code and GGUF on another host; local file, not S3",
        "label": a.label,
        "host": {
            "machine": platform.machine(),
            "kernel": platform.release(),
            "nproc": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpuinfo": cpu_fields(),
        },
        "threads": a.threads,
        "shim_init": shim.init_info,
        "load_s": round(load_s, 6),
        "steady_state_total_s": round(ss, 6),
        "steady_state_cpu_s": round(ss_cpu, 6),
        "steady_state_s": lat,
        "steady_state_cpu_s_by_request": cpu,
        "hidden_sha256": hidden_sha,
        "concat_probe": concat if probe is not None else None,
        "answers": answers,
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
