"""Run the baseline engine OUTSIDE Lambda, with the same code, weights and request set, for profilers Lambda denies.

  python offline_drive.py --pkg <unzipped function.zip> --gguf <s2b-h512-Q8_0.gguf> [--passes 10] [--out run.json]
                          [--ngl 0] [--threads 6] [--cprofile out.prof] [--sampler <lkaisamp.so> --samples out.bin]

What it measures is NOT Lambda evidence: the host, kernel, page cache and storage differ (the GGUF is read from a local
file, not a memfd filled from S3). Every result it writes says so in `evidence`. It exists so that perf, py-spy,
Nsight Systems and nvidia-smi can watch the same load and forward path that Lambda runs.

Phases (seconds, CLOCK_MONOTONIC), named as in baseline/phases.py where the code path is the same:
  imports            engine (numpy, pydantic, tokenizers, strands_decider)
  weight_read        read the GGUF into a memfd (stands in for weight_download; local disk, not S3)
  ggml_backend_load, model_load_mmap, context_init   decshim's own timers (dec_load_phases_ns)
  assets_load        tokenizer, head, config
  warmup_inference   PR #77's warm-up request
  steady_state_inference  every request in requests.json, `passes` times, in order
The process prints `PHASE <name> <begin|end>` lines on stderr so an external profiler's timeline can be cut by phase,
and, with --sampler, tags native samples with the phase.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time

T0 = time.monotonic()
TAGS = {"imports": 1, "weight_read": 2, "model_load": 3, "assets_load": 4, "warmup_inference": 5,
        "steady_state_inference": 6}


class Phases:
    def __init__(self, sampler=None):
        self.d: dict[str, float] = {}
        self.sampler = sampler

    def run(self, name, fn, *a, **k):
        print(f"PHASE {name} begin {time.monotonic():.6f}", file=sys.stderr, flush=True)
        if self.sampler is not None:
            self.sampler.samp_phase(TAGS.get(name, 0))
        t = time.monotonic()
        try:
            return fn(*a, **k)
        finally:
            self.d[name] = round(self.d.get(name, 0.0) + time.monotonic() - t, 6)
            if self.sampler is not None:
                self.sampler.samp_phase(0)
            print(f"PHASE {name} end {time.monotonic():.6f}", file=sys.stderr, flush=True)


def memfd_copy(path: str) -> str:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.memfd_create(b"gguf", 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            os.write(fd, chunk)
    return f"/proc/self/fd/{fd}"


def cpu_part() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("CPU part") or line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="unzipped function.zip (engine.py, lib/, assets/, strands_decider/)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--requests", default=os.path.join(os.path.dirname(__file__), "..", "baseline", "requests.json"))
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6, help="Lambda's 10,240 MB function has 6 vCPUs")
    ap.add_argument("--ngl", type=int, default=0, help="GPU layers (needs a decshim built with DEC_NGL support)")
    ap.add_argument("--no-memfd", action="store_true", help="mmap the GGUF file directly instead of a memfd copy")
    ap.add_argument("--cprofile")
    ap.add_argument("--sampler", help="path to lkaisamp.so")
    ap.add_argument("--samples", help="where to write the sampler's binary dump")
    ap.add_argument("--sample-period-us", type=int, default=1000)
    ap.add_argument("--nodeprof", help="E15: a CPU variant .so built with e15/patch_nodeprof.py; needs LKAI_NODEPROF=1")
    ap.add_argument("--nodeprof-out", help="E15: TSV of per-node wall time over steady_state_inference")
    ap.add_argument("--host-label", default=platform.node())
    ap.add_argument("--out")
    a = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("DEC_KV", "f32")
    os.environ.setdefault("DEC_UBATCH", "512")
    os.environ.setdefault("N_SEQ", "17")
    os.environ["DEC_NGL"] = str(a.ngl)
    sys.path.insert(0, a.pkg)
    samp = None
    if a.sampler:
        samp = ctypes.CDLL(a.sampler)
        if samp.samp_start(a.sample_period_us, 1 << 22) != 0:
            sys.exit("lkaisamp: samp_start failed")
    prof = None
    if a.cprofile:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
    ph = Phases(samp)
    eng_mod = ph.run("imports", __import__, "engine")
    gguf = a.gguf if a.no_memfd else ph.run("weight_read", memfd_copy, a.gguf)
    lib = os.path.join(a.pkg, "lib", "libdecider_hidden.so")
    engine, shim = ph.run("model_load", eng_mod.load_engine, lib, gguf, os.path.join(a.pkg, "assets"), a.threads,
                          max_request_tokens=24576)
    buf = (ctypes.c_int64 * 4)()
    shim.lib.dec_load_phases_ns(buf, 4)
    for name, ns in zip(("ggml_backend_load", "model_load_mmap", "context_init"), buf[:3], strict=True):
        ph.d[name] = round(ns / 1e9, 6)
    ph.d["assets_load"] = round(ph.d["model_load"] - buf[3] / 1e9, 6)
    warm = {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
    ph.run("warmup_inference", engine.evaluate, warm)
    reqs = json.load(open(a.requests))["requests"]
    lat: dict[str, list[float]] = {r["id"]: [] for r in reqs}
    answers: dict[str, dict] = {}
    nprof = None
    if a.nodeprof:
        # The same path ggml loaded, so dlopen returns the loaded variant's handle and its table.
        nprof = ctypes.CDLL(a.nodeprof)
        nprof.lkai_nodeprof_reset()
    print(f"PHASE steady_state_inference begin {time.monotonic():.6f}", file=sys.stderr, flush=True)
    if samp is not None:
        samp.samp_phase(TAGS["steady_state_inference"])
    t_ss = time.monotonic()
    for _ in range(a.passes):
        for r in reqs:
            t = time.monotonic()
            out = engine.evaluate(r["request"])
            lat[r["id"]].append(round(time.monotonic() - t, 6))
            answers.setdefault(r["id"], out.model_dump()["answers"])
    ph.d["steady_state_total"] = round(time.monotonic() - t_ss, 6)
    if nprof is not None and nprof.lkai_nodeprof_dump(a.nodeprof_out.encode()) != 0:
        sys.exit("lkai_nodeprof_dump failed")
    print(f"PHASE steady_state_inference end {time.monotonic():.6f}", file=sys.stderr, flush=True)
    if samp is not None:
        samp.samp_phase(0)
        samp.samp_stop()
        if a.samples:
            fd = os.open(a.samples, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            samp.samp_dump(fd)
            os.close(fd)
            # The maps file symbolises the dump: the samples are absolute addresses in this process.
            with open(a.samples + ".maps", "w") as f:
                f.write(open("/proc/self/maps").read())
    if prof is not None:
        prof.disable()
        prof.dump_stats(a.cprofile)
    res = {
        "schema": "lkai-offline/1",
        "evidence": "OFFLINE, NOT LAMBDA: same function.zip code and GGUF on another host; local file, not S3",
        "host": {"label": a.host_label, "machine": platform.machine(), "cpu": cpu_part(), "nproc": os.cpu_count(),
                 "kernel": platform.release()},
        "threads": a.threads,
        "ngl": a.ngl,
        "shim_init": shim.init_info,
        "phases_s": ph.d,
        "steady_state_s": lat,
        "answers": answers,
        "sampler": None if samp is None else {"count": samp.samp_count(), "dropped": samp.samp_dropped(),
                                              "period_us": a.sample_period_us},
        "wall_s": round(time.monotonic() - T0, 3),
    }
    js = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)
    print(js)


if __name__ == "__main__":
    main()
