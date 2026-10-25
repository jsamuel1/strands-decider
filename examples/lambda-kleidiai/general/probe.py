"""General-ladder probe (G): one process, one model load, then P1's requests and a full-window forward. FIXED HOST.

  python probe.py <pkg_dir> <gguf> <requests.json> <reference outputs.json> <out.json>
         [--mode local|memfd] [--n-ctx N] [--warm-passes 3] [--long-passes 2] [--arm NAME]

Runs inside P1's pinned public.ecr.aws/lambda/python:3.12 image (arm64, or its amd64 twin for the x86_64 path) with
an unzipped function package mounted read-only at <pkg_dir>, as Lambda mounts /var/task. Nothing here is Lambda
latency: every number is a fixed-host measurement of the same engine code path.

  --mode local   llama.cpp mmaps the GGUF file itself (the general-deployment path: EC2, container, laptop).
  --mode memfd   the GGUF is copied into a memfd first and mapped through /proc/self/fd/N, as PR #77's loader does
                 after its S3 download (weight_read here is a local-disk copy and is NOT comparable to S3).
  --n-ctx N      E10's one variable: replaces the n_ctx argument engine.load_engine passes to engine.Shim, exactly
                 as e10/e10_entry.py's wrap_shim does at lkai/fbb1-kv-size 60f4586 (requested and applied values
                 are recorded). Unset, P1's own value (16384 with N_SEQ=17) is used and recorded.

Measured, in order: imports (engine and loader, P1's `imports` set), weight_read (memfd only), P1's decshim load
phases (ggml_backend_load, model_load_mmap, context_init), assets, the handler's warm-up evaluate (the first
inference), the probe request (first request), every fixed request --warm-passes times, a forward that fills the
model's whole window (--long-passes times) and the process's memory high-water mark.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

T0 = time.monotonic()
WARMUP = {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
PHASES = ("ggml_backend_load", "model_load_mmap", "context_init", "dec_load_total")
# P1's deployed function environment (baseline/deploy.py). Without these the engine runs another KV type, ubatch and
# sequence count, and its answers are not P1's (E8's probe_e8.py learned this the hard way).
P1_ENV = (("OPENBLAS_NUM_THREADS", "1"), ("DEC_KV", "f32"), ("DEC_UBATCH", "512"), ("N_SEQ", "17"))


def long_request(window: int) -> dict:
    """One noul question over a state longer than the window: HiddenStateEngine.fit trims the state from the end so
    state + question is exactly `window` tokens (asserted by the caller from usage.input_tokens)."""
    para = (
        "Incident 4471 timeline. 09:02 deploy of checkout-api build 8812 starts. 09:06 p99 latency on /cart rises "
        "from 180 ms to 2.4 s; error rate 0.2% to 6%. 09:11 on-call sees connection-pool exhaustion against the "
        "orders database and suspects the retry change, which retries reads three times with no jitter. "
    )
    return {
        "state": para * (window // 40 + 8),
        "questions": {
            "retry": {"type": "noul", "instructions": "Did the retry change cause the outage?"}
        },
    }


def runtime_to_probe_s() -> float | None:
    """Process start to this module's first line (interpreter start), as baseline/phases.runtime_to_import_s."""
    try:
        with open("/proc/self/stat", encoding="ascii") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        boot_at_t0 = time.clock_gettime(time.CLOCK_BOOTTIME) - (time.monotonic() - T0)
        return round(boot_at_t0 - int(fields[19]) / os.sysconf("SC_CLK_TCK"), 4)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def to_memfd(path: str) -> str:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(b"model", 0)
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            os.write(fd, b)
    return f"/proc/self/fd/{fd}"


def wrap_n_ctx(engine_mod, n_ctx: int, seen: dict) -> None:
    """e10_entry.wrap_shim's n_ctx replacement (lkai/fbb1-kv-size 60f4586), without its Lambda entry point."""
    orig = engine_mod.Shim.__init__

    def __init__(self, lib_path, model_path, threads, *args, **kwargs):
        if args:
            seen["requested"].append(args[0])
            args = (n_ctx, *args[1:])
        else:
            seen["requested"].append(kwargs.get("n_ctx"))
            kwargs["n_ctx"] = n_ctx
        seen["applied"].append(n_ctx)
        orig(self, lib_path, model_path, threads, *args, **kwargs)

    engine_mod.Shim.__init__ = __init__


def e7_mechanism(shim, lib_dir: str) -> dict:
    """E7's readout (e7/e7_host.py): the patched loader's skip flag and CPU_KLEIDIAI bytes; 'absent' on other libs."""
    try:
        lib = ctypes.CDLL(os.path.join(lib_dir, "libllama.so"))
        f = lib.llama_model_lkai_tied_output_skipped
        g = lib.llama_model_lkai_buft_bytes
    except (AttributeError, OSError):
        return {"tied_output_skipped": "absent"}
    f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_int32
    g.argtypes, g.restype = [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_uint64
    model = ctypes.c_void_p.from_address(shim.h).value  # dec_t's first member is the llama_model *
    return {
        "tied_output_skipped": bool(f(model)),
        "buft_bytes": {n: int(g(model, n.encode())) for n in ("CPU", "CPU_KLEIDIAI", "CPU_REPACK")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("gguf")
    ap.add_argument("requests")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=("local", "memfd"), default="local")
    ap.add_argument("--n-ctx", type=int, default=0)
    ap.add_argument("--warm-passes", type=int, default=3)
    ap.add_argument("--long-passes", type=int, default=2)
    ap.add_argument("--arm", default="unknown")
    a = ap.parse_args()
    for k, v in P1_ENV:
        os.environ.setdefault(k, v)
    threads = int(os.environ.get("THREADS", "6"))
    sys.path.insert(0, a.pkg)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import host

    res: dict = {
        "schema": "lkai-general-probe/1",
        "evidence": "FIXED HOST, NOT LAMBDA",
        "arm": a.arm,
        "mode": a.mode,
        "threads": threads,
        "env": {
            k: os.environ.get(k)
            for k in (
                "DEC_KV",
                "DEC_UBATCH",
                "N_SEQ",
                "OPENBLAS_NUM_THREADS",
                "GGML_KLEIDIAI_REPACK_THREADS",
                "LKAI_SKIP_TIED_OUTPUT",
            )
        },
        "host_before": host.host_metadata(),
        "gguf": {"file": os.path.basename(a.gguf), "bytes": os.path.getsize(a.gguf)},
    }
    t = time.monotonic()
    # P1's snapstart/engine.py from the package (numpy, tokenizers, strands_decider), and PR #77's loader, as P1's
    # `imports` phase times both.
    import engine
    import loader  # noqa: F401

    ph: dict = {
        "process_to_probe": runtime_to_probe_s(),
        "probe_start_to_imports": round(t - T0, 4),
        "imports": round(time.monotonic() - t, 4),
    }
    seen: dict = {"requested": [], "applied": []}
    if a.n_ctx:
        wrap_n_ctx(engine, a.n_ctx, seen)
    path = a.gguf
    if a.mode == "memfd":
        t = time.monotonic()
        path = to_memfd(a.gguf)
        ph["weight_read"] = round(time.monotonic() - t, 4)
    t = time.monotonic()
    eng, shim = engine.load_engine(
        os.path.join(a.pkg, "lib", "libdecshim.so"), path, os.path.join(a.pkg, "assets"), threads
    )
    ph["load_engine"] = round(time.monotonic() - t, 4)
    buf = (ctypes.c_int64 * 4)()
    shim.lib.dec_load_phases_ns(buf, 4)
    ph |= {k: round(v / 1e9, 4) for k, v in zip(PHASES, buf, strict=True)}
    ph["assets_load"] = round(ph["load_engine"] - ph["dec_load_total"], 4)
    res["n_ctx"] = {
        "requested": seen["requested"] or [shim.n_ctx],
        "applied": seen["applied"] or [shim.n_ctx],
    }
    res["init_info"] = shim.init_info
    res["e7"] = e7_mechanism(shim, os.path.join(a.pkg, "lib"))
    res["status_after_load_kb"] = host.proc_status_kb(host.read_text("/proc/self/status"))

    t = time.monotonic()
    eng.evaluate(WARMUP)
    ph["warmup_inference"] = round(time.monotonic() - t, 4)
    reqs = json.load(open(a.requests))
    by_id = {r["id"]: r["request"] for r in reqs["requests"]}
    t = time.monotonic()
    first = eng.evaluate(by_id[reqs["probe"]]).model_dump(mode="json")["answers"]
    ph["first_request_inference"] = round(time.monotonic() - t, 4)
    ph["model_ready"] = round(time.monotonic() - T0, 4)

    ident = res["host_before"]["identity"]
    ref = json.load(open(a.reference))["outputs"]
    answers: dict = {reqs["probe"]: [first]}
    warm: dict = {}
    for _ in range(a.warm_passes):
        for rid, body in by_id.items():
            t = time.monotonic()
            ans = eng.evaluate(body).model_dump(mode="json")["answers"]
            warm.setdefault(rid, []).append(round(time.monotonic() - t, 4))
            answers.setdefault(rid, []).append(ans)
    vs_ref = {}
    for rid, alist in answers.items():
        want = (ref.get(rid) or {}).get(ident)
        if want is None:
            vs_ref[rid] = "no-reference-for-identity"
        else:
            vs_ref[rid] = "identical" if all(x == want for x in alist) else "DIFFERS"
    window = eng.cfg.max_length
    lr = long_request(window)
    long_s, long_ans, long_tokens = [], [], None
    for _ in range(a.long_passes):
        t = time.monotonic()
        out = eng.evaluate(lr).model_dump(mode="json")
        long_s.append(round(time.monotonic() - t, 4))
        long_ans.append(out["answers"])
        long_tokens = out["usage"]["input_tokens"]
    res |= {
        "phases_s": ph,
        "warm_s": warm,
        "long_forward": {
            "window": window,
            "input_tokens": long_tokens,
            "seconds": long_s,
            "answers": long_ans[0] if long_ans else None,
            "repeat_identical": all(x == long_ans[0] for x in long_ans),
        },
        "answers": {rid: alist[0] for rid, alist in answers.items()},
        "answers_repeat_identical": all(all(x == al[0] for x in al) for al in answers.values()),
        "vs_reference": vs_ref,
        "status_end_kb": host.proc_status_kb(host.read_text("/proc/self/status")),
        "host_after": {"loadavg": host.loadavg()},
        "wall_s": round(time.monotonic() - T0, 4),
    }
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(
        json.dumps(
            {
                "arm": a.arm,
                "mode": a.mode,
                "model_load_mmap": ph["model_load_mmap"],
                "vs_reference": vs_ref,
                "long_tokens": long_tokens,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
