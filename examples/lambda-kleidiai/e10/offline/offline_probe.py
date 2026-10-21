"""Offline (x86, NOT Lambda) probe for one n_ctx: buffer sizes from llama.cpp's own log, context_init time,
memory-clear time, RSS, and answers for a request set. One process per n_ctx.

  python offline_probe.py N_CTX PKG LIB GGUF REQUESTS.json OUT.json
"""
import ctypes
import json
import os
import re
import resource
import statistics
import sys
import tempfile
import time

n_ctx, pkg, libdir, gguf, reqs_path, out_path = sys.argv[1:7]
n_ctx = int(n_ctx)
os.environ.update(DEC_KV="f32", DEC_UBATCH="512", N_SEQ="17", DEC_GGML_DIR=libdir)
sys.path.insert(0, pkg)

# llama.cpp logs to stderr when verbose=1; capture fd 2 around the load.
log = tempfile.TemporaryFile()
saved = os.dup(2)
os.dup2(log.fileno(), 2)
try:
    import engine as eng  # noqa: E402  P1's snapstart/engine.py, unchanged
except BaseException:
    os.dup2(saved, 2)
    raise

lib = ctypes.CDLL(os.path.join(libdir, "libdecshim.so"))
lib.dec_load_seq.restype = ctypes.c_void_p
lib.dec_load_seq.argtypes = [ctypes.c_char_p] + [ctypes.c_int] * 5
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
try:
    h = lib.dec_load_seq(gguf.encode(), 16, n_ctx, 1, 1, 17)
finally:
    os.dup2(saved, 2)
load_s = time.perf_counter() - t
log.seek(0)
text = log.read().decode(errors="replace")
assert h, text[-2000:]
ph = (ctypes.c_int64 * 4)()
lib.dec_load_phases_ns(ph, 4)


def mib(pat):
    return [float(x) for x in re.findall(pat, text)]


buffers = {
    "kv_mib": mib(r"llama_kv_cache: +\S+ KV buffer size = +([\d.]+) MiB"),
    "rs_mib": mib(r"llama_memory_recurrent: +\S+ RS buffer size = +([\d.]+) MiB"),
    "compute_mib": mib(r"sched_reserve: +\S+ compute buffer size = +([\d.]+) MiB")
    or mib(r"compute buffer size = +([\d.]+) MiB"),
    "output_mib": mib(r"output buffer size = +([\d.]+) MiB"),
    "n_ctx_log": re.findall(r"llama_context: n_ctx += (\d+)", text),
    "kv_line": re.findall(r"llama_kv_cache: size = [^\n]+", text),
    "rs_line": re.findall(r"llama_memory_recurrent: size = [^\n]+", text),
}

# Time llama_memory_clear(mem, true) directly: the clear dec_eval/dec_eval_fork do before every forward.
llama = ctypes.CDLL(os.path.join(libdir, "libllama.so"))
llama.llama_get_memory.restype = ctypes.c_void_p
llama.llama_get_memory.argtypes = [ctypes.c_void_p]
llama.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
ctx = ctypes.c_void_p.from_address(h + 8).value  # dec_t {model, ctx, ...}
mem = llama.llama_get_memory(ctx)
clr = []
for _ in range(30):
    t = time.perf_counter_ns()
    llama.llama_memory_clear(mem, True)
    clr.append((time.perf_counter_ns() - t) / 1e9)

# The P1 engine over this handle.
shim = eng.Shim.__new__(eng.Shim)
lib.dec_eval.restype = ctypes.c_int
lib.dec_eval.argtypes = [ctypes.c_void_p, eng._I32, ctypes.c_int, eng._I32, ctypes.c_int, eng._F32]
lib.dec_eval_fork.restype = ctypes.c_int
lib.dec_eval_fork.argtypes = [ctypes.c_void_p, eng._I32, ctypes.c_int, eng._I32, eng._I32, ctypes.c_int,
                              eng._I32, eng._I32, eng._F32]
lib.dec_n_embd.argtypes = [ctypes.c_void_p]
shim.h, shim.lib, shim.n_ctx, shim.n_seq = h, lib, n_ctx, 17
shim.n_embd, shim.unrot = lib.dec_n_embd(h), None
e = eng.HeadRowsEngine.from_assets(os.path.join(pkg, "assets"), shim.forward_shared, max_request_tokens=24576)
rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

reqs = json.load(open(reqs_path))
answers, errors, timings = {}, {}, {}
for rid, body in reqs.items():
    try:
        ts = []
        for _ in range(2):
            t = time.perf_counter()
            r = e.evaluate(body).model_dump()
            ts.append(time.perf_counter() - t)
        answers[rid] = {"answers": r["answers"], "input_tokens": r["usage"]["input_tokens"]}
        timings[rid] = min(ts)
    except Exception as ex:  # noqa: BLE001  the over-limit behaviour is a result
        errors[rid] = f"{type(ex).__name__}: {ex}"
json.dump(
    {
        "n_ctx": n_ctx,
        "buffers": buffers,
        "dec_phases_s": {k: v / 1e9 for k, v in zip(("backend", "model", "context", "total"), ph)},
        "load_s": load_s,
        "memory_clear_s": {"p50": statistics.median(clr), "min": min(clr), "max": max(clr), "n": len(clr)},
        "rss_maxrss_kib": {"before_load": rss0, "after_load": rss_loaded,
                           "after_requests": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "answers": answers,
        "errors": errors,
        "best_of_2_s": timings,
        "host": {"cpu": next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                              if ln.startswith("model name")), "unknown"), "threads": 16},
    },
    open(out_path, "w"),
    indent=1,
)
print("ok", n_ctx, buffers["kv_mib"], buffers["rs_mib"], buffers["compute_mib"], len(answers), len(errors))
