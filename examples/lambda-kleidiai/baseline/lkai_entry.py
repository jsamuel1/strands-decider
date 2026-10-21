"""Lambda entry point of the instrumented baseline: handler "lkai_entry.handler".

Wraps snapstart/app.py (and through it PR #77's handler.py, loader.py and snapstart/engine.py)
without editing them. Each wrapped call is timed with phases.Recorder; nothing in the request path
changes. With LKAI_PHASES=0 the module imports app and hands every event straight to app.handler,
which is the REPORT-only arm of the overhead measurement: same zip, same code path, no timers.

Phases (seconds, CLOCK_MONOTONIC):
  runtime_to_import   process start -> this module (interpreter + runtime bootstrap)
  imports             numpy, pydantic, tokenizers, strands_decider, app modules
  weight_download     loader.download_to_memfd: HEAD + parallel ranged GETs into the memfd
  ggml_backend_load   decshim: dlopen of the libggml-cpu-* variants (best ISA for this host)
  model_load_mmap     decshim: llama_model_load_from_file on the memfd (mmap, tensor setup, KleidiAI repack)
  context_init        decshim: llama_init_from_model (KV cache, compute buffers)
  assets_load         tokenizer.json + head.safetensors + head_config.json (engine_load - shim_init)
  warmup_inference    the handler's warm-up evaluate before the snapshot (first inference ever)
  model_ready         handler._load end to end (PR #77's own init_seconds, independently timed)
Per invocation: start class (cold / restore / warm), the wait for a background load, the handler
total, PR #77's in-function engine latency (latency_ms) and, after a restore, the gap between the
after-restore hook and the first invocation. Lambda's REPORT line is left untouched.

Host identity (hostinfo.snapshot, H1): read once at init and again after every restore. The full record rides on
that environment's first invoke RESPONSE as `lkai.host`, with the invocation context's log stream name, so a row
names the CPU identity, features, HWCAPs, cores, cgroup limits and kernel of the environment that served it. Log
lines carry only `host_fingerprint` (the 4 KB log tail must keep its START line); the snapshot-building init's full
record rides on `before_snapshot`, which no tail holds. Only the instrumented arm reads it; REPORT-only is unchanged.
"""

# ruff: noqa: SLF001
# (This module wraps PR #77's handler, whose load future and CPU probe are module-private.)
from __future__ import annotations

import os
import time

T0_NS = time.monotonic_ns()
# app.py sets this before numpy loads; the instrumented path imports engine (and numpy) first, so set it here too.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import phases  # noqa: E402

ON = phases.enabled()
if ON:
    import hostinfo

if not ON:
    import app

    def handler(event, context):
        return app.handler(event, context)

else:
    import uuid

    REC = phases.Recorder(t0_ns=T0_NS)
    _rt = phases.runtime_to_import_s()
    if _rt is not None:
        REC.init["runtime_to_import"] = _rt
    _state = {"n": 0, "env_id": uuid.uuid4().hex[:12], "host": hostinfo.snapshot()}

    with REC.phase("imports"):
        import engine as _eng
        import loader as _loader

    _orig_download = _loader.download_to_memfd

    def _download(bucket: str, key: str) -> str:
        with REC.phase("weight_download"):
            path = _orig_download(bucket, key)
        try:
            size = os.fstat(int(path.rsplit("/", 1)[1])).st_size
            REC.counters["weight_bytes"] = size
            REC.counters["weight_mib_s"] = round(size / 2**20 / REC.init["weight_download"], 1)
        except (OSError, ValueError, ZeroDivisionError):
            pass
        return path

    _loader.download_to_memfd = _download

    _OrigShim = _eng.Shim
    _DEC_PHASES = ("ggml_backend_load", "model_load_mmap", "context_init", "dec_load_total")

    class _TimedShim(_OrigShim):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            with REC.phase("shim_init"):
                super().__init__(*args, **kwargs)
            fn = getattr(self.lib, "dec_load_phases_ns", None)
            if fn is None:  # a shim built without the timers
                return
            import ctypes

            buf = (ctypes.c_int64 * len(_DEC_PHASES))()
            fn.restype = ctypes.c_int
            fn(buf, len(_DEC_PHASES))
            for name, ns in zip(_DEC_PHASES, buf, strict=True):
                REC.add(name, ns / 1e9)

    _eng.Shim = _TimedShim
    _orig_load_engine = _eng.load_engine

    def _load_engine(*args, **kwargs):
        with REC.phase("engine_load"):
            out = _orig_load_engine(*args, **kwargs)
        REC.add("assets_load", REC.init["engine_load"] - REC.init.get("shim_init", 0.0))
        return out

    _eng.load_engine = _load_engine
    _eng.HeadRowsEngine.evaluate = phases.first_call_timed(
        REC, "warmup_inference", _eng.HeadRowsEngine.evaluate
    )

    # handler.py binds load_engine at its import and runs _load there (SnapStart) or in a thread.
    _t_app = time.monotonic_ns()
    import app
    import handler as _pr77

    if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") != "on-demand":
        # The load ran inside the import: model_ready is that import's duration.
        REC.add("model_ready", (time.monotonic_ns() - _t_app) / 1e9)
    # On demand the load runs in PR #77's background thread; the first invocation records its init_seconds
    # (handler._load's own perf_counter span) as model_ready, after waiting for it.
    REC.add("import_total", REC.since_t0())

    def _before_snapshot():
        REC.emit(
            "before_snapshot",
            env_id=_state["env_id"],
            init=REC.init,
            counters=REC.counters,
            host=_state["host"],
        )

    def _after_restore():
        REC.restored = True
        REC.restore_mono = time.monotonic_ns()
        REC.restore_wall = time.time_ns()
        _state["env_id"] = uuid.uuid4().hex[
            :12
        ]  # every restored copy starts as its own environment
        # A restored copy may run on another host than the one that built its snapshot: read it again. The log
        # line carries only the fingerprint: bench.py keeps a 4 KB log tail, and a full record could push the
        # invocation's START line out of it. The full record rides on the first invoke's response.
        _state["host"] = hostinfo.snapshot()
        REC.emit(
            "after_restore",
            env_id=_state["env_id"],
            cpu_part=_pr77._cpu_part(),
            host_fingerprint=_state["host"]["fingerprint"],
        )

    try:
        from snapshot_restore_py import register_after_restore, register_before_snapshot

        register_before_snapshot(_before_snapshot)
        register_after_restore(_after_restore)
        REC.counters["snapstart_hooks"] = True
    except ImportError:  # not on Lambda, or a runtime without the hooks
        REC.counters["snapstart_hooks"] = False

    def handler(event, context):
        t_in = time.monotonic_ns()
        _state["n"] += 1
        n = _state["n"]
        cls = phases.start_class(n, os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE"), REC.restored)
        _, init_s = _pr77._ENGINE.result()  # what app.handler would wait for anyway
        t_ready = time.monotonic_ns()
        if "model_ready" not in REC.init:
            REC.init["model_ready"] = init_s
        out = app.handler(event, context)
        t_out = time.monotonic_ns()
        inv = {
            "n": n,
            "class": cls,
            "wait_load_s": round((t_ready - t_in) / 1e9, 6),
            "handler_s": round((t_out - t_in) / 1e9, 6),
            "cpu_part": _pr77._cpu_part(),
        }
        if isinstance(out, dict) and "latency_ms" in out:
            inv["engine_s"] = round(out["latency_ms"] / 1000, 6)
        if n == 1 and REC.restore_mono is not None:
            inv["restore_to_invoke_s"] = round((t_in - REC.restore_mono) / 1e9, 6)
        rec = {
            "env_id": _state["env_id"],
            "request_id": getattr(context, "aws_request_id", None),
            "version": getattr(context, "function_version", None),
            "inv": inv,
        }
        if n == 1:
            rec["init"] = dict(REC.init)
            rec["counters"] = dict(REC.counters)
            rec["host_fingerprint"] = _state["host"]["fingerprint"]
        REC.emit("invoke", **rec)
        if isinstance(out, dict) and "statusCode" not in out:
            # Response only (see _after_restore): bench.py stores it as the row's lkai.host.
            if n == 1:
                rec = {
                    **rec,
                    "host": _state["host"],
                    "log_stream": getattr(context, "log_stream_name", None),
                }
            out["lkai"] = rec
        return out
