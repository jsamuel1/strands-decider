"""E14 sampler-bias probe: does lkaisamp's SIGPROF land on the thread that used the CPU, in Lambda and offline?

  Lambda: handler "sigprof_probe.handler" in a zip with lib/liblkaisamp.so (P3's arm64 build, from lkai-p3-sampler's
          deployed zip) beside it. Event {"threads": 6, "seconds": 2}.
  offline: python sigprof_probe.py <path to liblkaisamp.so> [threads] [seconds]

`threads` threads (the calling thread included) each copy an 8 MiB buffer with ctypes.memmove in a loop for the same
wall time. ctypes releases the GIL around each foreign call, so the copies run natively and in parallel, as ggml's
worker threads do.

lkaisamp records (pc, lr, tid, tag) per SIGPROF. The probe tags every worker thread's samples by its tid and returns
the share each thread received, the main (calling) thread's share, the fair share (1/threads), the kernel release and
the CPU part. With fair delivery every thread gets about 1/threads.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import struct
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REC = struct.Struct("<QQII")


def _cpu_part() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("CPU part"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def run(lib_path: str, threads: int = 6, seconds: float = 2.0) -> dict:
    samp = ctypes.CDLL(lib_path)
    buf_src = ctypes.create_string_buffer(8 << 20)
    tids: dict[int, int] = {}
    go = threading.Event()
    stop = time.monotonic() + seconds + 0.2

    def work(idx: int) -> None:
        tids[idx] = threading.get_native_id()
        dst = ctypes.create_string_buffer(8 << 20)
        go.wait()
        while time.monotonic() < stop:
            ctypes.memmove(
                dst, buf_src, 8 << 20
            )  # native copy; ctypes releases the GIL around the call

    ts = [threading.Thread(target=work, args=(i,)) for i in range(1, threads)]
    for t in ts:
        t.start()
    samp.samp_reset()  # a warm environment keeps the previous invocation's samples otherwise
    if samp.samp_start(1000, 1 << 20) != 0:
        raise RuntimeError("samp_start failed")
    tids[0] = threading.get_native_id()
    go.set()
    work_main_dst = ctypes.create_string_buffer(8 << 20)
    while time.monotonic() < stop:
        ctypes.memmove(work_main_dst, buf_src, 8 << 20)
    samp.samp_stop()
    for t in ts:
        t.join()
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        samp.samp_dump(fd)
        os.close(fd)
        data = open(path, "rb").read()
    finally:
        os.unlink(path)
    _period, n, _dropped, rec = struct.unpack_from("<QQQQ", data, 8)
    counts: dict[int, int] = {}
    for i in range(n):
        _pc, _lr, tid, _tag = REC.unpack_from(data, 40 + i * rec)
        counts[tid] = counts.get(tid, 0) + 1
    by_idx = [counts.get(tids[i], 0) for i in range(threads)]
    other = n - sum(by_idx)
    return {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_part": _cpu_part(),
        "nproc": os.cpu_count(),
        "threads": threads,
        "samples": n,
        "share": [round(c / n, 4) if n else 0.0 for c in by_idx],
        "other_threads_share": round(other / n, 4) if n else 0.0,
        "main_thread_share": round(by_idx[0] / n, 4) if n else 0.0,
        "fair_share": round(1 / threads, 4),
    }


def handler(event, context):
    e = event or {}
    out = run(
        os.path.join(HERE, "lib", "liblkaisamp.so"),
        int(e.get("threads", 6)),
        float(e.get("seconds", 2)),
    )
    out["log_stream"] = getattr(context, "log_stream_name", "unknown")
    print("E14_SIGPROF " + json.dumps(out, separators=(",", ":")))
    return out


if __name__ == "__main__":
    print(
        json.dumps(
            run(sys.argv[1], *(int(x) for x in sys.argv[2:3]), *(float(x) for x in sys.argv[3:4]))
        )
    )
