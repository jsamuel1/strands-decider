"""Lambda entry point of both E7 functions: handler "e7_entry.handler". One zip serves both arms.

The arms differ only in one environment variable that the patched llama.cpp reads at model load
(e7/patch_skip_output.py):
  ctl    LKAI_SKIP_TIED_OUTPUT unset   upstream's load: the tied output head is created and repacked into CPU_KLEIDIAI
  skip   LKAI_SKIP_TIED_OUTPUT=1       the tied output head is not created, so it is never repacked

This module imports P1's instrumented entry point (baseline/lkai_entry.py) unchanged, so P1's own timers time
`imports`, `model_load_mmap` and every other phase. On the first invocation of each environment, after P1's handler
has answered (the model is loaded by then: at import under SnapStart, or in PR #77's background thread on demand), it
reads the host metadata the initiative's HOST VARIABILITY rules ask for and the mechanism evidence from the loaded
model: whether the head was skipped, and the bytes of the CPU_KLEIDIAI and CPU weight buffers. Host metadata is read
then, not at import, because a SnapStart restore can land on another host than the snapshot init ran on; the init
host's CPU part is kept separately as `init_cpu_part`. The readout is outside every P1 phase, adds the same few
milliseconds to the first invocation's Duration on both arms, and its own duration is recorded. A field that cannot
be read is "unknown", never guessed.

The first invocation of each environment carries the record as `lkai.e7` in its response and as an `E7 {json}` log
line. Later invocations carry only P1's record.
"""

from __future__ import annotations

import json
import sys
import time

from e7_host import UNKNOWN, cpuinfo, host_metadata, mechanism, proc_status_kb, read_text

# Only the CPU part of the host this module is imported on (the snapshot init, under SnapStart). The full record is
# read at the first invocation, on the host that serves it.
INIT_CPU_PART = cpuinfo(read_text("/proc/cpuinfo"))["part"]

# P1's instrumented entry; its clock starts at its own import.
import lkai_entry  # noqa: E402

_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)  # waits for the load, then answers (P1's timing)
    _n["n"] += 1
    if _n["n"] == 1:
        t = time.monotonic_ns()
        import engine

        rec = host_metadata() | {
            "init_cpu_part": INIT_CPU_PART,
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
            "mechanism": mechanism(engine.SHIM),
            "proc_status_kb": proc_status_kb(read_text("/proc/self/status")),
        }
        rec["readout_s"] = round((time.monotonic_ns() - t) / 1e9, 6)
        sys.stdout.write("E7 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e7"] = rec
    return out
