"""Lambda entry point of all three E8 functions: handler "e8_entry.handler".

  base   P1's deployed function.zip, every entry unchanged, plus this file and e8_host.py
  off    the same zip with the six lib/libggml-cpu-*.so built with e8/patch_parallel_repack.py; switch unset
  par    the `off` zip, with GGML_KLEIDIAI_REPACK_THREADS=6 in the function's environment

This module imports P1's instrumented entry point (baseline/lkai_entry.py) unchanged, so P1's own timers time
`model_load_mmap` (decshim's clock around llama_model_load_from_file, which holds the repack) and every other phase.
On the first invocation of each environment, after P1's handler has answered (the model is loaded by then: at import
under SnapStart, or in PR #77's background thread on demand), it reads the host metadata the initiative's HOST
VARIABILITY rules ask for, the libggml-cpu variant this host loaded, and VmHWM. It reads them then, not at import,
because a SnapStart restore can land on another host than the snapshot init ran on; the init host's CPU part is kept
separately as `init_cpu_part`. The readout is outside every P1 phase and its own duration is recorded. A field that
cannot be read is "unknown", never guessed.

The first invocation of each environment carries the record as `lkai.e8` in its response and as an `E8 {json}` log
line. Later invocations carry only P1's record.
"""

from __future__ import annotations

import json
import sys
import time

from e8_host import UNKNOWN, cpuinfo, host_metadata, loaded_ggml_cpu, proc_status_kb, read_text

# Only the CPU part of the host this module is imported on (the snapshot init, under SnapStart).
INIT_CPU_PART = cpuinfo(read_text("/proc/cpuinfo"))["part"]

# P1's instrumented entry; its clock starts at its own import.
import lkai_entry  # noqa: E402

_n = {"n": 0}


def handler(event, context):
    out = lkai_entry.handler(event, context)  # waits for the load, then answers (P1's timing)
    _n["n"] += 1
    if _n["n"] == 1:
        t = time.monotonic_ns()
        rec = host_metadata() | {
            "init_cpu_part": INIT_CPU_PART,
            "log_stream": getattr(context, "log_stream_name", UNKNOWN),
            "ggml_cpu_loaded": loaded_ggml_cpu(read_text("/proc/self/maps", 1 << 22)),
            "proc_status_kb": proc_status_kb(read_text("/proc/self/status")),
        }
        rec["readout_s"] = round((time.monotonic_ns() - t) / 1e9, 6)
        sys.stdout.write("E8 " + json.dumps(rec, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if isinstance(out, dict) and isinstance(out.get("lkai"), dict):
            out["lkai"]["e8"] = rec
    return out
