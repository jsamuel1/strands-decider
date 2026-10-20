"""Lambda entry point of the E3 functions: handler "e3_entry.handler".

It routes P1's loader.download_to_memfd to the E3_DL strategy (e3dl.py), then hands over to P1's instrumented entry
point (baseline/lkai_entry.py) unchanged. lkai_entry wraps whatever loader.download_to_memfd is when it imports, so the
weight_download phase, weight_mib_s and every later phase are timed by P1's own code, and the e3_* counters ride in
P1's LKAI records.

The patch has to be in place before lkai_entry imports, because with SnapStart (and on demand, in PR #77's background
thread) the model load starts during that import. Importing loader here moves its import (ctypes, libc, about a
millisecond) out of P1's `imports` phase; e3_preimport_s records it.
"""

from __future__ import annotations

import time

_t = time.monotonic_ns()
import e3dl  # noqa: E402
import loader  # noqa: E402

loader.download_to_memfd = e3dl.make_download(loader.download_to_memfd)
_preimport_s = round((time.monotonic_ns() - _t) / 1e9, 6)

import lkai_entry  # noqa: E402  (runs the load at import under SnapStart)

if hasattr(lkai_entry, "REC"):  # LKAI_PHASES=0 has no recorder
    lkai_entry.REC.counters["e3_preimport_s"] = _preimport_s
handler = lkai_entry.handler
