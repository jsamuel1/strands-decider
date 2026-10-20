"""Lambda entry point of E4 (handler "e4_entry.handler"): P1's instrumented baseline (lkai_entry), with the weight
download and the model load overlapped when E4_STREAM=1.

Every E4 arm runs this file from the same zip; the arms differ only in their environment (deploy_e4.ARMS_ENV):
  base         E4_STREAM=0                       download, then load (P1's sequence; the llama.cpp hooks stay unset)
  stream       E4_STREAM=1 E4_MMAP_THRESHOLD=4 MiB   the load starts once the GGUF header is in the memfd and follows
                             the download front (stream_loader.py), with glibc's mmap threshold pinned
  basethr      E4_STREAM=0 E4_MMAP_THRESHOLD=4 MiB   base with the same pin: what the pin alone does
  streamfile   E4_STREAM=1, STREAM_ORDER=file    (first run only) the overlap in llama.cpp's own tensor order
  E4_MMAP_THRESHOLD unset: glibc's dynamic threshold, as P1 (first run's stream arm)

Both arms import loader, engine and stream_loader here, before lkai_entry, so their imports are the same; that time
is added to lkai_entry's `imports` phase. On every arm this module also records `download_and_load`: from the start
of the weight download to the return of the shim's load (llama_model_load_from_file + context init), the quantity
E4 changes. lkai_entry's other phases keep their meaning, except that on a streamed arm `model_load_mmap` includes
the time the loader waits for bytes (`stream_load_blocked`).
"""

from __future__ import annotations

import os
import time

_T0 = time.monotonic_ns()
# as app.py and lkai_entry.py, before numpy loads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import engine as _eng  # noqa: E402
import loader as _loader  # noqa: E402
import stream_loader  # noqa: E402

STREAM = os.environ.get("E4_STREAM", "0") == "1"
# Before any download thread allocates (E4_MMAP_THRESHOLD; stream_loader.pin_mmap_threshold says why).
MMAP_THRESHOLD = stream_loader.pin_mmap_threshold()
_PRE_IMPORTS_S = (time.monotonic_ns() - _T0) / 1e9
_t = {"download_start": None}

_orig_download = stream_loader.download_to_memfd_streaming if STREAM else _loader.download_to_memfd


def _download(bucket: str, key: str) -> str:
    _t["download_start"] = time.monotonic()
    return _orig_download(bucket, key)


_loader.download_to_memfd = _download
_orig_shim_init = _eng.Shim.__init__


def _shim_init(self, *args, **kwargs):
    try:
        _orig_shim_init(self, *args, **kwargs)
    except BaseException:
        if STREAM:
            stream_loader.finish_load()  # raises the download's own error if that is why the load failed
        raise
    if STREAM:
        stream_loader.finish_load()
    if _t["download_start"] is not None:
        stream_loader._record(download_and_load_s=time.monotonic() - _t["download_start"])  # noqa: SLF001


_eng.Shim.__init__ = _shim_init

import lkai_entry  # noqa: E402  (runs PR #77's load: at import under SnapStart, in a thread on demand)

_rec = getattr(lkai_entry, "REC", None)
if _rec is not None:
    _rec.add("imports", _PRE_IMPORTS_S)
    _rec.counters["e4_stream"] = STREAM
    _rec.counters["e4_stream_order"] = os.environ.get("STREAM_ORDER", "arrival") if STREAM else None
    _rec.counters["e4_mmap_threshold"] = MMAP_THRESHOLD

handler = lkai_entry.handler
