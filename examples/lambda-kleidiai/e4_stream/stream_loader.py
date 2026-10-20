"""E4: stream the GGUF from S3 into the memfd while llama.cpp loads it (handler arm E4_STREAM=1).

The download is PR #77's (examples/lambda/app/loader.py), unchanged in every parameter: one HeadObject, then 64 MiB
ranged GetObjects on 32 threads, 8 MiB body chunks, each range retried from the byte where it broke off, every byte
pwritten into one memfd. What changes is when the load starts. download_to_memfd_streaming returns the memfd path
as soon as the GGUF header is present, with the E4 llama.cpp hooks (e4_stream/patch_stream.py) installed:

- gguf_set_read_wait: the header parser blocks until the bytes it reads have arrived;
- llama_set_load_stream(wait, order): the loader blocks before reading each tensor until its bytes have arrived,
  and loads tensors in the order their bytes are expected to arrive. All 31 ranges are in flight at once and move
  at about the same rate, so a byte arrives at about (its offset within its range) / (range length) of the
  download: the order key is the largest such fraction a tensor needs (`order_key`). STREAM_ORDER=file keeps the
  loader's own (creation) order instead: the overlap without the reordering, a control.

Before llama_model_load_from_file returns, the patched loader waits for the whole file, so the model is complete
before the warm-up inference. A failed range makes every waiter return non-zero; the load then fails and the
download's own exception is raised. After the load, the hooks are cleared (the dormant state).

Counters (`STATS`, copied into lkai_entry's phase record when it is loaded): header wait, download wall time, bytes,
time the loader spent blocked in wait() and how many of its waits blocked.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import loader as _pr77_loader

log = logging.getLogger("decider.e4")

CHUNK = _pr77_loader.CHUNK
WORKERS = _pr77_loader.WORKERS
ATTEMPTS = _pr77_loader.ATTEMPTS
BODY_CHUNK = 8 * 1024 * 1024  # loader._download_part's iter_chunks size

WAIT_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)
ORDER_T = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64)

STATS: dict[str, Any] = {}
# The memfd's tracker and callbacks live as long as the process, like loader._fds.
_keep: list[Any] = []
_retry_sleep = time.sleep  # loader._download_part's back-off (tests replace it)


def parts_for(size: int, chunk: int = CHUNK) -> list[tuple[int, int]]:
    """loader.download_to_memfd's ranges: [(start, end_inclusive), ...]."""
    return [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]


def order_key(offs: int, size: int, chunk: int = CHUNK) -> int:
    """Bytes a range must have received before [offs, offs + size) is complete, maximised over the ranges it spans.

    With every range in flight at the same rate, that is when the tensor's last byte arrives, so sorting tensors by
    it follows the download front across all ranges."""
    if size <= 0:
        return 0
    end = offs + size
    first, last = offs // chunk, (end - 1) // chunk
    return max(min(end, (k + 1) * chunk) - k * chunk for k in range(first, last + 1))


class Progress:
    """Contiguous bytes received per range; wait() blocks until a byte span is covered or the download failed."""

    def __init__(self, size: int, chunk: int = CHUNK, clock=time.monotonic):
        self.size, self.chunk, self.clock = size, chunk, clock
        self.parts = parts_for(size, chunk)
        self.done = [0] * len(self.parts)  # bytes received from each range's start
        self.error: BaseException | None = None
        self.cv = threading.Condition()
        self.waits = self.blocked = 0
        self.blocked_s = 0.0

    def advance(self, part: int, pos: int) -> None:
        """Range `part` now holds every byte from its start up to (not including) file offset `pos`."""
        start = self.parts[part][0]
        with self.cv:
            if pos - start > self.done[part]:
                self.done[part] = pos - start
                self.cv.notify_all()

    def fail(self, e: BaseException) -> None:
        with self.cv:
            if self.error is None:
                self.error = e
            self.cv.notify_all()

    def covered(self, offs: int, size: int) -> bool:
        end = min(offs + size, self.size)
        if end <= offs:
            return True
        for k in range(offs // self.chunk, (end - 1) // self.chunk + 1):
            start, stop = self.parts[k][0], self.parts[k][1] + 1
            if self.done[k] < min(end, stop) - start:
                return False
        return True

    def complete(self) -> bool:
        return all(d == e - s + 1 for d, (s, e) in zip(self.done, self.parts, strict=True))

    def wait(self, offs: int, size: int, timeout: float | None = None) -> bool:
        """True once [offs, offs + size) has arrived; False if the download failed (or timed out)."""
        with self.cv:
            self.waits += 1
            if self.covered(offs, size):
                return True
            t0 = self.clock()
            self.blocked += 1
            ok = self.cv.wait_for(
                lambda: self.error is not None or self.covered(offs, size), timeout
            )
            self.blocked_s += self.clock() - t0
            return bool(ok) and self.error is None


def _download_part(
    client, bucket: str, key: str, fd: int, k: int, start: int, end: int, prog: Progress
) -> None:
    """loader._download_part, reporting progress after every pwrite."""
    pos = start
    for attempt in range(1, ATTEMPTS + 1):
        try:
            body = client().get_object(Bucket=bucket, Key=key, Range=f"bytes={pos}-{end}")["Body"]
            for chunk in body.iter_chunks(BODY_CHUNK):
                os.pwrite(fd, chunk, pos)
                pos += len(chunk)
                prog.advance(k, pos)
        except Exception as e:  # a reset connection, a timeout, a short read
            if attempt == ATTEMPTS:
                raise
            log.warning("range %d-%d broke off at %d (%s); retrying", start, end, pos, e)
            _retry_sleep(0.5 * 2**attempt)
            continue
        if pos == end + 1:
            return
        if attempt == ATTEMPTS:
            raise OSError(f"range {start}-{end} of {key} ended at {pos}")


def _lib_dir() -> str:
    return os.environ.get("DEC_GGML_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lib"
    )


class Hooks:
    """The E4 llama.cpp hooks bound to one Progress. install() before the load, clear() after it."""

    def __init__(self, prog: Progress, by_arrival: bool, lib_dir: str | None = None):
        d = lib_dir or _lib_dir()
        # Same files the shim's rpath resolves, so these are the instances llama_model_load_from_file uses.
        self.llama = ctypes.CDLL(os.path.join(d, "libllama.so"))
        self.base = ctypes.CDLL(os.path.join(d, "libggml-base.so"))
        self.llama.llama_set_load_stream.argtypes = [WAIT_T, ORDER_T, ctypes.c_void_p]
        self.llama.llama_set_load_stream.restype = None
        self.base.gguf_set_read_wait.argtypes = [WAIT_T, ctypes.c_void_p]
        self.base.gguf_set_read_wait.restype = None
        self.prog, self.by_arrival = prog, by_arrival
        self.order_calls = 0

        def wait(_ud, offs, size):
            try:
                return 0 if prog.wait(offs, size) else 1
            except BaseException:  # an exception must not cross into C
                return 1

        def order(_ud, offs, size):
            self.order_calls += 1
            try:
                return order_key(offs, size, prog.chunk)
            except BaseException:
                return 0

        self.wait_cb, self.order_cb = WAIT_T(wait), ORDER_T(order)

    def install(self) -> None:
        self.base.gguf_set_read_wait(self.wait_cb, None)
        self.llama.llama_set_load_stream(
            self.wait_cb, self.order_cb if self.by_arrival else ORDER_T(), None
        )

    def clear(self) -> None:
        self.base.gguf_set_read_wait(WAIT_T(), None)
        self.llama.llama_set_load_stream(WAIT_T(), ORDER_T(), None)


# Keys _record files as phase durations (seconds, lkai_entry REC.init); everything else is a counter.
PHASE_KEYS = frozenset(
    {"weight_download", "stream_header_wait", "stream_load_blocked", "download_and_load"}
)


def _record(**kv: Any) -> None:
    """Merge into STATS, and into lkai_entry's phase record when the instrumented entry point is loaded.

    A key `<phase>_s` with <phase> in PHASE_KEYS is a duration and goes to REC.init[<phase>]; the rest are counters."""
    STATS.update(kv)
    rec = getattr(sys.modules.get("lkai_entry"), "REC", None)
    if rec is None:
        return
    with rec._lock:  # noqa: SLF001
        for k, v in kv.items():
            phase = k.removesuffix("_s")
            if k.endswith("_s") and phase in PHASE_KEYS:
                rec.init[phase] = round(float(v), 6)
            else:
                rec.counters[k] = v


def download_to_memfd_streaming(bucket: str, key: str, header_bytes: int = 1 << 20) -> str:
    """Start loader.download_to_memfd's download and return /proc/self/fd/<n> once the first `header_bytes` (rounded
    up to whole 8 MiB body chunks by the range they live in) have arrived, with the hooks installed.

    The caller then loads the model from that path; the hooks block every read until its bytes are present."""
    client = _pr77_loader._client  # noqa: SLF001
    t0 = time.monotonic()
    size = client().head_object(Bucket=bucket, Key=key)["ContentLength"]
    fd = _pr77_loader._memfd(os.path.basename(key))  # noqa: SLF001
    try:
        os.ftruncate(fd, size)
    except OSError as e:
        os.close(fd)
        raise RuntimeError(f"not enough memory for {key} ({size / 2**20:.0f} MiB)") from e
    prog = Progress(size)
    hooks = Hooks(prog, by_arrival=os.environ.get("STREAM_ORDER", "arrival") != "file")
    _keep.extend([prog, hooks])
    _pr77_loader._fds.append(fd)  # noqa: SLF001
    pool = ThreadPoolExecutor(max_workers=WORKERS)

    def run() -> None:
        futs = [
            pool.submit(_download_part, client, bucket, key, fd, k, s, e, prog)
            for k, (s, e) in enumerate(prog.parts)
        ]
        try:
            for f in futs:
                f.result()
        except BaseException as e:
            prog.fail(e)
            STATS["download_error"] = repr(e)
            log.error("streamed download of s3://%s/%s failed: %r", bucket, key, e)
            pool.shutdown(wait=False, cancel_futures=True)
            return
        # every range is done: no idle worker threads left in a SnapStart snapshot
        pool.shutdown(wait=True)
        STATS["download_s"] = time.monotonic() - t0
        log.info(
            "streamed s3://%s/%s (%.0f MiB) in %.1fs",
            bucket,
            key,
            size / 2**20,
            STATS["download_s"],
        )

    th = threading.Thread(target=run, name="e4-download", daemon=True)
    th.start()
    if not prog.wait(0, min(header_bytes, size)):
        raise RuntimeError(
            f"streamed download of {key} failed before the header arrived"
        ) from prog.error
    hooks.install()
    STATS.update(t_download_start=t0, path=f"/proc/self/fd/{fd}", hooks=hooks, thread=th, size=size)
    STATS["header_wait_s"] = time.monotonic() - t0
    return f"/proc/self/fd/{fd}"


def finish_load() -> None:
    """After llama_model_load_from_file returned (or failed): clear the hooks, record the stream counters, and raise
    the download's exception if it failed.

    Records (into lkai_entry's phase record, overwriting what its download_to_memfd wrapper measured, which for a
    streamed load is only the header wait): weight_download = the whole download, HEAD to last byte;
    stream_header_wait; stream_load_blocked = time the loader spent blocked on bytes; stream_waits[_blocked]."""
    hooks: Hooks | None = STATS.pop("hooks", None)
    if hooks is None:
        return
    prog = hooks.prog
    hooks.clear()
    th = STATS.pop("thread", None)
    if th is not None and prog.complete():
        th.join()  # sets download_s
    kv: dict[str, Any] = {
        "stream_header_wait_s": STATS.get("header_wait_s", 0.0),
        "stream_load_blocked_s": prog.blocked_s,
        "stream_waits": prog.waits,
        "stream_waits_blocked": prog.blocked,
        "stream_order_calls": hooks.order_calls,
        "stream_order": "arrival" if hooks.by_arrival else "file",
    }
    if "download_s" in STATS:
        dl = STATS["download_s"]
        kv |= {
            "weight_download_s": dl,
            "weight_bytes": STATS["size"],
            "weight_mib_s": round(STATS["size"] / 2**20 / dl, 1),
        }
    _record(**kv)
    if prog.error is not None:
        raise RuntimeError("streamed weight download failed") from prog.error
    if not prog.complete():  # the patched loader waits for the whole file; this would be a defect
        raise RuntimeError("model load returned before the streamed download completed")
