"""E3 download strategies: the GGUF from S3 into a memfd, by one of several parallel strategies. No disk.

The single variable of experiment E3 is how the bytes get from S3 into memory. Every strategy writes into the same
destination as P1's loader (an anonymous memfd, returned as /proc/self/fd/<n> for llama.cpp to mmap), from the same
bucket and object, so the format, loader and everything after the download stay as P1 measured them.

E3_DL selects the strategy (the function's environment):

  loader                       P1's own loader.download_to_memfd, unchanged (the A/A arm: same bytes path as the
                               control, but through this wrapper and zip)
  threads:<chunk_mib>:<workers>
                               P1's algorithm (one HEAD, then ranged GETs on a thread pool, each resumed from the byte
                               where it broke off) with the part size and thread count as parameters.
                               threads:64:32 is P1's setting
  crt:<part_mib>:<gbps>        awscrt's S3Client (the CRT transfer engine boto3's CRT transfer manager uses): one
                               auto-ranged GET split into <part_mib> parts, sized for <gbps> Gbit/s, each body chunk
                               pwritten at its offset from CRT's callback. No HEAD: the first part tells CRT the size
  crtfile:<part_mib>:<gbps>    the same client, with CRT writing the body itself to /proc/self/fd/<n> (recv_filepath),
                               so no Python runs per chunk

E3_VERIFY=1 also hashes the memfd (sha256) after the download and records it. It is only set on the verification
invocations, never on measured ones, because hashing 2 GB adds seconds.

Counters (e3_*) go into the baseline Recorder (lkai_entry.REC), so they ride in the same `LKAI` records as the phases:
the strategy, GET and HEAD requests issued (CRT's are computed from the part size and labelled so), retries, the
process's peak RSS after the download, the /tmp bytes in use, and the client shutdown time. CRT's client is shut down
before the download returns, so no CRT thread or connection is live when SnapStart takes its snapshot.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import mmap
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger("decider.e3dl")

MIB = 1024 * 1024
ATTEMPTS = 5  # P1's loader.ATTEMPTS
BODY_CHUNK = 8 * MIB  # P1's iter_chunks size


def parse(spec: str) -> dict[str, Any]:
    """E3_DL as a dict: {"kind": loader|threads|crt|crtfile, ...}. Raises ValueError on anything else."""
    parts = spec.strip().split(":")
    kind = parts[0]
    if kind == "loader" and len(parts) == 1:
        return {"kind": "loader"}
    if kind == "threads" and len(parts) == 3:
        chunk, workers = int(parts[1]), int(parts[2])
        if chunk < 1 or workers < 1:
            raise ValueError(f"E3_DL {spec!r}: chunk and workers must be positive")
        return {"kind": "threads", "chunk": chunk * MIB, "workers": workers}
    if kind in ("crt", "crtfile") and len(parts) == 3:
        part, gbps = int(parts[1]), float(parts[2])
        if part < 5 or gbps <= 0:  # CRT's minimum part size is 5 MiB
            raise ValueError(f"E3_DL {spec!r}: part >= 5 MiB and gbps > 0")
        return {"kind": kind, "part": part * MIB, "gbps": gbps}
    raise ValueError(f"E3_DL {spec!r}: expected loader, threads:<mib>:<n>, crt:<mib>:<gbps> or crtfile:...")


def counters() -> dict[str, Any]:
    """The baseline Recorder's counters when lkai_entry is loaded (also mid-import), else a scratch dict."""
    rec = getattr(sys.modules.get("lkai_entry"), "REC", None)
    return rec.counters if rec is not None else _SCRATCH


_SCRATCH: dict[str, Any] = {}
_fds: list[int] = []  # like loader._fds: llama mmaps the memfd, so it stays open for the process lifetime


def memfd(name: str) -> int:
    return os.memfd_create(name, 0)


def peak_rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def tmp_used_mb(path: str = "/tmp") -> float | None:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return round((st.f_blocks - st.f_bfree) * st.f_frsize / MIB, 1)


def sha256_fd(fd: int) -> str:
    size = os.fstat(fd).st_size
    h = hashlib.sha256()
    if size:
        with mmap.mmap(fd, size, prot=mmap.PROT_READ) as m:
            for off in range(0, size, 64 * MIB):
                h.update(m[off : off + 64 * MIB])
    return h.hexdigest()


# ---------------------------------------------------------------------------- threads (P1's algorithm, parameterised)


class _Threads:
    def __init__(self, client: Any, bucket: str, key: str, fd: int):
        self.client, self.bucket, self.key, self.fd = client, bucket, key, fd
        self.gets = 0
        self.retries = 0
        self._lock = threading.Lock()

    def part(self, start: int, end: int) -> None:
        pos = start
        for attempt in range(1, ATTEMPTS + 1):
            with self._lock:
                self.gets += 1
            try:
                body = self.client.get_object(
                    Bucket=self.bucket, Key=self.key, Range=f"bytes={pos}-{end}"
                )["Body"]
                for chunk in body.iter_chunks(BODY_CHUNK):
                    os.pwrite(self.fd, chunk, pos)
                    pos += len(chunk)
            except Exception as e:  # a reset connection, a timeout, a short read
                if attempt == ATTEMPTS:
                    raise
                with self._lock:
                    self.retries += 1
                log.warning("range %d-%d broke off at %d (%s); retrying", start, end, pos, e)
                time.sleep(0.5 * 2**attempt)
                continue
            if pos == end + 1:
                return
            if attempt == ATTEMPTS:
                raise OSError(f"range {start}-{end} of {self.key} ended at {pos}")
            with self._lock:
                self.retries += 1


def ranges(size: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]


def download_threads(bucket: str, key: str, chunk: int, workers: int, client: Any = None) -> int:
    if client is None:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            config=Config(max_pool_connections=workers, retries={"max_attempts": ATTEMPTS}),
        )
    size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    fd = memfd(os.path.basename(key))
    try:
        os.ftruncate(fd, size)
        t = _Threads(client, bucket, key, fd)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda p: t.part(*p), ranges(size, chunk)))
    except BaseException:
        os.close(fd)
        raise
    c = counters()
    c.update(e3_get_requests=t.gets, e3_head_requests=1, e3_retries=t.retries)
    c["e3_requests_source"] = "counted"
    return fd


# ---------------------------------------------------------------------------- CRT


def download_crt(
    bucket: str, key: str, part: int, gbps: float, to_file: bool, region: str | None = None
) -> int:
    import awscrt
    from awscrt import auth, http, io, s3

    region = region or os.environ.get("AWS_REGION") or os.environ["AWS_DEFAULT_REGION"]
    fd = memfd(os.path.basename(key))
    elg = io.EventLoopGroup()
    bootstrap = io.ClientBootstrap(elg, io.DefaultHostResolver(elg))
    creds = auth.AwsCredentialsProvider.new_default_chain(bootstrap)
    client = s3.S3Client(
        bootstrap=bootstrap,
        region=region,
        credential_provider=creds,
        part_size=part,
        throughput_target_gbps=gbps,
    )
    host = f"{bucket}.s3.{region}.amazonaws.com"
    from urllib.parse import quote

    req = http.HttpRequest("GET", "/" + quote(key), http.HttpHeaders([("Host", host)]))
    got = {"bytes": 0, "status": None, "length": None}
    lock = threading.Lock()

    def on_headers(status_code: int, headers: list, **_: Any) -> None:
        got["status"] = status_code
        for k, v in headers:
            if k.lower() == "content-length":
                got["length"] = int(v)

    def on_body(chunk: bytes, offset: int, **_: Any) -> None:
        # pwrite past EOF grows the memfd, so no HEAD or ftruncate is needed first.
        os.pwrite(fd, chunk, offset)
        with lock:
            got["bytes"] += len(chunk)

    kw: dict[str, Any] = {"on_headers": on_headers}
    if to_file:
        kw["recv_filepath"] = f"/proc/self/fd/{fd}"
    else:
        kw["on_body"] = on_body
    shutdown = [client.shutdown_event, bootstrap.shutdown_event, elg.shutdown_event]
    r = None
    try:
        r = client.make_request(type=s3.S3RequestType.GET_OBJECT, request=req, **kw)
        r.finished_future.result()
    except BaseException:
        os.close(fd)
        raise
    finally:
        # No CRT thread or socket may outlive the download into a SnapStart snapshot. CRT releases its native
        # client only when the last Python reference goes, and the request object holds one.
        t = time.monotonic()
        del r, client, creds, bootstrap, elg, req, kw
        gc.collect()
        clean = all(e.wait(10) for e in shutdown)
        shutdown_s = round(time.monotonic() - t, 4)
    size = os.fstat(fd).st_size
    if got["length"] is not None and size != got["length"]:
        os.close(fd)
        raise OSError(f"crt wrote {size} bytes of {key}, Content-Length {got['length']}")
    c = counters()
    c.update(
        e3_get_requests=math.ceil(size / part),
        e3_head_requests=0,
        e3_retries=None,
        e3_requests_source="computed: ceil(size / part_size); CRT does not report its request count",
        e3_crt_shutdown_s=shutdown_s,
        e3_crt_shutdown_clean=clean,
        e3_awscrt=awscrt.__version__ if hasattr(awscrt, "__version__") else None,
    )
    if not to_file:
        c["e3_crt_body_bytes"] = got["bytes"]
    return fd


# ---------------------------------------------------------------------------- entry


def make_download(orig: Any, spec: str | None = None, verify: bool | None = None) -> Any:
    """A replacement for loader.download_to_memfd(bucket, key) -> "/proc/self/fd/<n>" for the E3_DL strategy."""
    cfg = parse(spec if spec is not None else os.environ.get("E3_DL", "loader"))
    if verify is None:
        verify = os.environ.get("E3_VERIFY") == "1"

    def download_to_memfd(bucket: str, key: str) -> str:
        c = counters()
        c["e3_strategy"] = spec if spec is not None else os.environ.get("E3_DL", "loader")
        if cfg["kind"] == "loader":
            path = orig(bucket, key)
            fd = int(path.rsplit("/", 1)[1])
            size = os.fstat(fd).st_size
            c.update(
                e3_get_requests=math.ceil(size / (64 * MIB)),
                e3_head_requests=1,
                e3_retries=None,
                e3_requests_source="computed: P1 loader, ceil(size / 64 MiB) + 1 HEAD",
            )
        else:
            if cfg["kind"] == "threads":
                fd = download_threads(bucket, key, cfg["chunk"], cfg["workers"])
            else:
                fd = download_crt(bucket, key, cfg["part"], cfg["gbps"], cfg["kind"] == "crtfile")
            _fds.append(fd)
            path = f"/proc/self/fd/{fd}"
        c["e3_peak_rss_mb_after_download"] = peak_rss_mb()
        c["e3_tmp_used_mb"] = tmp_used_mb()
        if verify:
            t = time.monotonic()
            c["e3_weight_sha256"] = sha256_fd(int(path.rsplit("/", 1)[1]))
            c["e3_hash_s"] = round(time.monotonic() - t, 3)
        log.info("e3 %s: %s", c["e3_strategy"], {k: v for k, v in c.items() if k.startswith("e3_")})
        return path

    return download_to_memfd
