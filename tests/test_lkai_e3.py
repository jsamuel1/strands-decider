"""examples/lambda-kleidiai/e3: the E3 download strategies and the entry point that installs them.

No AWS and no model: the threads strategy runs against an in-memory S3 stand-in that can break a body off mid-range,
and the entry point runs against the stand-ins test_lkai_baseline.py builds for P1's handler. They pin E3's contract:
every strategy leaves exactly the object's bytes in a memfd, the P1 setting issues P1's request count, and the
strategy is in place before P1's lkai_entry wraps the loader, so P1's weight_download phase times it.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import sys

import pytest

E3 = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e3")
sys.path.insert(0, os.path.abspath(E3))
sys.path.insert(0, os.path.dirname(__file__))

import e3dl  # noqa: E402
from test_lkai_baseline import _ctx, _install_fakes  # noqa: E402  (puts baseline/ on the path)

import phases  # noqa: E402  isort: skip

MIB = 1024 * 1024


def test_parse_specs():
    assert e3dl.parse("loader") == {"kind": "loader"}
    assert e3dl.parse("threads:64:32") == {"kind": "threads", "chunk": 64 * MIB, "workers": 32}
    assert e3dl.parse("crt:16:10") == {"kind": "crt", "part": 16 * MIB, "gbps": 10.0}
    assert e3dl.parse("crtfile:8:25")["kind"] == "crtfile"
    for bad in ("", "threads:64", "threads:0:4", "crt:4:10", "crt:16:0", "http:1:1", "loader:1"):
        with pytest.raises(ValueError):
            e3dl.parse(bad)


@pytest.mark.parametrize(("size", "chunk"), [(1, 4), (10, 4), (12, 4), (64 * MIB + 3, 64 * MIB)])
def test_ranges_cover_every_byte_once(size, chunk):
    rs = e3dl.ranges(size, chunk)
    assert rs[0][0] == 0 and rs[-1][1] == size - 1
    assert all(b[0] == a[1] + 1 for a, b in itertools.pairwise(rs))
    assert len(rs) == -(-size // chunk)


class _Body:
    def __init__(self, data: bytes, break_after: int | None):
        self.data, self.break_after = data, break_after

    def iter_chunks(self, n):
        sent = 0
        for i in range(0, len(self.data), 7):  # odd-sized chunks, so offsets are exercised
            if self.break_after is not None and sent >= self.break_after:
                raise ConnectionResetError("body broke off")
            piece = self.data[i : i + 7]
            sent += len(piece)
            yield piece


class _S3:
    """head_object/get_object over one in-memory object. The first GET of every range starting at a multiple of 3
    chunks breaks off after 5 bytes, so the resume-from-byte path runs."""

    def __init__(self, data: bytes):
        self.data, self.calls, self.broke = data, [], set()

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.data)}

    def get_object(self, Bucket, Key, Range):
        a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        self.calls.append((a, b))
        brk = None
        if a % 30 == 0 and a not in self.broke:
            self.broke.add(a)
            brk = 5
        return {"Body": _Body(self.data[a : b + 1], brk)}


def _read_fd(fd):
    return os.pread(fd, os.fstat(fd).st_size, 0)


def test_threads_writes_exact_bytes_and_counts_requests(monkeypatch):
    monkeypatch.setattr(e3dl.time, "sleep", lambda s: None)
    data = os.urandom(1000)
    s3 = _S3(data)
    e3dl._SCRATCH.clear()
    fd = e3dl.download_threads("b", "k", chunk=10, workers=8, client=s3)
    try:
        assert _read_fd(fd) == data
        assert e3dl.sha256_fd(fd) == hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)
    c = e3dl._SCRATCH
    assert c["e3_get_requests"] == len(s3.calls) == 100 + len(s3.broke)
    assert c["e3_retries"] == len(s3.broke) > 0
    assert c["e3_head_requests"] == 1


def test_threads_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(e3dl.time, "sleep", lambda s: None)

    class Always(_S3):
        def get_object(self, Bucket, Key, Range):
            return {"Body": _Body(b"x" * 10, 0)}

    with pytest.raises(ConnectionResetError):
        e3dl.download_threads("b", "k", chunk=10, workers=1, client=Always(b"y" * 10))


def test_loader_kind_is_p1_loader_with_p1_request_count(monkeypatch):
    fd = os.memfd_create("t", 0)
    os.ftruncate(fd, 2076712416)  # the deployed GGUF's size; sparse, so no memory is used
    calls = []

    def orig(bucket, key):
        calls.append((bucket, key))
        return f"/proc/self/fd/{fd}"

    e3dl._SCRATCH.clear()
    try:
        path = e3dl.make_download(orig, "loader", verify=False)("b", "k")
    finally:
        os.close(fd)
    assert path == f"/proc/self/fd/{fd}" and calls == [("b", "k")]
    # P1's README: "each one HEAD and 31 ranged GETs"
    assert e3dl._SCRATCH["e3_get_requests"] == 31 and e3dl._SCRATCH["e3_head_requests"] == 1
    assert "e3_weight_sha256" not in e3dl._SCRATCH


def test_verify_records_sha256(monkeypatch):
    data = b"weights" * 1000
    monkeypatch.setattr(e3dl, "download_threads", lambda b, k, chunk, workers: _fd_with(data))
    e3dl._SCRATCH.clear()
    path = e3dl.make_download(None, "threads:64:32", verify=True)("b", "k")
    os.close(int(path.rsplit("/", 1)[1]))
    assert e3dl._SCRATCH["e3_weight_sha256"] == hashlib.sha256(data).hexdigest()
    assert e3dl._SCRATCH["e3_strategy"] == "threads:64:32"
    assert e3dl._SCRATCH["e3_tmp_used_mb"] is not None


def _fd_with(data: bytes) -> int:
    fd = os.memfd_create("t", 0)
    os.write(fd, data)
    return fd


def test_entry_installs_strategy_before_p1_wraps_loader(monkeypatch, capsys):
    """e3_entry patches loader.download_to_memfd, then imports P1's lkai_entry, whose weight_download phase must time
    the E3 strategy (not P1's loader) and whose LKAI record must carry the e3_* counters."""
    monkeypatch.setenv("LKAI_PHASES", "1")
    monkeypatch.setenv("E3_DL", "threads:16:64")
    monkeypatch.delenv("E3_VERIFY", raising=False)
    reg = _install_fakes(monkeypatch, "snap-start")
    monkeypatch.delitem(sys.modules, "e3_entry", raising=False)
    used = []

    def fake_threads(bucket, key, chunk, workers):
        used.append((bucket, key, chunk, workers))
        e3dl.counters().update(e3_get_requests=124, e3_head_requests=1, e3_retries=0)
        return _fd_with(b"g" * 64)

    monkeypatch.setattr(e3dl, "download_threads", fake_threads)
    import e3_entry

    assert used == [("b", "k", 16 * MIB, 64)]
    reg["before"]()
    reg["after"]()
    out = e3_entry.handler({"state": "s"}, _ctx(1))
    assert out["answers"] == {"q": {"type": "noul", "noul": 0.25}}
    recs = phases.parse_lkai(capsys.readouterr().out)
    inv = next(r for r in recs if r["kind"] == "invoke")
    assert "weight_download" in inv["init"]
    assert inv["counters"]["weight_bytes"] == 64
    assert inv["counters"]["e3_strategy"] == "threads:16:64"
    assert inv["counters"]["e3_get_requests"] == 124
    assert inv["counters"]["e3_preimport_s"] >= 0


def test_entry_report_only_arm(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "0")
    monkeypatch.setenv("E3_DL", "loader")
    _install_fakes(monkeypatch, "snap-start")
    monkeypatch.delitem(sys.modules, "e3_entry", raising=False)
    import e3_entry

    assert e3_entry.handler({"state": "s"}, _ctx(1))["answers"]
    assert phases.parse_lkai(capsys.readouterr().out) == []


def test_sha256_fd_empty_and_large_stride():
    fd = os.memfd_create("t", 0)
    try:
        assert e3dl.sha256_fd(fd) == hashlib.sha256(b"").hexdigest()
        data = io.BytesIO(os.urandom(3 * MIB + 5)).getvalue()
        os.write(fd, data)
        assert e3dl.sha256_fd(fd) == hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)
