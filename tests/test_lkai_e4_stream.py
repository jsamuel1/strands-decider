"""examples/lambda-kleidiai/e4_stream: the streamed weight download and the llama.cpp load-stream patch.

No llama.cpp, model or AWS needed: the download runs against an in-memory fake S3 client that serves byte ranges
(optionally throttled or failing), and the patch is applied to synthetic files carrying its anchors. The real
end-to-end check (patched llama.cpp on arm64, a streamed load giving bit-identical hidden states) is
e4_stream/localtest.py, run on the build host.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
import types

import pytest

E4 = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e4_stream")
sys.path.insert(0, os.path.abspath(E4))

import patch_stream  # noqa: E402

MIB = 1 << 20


@pytest.fixture
def sl(monkeypatch):
    """stream_loader with a stand-in PR #77 loader (its constants and memfd helper; boto3 is never imported)."""
    for m in ("stream_loader", "loader"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    loader = types.ModuleType("loader")
    loader.CHUNK, loader.WORKERS, loader.ATTEMPTS = 64 * MIB, 32, 5
    loader._fds = []
    loader._memfd = lambda name: os.memfd_create(name)
    monkeypatch.setitem(sys.modules, "loader", loader)
    import stream_loader

    yield stream_loader
    for fd in loader._fds:
        os.close(fd)


# ------------------------------------------------------------------------------------------------ order and ranges


def test_parts_match_pr77_loader(sl):
    size = 2_076_712_416  # the baseline GGUF
    parts = sl.parts_for(size)
    assert len(parts) == 31
    assert parts[0] == (0, 64 * MIB - 1)
    assert parts[-1][1] == size - 1
    assert all(b[0] == a[1] + 1 for a, b in itertools.pairwise(parts))


def test_order_key_is_bytes_needed_from_the_slowest_range(sl):
    c = 64 * MIB
    assert sl.order_key(0, 10, c) == 10
    assert sl.order_key(c + 5, 10, c) == 15  # 15 bytes into range 1
    # spans the end of range 0 (needs all of it) and 7 bytes of range 1
    assert sl.order_key(c - 3, 10, c) == c
    assert sl.order_key(5 * c + 100, 0, c) == 0


def test_order_key_follows_the_download_front(sl):
    """Sorting by order_key puts a tensor early in range 5 before a tensor late in range 0."""
    c = 64 * MIB
    late_in_0, early_in_5 = (c - 4 * MIB, MIB), (5 * c + MIB, MIB)
    assert sl.order_key(*early_in_5, c) < sl.order_key(*late_in_0, c)


# ------------------------------------------------------------------------------------------------ Progress


def test_progress_covered_and_wait(sl):
    p = sl.Progress(10 * MIB, chunk=4 * MIB)  # ranges [0,4) [4,8) [8,10) MiB
    assert not p.covered(0, 1)
    p.advance(0, 2 * MIB)
    assert p.covered(0, 2 * MIB) and not p.covered(0, 2 * MIB + 1)
    p.advance(1, 4 * MIB + 10)
    assert not p.covered(3 * MIB, 2 * MIB)  # needs range 0 to its end
    p.advance(0, 4 * MIB)
    assert p.covered(3 * MIB, MIB + 10) and not p.covered(3 * MIB, MIB + 11)
    assert p.covered(9 * MIB, 0)  # empty span
    assert p.covered(9 * MIB, 50 * MIB) is False  # clipped to the file, range 2 empty

    got = []
    t = threading.Thread(target=lambda: got.append(p.wait(8 * MIB, 2 * MIB)))
    t.start()
    time.sleep(0.05)
    assert t.is_alive()
    p.advance(2, 10 * MIB)
    t.join(2)
    assert got == [True] and p.blocked == 1 and p.blocked_s > 0
    assert not p.complete()
    p.advance(1, 8 * MIB)
    assert p.complete()


def test_progress_advance_never_goes_backwards(sl):
    p = sl.Progress(8 * MIB, chunk=4 * MIB)
    p.advance(1, 6 * MIB)
    p.advance(
        1, 5 * MIB
    )  # a retried range restarts from where it stopped, never earlier; ignore regressions
    assert p.done[1] == 2 * MIB


def test_progress_failure_releases_waiters(sl):
    p = sl.Progress(8 * MIB, chunk=4 * MIB)
    got = []
    t = threading.Thread(target=lambda: got.append(p.wait(0, 8 * MIB)))
    t.start()
    time.sleep(0.05)
    p.fail(OSError("reset"))
    t.join(2)
    assert got == [False]
    assert p.wait(0, 1) is False  # later waiters fail at once


# ------------------------------------------------------------------------------------------------ streamed download


class _Body:
    def __init__(self, data: bytes, delay: float, fail_after: int | None):
        self.data, self.delay, self.fail_after = data, delay, fail_after

    def iter_chunks(self, n):
        for i in range(0, len(self.data), n):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionResetError("body broke off")
            time.sleep(self.delay)
            yield self.data[i : i + n]


class FakeS3:
    """Serves Range GETs of `blob`. `breaks` = {range_start: bytes served before the first attempt breaks off}."""

    def __init__(
        self, blob: bytes, delay: float = 0.0, breaks: dict | None = None, fatal: bool = False
    ):
        self.blob, self.delay, self.breaks, self.fatal = blob, delay, dict(breaks or {}), fatal
        self.span = MIB  # the range size the tests use
        self.gets: list[str] = []
        self.lock = threading.Lock()

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.blob)}

    def get_object(self, Bucket, Key, Range):
        a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        with self.lock:
            self.gets.append(Range)
            if (
                self.fatal
            ):  # every attempt at a range in `breaks`, resumed or not, breaks off at once
                fail = 0 if any(s <= a <= s + self.span for s in self.breaks) else None
            else:
                fail = self.breaks.pop(a, None)
        return {"Body": _Body(self.blob[a : b + 1], self.delay, fail)}


class _FakeHooks:
    def __init__(self, prog, by_arrival, lib_dir=None):
        self.prog, self.by_arrival, self.order_calls = prog, by_arrival, 0
        self.installed = self.cleared = 0

    def install(self):
        self.installed += 1

    def clear(self):
        self.cleared += 1


def _patch_small(sl, monkeypatch, s3, chunk=MIB, body=64 * 1024):
    monkeypatch.setattr(sl, "CHUNK", chunk)
    monkeypatch.setattr(sl, "BODY_CHUNK", body)
    monkeypatch.setattr(
        sl,
        "parts_for",
        lambda size, c=chunk: [(s, min(s + c, size) - 1) for s in range(0, size, c)],
    )
    real_progress = sl.Progress
    monkeypatch.setattr(sl, "Progress", lambda size: real_progress(size, chunk))
    monkeypatch.setattr(sl, "Hooks", _FakeHooks)
    sys.modules["loader"]._client = lambda: s3


def _memfd_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_streamed_download_returns_at_header_then_completes_identically(sl, monkeypatch):
    blob = os.urandom(5 * MIB + 12345)
    s3 = FakeS3(blob, delay=0.002)
    _patch_small(sl, monkeypatch, s3)
    t = time.monotonic()
    path = sl.download_to_memfd_streaming("b", "k", header_bytes=64 * 1024)
    t_header = time.monotonic() - t
    hooks = sl.STATS["hooks"]
    assert hooks.installed == 1
    assert not hooks.prog.complete()  # returned while the download is still running
    assert hooks.prog.wait(0, len(blob))  # the loader's final whole-file wait
    sl.finish_load()
    assert hooks.cleared == 1
    assert _memfd_bytes(path) == blob
    assert len(s3.gets) == 6  # one GET per range, as loader.download_to_memfd
    assert sl.STATS["download_s"] >= t_header


def test_streamed_download_resumes_a_broken_range_from_its_byte(sl, monkeypatch):
    blob = os.urandom(3 * MIB)
    s3 = FakeS3(blob, breaks={MIB: 3 * 64 * 1024})
    _patch_small(sl, monkeypatch, s3)
    monkeypatch.setattr(sl, "_retry_sleep", lambda s: None)  # retry back-off
    path = sl.download_to_memfd_streaming("b", "k", header_bytes=1)
    assert sl.STATS["hooks"].prog.wait(0, len(blob))
    sl.finish_load()
    assert _memfd_bytes(path) == blob
    assert f"bytes={MIB + 3 * 64 * 1024}-{2 * MIB - 1}" in s3.gets  # resumed where it broke off


def test_streamed_download_failure_fails_the_load(sl, monkeypatch):
    blob = os.urandom(3 * MIB)
    s3 = FakeS3(blob, breaks={2 * MIB: 64 * 1024}, fatal=True)  # range 2 breaks on every attempt
    _patch_small(sl, monkeypatch, s3)
    monkeypatch.setattr(sl, "_retry_sleep", lambda s: None)
    sl.download_to_memfd_streaming("b", "k", header_bytes=1)
    prog = sl.STATS["hooks"].prog
    assert prog.wait(0, len(blob)) is False  # the loader's wait returns non-zero, so the load fails
    with pytest.raises(RuntimeError, match="streamed weight download failed"):
        sl.finish_load()
    assert "download_error" in sl.STATS


def test_finish_load_refuses_an_incomplete_download(sl, monkeypatch):
    blob = os.urandom(2 * MIB)
    s3 = FakeS3(blob, delay=0.05)
    _patch_small(sl, monkeypatch, s3)
    sl.download_to_memfd_streaming("b", "k", header_bytes=1)
    prog = sl.STATS["hooks"].prog
    with pytest.raises(RuntimeError, match="before the streamed download completed"):
        sl.finish_load()  # as if the patched loader had returned without its final wait
    assert prog.wait(0, len(blob))  # let the download finish before the fixture closes the memfd


def test_record_routes_phases_and_counters(sl, monkeypatch):
    rec = types.SimpleNamespace(init={}, counters={}, _lock=threading.Lock())
    monkeypatch.setitem(sys.modules, "lkai_entry", types.SimpleNamespace(REC=rec))
    sl._record(weight_download_s=24.5, weight_mib_s=81.0, stream_waits=7, download_and_load_s=25)
    assert rec.init == {"weight_download": 24.5, "download_and_load": 25.0}
    assert rec.counters == {"weight_mib_s": 81.0, "stream_waits": 7}


# ------------------------------------------------------------------------------------------------ the patch


def _synthetic_tree(tmp_path):
    files = {
        "include/llama.h": "// llama.h\n" + patch_stream.HDR_ANCHOR + "// end\n",
        "src/llama-model-loader.cpp": patch_stream.GLOBALS_ANCHOR
        + "\nvoid f() {\n"
        + patch_stream.ORDER_ANCHOR
        + "\n    }\n"
        + patch_stream.MMAP_ANCHOR
        + "\n"
        + patch_stream.READ_ANCHOR
        + "\n"
        + patch_stream.FINAL_ANCHOR
        + "\n}\n",
        "ggml/include/gguf.h": patch_stream.GGUF_H_ANCHOR,
        "ggml/src/gguf.cpp": patch_stream.GGUF_C_ANCHOR + "    return 0;\n}\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return files


def test_patch_applies_once_and_is_idempotent(tmp_path):
    _synthetic_tree(tmp_path)
    assert patch_stream.patch(tmp_path) is True
    h = (tmp_path / "include/llama.h").read_text()
    s = (tmp_path / "src/llama-model-loader.cpp").read_text()
    g = (tmp_path / "ggml/src/gguf.cpp").read_text()
    assert (
        "llama_set_load_stream" in h
        and "gguf_set_read_wait" in (tmp_path / "ggml/include/gguf.h").read_text()
    )
    # the wait precedes the mmap branch's reads, the read branch's reads, and the end of the load
    mmap_wait = s.index(patch_stream.MMAP_ADD)
    assert mmap_wait < s.index("validation_result.emplace_back")
    assert "llama_stream_wait(ggml_get_name(cur), weight->offs, n_size);  //" in s  # read branch
    assert s.index('llama_stream_wait("(whole file)"') < s.index("// unmap offloaded tensors")
    assert s.index("if (g_load_wait && g_load_order)") < s.index("if (!use_mmap) {")
    assert "g_gguf_ready" in g
    before = {p: p.read_text() for p in tmp_path.rglob("*.*")}
    assert patch_stream.patch(tmp_path) is False
    assert {p: p.read_text() for p in tmp_path.rglob("*.*")} == before


def test_patch_refuses_missing_or_duplicate_anchor(tmp_path):
    _synthetic_tree(tmp_path)
    p = tmp_path / "src/llama-model-loader.cpp"
    p.write_text(p.read_text() + patch_stream.READ_ANCHOR)  # now twice
    with pytest.raises(AssertionError, match="exactly once"):
        patch_stream.patch(tmp_path)
    assert (
        "E4 stream patch" not in (tmp_path / "include/llama.h").read_text()
    )  # nothing half-written


def test_patch_refuses_a_half_applied_tree(tmp_path):
    _synthetic_tree(tmp_path)
    p = tmp_path / "ggml/src/gguf.cpp"
    p.write_text(p.read_text() + "// strands-decider E4 stream patch\n")
    with pytest.raises(AssertionError, match="half-applied"):
        patch_stream.patch(tmp_path)


def test_bench_e4_uses_a_fresh_lambda_client_per_wave(monkeypatch):
    """bench_e4 changes only the client: each wave builds a new one with the same config, then runs bench's wave."""
    import importlib.util

    if not (importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore")):
        botocore = types.ModuleType("botocore")
        cfg = types.ModuleType("botocore.config")
        cfg.Config = lambda **kw: types.SimpleNamespace(**kw)
        exc = types.ModuleType("botocore.exceptions")
        exc.ClientError = type("ClientError", (Exception,), {})
        boto3 = types.ModuleType("boto3")
        boto3.Session = lambda **kw: None
        for name, mod in {
            "boto3": boto3,
            "botocore": botocore,
            "botocore.config": cfg,
            "botocore.exceptions": exc,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
    for m in ("bench", "bench_e4", "deploy"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    import bench_e4

    made = []

    class Sess:
        def client(self, svc, config=None):
            c = types.SimpleNamespace(svc=svc, meta=types.SimpleNamespace(config=config or "cfg"))
            made.append(c)
            return c

    seen = []
    monkeypatch.setattr(bench_e4, "_wave", lambda self, *a, **k: seen.append(self.lam) or [])
    b = object.__new__(bench_e4.bench.Bench)
    monkeypatch.setattr(
        bench_e4, "_init", lambda self, sess, *a, **k: setattr(self, "lam", sess.client("lambda"))
    )
    bench_e4.bench.Bench.__init__(b, Sess())
    first = b.lam
    b.wave("base", "1", 0, "restore")
    b.wave("stream", "1", 0, "restore")
    assert seen[0] is not first and seen[1] is not seen[0]
    assert all(c.svc == "lambda" and c.meta.config is first.meta.config for c in seen)


def _arm_sets():
    src = open(os.path.join(E4, "deploy_e4.py")).read()
    ns: dict = {"os": types.SimpleNamespace(environ={})}
    exec(src.split("# --- arms ---", 1)[1].split("# --- end arms ---", 1)[0], ns)
    return ns["ARMS_SETS"], ns["ARMS_ENV"]


def test_e4_arms_differ_only_in_e4_environment():
    sets, default = _arm_sets()
    assert default is sets["r2"]
    r1, r2 = sets["r1"], sets["r2"]
    assert r1 == {
        "base": {"E4_STREAM": "0"},
        "stream": {"E4_STREAM": "1", "STREAM_ORDER": "arrival"},
        "streamfile": {"E4_STREAM": "1", "STREAM_ORDER": "file"},
    }
    thr = str(4 * MIB)
    assert r2["base"] == r1["base"]
    assert r2["streamdyn"] == r1["stream"]  # run 1's experiment arm, re-measured
    assert r2["stream"] == r1["stream"] | {"E4_MMAP_THRESHOLD": thr}
    assert r2["basethr"] == r1["base"] | {"E4_MMAP_THRESHOLD": thr}
    for arms in sets.values():
        assert all(
            set(e) <= {"E4_STREAM", "STREAM_ORDER", "E4_MMAP_THRESHOLD"} for e in arms.values()
        )


def test_pin_mmap_threshold(sl, monkeypatch):
    calls = []
    libc = types.SimpleNamespace(mallopt=lambda p, v: calls.append((p, v)) or 1)
    monkeypatch.setattr(sl.ctypes, "CDLL", lambda name: libc)
    monkeypatch.delenv("E4_MMAP_THRESHOLD", raising=False)
    assert (
        sl.pin_mmap_threshold() is None and calls == []
    )  # unset: glibc's dynamic threshold, as P1
    monkeypatch.setenv("E4_MMAP_THRESHOLD", "0")
    assert sl.pin_mmap_threshold() is None and calls == []
    monkeypatch.setenv("E4_MMAP_THRESHOLD", str(4 * MIB))
    assert sl.pin_mmap_threshold() == 4 * MIB
    assert calls == [(-3, 4 * MIB)]  # M_MMAP_THRESHOLD
    libc.mallopt = lambda p, v: 0
    with pytest.raises(OSError, match="mallopt"):
        sl.pin_mmap_threshold()


def _row(arm, rnd, cls, cpu, total, mem, tail=""):
    rep = {"Duration": 1000.0 * total, "Max Memory Used": mem}
    rep["Init Duration" if cls == "cold" else "Restore Duration"] = 0.0
    return {
        "arm": arm,
        "round": rnd,
        "expected_class": cls,
        "cpu_part": cpu,
        "report": rep,
        "error": None,
        "class_mismatch": False,
        "latency_ms": None,
        "lkai": {"env_id": f"{arm}{rnd}{cls}{cpu}{total}"},
        "log_tail": tail,
    }


def test_round_deltas_stay_within_one_cpu_part_and_report_the_sign():
    """A part only one arm drew contributes nothing, and a fast part in one arm cannot offset a slow part in the
    other: each round's delta is within one part, and `sign` is "mixed" unless every round agrees."""
    import summarize_e4

    rows = [
        _row("base", 0, "cold", "p1", 10.0, 100),
        _row("stream", 0, "cold", "p1", 8.0, 90),
        _row("base", 1, "cold", "p1", 10.5, 100),
        _row("stream", 1, "cold", "p1", 8.4, 95),
        _row("base", 0, "cold", "p2", 20.0, 100),  # stream never drew p2
        _row("stream", 0, "cold", "p3", 1.0, 100),  # base never drew p3
        _row("base", 0, "restore", "p1", 4.0, 100),
        _row("stream", 0, "restore", "p1", 4.1, 100),
        _row("base", 1, "restore", "p1", 4.0, 100),
        _row("stream", 1, "restore", "p1", 3.9, 100),
    ]
    d = summarize_e4.round_deltas(rows, "stream")
    cold = d["cold"]["first_request_total"]
    assert set(cold) == {"p1"}
    assert cold["p1"]["per_round"] == [-2.0, -2.1] and cold["p1"]["sign"] == "negative"
    assert d["cold"]["report_max_memory_mb"]["p1"]["per_round"] == [-10, -5]
    assert d["restore"]["first_request_total"]["p1"]["sign"] == "mixed"


def test_host_variability_names_what_it_did_not_record(tmp_path):
    """Observed fields come from the rows and saved logs; unrecorded ones are listed as unknown, and no Graviton
    generation is named anywhere in the block."""
    import gzip
    import json
    import re

    import summarize_e4

    (tmp_path / "logs").mkdir()
    with gzip.open(tmp_path / "logs" / "fn.jsonl.gz", "wt") as fh:
        for s in ("a", "b"):
            fh.write(
                json.dumps(
                    {
                        "logStreamName": s,
                        "message": "INIT_START Runtime Version: python:3.12.v1\tARN",
                    }
                )
                + "\n"
            )
    tail = "decider.engine init cpu_part=p1 ggml_cpu=libx.so,liby.so threads=6 ubatch=512"
    rows = [
        _row("base", 0, "cold", "p1", 10.0, 100, tail),
        _row("stream", 0, "cold", "p1", 8.0, 90, tail),
    ]
    cond = {
        "arms": ["base", "stream"],
        "runtime_version_arn": "arn:x",
        "functions": {
            a: {"MemorySize": 10240, "Architectures": ["arm64"]} for a in ("base", "stream")
        },
    }
    h = summarize_e4.host_variability(str(tmp_path), cond, rows)
    o = h["observed"]
    assert o["ggml_cpu_variants_by_cpu_part"] == {"p1": {"libx.so,liby.so": 2}}
    assert o["os_cpu_count_by_cpu_part"] == {"p1": {"6": 2}}
    assert o["init_start_runtime_versions"] == {"python:3.12.v1": 2}
    assert o["cpu_parts_by_class_and_arm"]["cold"] == {"base": {"p1": 1}, "stream": {"p1": 1}}
    assert "AT_HWCAP / AT_HWCAP2" in h["unknown"] and "kernel release" in h["unknown"]
    assert set(h["per_round_deltas"]) == {"stream"}
    assert not re.search(r"Graviton\d", json.dumps(h))


def test_localtest_compares_the_observed_cpu_part_and_fails_on_mismatch(tmp_path):
    import subprocess

    script = open(os.path.join(E4, "localtest.sh")).read().split("python3 - ", 1)[1]
    body = script.split("<<'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
    out = tmp_path / "out"
    out.mkdir()
    reference = tmp_path / "reference.json"
    reference.write_text('{"outputs":{"r":{"p1":{"q":{"choice":"yes"}}}}}')

    def invoke(answer):
        (out / "base-0.json").write_text(
            __import__("json").dumps(
                {
                    "init": {"cpu_part": "p1"},
                    "counters": {},
                    "answers": {"r": {"q": {"choice": answer}}},
                }
            )
        )
        return subprocess.run(
            [sys.executable, "-c", body, str(out), str(reference)],
            capture_output=True,
            text=True,
        )

    assert invoke("yes").returncode == 0
    bad = invoke("no")
    assert bad.returncode != 0
    assert "output differs from P1 reference for CPU part p1" in bad.stderr


def test_unknown_cpu_parts_are_excluded_and_counted():
    """Unknown identities are unassessed rows, never a synthetic shared CPU stratum."""
    import summarize_e4

    rows = [
        _row("base", 0, "cold", None, 10.0, 100),
        _row("stream", 0, "cold", None, 8.0, 90),
    ]
    comparison = summarize_e4.compare_arms(rows, "stream")
    assert "cold" not in comparison
    assert comparison["unknown_cpu_part_rows"] == {"cold": {"base": 1, "stream": 1}}
    assert summarize_e4.round_deltas(rows, "stream") == {}


def test_mixed_known_and_unknown_cpu_parts_keep_unknown_counts(tmp_path):
    import gzip

    import summarize_e4

    (tmp_path / "logs").mkdir()
    with gzip.open(tmp_path / "logs" / "fn.jsonl.gz", "wt"):
        pass
    rows = [
        _row("base", 0, "cold", "p1", 10.0, 100),
        _row("stream", 0, "cold", "p1", 8.0, 90),
        _row("base", 0, "cold", None, 20.0, 200),
        _row("stream", 0, "cold", None, 1.0, 10),
    ]
    comparison = summarize_e4.compare_arms(rows, "stream")
    assert set(comparison["cold"]["first_request_total"]["by_cpu_part"]) == {"p1"}
    assert comparison["unknown_cpu_part_rows"] == {"cold": {"base": 1, "stream": 1}}
    cond = {
        "arms": ["base", "stream"],
        "functions": {
            a: {"MemorySize": 10240, "Architectures": ["arm64"]} for a in ("base", "stream")
        },
    }
    observed = summarize_e4.host_variability(str(tmp_path), cond, rows)["observed"]
    assert observed["unknown_cpu_part_rows_by_class_and_arm"] == {"cold": {"base": 1, "stream": 1}}
    assert "None" not in str(observed["cpu_parts_by_class_and_arm"])


def test_host_variability_names_sequential_alternating_arm_waves(tmp_path):
    import gzip

    import summarize_e4

    (tmp_path / "logs").mkdir()
    with gzip.open(tmp_path / "logs" / "fn.jsonl.gz", "wt"):
        pass
    rows = [
        _row("base", 0, "cold", "p1", 10.0, 100),
        _row("stream", 0, "cold", "p1", 8.0, 90),
    ]
    cond = {
        "arms": ["base", "stream"],
        "functions": {
            a: {"MemorySize": 10240, "Architectures": ["arm64"]} for a in ("base", "stream")
        },
    }
    method = summarize_e4.host_variability(str(tmp_path), cond, rows)["method"]
    assert "sequentially in alternating order" in method
    assert "interleaved" not in method


def test_published_e4_texts_claim_no_graviton_generation():
    """The initiative's host rule: the runs recorded CPU part numbers only, so no published E4 text names a
    generation (the README's note that none is claimed is the one allowed mention)."""
    import json
    import re

    for name in ("meta.json", "meta-r1.json"):
        m = json.load(open(os.path.join(E4, name)))
        assert not re.search(r"Graviton\d", json.dumps(m)), name
        assert m["host_assessment"].startswith("Comparable; no rerun.")
    assert not re.search(r"Graviton\d", open(os.path.join(E4, "README.md")).read())
