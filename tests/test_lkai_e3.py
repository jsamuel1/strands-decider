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
import json
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


@pytest.fixture(autouse=True)
def _no_live_recorder(monkeypatch):
    """e3dl.counters() writes into lkai_entry.REC when lkai_entry is imported. An entry-point test (here or in
    test_lkai_baseline) can leave one in sys.modules; the unit tests below read e3dl._SCRATCH, so drop it."""
    monkeypatch.delitem(sys.modules, "lkai_entry", raising=False)
    e3dl._SCRATCH.clear()


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


# ---------------------------------------------------------------------------- summarize_e3


def _row(arm, rnd, wave, cpu, dl=None, lat=0.25, init_extra=None, counters=None, err=None):
    rec = {
        "arm": arm,
        "round": rnd,
        "wave": wave,
        "seq": 0,
        "request_id": "ticket-route-1q",
        "qualifier": "$LATEST",
        "rt_s": 30.0,
        "status": 200,
        "error": err,
        "report": {"Duration": 1000.0, "Billed Duration": 1000.0, "Max Memory Used": 5100.0}
        | ({"Init Duration": 800.0} if wave == "cold" else {}),
        "expected_class": wave,
        "observed_class": wave,
        "class_mismatch": False,
        "cpu_part": cpu,
        "answers": {"q": {"type": "noul", "noul": 0.25}},
        "latency_ms": lat * 1000,
    }
    inv = {"n": 1 if wave == "cold" else 2, "handler_s": 1.0, "wait_load_s": 0.0}
    rec["lkai"] = {"inv": inv}
    if wave == "cold":
        rec["lkai"]["init"] = {"weight_download": dl, "model_load_mmap": 8.0, "model_ready": 34.0}
        rec["lkai"]["init"] |= init_extra or {}
        rec["lkai"]["counters"] = {"weight_mib_s": round(1980 / dl, 1)} | (counters or {})
    return rec


def _write_run(d, rows, arms, e3_dl):
    os.makedirs(d, exist_ok=True)
    fn = {
        a: {
            "FunctionName": f"lkai-e3-{a}",
            "Runtime": "python3.12",
            "Architectures": ["arm64"],
            "MemorySize": 10240,
            "Timeout": 900,
            "EphemeralStorage": {"Size": 512},
            "SnapStart": {"ApplyOn": "PublishedVersions"},
            "Handler": "e3_entry.handler",
            "environment": {"E3_DL": e3_dl.get(a) or ""},
        }
        for a in arms
    }
    cond = {
        "region": "us-east-2",
        "rounds": 2,
        "concurrency": 2,
        "warm_per_arm_per_round": 1,
        "arms": arms,
        "e3_dl": e3_dl,
        "functions": fn,
        "requests": {"requests": [{"id": "ticket-route-1q"}]},
        "probe": "ticket-route-1q",
        "requests_file": "baseline/requests.json",
        "started": "2026-10-20T00:00:00Z",
        "finished": "2026-10-20T01:00:00Z",
        "client": {},
        "p1_zip_sha256": "d" * 64,
        "runtime_version_arn": "arn:aws:lambda:us-east-2::runtime:x",
        "build_manifest": {
            "repo_commit": "c" * 40,
            "repo_dirty": False,
            "model": {"gguf_sha256": "8" * 64, "gguf_bytes": 2076712416},
            "llama_cpp": {},
            "images": {},
            "toolchain": [],
            "assets_sha256": {},
            "function_zip": {"sha256": "e" * 64},
            "awscrt_wheel": {},
            "e3_sources_sha256": {},
        },
        "rounds_meta": [],
    }
    json.dump(cond, open(os.path.join(d, "conditions.json"), "w"))
    with open(os.path.join(d, "invocations.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _summarize(tmp_path, rows, verify_rows, arms, e3_dl):
    import summarize_e3

    cold, ver, out = tmp_path / "cold", tmp_path / "verify", tmp_path / "out"
    _write_run(str(cold), rows, arms, e3_dl)
    _write_run(str(ver), verify_rows, arms, e3_dl)
    ref = tmp_path / "ref.json"
    ref.write_text(
        json.dumps(
            {"outputs": {"ticket-route-1q": {"0xd40": {"q": {"type": "noul", "noul": 0.25}}}}}
        )
    )
    out.mkdir()
    argv = [
        "summarize_e3.py",
        "--cold",
        str(cold),
        "--verify",
        str(ver),
        "--reference",
        str(ref),
        "--baseline-commit",
        "7f066aa",
        "--run-id",
        "t",
        "--out",
        str(out),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        summarize_e3.main()
    finally:
        sys.argv = old
    return {a: json.load(open(out / a / "results.json")) for a in arms if a != "ctl"}, out


def test_summarize_status_per_setting_and_isolation(tmp_path):
    arms = ["ctl", "fast", "same"]
    e3_dl = {"ctl": None, "fast": "threads:16:64", "same": "threads:64:32"}
    rows, ver = [], []
    for rnd in range(2):
        for i in range(10):
            j = 0.01 * i
            rows.append(_row("ctl", rnd, "cold", "0xd40", dl=25.0 + j))
            rows.append(
                _row("fast", rnd, "cold", "0xd40", dl=12.0 + j, counters={"e3_get_requests": 124})
            )
            rows.append(
                _row("same", rnd, "cold", "0xd40", dl=25.0 + j, counters={"e3_get_requests": 31})
            )
            for a in arms:
                rows.append(_row(a, rnd, "warm", "0xd40"))
    good = {"e3_weight_sha256": "8" * 64}
    ver += [_row("fast", 0, "cold", "0xd40", dl=12, counters=good)]
    ver += [_row("same", 0, "cold", "0xd40", dl=25, counters=good)]
    res, out = _summarize(tmp_path, rows, ver, arms, e3_dl)
    assert res["fast"]["status"] == "improved"
    assert res["fast"]["e3"]["download"]["vs_ctl"]["ci95_s"][1] < 0
    assert res["fast"]["e3"]["download"]["s3_get_requests_per_download"] == [124]
    assert res["fast"]["cost"]["s3_get_requests_per_download"] == 124
    assert res["same"]["status"] == "no-change"
    assert res["fast"]["e3"]["hash"]["match"] is True
    assert set(res["fast"]["phases"]) == {"ctl", "fast"}  # only this setting and the control
    for a in (
        "ctl",
        "fast",
    ):  # no restore leg: P2's required restore phases are explicit nulls in both arms
        rs = res["fast"]["phases"][a]["restore"]
        assert rs["n"] == 0
        for k in ("report_restore_duration", "first_request_inference", "snapshot_init"):
            assert k in rs and rs[k] is None
    arms_in_rows = {json.loads(x)["arm"] for x in open(out / "fast" / "invocations.jsonl")}
    assert arms_in_rows == {"ctl", "fast"}
    assert res["fast"]["e3"]["compute"]["cold.model_load_mmap"]["pooled_delta_p50_s"] == 0
    assert "Status improved" in res["fast"]["conclusion"]
    assert (out / "summary.md").read_text().count("\n") == 4
    chk = res["fast"]["e3"]["host"]["status_check"]
    assert chk == chk | {
        "largest_cpu_part": "0xd40",
        "status_within_part": "improved",
        "holds": True,
    }


def test_summarize_host_check_flags_a_cpu_mix_confound(tmp_path):
    """An arm that only looks slower because half its colds landed on a CPU part the control never saw: the pooled
    status says regressed, the within-part check says no-change, and `holds` is False. Unrecorded host fields are
    listed as unknown and the ggml variant comes from the init log line."""
    arms = ["ctl", "mix"]
    e3_dl = {"ctl": None, "mix": "threads:64:64"}
    log = "decider.engine init cpu_part=0xd40 ggml_cpu=libggml-cpu-armv8.6_1.so threads=6\n"
    rows = []
    for i in range(20):
        j = 0.01 * i
        rows.append(_row("ctl", i % 2, "cold", "0xd40", dl=25.0 + j) | {"log_tail": log})
        rows.append(_row("mix", i % 2, "cold", "0xd40", dl=25.0 + j) | {"log_tail": log})
        rows.append(_row("mix", i % 2, "cold", "0xd0c", dl=30.0 + j))
    rows += [_row(a, 0, "warm", "0xd40") for a in arms]
    ver = [_row("mix", 0, "cold", "0xd40", dl=25, counters={"e3_weight_sha256": "8" * 64})]
    res, out = _summarize(tmp_path, rows, ver, arms, e3_dl)
    r = res["mix"]
    assert r["status"] == "regressed"
    host = r["e3"]["host"]
    assert set(host["download_by_cpu_part"]) == {"0xd40"}  # ctl never ran on 0xd0c
    assert host["status_check"]["status_within_part"] == "no-change"
    assert host["status_check"]["holds"] is False
    assert "changes this result" in r["conclusion"]
    rec = host["recorded"]
    assert rec["ggml_cpu_variant"]["mix:cold:libggml-cpu-armv8.6_1.so"] == 20
    assert rec["ggml_cpu_variant"]["mix:cold:unknown"] == 20
    assert rec["cpu_part"]["mix:cold:0xd0c"] == 20
    assert "CPU generation / instance family" in host["unknown"]
    assert "no-change (0xd40, n=20)" in (out / "summary.md").read_text()


def test_summarize_hash_mismatch_is_never_improved(tmp_path):
    arms = ["ctl", "fast"]
    e3_dl = {"ctl": None, "fast": "crt:8:25"}
    rows = [
        _row("ctl", r, "cold", "0xd40", dl=25.0 + 0.01 * i) for r in range(2) for i in range(10)
    ]
    rows += [
        _row("fast", r, "cold", "0xd40", dl=12.0 + 0.01 * i) for r in range(2) for i in range(10)
    ]
    rows += [_row(a, 0, "warm", "0xd40") for a in arms]
    ver = [_row("fast", 0, "cold", "0xd40", dl=12, counters={"e3_weight_sha256": "0" * 64})]
    res, _ = _summarize(tmp_path, rows, ver, arms, e3_dl)
    assert res["fast"]["e3"]["hash"]["match"] is False
    assert res["fast"]["status"] == "inconclusive"


def test_summarize_arm_without_rows_is_failed(tmp_path):
    arms = ["ctl", "dead"]
    e3_dl = {"ctl": None, "dead": "threads:8:128"}
    rows = [_row("ctl", 0, "cold", "0xd40", dl=25.0 + 0.01 * i) for i in range(5)]
    rows += [_row("dead", 0, "cold", "0xd40", dl=1.0, err="Unhandled")]
    res, _ = _summarize(tmp_path, rows, [], arms, e3_dl)
    assert res["dead"]["status"] == "failed"
    assert res["dead"]["errors"]["count"] == 1
    assert res["dead"]["e3"]["host"]["status_check"] is None
