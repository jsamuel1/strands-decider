"""examples/lambda-kleidiai/profiling: the P3 probe, native sampler, symboliser and profiler-run summary.

No AWS and no llama.cpp: the sampler is compiled from source with the system C compiler (skipped without one), the
summary is computed from synthetic bench rows and log events, and the probe runs against the test host.
"""

from __future__ import annotations

import ctypes
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import types
from concurrent.futures import Future
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "examples" / "lambda-kleidiai"
PROF = ROOT / "profiling"
sys.path.insert(0, str(PROF))
sys.path.insert(0, str(ROOT / "baseline"))

import lkai_probe  # noqa: E402
import lkai_symbolise as ls  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler")


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    """bench.py / deploy_arms.py import boto3 at module level; stand-ins when the SDK is absent."""
    import importlib.util

    if importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore"):
        return
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    botocore = types.ModuleType("botocore")
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    for name, mod in (("boto3", boto3), ("botocore", botocore), ("botocore.exceptions", exc), ("botocore.config", cfg)):
        monkeypatch.setitem(sys.modules, name, mod)


# ------------------------------------------------------------------------------------------------ sampler + symbols


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("samp")
    samp = d / "liblkaisamp.so"
    subprocess.run([CC, "-O2", "-shared", "-fPIC", "-o", str(samp), str(PROF / "lkaisamp.c")], check=True)
    src = d / "hot.c"
    src.write_text("double lkai_test_hot(long n){volatile double s=0;for(long i=0;i<n;i++)s+=i*0.5;return s;}\n")
    hot = d / "libhot.so"
    subprocess.run([CC, "-O1", "-shared", "-fPIC", "-o", str(hot), str(src)], check=True)
    return samp, hot


@needs_cc
def test_sampler_attributes_cpu_to_the_hot_function_and_phase(built):
    samp_path, hot_path = built
    samp, hot = ctypes.CDLL(str(samp_path)), ctypes.CDLL(str(hot_path))
    hot.lkai_test_hot.restype = ctypes.c_double
    hot.lkai_test_hot.argtypes = [ctypes.c_long]
    assert samp.samp_start(1000, 100000) == 0
    samp.samp_phase(5)
    threads = [threading.Thread(target=hot.lkai_test_hot, args=(150_000_000,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    samp.samp_phase(0)
    samp.samp_stop()
    res = ls.summary(samp, {5: "hot"})
    samp.samp_reset()
    assert res["dropped"] == 0 and res["n"] > 20
    hot_tab = res["phases"]["hot"]
    assert hot_tab["samples"] >= 0.8 * res["n"]
    top = hot_tab["top_self"][0]
    assert top["symbol"] == "libhot.so:lkai_test_hot", hot_tab["top_self"][:3]
    assert top["pct"] > 70


@needs_cc
def test_sampler_dump_format_and_overflow_counts_drops(built, tmp_path):
    samp_path, hot_path = built
    # A fresh process: the buffer capacity is fixed at the first samp_start.
    code = f"""
import ctypes, os, sys
s = ctypes.CDLL({str(samp_path)!r}); h = ctypes.CDLL({str(hot_path)!r})
h.lkai_test_hot.argtypes = [ctypes.c_long]
assert s.samp_start(1000, 5) == 0
h.lkai_test_hot(150_000_000)
s.samp_stop()
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
assert s.samp_dump(fd) == 0
os.close(fd)
"""
    out = tmp_path / "s.bin"
    subprocess.run([sys.executable, "-c", code, str(out)], check=True)
    hdr, recs = ls.read_samples(str(out))
    assert hdr["n"] == 5 == len(recs) and hdr["dropped"] > 0 and hdr["period_us"] == 1000


def test_read_samples_rejects_foreign_files(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"NOTLKSMP" + b"\0" * 32)
    with pytest.raises(ValueError):
        ls.read_samples(str(p))
    p.write_bytes(b"LKSAMP01" + struct.pack("<QQQQ", 1000, 0, 0, 99))
    with pytest.raises(ValueError, match="record size"):
        ls.read_samples(str(p))


@needs_cc
def test_symbolizer_resolves_and_remaps_paths(built, tmp_path):
    _, hot_path = built
    lib = ctypes.CDLL(str(hot_path))
    addr = ctypes.cast(lib.lkai_test_hot, ctypes.c_void_p).value
    maps = open("/proc/self/maps").read()
    assert ls.Symbolizer(maps)(addr) == "libhot.so:lkai_test_hot"
    # The same maps, as if captured in a container where the library lived under /pkg/.
    moved = maps.replace(str(hot_path.parent) + "/", "/pkg/")
    assert ls.Symbolizer(moved)(addr).startswith("libhot.so:+0x")  # /pkg/ is not on this host
    assert ls.Symbolizer(moved, {"/pkg/": str(hot_path.parent) + "/"})(addr) == "libhot.so:lkai_test_hot"
    assert ls.Symbolizer("")(addr) == "?"


def test_tables_rank_self_samples_and_caller_edges():
    sym = {1: "a.so:f", 2: "a.so:g", 3: "b.so:h"}.get
    recs = [(1, 3, 10, 6)] * 6 + [(2, 3, 11, 6)] * 3 + [(3, 0, 10, 6)]
    t = ls.tables(recs, sym, {6: "steady"})["steady"]
    assert t["samples"] == 10 and t["threads"] == 2
    assert [x["symbol"] for x in t["top_self"]] == ["a.so:f", "a.so:g", "b.so:h"]
    assert t["by_library"][0] == {"library": "a.so", "samples": 9, "pct": 90.0}
    assert t["top_caller_edges"][0] == {"edge": "b.so:h -> a.so:f", "samples": 6}


# ------------------------------------------------------------------------------------------------ probe


def test_probe_reports_facts_without_claiming_tools():
    r = lkai_probe.probe(py_spy_path="")
    assert r["schema"] == "lkai-probe/1"
    assert r["py_spy"] == {"result": "not-shipped"}
    assert isinstance(r["gpu"]["dev_nvidia"], list)
    assert r["gpu"]["nvidia_vendor_0x10de_present"] == ("0x10de" in r["gpu"]["pci_vendors"])
    # Every syscall attempt records an outcome (and an errno name when refused), never an exception.
    for ev in r["syscalls"]["perf_event_open"]:
        assert ev["result"] in ("opened", "denied", "opened-but-read-failed", "unsupported-arch")
        if ev["result"] == "denied":
            assert ev["errno"].isupper()
    assert r["syscalls"]["itimer_prof_sigprof"]["result"] in ("delivered", "none")
    assert r["syscalls"]["process_vm_readv_self"]["result"] in ("allowed", "denied", "unsupported-arch")
    # No environment value beyond the allow-list leaks.
    assert set(r["env_values"]) <= set(lkai_probe.ENV_NAMES)


def test_probe_syscall_args_are_widened_to_c_long(monkeypatch):
    seen = []

    class FakeLibc:
        def syscall(self, *a):
            seen.append(a)
            return -1

    monkeypatch.setattr(lkai_probe, "_libc", FakeLibc())
    monkeypatch.setitem(lkai_probe._NR, "bpf", 280)
    lkai_probe._sys("bpf", 0, 7)
    assert all(isinstance(x, ctypes.c_long) for x in seen[0])  # the nr, 0 and 7: none passed as a 32-bit int


# ------------------------------------------------------------------------------------------------ summary


def _row(arm, rnd, cls, *, report=None, lat=500.0, cpu="0xd40", rid="r1", answers=None):
    rep = report if report is not None else {"Duration": lat + 10, "Billed Duration": lat + 11, "Max Memory Used": 5000}
    return {"arm": arm, "round": rnd, "wave": cls, "expected_class": cls, "observed_class": cls,
            "class_mismatch": False, "error": None, "report": rep, "latency_ms": lat, "cpu_part": cpu,
            "request_id": rid, "rt_s": lat / 1000 + 0.05, "answers": answers or {"q": {"choice": "x"}}}


def test_overhead_vs_compares_one_arm_to_the_control_by_cpu_part():
    import summarize_prof as sp

    rows = [_row("rep", 0, "warm", lat=460.0), _row("pyspy", 0, "warm", lat=500.0),
            _row("pyspy", 0, "warm", lat=900.0, cpu="0xd0c"), _row("sampler", 0, "warm", lat=470.0)]
    ov = sp.overhead_vs(rows, "pyspy")
    cell = ov["warm"]["steady_state_inference"]
    assert list(cell["by_cpu_part"]) == ["0xd40"]  # part 0xd0c has no control sample
    assert cell["pooled_delta_p50_s"] == pytest.approx(0.04)
    assert ov["method"].startswith("median(pyspy) - median(rep)")
    assert sp.overhead_vs(rows, "sampler")["warm"]["steady_state_inference"]["pooled_delta_p50_s"] == pytest.approx(0.01)


def _logs(tmp_path, fn, messages):
    d = tmp_path / "logs"
    d.mkdir(exist_ok=True)
    with gzip.open(d / f"{fn}.jsonl.gz", "wt") as f:
        for i, m in enumerate(messages):
            f.write(json.dumps({"timestamp": i, "logStreamName": "2026/10/20/[$LATEST]x", "message": m}) + "\n")


def test_profile_extracts_merge_records_per_profiler(tmp_path):
    import summarize_prof as sp

    def prof(kind, **kw):
        return "LKAI_PROF " + json.dumps({"schema": "lkai-prof/1", "kind": kind, **kw})

    samp = {"n": 10, "dropped": 0, "phases": {"invoke": {
        "samples": 10, "top_self": [{"symbol": "libggml-cpu.so:kai_kernel", "samples": 7}],
        "by_library": [{"library": "libggml-cpu.so", "samples": 9}], "top_caller_edges": []}}}
    _logs(tmp_path, "lkai-p3-sampler", [prof("invoke", sampler=samp, cpu_s=0.5), prof("invoke", sampler=samp, cpu_s=0.5)])
    spy = {"rc": 0, "popen_s": 0.001, "samples": 5, "top_leaf": [{"frame": "_eval (engine.py:138)", "samples": 5}],
           "top_inclusive": []}
    _logs(tmp_path, "lkai-p3-pyspy", [prof("init", pyspy=spy), "noise", prof("invoke", pyspy=spy)])
    tel = [{"type": "platform.initReport", "record": {"metrics": {"durationMs": 800.0}}},
           {"type": "platform.runtimeDone", "record": {"spans": [{"name": "responseLatency", "durationMs": 1.5}]}}]
    _logs(tmp_path, "lkai-p3-telemetry", ["LKAI_TEL " + json.dumps(t) for t in tel])
    cond = {"functions": {"sampler": {"FunctionName": "lkai-p3-sampler"}, "pyspy": {"FunctionName": "lkai-p3-pyspy"},
                          "telemetry": {"FunctionName": "lkai-p3-telemetry"}}}
    out = sp.profile_extracts(str(tmp_path), cond)
    s = out["sampler"]["sampler_invoke"]
    assert s["samples"] == 20 and s["cpu_s"] == 1.0 and s["effective_hz_per_cpu_s"] == 20.0
    # 14 of the 20 recorded samples: the share is of all samples, not of the retained top rows (which hold 14)
    assert s["phases"]["invoke"]["top_self"][0] == {"symbol": "libggml-cpu.so:kai_kernel", "samples": 14, "share": 0.7}
    assert out["pyspy"]["pyspy_init"]["samples"] == 5 and out["pyspy"]["pyspy_invoke"]["records"] == 1
    t = out["telemetry"]["telemetry"]
    assert t["durations_ms"]["platform.initReport.durationMs"]["p50"] == 800.0
    assert t["spans_ms"]["invoke.responseLatency"]["p50"] == 1.5


def test_init_phase_shares_rank_cold_and_restore():
    import summarize_prof as sp

    d = lambda v: {"p50": v}  # noqa: E731
    rows = sp.init_phase_shares({"cold": {"first_request_total": d(36.0), "weight_download": d(25.0),
                                          "model_load_mmap": d(8.0)},
                                 "restore": {"first_request_total": d(4.0), "report_restore_duration": d(3.2)}})
    assert [r["phase"] for r in rows] == ["weight_download", "model_load_mmap", "report_restore_duration"]
    assert rows[0]["share_of_first_request"] == pytest.approx(25 / 36, abs=1e-4)


def test_bench_accepts_a_per_function_expected_code_sha(tmp_path, monkeypatch):
    import bench

    class FakeLambda:
        def get_function_configuration(self, FunctionName):
            return {"FunctionName": FunctionName, "Runtime": "python3.12", "Architectures": ["arm64"],
                    "MemorySize": 10240, "Timeout": 900, "EphemeralStorage": {"Size": 512},
                    "SnapStart": {"ApplyOn": "PublishedVersions"}, "Handler": "h", "CodeSha256": FunctionName,
                    "CodeSize": 1, "Environment": {"Variables": {}}, "TracingConfig": {"Mode": "Active"},
                    "Layers": [{"Arn": "arn:aws:lambda:us-east-2:000000000000:layer:lkai-p3-telemetry:3"}]}

        def get_runtime_management_config(self, FunctionName):
            return {}

    class FakeBench:
        def __init__(self, *a, **k):
            self.lam = FakeLambda()

    state = {"region": "us-east-2", "manifest": {"function_zip": {"sha256": "p1zip"}},
             "functions": {"rep": {"name": "p1zip"}, "pyspy": {"name": "profzip", "code_sha256_hex": "profzip"}}}
    monkeypatch.setattr(bench.deploy, "load_state", lambda b: state)
    monkeypatch.setattr(bench.deploy, "b64_to_hex", lambda v: v)  # CodeSha256 is the function name here
    monkeypatch.setattr(bench.boto3, "Session", lambda **k: object())
    monkeypatch.setattr(bench, "Bench", FakeBench)
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(tmp_path), "--rounds", "0", "--arms", "rep,pyspy"])
    bench.main()
    cond = json.load(open(tmp_path / "conditions.json"))
    assert cond["functions"]["pyspy"]["expected_code_sha256"] == "profzip"
    assert cond["functions"]["rep"]["expected_code_sha256"] == "p1zip"
    assert cond["functions"]["rep"]["layers"] == [["lkai-p3-telemetry", "3"]]  # name and version, no account id
    state["functions"]["pyspy"]["code_sha256_hex"] = "other"
    with pytest.raises(SystemExit, match="not the expected zip"):
        bench.main()


def test_deploy_arms_teardown_never_touches_other_prefixes(monkeypatch):
    import deploy_arms

    deleted = []

    class Pag:
        def paginate(self, FunctionName):
            return [{"Versions": [{"Version": "$LATEST"}, {"Version": "4"}]}]

    class FakeLambda:
        def get_paginator(self, _):
            return Pag()

        def delete_function(self, FunctionName, Qualifier=None):
            deleted.append((FunctionName, Qualifier))

        class exceptions:
            ResourceNotFoundException = KeyError

    sess = types.SimpleNamespace(client=lambda name: FakeLambda())
    st = {"functions": {"rep": {"name": "lkai-p3-rep"}, "x": {"name": "lkai-p1-rep"}}}
    deploy_arms.teardown(sess, st, everything=False)
    assert deleted == [("lkai-p3-rep", "4")]


def test_telemetry_extension_is_executable_and_stdlib_only():
    p = PROF / "extensions" / "lkai-telemetry"
    assert os.access(p, os.X_OK)
    src = p.read_text()
    assert src.startswith("#!/usr/bin/env python3")
    imports = {ln.split()[1].split(".")[0] for ln in src.splitlines() if ln.startswith(("import ", "from "))}
    assert imports <= {"http", "json", "os", "sys", "threading", "time", "urllib"}


# ------------------------------------------------------------------------------------------------ lkai_prof handler


def _load_prof(monkeypatch, tmp_path, mode, entry_handler, pyspy_script=None):
    """Import lkai_prof in `mode` against a stand-in lkai_entry; bin/py-spy is a fake under tmp_path."""
    import importlib

    entry = types.ModuleType("lkai_entry")
    entry.handler = entry_handler
    monkeypatch.setitem(sys.modules, "lkai_entry", entry)
    pr77 = types.ModuleType("handler")  # PR #77's handler: its background load, already finished
    done = Future()
    done.set_result(None)
    pr77._ENGINE = done
    monkeypatch.setitem(sys.modules, "handler", pr77)
    monkeypatch.setenv("LKAI_PROF", mode)
    monkeypatch.delitem(sys.modules, "lkai_prof", raising=False)
    if pyspy_script:
        (tmp_path / "bin").mkdir()
        spy = tmp_path / "bin" / "py-spy"
        spy.write_text(pyspy_script)
        spy.chmod(0o755)
    real = os.path.abspath
    monkeypatch.setattr(
        os.path, "abspath", lambda p: str(tmp_path / "lkai_prof.py") if str(p).endswith("lkai_prof.py") else real(p)
    )
    mod = importlib.import_module("lkai_prof")
    monkeypatch.setattr(os.path, "abspath", real)
    return mod


# Like py-spy cut by SIGINT mid-write: the raw output ends inside a UTF-8 sequence.
_BAD_SPY = """#!/bin/sh
while [ "$1" != "-o" ]; do shift; done
printf 'a (x.py:1);b (y.py:2) 3\\n\\376\\375' > "$2"
exec sleep 30
"""


def _records(capsys):
    return [json.loads(ln.split(" ", 1)[1]) for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("LKAI_PROF ")]


def test_truncated_pyspy_output_does_not_fail_the_request(monkeypatch, tmp_path, capsys):
    import time

    def entry(e, c):
        time.sleep(0.3)  # let the fake py-spy write before it is interrupted, as a real forward would
        return {"answers": {"q": 1}}

    mod = _load_prof(monkeypatch, tmp_path, "pyspy", entry, _BAD_SPY)
    time.sleep(0.3)
    assert mod.handler({}, None)["answers"] == {"q": 1}
    recs = _records(capsys)
    assert "profiler_error" not in [r["kind"] for r in recs]  # undecodable bytes are replaced, the rest parses
    assert next(r for r in recs if r["kind"] == "init")["pyspy"]["samples"] == 3
    assert next(r for r in recs if r["kind"] == "invoke")["pyspy"]["samples"] == 3


def test_profiler_exception_is_logged_and_the_handler_outcome_kept(monkeypatch, tmp_path, capsys):
    def boom(e, c):
        raise ValueError("engine failed")

    mod = _load_prof(monkeypatch, tmp_path, "none", boom)

    def bad_hook():
        raise UnicodeDecodeError("utf-8", b"\xfe", 0, 1, "invalid start byte")

    monkeypatch.setattr(mod, "_init_profile_emit", bad_hook)
    with pytest.raises(ValueError, match="engine failed"):  # the handler's own error, not the profiler's
        mod.handler({}, None)
    errs = [r for r in _records(capsys) if r["kind"] == "profiler_error"]
    assert len(errs) == 1 and errs[0]["where"] == "init" and errs[0]["error"].startswith("UnicodeDecodeError")


# ------------------------------------------------------------------------------------------------ paired overhead


def _prow(arm, rnd, cls, dur, *, rid="r1", cpu="0xd40", init=None, error=None, mismatch=False):
    rep = {"Duration": dur}
    if init is not None:
        rep["Init Duration"] = init
    return {"arm": arm, "round": rnd, "expected_class": cls, "error": error, "class_mismatch": mismatch,
            "cpu_part": cpu, "request_id": rid, "report": rep}


def test_paired_rounds_cancel_round_drift():
    import summarize_prof as sp

    rows = []
    for rnd, drift in enumerate((0, 100, -50)):  # the whole round is slower or faster: both arms move together
        rows += [_prow("rep", rnd, "warm", 700 + drift + i) for i in (0, 5, 10)]
        rows += [_prow("x", rnd, "warm", 720 + drift + i) for i in (0, 5, 10)]
    rows += [_prow("x", 0, "warm", 9999, cpu="0xd0c"), _prow("x", 1, "warm", 9999, error="boom"),
             _prow("x", 2, "warm", 9999, mismatch=True)]  # all excluded
    p = sp.paired_rounds(rows, "x")["warm.Duration"]
    assert p["rounds"] == 3 and p["delta_ms_by_round"] == [20.0, 20.0, 20.0] and p["skipped_rounds"] == []
    assert p["mean_ms"] == 20.0 and p["min_ms"] == p["max_ms"] == 20.0
    assert sp.noise_floor({"rep2": {"warm.Duration": p, "warm.Duration.by_request": {"cells": 1}}}) == {
        "warm.Duration": 20.0}


def test_paired_warm_by_request_removes_request_mix():
    import summarize_prof as sp

    # The arm got more of the long request; pooled medians would call it 3x slower. Per request it is +10%.
    rows = [_prow("rep", 0, "warm", 250, rid="short"), _prow("rep", 0, "warm", 1750, rid="long"),
            _prow("x", 0, "warm", 275, rid="short"), *[_prow("x", 0, "warm", 1925, rid="long")] * 5]
    w = sp.paired_warm_by_request(rows, "x")
    assert w == {"cells": 2, "median_pct": 10.0, "min_pct": 10.0, "max_pct": 10.0, "pct_by_cell": [10.0, 10.0],
                 "round_ids": [0], "pct_by_round": [10.0]}


def test_profile_extracts_split_warm_sampler_tables_by_cpu_part(tmp_path):
    import summarize_prof as sp

    def phase(n, cls, cpu):
        return "LKAI " + json.dumps({"schema": "lkai-phases/1", "kind": "invoke", "inv": {"n": n, "class": cls,
                                                                                       "cpu_part": cpu}})

    def samp(n, sym):
        t = {"samples": 4, "top_self": [{"symbol": sym, "samples": 4}], "by_library": [], "top_caller_edges": []}
        return "LKAI_PROF " + json.dumps({"kind": "invoke", "n": n, "cpu_s": 0.1,
                                          "sampler": {"n": 4, "dropped": 0, "phases": {"invoke": t}}})

    d = tmp_path / "logs"
    d.mkdir()
    ev = [("s1", phase(1, "cold", "0xd40")), ("s1", samp(1, "cold_sym")), ("s1", phase(2, "warm", "0xd40")),
          ("s1", samp(2, "i8mm")), ("s2", phase(1, "warm", "0xd0c")), ("s2", samp(1, "dotprod"))]
    with gzip.open(d / "lkai-p3-sampler.jsonl.gz", "wt") as f:
        for i, (s, m) in enumerate(ev):
            f.write(json.dumps({"timestamp": i, "logStreamName": s, "message": m}) + "\n")
    out = sp.profile_extracts(str(tmp_path), {"functions": {"sampler": {"FunctionName": "lkai-p3-sampler"}}})["sampler"]
    assert out["sampler_invoke"]["records"] == 3
    top = {k: out[k]["phases"]["invoke"]["top_self"][0]["symbol"] for k in out if k.startswith("sampler_invoke_warm_")}
    assert top == {"sampler_invoke_warm_0xd40": "i8mm", "sampler_invoke_warm_0xd0c": "dotprod"}


# ------------------------------------------------------------------------------------------------ review-2 regressions


def test_bench_makes_a_fresh_invoke_client_for_every_wave(monkeypatch):
    """A pooled connection must not outlive a wave (it can die during the publish/SnapStart wait)."""
    import bench

    made = []
    monkeypatch.setattr(bench, "invoke_client", lambda sess, c, t: made.append((c, t)) or f"client-{len(made)}")
    b = bench.Bench.__new__(bench.Bench)
    b.sess, b.concurrency, b.read_timeout, b.clients_made, b.probe = object(), 3, 120, 0, "r1"
    b.lam = "lifecycle"
    seen = []
    b.invoke = lambda arm, q, rid, rnd, wave, seq, client: seen.append(client) or {}
    b.wave("rep", "1", 0, "restore")
    b.wave("rep", "1", 0, "warm")
    assert made == [(3, 120), (3, 120)] and b.clients_made == 2
    assert seen == ["client-1"] * 3 + ["client-2"] * 3  # one client per wave, never the lifecycle client


def test_merge_top_shares_use_the_full_denominator_not_retained_rows():
    """Run 20261020T124451Z: init kept 83,864 of 106,648 samples in its top-N; shares over the retained rows read
    35.2% where the truth was 30.1%."""
    import summarize_prof as sp

    # Two records, each truncated to its top rows: 60 of 100 and 30 of 50 samples retained.
    lists = [[{"s": "a", "n": 40}, {"s": "b", "n": 20}], [{"s": "a", "n": 20}, {"s": "c", "n": 10}]]
    out = sp._merge_top(lists, "s", "n", [100, 50])
    assert out[0] == {"s": "a", "n": 60, "share": 0.4}  # 60/150, not 60/90
    assert [x["share"] for x in out] == [0.4, 0.1333, 0.0667]
    assert all(x["share"] is None for x in sp._merge_top(lists, "s", "n", [100, None]))  # no guessed denominator
    with pytest.raises(ValueError, match="exceed"):
        sp._merge_top(lists, "s", "n", [10, 10])  # exclusive rows can never sum past their total
    assert sp._merge_top(lists, "s", "n", [10, 10], inclusive=True)[0]["share"] == 3.0  # inclusive may


def test_profile_extracts_cprofile_and_pyspy_shares_are_of_all_time(tmp_path):
    import summarize_prof as sp

    cp_init = {"kind": "init", "tottime_total": 10.0,
               "top": {"all_threads": [{"fn": "ssl:read", "tottime": 4.0, "cumtime": 9.0}]}}
    cp_inv = {"kind": "invoke", "n": 1, "tottime_total": 2.0, "top": [{"fn": "engine:_fork", "tottime": 1.5,
                                                                         "cumtime": 1.9}]}
    _logs(tmp_path, "lkai-p3-cprofile", ["LKAI_PROF " + json.dumps(cp_init), "LKAI_PROF " + json.dumps(cp_inv)])
    spy = {"rc": 0, "popen_s": 0.0, "samples": 50, "top_leaf": [{"frame": "read", "samples": 20}],
           "top_inclusive": [{"frame": "main", "samples": 50}, {"frame": "read", "samples": 20}]}
    _logs(tmp_path, "lkai-p3-pyspy", ["LKAI_PROF " + json.dumps({"kind": "init", "pyspy": spy})])
    cond = {"functions": {"cprofile": {"FunctionName": "lkai-p3-cprofile"}, "pyspy": {"FunctionName": "lkai-p3-pyspy"}}}
    out = sp.profile_extracts(str(tmp_path), cond)
    assert out["cprofile"]["cprofile_init_top_tottime"][0]["share"] == 0.4
    assert out["cprofile"]["cprofile_init_tottime_total"] == 10.0
    assert out["cprofile"]["cprofile_invoke_top_tottime"][0]["share"] == 0.75
    assert out["pyspy"]["pyspy_init"]["top_leaf"][0]["share"] == 0.4
    assert out["pyspy"]["pyspy_init"]["top_inclusive"][0]["share"] == 1.0


def test_paired_rounds_skip_a_round_whose_control_is_one_sample():
    """Run 20261020T124451Z: rep kept 1 valid restore sample in rounds 0/2/4 and was still compared."""
    import summarize_prof as sp

    rows = []
    for rnd in range(3):
        rows += [_prow("x", rnd, "restore", 3300 + i) for i in range(5)]
        rows += [_prow("rep", rnd, "restore", 3000 + i) for i in range(1 if rnd == 1 else 5)]
    rows += [_prow("rep", 1, "restore", 0, error="client:ReadTimeoutError") for _ in range(4)]
    p = sp.paired_rounds(rows, "x")["restore.Duration"]
    assert p["rounds"] == 2 and p["skipped_rounds"] == [{"round": 1, "n_arm": 5, "n_ctrl": 1}]
    ev = sp.restore_evidence(rows, ["rep", "x"])
    assert ev["usable"] is False and ev["sparse_cells"] == [{"arm": "rep", "round": 1}]
    assert ev["by_arm_round"]["rep"]["1"] == {"sent": 5, "client_errors": 4, "valid_0xd40": 1}


def test_sign_flip_p_is_exact_and_floored_by_pair_count():
    import summarize_prof as sp

    # 5 pairs, all one sign: only the all-plus and all-minus assignments are as extreme.
    assert sp.sign_flip_p([10, 11, 12, 13, 14]) == round(2 / 32, 4)
    assert sp.sign_flip_p([-3, -1, -2, -5, -4]) == 0.0625
    assert sp.sign_flip_p([1, -1]) == 1.0
    assert sp.sign_flip_p([1]) is None
    assert sp.sign_flip_p([1.0] * 6) == round(2 / 64, 4)  # 6 pairs could reach 0.03


def test_null_test_keeps_round_pairing_and_cannot_go_below_its_floor():
    """arm - rep and rep2 - rep share each round's rep median: a 5-vs-5 unpaired permutation gave p=0.008, which a
    paired design of 5 rounds cannot support. The paired statistic is arm - rep2 per round."""
    import summarize_prof as sp

    def cell(d):
        return {"delta_ms_by_round": d, "round_ids": list(range(len(d))), "min_ms": min(d), "max_ms": max(d)}

    null = [-20.0, 15.0, -5.0, 10.0, 0.0]
    paired = {"rep2": {"cold.Init Duration": cell(null)},
              "xray": {"cold.Init Duration": cell([-10.0, 12.0, 3.0, -8.0, 5.0])},
              "cprofile": {"cold.Init Duration": cell([100.0, 110.0, 99.0, 117.0, 107.0])}}
    v = sp.null_verdicts(paired)
    assert "rep2" not in v
    c = v["cprofile"]["cold.Init Duration"]
    assert c["p"] == 0.0625 and c["min_attainable_p"] == 0.0625 and c["underpowered"] is True
    assert c["verdict"] == "not measurable"  # p < 0.05 is unattainable with 5 pairs
    assert c["paired_diff_by_round"] == [120.0, 95.0, 104.0, 107.0, 107.0]  # arm - rep2, round by round
    assert c["consistent_direction"] is True and c["null_max_abs"] == 20.0
    assert v["xray"]["cold.Init Duration"]["consistent_direction"] is False
    # Mutation probe: the old unpaired 5-vs-5 test on the same values would have reported p = 2/252 < 0.05.
    assert 2 / 252 < 0.05 < c["p"]
    # Pairing is by round id, not position: a round only one arm kept is dropped from the pairs.
    gappy = {"rep2": {"k": {"delta_ms_by_round": [1.0, 2.0, 3.0], "round_ids": [0, 1, 2]}},
             "x": {"k": {"delta_ms_by_round": [9.0, 9.0], "round_ids": [0, 2]}}}
    g = sp.null_verdicts(gappy)["x"]["k"]
    assert g["rounds"] == [0, 2] and g["paired_diff_by_round"] == [8.0, 6.0] and g["min_attainable_p"] == 0.5
    assert sp.null_verdicts({"rep2": {}, "x": {"warm.Duration": cell([1.0])}})["x"]["warm.Duration"]["verdict"] == (
        "insufficient")


def test_six_pairs_can_be_measurable():
    import summarize_prof as sp

    def cell(d):
        return {"delta_ms_by_round": d, "round_ids": list(range(len(d)))}

    v = sp.null_verdicts({"rep2": {"k": cell([1.0, -2.0, 0.0, 2.0, -1.0, 1.0])},
                          "x": {"k": cell([50.0, 52.0, 49.0, 55.0, 51.0, 50.0])}})["x"]["k"]
    assert v["p"] == round(2 / 64, 4) and v["underpowered"] is False and v["verdict"] == "measurable"


def _prov_repo(tmp_path):
    """A git repo laid out like profiling/: the packaged sources and build_prof_zip.sh with its PYSPY_SHA256 pin."""
    import hashlib

    r = tmp_path / "repo"
    (r / "extensions").mkdir(parents=True)
    files = {"lkai_prof.py": b"p", "lkai_probe.py": b"q", "lkai_symbolise.py": b"s", "lkaisamp.c": b"int x;",
             "extensions/lkai-telemetry": b"#!/usr/bin/env python3\n"}
    for f, b in files.items():
        (r / f).write_bytes(b)
    (r / "build_prof_zip.sh").write_text("PYSPY_SHA256=" + "a" * 64 + "\n")
    git = ["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "x"], check=True)
    commit = subprocess.run([*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    sha = {f: hashlib.sha256(b).hexdigest() for f, b in files.items()}
    pm = {"schema": "lkai-prof-build/2", "source_commit": commit, "pyspy_wheel_sha256": "a" * 64,
          "prof_zip_sha256": "z", "base_zip_sha256": "b", "telemetry_layer_sha256": "t", "sources": sha,
          "built_from": {"lkai_prof.py": "lkai_prof.py", "lkai_probe.py": "lkai_probe.py",
                         "lkai_symbolise.py": "lkai_symbolise.py", "lib/liblkaisamp.so": "lkaisamp.c",
                         "bin/py-spy": "pyspy_wheel", "telemetry-layer.zip": "extensions/lkai-telemetry"},
          "added": {"lkai_prof.py": sha["lkai_prof.py"], "lkai_probe.py": sha["lkai_probe.py"],
                    "lkai_symbolise.py": sha["lkai_symbolise.py"], "lib/liblkaisamp.so": "so", "bin/py-spy": "bin"}}
    return r, commit, pm


def test_provenance_binds_c_source_extension_and_wheel(tmp_path):
    import copy

    import summarize_prof as sp

    r, commit, pm = _prov_repo(tmp_path)
    assert sp.verify_prof_manifest(pm, commit, str(r))["result"] == "match"
    for mutate, msg in (
        (lambda m: m["sources"].__setitem__("lkaisamp.c", "0" * 64), "lkaisamp.c"),
        (lambda m: m["sources"].__setitem__("extensions/lkai-telemetry", "0" * 64), "lkai-telemetry"),
        (lambda m: m.__setitem__("pyspy_wheel_sha256", "b" * 64), "py-spy wheel"),
        (lambda m: m["added"].__setitem__("lkai_prof.py", "0" * 64), "verbatim"),
        (lambda m: m["added"].__setitem__("lib/extra.so", "x"), "no recorded source"),
        (lambda m: m["built_from"].__setitem__("lib/liblkaisamp.so", "other.c"), "no source hash"),
        (lambda m: m.pop("sources"), "non-Python"),  # an lkai-prof-build/1 manifest is refused
    ):
        bad = copy.deepcopy(pm)
        mutate(bad)
        with pytest.raises(SystemExit, match=msg):
            sp.verify_prof_manifest(bad, commit, str(r))
    with pytest.raises(SystemExit, match="not a commit"):
        sp.verify_prof_manifest(pm, "0" * 40, str(r))


def test_build_script_hashes_every_repository_file_it_packages_and_pins_the_wheel():
    import re

    src = (PROF / "build_prof_zip.sh").read_text()
    sources = set(re.search(r'^SOURCES="([^"]+)"', src, re.M).group(1).split())
    copied = set(re.findall(r'"\$HERE/([^"]+)"', src)) | set(re.findall(r"/p/(\S+\.c)\b", src))
    assert copied <= sources, copied - sources  # nothing from the repo is packaged without a source hash
    assert re.search(r"^PYSPY_SHA256=[0-9a-f]{64}$", src, re.M)  # pinned, not an optional override
    assert "${PYSPY_SHA256" not in src


def test_warm_by_request_verdict_tests_rounds_not_correlated_cells():
    """25 cells from 5 rounds are 5 environments: the test unit is the round."""
    import summarize_prof as sp

    def w(by_round):
        return {"warm.Duration.by_request": {"pct_by_cell": [x for x in by_round for _ in range(5)],
                                             "pct_by_round": by_round, "round_ids": list(range(len(by_round)))}}

    v = sp.null_verdicts({"rep2": w([-2.0, 1.0, 3.0, -1.0, 0.5]), "x": w([5.0, 6.0, 4.0, 7.0, 5.5])})
    cell = v["x"]["warm.Duration.by_request"]
    assert cell["n_pairs"] == 5 and cell["p"] == 0.0625  # 5 rounds, not 25 cells
    assert cell["verdict"] == "not measurable" and cell["consistent_direction"] is True and cell["null_max_abs"] == 3.0


def test_host_variability_reports_part_ids_unknowns_and_no_generation(tmp_path):
    import summarize_prof as sp

    tail = "START RequestId: {} Version: $LATEST\n... init cpu_part={} ggml_cpu=libggml-cpu-a.so,libggml-cpu-{}.so ...\n"
    rows = []
    for i, (arm, cls, cpu) in enumerate([("rep", "cold", "0xd40"), ("rep", "warm", "0xd40"), ("rep", "cold", "0xd0c"),
                                         ("x", "cold", "0xd4f"), ("x", "restore", "0xd40")]):
        r = _row(arm, i % 2, cls, cpu=cpu)
        r["log_tail"] = tail.format(f"req{i}", cpu, cpu)
        rows.append(r)
    rows[1]["log_tail"] = "no start line"
    d = tmp_path / "logs"
    d.mkdir()
    with gzip.open(d / "fn.jsonl.gz", "wt") as f:
        for i in (0, 2, 3, 4):
            f.write(json.dumps({"logStreamName": f"s{i}", "message": f"START RequestId: req{i} Version: $LATEST"}) + "\n")
        f.write(json.dumps({"logStreamName": "s0", "message": "INIT_START Runtime Version: python:3.12.v43\tARN"}) + "\n")
    probes = tmp_path / "probe"
    probes.mkdir()
    (probes / "p1.json").write_text(json.dumps({"cpuinfo": {"processors": 6, "first": {
        "CPU implementer": "0x41", "CPU variant": "0x1", "CPU part": "0xd40", "CPU revision": "1", "Features": "fp sve"}},
        "nproc": 6, "uname": {"release": "5.10"}, "python": "3.12.14"}))
    hv = sp.host_variability(rows, {"runtime_version_arn": "arn:rt", "region": "r"}, str(tmp_path), str(probes))
    o = hv["observed"]
    assert hv["new_measurements"] is False and hv["aws_calls"] == 0
    assert o["cpu_part_ids"] == ["0xd0c", "0xd40", "0xd4f"] and o["multiple_identities"] is True
    assert o["cpu_part_counts_by_arm_class"]["rep"] == {"cold": {"0xd0c": 1, "0xd40": 1}, "warm": {"0xd40": 1}}
    assert o["cpu_part_counts_by_arm_class"]["x"] == {"cold": {"0xd4f": 1}, "restore": {"0xd40": 1}}
    assert o["ggml_cpu_variants_loaded_by_part"]["0xd4f"] == {"libggml-cpu-a.so,libggml-cpu-0xd4f.so": 1}
    assert o["log_stream_linkage"]["rows_linked_to_stream"] == 4 and o["log_stream_linkage"]["rows_unlinked"] == 1
    assert o["log_stream_linkage"]["streams_with_more_than_one_part"] == 0
    assert o["runtime_versions_in_init_start"] == {"python:3.12.v43": 1}
    assert o["probe_environments"]["by_file"]["p1.json"]["implementer"] == "0x41"
    for k in ("cpu_features_hwcaps_per_invocation", "cgroup_cpu_memory_limits", "restored_core_count",
              "kernel_per_invocation", "availability_zone", "processor_generation_or_microarchitecture"):
        assert k in hv["unknown"]
    blob = json.dumps(hv)
    assert "Graviton" not in blob and "Neoverse" not in blob  # part IDs are labels, never mapped to a generation


GENERATION = __import__("re").compile(r"Graviton ?[2-5]|Neoverse|Graviton(?:2|3|4|5) sandboxes")


def test_p3_text_and_results_never_name_a_generation():
    """The initiative's host-variability contract forbids inferring a processor generation from a CPU part ID or
    from timings. P3's own docs, matrix, meta and committed results use part IDs only."""
    files = [PROF / "README.md", PROF / "applicability.json", PROF / "meta.json", PROF / "summarize_prof.py"]
    for run in ("20261020T231634Z", "20261020T124451Z"):
        d = PROF / "results" / run
        files += [d / f for f in ("README.md", "meta.json", "applicability.json", "results.json") if (d / f).exists()]
    for f in files:
        hits = GENERATION.findall(f.read_text())
        assert not hits, (f, hits[:3])


@pytest.mark.parametrize("run,extra", [("20261020T231634Z", []), ("20261020T124451Z", ["--superseded-by",
                                                                                         "20261020T231634Z"])])
def test_committed_results_regenerate_from_committed_evidence(tmp_path, run, extra):
    """Each committed results.json is what summarize_prof.py makes of the committed, redacted run directory, its
    prof-manifest.json and its meta copy (the commit binding is re-verified for the current run)."""
    d = PROF / "results" / run
    if not (d / "invocations.jsonl.gz").exists():
        pytest.skip("run evidence not present")
    if subprocess.run(["git", "-C", str(PROF), "rev-parse", "--verify", "-q", "0eabd60^{commit}"],
                      capture_output=True).returncode:
        pytest.skip("experiment commit not in this clone")
    work = tmp_path / run
    shutil.copytree(d, work)
    with gzip.open(work / "invocations.jsonl.gz") as f:
        (work / "invocations.jsonl").write_bytes(f.read())
    with gzip.open(work / "xray.jsonl.gz") as f:
        (work / "xray.jsonl").write_bytes(f.read())
    (work / "results.json").unlink()
    want = json.loads((d / "results.json").read_text())
    commit = want["commits"]["experiment"]
    subprocess.run([sys.executable, str(PROF / "summarize_prof.py"), str(work),
                    "--reference", str(ROOT / "baseline" / "reference" / "outputs.json"),
                    "--baseline-commit", want["commits"]["baseline"], "--meta", str(d / "meta.json"),
                    "--applicability", str(d / "applicability.json"), "--offline", str(PROF / "results" / "offline"),
                    "--probe", str(PROF / "results" / "lambda-probe" / "q1.json"), "--experiment-commit", commit,
                    "--prof-manifest", str(d / "prof-manifest.json"), *extra], check=True, capture_output=True)
    got = json.loads((work / "results.json").read_text())
    got["artifacts"]["invocations"] = "invocations.jsonl.gz"  # stage_run.sh points the link at the .gz
    for k in ("profiler_overhead_paired", "profiles", "hot_spots", "restore_evidence", "correctness_by_arm",
              "deployment", "conclusion", "status", "host_variability"):
        assert got.get(k) == want.get(k), k
