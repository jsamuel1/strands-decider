"""examples/lambda-kleidiai/baseline: phase timers, the instrumented entry point and the results summary.

The entry point is exercised against stand-ins for PR #77's handler/loader and the snapstart engine, so these tests
need neither llama.cpp nor AWS. They pin the contract the experiments rely on: the instrumented handler returns the
same answers as the uninstrumented one, records every init phase once, classifies cold/restore/warm, and the
summary's distributions, overhead and correctness diff are computed from the rows they claim.
"""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import os
import sys
import types
from concurrent.futures import Future
from pathlib import Path

import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "baseline")
sys.path.insert(0, os.path.abspath(BASE))

import phases  # noqa: E402
import summarize  # noqa: E402


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    """deploy.py and bench.py import boto3/botocore at module level; the test env need not have them. The tests
    replace every client they touch, so empty stand-ins suffice when the SDK is absent."""
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
    for name, mod in {
        "boto3": boto3,
        "botocore": botocore,
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in ("deploy", "bench"):
        monkeypatch.delitem(sys.modules, m, raising=False)


def test_parse_report_reads_lambda_fields():
    log = (
        "START RequestId: x Version: 3\n"
        'LKAI {"kind":"invoke","inv":{"n":1}}\n'
        "REPORT RequestId: x\tDuration: 812.31 ms\tBilled Duration: 1103 ms\tMemory Size: 10240 MB\t"
        "Max Memory Used: 4321 MB\tRestore Duration: 410.20 ms\tBilled Restore Duration: 290 ms\t\n"
    )
    rep = phases.parse_report(log)
    assert rep["Duration"] == 812.31
    assert rep["Restore Duration"] == 410.2
    assert rep["Billed Restore Duration"] == 290
    assert rep["Max Memory Used"] == 4321
    assert phases.parse_lkai(log) == [{"kind": "invoke", "inv": {"n": 1}}]


def test_parse_lkai_skips_garbage_and_finds_prefixed_lines():
    log = 'LKAI {bad json\n2026-10-20T00:00:00Z abc INFO LKAI {"kind":"x"}\nLKAI not-json\n'
    assert phases.parse_lkai(log) == [{"kind": "x"}]


def test_recorder_phases_with_fake_clock():
    t = iter([0, 1_000_000_000, 3_500_000_000, 4_000_000_000, 4_250_000_000])
    lines = []
    rec = phases.Recorder(t0_ns=0, clock=lambda: next(t), wall=lambda: 7, write=lines.append)
    with rec.phase("a"):
        pass
    with rec.phase("a"):  # accumulates
        pass
    assert rec.init == {"a": 1.5}  # (1.0 - 0) + (4.0 - 3.5)
    rec.emit("k", x=1)
    assert json.loads(lines[0].removeprefix(phases.PREFIX)) == {
        "schema": phases.SCHEMA,
        "kind": "k",
        "wall_ns": 7,
        "x": 1,
    }


def test_first_call_timed_only_times_once():
    calls = []
    rec = phases.Recorder(write=lambda s: None)
    f = phases.first_call_timed(rec, "first", lambda x: calls.append(x) or x)
    assert f(1) == 1 and f(2) == 2
    assert calls == [1, 2]
    assert list(rec.init) == ["first"]


@pytest.mark.parametrize(
    ("n", "init_type", "restored", "want"),
    [
        (1, "on-demand", False, "cold"),
        (1, "snap-start", True, "restore"),
        (2, "on-demand", False, "warm"),
        (5, "snap-start", True, "warm"),
        (1, "provisioned-concurrency", False, "init-other"),
    ],
)
def test_start_class(n, init_type, restored, want):
    assert phases.start_class(n, init_type, restored) == want


def test_enabled_flag():
    assert phases.enabled({}) and phases.enabled({"LKAI_PHASES": "1"})
    assert not phases.enabled({"LKAI_PHASES": "0"})


def test_wait_updated_waits_for_snapstart_optimization(monkeypatch):
    import deploy

    responses = [
        {
            "State": "Active",
            "LastUpdateStatus": "Successful",
            "SnapStart": {"ApplyOn": "PublishedVersions", "OptimizationStatus": "Off"},
        },
        {
            "State": "Active",
            "LastUpdateStatus": "Successful",
            "SnapStart": {"ApplyOn": "PublishedVersions", "OptimizationStatus": "On"},
        },
    ]

    class Lambda:
        def get_function_configuration(self, **kwargs):
            assert kwargs == {"FunctionName": "fn", "Qualifier": "7"}
            return responses.pop(0)

    sleeps = []
    monkeypatch.setattr(deploy.time, "sleep", sleeps.append)
    got = deploy.wait_updated(Lambda(), "fn", "7")
    assert got["SnapStart"]["OptimizationStatus"] == "On"
    assert sleeps == [3]
    assert responses == []


def test_run_script_arms_cleanup_trap_before_benchmark():
    script = (Path(BASE) / "run.sh").read_text()
    deploy = script.index('"$PY" "$HERE/deploy.py" up')
    trap = script.index("trap cleanup_versions EXIT")
    bench = script.index('"$PY" "$HERE/bench.py"')
    assert deploy < trap < bench
    assert "TEARDOWN must be versions, all, or none" in script


def test_bench_deletes_published_versions_if_later_publish_fails(tmp_path, monkeypatch):
    import bench

    class FakeLambda:
        def get_function_configuration(self, FunctionName):
            return {
                "FunctionName": FunctionName,
                "Runtime": "python3.12",
                "Architectures": ["arm64"],
                "MemorySize": 10240,
                "Timeout": 900,
                "EphemeralStorage": {"Size": 512},
                "SnapStart": {"ApplyOn": "PublishedVersions"},
                "Handler": "lkai_entry.handler",
                "CodeSha256": "eg==",
                "CodeSize": 1,
                "Environment": {"Variables": {}},
            }

        def get_runtime_management_config(self, FunctionName):
            return {}

    deleted = []

    class FakeBench:
        def __init__(self, *args, **kwargs):
            self.lam = FakeLambda()
            self.order = ["probe"]

        def set_cold_tag(self, arm, tag):
            pass

        def wave(self, *args, **kwargs):
            return [{"error": None, "rt_s": 1.0}]

        def publish(self, arm, tag):
            if arm == "rep":
                raise RuntimeError("publish failed")
            return {"version": "7", "publish_s": 1.0}

        def delete_version(self, arm, version):
            deleted.append((arm, version))

    state = {
        "region": "us-east-2",
        "functions": {a: {"name": a} for a in ("inst", "rep")},
        "manifest": {"function_zip": {"sha256": "7a"}},
    }
    monkeypatch.setattr(bench.deploy, "load_state", lambda build: state)
    monkeypatch.setattr(bench.deploy, "b64_to_hex", lambda value: "7a")
    monkeypatch.setattr(bench.boto3, "Session", lambda **kwargs: object())
    monkeypatch.setattr(bench, "Bench", FakeBench)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "--out", str(tmp_path), "--rounds", "1", "--concurrency", "1"],
    )
    with pytest.raises(RuntimeError, match="publish failed"):
        bench.main()
    assert deleted == [("inst", "7")]


# ------------------------------------------------------------------ entry point against stand-ins


def test_publish_deletes_version_if_snapshot_wait_fails(monkeypatch):
    import bench

    deleted = []

    class Lam:
        def publish_version(self, FunctionName, Description):
            return {"Version": "9"}

        def delete_function(self, FunctionName, Qualifier):
            deleted.append((FunctionName, Qualifier))

    b = bench.Bench.__new__(bench.Bench)
    b.lam, b.st = Lam(), {"functions": {"inst": {"name": "fn"}}}

    def fail(*args):
        raise SystemExit("fn:9: unexpected SnapStart optimization status 'Failed'")

    monkeypatch.setattr(bench.deploy, "wait_updated", fail)
    with pytest.raises(SystemExit):
        b.publish("inst", "t")
    assert deleted == [("fn", "9")]


ANSWER = {"q": {"type": "noul", "noul": 0.25}}


def _install_fakes(monkeypatch, init_type):
    """Stand-ins with the shapes lkai_entry wraps: loader.download_to_memfd, engine.Shim/load_engine/
    HeadRowsEngine.evaluate, handler._ENGINE/_cpu_part, app.handler (which calls handler's engine)."""
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", init_type)
    for m in ("lkai_entry", "engine", "loader", "handler", "app", "snapshot_restore_py"):
        monkeypatch.delitem(sys.modules, m, raising=False)

    loader = types.ModuleType("loader")
    loader.download_to_memfd = lambda bucket, key: "/proc/self/fd/0"

    engine = types.ModuleType("engine")

    class Shim:
        def __init__(self, *a, **k):
            self.lib = types.SimpleNamespace()  # no dec_load_phases_ns: a shim built without timers

    class HeadRowsEngine:
        def evaluate(self, req):
            return {"answers": ANSWER}

    def load_engine(*a, **k):
        return HeadRowsEngine(), engine.Shim()

    engine.Shim, engine.HeadRowsEngine, engine.load_engine = Shim, HeadRowsEngine, load_engine

    handler = types.ModuleType("handler")
    handler._cpu_part = lambda: "0xd40"

    def _load():
        import engine as e
        import loader as lo

        lo.download_to_memfd("b", "k")
        eng, _ = e.load_engine("lib", "g", "a", 6)
        eng.evaluate({"warm": "up"})
        return eng, 1.5

    app = types.ModuleType("app")

    def app_handler(event, context):
        import handler as h  # snapstart/app.py imports PR #77's handler at module level

        eng, _ = h._ENGINE.result()
        return {
            "latency_ms": 12.5,
            "answers": eng.evaluate(event)["answers"],
            "usage": {"input_tokens": 9},
        }

    app.handler = app_handler

    hooks = types.ModuleType("snapshot_restore_py")
    reg = {}
    hooks.register_before_snapshot = lambda f: reg.__setitem__("before", f)
    hooks.register_after_restore = lambda f: reg.__setitem__("after", f)

    # handler.py loads at import, so it must be importable when lkai_entry imports it: install the modules lazily
    # through a finder that builds `handler` (and runs _load) on first import, after lkai_entry patched engine/loader.
    monkeypatch.setitem(sys.modules, "loader", loader)
    monkeypatch.setitem(sys.modules, "engine", engine)
    monkeypatch.setitem(sys.modules, "app", app)
    monkeypatch.setitem(sys.modules, "snapshot_restore_py", hooks)

    class _Finder:
        def find_spec(self, name, path=None, target=None):
            if name != "handler":
                return None
            return importlib.util.spec_from_loader(name, self)

        def create_module(self, spec):
            fut2 = Future()
            fut2.set_result(_load())
            handler._ENGINE = fut2
            return handler

        def exec_module(self, module):
            pass

    finder = _Finder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    return reg


def _ctx(i):
    return types.SimpleNamespace(aws_request_id=f"req-{i}", function_version="3")


def test_entry_instrumented_records_phases_and_preserves_answers(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "1")
    reg = _install_fakes(monkeypatch, "snap-start")
    import lkai_entry

    reg["before"]()
    reg["after"]()
    out1 = lkai_entry.handler({"state": "s"}, _ctx(1))
    out2 = lkai_entry.handler({"state": "s"}, _ctx(2))
    assert out1["answers"] == ANSWER and out2["answers"] == ANSWER
    recs = phases.parse_lkai(capsys.readouterr().out)
    kinds = [r["kind"] for r in recs]
    assert kinds == ["before_snapshot", "after_restore", "invoke", "invoke"]
    inv1, inv2 = recs[2], recs[3]
    assert inv1["inv"]["class"] == "restore" and inv2["inv"]["class"] == "warm"
    assert inv1["env_id"] == recs[1]["env_id"] != recs[0]["env_id"]
    assert inv1["inv"]["engine_s"] == 0.0125
    assert "restore_to_invoke_s" in inv1["inv"]
    for ph in (
        "imports",
        "weight_download",
        "shim_init",
        "engine_load",
        "assets_load",
        "warmup_inference",
        "model_ready",
        "import_total",
    ):
        assert ph in inv1["init"], ph
    assert "init" not in inv2  # init phases ride on the first invocation only
    assert inv1["counters"]["snapstart_hooks"] is True
    assert out1["lkai"]["inv"]["n"] == 1


def test_entry_on_demand_is_cold(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "1")
    _install_fakes(monkeypatch, "on-demand")
    import lkai_entry

    lkai_entry.handler({"state": "s"}, _ctx(1))
    rec = phases.parse_lkai(capsys.readouterr().out)[-1]
    assert rec["inv"]["class"] == "cold"
    assert rec["init"]["model_ready"] == 1.5  # PR #77's init_seconds, from the background load


def test_entry_report_only_arm_emits_nothing(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "0")
    _install_fakes(monkeypatch, "on-demand")
    import lkai_entry

    out = lkai_entry.handler({"state": "s"}, _ctx(1))
    assert out["answers"] == ANSWER and "lkai" not in out
    assert phases.parse_lkai(capsys.readouterr().out) == []


# ------------------------------------------------------------------ summary


def test_pct_and_dist():
    xs = [float(i) for i in range(1, 11)]
    assert summarize.pct(xs, 50) == 5 and summarize.pct(xs, 90) == 9 and summarize.pct(xs, 99) == 10
    d = summarize.dist(xs)
    assert d["n"] == 10 and d["min"] == 1 and d["max"] == 10 and sum(d["histogram"]["counts"]) == 10
    assert summarize.dist([None]) is None
    assert summarize.dist([2.0, 2.0])["histogram"] == {"edges": [2.0, 2.0], "counts": [2]}


def _row(
    arm,
    rnd,
    wave,
    *,
    rt=1.0,
    lat=500.0,
    report=None,
    cpu="0xd40",
    answers=None,
    lkai=None,
    rid="r1",
):
    report = (
        report
        if report is not None
        else {"Duration": lat + 10, "Billed Duration": lat + 11, "Max Memory Used": 4000}
    )
    obs = phases.observed_class(report)
    return {
        "arm": arm,
        "round": rnd,
        "wave": wave,
        "seq": 0,
        "request_id": rid,
        "rt_s": rt,
        "error": None,
        "report": report,
        "expected_class": wave,
        "observed_class": obs,
        "class_mismatch": obs != wave,
        "cpu_part": cpu,
        "latency_ms": lat,
        "answers": answers or {"q": {"type": "noul", "noul": 0.5}},
        **({"lkai": lkai} if lkai else {}),
    }


def _run_dir(tmp_path, rows):
    cond = {
        "region": "us-east-2",
        "rounds": 2,
        "concurrency": 2,
        "warm_per_arm_per_round": 2,
        "arms": ["inst", "rep"],
        "probe": "r1",
        "requests_file": "requests.json",
        "started": "t0",
        "finished": "t1",
        "requests": {"requests": [{"id": "r1"}, {"id": "r2"}]},
        "client": {},
        "runtime_version_arn": "arn:aws:lambda:us-east-2::runtime:abc",
        "functions": {
            a: {
                "FunctionName": f"lkai-p1-{a}",
                "Runtime": "python3.12",
                "Architectures": ["arm64"],
                "MemorySize": 10240,
                "EphemeralStorage": {"Size": 512},
                "Timeout": 900,
                "SnapStart": {"ApplyOn": "PublishedVersions"},
                "Handler": "lkai_entry.handler",
                "environment": {"LKAI_PHASES": "1" if a == "inst" else "0"},
            }
            for a in ("inst", "rep")
        },
        "build_manifest": {
            "repo_commit": "abc",
            "repo_dirty": False,
            "model": {"id": "m", "gguf_bytes": 2 * 2**30, "gguf_sha256": "g"},
            "llama_cpp": {"commit": "f74e"},
            "function_zip": {
                "sha256": "z",
                "source_commit": "zip-source",
                "inputs_sha256": "inputs-key",
            },
            "images": {},
            "toolchain": [],
            "assets_sha256": {},
        },
        "rounds_meta": [
            {
                "t_versions_deleted": 100.0,
                "published": {
                    "inst": {"t_publish": 0.0, "version": "3"},
                    "rep": {"t_publish": 0.0, "version": "3"},
                },
            }
        ],
    }
    (tmp_path / "conditions.json").write_text(json.dumps(cond))
    (tmp_path / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "logs").mkdir()
    snap = {
        "kind": "before_snapshot",
        "wall_ns": 1,
        "init": {"weight_download": 3.0, "model_ready": 6.0},
    }
    with gzip.open(tmp_path / "logs" / "lkai-p1-inst.jsonl.gz", "wt") as f:
        for stream, wall in (
            ("2026/10/20/[3]abc", 1),
            ("2026/10/20/[2]old", 2),
        ):  # [2]: not this run's
            msg = "LKAI " + json.dumps(snap | {"wall_ns": wall})
            f.write(json.dumps({"timestamp": 0, "logStreamName": stream, "message": msg}) + "\n")
    return str(tmp_path)


def test_summarize_end_to_end(tmp_path):
    init = {"weight_download": 25.0, "model_ready": 30.0, "imports": 0.5}
    rows = []
    for rnd in range(2):
        for arm in ("inst", "rep"):
            for i in range(2):
                lk = (
                    {
                        "inv": {"handler_s": 31.0, "wait_load_s": 30.0},
                        "init": init,
                        "counters": {"weight_mib_s": 80.0},
                    }
                    if arm == "inst"
                    else None
                )
                rows.append(
                    _row(
                        arm,
                        rnd,
                        "cold",
                        rt=40.0 + i,
                        report={
                            "Init Duration": 9000.0 + rnd * 100,
                            "Duration": 31000.0,
                            "Billed Duration": 40001.0,
                            "Max Memory Used": 5000,
                        },
                        lkai=lk,
                    )
                )
                rows.append(
                    _row(
                        arm,
                        rnd,
                        "restore",
                        rt=6.0,
                        report={
                            "Restore Duration": 400.0,
                            "Billed Restore Duration": 300,
                            "Duration": 900.0 + (20 if arm == "inst" else 0),
                            "Billed Duration": 901.0,
                            "Max Memory Used": 4100,
                        },
                        lkai={
                            "inv": {
                                "handler_s": 0.9,
                                "wait_load_s": 0.0,
                                "restore_to_invoke_s": 0.05,
                            }
                        }
                        if arm == "inst"
                        else None,
                    )
                )
                rows.append(_row(arm, rnd, "warm", lat=500.0, rid=f"r{i + 1}"))
    bad = _row(
        "inst", 1, "restore", report={"Duration": 900.0}
    )  # expected restore, REPORT says warm
    rows.append(bad)
    res = summarize.summarize(_run_dir(tmp_path, rows), None, None, {})
    assert res["schema"] == "lkai-result/1"
    assert res["classification"]["mismatches"] == 1
    assert res["classification"]["expected_vs_observed"]["restore->warm"] == 1
    inst = res["phases"]["inst"]
    assert inst["cold"]["n"] == 4 and inst["restore"]["n"] == 4  # the mismatch row is excluded
    assert inst["cold"]["weight_download"]["p50"] == 25.0
    assert inst["cold"]["first_request_total"]["p50"] == 40.0
    assert inst["restore"]["snapshot_init"]["weight_download"]["p50"] == 3.0
    assert "snapshot_inits_per_version" not in inst["restore"]  # counts, not distributions
    assert res["cost"]["snapshot_inits_per_version"] == {"inst": {"3": 1}}
    assert res["status"] == "baseline" and res["changed_variable"] is None
    assert res["build"]["function_zip"] == {
        "sha256": "z",
        "source_commit": "zip-source",
        "inputs_sha256": "inputs-key",
    }
    assert res["cost"]["snapshot_inits"] == 2  # one observed on inst, the same assumed for rep
    assert set(inst["warm"]["steady_state_by_request"]) == {"r1", "r2"}
    assert (
        res["phases"]["rep"]["cold"].get("weight_download") is None
    )  # REPORT-only arm has no phase records
    ov = res["instrumentation_overhead"]["restore"]["report_duration"]
    assert ov["by_cpu_part"]["0xd40"]["delta_p50_s"] == 0.02
    assert ov["pooled_delta_p50_s"] == 0.02
    assert res["variance"]["inst"]["cold"]["per_round_median_s"] == [40.0, 40.1]
    c = res["cost"]
    assert c["cold_starts"] == 8 and c["restores"] == 8  # the mismatch row has no Restore Duration
    assert c["s3_get_requests_per_download"] == 32
    assert c["snapstart_cache_usd"] == round(
        2 * 3 * 3600 * 10 * summarize.PRICES["snapstart_cache_gb_s"], 4
    )


def test_build_script_uses_immutable_images_and_records_zip_source():
    script = (Path(BASE) / "build.sh").read_text()
    assert "LAMBDA_IMAGE=public.ecr.aws/lambda/python@sha256:" in script
    assert "PY_IMAGE=python@sha256:" in script
    assert "LAMBDA_IMAGE=public.ecr.aws/lambda/python:3.12" not in script
    assert "PY_IMAGE=python:3.12-bookworm" not in script
    assert ".zip-repo-commit" in script
    assert '"source_commit"' in script and '"inputs_sha256"' in script


def test_overhead_matches_cpu_parts():
    # inst ran on Graviton2 and 3, rep only on Graviton3: only the 0xd40 cell is compared.
    rows = [
        _row("inst", 0, "warm", lat=800.0, cpu="0xd0c"),
        _row("inst", 0, "warm", lat=470.0, cpu="0xd40"),
        _row("rep", 0, "warm", lat=460.0, cpu="0xd40"),
    ]
    ov = summarize.overhead(rows)["warm"]["steady_state_inference"]
    assert list(ov["by_cpu_part"]) == ["0xd40"]
    assert ov["pooled_delta_p50_s"] == pytest.approx(0.01)


def test_reference_and_compare():
    a = {
        "q": {
            "type": "choice",
            "choice": "x",
            "probabilities": {"x": 0.7, "y": 0.3},
            "confidence": 0.4,
        }
    }
    b = {
        "q": {
            "type": "choice",
            "choice": "y",
            "probabilities": {"x": 0.45, "y": 0.55},
            "confidence": 0.1,
        }
    }
    rows = [_row("inst", 0, "warm", answers=a), _row("inst", 0, "warm", answers=a, cpu="0xd0c")]
    ref = summarize.reference(rows)
    assert set(ref["r1"]) == {"0xd40", "0xd0c"}
    same = summarize.compare(rows, ref)
    assert same["compared"] == 2 and same["bit_identical"] == 2 and same["max_abs_delta"] == 0
    diff = summarize.compare([_row("inst", 0, "warm", answers=b)], ref)
    assert diff["choice_flips"] == 1 and diff["bit_identical"] == 0
    assert diff["max_abs_delta"] == pytest.approx(0.3)
    extra = {**a, "q": {**a["q"], "unexpected": 1}}
    extra_result = summarize.compare([_row("inst", 0, "warm", answers=extra)], ref)
    assert extra_result["bit_identical"] == 0
    assert extra_result["answer_shape_mismatches"] == 1
    missing_field = {"q": {"type": "choice", "choice": "x"}}
    missing_result = summarize.compare([_row("inst", 0, "warm", answers=missing_field)], ref)
    assert missing_result["bit_identical"] == 0
    assert missing_result["answer_shape_mismatches"] == 1
    other = summarize.compare([_row("inst", 0, "warm", answers=a, cpu="0xd4f")], ref)
    assert other["no_reference_for_cpu_part"] == 1 and other["compared"] == 0
