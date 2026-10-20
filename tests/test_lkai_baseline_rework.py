"""examples/lambda-kleidiai/baseline review-4 fixes: the memory condition, content-addressed build reuse, malformed
successes as errors, and cleanup that attempts every version. No AWS, Docker or model needed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / "examples" / "lambda-kleidiai" / "baseline"
sys.path.insert(0, str(BASE))

import buildcache  # noqa: E402
import summarize  # noqa: E402


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    """deploy.py and bench.py import boto3/botocore at module level; stand-ins when the SDK is absent."""
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


ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
NoSuchEntity = type("NoSuchEntityException", (Exception,), {})
ACCOUNT = "210987654321"


class Sts:
    def __init__(self, account=ACCOUNT):
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account}


def _clean_manifest(build, **extra):
    """A build/manifest.json of a clean build (lkai-build-manifest/2, no dirty inputs)."""
    m = {"schema": "lkai-build-manifest/2", "repo_dirty": False, "dirty_inputs": []} | extra
    Path(build).mkdir(parents=True, exist_ok=True)
    (Path(build) / "manifest.json").write_text(json.dumps(m))
    return m


@pytest.fixture
def clean_harness(monkeypatch):
    """deploy/bench see a clean harness; the real check is tested on its own below."""
    monkeypatch.setattr(
        buildcache, "harness_provenance", lambda repo=None: {"commit": "abc", "dirty_inputs": []}
    )


# ------------------------------------------------------------------ 1. memory is an input


@pytest.mark.parametrize("value", ["127", "10241", "abc", "1.5", ""])
def test_memory_rejects_out_of_range_and_non_integer(value):
    import deploy

    with pytest.raises(Exception, match="memory must be"):
        deploy.memory_mb(value)


def _fake_functions_session(created):
    class Logs:
        class exceptions:
            ResourceAlreadyExistsException = type("E", (Exception,), {})

        def create_log_group(self, **kw):
            pass

        def put_retention_policy(self, **kw):
            pass

    class Lam:
        class exceptions:
            ResourceNotFoundException = ResourceNotFound
            InvalidParameterValueException = type("IPV", (Exception,), {})

        def get_function(self, FunctionName):
            raise ResourceNotFound()

        def create_function(self, **kw):
            created[kw["FunctionName"]] = kw

        def get_function_configuration(self, FunctionName, **kw):
            return {
                "State": "Active",
                "LastUpdateStatus": "Successful",
                "FunctionArn": f"arn:{FunctionName}",
                "CodeSha256": "eg==",
                "CodeSize": 1,
                "MemorySize": created[FunctionName]["MemorySize"],
            }

    class Sess:
        def client(self, name):
            return {"lambda": Lam(), "logs": Logs(), "iam": object(), "sts": Sts()}[name]

    return Sess()


def test_deploy_up_sets_requested_memory_on_both_functions(tmp_path, monkeypatch, clean_harness):
    import deploy

    created: dict = {}
    sess = _fake_functions_session(created)
    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: sess)
    monkeypatch.setattr(deploy, "ensure_role", lambda iam, st, bucket: "arn:role")
    monkeypatch.setattr(deploy, "pin_runtime", lambda lam, st, logs=None: None)
    deploy.save_state(
        str(tmp_path),
        {
            "prefix": "p",
            "region": "us-east-2",
            "bucket": "b",
            "model_key": "m",
            "zip_key": "z",
            "manifest": {"function_zip": {"sha256": "ab" * 32}},
        },
    )
    _clean_manifest(tmp_path)
    deploy.main(["functions", "--build", str(tmp_path), "--memory", "4096", "--profile", "p"])
    assert {f["MemorySize"] for f in created.values()} == {4096}
    assert set(created) == {"p-inst", "p-rep"}
    st = deploy.load_state(str(tmp_path))
    assert st["memory_mb"] == 4096
    assert {f["memory_mb"] for f in st["functions"].values()} == {4096}


def _bench_main_fixture(tmp_path, monkeypatch, deployed_memory):
    import bench

    class FakeLambda:
        def get_function_configuration(self, FunctionName):
            return {
                "FunctionName": FunctionName,
                "Runtime": "python3.12",
                "Architectures": ["arm64"],
                "MemorySize": deployed_memory,
                "Timeout": 900,
                "EphemeralStorage": {"Size": 512},
                "SnapStart": {"ApplyOn": "PublishedVersions"},
                "Handler": "lkai_entry.handler",
                "CodeSha256": "eg==",
                "CodeSize": 1,
                "Environment": {"Variables": {"LKAI_PHASES": summarize.ARM_PHASES[FunctionName]}},
            }

        def get_runtime_management_config(self, FunctionName):
            return {}

    class FakeBench:
        def __init__(self, *a, **k):
            self.lam = FakeLambda()
            self.order = ["probe"]

        def set_cold_tag(self, arm, tag):
            pass

        def wave(self, *a, **k):
            return [{"error": None, "rt_s": 1.0}]

        def publish(self, arm, tag):
            return {"version": "7", "publish_s": 1.0}

        def delete_version(self, arm, v):
            pass

    state = {
        "region": "us-east-2",
        "memory_mb": 10240,
        "functions": {a: {"name": a} for a in ("inst", "rep")},
        "manifest": {
            "schema": "lkai-build-manifest/2",
            "dirty_inputs": [],
            "function_zip": {"sha256": "7a"},
        },
    }
    monkeypatch.setenv("AWS_PROFILE", "lkai-test")
    monkeypatch.setattr(bench.deploy, "load_state", lambda build: state)
    monkeypatch.setattr(bench.deploy, "b64_to_hex", lambda v: "7a")
    monkeypatch.setattr(bench.deploy, "check_account", lambda sess, st, expect=None: "1")
    monkeypatch.setattr(
        bench.buildcache, "harness_provenance", lambda: {"commit": "abc", "dirty_inputs": []}
    )
    monkeypatch.setattr(bench.boto3, "Session", lambda **kw: object())
    monkeypatch.setattr(bench, "Bench", FakeBench)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    monkeypatch.setattr(bench.Bench, "save_logs", lambda *a: 0, raising=False)
    return bench


def test_bench_refuses_when_deployed_memory_differs(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "--out", str(tmp_path / "run"), "--memory", "2048", "--rounds", "2"],
    )
    with pytest.raises(SystemExit, match="MemorySize 10240 MB, run wants 2048 MB"):
        bench.main()


def test_bench_records_memory_condition(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=2048)
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--out",
            str(out),
            "--memory",
            "2048",
            "--rounds",
            "2",
            "--warm",
            "5",
            "--log-lag",
            "0",
        ],
    )
    bench.Bench.save_logs = lambda self, t0, t1: 0
    bench.main()
    cond = json.loads((out / "conditions.json").read_text())
    assert cond["memory_mb"] == 2048
    assert cond["functions"]["inst"]["MemorySize"] == 2048


def test_bench_refuses_a_reused_run_directory(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    out = tmp_path / "run"
    out.mkdir()
    (out / "invocations.jsonl").write_text("{}\n")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out)])
    with pytest.raises(SystemExit, match="already holds a run"):
        bench.main()


# ------------------------------------------------------------------ 2. build reuse is content-addressed


def _repo(tmp_path):
    repo = tmp_path / "repo"
    for stage in buildcache.ORDER:
        for p in buildcache.INPUTS[stage]:
            f = repo / p
            if f.suffix:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"{p}\n")
            else:
                f.mkdir(parents=True, exist_ok=True)
                (f / "a.py").write_text(f"{p}\n")
    build = tmp_path / "build"
    (build / "model" / "assets").mkdir(parents=True)
    (build / "model" / "assets" / "head_config.json").write_text("{}")
    return repo, build


PINS = {
    "LLAMA_REV": "f74e",
    "CKPT": "m",
    "CKPT_REV": "216d",
    "PY_IMAGE": "python@sha256:aa",
    "LAMBDA_IMAGE": "lambda@sha256:bb",
    "QUANT_TYPE": "Q8_0",
}


def _keys(repo, build, pins=PINS):
    k: dict[str, str] = {}
    for s in buildcache.ORDER:
        k[s] = buildcache.stage_key(s, str(repo), str(build), pins, k)
    return k


def test_every_stage_key_covers_its_pins_inputs_and_upstream(tmp_path):
    repo, build = _repo(tmp_path)
    base = _keys(repo, build)
    # A pinned revision or image digest: its stage and everything downstream.
    cases = {
        "LLAMA_REV": {"llama", "weights", "zip", "quant"},
        "CKPT_REV": {"weights", "zip", "quant"},
        "PY_IMAGE": {"weights", "zip", "quant"},
        "LAMBDA_IMAGE": {"zip", "quant"},
        "QUANT_TYPE": {"quant"},
    }
    for pin, want in cases.items():
        k = _keys(repo, build, PINS | {pin: PINS[pin] + "x"})
        assert {s for s in base if k[s] != base[s]} == want, pin


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("examples/lambda-kleidiai/llama/patch_llama.py", {"llama", "weights", "zip", "quant"}),
        ("examples/lambda-kleidiai/baseline/weights.sh", {"weights", "zip", "quant"}),
        ("examples/lambda-kleidiai/weights/a.py", {"weights", "zip", "quant"}),
        ("examples/lambda/prepare_model.py", {"weights", "zip", "quant"}),
        ("src/strands_decider/a.py", {"weights", "zip", "quant"}),
        ("examples/lambda-kleidiai/baseline/container_build.sh", {"zip", "quant"}),
        ("examples/lambda-kleidiai/snapstart/a.py", {"zip", "quant"}),
        ("examples/lambda/app/a.py", {"zip", "quant"}),
    ],
)
def test_editing_an_uncommitted_input_file_changes_dependent_keys(tmp_path, path, want):
    repo, build = _repo(tmp_path)
    base = _keys(repo, build)
    (repo / path).write_text("changed, never committed\n")
    k = _keys(repo, build)
    assert {s for s in base if k[s] != base[s]} == want


def test_zip_key_covers_asset_bytes_and_new_or_deleted_files(tmp_path):
    repo, build = _repo(tmp_path)
    base = _keys(repo, build)
    (build / "model" / "assets" / "head_config.json").write_text('{"t": 2}')
    assert _keys(repo, build)["zip"] != base["zip"]
    repo2, build2 = _repo(tmp_path / "2")
    b2 = _keys(repo2, build2)
    (repo2 / "examples/lambda-kleidiai/shim/new.c").write_text("x")
    assert _keys(repo2, build2)["zip"] != b2["zip"]
    os.remove(repo2 / "examples/lambda-kleidiai/shim/new.c")
    os.remove(repo2 / "examples/lambda-kleidiai/shim/a.py")
    assert _keys(repo2, build2)["zip"] != b2["zip"]


def test_harness_only_changes_keep_every_key(tmp_path):
    repo, build = _repo(tmp_path)
    base = _keys(repo, build)
    for p in ("examples/lambda-kleidiai/baseline/bench.py", "README.md", "LICENSE"):
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text("edited\n")
    (repo / "src/strands_decider/__pycache__").mkdir()
    (repo / "src/strands_decider/__pycache__/a.cpython-312.pyc").write_bytes(b"\0")
    assert _keys(repo, build) == base
    assert not set(buildcache.NOT_KEYED["weights"]) & set(buildcache.INPUTS["weights"])


def test_stage_key_refuses_missing_pin_or_upstream(tmp_path):
    repo, build = _repo(tmp_path)
    with pytest.raises(ValueError, match="pins"):
        buildcache.stage_key("llama", str(repo), str(build), {}, {})
    with pytest.raises(ValueError, match="upstream"):
        buildcache.stage_key("quant", str(repo), str(build), PINS, {"weights": "w"})


def _outputs(build, stage):
    for o in buildcache.OUTPUTS[stage]:
        p = build / o
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(stage) if "." in p.name else p.mkdir()


def test_changed_key_invalidates_stage_and_downstream_outputs(tmp_path):
    build = tmp_path / "build"
    for s in buildcache.ORDER:
        # nothing stamped, nothing built: run it
        assert buildcache.check(str(build), s, "k1") == (False, [])
        buildcache.begin(str(build), s, "k1")
        _outputs(build, s)
        buildcache.stamp(str(build), s, "k1")
    assert all(buildcache.check(str(build), s, "k1")[0] for s in buildcache.ORDER)

    fresh, stale = buildcache.check(str(build), "weights", "k2")
    assert not fresh
    for s in ("weights", "zip", "quant"):
        for o in buildcache.OUTPUTS[s]:
            if s == "quant" or o != "model/s2b-h512-Q8_0.gguf":
                assert o in stale, o
    assert "llama.cpp" not in stale
    assert buildcache.stamped(str(build)) == {
        "llama": "k1",
        "weights": None,
        "zip": None,
        "quant": None,
    }
    # Even though the zip's own key did not change, its stamp is gone: it rebuilds from the new weights.
    assert buildcache.check(str(build), "zip", "k1")[0] is False


def test_unstamped_existing_outputs_are_stale(tmp_path):
    """An output from a build that predates the keys (or was interrupted under another key) is never reused."""
    build = tmp_path / "build"
    _outputs(build, "zip")
    _outputs(build, "quant")
    fresh, stale = buildcache.check(str(build), "zip", "k")
    assert not fresh and "function.zip" in stale and "model/s2b-h512-Q8_0.gguf" in stale


def test_check_cli_removes_stale_outputs_and_resumes_same_pending_key(tmp_path):
    build = tmp_path / "build"
    _outputs(build, "quant")
    (build / ".keys").mkdir()
    (build / ".keys" / "quant.key").write_text("q1\n")  # a finished earlier build
    assert buildcache.main(["check", "quant", "q2", "--build", str(build), "--remove"]) == 10
    assert not (build / "model/s2b-h512-Q8_0.gguf").exists()
    # The stage was interrupted before stamping: the same key resumes without wiping partial outputs.
    (build / "model/s2b-h512-Q8_0.gguf").write_text("partial")
    assert buildcache.check(str(build), "quant", "q2") == (False, [])
    assert (build / "model/s2b-h512-Q8_0.gguf").exists()
    buildcache.stamp(str(build), "quant", "q2")
    assert buildcache.main(["check", "quant", "q2", "--build", str(build)]) == 0


def test_stamp_refuses_missing_outputs_or_unbegun_key(tmp_path):
    build = tmp_path / "build"
    buildcache.check(str(build), "llama", "k")
    buildcache.begin(str(build), "llama", "k")
    with pytest.raises(SystemExit, match="missing"):
        buildcache.stamp(str(build), "llama", "k")
    _outputs(build, "llama")
    with pytest.raises(SystemExit, match="was not begun"):
        buildcache.stamp(str(build), "llama", "other")


def test_begin_refuses_while_stale_outputs_remain(tmp_path):
    """A failed delete must not leave a pending key that would let the next run resume over stale outputs."""
    build = tmp_path / "build"
    _outputs(build, "zip")
    fresh, stale = buildcache.check(str(build), "zip", "k")
    assert not fresh and stale
    with pytest.raises(SystemExit, match="stale outputs still present"):
        buildcache.begin(str(build), "zip", "k")
    assert buildcache.check(str(build), "zip", "k") == (False, stale)  # listed again, not resumed


def test_build_script_gates_every_stage_and_records_keys():
    script = (BASE / "build.sh").read_text()
    for s in buildcache.ORDER:
        assert f"fresh {s} " in script, s
        assert f'stamp {s} "$K_' in script, s
    assert 'begin "$1" "$2"' in script
    assert '[ ! -f "$G" ]' not in script  # no existence-only reuse
    assert 'rev-parse HEAD 2>/dev/null)" != "$LLAMA_REV"' not in script
    assert ".zip-inputs" not in script
    assert '"stage_keys"' in script
    assert "upstream llama=$K_LLAMA" in script and "upstream zip=$K_ZIP" in script
    weights = (BASE / "weights.sh").read_text()
    assert "[ -f $W/" not in weights  # every step renames a complete output into place
    assert weights.count(".tmp") >= 6


# ------------------------------------------------------------------ 3. malformed successes are errors


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        ({"latency_ms": 5.0}, "no-answers"),
        ({"answers": {}, "latency_ms": 5.0}, "no-answers"),
        ({"answers": None, "latency_ms": 5.0}, "no-answers"),
        ({"answers": [1], "latency_ms": 5.0}, "no-answers"),
        ({"answers": {"q": None}, "latency_ms": 5.0}, "answer-not-object"),
        ({"answers": {"q": {}}, "latency_ms": 5.0}, "answer-not-object"),
        ({"answers": {"q": {"noul": 1}}}, "no-latency"),
        ({"answers": {"q": {"noul": 1}}, "latency_ms": True}, "no-latency"),
        ({"answers": {"q": {"noul": 1}}, "latency_ms": float("nan")}, "no-latency"),
        ({"answers": {"q": {"noul": 1}}, "latency_ms": float("inf")}, "no-latency"),
        ({"statusCode": 422, "body": "{}"}, "http-422"),
        ({"raw": "not json"}, "no-answers"),
        ("text", "payload-not-object"),
    ],
)
def test_answer_problem(payload, problem):
    assert summarize.answer_problem(payload) == problem


def test_answer_problem_checks_question_ids():
    ok = {"answers": {"q": {"noul": 1}}, "latency_ms": 1}
    assert summarize.answer_problem(ok) is None
    assert summarize.answer_problem(ok, ["q"]) is None
    assert summarize.answer_problem(ok, ["q", "r"]) == "answers-do-not-match-questions"


def _bench_invoke(payload: dict, monkeypatch):
    import bench

    rows = []

    class Lam:
        def invoke(self, **kw):
            import io

            return {"Payload": io.BytesIO(json.dumps(payload).encode()), "StatusCode": 200}

    b = bench.Bench.__new__(bench.Bench)
    b.lam, b.st = Lam(), {"functions": {"inst": {"name": "fn"}}}
    b.reqs = {"r1": {"questions": {"q": {}}}}
    b.write = rows.append
    return b.invoke("inst", "$LATEST", "r1", 0, "warm", 0)


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", ""])
def test_positive_count_rejects_zero_negative_and_non_integer(value):
    import bench

    with pytest.raises(Exception, match=r"integer >= 1"):
        bench.positive_int(value)


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", ""])
def test_nonnegative_count_rejects_negative_and_non_integer(value):
    import bench

    with pytest.raises(Exception, match="non-negative integer"):
        bench.nonnegative_int(value)


@pytest.mark.parametrize("value", ["1", "17"])
def test_positive_count_accepts_positive_integer(value):
    import bench

    assert bench.positive_int(value) == int(value)


def test_wave_uses_fresh_lambda_client_per_wave():
    import bench

    seen = []

    class Lam:
        def __init__(self, name):
            self.name = name

        def invoke(self, **kw):
            import io

            seen.append(self.name)
            return {
                "Payload": io.BytesIO(b'{"answers":{"q":{"noul":0.5}},"latency_ms":1}'),
                "StatusCode": 200,
            }

    b = bench.Bench.__new__(bench.Bench)
    b.st = {"functions": {"inst": {"name": "fn"}}}
    b.reqs = {"r1": {"questions": {"q": {}}}}
    b.probe = "r1"
    b.concurrency = 2
    b.write = lambda row: None
    clients = iter([Lam("cold"), Lam("restore")])
    b.new_invoke_client = lambda: next(clients)

    b.wave("inst", "$LATEST", 0, "cold")
    b.wave("inst", "1", 0, "restore")
    assert seen == ["cold", "cold", "restore", "restore"]


def test_bench_marks_answerless_success_as_error(monkeypatch):
    rec = _bench_invoke({"latency_ms": 12.0, "usage": {}}, monkeypatch)
    assert rec["error"] == "malformed:no-answers"
    assert "answers" not in rec and rec["payload"] == {"latency_ms": 12.0, "usage": {}}
    good = _bench_invoke({"answers": {"q": {"noul": 0.5}}, "latency_ms": 12.0}, monkeypatch)
    assert good["error"] is None and good["answers"] == {"q": {"noul": 0.5}}
    wrong = _bench_invoke({"answers": {"x": {"noul": 0.5}}, "latency_ms": 12.0}, monkeypatch)
    assert wrong["error"] == "malformed:answers-do-not-match-questions"


def test_answerless_success_is_an_error_and_fails_the_baseline(tmp_path):
    import test_lkai_baseline as t

    good = t._row("inst", 0, "warm")
    bad = t._row("inst", 0, "warm")
    bad["answers"] = {}  # recorded as a success by a bench.py that did not check
    missing = t._row("rep", 0, "warm")
    del missing["answers"]
    run = t._run_dir(tmp_path, [good, bad, missing])
    ref = summarize.reference([good])
    res = summarize.summarize(run, None, None, {}, ref=ref)
    assert res["errors"]["count"] == 2
    assert {r["error"] for r in res["errors"]["rows"]} == {"malformed:no-answers"}
    c = res["correctness"]
    assert c["invocations"] == 3 and c["compared"] == 1 and c["bit_identical"] == 1
    assert c["without_answers"] == 2 and c["complete"] is False
    assert summarize.incomplete(res)
    # The malformed rows are not in the latency distributions either.
    assert res["phases"]["inst"]["warm"]["n"] == 1 and "warm" not in res["phases"]["rep"]


def test_compare_refuses_answerless_success_rows():
    import test_lkai_baseline as t

    r = t._row("inst", 0, "warm")
    r["answers"] = None
    with pytest.raises(ValueError, match="without answers"):
        summarize.compare([r], {})


def _mk(p: Path) -> Path:
    p.mkdir(parents=True)
    return p


def test_write_reference_refused_for_empty_baseline(tmp_path):
    import test_lkai_baseline as t

    run = t._run_dir(_mk(tmp_path / "run"), [])
    ref = tmp_path / "ref.json"
    assert summarize.main([run, "--write-reference", str(ref)]) == 3
    assert not ref.exists()
    res = json.loads((Path(run) / "results.json").read_text())
    assert res["correctness"]["invocations"] == 0
    assert "no invocations" in summarize.incomplete(res)


@pytest.mark.parametrize(
    "args",
    [
        ["--rounds", "0"],
        ["--rounds", "1"],
        ["--concurrency", "0"],
        ["--concurrency", "-3"],
        ["--warm", "0"],
        ["--log-lag", "-1"],
    ],
)
def test_bench_refuses_degenerate_sample_inputs(tmp_path, monkeypatch, capsys, args):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out), *args])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2 and "must be" in capsys.readouterr().err
    assert not out.exists()  # refused before any directory or AWS call


def _all_class_rows():
    import test_lkai_baseline as t

    rows = []
    for rnd in range(2):
        for arm in ("inst", "rep"):
            rows.append(
                t._row(arm, rnd, "cold", report={"Init Duration": 900.0, "Duration": 30000.0})
            )
            rows.append(
                t._row(arm, rnd, "restore", report={"Restore Duration": 400.0, "Duration": 900.0})
            )
            rows.append(t._row(arm, rnd, "warm"))
            rows.append(t._row(arm, rnd, "warm", rid="r2"))  # every fixed request is answered
    return rows


def test_baseline_missing_a_start_class_is_not_a_baseline(tmp_path):
    import test_lkai_baseline as t

    rows = [r for r in _all_class_rows() if not (r["arm"] == "rep" and r["wave"] == "warm")]
    run = t._run_dir(_mk(tmp_path / "run"), rows)
    ref = tmp_path / "ref.json"
    assert summarize.main([run, "--write-reference", str(ref)]) == 3
    assert not ref.exists()
    res = json.loads((Path(run) / "results.json").read_text())
    assert res["correctness"]["complete"] is True  # every row compared, still not a baseline
    assert "arm rep has no warm samples" in summarize.incomplete(res)


def test_baseline_with_one_round_is_not_a_baseline(tmp_path):
    import test_lkai_baseline as t

    run = t._run_dir(_mk(tmp_path / "run"), _all_class_rows())
    cond = json.loads((Path(run) / "conditions.json").read_text())
    cond["rounds"] = 1
    (Path(run) / "conditions.json").write_text(json.dumps(cond))
    ref = summarize.reference(_all_class_rows())
    res = summarize.summarize(run, None, None, {}, ref=ref)
    assert any("at least 2 are needed" in w for w in summarize.incomplete(res))


def test_write_reference_refused_for_incomplete_baseline(tmp_path):
    import test_lkai_baseline as t

    good = t._row("inst", 0, "warm")
    bad = t._row("inst", 0, "warm")
    bad["answers"] = {}
    run = t._run_dir(_mk(tmp_path / "run"), [good, bad])
    ref = tmp_path / "ref.json"
    assert summarize.main([run, "--write-reference", str(ref)]) == 3
    assert not ref.exists()
    assert json.loads((Path(run) / "results.json").read_text())["errors"]["count"] == 1


def test_complete_baseline_writes_reference(tmp_path):
    import test_lkai_baseline as t

    rows = _all_class_rows()
    run = t._run_dir(_mk(tmp_path / "run"), rows)
    ref = tmp_path / "ref.json"
    assert summarize.main([run, "--write-reference", str(ref)]) == 0
    res = json.loads((Path(run) / "results.json").read_text())
    assert res["correctness"]["complete"] is True
    assert res["correctness"]["compared"] == len(rows)
    assert res["conditions"]["memory_mb"] == 10240
    assert json.loads(ref.read_text())["outputs"]["r1"]["0xd40"] == rows[0]["answers"]


def test_published_baseline_is_complete():
    """The committed run regenerates as complete: every one of its invocations was compared."""
    run = BASE / "results" / "20261020T080835Z"
    meta = json.loads((run / "meta.json").read_text())
    res = summarize.summarize(str(run), str(BASE / "reference" / "outputs.json"), None, meta)
    c = res["correctness"]
    assert c["complete"] and c["compared"] == c["invocations"] == c["bit_identical"] == 1200
    assert summarize.incomplete(res) == []


def test_bucket_creation_omits_location_constraint_in_us_east_1():
    import deploy

    calls = []

    class S3:
        def create_bucket(self, **kw):
            calls.append(kw)

        def __getattr__(self, name):
            return lambda **kw: None

    st = {}
    deploy.ensure_bucket(S3(), "us-east-1", st, "p")
    assert set(calls[0]) == {"Bucket"}


def test_bucket_creation_sets_location_constraint_elsewhere():
    import deploy

    calls = []

    class S3:
        def create_bucket(self, **kw):
            calls.append(kw)

        def __getattr__(self, name):
            return lambda **kw: None

    deploy.ensure_bucket(S3(), "eu-west-1", {}, "p")
    assert calls[0]["CreateBucketConfiguration"] == {"LocationConstraint": "eu-west-1"}


def test_cost_refuses_region_without_pinned_prices(tmp_path):
    import test_lkai_baseline as t

    run = t._run_dir(_mk(tmp_path / "run"), [])
    cond = json.loads((Path(run) / "conditions.json").read_text())
    cond["region"] = "eu-west-1"
    with pytest.raises(summarize.UnpricedRegion, match="no pinned prices for eu-west-1"):
        summarize.costs(cond, [])
    # The pinned table is the one applied to the pinned Region, and a caller can supply its own.
    cond["region"] = "us-east-2"
    assert summarize.costs(cond, [])["prices"] is summarize.PRICES_BY_REGION["us-east-2"]
    cond["region"] = "eu-west-1"
    mine = dict(summarize.PRICES, source="mine")
    assert summarize.costs(cond, [], prices=mine)["prices"]["source"] == "mine"


def test_bench_refuses_unpriced_region_before_any_aws_call(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    monkeypatch.setattr(
        bench.boto3, "Session", lambda **kw: pytest.fail("AWS session for an unpriced Region")
    )
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out), "--region", "eu-west-1"])
    with pytest.raises(SystemExit, match="no pinned prices for eu-west-1"):
        bench.main()
    assert not out.exists()


@pytest.mark.parametrize("cmd", ["up", "upload", "functions"])
def test_deploy_refuses_unpriced_region_before_creating_anything(tmp_path, monkeypatch, cmd):
    import deploy

    monkeypatch.setattr(
        deploy.boto3, "Session", lambda **kw: pytest.fail("AWS session for an unpriced Region")
    )
    with pytest.raises(SystemExit, match="no pinned prices for us-east-1"):
        deploy.main([cmd, "--build", str(tmp_path), "--region", "us-east-1", "--profile", "p"])


@pytest.mark.parametrize("latency", ["NaN", "Infinity", "-Infinity"])
def test_legacy_row_with_non_finite_latency_is_revalidated_as_error(tmp_path, latency):
    """A row bench.py recorded as a success before it checked finiteness becomes an error on reload, stays out of
    the distributions, and the result is still strict JSON."""
    import test_lkai_baseline as t

    rows = [t._row("inst", 0, "warm"), t._row("inst", 0, "warm", rid="r2")]
    run = t._run_dir(_mk(tmp_path / "run"), rows)
    lines = (Path(run) / "invocations.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"latency_ms": 500.0', f'"latency_ms": {latency}')
    assert latency in lines[1]
    (Path(run) / "invocations.jsonl").write_text("\n".join(lines) + "\n")
    ref = tmp_path / "ref.json"
    assert summarize.main([run, "--write-reference", str(ref)]) == 3
    text = (Path(run) / "results.json").read_text()
    res = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON {c}"))
    assert [e["error"] for e in res["errors"]["rows"]] == ["malformed:no-latency"]
    assert res["phases"]["inst"]["warm"]["n"] == 1
    assert res["correctness"]["without_answers"] == 1


def test_answer_with_non_finite_number_is_not_evidence():
    for v in (float("nan"), float("inf")):
        p = {"answers": {"q": {"noul": v}}, "latency_ms": 1.0}
        assert summarize.answer_problem(p) == "answer-not-finite"
        nested = {"answers": {"q": {"probabilities": {"a": v}}}, "latency_ms": 1.0}
        assert summarize.answer_problem(nested) == "answer-not-finite"


def test_dist_refuses_non_finite_samples():
    with pytest.raises(ValueError, match="non-finite"):
        summarize.dist([1.0, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        summarize.dist([1.0, float("inf")])


# ------------------------------------------------------------------ 4. cleanup attempts every version


def test_round_cleanup_deletes_every_version_after_first_failure(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    attempts = []

    def delete_version(self, arm, v):
        attempts.append((arm, v))
        if arm == "inst":
            raise RuntimeError("throttled")

    fake = bench.Bench
    fake.delete_version = delete_version
    fake.save_logs = lambda self, t0, t1: 0
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "--out", str(out), "--rounds", "2", "--warm", "5", "--log-lag", "0"],
    )
    with pytest.raises(bench.CleanupError, match="inst:7"):
        bench.main()
    assert sorted(attempts) == [("inst", "7"), ("rep", "7")]
    cond = json.loads((out / "conditions.json").read_text())
    assert cond["rounds_meta"][0]["cleanup_failures"][0].startswith("inst:7")


def test_round_failure_survives_cleanup_failure(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    attempts = []

    def delete_version(self, arm, v):
        attempts.append(arm)
        raise RuntimeError("delete failed")

    def wave(self, arm, qualifier, rnd, wave, **kw):
        if wave == "restore":
            raise KeyError("benchmark failed")
        return [{"error": None, "rt_s": 1.0}]

    fake = bench.Bench
    fake.delete_version, fake.wave = delete_version, wave
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(tmp_path / "run"), "--rounds", "2"])
    with pytest.raises(KeyError, match="benchmark failed"):
        bench.main()
    assert sorted(attempts) == ["inst", "rep"]


def test_delete_versions_ignores_already_deleted():
    import bench

    calls = []

    class B:
        def delete_version(self, arm, v):
            calls.append(arm)
            if arm == "a":
                raise ResourceNotFound()
            if arm == "b":
                raise RuntimeError("x")

    assert bench.delete_versions(
        B(), {"a": {"version": "1"}, "b": {"version": "2"}, "c": {"version": "3"}}
    ) == ["b:2: RuntimeError: x"]
    assert calls == ["a", "b", "c"]


def _teardown_session(fail_on: set, deleted: list, versions: dict):
    class Pager:
        def __init__(self, fn):
            self.fn = fn

        def paginate(self, FunctionName):
            if ("list", FunctionName) in fail_on:
                raise RuntimeError("list failed")
            return [
                {
                    "Versions": [{"Version": "$LATEST"}]
                    + [{"Version": v} for v in versions[FunctionName]]
                }
            ]

    class Lam:
        class exceptions:
            ResourceNotFoundException = ResourceNotFound

        def get_paginator(self, name):
            return Pager(name)

        def delete_function(self, FunctionName, Qualifier=None):
            key = (FunctionName, Qualifier)
            deleted.append(key)
            if key in fail_on:
                raise RuntimeError(f"delete {key} failed")

    class Logs:
        class exceptions:
            ResourceNotFoundException = ResourceNotFound

        def delete_log_group(self, logGroupName):
            deleted.append(("log", logGroupName))
            if ("log", logGroupName) in fail_on:
                raise RuntimeError(f"delete {logGroupName} failed")

    class Iam:
        class exceptions:
            NoSuchEntityException = NoSuchEntity

        def delete_role_policy(self, **kw):
            deleted.append(("policy", kw["RoleName"]))

        def delete_role(self, **kw):
            deleted.append(("role", kw["RoleName"]))

    class Bucket:
        class objects:
            @staticmethod
            def all():
                return types.SimpleNamespace(delete=lambda: [{"Deleted": []}])

        def delete(self):
            deleted.append(("bucket", "b"))

    class Sess:
        def client(self, name):
            return {"lambda": Lam(), "logs": Logs(), "iam": Iam(), "sts": Sts()}[name]

        def resource(self, name):
            return types.SimpleNamespace(Bucket=lambda n: Bucket())

    return Sess()


def test_teardown_attempts_every_version_and_function_after_failures():
    import deploy

    deleted: list = []
    versions = {"f-inst": ["1", "2", "3"], "f-rep": ["4", "5"]}
    sess = _teardown_session({("f-inst", "1"), ("f-inst", None)}, deleted, versions)
    st = {
        "prefix": "f",
        "bucket": "b",
        "functions": {a: {"name": f"f-{a}"} for a in ("inst", "rep")},
    }
    failures = deploy.teardown(sess, st, everything=True)
    for name, vs in versions.items():
        for v in vs:
            assert (name, v) in deleted
        assert (name, None) in deleted
    assert ("role", "f-fn") in deleted and ("bucket", "b") in deleted
    assert ("log", "/aws/lambda/f-inst") in deleted
    assert ("log", "/aws/lambda/f-rep") in deleted
    assert len(failures) == 2 and any("f-inst:1" in f for f in failures)


def test_teardown_list_failure_does_not_stop_other_functions():
    import deploy

    deleted: list = []
    sess = _teardown_session({("list", "f-inst")}, deleted, {"f-inst": ["1"], "f-rep": ["4"]})
    st = {"prefix": "f", "functions": {a: {"name": f"f-{a}"} for a in ("inst", "rep")}}
    failures = deploy.teardown(sess, st, everything=False)
    assert ("f-rep", "4") in deleted and failures == [
        "list versions f-inst: RuntimeError: list failed"
    ]


def test_teardown_log_failure_does_not_stop_role_and_bucket_cleanup():
    import deploy

    deleted: list = []
    fail = {("log", "/aws/lambda/f-inst")}
    sess = _teardown_session(fail, deleted, {"f-inst": [], "f-rep": []})
    st = {
        "prefix": "f",
        "bucket": "b",
        "functions": {a: {"name": f"f-{a}"} for a in ("inst", "rep")},
    }
    failures = deploy.teardown(sess, st, everything=True)
    assert any("delete log group /aws/lambda/f-inst" in f for f in failures)
    assert ("log", "/aws/lambda/f-rep") in deleted
    assert ("role", "f-fn") in deleted and ("bucket", "b") in deleted


def test_teardown_cli_exits_nonzero_naming_failures(tmp_path, monkeypatch):
    import deploy

    deleted: list = []
    sess = _teardown_session({("f-inst", "1")}, deleted, {"f-inst": ["1", "2"], "f-rep": []})
    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: sess)
    deploy.save_state(
        str(tmp_path),
        {
            "prefix": "f",
            "region": "us-east-2",
            "functions": {a: {"name": f"f-{a}"} for a in ("inst", "rep")},
        },
    )
    with pytest.raises(SystemExit, match="1 step"):
        deploy.main(["teardown", "--build", str(tmp_path), "--profile", "p"])
    assert ("f-inst", "2") in deleted


# ------------------------------------------------------------------ run.sh, end to end with stand-ins


def _stub_run(
    tmp_path,
    bench_rc=0,
    teardown_rc=0,
    summarize_rc=0,
    memory=None,
    profile="lkai-test",
    dirty="",
    sts_rc=0,
):
    """run.sh copied beside stand-in build/deploy/bench/summarize/redact scripts; returns (exit code, call log)."""
    d = tmp_path / "baseline"
    d.mkdir()
    (d / "run.sh").write_text((BASE / "run.sh").read_text())
    log = tmp_path / "calls.log"
    stub = (
        'import sys; open({log!r}, "a").write("{name} " + " ".join(sys.argv[1:]) + "\\n"); {body}'
    )
    files = {
        "deploy.py": f"sys.exit({teardown_rc} if sys.argv[1:2] == ['teardown'] else 0)",
        "bench.py": f"sys.exit({bench_rc})",
        "summarize.py": f"sys.exit({summarize_rc})",
        "redact.py": "import os; os.makedirs(sys.argv[2], exist_ok=True) if not sys.argv[1].endswith('.json') else None",
        # `dirty --check` prints the dirty inputs and exits 1 when there are any.
        "buildcache.py": f"print({dirty!r}, end='') if {dirty!r} else None; sys.exit(1 if {dirty!r} else 0)",
    }
    for name, body in files.items():
        (d / name).write_text(stub.format(log=str(log), name=name, body=body))
    (d / "build.sh").write_text(f"#!/bin/bash\necho build.sh >> {log}\n")
    (d / "meta.json").write_text("{}")
    (d / "build").mkdir()
    (d / "build" / "deployment.json").write_text('{"bucket": "b"}')
    (d / "build" / "manifest.json").write_text("{}")
    (d / "reference").mkdir()
    (d / "reference" / "outputs.json").write_text("{}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "py"
    py.write_text(
        '#!/bin/bash\n[ "$1" = -c ] && [ "$2" = "import boto3" ] && exit 0\nexec '
        + sys.executable
        + ' "$@"\n'
    )
    aws = bindir / "aws"
    aws.write_text(
        f'#!/bin/bash\necho "aws $*" >> {log}\n[ {sts_rc} -eq 0 ] || exit {sts_rc}\necho 123456789012\n'
    )
    for f in (d / "run.sh", d / "build.sh", py, aws):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PYTHON": str(py), "PATH": f"{bindir}:{os.environ['PATH']}"}
    env.pop("AWS_PROFILE", None)
    if profile is not None:
        env["AWS_PROFILE"] = profile
    if memory is not None:
        env["MEMORY"] = memory
    p = subprocess.run([str(d / "run.sh"), "rid"], env=env, capture_output=True, text=True)
    return p.returncode, (log.read_text() if log.exists() else ""), p.stderr


def test_run_sh_passes_memory_to_deploy_and_bench(tmp_path):
    rc, calls, err = _stub_run(tmp_path, memory="4096")
    assert rc == 0, err
    assert "deploy.py up --region us-east-2 --prefix lkai-p1 --memory 4096" in calls
    assert "bench.py --region us-east-2 --profile lkai-test --memory 4096 " in calls
    assert calls.count("deploy.py teardown") == 1


@pytest.mark.parametrize("memory", ["64", "20000", "4k", ""])
def test_run_sh_rejects_bad_memory_before_building(tmp_path, memory):
    rc, calls, err = _stub_run(tmp_path, memory=memory)
    if memory == "":  # unset-equivalent: the default
        assert rc == 0, err
        assert "--memory 10240" in calls
        return
    assert rc == 2 and "MEMORY must be" in err and calls == ""


def test_run_sh_keeps_benchmark_failure_status_when_cleanup_fails(tmp_path):
    rc, calls, err = _stub_run(tmp_path, bench_rc=5, teardown_rc=1)
    assert rc == 5
    assert "deploy.py teardown" in calls and "CLEANUP FAILED" in err


def test_run_sh_fails_successful_run_whose_cleanup_fails(tmp_path):
    rc, _calls, err = _stub_run(tmp_path, teardown_rc=1)
    assert rc == 1 and "CLEANUP FAILED" in err


def test_run_sh_publishes_invalid_baseline_but_exits_nonzero(tmp_path):
    rc, calls, err = _stub_run(tmp_path, summarize_rc=3)
    assert rc == 3 and "not a valid baseline" in err
    assert "redact.py" in calls and "deploy.py teardown" in calls
