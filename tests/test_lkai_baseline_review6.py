"""examples/lambda-kleidiai/baseline review-6 fixes: explicit AWS identity before spend, the whole fixed request set,
both overhead arms, and build inputs bound to the commit that names them. No AWS, Docker or model needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / "examples" / "lambda-kleidiai" / "baseline"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import buildcache  # noqa: E402
import summarize  # noqa: E402
import test_lkai_baseline as t  # noqa: E402
import test_lkai_baseline_rework as rw  # noqa: E402
from test_lkai_baseline_rework import (  # noqa: E402
    _aws_sdk_stand_ins,  # noqa: F401  (autouse fixture)
    _bench_main_fixture,
    _stub_run,
)
from test_lkai_baseline_rework import clean_harness as clean_harness  # noqa: E402


def _mk(p: Path) -> Path:
    p.mkdir(parents=True)
    return p


def _good_rows():
    return rw._all_class_rows()


def _summarize(tmp_path, rows, edit_cond=None):
    run = t._run_dir(_mk(tmp_path / "run"), rows)
    if edit_cond:
        p = Path(run) / "conditions.json"
        cond = json.loads(p.read_text())
        edit_cond(cond)
        p.write_text(json.dumps(cond))
    ref = tmp_path / "ref.json"
    rc = summarize.main([run, "--write-reference", str(ref)])
    return rc, ref, json.loads((Path(run) / "results.json").read_text())


def test_fixture_baseline_is_valid(tmp_path):
    rc, ref, res = _summarize(tmp_path, _good_rows())
    assert rc == 0 and ref.exists() and res["status"] == "baseline"
    assert res["request_coverage"]["inst"]["missing"] == []


# ------------------------------------------------------------------ 1. explicit identity before any spend


def test_run_sh_requires_aws_profile_before_anything(tmp_path):
    rc, calls, err = _stub_run(tmp_path, profile=None)
    assert rc == 2 and "AWS_PROFILE is required" in err
    assert calls == ""  # no build, no deploy, no sts


def test_run_sh_resolves_identity_before_build_and_passes_profile_everywhere(tmp_path):
    rc, calls, err = _stub_run(tmp_path)
    assert rc == 0, err
    lines = calls.splitlines()
    sts = next(i for i, c in enumerate(lines) if c.startswith("aws sts get-caller-identity"))
    assert "--profile lkai-test" in lines[sts]
    assert sts < lines.index("build.sh")
    assert sum(c.startswith("aws ") for c in lines) == 1  # redaction reuses the preflight's account
    for c in lines:
        if c.startswith(("deploy.py", "bench.py")):
            assert "--profile lkai-test" in c, c
        if c.startswith("deploy.py"):
            assert "--expect-account 123456789012" in c, c
    assert any("--account 123456789012" in c for c in lines if c.startswith("redact.py"))


def test_run_sh_stops_when_identity_cannot_be_resolved(tmp_path):
    rc, calls, err = _stub_run(tmp_path, sts_rc=255)
    assert rc == 2 and "could not resolve AWS_PROFILE" in err
    assert "build.sh" not in calls and "deploy.py" not in calls


@pytest.mark.parametrize("cmd", ["up", "upload", "functions", "teardown"])
def test_deploy_requires_explicit_profile(tmp_path, monkeypatch, capsys, cmd):
    import deploy

    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: pytest.fail("implicit session"))
    with pytest.raises(SystemExit) as e:
        deploy.main([cmd, "--build", str(tmp_path)])
    assert e.value.code == 2 and "--profile or AWS_PROFILE is required" in capsys.readouterr().err


def test_bench_requires_explicit_profile(tmp_path, monkeypatch, capsys):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2 and "AWS_PROFILE is required" in capsys.readouterr().err
    assert not out.exists()


def test_check_account_refuses_preflight_or_state_mismatch():
    import deploy

    class Sess:
        def client(self, name):
            assert name == "sts"
            return rw.Sts("111111111111")

    st: dict = {}
    assert deploy.check_account(Sess(), st, "111111111111") == "111111111111"
    assert st["account"] == "111111111111"
    with pytest.raises(SystemExit, match="different account than the preflight"):
        deploy.check_account(Sess(), {}, "222222222222")
    with pytest.raises(SystemExit, match="deployed in another account"):
        deploy.check_account(Sess(), {"account": "222222222222", "prefix": "p"})


def test_deploy_checks_account_before_any_resource_call(tmp_path, monkeypatch, clean_harness):
    import deploy

    calls = []

    class Sess:
        region_name = "us-east-2"

        def client(self, name):
            calls.append(name)
            if name != "sts":
                pytest.fail(f"{name} client before the account check")
            return rw.Sts("111111111111")

    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: Sess())
    rw._clean_manifest(tmp_path)
    with pytest.raises(SystemExit, match="different account"):
        deploy.main(
            ["up", "--build", str(tmp_path), "--profile", "p", "--expect-account", "999999999999"]
        )
    assert calls == ["sts"]


def test_show_never_prints_account(tmp_path, capsys):
    import deploy

    deploy.save_state(str(tmp_path), {"region": "us-east-2", "account": rw.ACCOUNT})
    deploy.main(["show", "--build", str(tmp_path)])
    assert rw.ACCOUNT not in capsys.readouterr().out


# ------------------------------------------------------------------ 2. the whole fixed request set


def test_bench_refuses_fewer_warm_passes_than_requests(tmp_path, monkeypatch, capsys):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out), "--warm", "4"])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2 and "fewer than the 5 fixed requests" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("reqs", "why"),
    [
        ({"probe": "a", "requests": []}, "no requests"),
        ({"probe": "a", "requests": [{"request": {}}]}, "string id"),
        ({"probe": "a", "requests": [{"id": "a", "request": {}}] * 2}, "duplicate"),
        ({"probe": "a", "requests": [{"id": "a"}]}, "request body"),
        ({"probe": "z", "requests": [{"id": "a", "request": {}}]}, "probe"),
    ],
)
def test_request_set_problem(reqs, why):
    assert why in summarize.request_set_problem(reqs)


def test_committed_request_set_is_valid():
    reqs = json.loads((BASE / "requests.json").read_text())
    assert summarize.request_set_problem(reqs) is None
    assert summarize.warm_problem(len(reqs["requests"]), reqs) is None
    assert summarize.warm_problem(len(reqs["requests"]) - 1, reqs)


def test_partial_request_coverage_is_not_a_baseline(tmp_path):
    rows = [r for r in _good_rows() if r["request_id"] != "r2"]
    rc, ref, res = _summarize(tmp_path, rows)
    assert res["correctness"]["complete"] is True  # every present row compared: the old gate passed
    assert rc == 3 and not ref.exists() and res["status"] == "failed"
    why = res["baseline_invalid"]
    assert "arm inst has no successful answer for request(s) ['r2']" in why
    assert "arm rep has no successful answer for request(s) ['r2']" in why


def test_one_arm_missing_one_request_is_not_a_baseline(tmp_path):
    rows = [r for r in _good_rows() if not (r["arm"] == "rep" and r["request_id"] == "r2")]
    rc, _ref, res = _summarize(tmp_path, rows)
    assert rc == 3 and res["baseline_invalid"] == [
        "arm rep has no successful answer for request(s) ['r2']"
    ]


def test_warm_condition_below_request_count_is_not_a_baseline(tmp_path):
    rc, _ref, res = _summarize(tmp_path, _good_rows(), lambda c: c.update(warm_per_arm_per_round=1))
    assert rc == 3
    assert "1 warm pass(es) per round cannot cover the 2 fixed requests" in res["baseline_invalid"]


@pytest.mark.parametrize("rid", [None, "", "not-declared"])
def test_legacy_row_without_a_declared_request_id_is_an_error(tmp_path, rid):
    rows = _good_rows()
    extra = t._row("inst", 0, "warm")
    if rid is None:
        del extra["request_id"]
    else:
        extra["request_id"] = rid
    rc, _ref, res = _summarize(tmp_path, [*rows, extra])
    want = "unknown-request-id" if rid else "no-request-id"
    assert rc == 3 and [e["error"] for e in res["errors"]["rows"]] == [f"malformed:{want}"]


def test_legacy_conditions_without_request_bodies_are_not_a_baseline(tmp_path):
    rc, _ref, res = _summarize(
        tmp_path,
        _good_rows(),
        lambda c: c.update(requests={"requests": [{"id": "r1"}, {"id": "r2"}]}),
    )
    assert rc == 3 and "conditions record no fixed request ids" in res["baseline_invalid"]


# ------------------------------------------------------------------ 3. both overhead arms


@pytest.mark.parametrize("arms", ["inst", "rep", "inst,inst", "inst,rep,inst", "inst,ctl", ""])
def test_bench_refuses_any_arm_set_but_inst_and_rep(tmp_path, monkeypatch, capsys, arms):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out), "--arms", arms])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2 and "arms must be exactly" in capsys.readouterr().err
    assert not out.exists()


def test_bench_accepts_either_arm_order(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["bench.py", "--out", str(out), "--arms", "rep,inst", "--log-lag", "0"]
    )
    bench.Bench.save_logs = lambda self, t0, t1: 0
    bench.main()
    assert json.loads((out / "conditions.json").read_text())["arms"] == ["rep", "inst"]


def test_bench_refuses_an_arm_deployed_with_the_wrong_phase_flag(tmp_path, monkeypatch):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    fake = bench.Bench(None, None, None, None, None).lam
    orig = type(fake).get_function_configuration

    def cfg(self, FunctionName):
        c = orig(self, FunctionName)
        c["Environment"]["Variables"]["LKAI_PHASES"] = "1"  # rep instrumented too: no control
        return c

    monkeypatch.setattr(type(fake), "get_function_configuration", cfg)
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(tmp_path / "run")])
    with pytest.raises(SystemExit, match="rep: deployed LKAI_PHASES='1'"):
        bench.main()


def test_one_arm_run_is_not_a_baseline(tmp_path):
    rows = [r for r in _good_rows() if r["arm"] == "inst"]

    def one_arm(c):  # what bench.py --arms inst would have recorded
        c["arms"] = ["inst"]
        del c["functions"]["rep"]
        for m in c["rounds_meta"]:
            m["published"].pop("rep", None)

    rc, ref, res = _summarize(tmp_path, rows, one_arm)
    assert res["correctness"]["complete"] is True and res["errors"]["count"] == 0
    assert rc == 3 and not ref.exists() and res["status"] == "failed"
    assert any("arms must be exactly ['inst', 'rep']" in w for w in res["baseline_invalid"])
    assert any("overhead not measured" in w for w in res["baseline_invalid"])


def test_persisted_arm_with_wrong_phase_flag_is_not_a_baseline(tmp_path):
    rc, _ref, res = _summarize(
        tmp_path,
        _good_rows(),
        lambda c: c["functions"]["rep"]["environment"].update(LKAI_PHASES="1"),
    )
    assert rc == 3
    assert "arm rep was deployed with LKAI_PHASES='1', needs '0'" in res["baseline_invalid"]


def test_arms_on_disjoint_cpu_parts_measure_no_overhead(tmp_path):
    rows = _good_rows()
    for r in rows:
        if r["arm"] == "rep":
            r["cpu_part"] = "0xd0c"
    rc, _ref, res = _summarize(tmp_path, rows)
    assert rc == 3
    assert any("overhead not measured for cold/restore/warm" in w for w in res["baseline_invalid"])


# ------------------------------------------------------------------ 4. inputs bound to the recorded commit


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A git repo holding every stage input and harness script, committed clean."""
    r, _build = rw._repo(tmp_path)
    for p in buildcache.UNKEYED_SOURCES:
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(f"{p}\n")
    (r / ".gitignore").write_text("*.local\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    assert buildcache.dirty(str(r)) == []
    return r


@pytest.mark.parametrize(
    ("change", "want"),
    [
        (
            lambda r: (r / "src/strands_decider/a.py").write_text("edited\n"),
            "modified src/strands_decider/a.py",
        ),
        (
            lambda r: (r / "src/strands_decider/new.py").write_text("x\n"),
            "untracked src/strands_decider/new.py",
        ),
        (
            lambda r: (r / "examples/lambda/app/x.local").write_text("x\n"),
            "untracked examples/lambda/app/x.local",
        ),
        (lambda r: (r / "examples/lambda/app/a.py").unlink(), "deleted examples/lambda/app/a.py"),
        (
            lambda r: (r / "examples/lambda-kleidiai/llama/patch_llama.py").chmod(0o755),
            "mode examples/lambda-kleidiai/llama/patch_llama.py",
        ),
        (
            lambda r: (r / "examples/lambda-kleidiai/baseline/deploy.py").write_text("edited\n"),
            "modified examples/lambda-kleidiai/baseline/deploy.py",
        ),
        (
            lambda r: (r / "examples/lambda-kleidiai/baseline/requests.json").write_text("{}\n"),
            "modified examples/lambda-kleidiai/baseline/requests.json",
        ),
    ],
    ids=[
        "tracked-edit",
        "untracked",
        "ignored-untracked",
        "deleted",
        "exec-bit",
        "deploy-script",
        "request-set",
    ],
)
def test_dirty_finds_every_kind_of_input_change(repo, change, want):
    change(repo)
    assert buildcache.dirty(str(repo)) == [want]


def test_dirty_ignores_files_outside_the_inputs(repo):
    (repo / "README.md").write_text("doc edit\n")
    (repo / "notes.txt").write_text("scratch\n")
    (repo / "examples/lambda-kleidiai/baseline/README.md").write_text("doc\n")
    assert buildcache.dirty(str(repo)) == []


def test_dirty_covers_exactly_what_the_keys_hash(repo, tmp_path):
    """Any change that alters a stage key is reported dirty: the two walk the same file list."""
    build = tmp_path / "build"
    base = rw._keys(repo, build)
    (repo / "examples/lambda-kleidiai/snapstart/new.py").write_text("x\n")
    assert rw._keys(repo, build) != base
    assert buildcache.dirty(str(repo)) == ["untracked examples/lambda-kleidiai/snapstart/new.py"]


def test_dirty_check_cli_exit_status(repo, capsys):
    assert buildcache.main(["dirty", "--repo", str(repo), "--check"]) == 0
    (repo / "src/strands_decider/new.py").write_text("x\n")
    assert buildcache.main(["dirty", "--repo", str(repo), "--check"]) == 1
    assert "untracked src/strands_decider/new.py" in capsys.readouterr().out
    assert buildcache.main(["dirty", "--repo", str(repo)]) == 0  # listing only


def test_harness_provenance_reports_commit_and_dirty_scripts(repo):
    h = buildcache.harness_provenance(str(repo))
    assert len(h["commit"]) == 40 and h["dirty_inputs"] == []
    (repo / "examples/lambda-kleidiai/baseline/bench.py").write_text("edited\n")
    (repo / "src/strands_decider/a.py").write_text("edited\n")  # a stage input, not harness
    assert buildcache.harness_provenance(str(repo))["dirty_inputs"] == [
        "modified examples/lambda-kleidiai/baseline/bench.py"
    ]


def test_build_script_refuses_dirty_inputs_and_records_the_complete_list():
    s = (BASE / "build.sh").read_text()
    assert "--untracked-files=no" not in s
    refuse = s.index('dirty --repo "$REPO" --check')
    assert refuse < s.index("docker pull")  # before any work
    assert '"schema": "lkai-build-manifest/2"' in s
    assert '"dirty_inputs": dirty_inputs' in s and '"repo_dirty": bool(dirty_inputs)' in s


def test_run_sh_refuses_dirty_inputs_before_identity_or_build(tmp_path):
    rc, calls, err = _stub_run(tmp_path, dirty="untracked src/strands_decider/x.py")
    assert rc == 2 and "untracked src/strands_decider/x.py" in err
    assert "build.sh" not in calls and "deploy.py" not in calls and "aws " not in calls


@pytest.mark.parametrize(
    ("manifest", "why"),
    [
        (
            {"schema": "lkai-build-manifest/2", "dirty_inputs": ["untracked a.py"]},
            "differ from commit",
        ),
        ({"schema": "lkai-build-manifest/2", "repo_dirty": True, "dirty_inputs": []}, "repo_dirty"),
        ({"schema": "lkai-build-manifest/2"}, "no dirty_inputs"),
        ({"schema": "lkai-build-manifest/1", "repo_dirty": False}, "predates"),
        ({}, "predates"),
    ],
)
def test_deploy_refuses_a_build_that_is_not_its_commit(
    tmp_path, monkeypatch, clean_harness, manifest, why
):
    import deploy

    monkeypatch.setattr(
        deploy.boto3, "Session", lambda **kw: pytest.fail("session for a dirty build")
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match=why):
        deploy.main(["up", "--build", str(tmp_path), "--profile", "p"])


def test_deploy_refuses_dirty_harness(tmp_path, monkeypatch):
    import deploy

    monkeypatch.setattr(
        buildcache,
        "harness_provenance",
        lambda repo=None: {"commit": "abc", "dirty_inputs": ["modified examples/x/deploy.py"]},
    )
    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: pytest.fail("session"))
    rw._clean_manifest(tmp_path)
    with pytest.raises(SystemExit, match="deploy/harness files differ"):
        deploy.main(["up", "--build", str(tmp_path), "--profile", "p"])


def test_teardown_works_for_a_dirty_build(tmp_path, monkeypatch):
    """Cleanup is never blocked by provenance: a dirty build's billing versions must still be deletable."""
    import deploy

    deleted: list = []
    sess = rw._teardown_session(set(), deleted, {"f-inst": ["1"], "f-rep": []})
    monkeypatch.setattr(deploy.boto3, "Session", lambda **kw: sess)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "lkai-build-manifest/1", "repo_dirty": True})
    )
    deploy.save_state(
        str(tmp_path),
        {
            "prefix": "f",
            "region": "us-east-2",
            "functions": {a: {"name": f"f-{a}"} for a in ("inst", "rep")},
        },
    )
    deploy.main(["teardown", "--build", str(tmp_path), "--profile", "p"])
    assert ("f-inst", "1") in deleted


@pytest.mark.parametrize(
    ("dirty_inputs", "harness_dirty"), [(["untracked src/x.py"], []), ([], ["modified bench.py"])]
)
def test_bench_refuses_dirty_build_or_harness(tmp_path, monkeypatch, dirty_inputs, harness_dirty):
    bench = _bench_main_fixture(tmp_path, monkeypatch, deployed_memory=10240)
    st = bench.deploy.load_state(None)
    st["manifest"]["dirty_inputs"] = dirty_inputs
    monkeypatch.setattr(
        bench.buildcache,
        "harness_provenance",
        lambda: {"commit": "abc", "dirty_inputs": harness_dirty},
    )
    monkeypatch.setattr(bench.boto3, "Session", lambda **kw: pytest.fail("session"))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(out)])
    with pytest.raises(SystemExit, match="refusing to benchmark"):
        bench.main()
    assert not out.exists()


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (
            lambda c: c["build_manifest"].update(dirty_inputs=["untracked src/x.py"]),
            "differ from commit",
        ),
        (lambda c: c["build_manifest"].update(repo_dirty=True), "repo_dirty is true"),
        (lambda c: c["build_manifest"].pop("dirty_inputs"), "no dirty_inputs"),
        (lambda c: c["harness"].update(dirty_inputs=["modified bench.py"]), "harness files differ"),
        (lambda c: c.pop("harness"), "no harness provenance"),
    ],
)
def test_summarize_refuses_dirty_provenance_as_baseline(tmp_path, edit, why):
    rc, ref, res = _summarize(tmp_path, _good_rows(), edit)
    assert rc == 3 and not ref.exists() and res["status"] == "failed"
    assert any(why in w for w in res["baseline_invalid"])


def test_legacy_manifest_records_its_weaker_check(tmp_path):
    """A /1 manifest (the published run) checked tracked files only; the result says so instead of passing it off as
    the complete check."""

    def legacy(c):
        c["build_manifest"] = {
            k: v for k, v in c["build_manifest"].items() if k not in ("schema", "dirty_inputs")
        } | {"schema": "lkai-build-manifest/1"}
        c.pop("harness")

    rc, _ref, res = _summarize(tmp_path, _good_rows(), legacy)
    assert rc == 0
    assert res["commits"]["provenance_check"] == "tracked files only (lkai-build-manifest/1)"
    assert res["commits"]["dirty_inputs"] is None


def test_invalid_baseline_is_published_as_failed_and_passes_p2_status_rules(tmp_path):
    rows = [r for r in _good_rows() if r["arm"] == "inst"]

    def one_arm(c):  # what bench.py --arms inst would have recorded
        c["arms"] = ["inst"]
        del c["functions"]["rep"]
        for m in c["rounds_meta"]:
            m["published"].pop("rep", None)

    _rc, _ref, res = _summarize(tmp_path, rows, one_arm)
    assert res["status"] == "failed"
    assert res["changed_variable"] and res["conclusion"].startswith("Not a valid baseline: ")


def test_published_baseline_still_valid_under_new_gates():
    run = BASE / "results" / "20261020T080835Z"
    meta = json.loads((run / "meta.json").read_text())
    res = summarize.summarize(str(run), str(BASE / "reference" / "outputs.json"), None, meta)
    assert summarize.incomplete(res) == []
    cov = res["request_coverage"]
    ids = json.loads((BASE / "requests.json").read_text())
    assert {a: c["missing"] for a, c in cov.items()} == {"inst": [], "rep": []}
    assert set(cov["inst"]["answered"]) == {r["id"] for r in ids["requests"]}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
