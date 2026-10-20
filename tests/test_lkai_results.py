"""examples/lambda-kleidiai/results: the lkai-result/1 validator, the publisher and the dashboard renderer.

No AWS: the store is a LocalStore directory, which has the same key layout as the S3 store. The tests pin what the
experiment cards rely on: a valid run validates and a broken one names its problem; publishing writes only the
run's own prefix, is idempotent, and refuses to silently overwrite another run; the dashboard is self-contained
and shows failed and null results explicitly; fixtures are labelled and give way to real runs.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import re
import sys
import threading

import pytest

RESULTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "results")
)
sys.path.insert(0, RESULTS)

import lkai_results as lr  # noqa: E402
import publish_result  # noqa: E402

FIXTURES = sorted(glob.glob(os.path.join(RESULTS, "fixtures", "*", "*", "results.json")))


def load(name: str) -> dict:
    (path,) = [p for p in FIXTURES if f"/{name}/" in p]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fixture_dir(name: str) -> str:
    (path,) = [p for p in FIXTURES if f"/{name}/" in p]
    return os.path.dirname(path)


def real(res: dict, run_id: str | None = None) -> dict:
    r = copy.deepcopy(res)
    r.pop("fixture", None)
    if run_id:
        r["run_id"] = run_id
    return r


class FakeMcpProc:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.stdin = self
        self.stdout = self
        self.returncode = None
        self.writes: list[dict] = []

    def write(self, data: str) -> None:
        for line in data.splitlines():
            if line:
                self.writes.append(json.loads(line))

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        return json.dumps(self.responses.pop(0)) + "\n"

    def fileno(self) -> int:
        return 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout=None) -> int:
        return self.returncode or 0




def test_schema_file_is_json_and_names_every_brief_field():
    s = lr.load_schema()
    props = set(s["properties"])
    for k in (
        "hypothesis",
        "changed_variable",
        "commits",
        "deployment",
        "model",
        "conditions",
        "classification",
        "phases",
        "errors",
        "cost",
        "correctness",
        "profiling_artifacts",
        "artifacts",
        "conclusion",
        "reproduce",
        "status",
    ):
        assert k in props, k


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.split("fixtures/")[1])
def test_committed_fixtures_validate_and_are_labelled(path):
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    assert lr.validate(res) == []
    assert res["fixture"] is True


def test_fixtures_cover_baseline_improved_null_and_failed():
    assert {json.load(open(p))["status"] for p in FIXTURES} >= {
        "baseline",
        "improved",
        "no-change",
        "failed",
    }


def test_p1_summarize_output_shape_validates():
    """baseline/summarize.py's own block: same dist() keys, nested snapshot_init, correctness null on the
    reference-writing run. A baseline without a reference is still valid (correctness null)."""
    r = real(load("fixture-baseline"))
    r["correctness"] = None
    r["phases"]["inst"]["restore"]["snapshot_init"]["imports"] = None
    r["variance"] = {"inst": {"cold": {"per_round_median_s": [27.1, 27.4]}}}
    r["instrumentation_overhead"] = {}
    assert lr.validate(r) == []


def test_schema_requires_complete_measurement_contract():
    base = real(load("fixture-improved"))
    cases = [
        (lambda r: r["changed_variable"].pop("baseline"), "matches no allowed shape"),
        (lambda r: r["deployment"].pop("az"), "missing 'az'"),
        (lambda r: r["model"].pop("revision"), "missing 'revision'"),
        (lambda r: r["conditions"].pop("request_ids"), "fixed request id"),
        (lambda r: r["conditions"].pop("finished"), "started and finished"),
        (
            lambda r: r["phases"]["inst"].pop("warm"),
            "phases.inst: missing 'warm' start class",
        ),
        (
            lambda r: r["phases"]["inst"]["cold"].pop("model_ready"),
            "phases.inst.cold: missing required phase 'model_ready'",
        ),
        (
            lambda r: r["phases"]["inst"]["cold"].pop("report_max_memory_mb"),
            "phases.inst.cold: report_max_memory_mb MB distribution is required",
        ),
        (lambda r: r["cost"].pop("s3_request_usd"), "missing 's3_request_usd'"),
        (lambda r: r.update(artifacts={}), "needs at least 1"),
    ]
    for mutate, expected in cases:
        r = copy.deepcopy(base)
        mutate(r)
        assert any(expected in p for p in lr.validate(r)), (expected, lr.validate(r))


def test_measurement_contract_is_required_per_arm_and_start_class():
    r = real(load("fixture-improved"))
    r["phases"]["other"] = copy.deepcopy(r["phases"]["inst"])
    del r["phases"]["other"]["restore"]["report_restore_duration"]
    del r["phases"]["other"]["cold"]["report_max_memory_mb"]
    problems = lr.validate(r)
    assert any(
        "phases.other.restore: missing required phase 'report_restore_duration'" in p
        for p in problems
    )
    assert any(
        "phases.other.cold: report_max_memory_mb MB distribution is required" in p
        for p in problems
    )


def test_validator_rejects_non_finite_metrics():
    r = real(load("fixture-improved"))
    r["phases"]["inst"]["cold"]["weight_download"]["p50"] = float("nan")
    r["cost"]["total_usd"] = float("inf")
    problems = lr.validate(r)
    assert any("p50: number must be finite" in p for p in problems)
    assert any("total_usd: number must be finite" in p for p in problems)


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        (lambda r: r.pop("hypothesis"), "missing 'hypothesis'"),
        (lambda r: r.update(status="great"), "not one of"),
        (lambda r: r["commits"].update(experiment="not-a-sha"), "does not match"),
        (lambda r: r.update(reproduce=[]), "at least 1"),
        (lambda r: r.update(changed_variable=None), "expected object"),
        (lambda r: r.update(conclusion=None), "expected string"),
        (lambda r: r.update(correctness=None), "expected object"),
        (lambda r: r["errors"].update(count=-1), "< 0"),
        (
            lambda r: r.update(artifacts={"logs": "../../etc/passwd"}),
            "relative to the run directory",
        ),
        (lambda r: r.update(profiling_artifacts=[{"name": "perf"}]), "needs path or url"),
        (
            lambda r: r["deployment"].update(
                functions={"inst": "arn:aws:lambda:us-east-2:123456789012:function:x"}
            ),
            "account id",
        ),
    ],
)
def test_validator_names_the_problem(mutate, expect):
    r = real(load("fixture-improved"))
    mutate(r)
    problems = lr.validate(r)
    assert any(expect in p for p in problems), problems


def test_distribution_semantics_are_checked():
    r = real(load("fixture-improved"))
    d = r["phases"]["inst"]["cold"]["weight_download"]
    d["p90"], d["p50"] = d["p50"], d["p90"] + 1
    d["histogram"]["counts"][0] += 1
    problems = lr.validate(r)
    assert any("percentiles out of order" in p for p in problems)
    assert any("counts sum to" in p for p in problems)


def test_baseline_must_not_claim_a_changed_variable():
    r = real(load("fixture-baseline"))
    r["changed_variable"] = {"name": "threads", "baseline": 1, "experiment": 2}
    assert any("baseline changes nothing" in p for p in lr.validate(r))


def test_failed_run_needs_only_the_minimum_but_must_say_why():
    r = real(load("fixture-failed"))
    assert lr.validate(r) == []
    r["conclusion"] = ""
    assert lr.validate(r)


def test_dist_matches_p1_definition():
    d = lr.dist([1.0, 2.0, 3.0, 4.0])
    assert (d["n"], d["p50"], d["p90"], d["p99"], d["min"], d["max"]) == (
        4,
        2.0,
        4.0,
        4.0,
        1.0,
        4.0,
    )
    assert sum(d["histogram"]["counts"]) == 4
    assert lr.dist([None]) is None


# ---------------------------------------------------------------------------- publish


def test_publish_writes_only_its_own_prefix_and_is_idempotent(tmp_path):
    store = lr.LocalStore(str(tmp_path / "store"))
    r = real(load("fixture-improved"), "run-1")
    rep = lr.publish_run(r, fixture_dir("fixture-improved"), store)
    keys = store.list("")
    prefix = "runs/E3-parallel-range-get/run-1/"
    assert keys and all(k.startswith(prefix) for k in keys)
    assert f"{prefix}artifacts/invocations.jsonl" in keys
    assert rep.uploaded and not rep.unchanged
    before = {k: store.get(k) for k in keys}

    again = lr.publish_run(r, fixture_dir("fixture-improved"), store)
    assert again.uploaded == []
    assert sorted(again.unchanged) == sorted(keys)
    assert {k: store.get(k) for k in store.list("")} == before

    pub = json.loads(store.get(f"{prefix}result.json"))["published"]
    assert pub["artifacts"]["invocations.jsonl"]["uri"].endswith(
        f"{prefix}artifacts/invocations.jsonl"
    )
    assert "published" not in r  # caller's dict untouched


def test_publish_refuses_to_overwrite_a_different_run(tmp_path):
    store = lr.LocalStore(str(tmp_path))
    r = real(load("fixture-improved"), "run-1")
    lr.publish_run(r, fixture_dir("fixture-improved"), store)
    r2 = copy.deepcopy(r)
    r2["conclusion"] = "a different result under the same run id"
    with pytest.raises(lr.PublishConflict):
        lr.publish_run(r2, fixture_dir("fixture-improved"), store)
    lr.publish_run(r2, fixture_dir("fixture-improved"), store, replace=True)
    assert (
        "different result"
        in json.loads(store.get("runs/E3-parallel-range-get/run-1/result.json"))["conclusion"]
    )


def test_publish_rejects_invalid_and_unsafe_input(tmp_path):
    store = lr.LocalStore(str(tmp_path))
    bad = real(load("fixture-improved"))
    bad.pop("hypothesis")
    with pytest.raises(lr.ValidationError):
        lr.publish_run(bad, None, store)
    r = real(load("fixture-improved"))
    with pytest.raises(ValueError, match="outside the run directory"):
        lr.publish_run(r, fixture_dir("fixture-improved"), store, extra=["/etc/hostname"])
    assert store.list("") == []


def test_publish_refuses_missing_artifacts_without_writes(tmp_path):
    store = lr.LocalStore(str(tmp_path / "s"))
    r = real(load("fixture-improved"))
    r["artifacts"]["logs"] = "logs/"
    with pytest.raises(FileNotFoundError, match="required artifact"):
        lr.publish_run(r, fixture_dir("fixture-improved"), store)
    assert store.list("") == []


def test_publish_refuses_result_conflict_before_artifact_writes(tmp_path):
    store = lr.LocalStore(str(tmp_path / "s"))
    r = real(load("fixture-improved"), "same-run")
    lr.publish_run(r, fixture_dir("fixture-improved"), store)
    r["conclusion"] = "different result"
    run_dir = fixture_dir("fixture-improved")
    target = os.path.join(run_dir, "new-profile.txt")
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write("must not upload")
        with pytest.raises(lr.PublishConflict):
            lr.publish_run(r, run_dir, store, extra=[target])
    finally:
        os.unlink(target)
    assert not any(k.endswith("new-profile.txt") for k in store.list(""))


def test_publish_rejects_artifact_symlink_escape(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("outside")
    (run_dir / "escape").symlink_to(outside)
    r = real(load("fixture-improved"))
    r["artifacts"] = {"logs": "escape"}
    with pytest.raises(ValueError, match="resolves outside"):
        lr.publish_run(r, str(run_dir), lr.LocalStore(str(tmp_path / "store")))


def test_publish_requires_run_dir_for_listed_artifacts(tmp_path):
    with pytest.raises(ValueError, match="run_dir is required"):
        lr.publish_run(
            real(load("fixture-improved")), None, lr.LocalStore(str(tmp_path / "store"))
        )


def test_concurrent_publishers_never_collide(tmp_path):
    """Each experiment card publishes in parallel. Every run ends up in the store and on the page, and no
    publisher wrote outside its prefix (the only shared outputs are the derived site/ pages)."""
    store = lr.LocalStore(str(tmp_path))
    base = real(load("fixture-improved"))
    runs = []
    for i in range(8):
        r = copy.deepcopy(base)
        r["experiment_id"], r["run_id"] = f"E{i}", f"run-{i}"
        runs.append(r)

    errors: list[BaseException] = []

    def go(r):
        try:
            lr.publish_run(r, fixture_dir("fixture-improved"), store)
            lr.regenerate(store)
        except BaseException as e:  # surfaced below: a thread's exception must fail the test
            errors.append(e)

    ts = [threading.Thread(target=go, args=(r,)) for r in runs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    lr.regenerate(store)
    keys = store.list("")
    assert {k for k in keys if k.endswith("/result.json")} == {
        f"runs/E{i}/run-{i}/result.json" for i in range(8)
    }
    page = store.get("site/dashboard.html").decode()
    for i in range(8):
        assert f"E{i}/run-{i}" in page


def test_publish_command_end_to_end(tmp_path, capsys):
    store, out = str(tmp_path / "store"), str(tmp_path / "out")
    for p in FIXTURES:
        assert (
            publish_result.main(
                [p, "--store", store, "--out", out, "--skip-dashboard-upload"]
            )
            == 0
        )
    assert (
        publish_result.main(
            [
                FIXTURES[0],
                "--store",
                store,
                "--out",
                out,
                "--skip-dashboard-upload",
            ]
        )
        == 0
    )  # idempotent rerun
    stdout = capsys.readouterr().out
    assert "uploaded 0" in stdout
    assert "dashboard rebuilt from 4 run(s)" in stdout
    assert "ARTIFACTORY " not in stdout
    assert os.path.isfile(os.path.join(out, "dashboard.html"))
    assert os.path.isfile(os.path.join(out, "index.html"))
    assert len(glob.glob(os.path.join(out, "*", "*.html"))) == len(FIXTURES)


def test_publish_dashboard_uses_current_sha_and_verifies_ready(tmp_path, monkeypatch):
    page = tmp_path / "dashboard.html"
    page.write_text("<html>dashboard</html>")
    page_sha = hashlib.sha256(page.read_bytes()).hexdigest()

    def tool_result(request_id, payload):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
        }

    proc = FakeMcpProc(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            tool_result(2, {"artifact": {"sha256": "before"}}),
            tool_result(3, {"versionNumber": 7}),
            tool_result(
                4,
                {
                    "artifact": {
                        "currentVersionNumber": 7,
                        "sha256": page_sha,
                        "status": "READY",
                    },
                    "artifactUrl": publish_result.DASHBOARD_URL,
                },
            ),
        ]
    )
    monkeypatch.setattr(publish_result.shutil, "which", lambda _: "/bin/artifactory-mcp")
    monkeypatch.setattr(publish_result.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(publish_result.select, "select", lambda r, w, x, timeout: (r, w, x))

    receipt = publish_result.publish_dashboard(str(page), "artifact-id")
    assert receipt == {
        "artifactId": "artifact-id",
        "artifactUrl": publish_result.DASHBOARD_URL,
        "versionNumber": 7,
        "sha256": page_sha,
        "status": "READY",
    }
    calls = [row for row in proc.writes if row.get("method") == "tools/call"]
    assert [row["params"]["name"] for row in calls] == [
        "get_artifact",
        "upload_artifact_version",
        "get_artifact",
    ]
    upload = calls[1]["params"]["arguments"]
    assert upload["expectedCurrentSha256"] == "before"
    assert upload["filePath"] == str(page)
    assert upload["waitForReady"] is True


def test_publish_dashboard_rejects_preflight_drift_before_upload(tmp_path, monkeypatch):
    page = tmp_path / "dashboard.html"
    page.write_text("<html>dashboard</html>")

    def tool_result(request_id, payload):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
        }

    proc = FakeMcpProc(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            tool_result(2, {"artifact": {"sha256": "newer"}}),
        ]
    )
    monkeypatch.setattr(publish_result.shutil, "which", lambda _: "/bin/artifactory-mcp")
    monkeypatch.setattr(publish_result.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(publish_result.select, "select", lambda r, w, x, timeout: (r, w, x))

    with pytest.raises(publish_result.DashboardPublishError, match="changed while"):
        publish_result.publish_dashboard(
            str(page), "artifact-id", expected_current_sha256="older"
        )
    assert not any(
        row.get("params", {}).get("name") == "upload_artifact_version" for row in proc.writes
    )


def test_publish_dashboard_rejects_wrong_ready_bytes(tmp_path, monkeypatch):
    page = tmp_path / "dashboard.html"
    page.write_text("<html>dashboard</html>")

    def tool_result(request_id, payload):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
        }

    proc = FakeMcpProc(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            tool_result(2, {"artifact": {"sha256": "before"}}),
            tool_result(3, {"versionNumber": 7}),
            tool_result(
                4,
                {
                    "artifact": {
                        "currentVersionNumber": 7,
                        "sha256": "wrong",
                        "status": "READY",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(publish_result.shutil, "which", lambda _: "/bin/artifactory-mcp")
    monkeypatch.setattr(publish_result.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(publish_result.select, "select", lambda r, w, x, timeout: (r, w, x))

    with pytest.raises(publish_result.DashboardPublishError, match="does not match"):
        publish_result.publish_dashboard(str(page), "artifact-id")


def test_publish_command_uploads_dashboard_and_verifies_ready(tmp_path, monkeypatch, capsys):
    store, out = str(tmp_path / "store"), str(tmp_path / "out")
    observed = {}

    def preflight():
        observed["preflight_store_exists"] = os.path.exists(store)
        return "before"

    def publish(path, expected_current_sha256=None):
        observed["path"] = path
        observed["expected"] = expected_current_sha256
        return {
            "artifactId": publish_result.DASHBOARD_ARTIFACT_ID,
            "artifactUrl": publish_result.DASHBOARD_URL,
            "versionNumber": 9,
            "sha256": "a" * 64,
            "status": "READY",
        }

    monkeypatch.setattr(publish_result, "get_dashboard_sha", preflight)
    monkeypatch.setattr(publish_result, "publish_dashboard", publish)
    assert publish_result.main([FIXTURES[0], "--store", store, "--out", out]) == 0
    assert observed["path"] == os.path.abspath(os.path.join(out, "dashboard.html"))
    assert observed["expected"] == "before"
    assert observed["preflight_store_exists"] is False
    receipt = json.loads(capsys.readouterr().out.rsplit("ARTIFACTORY ", 1)[1])
    assert receipt["versionNumber"] == 9
    assert receipt["status"] == "READY"


def test_publish_command_fails_when_dashboard_upload_fails(tmp_path, monkeypatch):
    def refused(_path, expected_current_sha256=None):
        raise publish_result.DashboardPublishError("stale SHA")

    monkeypatch.setattr(publish_result, "get_dashboard_sha", lambda: "before")
    monkeypatch.setattr(publish_result, "publish_dashboard", refused)
    assert (
        publish_result.main(
            [FIXTURES[0], "--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]
        )
        == 4
    )


def test_publish_command_exit_codes(tmp_path):
    bad = real(load("fixture-improved"))
    bad.pop("hypothesis")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert publish_result.main([str(p), "--validate-only"]) == 2
    good = tmp_path / "good" / "results.json"
    good.parent.mkdir()
    (good.parent / "invocations.jsonl").write_text("{}\n")
    r = real(load("fixture-improved"), "x")
    good.write_text(json.dumps(r))
    store = str(tmp_path / "s")
    assert (
        publish_result.main(
            [
                str(good),
                "--store",
                store,
                "--out",
                str(tmp_path / "o"),
                "--skip-dashboard-upload",
            ]
        )
        == 0
    )
    r["conclusion"] = "changed"
    good.write_text(json.dumps(r))
    assert (
        publish_result.main(
            [
                str(good),
                "--store",
                store,
                "--out",
                str(tmp_path / "o"),
                "--skip-dashboard-upload",
            ]
        )
        == 3
    )


# ---------------------------------------------------------------------------- rendering


def all_fixtures() -> list[dict]:
    return [json.load(open(p, encoding="utf-8")) for p in FIXTURES]


def test_dashboard_is_self_contained():
    page = lr.render_site(all_fixtures())["dashboard.html"]
    assert "<script" not in page
    assert not re.search(r'(src|href)="(https?:)?//', page)
    assert "<link" not in page
    assert re.findall(r"https?://[^\"' <]+", page) == ["http://www.w3.org/2000/svg"] * page.count(
        "<svg"
    )


def test_failed_and_null_results_are_explicit():
    site = lr.render_site(all_fixtures())
    page = site["dashboard.html"]
    assert 'class="failed-row"' in page and "s-failed" in page
    assert "Failed run." in page
    assert "AccessDenied" in page  # the cause, not just a red badge
    assert "No latency measured." in page
    assert 'class="null-row"' in page and "s-no-change" in page
    assert "not measured (null)" in page  # E2 fixture's restore_to_invoke is null
    assert "not compared" in page
    failed = site["E1-s3-express/fixture-failed.html"]
    assert "Failed run." in failed and "Errors: 20" in failed


def test_fixtures_are_labelled_and_hidden_once_real_runs_exist():
    fx = all_fixtures()
    page = lr.render_site(fx)["dashboard.html"]
    assert "FIXTURE DATA." in page and page.count("badge fx") >= len(fx)

    real_base = real(load("fixture-baseline"), "2026-10-21-a")
    site = lr.render_site([*fx, real_base])
    page = site["dashboard.html"]
    assert "FIXTURE DATA." not in page
    assert "fixture-improved" not in page
    assert "4 fixture run(s) hidden" in page
    assert "Baseline: P1-baseline/2026-10-21-a" in page
    assert (
        "fixture-improved"
        in lr.render_site([*fx, real_base], include_fixtures=True)["dashboard.html"]
    )


def test_deltas_are_against_the_baseline_same_class_and_phase():
    runs = [real(load("fixture-baseline")), real(load("fixture-improved"))]
    b = runs[0]["phases"]["inst"]["cold"]["weight_download"]["p50"]
    e = runs[1]["phases"]["inst"]["cold"]["weight_download"]["p50"]
    want = f"({100 * (e - b) / b:+.1f}%)"
    page = lr.render_site(runs)["dashboard.html"]
    assert want in page
    assert "<svg" in page and "baseline p50" in page


def test_no_baseline_is_stated():
    page = lr.render_site([real(load("fixture-improved"))])["dashboard.html"]
    assert "No baseline published yet." in page


def test_rendered_text_is_escaped_and_masks_account_ids():
    r = real(load("fixture-improved"))
    r["conclusion"] = "<script>alert(1)</script> in 123456789012"
    r["published"] = {"result": "s3://b/runs/x", "result_link": "javascript:alert(1)"}
    page = lr.render_site([r])["dashboard.html"]
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "123456789012" not in page
    assert 'href="javascript:' not in page


def test_detail_pages_link_back_and_to_the_baseline():
    site = lr.render_site([real(load("fixture-baseline")), real(load("fixture-improved"))])
    p = site["E3-parallel-range-get/fixture-improved.html"]
    assert 'href="../index.html"' in p
    assert 'href="../P1-baseline/fixture-baseline.html"' in p
    assert 'href="E3-parallel-range-get/fixture-improved.html"' in site["index.html"]


def test_detail_anchors_do_not_collide_after_slugging():
    first = real(load("fixture-baseline"), "a.b")
    second = real(load("fixture-improved"), "a-b")
    second["experiment_id"] = first["experiment_id"]
    assert lr.anchor(first) != lr.anchor(second)
    page = lr.render_site([first, second])["dashboard.html"]
    assert f'id="{lr.anchor(first)}"' in page
    assert f'id="{lr.anchor(second)}"' in page


def test_regenerate_is_deterministic(tmp_path):
    store = lr.LocalStore(str(tmp_path))
    for r in all_fixtures():
        lr.publish_run(r, fixture_dir(r["run_id"]), store)
    lr.regenerate(store)
    first = {k: store.get(k) for k in store.list("site/")}
    lr.regenerate(store)
    assert {k: store.get(k) for k in store.list("site/")} == first
