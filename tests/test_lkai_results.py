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
        "phases.other.cold: report_max_memory_mb MB distribution is required" in p for p in problems
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
        lr.publish_run(real(load("fixture-improved")), None, lr.LocalStore(str(tmp_path / "store")))


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
            publish_result.main([p, "--store", store, "--out", out, "--skip-dashboard-upload"]) == 0
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


class FakeArtifactory:
    """In-memory artifactory-mcp: answers get_artifact / upload_artifact_version / list_artifact_versions like the
    real server (sha256 of stored bytes, version-pinned downloads into ARTIFACTORY_DOWNLOAD_DIR, CAS refusal).
    `transform` mimics the server's HTML rewrite before upload; `tamper` corrupts stored bytes after it."""

    def __init__(self, first: bytes = b"<html><head></head>v1</html>"):
        self.versions: list[dict] = []
        self.transform = lambda b: b
        self.tamper = None
        self.status_override: str | None = None
        self.report_sha: str | None = None
        self._add(first)
        self.uploads: list[dict] = []

    def _add(self, data: bytes) -> dict:
        row = {
            "versionNumber": len(self.versions) + 1,
            "sha256": hashlib.sha256(data).hexdigest(),
            "status": "READY",
            "data": data,
        }
        self.versions.append(row)
        return row

    @property
    def current_sha(self) -> str:
        return self.versions[-1]["sha256"]

    def popen(self, *args, env=None, **kwargs):
        return _FakeArtifactoryProc(self, (env or {}).get("ARTIFACTORY_DOWNLOAD_DIR"))

    def handle(self, tool: str, a: dict, download_dir: str | None) -> dict:
        if tool == "get_artifact":
            row = self.versions[a["version"] - 1] if "version" in a else self.versions[-1]
            path = None
            if download_dir:
                path = os.path.join(download_dir, a["artifactId"], f"v{row['versionNumber']}.html")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(row["data"])
            return {
                "artifact": {
                    "sha256": row["sha256"],
                    "status": row["status"],
                    "currentVersionNumber": len(self.versions),
                },
                "localPath": path,
                "artifactUrl": publish_result.DASHBOARD_URL,
            }
        if tool == "upload_artifact_version":
            self.uploads.append(a)
            if a["expectedCurrentSha256"] != self.current_sha:
                raise _ToolError("Upload (new version) failed (ConflictException): sha moved")
            with open(a["filePath"], "rb") as f:
                data = self.transform(f.read())
            row = self._add(data)
            reported = self.report_sha or row["sha256"]
            if self.tamper:
                row["data"] = self.tamper(row["data"])
                row["sha256"] = hashlib.sha256(row["data"]).hexdigest()
            if self.status_override:
                row["status"] = self.status_override
            return {
                "versionNumber": row["versionNumber"],
                "versionStatus": row["status"],
                "sha256": reported,
                "artifactUrl": publish_result.DASHBOARD_URL,
            }
        if tool == "list_artifact_versions":
            return {
                "versions": [{k: v for k, v in row.items() if k != "data"} for row in self.versions]
            }
        raise AssertionError(f"unexpected tool {tool}")


class _ToolError(Exception):
    pass


class _FakeArtifactoryProc(FakeMcpProc):
    def __init__(self, server: FakeArtifactory, download_dir: str | None):
        super().__init__([])
        self.server, self.download_dir = server, download_dir

    def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line:
                continue
            msg = json.loads(line)
            self.writes.append(msg)
            if "id" not in msg:
                continue
            if msg["method"] != "tools/call":
                self.responses.append({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
                continue
            p = msg["params"]
            try:
                out = self.server.handle(p["name"], p["arguments"], self.download_dir)
                result = {"content": [{"type": "text", "text": json.dumps(out)}]}
            except _ToolError as e:
                result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
            self.responses.append({"jsonrpc": "2.0", "id": msg["id"], "result": result})


# The bridge splice artifactory-mcp really performs (observed on the hosted v4): comments and scripts after <head>.
SPLICE = (
    b"\n<!-- __artifactory_data_bridge__ c7dcc82a -->\n<script>(function(){/*bridge*/})();</script>"
    b'<script type="application/json" id="m">{}</script>'
)


def _splice(b: bytes) -> bytes:
    at = b.index(b"<head>") + len(b"<head>")
    return b[:at] + SPLICE + b[at:]


@pytest.fixture
def artifactory(monkeypatch):
    server = FakeArtifactory()
    monkeypatch.setattr(publish_result.shutil, "which", lambda _: "/bin/artifactory-mcp")
    monkeypatch.setattr(publish_result.subprocess, "Popen", server.popen)
    monkeypatch.setattr(publish_result.select, "select", lambda r, w, x, timeout: (r, w, x))
    monkeypatch.setattr(publish_result, "READY_POLL_S", 0)
    return server


def _page(tmp_path, body: bytes = b"<p>dashboard</p>") -> tuple[str, bytes]:
    data = (
        b'<!doctype html><html><head><meta charset="utf-8"></head><body>'
        + body
        + b"</body></html>\n"
    )
    p = tmp_path / "dashboard.html"
    p.write_bytes(data)
    return str(p), data


def test_publish_dashboard_cas_uses_preflight_sha_and_verifies_bytes(tmp_path, artifactory):
    path, data = _page(tmp_path)
    artifactory.transform = _splice
    before = artifactory.current_sha
    receipt = publish_result.publish_dashboard(path, "artifact-id", expected_current_sha256=before)
    hosted = artifactory.versions[-1]["data"]
    assert receipt == {
        "artifactId": "artifact-id",
        "artifactUrl": publish_result.DASHBOARD_URL,
        "versionNumber": 2,
        "sha256": hashlib.sha256(hosted).hexdigest(),
        "generatedSha256": hashlib.sha256(data).hexdigest(),
        "status": "READY",
    }
    (upload,) = artifactory.uploads
    assert upload["expectedCurrentSha256"] == before
    assert upload["filePath"] == path
    assert upload["waitForReady"] is True


def test_publish_dashboard_requires_preflight_sha(tmp_path, artifactory):
    path, _ = _page(tmp_path)
    with pytest.raises(TypeError):
        publish_result.publish_dashboard(path, "artifact-id")  # no late-read fallback exists
    with pytest.raises(publish_result.DashboardPublishError, match="required"):
        publish_result.publish_dashboard(path, "artifact-id", expected_current_sha256="")
    assert artifactory.uploads == []


def test_publish_dashboard_refuses_when_newer_version_appeared(tmp_path, artifactory):
    """Reviewer's race: A reads S0, B publishes a newer complete page, A must not overwrite it."""
    path, _ = _page(tmp_path, b"stale")
    s0 = artifactory.current_sha
    artifactory._add(b"<html><head></head>newer page from B</html>")
    with pytest.raises(publish_result.DashboardPublishError, match="changed while"):
        publish_result.publish_dashboard(path, "artifact-id", expected_current_sha256=s0)
    assert artifactory.uploads == []
    assert artifactory.versions[-1]["data"].endswith(b"newer page from B</html>")


def test_publish_dashboard_cas_conflict_at_upload_is_refused(tmp_path, artifactory):
    """B wins between A's re-check and A's upload: the server-side CAS on the preflight SHA refuses A."""
    path, _ = _page(tmp_path, b"stale")
    s0 = artifactory.current_sha
    real_handle = artifactory.handle

    def racing(tool, a, d):
        if tool == "upload_artifact_version" and len(artifactory.versions) == 1:
            artifactory._add(b"<html><head></head>B</html>")
        return real_handle(tool, a, d)

    artifactory.handle = racing
    with pytest.raises(publish_result.DashboardPublishError, match="ConflictException"):
        publish_result.publish_dashboard(path, "artifact-id", expected_current_sha256=s0)
    assert artifactory.versions[-1]["data"] == b"<html><head></head>B</html>"


@pytest.mark.parametrize(
    "setup, match",
    [
        (
            lambda s: setattr(s, "tamper", lambda b: b.replace(b"dashboard", b"dashboarX")),
            "!= uploaded",
        ),
        (
            lambda s: setattr(s, "transform", lambda b: b.replace(b"dashboard", b"stale-run")),
            "does not match",
        ),
        (
            lambda s: setattr(
                s, "transform", lambda b: _splice(b).replace(SPLICE, SPLICE + b"<p>x</p>")
            ),
            "does not match",
        ),
        (lambda s: setattr(s, "transform", lambda b: b[:-5]), "does not match"),
        (lambda s: setattr(s, "report_sha", "0" * 64), "!= uploaded"),
        (lambda s: setattr(s, "status_override", "MISMATCHED"), "MISMATCHED"),
    ],
    ids=[
        "stored-bytes-differ",
        "body-rewritten",
        "non-platform-insert",
        "truncated",
        "wrong-sha",
        "mismatched",
    ],
)
def test_publish_dashboard_rejects_hosted_bytes_that_are_not_the_generated_page(
    tmp_path, artifactory, setup, match
):
    path, _ = _page(tmp_path)
    setup(artifactory)
    with pytest.raises(publish_result.DashboardPublishError, match=match):
        publish_result.publish_dashboard(
            path, "artifact-id", expected_current_sha256=artifactory.current_sha
        )


def test_publish_dashboard_verifies_its_own_version_even_if_superseded(tmp_path, artifactory):
    """A later publisher may be current by readback time; we verify OUR version number, not 'current'."""
    path, data = _page(tmp_path)
    real_handle = artifactory.handle

    def later(tool, a, d):
        out = real_handle(tool, a, d)
        if tool == "upload_artifact_version":
            artifactory._add(b"<html><head></head>C</html>")
        return out

    artifactory.handle = later
    receipt = publish_result.publish_dashboard(
        path, "artifact-id", expected_current_sha256=artifactory.current_sha
    )
    assert receipt["versionNumber"] == 2
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()


def test_platform_splice_only_matches_the_real_hosted_shape():
    gen = (
        b'<!doctype html><html lang="en"><head><meta charset="utf-8"></head><body>x</body></html>\n'
    )
    assert publish_result.platform_splice_only(gen, gen)
    assert publish_result.platform_splice_only(gen, _splice(gen))
    assert not publish_result.platform_splice_only(
        gen, _splice(gen).replace(b"x</body>", b"y</body>")
    )
    assert not publish_result.platform_splice_only(gen, gen.replace(b"<head>", b"<head><p>ad</p>"))
    assert not publish_result.platform_splice_only(b"no head", b"<script></script>no head")


def test_publish_command_reads_dashboard_sha_before_any_store_write(tmp_path, monkeypatch, capsys):
    store, out = str(tmp_path / "store"), str(tmp_path / "out")
    observed = {}

    def preflight():
        observed["preflight_store_exists"] = os.path.exists(store)
        return "before"

    def publish(path, *, expected_current_sha256):
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


def test_concurrent_publish_commands_never_regress_the_dashboard(tmp_path, artifactory, capsys):
    """End to end, through main(): A preflights, B publishes a later run completely, then A finishes. A is refused
    (exit 4) and the hosted page still lists B's run; a --regenerate-only retry then publishes both runs."""
    artifactory.transform = _splice
    store = str(tmp_path / "store")
    a_run, b_run = FIXTURES[0], FIXTURES[1]
    real_get = publish_result.get_dashboard_sha
    state = {}

    def a_preflight_then_b_runs():
        sha = real_get(publish_result.DASHBOARD_ARTIFACT_ID)
        if "b" not in state:
            state["b"] = None  # B's own preflight goes straight through
            state["b"] = publish_result.main(
                [b_run, "--store", store, "--out", str(tmp_path / "b")]
            )
        return sha

    publish_result.get_dashboard_sha = a_preflight_then_b_runs
    try:
        rc = publish_result.main([a_run, "--store", store, "--out", str(tmp_path / "a")])
    finally:
        publish_result.get_dashboard_sha = real_get
    assert state["b"] == 0 and rc == 4
    assert "changed while" in capsys.readouterr().err

    def key(p):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        return f"{r['experiment_id']}/{r['run_id']}".encode()

    a_key, b_key = key(a_run), key(b_run)
    assert b_key in artifactory.versions[-1]["data"]
    assert (
        publish_result.main(["--regenerate-only", "--store", store, "--out", str(tmp_path / "r")])
        == 0
    )
    assert a_key in artifactory.versions[-1]["data"] and b_key in artifactory.versions[-1]["data"]


def test_publish_command_fails_when_dashboard_upload_fails(tmp_path, monkeypatch):
    def refused(_path, *, expected_current_sha256):
        raise publish_result.DashboardPublishError("stale SHA")

    monkeypatch.setattr(publish_result, "get_dashboard_sha", lambda: "before")
    monkeypatch.setattr(publish_result, "publish_dashboard", refused)
    assert (
        publish_result.main(
            [FIXTURES[0], "--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]
        )
        == 4
    )


def test_publish_command_refuses_when_regeneration_never_settles(tmp_path, monkeypatch):
    def moving(*a, **k):
        raise lr.RegenerateDidNotSettle("run set changed")

    published = []
    monkeypatch.setattr(publish_result, "get_dashboard_sha", lambda: "before")
    monkeypatch.setattr(publish_result, "publish_dashboard", lambda *a, **k: published.append(a))
    monkeypatch.setattr(publish_result.lr, "regenerate", moving)
    assert (
        publish_result.main(
            [FIXTURES[0], "--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]
        )
        == 4
    )
    assert published == []


def test_regenerate_raises_instead_of_writing_an_unsettled_page(tmp_path, monkeypatch):
    store = lr.LocalStore(str(tmp_path / "s"))
    calls = iter(range(100))
    monkeypatch.setattr(
        lr, "load_runs", lambda s: [{"experiment_id": "E", "run_id": str(next(calls))}]
    )
    monkeypatch.setattr(
        lr, "render_site", lambda runs, include_fixtures=None: {"dashboard.html": "x"}
    )
    out = tmp_path / "out"
    with pytest.raises(lr.RegenerateDidNotSettle):
        lr.regenerate(store, str(out))
    assert not out.exists()


def test_racing_creates_of_one_run_id_cannot_both_win(tmp_path, monkeypatch):
    """Check-then-put on result.json: two publishers both see 'absent'; the create-only put lets one win."""
    store = lr.LocalStore(str(tmp_path / "s"))
    r1 = real(load("fixture-improved"), "same")
    r2 = copy.deepcopy(r1)
    r2["conclusion"] = "other publisher's measurement"
    real_get = store.get
    key = "runs/E3-parallel-range-get/same/result.json"

    def blind_get(k):
        return None if k == key else real_get(k)  # both publishers read before either writes

    monkeypatch.setattr(store, "get", blind_get)
    lr.publish_run(r1, fixture_dir("fixture-improved"), store)
    with pytest.raises(lr.PublishConflict, match="another publisher"):
        lr.publish_run(r2, fixture_dir("fixture-improved"), store)
    monkeypatch.setattr(store, "get", real_get)
    assert json.loads(store.get(key))["conclusion"] == r1["conclusion"]
    lr.publish_run(r1, fixture_dir("fixture-improved"), store)  # same bytes after a lost race: fine


def test_s3_store_puts_carry_checksum_and_create_only_condition():
    class ClientError(Exception):  # botocore's shape: .response["Error"]["Code"]
        def __init__(self, response, op):
            super().__init__(op)
            self.response = response

    class S3:
        def __init__(self):
            self.calls = []

        def put_object(self, **kw):
            self.calls.append(kw)
            if kw.get("IfNoneMatch") == "*" and len(self.calls) > 1:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")

    s = lr.S3Store.__new__(lr.S3Store)
    s.bucket, s.prefix, s.s3 = "b", "p", S3()
    s.put("k", b"abc", if_absent=True)
    with pytest.raises(lr.KeyExists):
        s.put("k", b"abc", if_absent=True)
    s.put("site/x", b"abc")
    first, _, plain = s.s3.calls
    import base64

    assert first["ChecksumSHA256"] == base64.b64encode(hashlib.sha256(b"abc").digest()).decode()
    assert first["IfNoneMatch"] == "*" and first["Key"] == "p/k"
    assert "IfNoneMatch" not in plain and plain["ChecksumSHA256"] == first["ChecksumSHA256"]


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
