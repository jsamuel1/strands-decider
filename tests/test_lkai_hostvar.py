"""examples/lambda-kleidiai/hostvar: H1's cross-run host-variability analysis.

Synthetic rows only (plus one tiny git repo for `extract`), so no AWS and no fork fetch. The tests pin: CPU parts
are counted per run/arm/class and never named as a generation; a row without a part is unassessed, never a shared
stratum; a pooled comparison that hides a within-part difference is flagged; the decision follows the observed
identities; and the generated file passes the P2 validator's account-id rule.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys

import pytest

HV = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "hostvar")
sys.path.insert(0, os.path.abspath(HV))

import hostvar  # noqa: E402

GENERATION = re.compile(r"graviton\s*\d?|neoverse|cortex-a\d+", re.I)
ACCOUNT = re.compile(r"\b\d{12}\b")


def row(
    run="R",
    arm="ctl",
    cls="warm",
    part="0xd40",
    engine=0.25,
    rnd=0,
    env="e",
    req=hostvar.PROBE,
    **kw,
):
    r = {
        "run": run,
        "arm": arm,
        "round": rnd,
        "wave": cls,
        "seq": 0,
        "t_start": 0.0,
        "cls": cls,
        "ok": True,
        "request": req,
        "cpu_part": part,
        "env_id": env,
        "lambda_request_id": None,
        "version": "$LATEST",
        "ggml_cpu": None,
        "host": None,
        "log_stream": None,
        "engine_s": engine,
        "duration_s": engine,
        "init_s": 0.8 if cls == "cold" else None,
        "restore_s": 3.2 if cls == "restore" else None,
        "memory_mb": 10240.0,
        "max_memory_mb": 5100.0,
    }
    r.update(kw)
    return r


def src(key="R", ctl=("ctl",), client="fresh-per-wave"):
    return {
        "key": key,
        "experiment_id": f"X-{key}",
        "run_id": "r",
        "branch": "b",
        "commit": "c" * 40,
        "dirs": ["d"],
        "control_arms": list(ctl),
        "invoke_client": client,
    }


# ------------------------------------------------------------------ pooled vs within


def _vals(rs):
    return [(hostvar.part_label(r["cpu_part"]), r["engine_s"]) for r in rs]


def test_pooled_result_hides_a_within_part_regression():
    """Candidate is ~18% slower on every part, but drew far fewer slow-part environments, so the pooled median
    looks faster than the control's."""
    ctl = [row(part="0xd40", engine=0.25)] * 45 + [row(part="0xd0c", engine=0.45)] * 55
    cand = [row(arm="c", part="0xd40", engine=0.295)] * 95 + [
        row(arm="c", part="0xd0c", engine=0.53)
    ] * 5
    cell = hostvar.compare_cell(_vals(ctl), _vals(cand))
    assert cell["pooled_delta"] == pytest.approx(0.295 - 0.45)  # pooled: candidate 155 ms "faster"
    assert {p: d["delta"] for p, d in cell["within_part"].items()} == {
        "0xd0c": pytest.approx(0.08),
        "0xd40": pytest.approx(0.045),
    }
    assert cell["flag"] == "sign_flip"


def test_pooled_sign_flip_is_flagged_near_balanced_mix():
    ctl = [row(part="0xd40", engine=0.25)] * 49 + [row(part="0xd0c", engine=0.45)] * 51
    cand = [row(arm="c", part="0xd40", engine=0.26)] * 90 + [
        row(arm="c", part="0xd0c", engine=0.47)
    ] * 10
    cell = hostvar.compare_cell(_vals(ctl), _vals(cand))
    assert cell["pooled_delta"] < 0  # pooled: candidate "faster"
    assert all(d["delta"] > 0 for d in cell["within_part"].values())  # every part: candidate slower
    assert cell["flag"] == "sign_flip"


def test_matched_mix_is_consistent():
    ctl = [row(part="0xd40", engine=0.25)] * 90 + [row(part="0xd0c", engine=0.45)] * 10
    cand = [row(arm="c", part="0xd40", engine=0.26)] * 90 + [
        row(arm="c", part="0xd0c", engine=0.47)
    ] * 10
    assert hostvar.compare_cell(_vals(ctl), _vals(cand))["flag"] == "consistent"


def test_sub_floor_differences_are_not_flagged():
    ctl = [row(part="0xd40", engine=0.25)] * 90 + [row(part="0xd0c", engine=0.45)] * 10
    cand = [row(arm="c", part="0xd40", engine=0.2501)] * 89 + [
        row(arm="c", part="0xd0c", engine=0.4501)
    ] * 11
    assert hostvar.compare_cell(_vals(ctl), _vals(cand))["flag"] == "consistent"


def test_missing_parts_are_unassessed_not_a_stratum():
    ctl = [row(part=None, engine=0.25)] * 10
    cand = [row(arm="c", part=None, engine=0.50)] * 10
    cell = hostvar.compare_cell(_vals(ctl), _vals(cand))
    assert cell["within_part"] == {} and cell["flag"] == "not assessable"
    assert cell["unassessed_rows"] == [10, 10]


def test_mixed_known_and_missing_parts_count_unknown_rows():
    ctl = [row(part="0xd40")] * 5 + [row(part=None)] * 3
    cand = [row(arm="c", part="0xd40")] * 5 + [row(arm="c", part=None)] * 2
    cell = hostvar.compare_cell(_vals(ctl), _vals(cand))
    assert list(cell["within_part"]) == ["0xd40"] and cell["unassessed_rows"] == [3, 2]


def test_part_with_too_few_rows_is_not_compared():
    ctl = [row(part="0xd40")] * 5 + [row(part="0xd4f", engine=0.2)] * (hostvar.MIN_N - 1)
    cand = [row(arm="c", part="0xd40")] * 5 + [row(arm="c", part="0xd4f", engine=0.2)] * 5
    assert list(hostvar.compare_cell(_vals(ctl), _vals(cand))["within_part"]) == ["0xd40"]


# ------------------------------------------------------------------ identity and decision


def _streams(run="R"):
    return [
        {
            "run": run,
            "function": "f",
            "stream": "s1",
            "version": "3",
            "kind": "snapshot_build",
            "init_cpu_parts": ["0xd40"],
            "ggml_cpu": ["v86"],
            "request_ids": [],
        },
        {
            "run": run,
            "function": "f",
            "stream": "s2",
            "version": "3",
            "kind": "snapshot_build",
            "init_cpu_parts": ["0xd0c"],
            "ggml_cpu": ["v82"],
            "request_ids": [],
        },
        {
            "run": run,
            "function": "f",
            "stream": "s3",
            "version": "$LATEST",
            "kind": "cold",
            "init_cpu_parts": ["0xd40"],
            "ggml_cpu": ["v86"],
            "request_ids": ["q1"],
        },
    ]


def test_cpu_identity_counts_stability_and_snapshot_parts():
    rows = [
        row(cls="cold", env="a", lambda_request_id="q1"),
        row(cls="warm", env="a"),
        row(cls="cold", env="b", part="0xd0c"),
        row(cls="restore", env="c", part=None),
    ]
    ident = hostvar.cpu_identity(rows, _streams())
    assert ident["by_run_arm_class"]["R"]["ctl"] == {
        "cold": {"0xd0c": 1, "0xd40": 1},
        "restore": {"unknown": 1},
        "warm": {"0xd40": 1},
    }
    assert ident["environment_stability"]["environments_with_more_than_one_part"] == 0
    assert ident["log_stream_stability"]["rows_linked_to_a_stream"] == 1
    assert ident["snapshot_build"]["versions_by_part_set"] == {"0xd0c+0xd40": 1}
    assert ident["ggml_cpu_variant_by_part"] == {"0xd0c": {"v82": 1}, "0xd40": {"v86": 2}}


def test_environment_with_two_parts_is_reported():
    rows = [row(env="a", part="0xd40"), row(env="a", part="0xd0c")]
    assert (
        hostvar.cpu_identity(rows, [])["environment_stability"][
            "environments_with_more_than_one_part"
        ]
        == 1
    )


def test_stream_with_two_parts_is_reported():
    rows = [row(lambda_request_id="q1", part="0xd0c", env=None)]  # stream s3's init said 0xd40
    assert (
        hostvar.cpu_identity(rows, _streams())["log_stream_stability"][
            "streams_with_more_than_one_part"
        ]
        == 1
    )


def test_single_identity_is_controlled_and_multiple_are_not():
    one = [row(cls=c, part="0xd40") for c in hostvar.CLASSES]
    d = hostvar.decide(hostvar.cpu_identity(one, []), one)
    assert d["placement_controlled"] is True and d["evidence"]["cpu_part_ids_observed"] == ["0xd40"]
    two = [*one, row(cls="warm", part="0xd0c", engine=0.45)]
    d = hostvar.decide(hostvar.cpu_identity(two, []), two)
    assert d["placement_controlled"] is False and d["verdict"] == "not controlled"
    assert d["evidence"]["warm_probe_engine_p50_ratio_by_part"] == {"0xd0c/0xd40": 1.8}


def test_unknown_part_never_counts_as_an_identity():
    rows = [row(part="0xd40"), row(part=None)]
    d = hostvar.decide(hostvar.cpu_identity(rows, []), rows)
    assert d["evidence"]["cpu_part_ids_observed"] == ["0xd40"]


def test_order_pattern_names_alternation_not_interleaving():
    p = hostvar._order_pattern([["a", "b"], ["b", "a"], ["a", "b"]])
    assert p.startswith("alternating") and "not interleaved" in p
    assert hostvar._order_pattern([]).startswith("unknown")


# ------------------------------------------------------------------ end to end


def _cfg(arm, code="X"):
    return {
        "code_sha256": code,
        "memory_mb": 10240,
        "architecture": "arm64",
        "runtime": "python3.12",
        "handler": "h",
        "tracing": None,
        "layers": [],
        "environment": {"A": "1"},
    }


def _cond(arms, code="X"):
    return [
        {
            "region": "us-east-2",
            "rounds": 2,
            "concurrency": 10,
            "runtime_version_arn": "arn:aws:lambda:us-east-2::runtime:abc",
            "started": "2026-10-20T00:00:00Z",
            "finished": "2026-10-20T01:00:00Z",
            "orders": [arms, arms[::-1]],
            "client": {},
            "functions": {a: _cfg(a, code) for a in arms},
        }
    ]


def _fixture_rows():
    rows = []
    for run in ("A", "B"):
        for rnd in (0, 1):
            for arm in ("ctl", "c"):
                slow = 0.05 if arm == "c" else 0.0
                for i in range(8):
                    part = "0xd0c" if (i < (4 if arm == "ctl" else 1)) else "0xd40"
                    base = 0.45 if part == "0xd0c" else 0.25
                    for cls in hostvar.CLASSES:
                        rows.append(
                            row(
                                run=run,
                                arm=arm,
                                cls=cls,
                                rnd=rnd,
                                part=part,
                                engine=base + slow,
                                env=f"{run}{rnd}{arm}{i}",
                            )
                        )
    return rows


def test_build_result_is_p2_shaped_and_names_no_generation():
    rows = _fixture_rows()
    sources = {"runs": [src("A", client="pooled"), src("B")], "excluded": {}}
    conds = {"A": _cond(["ctl", "c"]), "B": _cond(["ctl", "c"])}
    res = hostvar.build_result(
        rows,
        _streams("A"),
        conds,
        sources,
        {"used": False},
        {"baseline": "a" * 40, "experiment": "b" * 40},
    )
    hv = res["host_variability"]
    assert res["status"] == "inconclusive" and res["changed_variable"]["name"]
    assert hv["decision"]["placement_controlled"] is False
    assert set(hv["same_code_control_pool"]["runs"]) == {"A", "B"}
    assert set(res["phases"]) == {"part_0xd0c", "part_0xd40"}
    for p in res["phases"].values():
        assert set(p) == set(hostvar.CLASSES)
        assert p["cold"]["weight_download"] is None  # not pooled across runs, said so
    assert hv["restore_latency_caveat"]["runs_with_pooled_or_unknown_invoke_client"] == ["A"]
    assert hv["pooled_vs_within"]["comparisons"] > 0
    # The fixture's part mix differs by arm, so at least one comparison is flagged.
    assert any(c["flag"] != "consistent" for c in hv["pooled_vs_within"]["all"])
    for k in (
        "cpu_feature_flags",
        "hwcap_hwcap2",
        "cgroup_cpu_memory_limits",
        "availability_zone",
        "processor_generation",
    ):
        assert hv["unknown"][k]
    text = json.dumps(res)
    assert not GENERATION.search(text)
    assert not ACCOUNT.search(text)


def test_different_code_is_not_pooled():
    rows = _fixture_rows()
    sources = {"runs": [src("A"), src("B")], "excluded": {}}
    conds = {"A": _cond(["ctl", "c"], code="X"), "B": _cond(["ctl", "c"], code="Y")}
    res = hostvar.build_result(
        rows, [], conds, sources, {"used": False}, {"baseline": "a" * 7, "experiment": "b" * 7}
    )
    pool = res["host_variability"]["same_code_control_pool"]
    assert len(pool["runs"]) == 1 and len(pool["other_groups"]) == 1


def test_fixed_host_is_labelled_not_lambda(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(
        json.dumps(
            {
                "graviton": {
                    "what": "offline",
                    "host": [
                        "Linux x 6.1 aarch64",
                        "CPU part\t: 0xd40",
                        "16",
                        "max 100000",
                        "Python 3.12.15",
                    ],
                    "runs": {
                        "plain": {
                            "host": {"cpu": "0xd40", "nproc": 16, "kernel": "6.1"},
                            "threads": 6,
                            "ggml_cpu": ["v"],
                            "steady_state_p50_s": {"q": 0.24},
                            "phases_s": {"model_load_mmap": 7.9},
                        }
                    },
                }
            }
        )
    )
    fh = hostvar.fixed_host_from_offline(str(p))
    assert fh["used"] and "not Lambda" in fh["label"] and fh["pinned"]["cpu_part"] == "0xd40"
    assert fh["unknown"]["instance_type"]
    assert hostvar.fixed_host_from_offline(None)["used"] is False


# ------------------------------------------------------------------ extract


def test_compact_row_and_stream_rows_read_only_recorded_fields():
    tail = (
        "START RequestId: q9 Version: $LATEST\n2026 decider.engine init cpu_part=0xd0c ggml_cpu=a.so,b.so threads=6\n"
        "REPORT RequestId: q9\tDuration: 100.0 ms\n"
    )
    r = hostvar.compact_row(
        src(),
        {
            "arm": "ctl",
            "round": 1,
            "expected_class": "cold",
            "cpu_part": "0xd0c",
            "log_tail": tail,
            "latency_ms": 250.0,
            "report": {"Duration": 100.0, "Init Duration": 800.0, "Memory Size": 10240.0},
        },
    )
    assert r["lambda_request_id"] == "q9" and r["ggml_cpu"] == "a.so,b.so" and r["init_s"] == 0.8
    assert r["env_id"] is None and r["restore_s"] is None and r["host"] is None
    ev = [
        {"logStreamName": "2026/10/20/[3]abc", "message": "INIT_START Runtime\n"},
        {
            "logStreamName": "2026/10/20/[3]abc",
            "message": "x decider.engine init cpu_part=0xd40 ggml_cpu=v threads=6",
        },
        {"logStreamName": "2026/10/20/[3]def", "message": "RESTORE_START Runtime\n"},
        {"logStreamName": "2026/10/20/[3]def", "message": "START RequestId: q1 Version: 3\n"},
    ]
    st = {s["stream"]: s for s in hostvar.stream_rows(src(), "f", ev)}
    assert st["2026/10/20/[3]abc"]["kind"] == "snapshot_build" and st["2026/10/20/[3]abc"][
        "init_cpu_parts"
    ] == ["0xd40"]
    assert (
        st["2026/10/20/[3]def"]["kind"] == "restore"
        and st["2026/10/20/[3]def"]["init_cpu_parts"] == []
    )


def test_extract_reads_the_pinned_commit_not_the_working_tree(tmp_path):
    repo = tmp_path / "repo"
    d = repo / "runs" / "r1"
    (d / "logs").mkdir(parents=True)
    rec = {
        "arm": "ctl",
        "round": 0,
        "expected_class": "warm",
        "cpu_part": "0xd40",
        "latency_ms": 250.0,
        "report": {"Duration": 250.0},
        "log_tail": "",
    }
    (d / "invocations.jsonl").write_text(json.dumps(rec) + "\n")
    (d / "conditions.json").write_text(
        json.dumps(_cond(["ctl"])[0] | {"rounds_meta": [], "functions": {}})
    )
    (d / "logs" / "f.jsonl.gz").write_bytes(
        gzip.compress(b'{"logStreamName":"s","message":"INIT_START\\n"}\n')
    )

    def git(*a):
        return subprocess.run(
            ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
        )

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    sha = git("rev-parse", "HEAD").stdout.strip()
    # Uncommitted edit: extract must not see it.
    (d / "invocations.jsonl").write_text(json.dumps(rec | {"cpu_part": "0xd0c"}) + "\n")
    s = src("R") | {"commit": sha, "dirs": ["runs/r1"]}
    rows, streams, conds = hostvar.extract({"runs": [s]}, str(repo))
    assert [r["cpu_part"] for r in rows] == ["0xd40"]
    assert len(streams) == 1 and conds["R"][0]["region"] == "us-east-2"


# ------------------------------------------------------------------ committed result


RUN = os.path.join(HV, "results")


def _committed():
    out = []
    for d in sorted(os.listdir(RUN)) if os.path.isdir(RUN) else []:
        p = os.path.join(RUN, d, "results.json")
        if os.path.exists(p):
            out.append(p)
    return out


@pytest.mark.parametrize("path", _committed())
def test_committed_result_regenerates_and_names_no_generation(path, tmp_path):
    run = os.path.dirname(path)
    res = json.load(open(path))
    assert not GENERATION.search(json.dumps(res)) and not ACCOUNT.search(json.dumps(res))
    for k in ("rows.jsonl.gz", "streams.jsonl.gz", "conditions.json", "sources.json"):
        os.symlink(os.path.join(os.path.abspath(run), k), tmp_path / k)
    fixed = os.path.join(run, "offline-summary.json")
    argv = [
        "analyze",
        str(tmp_path),
        "--experiment-commit",
        res["commits"]["experiment"],
        "--baseline-commit",
        res["commits"]["baseline"],
    ]
    if os.path.exists(fixed):
        argv += ["--fixed-host", fixed]
    assert hostvar.main(argv) == 0
    again = json.load(open(tmp_path / "results.json"))
    assert again == {k: v for k, v in res.items() if k != "published"}


@pytest.mark.parametrize("path", _committed()[-1:])
def test_readme_tables_are_the_newest_result(path):
    doc = open(os.path.join(HV, "README.md")).read()
    assert hostvar.tables(json.load(open(path))) in doc
    assert not GENERATION.search(doc)
