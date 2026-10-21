"""examples/lambda-kleidiai/experiments/e2-onezone-ia: deploy checks, comparison, decision rule, cost and hashes.

No AWS and no model: the deploy test drives stand-in clients, and the summary tests build a synthetic bench.py run
directory in the shape P1's harness writes.
"""

from __future__ import annotations

import base64
import gzip
import importlib
import importlib.util
import json
import os
import sys
import types

import pytest

E2 = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "lambda-kleidiai",
        "experiments",
        "e2-onezone-ia",
    )
)
GGUF_SHA = "8dbb8872e6af312804b8d9d82b6d023bd025a0cfc81d55703b79837d0f042d8e"
SIZE = 2076712416


@pytest.fixture(autouse=True)
def _sdk(monkeypatch):
    if not (importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore")):
        boto3 = types.ModuleType("boto3")
        boto3.Session = lambda **kw: None
        botocore = types.ModuleType("botocore")
        exc = types.ModuleType("botocore.exceptions")
        exc.ClientError = type("ClientError", (Exception,), {"response": {}})
        cfg = types.ModuleType("botocore.config")
        cfg.Config = lambda **kw: None
        for name, mod in {
            "boto3": boto3,
            "botocore": botocore,
            "botocore.exceptions": exc,
            "botocore.config": cfg,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
    # Both experiment modules and their baseline counterparts are called deploy/summarize: load E2's by path.
    for m in ("deploy", "summarize", "e2_deploy", "e2_summarize"):
        monkeypatch.delitem(sys.modules, m, raising=False)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"e2_{name}", os.path.join(E2, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _b64(hexs: str) -> str:
    return base64.b64encode(bytes.fromhex(hexs)).decode()


def test_object_problems_requires_class_size_and_full_object_sha256():
    d = _load("deploy")
    good = {
        "StorageClass": "ONEZONE_IA",
        "ContentLength": SIZE,
        "ChecksumSHA256": _b64(GGUF_SHA),
        "ChecksumType": "FULL_OBJECT",
    }
    assert d.object_problems(good, "ONEZONE_IA", SIZE, GGUF_SHA) == []
    # S3 omits StorageClass for STANDARD objects.
    std = {k: v for k, v in good.items() if k != "StorageClass"}
    assert d.object_problems(std, "STANDARD", SIZE, GGUF_SHA) == []
    assert d.object_problems(std, "ONEZONE_IA", SIZE, GGUF_SHA)
    bad = good | {"ChecksumSHA256": _b64("00" * 32)}
    assert any("ChecksumSHA256" in p for p in d.object_problems(bad, "ONEZONE_IA", SIZE, GGUF_SHA))
    comp = good | {"ChecksumType": "COMPOSITE"}
    assert any("FULL_OBJECT" in p for p in d.object_problems(comp, "ONEZONE_IA", SIZE, GGUF_SHA))
    assert d.object_problems(good | {"ContentLength": 1}, "ONEZONE_IA", SIZE, GGUF_SHA)


def test_weight_keys_differ_only_by_storage_class():
    d = _load("deploy")
    a, b = (d.weight_key(GGUF_SHA, sc) for sc in ("STANDARD", "ONEZONE_IA"))
    assert a != b
    assert a.replace("/standard/", "/x/") == b.replace("/onezone_ia/", "/x/")


def test_function_configs_differ_only_in_name_key_and_description():
    d = _load("deploy")
    st = {
        "prefix": "lkai-e2",
        "bucket": "b",
        "manifest": {"function_zip": {"sha256": d.PINNED_ZIP_SHA256}},
        "weights": {a: {"key": d.weight_key(GGUF_SHA, sc)} for a, sc in d.ARMS.items()},
    }
    s, o = (d.function_config(st, a, "role") for a in ("std", "ozia"))
    diff = {k for k in s if s[k] != o[k]}
    assert diff == {"FunctionName", "Environment", "Description"}
    es, eo = s["Environment"]["Variables"], o["Environment"]["Variables"]
    assert {k for k in es if es[k] != eo[k]} == {"MODEL_KEY"}
    assert es["LKAI_PHASES"] == "1"
    assert s["SnapStart"] == {"ApplyOn": "PublishedVersions"}


def test_upload_refuses_a_copy_whose_checksum_does_not_match(monkeypatch):
    d = _load("deploy")
    calls = []

    class S3:
        def get_object(self, **kw):
            m = {
                "function_zip": {"sha256": d.PINNED_ZIP_SHA256},
                "model": {"gguf_sha256": GGUF_SHA, "gguf_bytes": SIZE},
            }
            return {"Body": types.SimpleNamespace(read=lambda: json.dumps(m).encode())}

        def put_bucket_tagging(self, **kw):
            pass

        def head_object(self, **kw):
            k = kw["Key"]
            if k.endswith(".gguf"):
                sc = "ONEZONE_IA" if "/onezone_ia/" in k else None
                h = {
                    "ContentLength": SIZE,
                    "ChecksumSHA256": _b64("11" * 32),
                    "ChecksumType": "FULL_OBJECT",
                    "LastModified": "t",
                }
                return h | ({"StorageClass": sc} if sc else {})
            return {}

        def copy_object(self, **kw):
            calls.append(kw)

    sess = types.SimpleNamespace(client=lambda *a, **k: S3(), region_name="us-east-2")
    monkeypatch.setattr(d.base, "ensure_bucket", lambda *a: "b")
    with pytest.raises(SystemExit, match="ChecksumSHA256"):
        d.upload(sess, {"prefix": "lkai-e2"}, "src")


def test_decide_maps_interval_to_status():
    s = _load("summarize")
    assert s.decide([0.99, 1.01]) == "no-change"
    assert s.decide([0.90, 0.97]) == "improved"
    assert s.decide([1.03, 1.10]) == "regressed"
    assert s.decide([0.97, 1.01]) == "inconclusive"
    assert s.decide(None) == "inconclusive"


def test_boot_ratio_is_seeded_and_brackets_the_point_estimate():
    s = _load("summarize")
    a = [25.0 + 0.01 * i for i in range(50)]
    b = [25.0 + 0.01 * i for i in range(50)]
    ci = s.boot_ratio(a, b)
    assert ci == s.boot_ratio(a, b)
    assert ci[0] <= 1.0 <= ci[1]
    assert s.boot_ratio([1.0], b) is None


def test_download_cost_per_class():
    s = _load("summarize")
    gets = 31
    std = s.download_cost("STANDARD", SIZE, gets)
    oz = s.download_cost("ONEZONE_IA", SIZE, gets)
    assert std == pytest.approx(32 * 0.0004 / 1000)
    assert oz == pytest.approx(32 * 0.001 / 1000 + SIZE / 1e9 * 0.01)
    assert oz > std


def _row(arm, rnd, wave, *, wd=25.0, cpu="0xd40", answers=None, rid="r1", lat=240.0):
    if wave == "cold":
        report = {
            "Init Duration": 800.0,
            "Duration": 35000.0,
            "Billed Duration": 35800,
            "Max Memory Used": 5100,
        }
    elif wave == "restore":
        report = {
            "Restore Duration": 3270.0,
            "Billed Restore Duration": 400,
            "Duration": 700.0,
            "Billed Duration": 701,
            "Max Memory Used": 5100,
        }
    else:
        report = {"Duration": 660.0, "Billed Duration": 661, "Max Memory Used": 5100}
    lk = {
        "inv": {"handler_s": 0.3, "wait_load_s": 0.0, "restore_to_invoke_s": 0.01, "cpu_part": cpu}
    }
    if wave == "cold":
        lk |= {
            "init": {
                "weight_download": wd,
                "model_load_mmap": 8.0,
                "model_ready": 34.0,
                "imports": 0.7,
            },
            "counters": {"weight_mib_s": round(SIZE / 2**20 / wd, 1)},
        }
    return {
        "arm": arm,
        "round": rnd,
        "wave": wave,
        "seq": 0,
        "request_id": rid,
        "rt_s": 1.0,
        "error": None,
        "report": report,
        "expected_class": wave,
        "observed_class": wave,
        "class_mismatch": False,
        "cpu_part": cpu,
        "latency_ms": lat,
        "answers": answers or {"q": {"type": "noul", "noul": 0.5}},
        "lkai": lk,
    }


def _run(tmp_path, ozia_wd):
    rows = []
    for rnd in range(2):
        for arm in ("std", "ozia"):
            for i in range(10):
                wd = (ozia_wd if arm == "ozia" else 25.0) + 0.01 * i
                rows.append(_row(arm, rnd, "cold", wd=wd))
                rows.append(_row(arm, rnd, "restore"))
                rows.append(_row(arm, rnd, "warm", rid="r2", lat=660.0))
    fns = {
        a: {
            "FunctionName": f"lkai-e2-{a}",
            "Runtime": "python3.12",
            "Architectures": ["arm64"],
            "MemorySize": 10240,
            "EphemeralStorage": {"Size": 512},
            "Timeout": 900,
            "SnapStart": {"ApplyOn": "PublishedVersions"},
            "Handler": "lkai_entry.handler",
            "environment": {"LKAI_PHASES": "1"},
        }
        for a in ("std", "ozia")
    }
    cond = {
        "region": "us-east-2",
        "rounds": 2,
        "concurrency": 10,
        "warm_per_arm_per_round": 1,
        "arms": ["std", "ozia"],
        "probe": "r1",
        "requests_file": "requests.json",
        "started": "t0",
        "finished": "t1",
        "requests": {"requests": [{"id": "r1"}, {"id": "r2"}]},
        "client": {},
        "runtime_version_arn": "arn:aws:lambda:us-east-2::runtime:abc",
        "functions": fns,
        "build_manifest": {
            "repo_commit": "4412b31",
            "repo_dirty": False,
            "model": {"id": "m", "gguf_bytes": SIZE, "gguf_sha256": GGUF_SHA},
            "llama_cpp": {"commit": "f74e"},
            "function_zip": {"sha256": "26eb", "source_commit": "fd96d7f", "inputs_sha256": "k"},
            "images": {},
            "toolchain": [],
            "assets_sha256": {},
        },
        "rounds_meta": [
            {
                "t_versions_deleted": 100.0,
                "published": {
                    "std": {"t_publish": 0.0, "version": "1"},
                    "ozia": {"t_publish": 0.0, "version": "1"},
                },
            }
        ],
    }
    (tmp_path / "conditions.json").write_text(json.dumps(cond))
    (tmp_path / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "logs").mkdir()
    for arm, wd in (("std", 24.6), ("ozia", 24.7)):
        with gzip.open(tmp_path / "logs" / f"lkai-e2-{arm}.jsonl.gz", "wt") as f:
            for w in (1, 2):
                rec = {
                    "kind": "before_snapshot",
                    "wall_ns": w,
                    "init": {"weight_download": wd, "model_ready": 33.0},
                }
                f.write(
                    json.dumps(
                        {
                            "timestamp": 0,
                            "logStreamName": "2026/10/20/[1]x",
                            "message": "LKAI " + json.dumps(rec),
                        }
                    )
                    + "\n"
                )
    ref = {
        "outputs": {
            "r1": {"0xd40": {"q": {"type": "noul", "noul": 0.5}}},
            "r2": {"0xd40": {"q": {"type": "noul", "noul": 0.5}}},
        }
    }
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    dep = {
        "weights": {
            a: {
                "storage_class": sc,
                "key": f"weights/{GGUF_SHA}/{sc.lower()}/x.gguf",
                "checksum_sha256_b64": _b64(GGUF_SHA),
                "checksum_type": "FULL_OBJECT",
                "sha256_matches_manifest": True,
            }
            for a, sc in (("std", "STANDARD"), ("ozia", "ONEZONE_IA"))
        }
    }
    (tmp_path / "dep.json").write_text(json.dumps(dep))
    return str(tmp_path)


@pytest.mark.parametrize(
    "ozia_wd,want", [(25.0, "no-change"), (27.0, "regressed"), (22.0, "improved")]
)
def test_summarize_end_to_end(tmp_path, ozia_wd, want):
    s = _load("summarize")
    run = _run(tmp_path, ozia_wd)
    meta = json.load(open(os.path.join(E2, "meta.json")))
    res = s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)
    assert res["status"] == want
    assert res["primary_arm"] == "ozia"
    wd = res["comparison"]["cold"]["weight_download"]
    assert wd["n"] == [20, 20]
    assert wd["ratio_ci95"][0] <= wd["ratio_p50"] <= wd["ratio_ci95"][1]
    assert res["comparison"]["snapshot_init"]["weight_download"]["n"] == [2, 2]
    c = res["cost"]
    assert c["by_arm"]["ozia"]["weight_downloads"] == 22  # 20 cold + 2 snapshot inits
    assert c["by_arm"]["std"]["s3_retrieval_usd"] == 0.0
    assert c["by_arm"]["ozia"]["s3_retrieval_usd"] == pytest.approx(
        22 * SIZE / 1e9 * 0.01, abs=1e-4
    )
    assert c["s3_onezone_ia_30_day_minimum_usd"] == pytest.approx(SIZE / 1e9 * 0.01, abs=1e-4)
    pc = c["per_storage_class"]["s3_usd_per_1000_cold_starts"]
    assert pc["ONEZONE_IA"] > pc["EXPRESS_ONEZONE"] > 0 and pc["ONEZONE_IA"] > pc["STANDARD"]
    k = res["correctness"]
    assert k["compared"] == k["bit_identical"] == 120
    assert k["answer_sha256"]["std"]["sha256"] == k["answer_sha256"]["ozia"]["sha256"]
    assert (
        k["weights"]["std"]["s3_checksum_sha256_b64"]
        == k["weights"]["ozia"]["s3_checksum_sha256_b64"]
    )


def test_run_script_parses_and_cleans_up_before_benchmark():
    import subprocess

    path = os.path.join(E2, "run.sh")
    subprocess.run(["bash", "-n", path], check=True)
    src = open(path).read()
    assert src.index("trap cleanup_versions EXIT") < src.index('"$BASE/bench.py"')
    assert "--arms std,ozia" in src


def test_answer_hash_detects_a_changed_answer(tmp_path):
    s = _load("summarize")
    rows = [
        _row("std", 0, "warm"),
        _row("ozia", 0, "warm", answers={"q": {"type": "noul", "noul": 0.51}}),
    ]
    assert s.answer_hash(rows, "std")["sha256"] != s.answer_hash(rows, "ozia")["sha256"]


def test_cluster_boot_ratio_is_paired_seeded_and_wider_than_iid_under_round_shifts():
    s = _load("summarize")
    # Every round shifts both arms by the same amount: the shared-wave dependence the cluster bootstrap must keep.
    shift = {0: 0.0, 1: 0.5, 2: -0.4, 3: 0.3, 4: -0.2}
    a = {r: [25.0 + d + 0.01 * i for i in range(10)] for r, d in shift.items()}
    b = {r: [25.0 + d + 0.01 * i for i in range(10)] for r, d in shift.items()}
    ci = s.cluster_boot_ratio(a, b)
    assert ci == s.cluster_boot_ratio(a, b)  # seeded
    # Identical arms, paired by round: every resample has ratio exactly 1.
    assert ci == [1.0, 1.0]
    iid = s.boot_ratio([v for r in a for v in a[r]], [v for r in b for v in b[r]])
    assert iid[0] < 1.0 < iid[1]
    # An arm that is 1% slower in every round stays 1% slower in every paired resample.
    slow = {r: [v * 1.01 for v in vs] for r, vs in a.items()}
    lo, hi = s.cluster_boot_ratio(slow, b)
    assert lo == pytest.approx(1.01, abs=1e-3) and hi == pytest.approx(1.01, abs=1e-3)
    # Only shared rounds count; fewer than 2 shared rounds gives no interval.
    assert s.cluster_boot_ratio({0: [1.0], 9: [1.0]}, {0: [1.0], 8: [1.0]}) is None


def test_comparison_reports_round_cluster_sensitivity_without_changing_status(tmp_path):
    s = _load("summarize")
    run = _run(tmp_path, 25.0)
    meta = json.load(open(os.path.join(E2, "meta.json")))
    res = s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)
    cmp_ = res["comparison"]
    sens = cmp_["sensitivity"]
    assert sens["rounds"] == 2
    assert sens["ci95"] == cmp_["cold"]["weight_download"]["ratio_ci95_round_cluster"]
    assert sens["outcome_under_same_band"] == s.decide(sens["ci95"])
    assert sens["agrees_with_primary"] is (sens["outcome_under_same_band"] == cmp_["outcome"])
    assert (
        res["status"] == cmp_["outcome"] == s.decide(cmp_["cold"]["weight_download"]["ratio_ci95"])
    )
    assert "ratio_ci95_round_cluster" in cmp_["cold"]["by_cpu_part"]["0xd40"]["weight_download"]


def test_total_usd_is_exactly_the_sum_of_named_charges(tmp_path):
    s = _load("summarize")
    run = _run(tmp_path, 25.0)
    meta = json.load(open(os.path.join(E2, "meta.json")))
    c = s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)[
        "cost"
    ]
    assert c["total_components"] == list(s.CHARGE_KEYS)
    assert c["total_usd"] == pytest.approx(sum(c[k] for k in s.CHARGE_KEYS), abs=1e-9)
    # Every *_usd key is either a named charge or excluded on purpose (a monthly rate, or the total itself).
    rest = {k for k in c if k.endswith("_usd")} - set(s.CHARGE_KEYS)
    assert rest == {"total_usd"}


def test_cost_refuses_an_unclassified_usd_key(tmp_path, monkeypatch):
    s = _load("summarize")
    run = _run(tmp_path, 25.0)
    meta = json.load(open(os.path.join(E2, "meta.json")))
    real = s.base.costs
    monkeypatch.setattr(s.base, "costs", lambda *a: real(*a) | {"new_fee_usd": 1.0})
    with pytest.raises(ValueError, match="new_fee_usd"):
        s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)


def test_committed_result_total_matches_its_components():
    p = os.path.join(E2, "results", "20261020T1125Z-e2", "results.json")
    c = json.load(open(p))["cost"]
    assert c["total_usd"] == pytest.approx(sum(c[k] for k in c["total_components"]), abs=1e-9)
    assert c["total_usd"] == pytest.approx(3.639428, abs=1e-6)


def _host_log(tmp_path, fn, envs):
    """envs: [(stream, cpu_part, variant, snapshot?)] -> a saved CloudWatch log in bench.py's shape."""
    with gzip.open(tmp_path / "logs" / f"{fn}.jsonl.gz", "wt") as f:
        for i, (stream, part, variant, snap) in enumerate(envs):
            msgs = [
                f"x decider.engine init cpu_part={part} ggml_cpu={variant} threads=6 ubatch=512"
            ]
            if snap:
                msgs.append("LKAI " + json.dumps({"kind": "before_snapshot", "wall_ns": i + 1}))
            for m in msgs:
                f.write(json.dumps({"timestamp": 0, "logStreamName": stream, "message": m}) + "\n")


def test_host_variability_reports_cpu_parts_unknowns_and_within_part_outcome(tmp_path):
    s = _load("summarize")
    run = _run(tmp_path, 25.0)
    rows = [json.loads(x) for x in open(os.path.join(run, "invocations.jsonl"))]
    # Put two std cold starts on another CPU part; give every row an env id.
    for i, r in enumerate(rows):
        r["lkai"]["env_id"] = f"e{i}"
        r["timing"] = {"threads": 6}
        r["report"]["Memory Size"] = 10240.0
    colds = [r for r in rows if r["arm"] == "std" and r["expected_class"] == "cold"]
    for r in colds[:2]:
        r["cpu_part"] = "0xd0c"
    (tmp_path / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for fn in ("lkai-e2-std", "lkai-e2-ozia"):
        _host_log(
            tmp_path,
            fn,
            [("2026/10/20/[1]a", "0xd0c", "v82", True), ("2026/10/20/[1]b", "0xd40", "v86", True)],
        )
    meta = json.load(open(os.path.join(E2, "meta.json")))
    res = s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)
    h = res["host_variability"]
    o = h["observed"]
    assert o["cpu_part_by_arm_and_class"]["std"]["cold"] == {"0xd0c": 2, "0xd40": 18}
    assert o["cpu_part_by_arm_and_class"]["ozia"]["cold"] == {"0xd40": 20}
    assert o["environments_with_more_than_one_cpu_part"] == 0
    assert o["ggml_cpu_variant_by_cpu_part"]["std"] == {"0xd0c": ["v82"], "0xd40": ["v86"]}
    assert o["snapshot_init_cpu_parts"]["ozia"] == {"0xd0c": 1, "0xd40": 1}
    assert o["threads"] == [6]
    # What was not logged is named as unknown; no processor generation is claimed.
    assert any("implementer" in u for u in h["unobserved"])
    assert any("HWCAP" in u for u in h["unobserved"])
    assert h["generation"].startswith("unknown")
    assert "graviton" not in json.dumps(h).lower()
    w = h["primary_within_dominant_cpu_part"]
    assert w["cpu_part"] == "0xd40" and w["n"] == [18, 20]
    assert w["outcome"] == s.decide(w["ratio_ci95"]) == "no-change"
    assert w["agrees_with_primary"] is True
    cells = h["primary_by_round_and_cpu_part"]
    assert cells["shared_cells"] == 2
    assert cells["ratio_range"] == [0.999601, 1.0]
    assert cells["delta_signs"] == ["negative", "zero"]
    assert cells["all_inside_equivalence_band"] is True
    assert h["rerun_assessment"]["needed"] is False
    assert h["rerun_assessment"]["decision"] == "no rerun"


def test_host_variability_flags_a_conclusion_that_flips_within_one_cpu_part(tmp_path):
    s = _load("summarize")
    run = _run(tmp_path, 25.0)
    rows = [json.loads(x) for x in open(os.path.join(run, "invocations.jsonl"))]
    # Pooled the arms look equal, but within the dominant part One Zone-IA is 10% slower: a host-mix artefact.
    for r in rows:
        if r["expected_class"] != "cold":
            continue
        i = int(r["lkai"]["init"]["weight_download"] * 100) % 10
        if r["arm"] == "ozia":
            r["cpu_part"] = "0xd40" if i < 6 else "0xd4f"
            r["lkai"]["init"]["weight_download"] = 27.5 if i < 6 else 22.0
        else:
            r["cpu_part"] = "0xd40" if i < 6 else "0xd4f"
    (tmp_path / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = json.load(open(os.path.join(E2, "meta.json")))
    res = s.summarize(run, str(tmp_path / "ref.json"), str(tmp_path / "dep.json"), "aac59d4", meta)
    w = res["host_variability"]["primary_within_dominant_cpu_part"]
    assert w["cpu_part"] == "0xd40"
    assert w["outcome"] == "regressed"
    assert w["agrees_with_primary"] is False
    assert res["host_variability"]["rerun_assessment"]["needed"] is True
    assert res["host_variability"]["rerun_assessment"]["decision"] == "rerun affected comparison"


def test_round_cpu_primary_records_range_signs_and_sparse_cells():
    s = _load("summarize")
    rows = [
        _row("std", 0, "cold", wd=25.0, cpu="a"),
        _row("ozia", 0, "cold", wd=24.9, cpu="a"),
        _row("std", 1, "cold", wd=25.0, cpu="a"),
        _row("ozia", 1, "cold", wd=25.1, cpu="a"),
        _row("std", 0, "cold", wd=25.0, cpu="sparse"),
    ]
    out = s._round_cpu_primary(rows)
    assert out["shared_cells"] == 2
    assert out["ratio_range"] == [0.996, 1.004]
    assert out["delta_s_range"] == [-0.1, 0.1]
    assert out["delta_signs"] == ["negative", "positive"]
    assert out["all_inside_equivalence_band"] is True
    assert all(c["cpu_part"] == "a" for c in out["cells"])
    assert "absent from either arm" in out["missing"]


def test_committed_result_names_cpu_parts_not_generations():
    base = os.path.join(E2, "results", "20261020T1125Z-e2")
    res = json.load(open(os.path.join(base, "results.json")))
    h = res["host_variability"]
    assert h["primary_within_dominant_cpu_part"]["outcome"] == res["status"] == "no-change"
    assert h["observed"]["environments_with_more_than_one_cpu_part"] == 0
    assert h["primary_by_round_and_cpu_part"]["shared_cells"] == 8
    assert h["primary_by_round_and_cpu_part"]["ratio_range"] == [0.994925, 1.009692]
    assert h["primary_by_round_and_cpu_part"]["delta_signs"] == ["negative", "positive"]
    assert h["primary_by_round_and_cpu_part"]["all_inside_equivalence_band"] is True
    assert h["rerun_assessment"]["needed"] is False
    for p in (
        os.path.join(base, "results.json"),
        os.path.join(base, "meta.json"),
        os.path.join(E2, "README.md"),
    ):
        assert "graviton" not in open(p).read().lower(), p
