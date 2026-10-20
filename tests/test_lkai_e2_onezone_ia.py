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
