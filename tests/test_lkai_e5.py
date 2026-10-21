"""examples/lambda-kleidiai/e5: the no-vocab GGUF conversion, its arm-vs-arm summary and its deployment wiring.

The conversion is checked on a small GGUF written with gguf-py when it is importable (llama.cpp's gguf-py is not a
repository dependency); the summary and deployment wiring need neither llama.cpp nor AWS.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest

E5 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e5")
)
BASE = os.path.join(os.path.dirname(E5), "baseline")
sys.path.insert(0, E5)
sys.path.insert(0, BASE)


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    if importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore"):
        return
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
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
    for m in ("deploy", "deploy_e5"):
        monkeypatch.delitem(sys.modules, m, raising=False)


def _row(arm, wave, cpu, *, init=None, report=None, lat=500.0):
    report = report or {"Duration": lat + 10, "Max Memory Used": 4000}
    obs = (
        "restore"
        if "Restore Duration" in report
        else "cold"
        if "Init Duration" in report
        else "warm"
    )
    return {
        "arm": arm,
        "expected_class": wave,
        "observed_class": obs,
        "class_mismatch": obs != wave,
        "error": None,
        "cpu_part": cpu,
        "report": report,
        "latency_ms": lat,
        **({"lkai": {"init": init}} if init else {}),
    }


def test_compare_arms_matches_cpu_parts_and_pools():
    import summarize_e5

    rows = []
    # CPU part 0xd40: e5 loads 0.5 s faster. Part 0xd0c: only e5 drew it, so it must not enter the comparison.
    for arm, mmap in (("base", 8.0), ("e5", 7.5)):
        for _ in range(3):
            rows.append(
                _row(
                    arm,
                    "cold",
                    "0xd40",
                    init={"model_load_mmap": mmap},
                    report={"Init Duration": 800.0, "Duration": 30000.0, "Max Memory Used": 5000},
                )
            )
    rows.append(
        _row(
            "e5",
            "cold",
            "0xd0c",
            init={"model_load_mmap": 1.0},
            report={"Init Duration": 1.0, "Duration": 1.0, "Max Memory Used": 1},
        )
    )
    rows.append(_row("e5", "cold", "0xd40", report={"Duration": 1.0}))  # mismatch: excluded
    out = summarize_e5.compare_arms(rows)
    m = out["cold"]["model_load_mmap"]
    assert list(m["by_cpu_part"]) == ["0xd40"]
    assert m["by_cpu_part"]["0xd40"]["n"] == [3, 3]
    assert m["pooled_delta_p50"] == pytest.approx(-0.5)
    assert out["cold"]["report_max_memory_mb"]["pooled_delta_p50"] == 0
    assert "restore" not in out


def test_deploy_e5_arms_differ_only_in_model_key(monkeypatch):
    import deploy
    import deploy_e5

    seen = []

    def fake_functions(sess, st):
        assert set(deploy.ARMS) == {next(iter(deploy.ARMS))} and set(deploy.ARMS.values()) == {"1"}
        arm = next(iter(deploy.ARMS))
        seen.append((arm, st["model_key"]))
        st.setdefault("functions", {})[arm] = {"name": f"lkai-e5-{arm}"}

    class FakeLam:
        def __init__(self):
            self.pins = []

        def put_runtime_management_config(self, **kw):
            self.pins.append(kw)

        def get_function_configuration(self, FunctionName):
            arm = FunctionName.rsplit("-", 1)[1]
            return {"Environment": {"Variables": {"MODEL_KEY": f"weights/{arm}.gguf"}}}

    class FakeSess:
        def client(self, name):
            return FakeLam()

    monkeypatch.setattr(deploy, "functions", fake_functions)
    st = {
        "prefix": "lkai-e5",
        "region": "us-east-2",
        "bucket": "b",
        "model_key_base": "weights/base.gguf",
        "model_key_e5": "weights/e5.gguf",
    }
    deploy_e5.functions(FakeSess(), st)
    assert seen == [("base", "weights/base.gguf"), ("e5", "weights/e5.gguf")]
    assert set(st["functions"]) == {"base", "e5"}
    assert "model_key" not in st
    assert deploy.ARMS == {"base": "1", "e5": "1"}
    assert deploy.ensure_role.__module__ == "deploy" and deploy.pin_runtime.__module__ == "deploy"


def test_deploy_e5_refuses_wrong_model_key(monkeypatch):
    import deploy
    import deploy_e5

    def fake_functions(sess, st):
        arm = next(iter(deploy.ARMS))
        st.setdefault("functions", {})[arm] = {"name": f"lkai-e5-{arm}"}

    class FakeLam:
        def get_function_configuration(self, FunctionName):
            return {"Environment": {"Variables": {"MODEL_KEY": "weights/base.gguf"}}}

    class FakeSess:
        def client(self, name):
            return FakeLam()

    monkeypatch.setattr(deploy, "functions", fake_functions)
    st = {
        "prefix": "p",
        "region": "r",
        "bucket": "b",
        "model_key_base": "weights/base.gguf",
        "model_key_e5": "weights/e5.gguf",
    }
    with pytest.raises(SystemExit, match="lkai-e5-e5"):
        deploy_e5.functions(FakeSess(), st)


def _manifest_fixture(tmp_path, *, converted_record_sha=True):
    import manifest_e5

    build = tmp_path / "build"
    model = build / "model"
    model.mkdir(parents=True)
    (build / "function.zip").write_bytes(b"zip")
    base = model / "s2b-h512-Q8_0.gguf"
    converted = model / "s2b-h512-Q8_0-novocab.gguf"
    base.write_bytes(b"base")
    converted.write_bytes(b"converted")
    manifest = {
        "function_zip": {"sha256": manifest_e5.sha(str(build / "function.zip"))},
        "model": {"gguf_sha256": manifest_e5.sha(str(base))},
        "llama_cpp": {"commit": "expected"},
        "images": {
            "weights": "python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b"
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    record_sha = manifest_e5.sha(str(converted)) if converted_record_sha else "stale"
    (model / "convert.json").write_text(
        json.dumps(
            {
                "image": "python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b",
                "converted_sha256": record_sha,
                "tools": {"llama_cpp": "wrong", "gguf_py": "0.19.0"},
                "tensors": {
                    "identical": True,
                    "count": [335, 335],
                    "digest": ["same", "same"],
                    "data_offset": [32, 32],
                },
                "removed_keys": [],
                "added_keys": {},
            }
        )
    )
    return manifest_e5, build, manifest_path, converted


def test_manifest_refuses_converter_from_wrong_llama_checkout(tmp_path):
    manifest_e5, build, manifest_path, _ = _manifest_fixture(tmp_path)

    with pytest.raises(
        SystemExit, match=r"conversion used llama\.cpp 'wrong', expected expected"
    ):
        manifest_e5.main(str(build), str(manifest_path))


def test_manifest_refuses_stale_conversion_record(tmp_path):
    manifest_e5, build, manifest_path, _ = _manifest_fixture(tmp_path, converted_record_sha=False)
    record = json.loads((build / "model" / "convert.json").read_text())
    record["tools"]["llama_cpp"] = "expected"
    (build / "model" / "convert.json").write_text(json.dumps(record))

    with pytest.raises(SystemExit, match="conversion record names converted sha256 'stale'"):
        manifest_e5.main(str(build), str(manifest_path))


def test_host_variability_splits_by_cpu_part_and_flags_mixed_signs():
    import host_variability as H

    def cold(arm, part, mmap, rd, env):
        r = _row(
            arm,
            "cold",
            part,
            init={"model_load_mmap": mmap},
            report={"Init Duration": 500.0, "Duration": 1000.0 * mmap, "Max Memory Used": 5000},
        )
        r["round"], r["lkai"]["env_id"] = rd, env
        return r

    rows = []
    for rd in range(2):
        for i in range(3):
            # part A: e5 faster; part B: e5 slower -> signs disagree
            rows.append(cold("base", "0xaaa", 8.0, rd, f"a-b-{rd}-{i}"))
            rows.append(cold("e5", "0xaaa", 7.5, rd, f"a-e-{rd}-{i}"))
            rows.append(cold("base", "0xbbb", 7.0, rd, f"b-b-{rd}-{i}"))
            rows.append(cold("e5", "0xbbb", 7.2, rd, f"b-e-{rd}-{i}"))
    rows.append(cold("e5", "0xccc", 1.0, 0, "c"))  # only one arm drew it: not compared
    by = H.compare(rows)
    m = by["cold"]["model_load_mmap"]["by_cpu_part"]
    assert set(m) == {"0xaaa", "0xbbb"}
    assert m["0xaaa"]["delta_p50"] == pytest.approx(-0.5)
    assert m["0xaaa"]["rounds_e5_below_base"] == [2, 2]
    assert m["0xbbb"]["rounds_e5_below_base"] == [0, 2]
    v = H.verdict(by)["cold.model_load_mmap"]
    assert v["same_sign_in_every_part"] is False
    assert v["parts_with_n_ge_min"] == ["0xaaa", "0xbbb"]
    mx = H.mix(rows)
    assert mx["environments_with_more_than_one_cpu_part"] == 0
    assert mx["invocations"]["cold:e5:0xccc"] == 1


def test_host_variability_labels_unrecorded_fields_unknown_and_infers_no_generation():
    import host_variability as H

    for field in (
        "cpu_implementer",
        "cpu_feature_flags",
        "hwcap_hwcap2",
        "online_core_count",
        "kernel",
    ):
        assert H.METADATA[field].startswith("unknown")
    assert H.METADATA["graviton_generation"].startswith("unknown")
    assert H.METADATA["cpu_part"].startswith("observed")


def test_host_variability_method_states_sequential_within_round(tmp_path):
    import host_variability as H

    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    rows = []
    for arm, mmap in (("base", 8.0), ("e5", 7.5)):
        for i in range(3):
            r = _row(
                arm,
                "cold",
                "0xd40",
                init={"model_load_mmap": mmap},
                report={"Init Duration": 500.0, "Duration": 1000.0, "Max Memory Used": 5000},
            )
            r["round"], r["lkai"]["env_id"] = 0, f"{arm}-{i}"
            rows.append(r)
    (run / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    method = H.assess(str(run))["method"]
    assert "alternates between rounds" in method
    assert "cold and restore waves run one arm after the other" in method
    assert "warm passes alternate arm order pass by pass" in method
    assert "time-within-round drift remains a limitation for cold and restore" in method


def test_run_rebuilds_when_conversion_record_is_stale():
    run = open(os.path.join(E5, "run.sh")).read()
    assert "conversion_current()" in run
    assert 'rec.get("converted_sha256") == h.hexdigest()' in run
    assert 'rm -f "$CONVERTED" "$CONVERT_RECORD"' in run


def test_container_conversion_records_output_digest():
    script = open(os.path.join(E5, "container_convert.sh")).read()
    assert 'python - "$rec" "$out"' in script
    assert 'hashlib.file_digest(fh, "sha256").hexdigest()' in script
    assert '"converted_sha256": converted_sha256' in script


def test_benchmark_docs_do_not_claim_within_round_pairing():
    readme = open(os.path.join(E5, "README.md")).read()
    run = open(os.path.join(E5, "run.sh")).read()
    assert "alternating between rounds" in readme
    assert "cold and restore waves\nrun one arm after the other" in readme
    assert "warm passes alternate arm order pass by pass" in readme
    assert "alternating-order" in run


def test_bench_alternates_warm_pass_order_within_round():
    # The method prose above claims warm passes alternate arm order within a round; pin that to bench.py.
    bench = open(os.path.join(E5, "..", "baseline", "bench.py")).read()
    assert "order = arms if rnd % 2 == 0 else arms[::-1]" in bench
    assert "for arm in order if i % 2 == 0 else order[::-1]:" in bench


def test_convert_no_vocab_keeps_tensor_bytes(tmp_path):
    gguf = pytest.importorskip("gguf", reason="llama.cpp's gguf-py is not a repository dependency")
    import convert_no_vocab
    import numpy as np
    import tensor_identity

    src, dst = tmp_path / "a.gguf", tmp_path / "b.gguf"
    w = gguf.GGUFWriter(str(src), arch="qwen35")
    w.add_uint32("qwen35.block_count", 1)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["a", "b", "c"])
    w.add_token_merges(["a b"])
    w.add_tensor("token_embd.weight", np.arange(24, dtype=np.float32).reshape(3, 8))
    w.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    convert_no_vocab.main(str(src), str(dst))
    r = gguf.GGUFReader(str(dst))
    assert r.fields["tokenizer.ggml.model"].contents() == "none"
    assert r.fields["qwen35.vocab_size"].contents() == 3
    assert "tokenizer.ggml.tokens" not in r.fields and "tokenizer.ggml.merges" not in r.fields
    assert r.fields["qwen35.block_count"].contents() == 1
    assert tensor_identity.main(str(src), str(dst)) == 0
    rec = json.loads(os.popen(f"{sys.executable} {E5}/tensor_identity.py {src} {dst}").read())
    assert rec["tensors"]["identical"] and rec["tensors"]["order_same"]
    assert "tokenizer.ggml.tokens" in rec["removed_keys"]
