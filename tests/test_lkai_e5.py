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
    # Graviton3: e5 loads 0.5 s faster. Graviton2: only e5 drew it, so it must not enter the comparison.
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


def test_manifest_refuses_converter_from_wrong_llama_checkout(tmp_path):
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
    (model / "convert.json").write_text(
        json.dumps(
            {
                "image": "python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b",
                "tools": {"llama_cpp": "wrong", "gguf_py": "0.19.0"},
                "tensors": {
                    "identical": True,
                    "count": [335, 335],
                    "digest": ["same", "same"],
                    "data_offset": [32, 32],
                },
            }
        )
    )

    with pytest.raises(
        SystemExit, match=r"conversion used llama\.cpp 'wrong', expected expected"
    ):
        manifest_e5.main(str(build), str(manifest_path))


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
