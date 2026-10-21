"""examples/lambda-kleidiai/e7: the dormant skip-tied-output patch, the entry wrapper and the summary.

No AWS and no arm64 build: the patch runs on a minimal stand-in for the llama.cpp files it edits (the exact anchors
P1's pinned tree holds after llama/patch_llama.py), e7_host's parsers run on canned /proc text, and summarize_e7's
statistics run on synthetic rows. The tests pin E7's one-variable contract: unset is upstream's code path, only
LKAI_SKIP_TIED_OUTPUT=1 skips the head, and both arms are compared within one CPU part.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import struct
import sys
import types

import pytest

E7 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e7")
)
BASE = os.path.abspath(os.path.join(E7, "..", "baseline"))
sys.path.insert(0, E7)
sys.path.insert(0, BASE)

import e7_host  # noqa: E402
import patch_skip_output as patch  # noqa: E402
import summarize_e7 as s7  # noqa: E402


def _sdk_stand_ins(monkeypatch) -> None:
    """deploy_e7.py and bench_e7.py import boto3/botocore at module level; the test env need not have them. No test
    here calls AWS, so empty stand-ins suffice when the SDK is absent (as tests/test_lkai_baseline.py does)."""
    for m in ("deploy_e7", "bench_e7", "deploy", "bench"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    if (
        "boto3" not in sys.modules
        and importlib.util.find_spec("boto3")
        and importlib.util.find_spec("botocore")
    ):
        return
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    for name, mod in {
        "boto3": boto3,
        "botocore": types.ModuleType("botocore"),
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def deploy_e7(monkeypatch):
    _sdk_stand_ins(monkeypatch)
    return importlib.import_module("deploy_e7")


@pytest.fixture
def bench_e7(monkeypatch):
    _sdk_stand_ins(monkeypatch)
    return importlib.import_module("bench_e7")


# --- patch -----------------------------------------------------------------------------------------------------------


def _tree(tmp_path):
    """The four files patch_skip_output.py edits, holding exactly the anchors (and some context) it expects."""
    (tmp_path / "src/models").mkdir(parents=True)
    (tmp_path / "include").mkdir()
    (tmp_path / "src/models/qwen35.cpp").write_text(
        patch.INCLUDES_OLD
        + "\nvoid load() {\n"
        + patch.LOAD_OLD
        + "}\n\nvoid graph() {\n"
        + "    if (cparams.embeddings) {  // strands-decider prefill patch\n        return;\n    }\n\n"
        + patch.GRAPH_OLD
        + "\nvoid mtp() {\n"
        + patch.MTP_OLD
        + "\n}\n"
    )
    (tmp_path / "src/llama-model.h").write_text(
        "struct llama_model {\n" + patch.MODEL_H_OLD + "};\n"
    )
    (tmp_path / "include/llama.h").write_text(patch.API_H_OLD)
    (tmp_path / "src/llama-model.cpp").write_text(patch.API_C_OLD)
    return tmp_path


def test_patch_gates_the_duplicate_on_the_switch_and_keeps_the_upstream_call(tmp_path):
    root = _tree(tmp_path)
    patch.patch(root)
    q = (root / "src/models/qwen35.cpp").read_text()
    # the upstream create_tensor survives unchanged, inside the not-skipped branch
    call = 'output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);'
    assert q.count(call) == 1
    assert q.index('getenv("LKAI_SKIP_TIED_OUTPUT")') < q.index(call)
    # only the exact value "1" skips
    assert 'strcmp(skip, "1") == 0' in q
    assert "lkai_tied_output_skipped = true;" in q
    # the LM head refuses a skipped head instead of dereferencing NULL; the embeddings early return stays first
    assert q.index("if (cparams.embeddings)") < q.index("GGML_ASSERT(model.output")
    assert "#include <cstdlib>" in q and "#include <cstring>" in q


def test_patch_exports_the_two_queries(tmp_path):
    root = _tree(tmp_path)
    patch.patch(root)
    h = (root / "include/llama.h").read_text()
    c = (root / "src/llama-model.cpp").read_text()
    for fn in ("llama_model_lkai_tied_output_skipped", "llama_model_lkai_buft_bytes"):
        assert "LLAMA_API" in h and fn in h and fn in c
    assert "memory_breakdown()" in c
    assert "bool lkai_tied_output_skipped = false;" in (root / "src/llama-model.h").read_text()


def test_patch_refuses_a_tree_without_its_anchors_and_cannot_apply_twice(tmp_path):
    root = _tree(tmp_path)
    patch.patch(root)
    with pytest.raises(AssertionError, match="anchor"):
        patch.patch(root)  # the anchors are gone: a second application is refused, not stacked
    bad = tmp_path / "other"
    bad.mkdir()
    _tree(bad)
    (bad / "src/models/qwen35.cpp").write_text("// upstream moved this code\n")
    with pytest.raises(AssertionError, match=r"qwen35\.cpp"):
        patch.patch(bad)


def test_patch_anchors_match_the_pinned_tree_when_available():
    """When a pinned, P1-patched llama.cpp checkout is at hand (LKAI_LLAMA_TREE), every anchor is in it once."""
    tree = os.environ.get("LKAI_LLAMA_TREE")
    if not tree:
        pytest.skip("LKAI_LLAMA_TREE not set")
    files = {
        "src/models/qwen35.cpp": (
            patch.INCLUDES_OLD,
            patch.LOAD_OLD,
            patch.GRAPH_OLD,
            patch.MTP_OLD,
        ),
        "src/llama-model.h": (patch.MODEL_H_OLD,),
        "include/llama.h": (patch.API_H_OLD,),
        "src/llama-model.cpp": (patch.API_C_OLD,),
    }
    for f, anchors in files.items():
        s = open(os.path.join(tree, f)).read()
        for a in anchors:
            assert s.count(a) == 1, (f, a[:40])


# --- deploy: one variable ------------------------------------------------------------------------------------------


def test_arms_differ_only_in_the_switch(deploy_e7):
    p1_env = {
        "MODEL_BUCKET": "b",
        "MODEL_KEY": "k",
        "DEC_KV": "f32",
        "N_SEQ": "17",
        "LKAI_PHASES": "0",
    }
    ctl = deploy_e7.arm_environment(p1_env, "ctl")
    skip = deploy_e7.arm_environment(p1_env, "skip")
    assert "LKAI_SKIP_TIED_OUTPUT" not in ctl
    assert skip.pop("LKAI_SKIP_TIED_OUTPUT") == "1"
    assert ctl == skip
    assert ctl["LKAI_PHASES"] == "1" and ctl["COLD_TAG"] == "0"


def test_bench_refuses_an_arm_whose_switch_is_wrong(bench_e7):
    assert bench_e7.arm_problem("ctl", {"LKAI_PHASES": "1"}) is None
    assert bench_e7.arm_problem("skip", {"LKAI_PHASES": "1", "LKAI_SKIP_TIED_OUTPUT": "1"}) is None
    assert "needs None" in bench_e7.arm_problem(
        "ctl", {"LKAI_PHASES": "1", "LKAI_SKIP_TIED_OUTPUT": "1"}
    )
    assert "needs '1'" in bench_e7.arm_problem("skip", {"LKAI_PHASES": "1"})
    assert "LKAI_PHASES" in bench_e7.arm_problem(
        "skip", {"LKAI_PHASES": "0", "LKAI_SKIP_TIED_OUTPUT": "1"}
    )
    assert bench_e7.read_timeout_from(["--rounds", "5", "--read-timeout", "300", "--out", "x"]) == (
        300,
        ["--rounds", "5", "--out", "x"],
    )
    assert bench_e7.read_timeout_from(["--out", "x"]) == (900, ["--out", "x"])


# --- host metadata ---------------------------------------------------------------------------------------------------

CPUINFO = """processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
CPU revision\t: 1

processor\t: 1
CPU part\t: 0xd40
"""


def test_cpuinfo_reads_the_first_block_and_labels_missing_fields_unknown():
    c = e7_host.cpuinfo(CPUINFO)
    assert c["part"] == "0xd40" and c["variant"] == "0x1" and c["implementer"] == "0x41"
    assert c["processors"] == 2
    assert e7_host.cpuinfo(None) == dict.fromkeys(
        ("implementer", "architecture", "variant", "part", "revision", "features", "processors"),
        "unknown",
    )
    assert e7_host.cpuinfo("processor : 0\n")["part"] == "unknown"


def test_auxv_extracts_hwcaps_and_rejects_a_truncated_vector():
    raw = struct.pack("<QQQQQQ", 16, 0xFFFFFFF, 26, 0x1F201, 0, 0)
    assert e7_host.auxv(raw) == {"hwcap": "0xfffffff", "hwcap2": "0x1f201"}
    assert e7_host.auxv(raw[:-3]) == {"hwcap": "unknown", "hwcap2": "unknown"}


def test_proc_status_kb_and_mechanism_without_a_model():
    s = e7_host.proc_status_kb("Name:\tpython\nVmHWM:\t 5300000 kB\nVmRSS:\t 5200000 kB\n")
    assert s == {"VmRSS": 5200000, "VmHWM": 5300000}
    assert e7_host.proc_status_kb(None) == {"VmRSS": "unknown", "VmHWM": "unknown"}
    m = e7_host.mechanism(None)
    assert m["tied_output_skipped"] == "unknown" and set(m["buft_bytes"].values()) == {"unknown"}


def test_handler_attaches_host_and_mechanism_to_the_first_invocation_only(monkeypatch, capsys):
    calls = []
    fake = types.SimpleNamespace(
        handler=lambda e, c: calls.append(e) or {"answers": {}, "lkai": {"inv": {}}}
    )
    monkeypatch.setitem(sys.modules, "lkai_entry", fake)
    monkeypatch.setitem(sys.modules, "engine", types.SimpleNamespace(SHIM=None))
    sys.modules.pop("e7_entry", None)
    import e7_entry

    ctx = types.SimpleNamespace(log_stream_name="2026/10/21/[$LATEST]abc")
    first = e7_entry.handler({"q": 1}, ctx)
    second = e7_entry.handler({"q": 2}, ctx)
    assert first["lkai"]["e7"]["log_stream"] == ctx.log_stream_name
    assert first["lkai"]["e7"]["mechanism"]["tied_output_skipped"] == "unknown"
    assert "init_cpu_part" in first["lkai"]["e7"] and "cpu" in first["lkai"]["e7"]
    assert "e7" not in second["lkai"]
    assert capsys.readouterr().out.count("E7 {") == 1
    assert calls == [{"q": 1}, {"q": 2}]


# --- summary ---------------------------------------------------------------------------------------------------------


def _row(
    arm, cls, cpu, rnd, *, mmap=8.0, total=36.0, eng=0.24, mem=5160, answers=None, rid="probe"
):
    rep = {"Duration": 1000.0 * (total - 0.8), "Max Memory Used": mem}
    if cls == "cold":
        rep["Init Duration"] = 800.0
    elif cls == "restore":
        rep["Restore Duration"] = 1000.0 * (total - 0.7)
        rep["Duration"] = 700.0
    return {
        "arm": arm,
        "round": rnd,
        "expected_class": cls,
        "observed_class": cls,
        "class_mismatch": False,
        "error": None,
        "cpu_part": cpu,
        "request_id": rid,
        "report": rep,
        "latency_ms": eng * 1000,
        "answers": answers if answers is not None else {"q": {"choice": "a"}},
        "lkai": {"init": {"model_load_mmap": mmap}},
    }


def test_delta_is_within_cpu_part_and_never_strata_on_a_missing_part():
    rows = []
    for rnd in range(3):
        for i in range(3):
            rows += [
                _row("ctl", "cold", "0xd40", rnd, mmap=8.0 + 0.01 * i),
                _row("skip", "cold", "0xd40", rnd, mmap=6.3 + 0.01 * i),
                _row("ctl", "cold", "0xd0c", rnd, mmap=12.0 + 0.01 * i),
                _row("skip", "cold", "0xd0c", rnd, mmap=9.5 + 0.01 * i),
            ]
    rows += [_row("ctl", "cold", None, 0, mmap=1.0), _row("skip", "cold", "?", 0, mmap=99.0)]
    d = s7.delta(rows, "cold", {"model_load_mmap": s7.COLD_KEYS["model_load_mmap"]})[
        "model_load_mmap"
    ]
    assert set(d["by_cpu_part"]) == {"0xd0c", "0xd40"}
    assert d["by_cpu_part"]["0xd40"]["delta_p50"] == pytest.approx(-1.7)
    assert d["by_cpu_part"]["0xd0c"]["delta_p50"] == pytest.approx(-2.5)
    assert d["by_cpu_part"]["0xd40"]["ci95"][1] < 0
    assert len(d["cells"]) == 6 and d["cell_delta_signs"] == [-1]
    assert d["rows_without_cpu_part"] == {"ctl": 1, "skip": 1}


def test_cells_record_a_part_that_only_one_arm_drew_in_a_round():
    rows = [
        _row("ctl", "cold", "0xd40", 0),
        _row("skip", "cold", "0xd40", 0),
        _row("ctl", "cold", "0xd4f", 0),
    ]
    rows += [_row("ctl", "cold", "0xd4f", 1), _row("skip", "cold", "0xd4f", 1)]
    d = s7.delta(rows, "cold", {"model_load_mmap": s7.COLD_KEYS["model_load_mmap"]})[
        "model_load_mmap"
    ]
    assert {"round": 0, "cpu_part": "0xd4f", "n": [1, 0]} in d["cells_missing_an_arm"]


def test_arm_agreement_flags_a_differing_answer_and_a_missing_ctl_part():
    rows = [
        _row("ctl", "warm", "0xd40", 0, rid="a", answers={"q": 1}),
        _row("skip", "warm", "0xd40", 0, rid="a", answers={"q": 1}),
        _row("skip", "warm", "0xd40", 0, rid="a", answers={"q": 2}),
        _row("skip", "warm", "0xd4f", 0, rid="a", answers={"q": 1}),
    ]
    g = s7.arm_agreement(rows)
    assert g["compared"] == 2 and g["bit_identical"] == 1 and g["no_ctl_answer_for_cpu_part"] == 1


def _res(cold_ci, restore_ci=(-0.1, 0.1), warm_ci=(-0.01, 0.01), clean=True):
    def head(ci):
        return {"by_cpu_part": {"0xd40": {"ci95": list(ci)}}}

    return {
        "errors": {"count": 0 if clean else 1},
        "correctness": {"compared": 10, "bit_identical": 10, "choice_flips": 0},
        "e7": {
            "delta": {
                "cold": {"first_request_total": head(cold_ci)},
                "restore": {"first_request_total": head(restore_ci)},
                "warm": {"steady_state_inference": head(warm_ci)},
            },
            "arm_agreement": {
                "compared": 10,
                "bit_identical": 10,
                "no_ctl_answer_for_cpu_part": 0,
                "ctl_internal_disagreements": 0,
            },
        },
    }


def test_status_follows_the_headlines():
    assert s7.status_of(_res((-1.9, -1.5))) == "improved"
    assert s7.status_of(_res((-0.2, 0.1))) == "no-change"
    assert s7.status_of(_res((-1.9, -1.5), restore_ci=(0.05, 0.2))) == "regressed"
    assert s7.status_of(_res((-1.9, -1.5), clean=False)) == "inconclusive"


def test_no_graviton_generation_is_named_in_e7_sources():
    """Host identity is reported as read (CPU part, features); a generation name would be an inference."""
    for f in sorted(os.listdir(E7)):
        if f.endswith((".py", ".sh", ".json", ".md")):
            text = open(os.path.join(E7, f)).read()
            assert (
                "Graviton2" not in text and "Graviton3" not in text and "Graviton4" not in text
            ), f


def _hv_res(rows):
    res = _res((-1.9, -1.5))
    res["e7"]["delta"] = {
        "cold": s7.delta(rows, "cold", {"model_load_mmap": s7.COLD_KEYS["model_load_mmap"]}),
        "restore": {},
        "warm": {},
    }
    res["e7"]["host"] = {"unknown_fields": ["cgroup cpu.max"]}
    return s7.host_variability(res, rows)


def test_host_variability_needs_no_rerun_when_every_part_and_cell_agree():
    rows = []
    for rnd in range(3):
        for i in range(3):
            rows += [
                _row("ctl", "cold", "0xd40", rnd, mmap=8.0 + 0.01 * i),
                _row("skip", "cold", "0xd40", rnd, mmap=6.3 + 0.01 * i),
                _row("ctl", "cold", "0xd0c", rnd, mmap=12.0 + 0.01 * i),
                _row("skip", "cold", "0xd0c", rnd, mmap=9.5 + 0.01 * i),
            ]
    hv = _hv_res(rows)
    c = hv["checks"]["cold.model_load_mmap"]
    assert c["consistent"] and c["ci_sign_by_part"] == {"0xd0c": -1, "0xd40": -1}
    assert hv["rerun_needed"] is False
    assert hv["observed"]["skip:cold:0xd40"] == 9


def test_host_variability_flags_an_effect_whose_sign_depends_on_the_host():
    rows = []
    for rnd in range(3):
        for i in range(3):
            rows += [
                _row("ctl", "cold", "0xd40", rnd, mmap=8.0 + 0.01 * i),
                _row("skip", "cold", "0xd40", rnd, mmap=6.3 + 0.01 * i),
                _row("ctl", "cold", "0xd0c", rnd, mmap=9.5 + 0.01 * i),
                _row("skip", "cold", "0xd0c", rnd, mmap=12.0 + 0.01 * i),
            ]
    hv = _hv_res(rows)
    c = hv["checks"]["cold.model_load_mmap"]
    assert not c["consistent"] and c["ci_sign_by_part"] == {"0xd0c": 1, "0xd40": -1}
    assert c["cell_delta_signs"] == [-1, 1]
    assert hv["rerun_needed"] is True
