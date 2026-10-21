"""examples/lambda-kleidiai/e12: the dormant DEC_CLEAR_DATA switch, the zip builder, the deploy's one-variable arms,
the host-metadata wrapper, the bench's interleaving and the offline comparator.

No AWS, no llama.cpp and no arm64: the shim test reads decshim.c's source, build_zip.py runs on a synthetic P1-shaped
zip, and the comparator runs on canned rows. The tests pin E12's one-variable contract: the deployed zip is P1's zip
with only lib/libdecshim.so swapped and the wrapper added, the arms differ only in DEC_CLEAR_DATA, and the baseline
behaviour (zero the buffers) is what the switch does when unset.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import types
import zipfile

import pytest

E12 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e12")
)
BASE = os.path.abspath(os.path.join(E12, "..", "baseline"))
SHIM_C = os.path.abspath(os.path.join(E12, "..", "shim", "decshim.c"))
sys.path.insert(0, E12)
sys.path.insert(0, BASE)

import build_zip  # noqa: E402
import offline_compare  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)


# --- the shim switch -------------------------------------------------------------------------------------------


def _shim() -> str:
    return open(SHIM_C).read()


def test_every_per_request_clear_goes_through_the_switch():
    src = _shim()
    code = re.sub(r"//[^\n]*", "", src)  # comments name the call too
    calls = re.findall(r"llama_memory_clear\(([^;]*)\);", code)
    assert len(calls) == 2, calls  # dec_eval and dec_eval_fork
    assert all(c.rstrip().endswith("d->clear_data") for c in calls), calls
    assert "llama_memory_clear(llama_get_memory(d->ctx), true)" not in code


def test_switch_defaults_to_the_baseline_full_clear():
    """Unset, empty or anything but exactly "0" keeps P1's clear(data=true)."""
    src = _shim()
    m = re.search(r'getenv\("DEC_CLEAR_DATA"\);\s*d->clear_data = (.*?);', src, re.S)
    assert m, "clear_data is not set from DEC_CLEAR_DATA"
    expr = m.group(1)
    assert expr.replace(" ", "") == '!(cd&&strcmp(cd,"0")==0)', expr

    def evaluate(cd):  # the C expression, transcribed
        return not (cd is not None and cd == "0")

    assert (
        evaluate(None) and evaluate("") and evaluate("1") and evaluate("false") and evaluate("00")
    )
    assert not evaluate("0")


def test_shim_reports_its_mode():
    assert re.search(r"int dec_clear_data\(void \* h\)", _shim())


# --- build_zip.py: one zip, P1's bytes plus two changes ----------------------------------------------------------


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    path = tmp_path / "p1.zip"
    entries = {
        "app.py": b"X = 1\n",
        "engine.py": b"SHIM = None\n",
        "lib/libdecshim.so": b"\x7fELF-p1-shim",
        "lib/libllama.so.0.7.0": b"\x7fELF-llama",
        "assets/tokenizer.json": b"{}",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.external_attr = (0o755 if name.endswith(".so") or ".so." in name else 0o644) << 16
            z.writestr(info, data)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "function_zip": {"sha256": sha, "bytes": path.stat().st_size},
                "model": {"gguf_sha256": "m"},
                "llama_cpp": {"commit": "f74e2fa6a"},
                "images": {},
            }
        )
    )
    return str(path), str(man)


def test_zip_is_p1_with_only_the_shim_swapped_and_the_wrapper_added(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    shim = tmp_path / "libdecshim.so"
    shim.write_bytes(b"\x7fELF-e12-shim")
    build_zip.main(
        ["--p1-zip", p1, "--p1-manifest", man, "--shim", str(shim), "--out", str(tmp_path / "b")]
    )
    out = tmp_path / "b" / "e12.zip"
    with zipfile.ZipFile(p1) as a, zipfile.ZipFile(out) as b:
        ia = {i.filename: i for i in a.infolist()}
        ib = {i.filename: i for i in b.infolist()}
        assert set(ib) - set(ia) == {"e12_entry.py"}
        assert set(ia) <= set(ib)
        for n in ia:
            if n == "lib/libdecshim.so":
                assert b.read(n) == b"\x7fELF-e12-shim"
                assert ib[n].external_attr == ia[n].external_attr  # keeps P1's mode
                continue
            assert a.read(n) == b.read(n), n
            assert (ia[n].external_attr, ia[n].date_time) == (
                ib[n].external_attr,
                ib[n].date_time,
            ), n
        assert b.read("e12_entry.py") == open(os.path.join(E12, "e12_entry.py"), "rb").read()
    m = json.load(open(tmp_path / "b" / "manifest.json"))
    assert m["unchanged_entries"] == 4
    assert m["function_zip"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert m["zips"]["e12"]["sha256"] == m["function_zip"]["sha256"]
    assert (
        m["changes"]["replaced"]["lib/libdecshim.so"]["e12_sha256"]
        == hashlib.sha256(b"\x7fELF-e12-shim").hexdigest()
    )
    assert isinstance(m["dirty_inputs"], list)


def test_zip_refuses_a_zip_its_manifest_does_not_describe(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    m = json.load(open(man))
    m["function_zip"]["sha256"] = "0" * 64
    json.dump(m, open(man, "w"))
    shim = tmp_path / "s.so"
    shim.write_bytes(b"x")
    with pytest.raises(SystemExit):
        build_zip.main(
            [
                "--p1-zip",
                p1,
                "--p1-manifest",
                man,
                "--shim",
                str(shim),
                "--out",
                str(tmp_path / "b"),
            ]
        )


def test_unchanged_entries_catches_any_other_difference(tmp_path):
    p1, _ = _p1_like_zip(tmp_path)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(p1) as a, zipfile.ZipFile(bad, "w") as b:
        for i in a.infolist():
            data = b"changed" if i.filename == "engine.py" else a.read(i)
            b.writestr(i, data)
        b.writestr(zipfile.ZipInfo("e12_entry.py", date_time=STAMP), b"")
    with pytest.raises(SystemExit, match=r"engine\.py"):
        build_zip.unchanged_entries(p1, str(bad))


# --- e12deploy.py: the arms differ in one variable ---------------------------------------------------------------


def _import_with_stub_boto3(name: str):
    """Import an E12 module that imports boto3/botocore at module level. The SDK stand-ins (when the SDK is absent)
    are installed per test by the autouse fixture below, through monkeypatch, so nothing leaks to other files."""
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    """P1's test pattern (tests/test_lkai_baseline.py): empty stand-ins when the SDK is absent."""
    for m in ("deploy", "bench", "e12deploy", "e12bench"):
        monkeypatch.delitem(sys.modules, m, raising=False)
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


def test_arm_environments_differ_only_in_the_switch():
    d = _import_with_stub_boto3("e12deploy")
    p1_env = {
        "MODEL_BUCKET": "b",
        "MODEL_KEY": "k",
        "DEC_KV": "f32",
        "DEC_UBATCH": "512",
        "N_SEQ": "17",
    }
    dor, e12 = d.arm_environment(p1_env, "dor"), d.arm_environment(p1_env, "e12")
    assert "DEC_CLEAR_DATA" not in dor
    assert e12["DEC_CLEAR_DATA"] == "0"
    strip = {"DEC_CLEAR_DATA", "E12_ARM"}
    assert {k: v for k, v in dor.items() if k not in strip} == {
        k: v for k, v in e12.items() if k not in strip
    }
    assert dor["LKAI_PHASES"] == e12["LKAI_PHASES"] == "1"
    for k, v in p1_env.items():
        assert dor[k] == v and e12[k] == v


def test_arm_environment_refuses_a_p1_environment_that_already_sets_the_switch():
    d = _import_with_stub_boto3("e12deploy")
    with pytest.raises(SystemExit):
        d.arm_environment({"DEC_CLEAR_DATA": "1"}, "dor")


# --- e12bench.py: interleaving and the request plan -------------------------------------------------------------


def test_bench_alternates_order_and_covers_every_request():
    bn = _import_with_stub_boto3("e12bench")
    assert bn.order_for(["dor", "e12"], 0) == ["dor", "e12"]
    assert bn.order_for(["dor", "e12"], 1) == ["e12", "dor"]
    p1_ids = ["a", "b", "c", "d", "e"]
    extra = ["a", "b", "c", "d", "e", "w", "s"]
    plan = bn.warm_ids(p1_ids, extra, 10)
    assert plan[:5] == p1_ids and set(plan) == set(extra)


def test_bench_runs_both_arms_at_the_same_moment():
    import threading

    bn = _import_with_stub_boto3("e12bench")
    barrier = threading.Barrier(2, timeout=5)

    def fn(arm):  # deadlocks (BrokenBarrierError) unless both arms run concurrently
        barrier.wait()
        return [arm]

    assert bn.together(None, ["dor", "e12"], fn) == {"dor": ["dor"], "e12": ["e12"]}


def test_merged_requests_rejects_colliding_ids():
    bn = _import_with_stub_boto3("e12bench")
    p1 = {"probe": "a", "requests": [{"id": "a", "request": {}}]}
    assert [
        r["id"]
        for r in bn.merged_requests(p1, {"requests": [{"id": "z", "request": {}}]})["requests"]
    ] == [
        "a",
        "z",
    ]
    with pytest.raises(SystemExit):
        bn.merged_requests(p1, {"requests": [{"id": "a", "request": {}}]})


def test_stress_requests_include_the_widest_fork():
    reqs = json.load(open(os.path.join(E12, "requests.json")))["requests"]
    p1 = {r["id"] for r in json.load(open(os.path.join(BASE, "requests.json")))["requests"]}
    widest = max(len(r["request"]["questions"]) for r in reqs)
    assert widest == 16  # N_SEQ=17: the state plus 16 question sequences
    assert not p1 & {r["id"] for r in reqs}
    assert any(len(r["request"]["questions"]) == 1 for r in reqs)  # the dec_eval path too


# --- e12_entry.py parsers --------------------------------------------------------------------------------------


def test_entry_parsers_report_unknown_rather_than_guess(monkeypatch):
    sys.modules.setdefault("lkai_entry", types.ModuleType("lkai_entry"))
    import e12_entry

    c = e12_entry.cpuinfo(
        "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\nFeatures\t: fp asimd i8mm\n\nprocessor\t: 1\n"
    )
    assert c["part"] == "0xd40" and c["implementer"] == "0x41" and c["processors"] == 2
    assert c["variant"] == "unknown" and c["revision"] == "unknown"
    assert e12_entry.cpuinfo(None)["part"] == "unknown"
    raw = (16).to_bytes(8, "little") + (0xFF).to_bytes(8, "little")
    assert e12_entry.auxv(raw) == {"hwcap": "0xff", "hwcap2": "unknown"}
    assert e12_entry.auxv(b"\x01") == {"hwcap": "unknown", "hwcap2": "unknown"}
    src = open(os.path.join(E12, "e12_entry.py")).read()
    assert "Graviton" not in src  # no generation is named from a part number


# --- offline_compare.py -----------------------------------------------------------------------------------------


def _rows(arm_clear, answers, hidden, ids=("a", "b", "a")):
    init = {"kind": "init", "clear_data": arm_clear, "load_s": 1.0}
    return init, [
        {
            "kind": "row",
            "id": i,
            "answers": answers.get(i),
            "hidden_sha256": hidden.get(i),
            "cpu": "0xd40",
            "latency_ms": 1.0,
        }
        for i in ids
    ]


def test_offline_compare_passes_identical_arms():
    ans, hid = {"a": {"q": 1}, "b": {"q": 2}}, {"a": ["h1"], "b": ["h2"]}
    runs = {"p1-0": _rows(None, ans, hid), "dor-1": _rows(1, ans, hid), "e12-2": _rows(0, ans, hid)}
    ref = {"a": {"0xd40": {"q": 1}}}
    res = offline_compare.compare(runs, ref)
    assert res["pass"], res["failures"]
    assert res["runs"]["e12-2"]["reference"]["bit_identical"] == 2


def test_offline_compare_catches_stale_state_and_a_wrong_switch():
    ans, hid = {"a": {"q": 1}, "b": {"q": 2}}, {"a": ["h1"], "b": ["h2"]}
    init, leaky = _rows(0, ans, hid)
    leaky[2] = dict(leaky[2], hidden_sha256=["other"])  # request a after b differs: a leak
    runs = {"p1-0": _rows(None, ans, hid), "e12-2": (init, leaky), "dor-1": _rows(0, ans, hid)}
    res = offline_compare.compare(runs, None)
    assert not res["pass"]
    joined = " ".join(res["failures"])
    assert "e12-2: 1 rows differ" in joined and "vary across positions for ['a']" in joined
    assert "dor-1: clear_data=0, expected 1" in joined
