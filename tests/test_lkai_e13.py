"""examples/lambda-kleidiai/e13: the OpenMP wait-policy arms, the zip builder, the wrapper's parsers, the bench order.

No AWS and no llama.cpp. The tests pin E13's one-variable contract: every arm deploys the same zip (P1's, byte for
byte, plus the wrapper) and P1's environment, and the arms differ only in OMP_WAIT_POLICY / GOMP_SPINCOUNT.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import struct
import sys
import types
import zipfile
from collections import Counter

import pytest

E13 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e13")
)
BASE = os.path.abspath(os.path.join(E13, "..", "baseline"))
sys.path.insert(0, E13)
sys.path.insert(0, BASE)

import e13_zip  # noqa: E402


def _sdk_stand_ins() -> dict[str, types.ModuleType]:
    """e13deploy/e13bench import boto3/botocore at module level; the test env need not have them. No test makes
    an AWS call, so empty stand-ins suffice when the SDK is absent."""
    if importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore"):
        return {}
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    return {
        "boto3": boto3,
        "botocore": types.ModuleType("botocore"),
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }


_saved = {k: sys.modules.get(k) for k in _sdk_stand_ins()}
sys.modules.update(_sdk_stand_ins())
try:
    import e13bench
    import e13deploy
finally:
    for _k, _v in _saved.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v
    for _m in (
        "deploy",
        "bench",
    ):  # leave the baseline tests their own import under their own stand-ins
        sys.modules.pop(_m, None)

STAMP = (2026, 1, 1, 0, 0, 0)
P1_ENV = {
    "N_SEQ": "17",
    "MODEL_KEY": "weights/x/s2b.gguf",
    "MODEL_BUCKET": "b",
    "DEC_KV": "f32",
    "DEC_UBATCH": "512",
}


# --- the one variable --------------------------------------------------------------------------------------------


def test_arms_differ_only_in_wait_policy_variables():
    envs = {arm: e13deploy.arm_environment(P1_ENV, arm) for arm in e13deploy.ARMS}
    keys = set().union(*envs.values())
    differing = {k for k in keys if len({e.get(k) for e in envs.values()}) > 1}
    assert differing == {"E13_ARM", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"}
    for env in envs.values():
        assert {k: env[k] for k in P1_ENV} == P1_ENV  # P1's environment is kept as is
        assert env["LKAI_PHASES"] == "1"
        assert env["OMP_DISPLAY_ENV"] == "VERBOSE"  # same on every arm: the log proof
        assert "THREADS" not in env and "OMP_NUM_THREADS" not in env  # thread count stays P1's


def test_control_is_libgomp_default():
    env = e13deploy.arm_environment(P1_ENV, "ctl")
    assert "OMP_WAIT_POLICY" not in env and "GOMP_SPINCOUNT" not in env
    assert e13deploy.EXPECTED_SPINCOUNT["ctl"] == 300000  # gcc 11 libgomp/env.c, wait policy unset


@pytest.mark.parametrize("key", ["OMP_WAIT_POLICY", "THREADS", "OMP_NUM_THREADS", "GOMP_SPINCOUNT"])
def test_p1_environment_that_sets_an_omp_variable_is_refused(key):
    with pytest.raises(SystemExit):
        e13deploy.arm_environment(P1_ENV | {key: "1"}, "ctl")


# --- the zip -----------------------------------------------------------------------------------------------------


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    path = tmp_path / "p1.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in {
            "lkai_entry.py": b"X = 1\n",
            "lib/libgomp.so.1": b"\x7fELF",
            "assets/tokenizer.json": b"{}",
        }.items():
            z.writestr(zipfile.ZipInfo(name, date_time=STAMP), data)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "function_zip": {"sha256": sha, "bytes": path.stat().st_size},
                "model": {"id": "m"},
                "llama_cpp": {"commit": "c"},
                "images": {},
                "toolchain": "t",
                "assets_sha256": {},
            }
        )
    )
    return str(path), str(man)


def test_zip_is_p1_byte_for_byte_plus_wrapper(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    m = e13_zip.build(p1, man, str(tmp_path / "out"))
    with zipfile.ZipFile(p1) as a, zipfile.ZipFile(tmp_path / "out" / "e13.zip") as b:
        assert [i.filename for i in b.infolist()] == [i.filename for i in a.infolist()] + [
            "e13_entry.py"
        ]
        for ia in a.infolist():
            ib = b.getinfo(ia.filename)
            assert a.read(ia) == b.read(ib)
            assert (ia.date_time, ia.external_attr) == (ib.date_time, ib.external_attr)
        assert b.read("e13_entry.py") == open(os.path.join(E13, "e13_entry.py"), "rb").read()
    assert m["zip"]["handler"] == "e13_entry.handler"
    assert m["function_zip"]["sha256"] == m["zip"]["sha256"]


def test_zip_refuses_a_zip_its_manifest_does_not_describe(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    with open(p1, "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit):
        e13_zip.build(p1, man, str(tmp_path / "out"))


# --- the bench's order -------------------------------------------------------------------------------------------

ARMS = list(e13deploy.ARMS)


def test_round_order_puts_every_arm_in_every_position():
    for start in (0, len(ARMS)):  # each cycle of len(ARMS) rounds, forward and reversed
        seen = Counter()
        for rnd in range(start, start + len(ARMS)):
            order = e13bench.round_order(ARMS, rnd)
            assert sorted(order) == sorted(ARMS)
            seen.update((pos, arm) for pos, arm in enumerate(order))
        assert all(seen[(p, a)] == 1 for p in range(len(ARMS)) for a in ARMS)
    assert e13bench.round_order(ARMS, len(ARMS)) == ARMS[::-1]


def test_warm_passes_interleave_and_lead_equally_within_a_round():
    plan = e13bench.warm_plan(["a", "b", "c", "d", "e"], 4)
    assert len(plan) == 20 and Counter(plan) == dict.fromkeys("abcde", 4)
    for rnd in range(6):
        firsts = Counter(e13bench.pass_order(ARMS, rnd, i)[0] for i in range(len(plan)))
        assert set(firsts.values()) == {len(plan) // len(ARMS)}
        for i in range(len(plan)):
            assert sorted(e13bench.pass_order(ARMS, rnd, i)) == sorted(ARMS)


# --- the wrapper -------------------------------------------------------------------------------------------------


@pytest.fixture
def entry(monkeypatch):
    """e13_entry imported against a stand-in lkai_entry (P1's needs the engine)."""
    calls = []

    def handler(event, context):
        calls.append(event)
        return {"answers": {}, "lkai": {"inv": {"cpu_part": "0xd40"}}}

    monkeypatch.setitem(sys.modules, "lkai_entry", types.SimpleNamespace(handler=handler))
    monkeypatch.setenv("E13_ARM", "passive")
    monkeypatch.setenv("OMP_WAIT_POLICY", "passive")
    sys.modules.pop("e13_entry", None)
    mod = importlib.import_module("e13_entry")
    yield mod
    sys.modules.pop("e13_entry", None)


def test_cpuinfo_reads_first_block_and_marks_missing_unknown(entry):
    text = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\nFeatures\t: fp asimd i8mm\n\nprocessor\t: 1\nCPU part\t: 0xd0c\n"
    c = entry.cpuinfo(text)
    assert c["part"] == "0xd40" and c["implementer"] == "0x41" and c["processors"] == 2
    assert c["variant"] == "unknown" and c["revision"] == "unknown"
    assert entry.cpuinfo(None)["part"] == "unknown"


def test_auxv_hwcaps(entry):
    raw = struct.pack("<QQQQQQ", 16, 0xFF, 26, 0x1, 0, 0)
    assert entry.auxv(raw) == {"hwcap": "0xff", "hwcap2": "0x1"}
    assert entry.auxv(b"\0" * 7) == {"hwcap": "unknown", "hwcap2": "unknown"}


def test_rusage_delta(entry):
    a = {"utime": 1.0, "stime": 0.5, "nvcsw": 10, "nivcsw": 2}
    b = {"utime": 3.4, "stime": 0.6, "nvcsw": 40, "nivcsw": 5}
    d = entry.rusage_delta(a, b, 0.5)
    assert d == {
        "wall_s": 0.5,
        "utime_s": 2.4,
        "stime_s": 0.1,
        "cpu_per_wall": 5.0,
        "nvcsw": 30,
        "nivcsw": 3,
    }
    assert entry.rusage_delta(a, b, 0)["cpu_per_wall"] is None


def test_handler_adds_record_on_first_invoke_only(entry, capsys):
    ctx = types.SimpleNamespace(log_stream_name="s/1")
    first = entry.handler({}, ctx)
    second = entry.handler({}, ctx)
    assert first["lkai"]["e13"]["omp_env"]["OMP_WAIT_POLICY"] == "passive"
    assert first["lkai"]["e13"]["e13_arm"] == "passive"
    assert first["lkai"]["e13"]["log_stream"] == "s/1"
    assert "e13" not in second["lkai"]
    assert set(second["lkai"]["e13_inv"]) == {
        "wall_s",
        "utime_s",
        "stime_s",
        "cpu_per_wall",
        "nvcsw",
        "nivcsw",
    }
    out = capsys.readouterr().out.splitlines()
    assert len([ln for ln in out if ln.startswith("E13 ")]) == 1


def test_wrapper_names_no_cpu_generation(entry):
    src = open(os.path.join(E13, "e13_entry.py")).read()
    for word in ("Graviton2", "Graviton3", "Graviton4", "Neoverse"):
        assert word not in src
