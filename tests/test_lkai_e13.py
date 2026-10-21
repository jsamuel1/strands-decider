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
import statistics
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


# --- the summary -------------------------------------------------------------------------------------------------

import summarize_e13 as s13  # noqa: E402

REQS = ["a", "b", "c", "d", "e"]
BASE_MS = {"a": 250, "b": 400, "c": 600, "d": 1000, "e": 1750}


def _warm(arm, rnd, rid, ms, cpu="0xd40", cpu_s=2.0):
    return {
        "arm": arm,
        "round": rnd,
        "request_id": rid,
        "expected_class": "warm",
        "observed_class": "warm",
        "class_mismatch": False,
        "error": None,
        "cpu_part": cpu,
        "report": {"Duration": ms},
        "latency_ms": ms - 5,
        "answers": {"q": {"choice": "x"}},
        "lkai": {"e13_inv": {"utime_s": cpu_s, "stime_s": 0.1, "nvcsw": 10, "nivcsw": 1}},
    }


def _rows(effect_pct: dict[str, float], rounds=6, n=10, cpu="0xd40", jitter=0.002, seed=1):
    import random as _r

    g = _r.Random(seed)
    rows = []
    for rnd in range(rounds):
        drift = 1 + 0.05 * (rnd % 3)  # round-to-round drift, shared by every arm in the round
        for rid in REQS:
            for arm, pct in effect_pct.items():
                for _ in range(n):
                    ms = BASE_MS[rid] * drift * (1 + pct / 100) * (1 + g.uniform(-jitter, jitter))
                    rows.append(_warm(arm, rnd, rid, ms, cpu))
    return rows


def test_paired_warm_detects_a_consistent_speedup_despite_round_drift():
    rows = _rows({"ctl": 0, "passive": -4})
    wp = s13.paired_by_request(rows, "passive", s13.WARM_KEYS["report_duration"], "0xd40")
    assert wp["cells"] == 30 and wp["rounds_below_zero"] == 6
    assert -4.5 < wp["mean_round_pct"] < -3.5
    assert wp["sign_flip_p"] == wp["min_attainable_p"] == round(2 / 64, 5)
    null = s13.control_null(rows, s13.WARM_KEYS["report_duration"], "0xd40")
    assert null["max_abs_round_pct"] < 1
    assert s13.warm_verdict(wp, null) == "faster"


def test_paired_warm_null_effect_is_no_change():
    rows = _rows({"ctl": 0, "spin30k": 0}, jitter=0.03)
    wp = s13.paired_by_request(rows, "spin30k", s13.WARM_KEYS["report_duration"], "0xd40")
    null = s13.control_null(rows, s13.WARM_KEYS["report_duration"], "0xd40")
    assert s13.warm_verdict(wp, null) == "no-change"


def test_pooled_cpu_mix_cannot_create_an_effect():
    """ctl lands mostly on the slow part, the arm mostly on the fast part: a pooled comparison would call the arm
    faster. Within each part there is no effect, so the paired verdict must be no-change on both."""
    slow = [
        r | {"report": {"Duration": r["report"]["Duration"] * 1.8}}
        for r in _rows({"ctl": 0, "active": 0}, cpu="0xd0c", jitter=0.03, seed=2)
    ]
    fast = _rows({"ctl": 0, "active": 0}, cpu="0xd40", jitter=0.03, seed=3)
    rows = [r for r in slow if r["arm"] == "ctl"] + [r for r in slow if r["arm"] == "active"][::5]
    rows += [r for r in fast if r["arm"] == "active"] + [r for r in fast if r["arm"] == "ctl"][::5]
    for cpu in ("0xd0c", "0xd40"):
        wp = s13.paired_by_request(rows, "active", s13.WARM_KEYS["report_duration"], cpu)
        null = s13.control_null(rows, s13.WARM_KEYS["report_duration"], cpu)
        assert s13.warm_verdict(wp, null) in ("no-change", "insufficient")
    pooled_ctl = statistics.median(r["report"]["Duration"] for r in rows if r["arm"] == "ctl")
    pooled_arm = statistics.median(r["report"]["Duration"] for r in rows if r["arm"] == "active")
    assert pooled_arm < 0.8 * pooled_ctl  # what a pooled comparison would wrongly report


def test_rows_without_cpu_part_are_never_compared():
    rows = _rows({"ctl": 0, "passive": -4}, cpu=None)
    assert s13.analyse(rows, "passive")["warm_paired"]["report_duration"] == {}
    hv = s13.host_variability(rows, {"passive": s13.analyse(rows, "passive")})
    assert hv["unknown"]["rows_without_cpu_part"]["passive:warm"] == len(rows) // 2


def test_status_regressed_when_warm_is_slower():
    rows = _rows({"ctl": 0, "active": +6})
    ana = s13.analyse(rows, "active")
    assert ana["warm_paired"]["report_duration"]["0xd40"]["verdict"] == "slower"
    assert s13.status_of(ana, 0, {"compared": 1, "bit_identical": 1}) == "regressed"


def test_status_inconclusive_when_an_answer_differs():
    rows = _rows({"ctl": 0, "passive": -4})
    rows[-1]["answers"] = {"q": {"choice": "y"}}
    ana = s13.analyse(rows, "passive")
    assert ana["arm_agreement"]["bit_identical"] == ana["arm_agreement"]["compared"] - 1
    assert s13.status_of(ana, 0, {"compared": 1, "bit_identical": 1}) == "inconclusive"


def test_sign_flip_p_floor():
    assert s13.sign_flip_p([1.0, 2.0, 3.0, 4.0, 5.0]) == round(2 / 32, 5)
    assert s13.sign_flip_p([1.0]) is None
    assert s13.sign_flip_p([1.0, -1.0]) == 1.0


def test_spincount_reads_libgomp_display(tmp_path):
    import gzip as _gz

    (tmp_path / "logs").mkdir()
    with _gz.open(tmp_path / "logs" / "lkai-e13-active.jsonl.gz", "wt") as f:
        for m in ("[host] OMP_WAIT_POLICY = 'ACTIVE'", "[host] GOMP_SPINCOUNT = '30000000000'"):
            f.write(json.dumps({"timestamp": 0, "logStreamName": "s", "message": m}) + "\n")
    cond = {
        "functions": {
            "active": {"FunctionName": "lkai-e13-active"},
            "ctl": {"FunctionName": "lkai-e13-ctl"},
        }
    }
    out = s13.spincount(str(tmp_path), cond)
    assert out["active"] == {"gomp_spincount": {"30000000000": 1}, "omp_wait_policy": {"ACTIVE": 1}}
    assert out["ctl"] == {"gomp_spincount": {}, "omp_wait_policy": {}}


def test_expected_spincounts_match_libgomp_defaults():
    assert e13deploy.EXPECTED_SPINCOUNT == {
        "ctl": 300000,
        "passive": 0,
        "spin30k": 30000,
        "active": 30000000000,
    }


def test_underpowered_part_is_not_a_verdict():
    rows = _rows({"ctl": 0, "passive": -4}, rounds=4)
    wp = s13.paired_by_request(rows, "passive", s13.WARM_KEYS["report_duration"], "0xd40")
    assert wp["min_attainable_p"] == 0.125
    assert s13.warm_verdict(wp, {"max_abs_round_pct": 0.1}) == "underpowered"


def test_improved_needs_no_contrary_sign_on_an_underpowered_part():
    rows = _rows({"ctl": 0, "passive": -4}) + _rows(
        {"ctl": 0, "passive": +4}, rounds=3, cpu="0xd0c", seed=9
    )
    ana = s13.analyse(rows, "passive")
    assert ana["warm_paired"]["report_duration"]["0xd40"]["verdict"] == "faster"
    assert ana["warm_paired"]["report_duration"]["0xd0c"]["verdict"] == "underpowered"
    assert s13.status_of(ana, 0, {"compared": 1, "bit_identical": 1}) == "no-change"
    agree = _rows({"ctl": 0, "passive": -4}) + _rows(
        {"ctl": 0, "passive": -4}, rounds=3, cpu="0xd0c", seed=9
    )
    assert (
        s13.status_of(s13.analyse(agree, "passive"), 0, {"compared": 1, "bit_identical": 1})
        == "improved"
    )


def test_client_error_is_transport_only_without_an_unmatched_report():
    rows = _rows({"ctl": 0, "passive": -4}, rounds=2, n=1)
    for i, r in enumerate(rows):
        r["log_tail"] = f"REPORT RequestId: id-{i}\tDuration: 1 ms"
    err = {"arm": "passive", "round": 1, "wave": "warm", "error": "client:SSLError", "detail": ""}
    ids = {"passive": {f"id-{i}" for i, r in enumerate(rows) if r["arm"] == "passive"}}
    k = s13.error_kinds([err], rows, ids)
    assert (k["client_transport_errors"], k["function_errors"]) == (1, 0)
    ids["passive"].add("ran-but-lost")  # Lambda ran something the client never saw
    k = s13.error_kinds([err], rows, ids)
    assert (k["client_transport_errors"], k["function_errors"]) == (0, 1)
    fn = err | {"error": "Unhandled"}
    assert s13.error_kinds([fn], rows, {"passive": set()})["function_errors"] == 1


def test_between_arms_uses_shared_rounds():
    rows = _rows({"ctl": 0, "passive": -4, "spin30k": -5})
    per = {a: s13.analyse(rows, a) for a in ("passive", "spin30k")}
    d = s13.between_arms(per, "spin30k", "passive")
    assert len(d["rounds"]) == 6 and d["rounds_below_zero"] == 6
    assert -1.2 < d["mean_pp"] < -0.8
    assert s13.between_arms(per, "spin30k", "active") is None
