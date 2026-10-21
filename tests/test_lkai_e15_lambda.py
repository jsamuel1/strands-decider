"""examples/lambda-kleidiai/e15 Lambda harness: the arms, the zips, the wrapper, the bench order and the summary.

Ported from E13's tests (lkai/a088-omp-wait 0dc2eed). No AWS and no llama.cpp. The tests pin E15's one-variable
contract: ctl runs P1's zip plus the wrapper; off and a run one candidate zip that differs from ctl's only in the six
lib/libggml-cpu-*.so; a differs from off only in LKAI_E15_A=1.
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

E15 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e15")
)
BASE = os.path.abspath(os.path.join(E15, "..", "baseline"))
sys.path.insert(0, E15)
sys.path.insert(0, BASE)

import e15_zip  # noqa: E402


def _sdk_stand_ins() -> dict[str, types.ModuleType]:
    """e15deploy/e15bench import boto3/botocore at module level; the test env need not have them. No test makes
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
    import e15bench
    import e15deploy
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


def test_arms_differ_only_in_the_switch_and_the_label():
    envs = {arm: e15deploy.arm_environment(P1_ENV, arm) for arm in e15deploy.ARMS}
    keys = set().union(*envs.values())
    differing = {k for k in keys if len({e.get(k) for e in envs.values()}) > 1}
    assert differing == {"E15_ARM", "LKAI_E15_A"}
    assert "LKAI_E15_A" not in envs["ctl"] and "LKAI_E15_A" not in envs["off"]
    assert envs["a"]["LKAI_E15_A"] == "1"
    for env in envs.values():
        assert {k: env[k] for k in P1_ENV} == P1_ENV  # P1's environment is kept as is
        assert env["LKAI_PHASES"] == "1"
        for k in ("THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "LKAI_E15_B", "LKAI_NODEPROF"):
            assert k not in env


def test_off_and_a_share_one_zip_and_ctl_has_its_own():
    assert e15deploy.ARM_ZIP == {"ctl": "ctl", "off": "cand", "a": "cand"}


@pytest.mark.parametrize("key", ["LKAI_E15_A", "LKAI_E15_B", "LKAI_NODEPROF", "E15_ARM"])
def test_p1_environment_that_sets_a_switch_is_refused(key):
    with pytest.raises(SystemExit):
        e15deploy.arm_environment(P1_ENV | {key: "1"}, "ctl")


# --- the zips ----------------------------------------------------------------------------------------------------


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    path = tmp_path / "p1.zip"
    files = {
        "lkai_entry.py": b"X = 1\n",
        "lib/libgomp.so.1": b"\x7fELF gomp",
        "assets/tokenizer.json": b"{}",
    }
    files |= {name: b"\x7fELF p1 " + name.encode() for name in e15_zip.VARIANTS}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.external_attr = (0o755 if name.endswith(".so") else 0o644) << 16
            z.writestr(info, data)
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


def _cand_libs(tmp_path, extra: bytes = b"") -> str:
    d = tmp_path / "cand"
    d.mkdir(exist_ok=True)
    for name in e15_zip.VARIANTS:
        body = b"\x7fELF cand " + name.encode()
        if name.endswith("armv8.6_1.so"):
            body += b" lkai_e15_set_a"
        (d / os.path.basename(name)).write_bytes(body + extra)
    (d / "toolchain.txt").write_text("gcc 11\n")
    return str(d)


def test_ctl_zip_is_p1_byte_for_byte_plus_wrapper(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    m = e15_zip.build(p1, man, _cand_libs(tmp_path), str(tmp_path / "out"))
    with zipfile.ZipFile(p1) as a, zipfile.ZipFile(tmp_path / "out" / "ctl.zip") as b:
        assert [i.filename for i in b.infolist()] == [i.filename for i in a.infolist()] + [
            "e15_entry.py"
        ]
        for ia in a.infolist():
            ib = b.getinfo(ia.filename)
            assert a.read(ia) == b.read(ib)
            assert (ia.date_time, ia.external_attr) == (ib.date_time, ib.external_attr)
        assert b.read("e15_entry.py") == open(os.path.join(E15, "e15_entry.py"), "rb").read()
    assert m["zips"]["ctl"]["handler"] == "e15_entry.handler"
    assert m["function_zip"]["sha256"] == m["zips"]["ctl"]["sha256"]


def test_cand_zip_differs_from_ctl_only_in_the_variant_libraries(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    libs = _cand_libs(tmp_path)
    m = e15_zip.build(p1, man, libs, str(tmp_path / "out"))
    with (
        zipfile.ZipFile(tmp_path / "out" / "ctl.zip") as c,
        zipfile.ZipFile(tmp_path / "out" / "cand.zip") as d,
    ):
        assert c.namelist() == d.namelist()
        differ = set()
        for ic in c.infolist():
            idd = d.getinfo(ic.filename)
            assert (ic.date_time, ic.external_attr, ic.compress_type) == (
                idd.date_time,
                idd.external_attr,
                idd.compress_type,
            )
            if c.read(ic) != d.read(idd):
                differ.add(ic.filename)
        assert differ == set(e15_zip.VARIANTS)
        for name in e15_zip.VARIANTS:
            assert d.read(name) == open(os.path.join(libs, os.path.basename(name)), "rb").read()
    for name, h in m["changed_entries"].items():
        assert h["p1_sha256"] != h["cand_sha256"]
        assert (
            h["cand_sha256"]
            == hashlib.sha256(
                open(os.path.join(libs, os.path.basename(name)), "rb").read()
            ).hexdigest()
        )
    assert m["zips"]["ctl"]["sha256"] != m["zips"]["cand"]["sha256"]


@pytest.mark.parametrize("marker", [b"lkai_np_", b"lkai_gdn_rowblock"])
def test_cand_zip_refuses_libraries_with_the_profiler_or_candidate_b(tmp_path, marker):
    p1, man = _p1_like_zip(tmp_path)
    with pytest.raises(SystemExit, match=r"patch_a\.py only"):
        e15_zip.build(p1, man, _cand_libs(tmp_path, marker), str(tmp_path / "out"))


def test_cand_zip_refuses_libraries_without_candidate_a(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    libs = _cand_libs(tmp_path)
    p = os.path.join(libs, "libggml-cpu-armv8.6_1.so")
    open(p, "wb").write(b"\x7fELF no switch")
    with pytest.raises(SystemExit, match="lacks candidate A"):
        e15_zip.build(p1, man, libs, str(tmp_path / "out"))


def test_zip_refuses_a_zip_its_manifest_does_not_describe(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    with open(p1, "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit):
        e15_zip.build(p1, man, _cand_libs(tmp_path), str(tmp_path / "out"))


# --- the bench's order -------------------------------------------------------------------------------------------

ARMS = list(e15deploy.ARMS)


def test_round_order_puts_every_arm_in_every_position():
    for start in (0, len(ARMS)):  # each cycle of len(ARMS) rounds, forward and reversed
        seen = Counter()
        for rnd in range(start, start + len(ARMS)):
            order = e15bench.round_order(ARMS, rnd)
            assert sorted(order) == sorted(ARMS)
            seen.update((pos, arm) for pos, arm in enumerate(order))
        assert all(seen[(p, a)] == 1 for p in range(len(ARMS)) for a in ARMS)


def test_warm_passes_interleave_and_lead_equally_within_a_round():
    plan = e15bench.warm_plan(["a", "b", "c", "d", "e"], 3)
    assert len(plan) == 15 and Counter(plan) == dict.fromkeys("abcde", 3)
    for rnd in range(6):
        firsts = Counter(e15bench.pass_order(ARMS, rnd, i)[0] for i in range(len(plan)))
        assert set(firsts.values()) == {len(plan) // len(ARMS)}
        for i in range(len(plan)):
            assert sorted(e15bench.pass_order(ARMS, rnd, i)) == sorted(ARMS)


# --- the wrapper -------------------------------------------------------------------------------------------------


@pytest.fixture
def entry(monkeypatch):
    """e15_entry imported against a stand-in lkai_entry (P1's needs the engine)."""

    def handler(event, context):
        return {"answers": {}, "lkai": {"inv": {"cpu_part": "0xd40"}}}

    monkeypatch.setitem(sys.modules, "lkai_entry", types.SimpleNamespace(handler=handler))
    monkeypatch.setenv("E15_ARM", "a")
    monkeypatch.setenv("LKAI_E15_A", "1")
    sys.modules.pop("e15_entry", None)
    mod = importlib.import_module("e15_entry")
    yield mod
    sys.modules.pop("e15_entry", None)


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


def test_handler_adds_record_on_first_invoke_only(entry, capsys):
    ctx = types.SimpleNamespace(log_stream_name="s/1")
    first = entry.handler({}, ctx)
    second = entry.handler({}, ctx)
    rec = first["lkai"]["e15"]
    assert rec["switch_env"]["LKAI_E15_A"] == "1" and rec["switch_env"]["LKAI_E15_B"] == "unset"
    assert rec["e15_arm"] == "a" and rec["log_stream"] == "s/1"
    assert "ggml_cpu_mapped" in rec
    assert "e15" not in second["lkai"]
    assert set(second["lkai"]["e15_inv"]) == {
        "wall_s",
        "utime_s",
        "stime_s",
        "cpu_per_wall",
        "nvcsw",
        "nivcsw",
    }
    out = capsys.readouterr().out.splitlines()
    assert len([ln for ln in out if ln.startswith("E15 ")]) == 1


def test_e15_sources_name_no_cpu_generation():
    for f in ("e15_entry.py", "e15deploy.py", "e15bench.py", "summarize_e15.py", "meta.json"):
        src = open(os.path.join(E15, f)).read()
        for word in ("Graviton2", "Graviton3", "Graviton4", "Graviton5", "Neoverse"):
            assert word not in src, (f, word)


# --- the summary -------------------------------------------------------------------------------------------------

import summarize_e15 as s15  # noqa: E402

REQS = ["a", "b", "c", "d", "e"]
BASE_MS = {"a": 250, "b": 400, "c": 600, "d": 1000, "e": 1750}
ANS = {
    "q": {"type": "choice", "choice": "x", "probabilities": {"x": 0.6, "y": 0.4}},
    "u": {"noul": 0.5},
}


def _variant_rec(cpu):
    return {
        "ggml_cpu_mapped": [
            "libggml-cpu-armv8.2_2.so" if cpu == "0xd0c" else "libggml-cpu-armv8.6_1.so"
        ]
    }


def _warm(arm, rnd, rid, ms, cpu="0xd40", answers=None):
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
        "answers": answers or ANS,
        "lkai": {
            "e15_inv": {"utime_s": 2.0, "stime_s": 0.1, "nvcsw": 10, "nivcsw": 1},
            "e15": _variant_rec(cpu),
        },
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


def _shift(ans: dict, dp: float) -> dict:
    out = json.loads(json.dumps(ans))
    out["q"]["probabilities"] = {"x": 0.6 + dp, "y": 0.4 - dp}
    return out


OK_REF = {"compared": 1, "bit_identical": 0, "within_tolerance": True}


def test_paired_warm_detects_a_consistent_speedup_despite_round_drift():
    rows = _rows({"ctl": 0, "a": -2})
    wp = s15.paired_by_request(rows, "a", s15.WARM_KEYS["report_duration"], "0xd40")
    assert wp["cells"] == 30 and wp["rounds_below_zero"] == 6
    assert -2.5 < wp["mean_round_pct"] < -1.5
    null = s15.control_null(rows, s15.WARM_KEYS["report_duration"], "0xd40")
    assert s15.warm_verdict(wp, null) == "faster"


def test_switch_comparison_uses_off_as_the_control():
    rows = _rows({"ctl": 0, "off": 0, "a": -2})
    sw = s15.analyse(rows, "a", ctl="off")
    assert sw["control"] == "off"
    assert sw["round_cells"]["warm_report_duration"]["control"] == "off"
    assert sw["warm_paired"]["report_duration"]["0xd40"]["verdict"] == "faster"


def test_pooled_cpu_mix_cannot_create_an_effect():
    slow = [
        r | {"report": {"Duration": r["report"]["Duration"] * 1.8}}
        for r in _rows({"ctl": 0, "a": 0}, cpu="0xd0c", jitter=0.03, seed=2)
    ]
    fast = _rows({"ctl": 0, "a": 0}, cpu="0xd40", jitter=0.03, seed=3)
    rows = [r for r in slow if r["arm"] == "ctl"] + [r for r in slow if r["arm"] == "a"][::5]
    rows += [r for r in fast if r["arm"] == "a"] + [r for r in fast if r["arm"] == "ctl"][::5]
    for cpu in ("0xd0c", "0xd40"):
        wp = s15.paired_by_request(rows, "a", s15.WARM_KEYS["report_duration"], cpu)
        null = s15.control_null(rows, s15.WARM_KEYS["report_duration"], cpu)
        assert s15.warm_verdict(wp, null) in ("no-change", "insufficient")
    pooled_ctl = statistics.median(r["report"]["Duration"] for r in rows if r["arm"] == "ctl")
    pooled_arm = statistics.median(r["report"]["Duration"] for r in rows if r["arm"] == "a")
    assert pooled_arm < 0.8 * pooled_ctl  # what a pooled comparison would wrongly report


def test_rows_without_cpu_part_are_never_compared():
    rows = _rows({"ctl": 0, "a": -2}, cpu=None)
    assert s15.analyse(rows, "a")["warm_paired"]["report_duration"] == {}
    hv = s15.host_variability(rows, {"a": s15.analyse(rows, "a")})
    assert hv["unknown"]["rows_without_cpu_part"]["a:warm"] == len(rows) // 2


def test_agreement_reports_probability_delta_and_tolerance():
    rows = _rows({"ctl": 0, "a": -2}, rounds=2, n=2)
    for r in rows:
        if r["arm"] == "a":
            r["answers"] = _shift(ANS, 0.015)
    g = s15.arm_agreement(rows, "a")
    assert g["bit_identical"] == 0 and g["choice_flips"] == 0
    assert abs(g["max_abs_probability_delta"] - 0.015) < 1e-9 and g["within_tolerance"]
    for r in rows:
        if r["arm"] == "a":
            r["answers"] = _shift(ANS, 0.025)
    assert not s15.arm_agreement(rows, "a")["within_tolerance"]


def test_a_choice_flip_is_outside_tolerance():
    rows = _rows({"ctl": 0, "a": -2}, rounds=2, n=1)
    flipped = _shift(ANS, 0.001)
    flipped["q"]["choice"] = "y"
    rows[-1]["answers"] = flipped
    g = s15.arm_agreement(rows, "a")
    assert g["choice_flips"] == 1 and not g["within_tolerance"]


def test_status_a_improved_within_tolerance():
    rows = _rows({"ctl": 0, "a": -2})
    for r in rows:
        if r["arm"] == "a":
            r["answers"] = _shift(ANS, 0.01)
    ana = s15.analyse(rows, "a")
    assert s15.status_of("a", ana, 0, OK_REF) == "improved"
    assert s15.status_of("a", ana, 0, OK_REF | {"within_tolerance": False}) == "inconclusive"
    assert s15.status_of("a", ana, 1, OK_REF) == "inconclusive"


def test_status_off_must_be_bit_identical():
    rows = _rows({"ctl": 0, "off": 0}, jitter=0.03)
    ana = s15.analyse(rows, "off")
    exact = {"compared": 5, "bit_identical": 5, "within_tolerance": True}
    assert s15.status_of("off", ana, 0, exact) == "no-change"
    rows[-1]["answers"] = _shift(ANS, 0.001)
    ana = s15.analyse(rows, "off")
    assert s15.status_of("off", ana, 0, exact) == "inconclusive"


def test_status_ignores_parts_the_candidate_cannot_change_but_needs_them_exact():
    # 0xd40 loads armv8.6_1 (changed): faster. 0xd0c loads armv8.2_2 (unchanged): slower but not a verdict.
    rows = _rows({"ctl": 0, "a": -2}) + _rows({"ctl": 0, "a": +3}, cpu="0xd0c", seed=7)
    ana = s15.analyse(rows, "a")
    assert ana["applicability"]["0xd0c"]["changed_by_candidate"] is False
    assert ana["applicability"]["0xd40"]["changed_by_candidate"] is True
    assert ana["warm_paired"]["report_duration"]["0xd0c"]["verdict"] == "slower"
    assert s15.status_of("a", ana, 0, OK_REF) == "improved"
    # an answer change on an unchanged part means the patch touched something it should not
    for r in rows:
        if r["arm"] == "a" and r["cpu_part"] == "0xd0c":
            r["answers"] = _shift(ANS, 0.001)
            break
    assert s15.status_of("a", s15.analyse(rows, "a"), 0, OK_REF) == "inconclusive"


def test_status_regressed_when_a_changed_part_is_slower():
    rows = _rows({"ctl": 0, "a": +6})
    ana = s15.analyse(rows, "a")
    assert ana["warm_paired"]["report_duration"]["0xd40"]["verdict"] == "slower"
    assert s15.status_of("a", ana, 0, OK_REF) == "regressed"


def test_improved_needs_no_contrary_sign_on_an_underpowered_changed_part():
    rows = _rows({"ctl": 0, "a": -2}) + _rows({"ctl": 0, "a": +4}, rounds=3, cpu="0xd4f", seed=9)
    ana = s15.analyse(rows, "a")
    assert ana["applicability"]["0xd4f"]["changed_by_candidate"] is True
    assert ana["warm_paired"]["report_duration"]["0xd4f"]["verdict"] == "underpowered"
    assert s15.status_of("a", ana, 0, OK_REF) == "no-change"


def test_applicability_unknown_without_a_variant_record():
    rows = _rows({"ctl": 0, "a": -2}, rounds=2, n=1)
    for r in rows:
        r["lkai"].pop("e15")
    assert s15.applicability(rows)["0xd40"] == {"variants": [], "changed_by_candidate": "unknown"}


def test_reference_tolerance(tmp_path):
    rows = _rows({"ctl": 0, "a": -2}, rounds=1, n=1)
    for r in rows:
        if r["arm"] == "a":
            r["answers"] = _shift(ANS, 0.012)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"outputs": {rid: {"0xd40": ANS} for rid in REQS}}))
    t = s15.reference_tolerance(rows, str(ref), "a")
    assert t["compared"] == 5 and t["bit_identical"] == 0 and t["within_tolerance"]
    assert abs(t["max_abs_probability_delta"] - 0.012) < 1e-9
    assert s15.reference_tolerance(rows, str(ref), "ctl")["bit_identical"] == 5


def test_sign_flip_p_floor():
    assert s15.sign_flip_p([1.0, 2.0, 3.0, 4.0, 5.0]) == round(2 / 32, 5)
    assert s15.sign_flip_p([1.0]) is None


def test_client_error_is_transport_only_without_an_unmatched_report():
    rows = _rows({"ctl": 0, "a": -2}, rounds=2, n=1)
    for i, r in enumerate(rows):
        r["log_tail"] = f"REPORT RequestId: id-{i}\tDuration: 1 ms"
    err = {"arm": "a", "round": 1, "wave": "warm", "error": "client:SSLError", "detail": ""}
    ids = {"a": {f"id-{i}" for i, r in enumerate(rows) if r["arm"] == "a"}}
    k = s15.error_kinds([err], rows, ids)
    assert (k["client_transport_errors"], k["function_errors"]) == (1, 0)
    ids["a"].add("ran-but-lost")
    k = s15.error_kinds([err], rows, ids)
    assert (k["client_transport_errors"], k["function_errors"]) == (0, 1)


def test_conclusion_templates_format_with_missing_values():
    meta = json.load(open(os.path.join(E15, "meta.json")))
    values = dict.fromkeys(
        (
            "status cpu rounds cells w_mean w_lo w_hi w_below w_p w_null w_verdict w_ctl_ms w_arm_ms eng_mean "
            "cpu_mean sw_mean sw_p sw_below other_parts applicability cold_d cold_lo cold_hi cold_n rst_d "
            "rst_lo rst_hi rst_n mem_d agree agree_n agree_max agree_flips ref ref_n ref_max ref_flips "
            "ref_missing transport function_errors restore_timeouts"
        ).split(),
        "n/a",
    ) | {"cost": 1.0}
    for arm in ("off", "a"):
        assert "n/a" in meta["arms"][arm]["conclusion_template"].format(**values)
