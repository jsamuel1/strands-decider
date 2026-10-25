"""General ladder (examples/lambda-kleidiai/general): interleaving, host identity, paired verdicts, P2 shape."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

G = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "general")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"lkai_general_{name}", os.path.join(G, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


host = _load("host")
run_ladder = _load("run_ladder")
summ = _load("summarize")
make_arms = _load("make_arms")


# ---------------------------------------------------------------------------------------------------------- host
ARM_CPUINFO = """processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd i8mm
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
CPU revision\t: 1

processor\t: 1
CPU part\t: 0xd40
"""
X86_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 173
model name\t: Intel(R) Xeon(R) 6975P-C
stepping\t: 1
flags\t\t: avx2 avx512f
"""


def test_arm_identity_is_the_printed_part_and_no_generation():
    cpu = host.cpuinfo(ARM_CPUINFO)
    assert cpu["part"] == "0xd40" and cpu["processors"] == 2 and cpu["variant"] == "0x1"
    assert host.identity(cpu) == "0xd40"
    assert "graviton" not in json.dumps(cpu).lower()


def test_x86_identity_and_unknowns():
    cpu = host.cpuinfo(X86_CPUINFO)
    assert host.identity(cpu) == "GenuineIntel 6/173/1"
    assert cpu["part"] == host.UNKNOWN
    assert host.identity(host.cpuinfo(None)) == host.UNKNOWN
    assert host.cpuinfo(None)["processors"] == host.UNKNOWN


def test_auxv_and_status_parsers():
    raw = (
        (16).to_bytes(8, "little")
        + (0xCF).to_bytes(8, "little")
        + (26).to_bytes(8, "little")
        + (3).to_bytes(8, "little")
    )
    assert host.auxv(raw) == {"hwcap": "0xcf", "hwcap2": "0x3"}
    assert host.auxv(b"\0" * 7) == {"hwcap": host.UNKNOWN, "hwcap2": host.UNKNOWN}
    st = host.proc_status_kb("VmRSS:\t  10 kB\nVmHWM:\t 20 kB\n")
    assert st["VmRSS"] == 10 and st["VmHWM"] == 20 and st["RssShmem"] == host.UNKNOWN


# ----------------------------------------------------------------------------------------------- interleaving
def test_rotation_is_a_latin_square_over_rounds():
    n = 16
    orders = [run_ladder.order(n, r) for r in range(n)]
    for o in orders:
        assert sorted(o) == list(range(n))
    for pos in range(n):  # every cell takes every position once over n rounds
        assert sorted(o[pos] for o in orders) == list(range(n))


def test_cells_and_only_filter():
    arms = [{"name": "p1", "modes": ["local", "memfd"]}, {"name": "e6"}]
    assert [(a["name"], m) for a, m in run_ladder.cells(arms)] == [
        ("p1", "local"),
        ("p1", "memfd"),
        ("e6", "local"),
    ]
    assert [(a["name"], m) for a, m in run_ladder.cells(arms, {"e6"})] == [("e6", "local")]


def test_docker_cmd_pins_cpus_mounts_ro_and_keeps_arm_env(tmp_path):
    class A:
        platform, cpuset, threads, image, warm_passes, long_passes = (
            "linux/arm64",
            "10-15",
            6,
            "img",
            3,
            2,
        )
        out = str(tmp_path)

    g = tmp_path / "m.gguf"
    g.write_bytes(b"x")
    arm = {
        "name": "e8",
        "pkg": "/p",
        "gguf": str(g),
        "env": {"GGML_KLEIDIAI_REPACK_THREADS": "6"},
        "args": ["--n-ctx", 4096],
    }
    cmd = run_ladder.docker_cmd({"requests": "/r", "reference": "/f"}, arm, "memfd", "x.json", A)
    s = " ".join(cmd)
    assert "--cpuset-cpus 10-15" in s and "--network none" in s and "THREADS=6" in s
    assert "/p:/var/task:ro" in s and f"{g}:/m/model.gguf:ro" in s
    assert (
        "GGML_KLEIDIAI_REPACK_THREADS=6" in s
        and "--mode memfd" in s
        and cmd[-2:] == ["--n-ctx", "4096"]
    )


def test_make_arms_one_variable_each(tmp_path):
    arms = {a["name"]: a for a in make_arms.arms(str(tmp_path / "s"), str(tmp_path / "a"))}
    p1 = arms["p1"]
    # E8, E7: the switch is the only environment difference; E10: only the n_ctx argument; E5/E9: only files.
    assert arms["e8"]["env"] == {"GGML_KLEIDIAI_REPACK_THREADS": "6"} and "env" not in arms["e8off"]
    assert arms["e8"]["pkg"] == arms["e8off"]["pkg"] and arms["e8"]["gguf"] == p1["gguf"]
    assert (
        arms["e7"]["env"] == {"LKAI_SKIP_TIED_OUTPUT": "1"}
        and arms["e7"]["pkg"] == arms["e7off"]["pkg"]
    )
    assert arms["e10"]["pkg"] == p1["pkg"] and arms["e10"]["args"] == ["--n-ctx", "4096"]
    assert arms["e5"]["pkg"] == p1["pkg"] and arms["e5"]["gguf"] != p1["gguf"]
    assert arms["e9p1"]["pkg"] == arms["e9"]["pkg"] and arms["e9p1"]["gguf"] == p1["gguf"]
    assert arms["e6"]["gguf"] == p1["gguf"] and arms["e6"]["pkg"] != p1["pkg"]


# --------------------------------------------------------------------------------------------- paired verdicts
def _row(arm, r, load, warm=0.5, ident="0xd40", mode="local", answers=None, long_ans=None):
    return {
        "_arm": arm,
        "_round": r,
        "_mode": mode,
        "_identity": ident,
        "_run": "t",
        "threads": 6,
        "phases_s": {
            "imports": 0.5,
            "model_load_mmap": load - 0.2,
            "context_init": 0.2,
            "dec_load_total": load,
            "warmup_inference": 0.2,
            "first_request_inference": 0.2,
            "model_ready": load + 1.0,
        },
        "warm_s": {"a": [warm, warm]},
        "long_forward": {
            "seconds": [16.0],
            "answers": long_ans or {"q": {"noul": 0.5}},
            "repeat_identical": True,
        },
        "status_end_kb": {"VmHWM": 5_000_000},
        "answers": answers or {"a": {"q": {"noul": 0.5, "type": "noul"}}},
        "answers_repeat_identical": True,
        "vs_reference": {"a": "identical"},
        "host_before": {"identity": ident, "machine": "aarch64", "sched_affinity": [10, 11]},
    }


def test_clear_effect_outside_spread_is_better():
    rows = [_row("p1", r, 8.0 + 0.05 * r) for r in range(5)] + [
        _row("e8", r, 2.3) for r in range(5)
    ]
    m = summ.paired(rows, "e8", "local", "0xd40")["metrics"]["load_s"]
    assert (
        m["outcome"] == "better"
        and m["delta_median"] < -5
        and m["control_spread"] == pytest.approx(0.2)
    )


def test_effect_inside_control_spread_is_no_change():
    rows = [_row("p1", r, 8.0 + (0.3 if r % 2 else 0)) for r in range(5)]
    rows += [_row("e5", r, 7.9 + (0.3 if r % 2 else 0)) for r in range(5)]
    assert summ.paired(rows, "e5", "local", "0xd40")["metrics"]["load_s"]["outcome"] == "no change"


def test_sign_change_across_rounds_is_no_change_even_if_median_is_large():
    rows = [_row("p1", r, 8.0) for r in range(5)]
    rows += [_row("x", r, v) for r, v in enumerate([7.0, 7.0, 7.0, 9.5, 9.5])]
    m = summ.paired(rows, "x", "local", "0xd40")["metrics"]["load_s"]
    assert m["control_spread"] == 0 and m["outcome"] == "no change"


def test_too_few_rounds_is_unassessed():
    rows = [_row("p1", 0, 8.0), _row("e8", 0, 2.0)]
    assert summ.paired(rows, "e8", "local", "0xd40")["metrics"]["load_s"]["outcome"] == "unassessed"


def test_pairs_only_shared_rounds_and_same_identity_and_mode():
    rows = [_row("p1", r, 8.0) for r in range(4)] + [_row("e8", r, 2.0) for r in (1, 2, 3, 4)]
    rows += [_row("e8", 0, 99.0, ident="0xd0c"), _row("e8", 0, 99.0, mode="memfd")]
    p = summ.paired(rows, "e8", "local", "0xd40")
    assert p["rounds"] == [("t", 1), ("t", 2), ("t", 3)] and p["metrics"]["load_s"]["deltas"] == [
        -6.0,
        -6.0,
        -6.0,
    ]


def test_pairs_never_cross_runs():
    rows = [_row("p1", r, 8.0) for r in range(3)]
    rows += [dict(_row("e6", r, 7.0), _run="other") for r in range(3)]
    assert summ.paired(rows, "e6", "local", "0xd40")["n"] == 0


def test_peak_rss_within_one_percent_is_no_change_even_with_zero_spread():
    rows = [_row("p1", r, 8.0) for r in range(3)]
    for r in range(3):
        x = _row("e8", r, 8.0)
        x["status_end_kb"] = {"VmHWM": 5_000_000 + 17 * 1024}
        rows.append(x)
    m = summ.paired(rows, "e8", "local", "0xd40")["metrics"]["peak_rss_mb"]
    assert m["control_spread"] == 0 and m["outcome"] == "no change"
    for r in rows[3:]:
        r["status_end_kb"] = {"VmHWM": 4_500_000}
    assert (
        summ.paired(rows, "e8", "local", "0xd40")["metrics"]["peak_rss_mb"]["outcome"] == "better"
    )


def test_worse_is_detected():
    rows = [_row("p1", r, 8.0, warm=0.50 + 0.001 * r) for r in range(5)]
    rows += [_row("x", r, 8.0, warm=0.60) for r in range(5)]
    assert summ.paired(rows, "x", "local", "0xd40")["metrics"]["warm_s"]["outcome"] == "worse"


def test_correctness_counts_and_choice_flips():
    a = {"a": {"q": {"choice": "x", "probabilities": {"x": 0.6, "y": 0.4}, "type": "choice"}}}
    b = {"a": {"q": {"choice": "y", "probabilities": {"x": 0.4, "y": 0.6}, "type": "choice"}}}
    rows = [_row("p1", 0, 8, answers=a), _row("e9", 0, 3, answers=a), _row("e7", 0, 3, answers=b)]
    c9 = summ.correctness(rows, "e9", "0xd40")
    assert (
        c9["vs_run_p1_requests"] == {"identical": 1, "compared": 1}
        and c9["choice_flips_vs_p1"] == 0
    )
    c7 = summ.correctness(rows, "e7", "0xd40")
    assert c7["vs_run_p1_requests"]["identical"] == 0 and c7["choice_flips_vs_p1"] == 1
    assert c7["max_abs_delta_vs_p1"] == pytest.approx(0.2)


def test_nan_answers_compare_by_text_and_are_listed():
    nan_a = {"a": {"q": {"noul": float("nan"), "type": "noul"}}}
    other = {"a": {"q": {"noul": float("nan"), "type": "noul"}}}  # a distinct NaN object
    rows = [_row("p1", 0, 8, answers=nan_a), _row("e8", 0, 3, answers=other)]
    c = summ.correctness(rows, "e8", "0xd40")
    assert c["vs_run_p1_requests"] == {"identical": 1, "compared": 1}
    assert c["nan_fields"] == ["a/q/noul"] and c["max_abs_delta_vs_p1"] == 0.0


def test_reference_differs_and_missing_identity_are_counted():
    r = _row("e10", 0, 8)
    r["vs_reference"] = {"a": "DIFFERS", "b": "no-reference-for-identity"}
    c = summ.correctness([_row("p1", 0, 8), r], "e10", "0xd40")
    assert c["vs_reference_requests"] == {"identical": 0, "differs": 1, "no_reference": 1}


def test_summarize_includes_e8_vs_e9_and_dormant():
    rows = []
    for r in range(3):
        rows += [
            _row("p1", r, 8.0 + 0.05 * r),
            _row("e8", r, 2.3),
            _row("e9", r, 3.2),
            _row("e8off", r, 8.01),
        ]
    s = summ.summarize(rows)
    assert s["e8_vs_e9"]["0xd40|local"]["metrics"]["load_s"]["delta_median"] == pytest.approx(0.9)
    assert s["dormant"]["e8"]["0xd40"]["timing"]["metrics"]["load_s"]["outcome"] == "no change"
    assert "e8" in s["rungs"] and "0xd40|correctness" in s["rungs"]["e8"]


def test_operator_rule_refuses_general_default_when_any_path_regresses():
    rows = []
    for r in range(3):
        rows += [
            _row("p1", r, 8.0 + 0.05 * r, warm=0.5 + 0.001 * r),
            _row("e8", r, 2.3, warm=0.5),
            _row("p1", r, 9.0 + 0.05 * r, ident="GenuineIntel 6/173/1", warm=0.4 + 0.001 * r),
            _row("e8", r, 9.0, ident="GenuineIntel 6/173/1", warm=0.48),
        ]
    s = summ.summarize(rows)
    why = summ.guard("e8", s, "general default")
    assert why and "GenuineIntel" in why[0] and "warm_s worse" in why[0]
    assert summ.guard("e8", s, "general opt-in") == []


def test_operator_rule_refuses_general_default_when_answers_differ():
    rows = [_row("p1", r, 8.0 + 0.05 * r) for r in range(3)]
    rows += [
        _row("e7", r, 6.0, answers={"a": {"q": {"noul": 0.51, "type": "noul"}}}) for r in range(3)
    ]
    assert any(
        "answers differ" in w for w in summ.guard("e7", summ.summarize(rows), "general default")
    )


def test_failed_loads_carry_their_run(tmp_path):
    d = tmp_path / "20261025T000000Z-x86"
    d.mkdir()
    (d / "runlog.jsonl").write_text(
        json.dumps({"arm": "e9", "rc": 1}) + "\n" + json.dumps({"arm": "p1", "rc": 0}) + "\n"
    )
    assert summ.failed_loads(str(d)) == [{"arm": "e9", "rc": 1, "run": "20261025T000000Z-x86"}]


def test_shipped_verdicts_obey_the_operator_rule_shape():
    vs = json.load(open(os.path.join(G, "verdicts.json")))
    assert set(vs["rungs"]) == set(summ.RUNGS)
    for r, v in vs["rungs"].items():
        assert v["general"] in ("general default", "general opt-in", "Lambda-only")
        assert len(v["commit"]) == 40 and "FIXED HOST, NOT LAMBDA" in v["conclusion"], r


# ------------------------------------------------------------------------------------------------------ P2 shape
def test_p2_result_validates_with_the_publisher_rules():
    lk_path = os.environ.get("LKAI_RESULTS_PY") or os.path.join(
        os.path.dirname(G), "results", "lkai_results.py"
    )
    if not os.path.exists(lk_path):
        pytest.skip(
            "P2 publisher (feat/lambda-kleidiai-results) not on this branch; set LKAI_RESULTS_PY"
        )
    lk = _load_path(lk_path)
    rows = []
    for r in range(3):
        rows += [_row("p1", r, 8.0), _row("e8", r, 2.3), _row("e8off", r, 8.0)]
    s = summ.summarize(rows)
    verdict = {
        "status": "improved",
        "hypothesis": "h",
        "started": "2026-10-25T00:00:00Z",
        "finished": "2026-10-25T01:00:00Z",
        "errors": 0,
        "error_rows": [],
        "reproduce": ["x"],
        "conclusion": "c",
        "general": "general default",
    }
    res = summ.p2_result("e8", rows, s, verdict, "t1", "abcdef1", "local", "0xd40", None)
    assert lk.validate(res) == []


def _load_path(path: str):
    spec = importlib.util.spec_from_file_location("lkai_results_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_p2_result_has_no_account_ids_and_labels_fixed_host():
    rows = []
    for r in range(3):
        rows += [_row("p1", r, 8.0), _row("e10", r, 7.9)]
    s = summ.summarize(rows)
    verdict = {
        "status": "no-change",
        "hypothesis": "h",
        "started": "s",
        "finished": "f",
        "errors": 0,
        "error_rows": [],
        "reproduce": ["x"],
        "conclusion": "c",
        "general": "general default",
    }
    res = summ.p2_result("e10", rows, s, verdict, "t1", "abcdef1", "local", "0xd40", None)
    text = json.dumps(res)
    assert "FIXED HOST, NOT LAMBDA" in text and res["deployment"]["region"].startswith("fixed-host")
    import re

    assert not re.search(r"\b\d{12}\b", text)
    assert res["phases"]["e10"]["restore"]["report_restore_duration"] is None
