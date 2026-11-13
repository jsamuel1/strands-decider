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
PRODUCTION_ROUNDS = 6  # README / verdicts.json min_rounds: the round count the ladder actually runs


@pytest.mark.parametrize("n", [2, 3, 10, 16])
def test_schedule_balances_every_pair_within_the_production_round_count(n):
    sched = run_ladder.schedule(n, PRODUCTION_ROUNDS, 20261113)
    assert len(sched) == PRODUCTION_ROUNDS
    for o in sched:
        assert sorted(o) == list(range(n))
    for i in range(n):
        for j in range(n):
            if i != j:  # each candidate runs before its control exactly as often as after it
                assert run_ladder.balance(sched, i, j) == (PRODUCTION_ROUNDS // 2,) * 2
        # and its mean position is the middle of the round: a linear drift adds the same to every cell
        assert sum(o.index(i) for o in sched) * 2 == PRODUCTION_ROUNDS * (n - 1)


def test_schedule_varies_neighbours_and_is_reproducible():
    a = run_ladder.schedule(16, PRODUCTION_ROUNDS, 7)
    assert a == run_ladder.schedule(16, PRODUCTION_ROUNDS, 7)
    assert a[0] != a[2] != a[4]  # each reversal pair draws its own permutation


@pytest.mark.parametrize("rounds", [0, 1, 5])
def test_schedule_refuses_an_odd_round_count(rounds):
    with pytest.raises(ValueError):
        run_ladder.schedule(4, rounds, 1)


def test_prime_reads_every_artifact_once(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "a.so").write_bytes(b"x" * 10)
    (pkg / "e.py").write_bytes(b"y" * 5)
    g = tmp_path / "m.gguf"
    g.write_bytes(b"z" * 7)
    r, f = tmp_path / "r.json", tmp_path / "f.json"
    r.write_bytes(b"{}")
    f.write_bytes(b"{}")
    calls = []
    monkeypatch.setattr(
        run_ladder.subprocess,
        "run",
        lambda c, **k: calls.append(c) or type("P", (), {"returncode": 0})(),
    )

    class A:
        image, platform = "img", "linux/arm64"

    cfg = {"requests": str(r), "reference": str(f)}
    arm = {"pkg": str(pkg), "gguf": str(g)}
    pr = run_ladder.prime(cfg, [(arm, "local"), (arm, "memfd")], A)
    assert pr["files"] == 5 and pr["bytes"] == 26 and pr["image_start_rc"] == 0 and len(calls) == 1


def test_the_committed_runs_follow_a_balanced_schedule():
    runs = os.path.join(G, "runs")
    for run in sorted(os.listdir(runs)):
        sp = os.path.join(runs, run, "schedule.json")
        assert os.path.exists(sp), run
        plan = json.load(open(sp))
        assert os.path.exists(os.path.join(runs, run, "prime.json")), (
            run
        )  # warm cache before round 0
        assert plan["rounds"] >= PRODUCTION_ROUNDS and plan["rounds"] % 2 == 0, run
        log = [json.loads(x) for x in open(os.path.join(runs, run, "runlog.jsonl"))]
        ran = {}
        for x in log:
            ran.setdefault(x["round"], []).append((x["pos"], f"{x['arm']}-{x['mode']}"))
        for r, cells in ran.items():  # the runlog is the schedule, position by position
            assert [c for _, c in sorted(cells)] == plan["order"][r], (run, r)
        idx = {c: i for i, c in enumerate(plan["cells"])}
        sched = [[idx[c] for c in o] for o in plan["order"]]
        for c in plan["cells"]:
            if not c.startswith("p1-"):
                ctl = "p1-" + c.split("-")[1]
                assert run_ladder.balance(sched, idx[c], idx[ctl]) == (plan["rounds"] // 2,) * 2, (
                    run,
                    c,
                )


def test_ladder_refuses_to_resume_under_another_schedule(tmp_path, monkeypatch):
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps({"requests": "/r", "reference": "/f", "arms": [{"name": "p1"}, {"name": "e8"}]})
    )
    out = tmp_path / "run"
    out.mkdir()
    (out / "schedule.json").write_text(
        json.dumps({"rounds": 2, "seed": 1, "cells": [], "order": []})
    )
    monkeypatch.setattr(sys, "argv", ["run_ladder.py", str(arms), str(out), "--rounds", "2"])
    with pytest.raises(SystemExit, match="another schedule"):
        run_ladder.main()


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
def _row(
    arm, r, load, warm=0.5, ident="0xd40", mode="local", answers=None, long_ans=None, pos=None
):
    return {
        "_arm": arm,
        "_round": r,
        "_mode": mode,
        "_identity": ident,
        "_run": "t",
        # Balanced by default: p1 at 1, the arm before it in even rounds and after it in odd ones.
        "_pos": pos if pos is not None else (1 if arm == "p1" else (0 if r % 2 == 0 else 2)),
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
            "window": 4096,
            "input_tokens": 4096,
        },
        "status_end_kb": {"VmHWM": 5_000_000},
        "answers": answers or {"a": {"q": {"noul": 0.5, "type": "noul"}}},
        "answers_repeat_identical": True,
        "repeat_compare": "canonical-json",
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
    assert any("GenuineIntel" in w and "warm_s worse" in w for w in why)
    assert summ.guard("e8", s, "general opt-in") == []


def test_operator_rule_refuses_general_default_when_answers_differ():
    rows = [_row("p1", r, 8.0 + 0.05 * r) for r in range(4)]
    rows += [
        _row("e7", r, 6.0, answers={"a": {"q": {"noul": 0.51, "type": "noul"}}}) for r in range(4)
    ]
    assert any(
        "answers differ" in w for w in summ.guard("e7", summ.summarize(rows), "general default")
    )


# The operator rule fails closed: a complete, clean default passes; each missing or wrong piece of evidence refuses.
REQ = ["0xd40|local"]


def _clean(rounds=4, arm="e10"):
    rows = []
    for r in range(rounds):
        rows += [_row("p1", r, 8.0 + 0.05 * r), _row(arm, r, 7.0)]
    return rows


def _guard(rows, rung="e10", **kw):
    s = summ.summarize(rows)
    s["failed_loads"] = kw.pop("failed", [])
    s["control_nan_fields"] = {"0xd40": summ.correctness(rows, "p1", "0xd40")["nan_fields"]}
    kw.setdefault("required", REQ)
    kw.setdefault("min_rounds", 4)
    return summ.guard(rung, s, "general default", **kw)


def test_guard_allows_a_complete_clean_default():
    assert _guard(_clean()) == []


def test_guard_refuses_a_missing_required_cell():
    why = _guard(_clean(), required=[*REQ, "GenuineIntel 6/173/1|local"])
    assert any("GenuineIntel 6/173/1|local: required cell missing" in w for w in why)
    assert any("required cell missing" in w for w in _guard([], rung="e10"))


def test_guard_refuses_too_few_rounds_and_unassessed_metrics():
    why = _guard(_clean(rounds=2))
    assert any("2 paired rounds, needs 4" in w for w in why)
    assert any("load_s unassessed" in w for w in why)


def test_guard_refuses_4096_answers_that_differ():
    rows = _clean()
    for d in rows:
        if d["_arm"] == "e10":
            d["long_forward"]["answers"] = {"q": {"noul": 0.6}}
    assert any("4096-token answers differ" in w for w in _guard(rows))


def test_guard_refuses_a_long_forward_that_did_not_fill_the_window():
    rows = _clean()
    rows[1]["long_forward"]["input_tokens"] = 4000
    assert any("did not fill the window" in w for w in _guard(rows))


def test_guard_refuses_unexpected_failed_loads_but_not_declared_ones():
    f = [{"run": "t", "arm": "e10", "rc": 1, "round": 2}]
    assert any("unexpected failed load" in w for w in _guard(_clean(), failed=f))
    assert any("t/p1: unexpected" in w for w in _guard(_clean(), failed=[f[0] | {"arm": "p1"}]))
    assert _guard(_clean(), failed=f, expected_failures=["t/e10"]) == []


def test_guard_refuses_non_finite_timings_and_missing_metrics():
    rows = _clean()
    rows[1]["phases_s"]["dec_load_total"] = float("nan")
    assert any("load_s non-finite" in w for w in _guard(rows))
    rows = _clean()
    del rows[3]["phases_s"]["model_ready"]
    assert any("model_ready_s missing" in w for w in _guard(rows))


def test_guard_refuses_non_finite_answers_p1_does_not_return():
    rows = _clean()
    for d in rows:
        if d["_arm"] == "e10":
            d["answers"] = {"a": {"q": {"noul": float("nan"), "type": "noul"}}}
    why = _guard(rows)
    assert any("non-finite answers P1 does not return" in w for w in why)


def test_guard_refuses_undeclared_control_nan_and_allows_declared():
    rows = _clean()
    for d in rows:
        d["answers"] = {"a": {"q": {"noul": float("nan"), "type": "noul"}}}
    assert any("P1 returns undeclared non-finite" in w for w in _guard(rows))
    assert _guard(rows, control_nan={"0xd40": ["a/q/noul"]}) == []


def test_guard_refuses_repeat_integrity_failures():
    rows = _clean()
    rows[1]["answers_repeat_identical"] = False
    assert any("repeats within a load" in w for w in _guard(rows))
    rows = _clean()
    rows[1]["repeat_compare"] = None  # an old probe compared by object identity: not evidence
    assert any("repeats within a load" in w for w in _guard(rows))


def test_guard_refuses_reference_mismatch():
    rows = _clean()
    rows[1]["vs_reference"] = {"a": "DIFFERS"}
    assert any("published reference" in w for w in _guard(rows))


def test_guard_refuses_an_order_unbalanced_against_p1():
    rows = _clean()
    for d in rows:
        if d["_arm"] == "e10":
            d["_pos"] = 0  # always before p1
    assert any("order unbalanced" in w for w in _guard(rows))
    rows = _clean()
    rows[1]["_pos"] = None
    assert any("order unbalanced" in w for w in _guard(rows))


def test_expected_refusals_must_refuse_every_round():
    s = {"failed_loads": [{"run": "x", "arm": "e9", "rc": 1}] * 5, "rows_by_run_arm": {}}
    assert summ.refusals_hold(s, ["x/e9"], 6)
    s["failed_loads"].append({"run": "x", "arm": "e9", "rc": 1})
    assert summ.refusals_hold(s, ["x/e9"], 6) == []
    s["rows_by_run_arm"] = {"x/e9": 1}
    assert summ.refusals_hold(s, ["x/e9"], 6)


def test_published_status_must_follow_the_primary_cell():
    s = summ.summarize(_clean())
    assert summ.status_holds("e10", s, {"status": "improved"}, "0xd40") == []
    assert summ.status_holds("e10", s, {"status": "no-change"}, "0xd40")
    assert summ.status_holds("e5", s, {"status": "improved"}, "0xd40")


def test_shipped_verdicts_pass_the_guard_on_the_committed_runs():
    vs = json.load(open(os.path.join(G, "verdicts.json")))
    s = json.load(open(os.path.join(G, "out", "ladder.json")))
    assert vs["min_rounds"] >= PRODUCTION_ROUNDS
    expected = [x for v in vs["rungs"].values() for x in v.get("expected_refusals", [])]
    for r, v in vs["rungs"].items():
        assert (
            summ.guard(
                r,
                s,
                v["general"],
                required=v.get("required_cells", vs["required_cells"]),
                min_rounds=vs["min_rounds"],
                expected_failures=expected,
                control_nan=vs.get("control_nan_fields", {}),
            )
            == []
        ), r
        assert summ.refusals_hold(s, v.get("expected_refusals", []), vs["min_rounds"]) == [], r
        assert summ.status_holds(r, s, v, vs["primary_identity"]) == [], r


# ------------------------------------------------------------------------------------------------- E6 bytecode
SH = os.path.join(G, "x86_e6.sh")


def _pkg(tmp_path, bad=False):
    src = tmp_path / "pkg"
    (src / "m").mkdir(parents=True)
    (src / "m" / "ok.py").write_text("X = 1\n")
    if bad:
        (src / "m" / "broken.py").write_text("def (:\n")
    return src


def _x86_e6(src, out):
    import subprocess

    env = dict(os.environ, X86_E6_PYTHON=sys.executable)
    return subprocess.run(["bash", SH, str(src), str(out)], capture_output=True, text=True, env=env)


def test_x86_e6_compiles_everything_and_writes_a_manifest(tmp_path):
    p = _x86_e6(_pkg(tmp_path), tmp_path / "e6")
    assert p.returncode == 0 and "X86_E6_DONE" in p.stdout, p.stderr
    m = json.load(open(str(tmp_path / "e6") + ".manifest.json"))
    assert m["compiled"] == 1 and m["failed"] == [] and len(m["tree_sha256"]) == 64
    assert any(f.endswith(".pyc") for _, _, fs in os.walk(tmp_path / "e6") for f in fs)


def test_x86_e6_fails_when_one_file_does_not_compile(tmp_path):
    p = _x86_e6(_pkg(tmp_path, bad=True), tmp_path / "e6")
    assert p.returncode != 0 and "X86_E6_DONE" not in p.stdout
    m = json.load(open(str(tmp_path / "e6") + ".manifest.json"))
    assert m["failed"] == ["m/broken.py"] and m["compiled"] == 1


def test_stage_refuses_incomplete_arm64_bytecode():
    text = open(os.path.join(G, "stage.sh")).read()
    i = text.index("pyc_zip.py --p1-zip")
    assert "FATAL: E6 bytecode incomplete" in text[i : i + 600]


def test_committed_x86_e6_package_compiled_without_failures():
    runs = os.path.join(G, "runs")
    x86 = [r for r in os.listdir(runs) if r.endswith("-x86")]
    assert x86
    for (
        r
    ) in x86:  # every x86 run that measured e6 carries its package's zero-failure compile manifest
        arms = json.load(open(os.path.join(runs, r, "arms.json")))["arms"]
        if not any(a["name"] == "e6" for a in arms):
            continue
        p = os.path.join(runs, r, "pkg-e6.manifest.json")
        m = json.load(open(p))
        assert m["failed"] == [] and m["compiled"] > 1000, p
        assert m["compiler"]["python"].startswith("3.12"), p


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
