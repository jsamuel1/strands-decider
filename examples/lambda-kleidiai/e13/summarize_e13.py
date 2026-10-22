"""E13 summary: one lkai-result/1 file per experiment arm (passive, spin30k, active), each against ctl from the same run.

  python summarize_e13.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <P1 sha> --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

Writes <dir>/<arm>/results.json for every experiment arm, and the merged run (invocations.jsonl, logs/,
conditions.json) once in <dir>. P1's summarize.py computes every distribution (phases, variance, cost, the
reference comparison). E13 adds, per arm against ctl:

- `e13.warm_paired`: the primary metric. Warm REPORT Duration, paired by (round, request id) within one CPU part, as
  P3's paired_warm_by_request: per cell median(arm) / median(ctl) - 1 in percent. The round is the unit (a round's
  cells share its environments), so the test is an exact two-sided sign-flip on the per-round means. The same
  pairing is applied to the engine's own latency_ms and to the process CPU seconds per invocation (rusage), which is
  the mechanism: spinning burns CPU, sleeping does not.
- `e13.delta`: cold and restore side effects, arm minus ctl median within each CPU part with a bootstrap 95% CI over
  environments (E6's method).
- `e13.arm_agreement`: every arm answer against ctl's for the same request and CPU part.
- `e13.spincount`: the GOMP_SPINCOUNT libgomp printed (OMP_DISPLAY_ENV=VERBOSE) per arm, from the saved logs.
- `host_variability`: CPU identity per arm and start class, unknown fields, and the primary metric per shared
  (round, CPU part) cell with its control spread.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import random
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import summarize as p1  # noqa: E402

CONTROL = "ctl"
EXPERIMENTS = ("passive", "spin30k", "active")
UNKNOWN_CPU = (None, "", "unknown", "?")


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _cpu(r: dict, k: str) -> float | None:
    inv = (r.get("lkai") or {}).get("e13_inv") or {}
    return inv.get(k)


def _cpu_s(r: dict) -> float | None:
    inv = (r.get("lkai") or {}).get("e13_inv") or {}
    if "utime_s" not in inv:
        return None
    return inv["utime_s"] + inv["stime_s"]


COLD_KEYS = {
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_init_duration": lambda r: _rep(r, "Init Duration"),
    "report_duration": lambda r: _rep(r, "Duration"),
    "model_load_mmap": lambda r: _init(r, "model_load_mmap"),
    "context_init": lambda r: _init(r, "context_init"),
    "weight_download": lambda r: _init(r, "weight_download"),
    "first_request_inference": lambda r: (r.get("latency_ms") or 0) / 1000 or None,
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
RESTORE_KEYS = {
    "first_request_total": lambda r: (
        (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
    "first_request_inference": lambda r: (r.get("latency_ms") or 0) / 1000 or None,
}
WARM_KEYS = {
    "report_duration": lambda r: _rep(r, "Duration"),
    "steady_state_inference": lambda r: (r.get("latency_ms") or 0) / 1000 or None,
    "process_cpu_s": _cpu_s,
    "voluntary_switches": lambda r: _cpu(r, "nvcsw"),
}


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def known_cpu(r: dict) -> bool:
    return r.get("cpu_part") not in UNKNOWN_CPU


def sign_flip_p(d: list[float]) -> float | None:
    """Exact two-sided paired sign-flip p-value of |mean(d)| over all 2^n sign assignments (P3's test). The smallest
    attainable value is 2 / 2^n. None below 2 pairs."""
    if len(d) < 2:
        return None
    obs = abs(statistics.fmean(d)) - 1e-12
    flips = list(itertools.product((1, -1), repeat=len(d)))
    hits = sum(
        abs(statistics.fmean([s * x for s, x in zip(sg, d, strict=True)])) >= obs for sg in flips
    )
    return round(hits / len(flips), 5)


def paired_by_request(rows: list[dict], arm: str, f, cpu: str) -> dict:
    """Warm only, one CPU part: per (round, request id) median(arm) / median(ctl) - 1 in percent, then per-round means
    (the unit) and an exact sign-flip test over rounds. P3's paired_warm_by_request, generalised to any metric."""
    per: dict[tuple, list[float]] = defaultdict(list)
    for r in ok_rows(rows, "warm"):
        if r["arm"] in (arm, CONTROL) and str(r.get("cpu_part")) == cpu:
            v = f(r)
            if v is not None:
                per[(r["arm"], r["round"], r["request_id"])].append(v)
    cells = sorted({k[1:] for k in per})
    pc = []
    for c in cells:
        a, b = per.get((arm, *c)), per.get((CONTROL, *c))
        if a and b and statistics.median(b):
            pc.append((c, 100 * (statistics.median(a) / statistics.median(b) - 1)))
    if not pc:
        return {}
    by_round: dict = defaultdict(list)
    for (rnd, _rid), x in pc:
        by_round[rnd].append(x)
    rounds = sorted(by_round)
    pr = [statistics.fmean(by_round[k]) for k in rounds]
    pct = [x for _, x in pc]
    return {
        "cells": len(pct),
        "median_cell_pct": round(statistics.median(pct), 3),
        "min_cell_pct": round(min(pct), 3),
        "max_cell_pct": round(max(pct), 3),
        "round_ids": rounds,
        "pct_by_round": [round(x, 3) for x in pr],
        "mean_round_pct": round(statistics.fmean(pr), 3),
        "rounds_below_zero": sum(x < 0 for x in pr),
        "rounds_above_zero": sum(x > 0 for x in pr),
        "sign_flip_p": sign_flip_p(pr),
        "min_attainable_p": round(2 / 2 ** len(pr), 5) if len(pr) >= 2 else None,
        "pct_by_cell": [{"round": c[0], "request_id": c[1], "pct": round(x, 3)} for c, x in pc],
    }


def control_null(rows: list[dict], f, cpu: str, seed: int = 13) -> dict:
    """The noise floor of the paired warm metric inside the control arm: each ctl (round, request) cell's
    observations are split at random into two halves and compared as if they were two arms. A per-round mean beyond
    this spread is not explained by within-cell variation."""
    rnd = random.Random(seed)
    per: dict[tuple, list[float]] = defaultdict(list)
    for r in ok_rows(rows, "warm"):
        if r["arm"] == CONTROL and str(r.get("cpu_part")) == cpu:
            v = f(r)
            if v is not None:
                per[(r["round"], r["request_id"])].append(v)
    by_round: dict = defaultdict(list)
    for (rnd_id, _rid), vs in sorted(per.items()):
        if len(vs) < 4:
            continue
        vs = vs[:]
        rnd.shuffle(vs)
        h = len(vs) // 2
        a, b = statistics.median(vs[:h]), statistics.median(vs[h : 2 * h])
        if b:
            by_round[rnd_id].append(100 * (a / b - 1))
    pr = [statistics.fmean(v) for _, v in sorted(by_round.items())]
    return {
        "pct_by_round": [round(x, 3) for x in pr],
        "max_abs_round_pct": round(max(abs(x) for x in pr), 3) if pr else None,
        "method": "ctl cells split at random into halves (seed 13), same pairing and round mean as the arms",
    }


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 13) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling environments independently per arm."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def delta(rows: list[dict], arm: str, cls: str, keys: dict) -> dict:
    """arm minus ctl median per metric within each known CPU part, bootstrap CI over environments."""
    rs = [r for r in ok_rows(rows, cls) if known_cpu(r)]
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted({str(r["cpu_part"]) for r in rs}):
            a = [
                v
                for r in rs
                if r["arm"] == CONTROL and str(r["cpu_part"]) == cpu
                if (v := f(r)) is not None
            ]
            b = [
                v
                for r in rs
                if r["arm"] == arm and str(r["cpu_part"]) == cpu
                if (v := f(r)) is not None
            ]
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[cpu] = {
                "ctl_p50": round(ma, 6),
                "arm_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        if per:
            out[name] = {"by_cpu_part": per}
    return out


def _cell_outcome(delta: float, control_spread: float | None) -> str:
    if control_spread is None:
        return "unassessed"
    if abs(delta) <= control_spread:
        return "in-band"
    return "faster" if delta < 0 else "slower"


def _aggregate_cells(cells: list[dict]) -> dict:
    """Fail-safe outcome from homogeneous cells; sparse opposite signs remain visible conflicts."""
    counts = Counter(c["outcome"] for c in cells)
    directional = {x for x in ("faster", "slower") if counts[x]}
    sparse_signs = {c["sign"] for c in cells if c["outcome"] == "unassessed"}
    opposite_sparse = ("faster" in directional and "positive" in sparse_signs) or (
        "slower" in directional and "negative" in sparse_signs
    )
    if len(directional) > 1 or opposite_sparse:
        outcome = "host-conditioned"
    elif directional and counts["in-band"]:
        outcome = "mixed"
    elif directional:
        outcome = next(iter(directional))
    elif counts["in-band"]:
        outcome = "null"
    else:
        outcome = "unknown"
    return {
        "outcome": outcome,
        "comparable": outcome in ("faster", "slower", "null"),
        "counts": dict(sorted(counts.items())),
        "unassessed_signs": sorted(sparse_signs),
        "unassessed_conflict": opposite_sparse,
    }


def round_cells(rows: list[dict], arm: str, cls: str, f) -> dict:
    """Metrics by homogeneous (round, CPU part, request) cell, with control spread as the margin."""
    rs = [r for r in ok_rows(rows, cls) if known_cpu(r)]
    keys = sorted({(r["round"], str(r["cpu_part"]), r["request_id"]) for r in rs})
    shared, missing = [], []
    by_part: dict = defaultdict(list)
    for rnd, cpu, request_id in keys:
        v = {
            a: [
                x
                for r in rs
                if r["arm"] == a
                and r["round"] == rnd
                and str(r["cpu_part"]) == cpu
                and r["request_id"] == request_id
                if (x := f(r)) is not None
            ]
            for a in (CONTROL, arm)
        }
        key = {"round": rnd, "cpu_part": cpu, "request_id": request_id}
        if not all(v.values()):
            missing.append(key | {"n": {a: len(v[a]) for a in v}})
            continue
        mc, ma = statistics.median(v[CONTROL]), statistics.median(v[arm])
        spread = max(v[CONTROL]) - min(v[CONTROL]) if len(v[CONTROL]) > 1 else None
        delta = ma - mc
        cell = key | {
            "n": {a: len(v[a]) for a in v},
            "median": {CONTROL: round(mc, 6), arm: round(ma, 6)},
            "control_within_cell_spread": round(spread, 6) if spread is not None else None,
            "delta": round(delta, 6),
            "ratio": round(ma / mc, 6) if mc else None,
            "sign": "negative" if delta < 0 else "positive" if delta > 0 else "zero",
            "outcome": _cell_outcome(delta, spread),
        }
        shared.append(cell)
        by_part[cpu].append(cell)
    by_cpu = {
        cpu: {
            "cells": len(cs),
            "delta_range": [min(c["delta"] for c in cs), max(c["delta"] for c in cs)],
            "delta_signs": sorted({c["sign"] for c in cs}),
            **_aggregate_cells(cs),
        }
        for cpu, cs in sorted(by_part.items())
    }
    assessed = [v for v in by_cpu.values() if v["outcome"] != "unknown"]
    overall = _aggregate_cells(shared)
    # Part-level mixed/null/conditioned states cannot be collapsed back into a directional headline.
    part_outcomes = {v["outcome"] for v in assessed}
    if "host-conditioned" in part_outcomes or {"faster", "slower"} <= part_outcomes:
        overall["outcome"], overall["comparable"] = "host-conditioned", False
    elif "mixed" in part_outcomes or (
        part_outcomes & {"faster", "slower"} and "null" in part_outcomes
    ):
        overall["outcome"], overall["comparable"] = "mixed", False
    elif part_outcomes == {"null"}:
        overall["outcome"], overall["comparable"] = "null", True
    elif len(part_outcomes) == 1 and part_outcomes <= {"faster", "slower"}:
        overall["outcome"], overall["comparable"] = next(iter(part_outcomes)), True
    else:
        overall["outcome"], overall["comparable"] = "unknown", False
    return {
        "class": cls,
        "cell_key": ["round", "cpu_part", "request_id"],
        "shared_cells": shared,
        "missing_cells": missing,
        "by_cpu_part": by_cpu,
        "assessment": overall,
    }


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm and start class (counts), as e13_entry read it on each environment's first
    invocation. Fields the sandbox does not expose stay 'unknown'."""
    ids: dict = defaultdict(Counter)
    for r in rows:
        e = (r.get("lkai") or {}).get("e13")
        if not e or r["error"]:
            continue
        c = e.get("cpu", {})
        key = json.dumps(
            {
                "implementer": c.get("implementer"),
                "part": c.get("part"),
                "variant": c.get("variant"),
                "revision": c.get("revision"),
                "processors": c.get("processors"),
                "features": c.get("features"),
                "hwcap": e.get("hwcap"),
                "hwcap2": e.get("hwcap2"),
                "os_cpu_count": e.get("os_cpu_count"),
                "sched_affinity": e.get("sched_affinity"),
                # /proc/self/cgroup lists the same v1 controllers in a per-environment order: kept out of the key
                # (it would split identical hosts) and summarised below.
                "cgroup": {
                    k: v for k, v in (e.get("cgroup") or {}).items() if k != "/proc/self/cgroup"
                },
                "kernel": e.get("kernel"),
                "python": e.get("python"),
                "execution_env": e.get("execution_env"),
                "ggml_cpu_mapped": e.get("ggml_cpu_mapped"),
                "libgomp_mapped": e.get("libgomp_mapped"),
                "threads_after_first_invoke": e.get("threads_after_first_invoke"),
            },
            sort_keys=True,
        )
        ids[f"{r['arm']}:{r['expected_class']}"][key] += 1
    omp = defaultdict(Counter)
    for r in rows:
        e = (r.get("lkai") or {}).get("e13")
        if e and not r["error"]:
            omp[r["arm"]][json.dumps(e.get("omp_env"), sort_keys=True)] += 1
    controllers: Counter = Counter()
    for r in rows:
        e = (r.get("lkai") or {}).get("e13")
        if e and not r["error"]:
            text = (e.get("cgroup") or {}).get("/proc/self/cgroup", "unknown")
            names = sorted(
                ln.split(":", 2)[1] + ":" + ln.split(":", 2)[2]
                for ln in text.splitlines()
                if ln.count(":") >= 2
            )
            controllers[json.dumps(names) if names else "unknown"] += 1
    return {
        "proc_self_cgroup": [
            {"environments": n, "entries": json.loads(k) if k != "unknown" else k}
            for k, n in controllers.most_common()
        ],
        "identities": {
            k: [{"environments": n, **json.loads(s)} for s, n in cnt.most_common()]
            for k, cnt in sorted(ids.items())
        },
        "omp_env_by_arm": {
            arm: [{"environments": n, **json.loads(s)} for s, n in cnt.most_common()]
            for arm, cnt in sorted(omp.items())
        },
        "note": "Read inside the function from /proc/cpuinfo, /proc/self/auxv, os.sched_getaffinity, /proc/self/maps "
        "and the cgroup files. 'unknown' = not exposed in the sandbox. CPU part and features are reported as read; no "
        "CPU generation is inferred from them or from timings.",
    }


_SPIN = re.compile(r"GOMP_SPINCOUNT = '(\d+)'")
_WAIT = re.compile(r"OMP_WAIT_POLICY = '(\w+)'")


def spincount(run: str, cond: dict) -> dict:
    """The effective GOMP_SPINCOUNT / OMP_WAIT_POLICY libgomp printed at start-up (OMP_DISPLAY_ENV=VERBOSE), per arm."""
    out = {}
    for arm, f in cond["functions"].items():
        path = os.path.join(run, "logs", f"{f['FunctionName']}.jsonl.gz")
        spins, waits = Counter(), Counter()
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    msg = json.loads(line)["message"]
                    if m := _SPIN.search(msg):
                        spins[m.group(1)] += 1
                    if m := _WAIT.search(msg):
                        waits[m.group(1)] += 1
        out[arm] = {"gomp_spincount": dict(spins), "omp_wait_policy": dict(waits)}
    return out


def arm_agreement(rows: list[dict], arm: str) -> dict:
    """arm's answers against ctl's from the same run, for the same request and CPU part."""
    ref: dict = {}
    ctl_self = 0
    for r in rows:
        if r["arm"] == CONTROL and not r["error"]:
            k = (r["request_id"], r.get("cpu_part"))
            if k in ref and ref[k] != r["answers"]:
                ctl_self += 1
            ref.setdefault(k, r["answers"])
    n = same = missing = 0
    for r in rows:
        if r["arm"] != arm or r["error"]:
            continue
        base = ref.get((r["request_id"], r.get("cpu_part")))
        if base is None:
            missing += 1
            continue
        n += 1
        same += r["answers"] == base
    return {
        "compared": n,
        "bit_identical": same,
        "no_ctl_answer_for_cpu_part": missing,
        "ctl_internal_disagreements": ctl_self,
        "method": f"complete answer object of every successful {arm} invocation vs the ctl answer for the same "
        "request id and CPU part in this run",
    }


def merge_runs(cold: str, restore: str | None, out_dir: str) -> str:
    """One run directory holding both legs' invocations, logs and conditions (cold first). Restore rounds become
    100 + round so per-round pairing stays within a leg."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    legs = [("cold", cold)] + ([("restore", restore)] if restore else [])
    with open(os.path.join(out_dir, "invocations.jsonl"), "w") as w:
        for leg, d in legs:
            for line in open(os.path.join(d, "invocations.jsonl")):
                r = json.loads(line)
                r["leg"] = leg
                if leg == "restore":
                    r["round"] = 100 + r["round"]
                w.write(json.dumps(r, separators=(",", ":")) + "\n")
    c = json.load(open(os.path.join(cold, "conditions.json")))
    if restore:
        rc = json.load(open(os.path.join(restore, "conditions.json")))
        c["legs"] = {
            "cold": {
                k: c.get(k)
                for k in ("rounds", "warm_reps", "started", "finished", "read_timeout_s")
            },
            "restore": {
                k: rc.get(k)
                for k in ("rounds", "warm_reps", "started", "finished", "read_timeout_s")
            },
        }
        c["rounds_meta"] = c["rounds_meta"] + [
            m | {"round": 100 + m["round"]} for m in rc["rounds_meta"]
        ]
        c["finished"] = rc.get("finished")
        c["read_timeout_s"] = {"cold": c.get("read_timeout_s"), "restore": rc.get("read_timeout_s")}
    json.dump(c, open(os.path.join(out_dir, "conditions.json"), "w"), indent=1)
    for _leg, d in legs:
        for f in sorted(os.listdir(os.path.join(d, "logs"))):
            with (
                open(os.path.join(d, "logs", f), "rb") as src,
                open(os.path.join(out_dir, "logs", f), "ab") as dst,
            ):
                shutil.copyfileobj(src, dst)  # concatenated gzip members read back as one stream
    return out_dir


def warm_verdict(wp: dict, null: dict, alpha: float = 0.05) -> str:
    """One CPU part's paired warm REPORT Duration: faster / slower when every round has the same sign, the sign-flip
    p is below alpha and the mean exceeds the control's own split-half noise. `underpowered` when the part has too
    few rounds for any p below alpha (2 / 2^n >= alpha); its round signs are still reported. no-change otherwise."""
    if not wp or len(wp["pct_by_round"]) < 2:
        return "insufficient"
    if wp["min_attainable_p"] >= alpha:
        return "underpowered"
    floor = null.get("max_abs_round_pct") or 0.0
    p = wp["sign_flip_p"]
    m = wp["mean_round_pct"]
    if p is not None and p < alpha and abs(m) > floor:
        if wp["rounds_below_zero"] == len(wp["pct_by_round"]):
            return "faster"
        if wp["rounds_above_zero"] == len(wp["pct_by_round"]):
            return "slower"
    return "no-change"


def report_request_ids(run: str, cond: dict) -> dict[str, set[str]]:
    """Request ids of every REPORT line in each arm's saved logs."""
    out = {}
    for arm, f in cond["functions"].items():
        path = os.path.join(run, "logs", f"{f['FunctionName']}.jsonl.gz")
        ids: set[str] = set()
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    msg = json.loads(line)["message"]
                    if msg.startswith("REPORT") and (m := re.search(r"RequestId: (\S+)", msg)):
                        ids.add(m.group(1))
        out[arm] = ids
    return out


def error_kinds(errors: list[dict], rows: list[dict], report_ids: dict[str, set[str]]) -> dict:
    """Classify each error row. `client-transport`: the client failed (client:*) and the arm's saved logs hold no
    REPORT line that a successful row does not account for, so Lambda never ran the request. Anything else is
    `function`. Only client-transport errors leave the status decidable; every error is still published."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not r["error"]:
            m = re.search(r"RequestId: (\S+)", r.get("log_tail") or "")
            if m:
                seen[r["arm"]].add(m.group(1))
    out = []
    for e in errors:
        unmatched = report_ids.get(e["arm"], set()) - seen[e["arm"]]
        transport = str(e["error"]).startswith("client:") and not unmatched
        out.append(
            e
            | {
                "kind": "client-transport" if transport else "function",
                "unmatched_report_lines": len(unmatched),
            }
        )
    return {
        "rows": out,
        "function_errors": sum(x["kind"] == "function" for x in out),
        "client_transport_errors": sum(x["kind"] == "client-transport" for x in out),
        "method": "a client:* error is client-transport when the arm's saved logs hold no REPORT line that a "
        "successful row does not account for: Lambda never ran it, so it has no answer and no latency",
    }


def status_of(arm_res: dict, function_errors: int, correctness: dict | None) -> str:
    """The deployment outcome for one arm, judged on the primary metric (paired warm REPORT Duration) per CPU part.

    improved: every powered part (enough rounds for p < 0.05) is faster, no underpowered part has a round of the
    opposite sign, and neither the cold nor the restore first-request CI lies entirely above 0 on any part.
    regressed: any powered part is slower, or a cold/restore first-request CI lies entirely above 0. This is a
    conservative do-not-use result even when sparse host cells are mixed; that limits effect-size generalization.
    inconclusive: a function error, an answer that differs, no powered part, or a claimed speedup whose homogeneous
    host cells are not all directional. no-change otherwise."""
    g = arm_res["arm_agreement"]
    c = correctness or {}
    clean = (
        function_errors == 0
        and c.get("bit_identical") == c.get("compared")
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["ctl_internal_disagreements"] == 0
    )
    wp = arm_res["warm_paired"]["report_duration"]
    powered = {
        cpu: v["verdict"]
        for cpu, v in wp.items()
        if v["verdict"] in ("faster", "slower", "no-change")
    }
    if not clean or not powered:
        return "inconclusive"
    side = [
        v
        for cls in ("cold", "restore")
        for v in arm_res["delta"][cls]
        .get("first_request_total", {})
        .get("by_cpu_part", {})
        .values()
    ]
    host_outcome = arm_res["round_cells"]["warm_report_duration"]["assessment"]["outcome"]
    if "slower" in powered.values() or any(v["ci95"][0] > 0 for v in side):
        return "regressed"
    if set(powered.values()) == {"faster"}:
        contrary = [
            cpu
            for cpu, v in wp.items()
            if v["verdict"] in ("underpowered", "insufficient") and v["rounds_above_zero"] > 0
        ]
        if contrary:
            return "no-change"
        return "improved" if host_outcome == "faster" else "inconclusive"
    return "no-change"


def analyse(rows: list[dict], arm: str) -> dict:
    """Everything E13 adds for one experiment arm against ctl."""
    warm = [r for r in ok_rows(rows, "warm") if known_cpu(r)]
    parts = sorted(
        {str(r["cpu_part"]) for r in warm if r["arm"] == arm}
        & {str(r["cpu_part"]) for r in warm if r["arm"] == CONTROL}
    )
    wp: dict = {}
    for name, f in WARM_KEYS.items():
        wp[name] = {}
        for cpu in parts:
            res = paired_by_request(rows, arm, f, cpu)
            if not res:
                continue
            if name == "report_duration":
                res["control_split_half"] = control_null(rows, f, cpu)
                res["verdict"] = warm_verdict(res, res["control_split_half"])
            wp[name][cpu] = res
    return {
        "warm_paired": wp,
        "delta": {
            "cold": delta(rows, arm, "cold", COLD_KEYS),
            "restore": delta(rows, arm, "restore", RESTORE_KEYS),
            "warm": delta(rows, arm, "warm", WARM_KEYS),
        },
        "round_cells": {
            "warm_report_duration": round_cells(rows, arm, "warm", WARM_KEYS["report_duration"]),
            "cold_first_request_total": round_cells(
                rows, arm, "cold", COLD_KEYS["first_request_total"]
            ),
            "restore_first_request_total": round_cells(
                rows, arm, "restore", RESTORE_KEYS["first_request_total"]
            ),
        },
        "arm_agreement": arm_agreement(rows, arm),
    }


def host_variability(rows: list[dict], per_arm: dict) -> dict:
    counts: dict = defaultdict(Counter)
    for r in rows:
        counts[f"{r['arm']}:{r['expected_class']}"][str(r.get("cpu_part"))] += 1
    unknown = {
        k: sum(n for cpu, n in c.items() if cpu in ("None", "", "unknown", "?"))
        for k, c in sorted(counts.items())
    }
    return {
        "schema": "lkai-host-variability/1",
        "observed": {
            "cpu_part_rows_by_arm_and_class": {
                k: dict(sorted(c.items())) for k, c in sorted(counts.items())
            },
            "primary_metric_cells": {
                arm: a["round_cells"]["warm_report_duration"] for arm, a in per_arm.items()
            },
        },
        "unknown": {
            "rows_without_cpu_part": unknown,
            "environment_fields": ["availability zone", "physical host", "CPU frequency"],
            "cgroup": "per environment in e13.host.identities; any file the sandbox hides is 'unknown'",
        },
        "method": "Every comparison is made within one observed CPU part. Warm (the primary metric): within each "
        "round the arms alternate pass by pass (the arm order rotates every pass), so the arms are interleaved "
        "within the round, and each (round, request) cell is compared on the same CPU part. Cold and restore waves "
        "are sequential arm waves whose position rotates across rounds (a Latin square over 4 rounds): that controls "
        "order and round drift but not time-within-round drift. Rows without a CPU part are counted, never compared.",
        "assessment": {
            arm: {
                "overall": a["round_cells"]["warm_report_duration"]["assessment"],
                "by_cpu_part": a["round_cells"]["warm_report_duration"]["by_cpu_part"],
            }
            for arm, a in per_arm.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold")
    ap.add_argument("--restore")
    ap.add_argument("--run-id")
    ap.add_argument("--baseline-commit")
    ap.add_argument("--reference")
    ap.add_argument("--meta")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--host-only",
        metavar="RUN_DIR",
        help="recompute host/cell analysis and conclusions in an existing committed run; raw measurements stay unchanged",
    )
    ap.add_argument(
        "--stage",
        metavar="RUN_DIR",
        help="only copy a committed run into per-arm publish directories in --out",
    )
    a = ap.parse_args(argv)
    if a.host_only:
        amend_host_analysis(a.host_only, a.meta or os.path.join(a.host_only, "meta.json"))
        return 0
    if a.stage:
        for p in stage_for_publish(a.stage, a.out):
            print("staged", p)
        return 0
    for k in ("cold", "run_id", "baseline_commit", "reference", "meta"):
        if not getattr(a, k):
            ap.error(f"--{k.replace('_', '-')} is required")
    run = merge_runs(a.cold, a.restore, a.out)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = dict.fromkeys((CONTROL, *EXPERIMENTS), "1")  # every E13 arm is instrumented
    cond, rows = p1.load_run(run)
    per_arm = {arm: analyse(rows, arm) for arm in EXPERIMENTS if arm in cond["arms"]}
    spins = spincount(run, cond)
    hv = host_variability(rows, per_arm)
    hst = host(rows)
    rids = report_request_ids(run, cond)
    for arm, ana in per_arm.items():
        m = meta["arms"][arm]
        res = p1.summarize(run, a.reference, a.baseline_commit, meta | m)
        errs = error_kinds(res["errors"]["rows"], rows, rids)
        res["errors"]["rows"] = errs["rows"]
        ana = ana | {"error_kinds": {k: v for k, v in errs.items() if k != "rows"}}
        res["experiment_id"] = meta["experiment_id"]
        res["run_id"] = f"{a.run_id}-{arm}"
        res["primary_arm"] = arm
        res["commits"]["experiment"] = cond["build_manifest"]["repo_commit"]
        res["build"] = {
            k: cond["build_manifest"][k]
            for k in ("images", "toolchain", "assets_sha256", "zip", "base", "wrapper_sha256")
        }
        res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
        res["conditions"]["legs"] = cond.get("legs")
        res["conditions"]["arm_variables"] = cond.get("arm_variables")
        res["instrumentation_overhead"] = {
            "measured": False,
            "note": "every E13 arm runs P1's phase timers and the same wrapper; overhead is P1's "
            "(P1-baseline/20261020T080835Z)",
        }
        res["host_variability"] = hv
        res["e13"] = ana | {
            "spincount": spins,
            "host": hst,
            "sibling_arms": sorted(per_arm),
            "vs_sibling_arms": {
                o: between_arms(per_arm, arm, o) for o in sorted(per_arm) if o != arm
            },
        }
        res["status"] = status_of(ana, errs["function_errors"], res["correctness"])
        res["e13"]["_warm_inv"] = {
            a2: {
                cpu: {
                    "nivcsw": [
                        r["lkai"]["e13_inv"]["nivcsw"]
                        for r in ok_rows(rows, "warm")
                        if r["arm"] == a2
                        and str(r.get("cpu_part")) == cpu
                        and "e13_inv" in (r.get("lkai") or {})
                    ]
                }
                for cpu in ("0xd40",)
            }
            for a2 in (CONTROL, arm)
        }
        tpl = m.get("conclusion_template") or meta.get("conclusion_template")
        res["conclusion"] = tpl.format(**fmt(res, arm)) if tpl else m.get("conclusion")
        res["reproduce"] = meta.get("reproduce", [])
        del res["e13"]["_warm_inv"]
        res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
        res["artifacts"] = {
            "invocations": "invocations.jsonl.gz",
            "logs": "logs/",
            "conditions": "conditions.json",
            "build_manifest": "build-manifest.json",
            "meta": "meta.json",
        }
        os.makedirs(os.path.join(a.out, arm), exist_ok=True)
        path = os.path.join(a.out, arm, "results.json")
        json.dump(res, open(path, "w"), indent=1, allow_nan=False)
        print("wrote", path, res["status"])
    return 0


def between_arms(per_arm: dict, a: str, b: str, cpu: str = "0xd40") -> dict | None:
    """a against b on the paired warm metric: both are paired with the same ctl cells, so per round
    (a - ctl) - (b - ctl) is their difference. Sign-flip test over the rounds both have."""
    try:
        ra = per_arm[a]["warm_paired"]["report_duration"][cpu]
        rb = per_arm[b]["warm_paired"]["report_duration"][cpu]
    except KeyError:
        return None
    pa = dict(zip(ra["round_ids"], ra["pct_by_round"], strict=True))
    pb = dict(zip(rb["round_ids"], rb["pct_by_round"], strict=True))
    rounds = sorted(set(pa) & set(pb))
    d = [pa[k] - pb[k] for k in rounds]
    if len(d) < 2:
        return None
    return {
        "arms": [a, b],
        "cpu_part": cpu,
        "rounds": rounds,
        "pp_by_round": [round(x, 3) for x in d],
        "mean_pp": round(statistics.fmean(d), 3),
        "rounds_below_zero": sum(x < 0 for x in d),
        "sign_flip_p": sign_flip_p(d),
        "method": "difference of the two arms' per-round paired percentages (same ctl cells), percentage points",
    }


def _parts_text(wp: dict, primary: str) -> str:
    """The other CPU parts' paired warm result in words: per-round signs and mean, and why it is not a verdict."""
    out = []
    for cpu, v in sorted(wp.items()):
        if cpu == primary:
            continue
        n = len(v["pct_by_round"])
        out.append(
            f"{cpu}: {v['mean_round_pct']:+.1f}% over {n} round(s) ({v['rounds_below_zero']} below 0, "
            f"{v['rounds_above_zero']} above), {v['verdict']}"
        )
    return "; ".join(out) or "none"


def _inv(res: dict, arm: str, k: str, cpu: str) -> list[float]:
    return res["e13"]["_warm_inv"][arm][cpu][k]


def fmt(res: dict, arm: str, cpu: str = "0xd40") -> dict:
    """Every number a conclusion template may cite, read from the result itself."""
    e = res["e13"]
    wp = e["warm_paired"]
    w = wp["report_duration"][cpu]
    dw, dc, dr = e["delta"]["warm"], e["delta"]["cold"], e["delta"]["restore"]
    c, g = res["correctness"], e["arm_agreement"]

    def d(block, k):
        return block[k]["by_cpu_part"][cpu]

    return {
        "cpu": cpu,
        "rounds": len(w["pct_by_round"]),
        "cells": w["cells"],
        "w_mean": w["mean_round_pct"],
        "w_lo": min(w["pct_by_round"]),
        "w_hi": max(w["pct_by_round"]),
        "w_p": w["sign_flip_p"],
        "w_null": w["control_split_half"]["max_abs_round_pct"],
        "w_ctl_ms": 1000 * d(dw, "report_duration")["ctl_p50"],
        "w_arm_ms": 1000 * d(dw, "report_duration")["arm_p50"],
        "eng_mean": wp["steady_state_inference"][cpu]["mean_round_pct"],
        "cpu_mean": wp["process_cpu_s"][cpu]["mean_round_pct"],
        "vcsw_ctl": d(dw, "voluntary_switches")["ctl_p50"],
        "vcsw_arm": d(dw, "voluntary_switches")["arm_p50"],
        "ivcsw_ctl": statistics.median(_inv(res, "ctl", "nivcsw", cpu)),
        "ivcsw_arm": statistics.median(_inv(res, arm, "nivcsw", cpu)),
        "error_where": ", ".join(
            f"{r['arm']} {r['wave']}, "
            + (
                f"restore leg round {r['round'] - 100}"
                if r["round"] >= 100
                else f"cold leg round {r['round']}"
            )
            for r in res["errors"]["rows"]
        )
        or "none",
        "restore_timeouts": sum(
            "ReadTimeout" in str(r["error"])
            for r in res["errors"]["rows"]
            if r["wave"] == "restore"
        ),
        "other_parts": _parts_text(wp["report_duration"], cpu),
        "cold_d": d(dc, "first_request_total")["delta_p50"],
        "cold_lo": d(dc, "first_request_total")["ci95"][0],
        "cold_hi": d(dc, "first_request_total")["ci95"][1],
        "cold_n": d(dc, "first_request_total")["n"],
        "cold_inf_ms": 1000 * d(dc, "first_request_inference")["delta_p50"],
        "rst_d": d(dr, "first_request_total")["delta_p50"],
        "rst_lo": d(dr, "first_request_total")["ci95"][0],
        "rst_hi": d(dr, "first_request_total")["ci95"][1],
        "rst_n": d(dr, "first_request_total")["n"],
        "rst_inf_ms": 1000 * d(dr, "first_request_inference")["delta_p50"],
        "rst_inf_lo": 1000 * d(dr, "first_request_inference")["ci95"][0],
        "rst_inf_hi": 1000 * d(dr, "first_request_inference")["ci95"][1],
        "mem_d": d(dc, "report_max_memory_mb")["delta_p50"],
        "agree": g["bit_identical"],
        "agree_n": g["compared"],
        "ref": c["bit_identical"],
        "ref_n": c["compared"],
        "transport": e["error_kinds"]["client_transport_errors"],
        "function_errors": e["error_kinds"]["function_errors"],
        "spin": ",".join(sorted(e["spincount"][arm]["gomp_spincount"])),
        "ctl_spin": ",".join(sorted(e["spincount"]["ctl"]["gomp_spincount"])),
        "host_outcome": e["round_cells"]["warm_report_duration"]["assessment"]["outcome"],
        "host_primary_counts": ", ".join(
            f"{k}={v}"
            for k, v in sorted(
                e["round_cells"]["warm_report_duration"]["by_cpu_part"][cpu]["counts"].items()
            )
        ),
        "cost": res["cost"]["total_usd"],
        **{
            f"vs_{o}_{k}": v[k]
            for o, v in (e.get("vs_sibling_arms") or {}).items()
            if v
            for k in ("mean_pp", "rounds_below_zero", "sign_flip_p")
        },
        **{
            f"vs_{o}_rounds": len(v["rounds"])
            for o, v in (e.get("vs_sibling_arms") or {}).items()
            if v
        },
    }


def amend_host_analysis(run_dir: str, meta_path: str) -> None:
    """Recompute only analysis derived from committed rows; preserve raw measurements and unrelated result fields."""
    inv = os.path.join(run_dir, "invocations.jsonl.gz")
    with gzip.open(inv, "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    meta = json.load(open(meta_path))
    per_arm = {arm: analyse(rows, arm) for arm in EXPERIMENTS}
    hv = host_variability(rows, per_arm)
    for arm, ana in per_arm.items():
        path = os.path.join(run_dir, arm, "results.json")
        res = json.load(open(path))
        res["e13"]["round_cells"] = ana["round_cells"]
        res["host_variability"] = hv
        function_errors = res["e13"]["error_kinds"]["function_errors"]
        res["status"] = status_of(ana, function_errors, res.get("correctness"))
        # Keep all measured values from the original result while deriving prose from the corrected host verdict.
        original = res["e13"]
        ana["spincount"] = original["spincount"]
        ana["error_kinds"] = original["error_kinds"]
        ana["_warm_inv"] = {
            a2: {
                "0xd40": {
                    "nivcsw": [
                        r["lkai"]["e13_inv"]["nivcsw"]
                        for r in ok_rows(rows, "warm")
                        if r["arm"] == a2
                        and str(r.get("cpu_part")) == "0xd40"
                        and "e13_inv" in (r.get("lkai") or {})
                    ]
                }
            }
            for a2 in (CONTROL, arm)
        }
        ana["vs_sibling_arms"] = {
            other: between_arms(per_arm, arm, other) for other in sorted(per_arm) if other != arm
        }
        res["e13"] = original | ana
        values = fmt(res, arm)
        res["conclusion"] = meta["arms"][arm]["conclusion_template"].format(**values)
        del res["e13"]["_warm_inv"]
        json.dump(res, open(path, "w"), indent=1, allow_nan=False)
        print("amended", path, res["status"], values["host_outcome"])


SHARED = ("invocations.jsonl.gz", "logs", "conditions.json", "build-manifest.json", "meta.json")


def stage_for_publish(run_dir: str, out: str) -> list[str]:
    """P2's publisher uploads artifacts relative to each results.json and refuses '..'. The committed run keeps one
    copy of the shared raw data beside <arm>/results.json; this copies it into one publishable directory per arm."""
    paths = []
    for arm in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, arm, "results.json")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out, arm)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        shutil.copy2(src, dst)
        for name in SHARED:
            s = os.path.join(run_dir, name)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(dst, name))
            else:
                shutil.copy2(s, dst)
        paths.append(os.path.join(dst, "results.json"))
    return paths


if __name__ == "__main__":
    sys.exit(main())
