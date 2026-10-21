"""E15 summary: one lkai-result/1 file per experiment arm (off, a), each against ctl from the same run.

  python summarize_e15.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <P1 sha> --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

Ported from E13's summarize_e13.py (lkai/a088-omp-wait 0dc2eed): same pairing, tests and host-variability cells.
Writes <dir>/<arm>/results.json for every experiment arm, and the merged run (invocations.jsonl, logs/,
conditions.json) once in <dir>. P1's summarize.py computes every distribution (phases, variance, cost, the
reference comparison). E15 adds, per arm against ctl (and for `a` also against `off`, which isolates the switch):

- `e15.warm_paired`: the primary metric. Warm REPORT Duration, paired by (round, request id) within one CPU part:
  per cell median(arm) / median(control) - 1 in percent; the round is the unit; exact two-sided sign-flip test.
- `e15.delta`: cold and restore side effects, arm minus control median within each CPU part, bootstrap 95% CI.
- `e15.arm_agreement`: every arm answer against the control's for the same request and CPU part, with the maximum
  |delta probability| (choice probabilities and noul) and choice flips, the operator's tolerance (TOLERANCE).
- `e15.applicability`: which ggml CPU variant each CPU part loaded. Candidate A changes only the SVE / i8mm variants;
  on any other part it is a no-op by construction, so those parts are a negative control (answers must be
  bit-identical) and are not counted for or against the verdict.
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
EXPERIMENTS = ("off", "a")
# The operator's tolerance (card evidence 2026-10-21): against P1's reference for the same CPU part, and here also
# against ctl in the same run.
TOLERANCE = {"max_abs_probability_delta": 0.02, "choice_flips": 0}
# The CPU variants candidate A changes (ggml-cpu.c undefines GGML_USE_LLAMAFILE when SVE or MATMUL_INT8 is set).
CHANGED_VARIANTS = {"armv8.2_3", "armv8.6_1", "armv8.6_2"}
UNKNOWN_CPU = (None, "", "unknown", "?")


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _cpu(r: dict, k: str) -> float | None:
    inv = (r.get("lkai") or {}).get("e15_inv") or {}
    return inv.get(k)


def _cpu_s(r: dict) -> float | None:
    inv = (r.get("lkai") or {}).get("e15_inv") or {}
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


def paired_by_request(rows: list[dict], arm: str, f, cpu: str, ctl: str = CONTROL) -> dict:
    """Warm only, one CPU part: per (round, request id) median(arm) / median(ctl) - 1 in percent, then per-round means
    (the unit) and an exact sign-flip test over rounds. P3's paired_warm_by_request, generalised to any metric."""
    per: dict[tuple, list[float]] = defaultdict(list)
    for r in ok_rows(rows, "warm"):
        if r["arm"] in (arm, ctl) and str(r.get("cpu_part")) == cpu:
            v = f(r)
            if v is not None:
                per[(r["arm"], r["round"], r["request_id"])].append(v)
    cells = sorted({k[1:] for k in per})
    pc = []
    for c in cells:
        a, b = per.get((arm, *c)), per.get((ctl, *c))
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


def control_null(rows: list[dict], f, cpu: str, seed: int = 13, ctl: str = CONTROL) -> dict:
    """The noise floor of the paired warm metric inside the control arm: each ctl (round, request) cell's
    observations are split at random into two halves and compared as if they were two arms. A per-round mean beyond
    this spread is not explained by within-cell variation."""
    rnd = random.Random(seed)
    per: dict[tuple, list[float]] = defaultdict(list)
    for r in ok_rows(rows, "warm"):
        if r["arm"] == ctl and str(r.get("cpu_part")) == cpu:
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


def delta(rows: list[dict], arm: str, cls: str, keys: dict, ctl: str = CONTROL) -> dict:
    """arm minus ctl median per metric within each known CPU part, bootstrap CI over environments."""
    rs = [r for r in ok_rows(rows, cls) if known_cpu(r)]
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted({str(r["cpu_part"]) for r in rs}):
            a = [
                v
                for r in rs
                if r["arm"] == ctl and str(r["cpu_part"]) == cpu
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


def round_cells(rows: list[dict], arm: str, cls: str, f, ctl: str = CONTROL) -> dict:
    """Primary metrics by shared (round, CPU part) cell, with the control's within-cell spread as the margin."""
    rs = [r for r in ok_rows(rows, cls) if known_cpu(r)]
    keys = sorted({(r["round"], str(r["cpu_part"])) for r in rs})
    shared, missing = [], []
    by_part: dict = defaultdict(list)
    for rnd, cpu in keys:
        v = {
            a: [
                x
                for r in rs
                if r["arm"] == a and r["round"] == rnd and str(r["cpu_part"]) == cpu
                if (x := f(r)) is not None
            ]
            for a in (ctl, arm)
        }
        if not all(v.values()):
            missing.append({"round": rnd, "cpu_part": cpu, "n": {a: len(v[a]) for a in v}})
            continue
        mc, ma = statistics.median(v[ctl]), statistics.median(v[arm])
        cell = {
            "round": rnd,
            "cpu_part": cpu,
            "n": {a: len(v[a]) for a in v},
            "median": {ctl: round(mc, 6), arm: round(ma, 6)},
            "control_spread": round(max(v[ctl]) - min(v[ctl]), 6) if len(v[ctl]) > 1 else None,
            "delta": round(ma - mc, 6),
            "ratio": round(ma / mc, 6) if mc else None,
            "sign": "negative" if ma < mc else "positive" if ma > mc else "zero",
        }
        shared.append(cell)
        by_part[cpu].append(cell)
    return {
        "class": cls,
        "control": ctl,
        "shared_cells": shared,
        "missing_cells": missing,
        "by_cpu_part": {
            cpu: {
                "cells": len(cs),
                "delta_range": [min(c["delta"] for c in cs), max(c["delta"] for c in cs)],
                "delta_signs": sorted({c["sign"] for c in cs}),
            }
            for cpu, cs in sorted(by_part.items())
        },
    }


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm and start class (counts), as e15_entry read it on each environment's first
    invocation. Fields the sandbox does not expose stay 'unknown'."""
    ids: dict = defaultdict(Counter)
    for r in rows:
        e = (r.get("lkai") or {}).get("e15")
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
        e = (r.get("lkai") or {}).get("e15")
        if e and not r["error"]:
            omp[r["arm"]][json.dumps(e.get("omp_env"), sort_keys=True)] += 1
    controllers: Counter = Counter()
    for r in rows:
        e = (r.get("lkai") or {}).get("e15")
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


def prob_deltas(ans: dict, base: dict) -> tuple[float, int, float]:
    """Max |delta| over choice probabilities and noul, choice flips, and max |delta| over score/confidence."""
    # P1's own answer flattening, so the paths compared are the ones P1 compares
    a, b = p1._flat(ans), p1._flat(base)  # noqa: SLF001
    worst = other = 0.0
    flips = 0
    for k in set(a) | set(b):
        v, w = b.get(k), a.get(k)
        if k.endswith(".choice"):
            flips += v != w
        elif v is None or w is None:
            continue
        elif ".p[" in k or k.endswith(".noul"):
            worst = max(worst, abs(float(w) - float(v)))
        else:
            other = max(other, abs(float(w) - float(v)))
    return worst, flips, other


def arm_agreement(rows: list[dict], arm: str, ctl: str = CONTROL) -> dict:
    """arm's answers against the control's from the same run, for the same request and CPU part."""
    ref: dict = {}
    ctl_self = 0
    for r in rows:
        if r["arm"] == ctl and not r["error"]:
            k = (r["request_id"], r.get("cpu_part"))
            if k in ref and ref[k] != r["answers"]:
                ctl_self += 1
            ref.setdefault(k, r["answers"])
    n = same = missing = flips = 0
    worst = other = 0.0
    by_part: dict = defaultdict(
        lambda: {
            "compared": 0,
            "bit_identical": 0,
            "max_abs_probability_delta": 0.0,
            "choice_flips": 0,
        }
    )
    for r in rows:
        if r["arm"] != arm or r["error"]:
            continue
        base = ref.get((r["request_id"], r.get("cpu_part")))
        if base is None:
            missing += 1
            continue
        n += 1
        same += r["answers"] == base
        w, f, o = prob_deltas(r["answers"], base)
        worst, other, flips = max(worst, w), max(other, o), flips + f
        bp = by_part[str(r.get("cpu_part"))]
        bp["compared"] += 1
        bp["bit_identical"] += r["answers"] == base
        bp["max_abs_probability_delta"] = round(max(bp["max_abs_probability_delta"], w), 6)
        bp["choice_flips"] += f
    return {
        "control": ctl,
        "compared": n,
        "bit_identical": same,
        "max_abs_probability_delta": round(worst, 6),
        "max_abs_score_or_confidence_delta": round(other, 6),
        "choice_flips": flips,
        "within_tolerance": n > 0
        and worst <= TOLERANCE["max_abs_probability_delta"]
        and flips <= TOLERANCE["choice_flips"],
        "by_cpu_part": dict(sorted(by_part.items())),
        "no_ctl_answer_for_cpu_part": missing,
        "ctl_internal_disagreements": ctl_self,
        "tolerance": TOLERANCE,
        "method": f"complete answer object of every successful {arm} invocation vs the {ctl} answer for the same "
        "request id and CPU part in this run; probability delta over choice probabilities and noul",
    }


def reference_tolerance(rows: list[dict], reference: str, arm: str) -> dict:
    """arm's answers against P1's reference/outputs.json for the same request and CPU part, under TOLERANCE."""
    ref = json.load(open(reference))["outputs"]
    n = same = flips = missing = 0
    worst = 0.0
    for r in rows:
        if r["arm"] != arm or r["error"]:
            continue
        base = ref.get(r["request_id"], {}).get(str(r.get("cpu_part")))
        if base is None:
            missing += 1
            continue
        n += 1
        same += r["answers"] == base
        w, f, _o = prob_deltas(r["answers"], base)
        worst, flips = max(worst, w), flips + f
    return {
        "compared": n,
        "bit_identical": same,
        "max_abs_probability_delta": round(worst, 6),
        "choice_flips": flips,
        "no_reference_for_cpu_part": missing,
        "within_tolerance": n > 0
        and worst <= TOLERANCE["max_abs_probability_delta"]
        and flips <= TOLERANCE["choice_flips"],
        "tolerance": TOLERANCE,
    }


def applicability(rows: list[dict]) -> dict:
    """The ggml CPU variant each CPU part loaded (from e15_entry's /proc/self/maps read), and whether candidate A
    changes that variant. A part with no variant record, or with more than one variant, is 'unknown'."""
    seen: dict = defaultdict(set)
    for r in rows:
        e = (r.get("lkai") or {}).get("e15")
        if not e or r["error"] or not known_cpu(r):
            continue
        for lib in e.get("ggml_cpu_mapped") or []:
            if isinstance(lib, str) and lib.startswith("libggml-cpu-"):
                seen[str(r["cpu_part"])].add(lib[len("libggml-cpu-") : -len(".so")])
    out = {}
    for cpu in sorted({str(r["cpu_part"]) for r in rows if known_cpu(r)}):
        v = sorted(seen.get(cpu, ()))
        out[cpu] = {
            "variants": v,
            "changed_by_candidate": (v[0] in CHANGED_VARIANTS) if len(v) == 1 else "unknown",
        }
    return out


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


def status_of(arm: str, arm_res: dict, function_errors: int, ref_tol: dict) -> str:
    """The deployment outcome for one arm, judged on the primary metric (paired warm REPORT Duration vs ctl) on the
    CPU parts whose loaded variant candidate A changes (`applicability`). Other parts are a negative control.

    Correctness gate (else inconclusive): no function error; ctl answers repeat; for `off` every answer is
    bit-identical to ctl's and to P1's reference (the switch is off, so the arithmetic must be P1's); for `a` the
    answers are within TOLERANCE of ctl's and of P1's reference, and bit-identical on parts candidate A cannot change.
    improved: every powered changed part is faster, no underpowered changed part has a round of the opposite sign,
    and no cold/restore first-request CI lies entirely above 0. regressed: a powered changed part is slower, or a
    cold/restore first-request CI lies entirely above 0. inconclusive: no powered changed part. no-change otherwise."""
    g = arm_res["arm_agreement"]
    app = arm_res["applicability"]
    changed = {cpu for cpu, v in app.items() if v["changed_by_candidate"] is True}
    unchanged_exact = all(
        v["bit_identical"] == v["compared"]
        for cpu, v in g["by_cpu_part"].items()
        if app.get(cpu, {}).get("changed_by_candidate") is False
    )
    if arm == "off":
        answers_ok = (
            g["compared"] > 0
            and g["bit_identical"] == g["compared"]
            and ref_tol["bit_identical"] == ref_tol["compared"]
        )
    else:
        answers_ok = g["within_tolerance"] and ref_tol["within_tolerance"] and unchanged_exact
    clean = function_errors == 0 and answers_ok and g["ctl_internal_disagreements"] == 0
    wp = arm_res["warm_paired"]["report_duration"]
    powered = {
        cpu: v["verdict"]
        for cpu, v in wp.items()
        if cpu in changed and v["verdict"] in ("faster", "slower", "no-change")
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
    if "slower" in powered.values() or any(v["ci95"][0] > 0 for v in side):
        return "regressed"
    if set(powered.values()) == {"faster"}:
        contrary = [
            cpu
            for cpu, v in wp.items()
            if cpu in changed
            and v["verdict"] in ("underpowered", "insufficient")
            and v["rounds_above_zero"] > 0
        ]
        return "no-change" if contrary else "improved"
    return "no-change"


def analyse(rows: list[dict], arm: str, ctl: str = CONTROL) -> dict:
    """Everything E15 adds for one experiment arm against a control arm of the same run."""
    warm = [r for r in ok_rows(rows, "warm") if known_cpu(r)]
    parts = sorted(
        {str(r["cpu_part"]) for r in warm if r["arm"] == arm}
        & {str(r["cpu_part"]) for r in warm if r["arm"] == ctl}
    )
    wp: dict = {}
    for name, f in WARM_KEYS.items():
        wp[name] = {}
        for cpu in parts:
            res = paired_by_request(rows, arm, f, cpu, ctl)
            if not res:
                continue
            if name == "report_duration":
                res["control_split_half"] = control_null(rows, f, cpu, ctl=ctl)
                res["verdict"] = warm_verdict(res, res["control_split_half"])
            wp[name][cpu] = res
    return {
        "control": ctl,
        "warm_paired": wp,
        "delta": {
            "cold": delta(rows, arm, "cold", COLD_KEYS, ctl),
            "restore": delta(rows, arm, "restore", RESTORE_KEYS, ctl),
            "warm": delta(rows, arm, "warm", WARM_KEYS, ctl),
        },
        "round_cells": {
            "warm_report_duration": round_cells(
                rows, arm, "warm", WARM_KEYS["report_duration"], ctl
            ),
            "cold_first_request_total": round_cells(
                rows, arm, "cold", COLD_KEYS["first_request_total"], ctl
            ),
            "restore_first_request_total": round_cells(
                rows, arm, "restore", RESTORE_KEYS["first_request_total"], ctl
            ),
        },
        "arm_agreement": arm_agreement(rows, arm, ctl),
        "applicability": applicability(rows),
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
            "cgroup": "per environment in e15.host.identities; any file the sandbox hides is 'unknown'",
        },
        "method": "Every comparison is made within one observed CPU part. Warm (the primary metric): within each "
        "round the arms alternate pass by pass (the arm order rotates every pass), so the arms are interleaved "
        "within the round, and each (round, request) cell is compared on the same CPU part. Cold and restore waves "
        "are sequential arm waves whose position rotates across rounds (a Latin square over 4 rounds): that controls "
        "order and round drift but not time-within-round drift. Rows without a CPU part are counted, never compared.",
        "assessment": {
            arm: {cpu: v.get("verdict") for cpu, v in a["warm_paired"]["report_duration"].items()}
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
        "--stage",
        metavar="RUN_DIR",
        help="only copy a committed run into per-arm publish directories in --out",
    )
    a = ap.parse_args(argv)
    if a.stage:
        for p in stage_for_publish(a.stage, a.out):
            print("staged", p)
        return 0
    for k in ("cold", "run_id", "baseline_commit", "reference", "meta"):
        if not getattr(a, k):
            ap.error(f"--{k.replace('_', '-')} is required")
    run = merge_runs(a.cold, a.restore, a.out)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = dict.fromkeys((CONTROL, *EXPERIMENTS), "1")  # every E15 arm is instrumented
    cond, rows = p1.load_run(run)
    per_arm = {arm: analyse(rows, arm) for arm in EXPERIMENTS if arm in cond["arms"]}
    # the switch alone: a against off (same zip, only LKAI_E15_A differs)
    switch = analyse(rows, "a", ctl="off") if {"a", "off"} <= set(cond["arms"]) else None
    hv = host_variability(rows, per_arm)
    hst = host(rows)
    rids = report_request_ids(run, cond)
    for arm, ana in per_arm.items():
        m = meta["arms"][arm]
        res = p1.summarize(run, a.reference, a.baseline_commit, meta | m)
        errs = error_kinds(res["errors"]["rows"], rows, rids)
        res["errors"]["rows"] = errs["rows"]
        ana = ana | {"error_kinds": {k: v for k, v in errs.items() if k != "rows"}}
        ref_tol = reference_tolerance(rows, a.reference, arm)
        res["experiment_id"] = meta["experiment_id"]
        res["run_id"] = f"{a.run_id}-{arm}"
        res["primary_arm"] = arm
        res["commits"]["experiment"] = cond["build_manifest"]["repo_commit"]
        res["build"] = {
            k: cond["build_manifest"][k]
            for k in (
                "images",
                "toolchain",
                "assets_sha256",
                "zips",
                "base",
                "wrapper_sha256",
                "changed_entries",
                "cand_build",
            )
        }
        res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
        res["conditions"]["legs"] = cond.get("legs")
        res["conditions"]["arm_variables"] = cond.get("arm_variables")
        res["conditions"]["arm_zip"] = cond.get("arm_zip")
        res["instrumentation_overhead"] = {
            "measured": False,
            "note": "every E15 arm runs P1's phase timers and the same wrapper; overhead is P1's "
            "(P1-baseline/20261020T080835Z)",
        }
        res["host_variability"] = hv
        res["e15"] = ana | {
            "host": hst,
            "reference_tolerance": ref_tol,
            "sibling_arms": sorted(per_arm),
        }
        if arm == "a" and switch is not None:
            res["e15"]["vs_off"] = switch
        res["status"] = status_of(arm, ana, errs["function_errors"], ref_tol)
        tpl = m.get("conclusion_template") or meta.get("conclusion_template")
        res["conclusion"] = tpl.format(**fmt(res, arm)) if tpl else m.get("conclusion")
        res["reproduce"] = meta.get("reproduce", [])
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


def fmt(res: dict, arm: str, cpu: str = "0xd40") -> dict:
    """Every number a conclusion template may cite, read from the result itself. Missing values are 'n/a'."""
    e = res["e15"]
    na = "n/a"

    def get(f):
        try:
            return f()
        except (KeyError, IndexError, TypeError):
            return na

    def wpv(blk, k):
        return get(lambda: blk["warm_paired"]["report_duration"][cpu][k])

    def d(blk, cls, k, f):
        return get(lambda: blk["delta"][cls][k]["by_cpu_part"][cpu][f])

    g, rt = e["arm_agreement"], e["reference_tolerance"]
    sw = e.get("vs_off") or {}
    w = get(lambda: e["warm_paired"]["report_duration"][cpu])
    out = {
        "cpu": cpu,
        "rounds": get(lambda: len(w["pct_by_round"])),
        "cells": wpv(e, "cells"),
        "w_mean": wpv(e, "mean_round_pct"),
        "w_lo": get(lambda: min(w["pct_by_round"])),
        "w_hi": get(lambda: max(w["pct_by_round"])),
        "w_below": wpv(e, "rounds_below_zero"),
        "w_p": wpv(e, "sign_flip_p"),
        "w_null": get(lambda: w["control_split_half"]["max_abs_round_pct"]),
        "w_verdict": wpv(e, "verdict"),
        "w_ctl_ms": get(
            lambda: 1000 * e["delta"]["warm"]["report_duration"]["by_cpu_part"][cpu]["ctl_p50"]
        ),
        "w_arm_ms": get(
            lambda: 1000 * e["delta"]["warm"]["report_duration"]["by_cpu_part"][cpu]["arm_p50"]
        ),
        "eng_mean": get(lambda: e["warm_paired"]["steady_state_inference"][cpu]["mean_round_pct"]),
        "cpu_mean": get(lambda: e["warm_paired"]["process_cpu_s"][cpu]["mean_round_pct"]),
        "sw_mean": wpv(sw, "mean_round_pct") if sw else na,
        "sw_p": wpv(sw, "sign_flip_p") if sw else na,
        "sw_below": wpv(sw, "rounds_below_zero") if sw else na,
        "other_parts": _parts_text(e["warm_paired"].get("report_duration", {}), cpu),
        "applicability": "; ".join(
            f"{p}: {','.join(v['variants']) or 'unknown'} "
            f"({'changed' if v['changed_by_candidate'] is True else 'unchanged' if v['changed_by_candidate'] is False else 'unknown'})"
            for p, v in e["applicability"].items()
        ),
        "cold_d": d(e, "cold", "first_request_total", "delta_p50"),
        "cold_lo": get(
            lambda: e["delta"]["cold"]["first_request_total"]["by_cpu_part"][cpu]["ci95"][0]
        ),
        "cold_hi": get(
            lambda: e["delta"]["cold"]["first_request_total"]["by_cpu_part"][cpu]["ci95"][1]
        ),
        "cold_n": d(e, "cold", "first_request_total", "n"),
        "rst_d": d(e, "restore", "first_request_total", "delta_p50"),
        "rst_lo": get(
            lambda: e["delta"]["restore"]["first_request_total"]["by_cpu_part"][cpu]["ci95"][0]
        ),
        "rst_hi": get(
            lambda: e["delta"]["restore"]["first_request_total"]["by_cpu_part"][cpu]["ci95"][1]
        ),
        "rst_n": d(e, "restore", "first_request_total", "n"),
        "mem_d": d(e, "cold", "report_max_memory_mb", "delta_p50"),
        "agree": g["bit_identical"],
        "agree_n": g["compared"],
        "agree_max": g["max_abs_probability_delta"],
        "agree_flips": g["choice_flips"],
        "ref": rt["bit_identical"],
        "ref_n": rt["compared"],
        "ref_max": rt["max_abs_probability_delta"],
        "ref_flips": rt["choice_flips"],
        "ref_missing": rt["no_reference_for_cpu_part"],
        "transport": e["error_kinds"]["client_transport_errors"],
        "function_errors": e["error_kinds"]["function_errors"],
        "restore_timeouts": sum(
            "ReadTimeout" in str(r["error"])
            for r in res["errors"]["rows"]
            if r["wave"] == "restore"
        ),
        "cost": res["cost"]["total_usd"],
        "status": res["status"],
    }

    def r(v, nd=2, sign=True):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
        return v

    for k in ("w_mean", "w_lo", "w_hi", "eng_mean", "cpu_mean", "sw_mean"):
        out[k] = r(out[k])
    for k in ("cold_d", "cold_lo", "cold_hi", "rst_d", "rst_lo", "rst_hi"):
        out[k] = r(out[k], 3)
    for k in ("w_ctl_ms", "w_arm_ms", "mem_d"):
        out[k] = r(out[k], 0, sign=k == "mem_d")
    return out


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
