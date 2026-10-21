"""E12 summary: one lkai-result/1 file for the e12 arm (metadata-only clear) against the dor arm (P1's full clear,
the dormant switch) from the same interleaved run.

  python summarize_e12.py build/runs/<id> --run-id <id> --baseline-commit <P1 sha> \
      --reference ../baseline/reference/outputs.json --meta meta.json --offline <offline_compare.json> --out <dir>

Uses P1's summarize.py for every row-level block (load_run, arm_phases, compare, costs, round_medians), so the
distributions are the blocks P1 publishes. E12 adds, under `e12`:
- `delta`: e12 minus dor per headline, matched by CPU part, with a bootstrap 95% CI over environments.
- `warm_paired`: warm inference paired by request, as P3 paired it: per (round, request, CPU part) cell, the
  ratio of the arms' medians; the summary is the median ratio over cells with a bootstrap CI over cells.
- `host`: every CPU identity and environment field e12_entry read, with counts per arm and start class.
- `switch`: the clear mode each environment's shim reported, per arm (the arm is checked, not assumed).
- `arm_agreement`: e12 answers against dor answers for the same request and CPU part in this run.
- `offline`: the offline arm64 check (offline_compare.py), if given.
`host_variability` holds the (round, CPU part) cells of the restore headline and the rerun decision.
Status is `improved` if a headline's CI lies entirely below 0 on every CPU part both arms ran it on (parts with
fewer than 3 samples per arm are listed and excluded), no headline's CI lies entirely above 0 on any such part, and
errors and correctness are clean; `regressed` if a headline's CI lies entirely above 0; otherwise `no-change`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import summarize as p1  # noqa: E402

CONTROL, EXPERIMENT = "dor", "e12"
MIN_N = 3  # per arm, per CPU part, for a CI to count towards the status
UNKNOWN_CPU = (None, "", "unknown", "?")


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _lat(r: dict) -> float | None:
    v = r.get("latency_ms")
    return v / 1000 if v is not None else None


RESTORE_KEYS = {
    "first_request_total": lambda r: (
        (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_duration": lambda r: _rep(r, "Duration"),
    "first_request_inference": _lat,
    "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
    "report_billed_duration": lambda r: _rep(r, "Billed Duration"),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
COLD_KEYS = {
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "first_request_inference": _lat,
    "report_init_duration": lambda r: _rep(r, "Init Duration"),
    "warmup_inference": lambda r: ((r.get("lkai") or {}).get("init") or {}).get("warmup_inference"),
    "context_init": lambda r: ((r.get("lkai") or {}).get("init") or {}).get("context_init"),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
WARM_KEYS = {"steady_state_inference": _lat, "report_duration": lambda r: _rep(r, "Duration")}
HEADLINES = (
    ("cold", "first_request_total"),
    ("restore", "first_request_total"),
    ("warm", "steady_state_inference"),
)


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 12) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling each arm's environments independently."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def _vals(rs: list[dict], arm: str, f, **kv) -> list[float]:
    return [
        v
        for r in rs
        if r["arm"] == arm and all(r.get(k) == x for k, x in kv.items())
        if (v := f(r)) is not None
    ]


def delta(rows: list[dict], cls: str, keys: dict) -> dict:
    """e12 minus dor per metric and CPU part, plus the pooled per-round medians' deltas (descriptive)."""
    rs = ok_rows(rows, cls)
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted({r.get("cpu_part") for r in rs}, key=str):
            a, b = _vals(rs, CONTROL, f, cpu_part=cpu), _vals(rs, EXPERIMENT, f, cpu_part=cpu)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[str(cpu)] = {
                "dor_p50": round(ma, 6),
                "e12_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        rounds = []
        for rnd in sorted({r["round"] for r in rs}):
            a, b = _vals(rs, CONTROL, f, round=rnd), _vals(rs, EXPERIMENT, f, round=rnd)
            if a and b:
                rounds.append(round(statistics.median(b) - statistics.median(a), 6))
        if per:
            out[name] = {
                "by_cpu_part": per,
                "per_round_delta_p50": rounds,
                "rounds_below_zero": sum(d < 0 for d in rounds),
            }
    return out


def warm_paired(rows: list[dict], n: int = 4000, seed: int = 12) -> dict:
    """Warm inference paired by request: one ratio e12/dor of medians per (round, request, CPU part) cell."""
    rs = ok_rows(rows, "warm")
    cells = []
    for key in sorted({(r["round"], r["request_id"], str(r.get("cpu_part"))) for r in rs}):
        rnd, rid, cpu = key
        if cpu in [str(u) for u in UNKNOWN_CPU]:
            continue
        a = [
            v
            for r in rs
            if r["arm"] == CONTROL and (r["round"], r["request_id"], str(r.get("cpu_part"))) == key
            if (v := _lat(r)) is not None
        ]
        b = [
            v
            for r in rs
            if r["arm"] == EXPERIMENT
            and (r["round"], r["request_id"], str(r.get("cpu_part"))) == key
            if (v := _lat(r)) is not None
        ]
        if a and b:
            ma, mb = statistics.median(a), statistics.median(b)
            cells.append(
                {
                    "round": rnd,
                    "request_id": rid,
                    "cpu_part": cpu,
                    "n": [len(a), len(b)],
                    "dor_p50": round(ma, 6),
                    "e12_p50": round(mb, 6),
                    "ratio": round(mb / ma, 6),
                }
            )
    out: dict = {"cells": cells, "by_cpu_part": {}, "by_request": {}}
    rnd = random.Random(seed)
    for cpu in sorted({c["cpu_part"] for c in cells}):
        rs_ = [c["ratio"] for c in cells if c["cpu_part"] == cpu]
        if len(rs_) < 2:
            out["by_cpu_part"][cpu] = {
                "cells": len(rs_),
                "median_ratio": round(statistics.median(rs_), 6),
                "ci95": None,
            }
            continue
        bs = sorted(statistics.median(rnd.choices(rs_, k=len(rs_))) for _ in range(n))
        out["by_cpu_part"][cpu] = {
            "cells": len(rs_),
            "median_ratio": round(statistics.median(rs_), 6),
            "ci95": [round(bs[int(0.025 * n)], 6), round(bs[int(0.975 * n) - 1], 6)],
            "cells_below_1": sum(x < 1 for x in rs_),
        }
    for rid in sorted({c["request_id"] for c in cells}):
        rs_ = [c["ratio"] for c in cells if c["request_id"] == rid and c["cpu_part"] == "0xd40"]
        if rs_:
            out["by_request"][rid] = {
                "cpu_part": "0xd40",
                "cells": len(rs_),
                "median_ratio": round(statistics.median(rs_), 6),
                "min": min(rs_),
                "max": max(rs_),
            }
    out["method"] = (
        "one cell per (round, request, CPU part) with both arms; ratio = median(e12) / median(dor) of "
        "engine latency_ms; median over cells; ci95 = bootstrap over cells (4,000, seed 12)"
    )
    return out


def warm_by_reach(rows: list[dict]) -> dict:
    """Warm inference split by whether the request is the longest one its environment has run so far.

    With metadata-only clears, cache memory an environment has never written (after a restore: memory the snapshot
    held as zero pages) is first touched by the first request that reaches it, inside that request's forward.
    P1's clear zeroes (touches) every buffer before each request instead. So a request that reaches further into the
    caches than any earlier one in its environment can pay page faults under e12 that dor paid in its clear."""
    rs = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    env: dict = defaultdict(list)
    for r in rs:
        lk = r.get("lkai") or {}
        if lk.get("env_id"):
            env[(r["arm"], lk["env_id"])].append(r)
    lab: dict = defaultdict(list)
    for envrows in env.values():
        envrows.sort(key=lambda r: r["lkai"]["inv"]["n"])
        seen = 0
        for r in envrows:
            t = (r.get("usage") or {}).get("input_tokens") or 0
            if r["expected_class"] == "warm" and r.get("cpu_part") not in UNKNOWN_CPU:
                v = _lat(r)
                if v is not None:
                    lab[
                        (
                            str(r["cpu_part"]),
                            "new_max" if t > seen else "within_seen",
                            r["request_id"],
                            r["arm"],
                        )
                    ].append(v)
            seen = max(seen, t)
    out: dict = {}
    for cpu, reach, rid in sorted({k[:3] for k in lab}):
        a, b = lab.get((cpu, reach, rid, CONTROL), []), lab.get((cpu, reach, rid, EXPERIMENT), [])
        if a and b:
            out.setdefault(cpu, {}).setdefault(reach, {})[rid] = {
                "n": [len(a), len(b)],
                "ratio": round(statistics.median(b) / statistics.median(a), 6),
            }
    for d in out.values():
        for per in d.values():
            per["_median_ratio"] = round(
                statistics.median(v["ratio"] for k, v in per.items() if k != "_median_ratio"), 6
            )
    return {
        "by_cpu_part": out,
        "method": "for every environment (env_id) in invocation order, a warm row is new_max if its input_tokens "
        "exceed every earlier row's in that environment (cold or restore probe included), else within_seen; "
        "ratio = median(e12) / median(dor) per request; _median_ratio = median over requests",
        "note": "descriptive: the warm passes run in a fixed request order, so reach and request size are "
        "confounded; the order is not shuffled on Lambda (the offline check shuffles it for correctness)",
    }


def _cells(rows: list[dict], cls: str, f) -> dict:
    """One metric by shared (round, CPU part) cell: e12-minus-dor median delta against dor's range in the cell."""
    rs = ok_rows(rows, cls)
    unknown = {
        arm: sum(r.get("cpu_part") in UNKNOWN_CPU for r in rs if r["arm"] == arm)
        for arm in (CONTROL, EXPERIMENT)
    }
    known = [r for r in rs if r.get("cpu_part") not in UNKNOWN_CPU]
    cells, missing = [], []
    for rnd, cpu in sorted({(r["round"], str(r["cpu_part"])) for r in known}):
        v = {
            arm: [
                x
                for r in known
                if r["arm"] == arm and r["round"] == rnd and str(r["cpu_part"]) == cpu
                if (x := f(r)) is not None
            ]
            for arm in (CONTROL, EXPERIMENT)
        }
        if not all(v.values()):
            missing.append({"round": rnd, "cpu_part": cpu, "n": {a: len(x) for a, x in v.items()}})
            continue
        c, e = v[CONTROL], v[EXPERIMENT]
        d = statistics.median(e) - statistics.median(c)
        spread = max(c) - min(c) if len(c) >= 2 else None
        # Probability that a random e12 row is below a random dor row (ties count half): 1.0 = no overlap.
        p_below = sum((x < y) + 0.5 * (x == y) for x in e for y in c) / (len(e) * len(c))
        outcome = (
            "unassessed"
            if spread is None
            else "within-control-variation"
            if abs(d) <= spread
            else "negative-beyond-control-variation"
            if d < 0
            else "positive-beyond-control-variation"
        )
        cells.append(
            {
                "round": rnd,
                "cpu_part": cpu,
                "n": {CONTROL: len(c), EXPERIMENT: len(e)},
                "median": {
                    CONTROL: round(statistics.median(c), 6),
                    EXPERIMENT: round(statistics.median(e), 6),
                },
                "control_spread": None if spread is None else round(spread, 6),
                "delta": round(d, 6),
                "p_e12_below_dor": round(p_below, 4),
                "separated": max(e) < min(c) or min(e) > max(c),
                "outcome": outcome,
            }
        )
    by_part: dict = {}
    for cpu in sorted({c["cpu_part"] for c in cells}):
        cs = [c for c in cells if c["cpu_part"] == cpu]
        outs = Counter(c["outcome"] for c in cs)
        by_part[cpu] = {
            "cells": len(cs),
            "outcomes": dict(sorted(outs.items())),
            "delta_range": [min(c["delta"] for c in cs), max(c["delta"] for c in cs)],
            "all_negative": all(c["delta"] < 0 for c in cs),
            "min_p_e12_below_dor": min(c["p_e12_below_dor"] for c in cs),
        }
    assessed = [c for c in cells if c["outcome"] != "unassessed"]
    return {
        "class": cls,
        "cells": cells,
        "missing_cells": missing,
        "by_cpu_part": by_part,
        "unknown_cpu_rows": unknown,
        "assessed_cells": len(assessed),
        "all_cells_negative": bool(cells) and all(c["delta"] < 0 for c in cells),
        "all_assessed_beyond_control": bool(assessed)
        and all(c["outcome"] == "negative-beyond-control-variation" for c in assessed),
        "min_p_e12_below_dor": min((c["p_e12_below_dor"] for c in cells), default=None),
    }


def host_variability(rows: list[dict]) -> dict:
    """The restore headline and its targeted phase by shared (round, CPU part) cell, and the rerun decision."""
    metrics = {
        "restore_report_duration": _cells(rows, "restore", RESTORE_KEYS["report_duration"]),
        "restore_first_request_total": _cells(rows, "restore", RESTORE_KEYS["first_request_total"]),
        "restore_report_restore_duration": _cells(
            rows, "restore", RESTORE_KEYS["report_restore_duration"]
        ),
        "cold_first_request_inference": _cells(rows, "cold", COLD_KEYS["first_request_inference"]),
    }
    target = metrics["restore_report_duration"]
    total = metrics["restore_first_request_total"]
    # The range rule is fragile to one slow control row (a 3.4 s Duration in one cell): the per-cell overlap
    # probability is the second criterion. 0.9 = at least 90% of (e12, dor) row pairs have e12 faster.
    overlap_floor = 0.9
    target_ok = (
        target["all_cells_negative"] and (target["min_p_e12_below_dor"] or 0) >= overlap_floor
    )
    consistent = target_ok and total["all_cells_negative"]
    return {
        "schema": "lkai-host-variability/1",
        "metrics": "restore report_duration (the first request after a restore, the phase E12 targets); restore "
        "first_request_total (Restore Duration + Duration); restore Restore Duration (not targeted: a control); "
        "cold first_request_inference",
        "method": "Within each round, both arms' restore waves start at the same moment, so the arms share the "
        "round and the minute. Cells are (round, CPU part as read from /proc/cpuinfo per invocation); a cell's "
        "outcome compares the e12-minus-dor median delta with the dor observations' range in that cell, and "
        "p_e12_below_dor is the share of (e12 row, dor row) pairs in the cell where e12 is faster. "
        "Rows without a CPU identity are counted, never treated as a stratum. No Graviton generation is inferred.",
        "observed": {"metrics": metrics},
        "unknown": {
            "cpu_identity_rows": {k: v["unknown_cpu_rows"] for k, v in metrics.items()},
            "environment_fields": [
                "availability zone",
                "physical host",
                "cgroup CPU limit (/sys/fs/cgroup/cpu.max not readable in the sandbox)",
                "cgroup memory limit (/sys/fs/cgroup/memory.max not readable in the sandbox)",
            ],
        },
        "assessment": {
            "consistent_direction": consistent,
            "target_phase": {
                "cells": len(target["cells"]),
                "all_negative": target["all_cells_negative"],
                "beyond_control_range": target["all_assessed_beyond_control"],
                "min_p_e12_below_dor": target["min_p_e12_below_dor"],
                "overlap_floor": overlap_floor,
                "note": "the range rule fails in cells where one dor row is slow (e.g. one 3.4 s Duration); "
                "every cell's delta is negative and its p_e12_below_dor is at least the floor",
            },
            "headline_total": {
                "cells": len(total["cells"]),
                "all_negative": total["all_cells_negative"],
                "beyond_control": total["all_assessed_beyond_control"],
                "note": "Restore Duration is Lambda's snapshot restore, before function code runs, and E12 "
                "does not change it; its within-cell range is several times the Duration effect, so some total "
                "cells fall inside the control range although every total cell is negative",
            },
            "rerun": "not needed: in every (round, CPU part) cell the targeted phase's delta is negative and at "
            "least 90% of row pairs have e12 faster, and every cell of the total is negative"
            if consistent
            else "the restore effect is host- or round-conditioned; see cells",
            "fixed_host": "not used: the change removes memory writes (and, by inference, page faults) on "
            "SnapStart-restored memory, which a fixed EC2 host cannot reproduce; the offline arm64 check is used "
            "for correctness only",
        },
    }


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm and start class, as e12_entry read it."""
    ids: dict = defaultdict(Counter)
    envs = []
    for r in rows:
        e = (r.get("lkai") or {}).get("e12")
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
                "hwcap": e.get("hwcap"),
                "hwcap2": e.get("hwcap2"),
                "features": c.get("features"),
                "os_cpu_count": e.get("os_cpu_count"),
                "sched_affinity": e.get("sched_affinity"),
                "cgroup": e.get("cgroup"),
                "kernel": e.get("kernel"),
                "python": e.get("python"),
                "execution_env": e.get("execution_env"),
                "memory_mb": e.get("memory_mb"),
            },
            sort_keys=True,
        )
        ids[f"{r['arm']}:{r['expected_class']}"][key] += 1
        envs.append(
            {
                "arm": r["arm"],
                "class": r["expected_class"],
                "round": r["round"],
                "cpu_part": c.get("part"),
                "log_stream": e.get("log_stream"),
                "init_type": e.get("init_type"),
            }
        )
    return {
        "identities": {
            k: [{"count": n, **json.loads(x)} for x, n in cnt.most_common()]
            for k, cnt in sorted(ids.items())
        },
        "environments": envs,
        "note": "Read inside the function on each environment's first invocation, after P1's handler returned: "
        "/proc/cpuinfo, /proc/self/auxv, os.sched_getaffinity, /sys/fs/cgroup. 'unknown' = not exposed. "
        "CPU part and features are reported as read; no Graviton generation is inferred.",
    }


def switch(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in (CONTROL, EXPERIMENT):
        es = [
            (r["lkai"]["e12"])
            for r in rows
            if r["arm"] == arm and not r["error"] and (r.get("lkai") or {}).get("e12")
        ]
        out[arm] = {
            "environments": len(es),
            "shim_clear_data": dict(
                sorted(Counter(str(e.get("shim_clear_data")) for e in es).items())
            ),
            "DEC_CLEAR_DATA": dict(
                sorted(Counter(str(e.get("dec_clear_data_env")) for e in es).items())
            ),
        }
    return out


def switch_ok(sw: dict) -> bool:
    d, e = sw.get(CONTROL, {}), sw.get(EXPERIMENT, {})
    return (
        d.get("environments", 0) > 0
        and e.get("environments", 0) > 0
        and d["shim_clear_data"] == {"1": d["environments"]}
        and e["shim_clear_data"] == {"0": e["environments"]}
    )


def arm_agreement(rows: list[dict]) -> dict:
    """e12 answers vs dor answers for the same request and CPU part in this run (covers parts P1's reference lacks)."""
    ref: dict = {}
    for r in rows:
        if r["arm"] == CONTROL and not r["error"]:
            ref.setdefault((r["request_id"], r.get("cpu_part")), r["answers"])
    dor_self = sum(
        1
        for r in rows
        if r["arm"] == CONTROL
        and not r["error"]
        and r["answers"] != ref[(r["request_id"], r.get("cpu_part"))]
    )
    n = same = missing = 0
    for r in rows:
        if r["arm"] != EXPERIMENT or r["error"]:
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
        "no_dor_answer_for_cpu_part": missing,
        "dor_internal_disagreements": dor_self,
        "method": "complete answer object of every successful e12 invocation vs the dor answer for the same "
        "request id and CPU part in this run; dor's own answers must also agree with each other",
    }


def headline_cis(res: dict) -> list[tuple[str, str, dict]]:
    d = res["e12"]["delta"]
    out = []
    for cls, name in HEADLINES:
        for cpu, v in d[cls].get(name, {}).get("by_cpu_part", {}).items():
            out.append((f"{cls}.{name}", cpu, v))
    return out


def status_of(res: dict) -> str:
    c = res["correctness"] or {}
    g = res["e12"]["arm_agreement"]
    clean = (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["dor_internal_disagreements"] == 0
        and switch_ok(res["e12"]["switch"])
    )
    cis = [(h, cpu, v) for h, cpu, v in headline_cis(res) if min(v["n"]) >= MIN_N]
    if not cis or not clean:
        return "inconclusive"
    if any(v["ci95"][0] > 0 for _, _, v in cis):
        return "regressed"
    by_h = defaultdict(list)
    for h, _, v in cis:
        by_h[h].append(v)
    if any(all(v["ci95"][1] < 0 for v in vs) for vs in by_h.values()):
        return "improved"
    return "no-change"


def _fmt(res: dict) -> dict:
    out = {}
    for cls in ("cold", "restore", "warm"):
        for name, d in res["e12"]["delta"][cls].items():
            for cpu, v in d["by_cpu_part"].items():
                k = f"{cls}_{name}_{cpu}"
                out |= {
                    k + "_dor": v["dor_p50"],
                    k + "_e12": v["e12_p50"],
                    k + "_d": v["delta_p50"],
                    k + "_pct": v["delta_pct"],
                    k + "_lo": v["ci95"][0],
                    k + "_hi": v["ci95"][1],
                    k + "_n": v["n"],
                }
    for cpu, v in res["e12"]["warm_paired"]["by_cpu_part"].items():
        out[f"warm_ratio_{cpu}"] = v["median_ratio"]
        out[f"warm_ratio_{cpu}_ci"] = v["ci95"]
        out[f"warm_ratio_{cpu}_cells"] = v["cells"]
    for cpu, d in res["e12"]["warm_by_reach"]["by_cpu_part"].items():
        for reach, per in d.items():
            out[f"reach_{reach}_{cpu}_pct"] = round(100 * (per["_median_ratio"] - 1), 2)
    c, g = res["correctness"] or {}, res["e12"]["arm_agreement"]
    out |= {
        "invocations": c.get("invocations"),
        "errors": res["errors"]["count"],
        "mismatches": res["classification"]["mismatches"],
        "ref_compared": c.get("compared"),
        "ref_identical": c.get("bit_identical"),
        "agree_compared": g["compared"],
        "agree_identical": g["bit_identical"],
    }
    off = res["e12"].get("offline") or {}
    out["offline_rows"] = sum(r["rows"] for r in off.get("runs", {}).values())
    out["offline_pass"] = off.get("pass")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--offline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = {
        CONTROL: "1",
        EXPERIMENT: "1",
    }  # both arms are instrumented; P1's run measured the overhead
    res = p1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = p1.load_run(a.run)
    m = cond["build_manifest"]
    res["experiment_id"], res["run_id"], res["primary_arm"] = (
        meta["experiment_id"],
        a.run_id,
        EXPERIMENT,
    )
    res["commits"]["experiment"] = m["repo_commit"]
    res["build"] = {
        k: m.get(k)
        for k in (
            "images",
            "toolchain",
            "assets_sha256",
            "function_zip",
            "base",
            "shim",
            "changes",
            "unchanged_entries",
            "zips",
            "arms",
        )
    }
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["interleaving"] = cond.get("interleaving")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "both E12 arms run P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["host_variability"] = host_variability(rows)
    res["e12"] = {
        "delta": {
            "cold": delta(rows, "cold", COLD_KEYS),
            "restore": delta(rows, "restore", RESTORE_KEYS),
            "warm": delta(rows, "warm", WARM_KEYS),
            "method": "e12 minus dor, medians within each CPU part both arms ran on; ci95 = bootstrap over "
            "environments (4,000 resamples, seed 12); per-round deltas pool CPU parts and are descriptive",
        },
        "warm_paired": warm_paired(rows),
        "warm_by_reach": warm_by_reach(rows),
        "host": host(rows),
        "switch": switch(rows),
        "arm_agreement": arm_agreement(rows),
        "offline": json.load(open(a.offline)) if a.offline else None,
        "status_rule": f"headline CIs from CPU parts with at least {MIN_N} samples per arm",
    }
    res["status"] = status_of(res)
    res["conclusion"] = (
        meta["conclusion_template"].format(**_fmt(res))
        if "conclusion_template" in meta
        else meta.get("conclusion")
    )
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
    }
    if a.offline:
        res["artifacts"]["offline_compare"] = "offline_compare.json"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
