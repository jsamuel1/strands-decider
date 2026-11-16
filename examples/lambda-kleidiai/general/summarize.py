#!/usr/bin/env python3
"""Summarize a general-ladder run (FIXED HOST, NOT LAMBDA) into per-rung tables, verdict inputs and P2 result files.

  python summarize.py <run dir> [<run dir> ...] --out <dir> [--verdicts verdicts.json --commit SHA --run-id ID]

Reads every r<round>-<arm>-<mode>.json probe file (probe.py) of each run. For each rung it compares the rung's arm
against P1 (`p1`) on the same host identity, the same mode and the SAME ROUNDS only: the paired value per round is
arm minus p1 in that round. The effect is the median of the per-round paired deltas, and the spread is the control's
own run-to-run range (max minus min of p1's per-round values). "no change" is called when |median delta| is inside
that spread, or when the paired deltas do not all have one sign. Standard library only, so it runs on the DevSpace's
system python too.

Writes <out>/ladder.json (everything), <out>/table.md and, per rung, <out>/p2/<rung>/results.json in lkai-result/1
with status from the same rule. Fixed-host numbers are published under P2's phase names with the labels below, so
the dashboard's columns read: cold first request = load + first request (model_ready), warm = steady-state inference.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
FNAME = re.compile(r"^r(\d+)-([A-Za-z0-9]+)-(local|memfd)\.json$")
# The rungs, their arm, the dormant check arm, and what the variable needs from a deployment.
RUNGS = {
    "e8": {
        "arm": "e8",
        "dormant": "e8off",
        "experiment": "E8-parallel-repack",
        "name": "GGML_KLEIDIAI_REPACK_THREADS (KleidiAI load-time repack threads)",
        "baseline": "unset (serial)",
        "value": "6",
    },
    "e9": {
        "arm": "e9",
        "dormant": "e9p1",
        "experiment": "E9-prepacked",
        "name": "KleidiAI-prepacked GGUF + E9 loader",
        "baseline": "P1 GGUF, runtime repack",
        "value": "s2b-h512-Q8_0-e9kr8.gguf + patch_e9.py",
    },
    "e7": {
        "arm": "e7",
        "dormant": "e7off",
        "experiment": "E7-skip-tied-output",
        "name": "LKAI_SKIP_TIED_OUTPUT (skip the tied output head)",
        "baseline": "unset",
        "value": "1",
    },
    "e10": {
        "arm": "e10",
        "dormant": None,
        "experiment": "E10-kv-size",
        "name": "llama.cpp n_ctx",
        "baseline": "16384",
        "value": "4096",
    },
    "e5": {
        "arm": "e5",
        "dormant": None,
        "experiment": "E5-no-vocab-gguf",
        "name": "GGUF tokenizer metadata",
        "baseline": "gpt2 BPE vocabulary embedded",
        "value": "tokenizer.ggml.model=none",
    },
    "e6": {
        "arm": "e6",
        "dormant": None,
        "experiment": "E6-bytecode",
        "name": "shipped Python bytecode",
        "baseline": "P1's (149 stale .pyc)",
        "value": "complete unchecked-hash .pyc",
    },
}
# metric -> (unit, extractor). Lower is better for every one.
METRICS = {
    "imports_s": ("s", lambda d: d["phases_s"].get("imports")),
    "model_load_mmap_s": ("s", lambda d: d["phases_s"].get("model_load_mmap")),
    "context_init_s": ("s", lambda d: d["phases_s"].get("context_init")),
    "load_s": ("s", lambda d: d["phases_s"].get("dec_load_total")),
    "first_inference_s": ("s", lambda d: d["phases_s"].get("warmup_inference")),
    "first_request_s": ("s", lambda d: d["phases_s"].get("first_request_inference")),
    "model_ready_s": ("s", lambda d: d["phases_s"].get("model_ready")),
    "warm_s": ("s", lambda d: statistics.median(v for vs in d["warm_s"].values() for v in vs)),
    "forward4096_s": ("s", lambda d: statistics.median(d["long_forward"]["seconds"])),
    "peak_rss_mb": ("MB", lambda d: d["status_end_kb"]["VmHWM"] / 1024),
}
HEADLINE = ("load_s", "model_ready_s", "warm_s", "forward4096_s", "peak_rss_mb")
# Below MIN_ROUNDS the control has no usable run-to-run range: the cell is reported, never judged.
MIN_ROUNDS = 3
# Peak RSS: changes under 1% of the control are "no change".
MEM_FLOOR = 0.01


def positions(run_dir: str) -> dict[tuple[int, str, str], int]:
    """(round, arm, mode) -> position in that round's order, for the successful loads in runlog.jsonl."""
    p = os.path.join(run_dir, "runlog.jsonl")
    if not os.path.exists(p):
        return {}
    return {
        (r["round"], r["arm"], r["mode"]): r["pos"]
        for r in map(json.loads, open(p))
        if not r.get("rc")
    }


def load_run(run_dir: str) -> list[dict]:
    rows = []
    pos = positions(run_dir)
    for p in sorted(glob.glob(os.path.join(run_dir, "r*-*.json"))):
        m = FNAME.match(os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        d["_round"], d["_arm"], d["_mode"], d["_run"] = (
            int(m[1]),
            m[2],
            m[3],
            os.path.basename(run_dir),
        )
        d["_pos"] = pos.get((d["_round"], d["_arm"], d["_mode"]))
        d["_identity"] = (d.get("host_before") or {}).get("identity", "unknown")
        d["_machine"] = (d.get("host_before") or {}).get("machine", "unknown")
        rows.append(d)
    return rows


def failed_loads(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "runlog.jsonl")
    if not os.path.exists(p):
        return []
    run = os.path.basename(os.path.normpath(run_dir))
    return [r | {"run": run} for r in map(json.loads, open(p)) if r.get("rc")]


def cell(rows: list[dict], arm: str, mode: str, ident: str) -> dict[tuple[str, int], dict]:
    """One (arm, mode, identity) cell keyed by (run, round): a pairing never crosses runs."""
    return {
        (d["_run"], d["_round"]): d
        for d in rows
        if d["_arm"] == arm and d["_mode"] == mode and d["_identity"] == ident
    }


def paired(rows: list[dict], arm: str, mode: str, ident: str, ctl: str = "p1") -> dict:
    """Per-metric comparison of `arm` with `ctl` over the rounds both ran (same identity, same mode)."""
    a, c = cell(rows, arm, mode, ident), cell(rows, ctl, mode, ident)
    rounds = sorted(set(a) & set(c))
    known = [r for r in rounds if a[r].get("_pos") is not None and c[r].get("_pos") is not None]
    out = {
        "rounds": rounds,
        "n": len(rounds),
        "metrics": {},
        "missing": [],
        "non_finite": [],
        # How often the arm ran before / after its control in the paired rounds (runlog positions).
        "order": {
            "arm_first": sum(1 for r in known if a[r]["_pos"] < c[r]["_pos"]),
            "control_first": sum(1 for r in known if a[r]["_pos"] > c[r]["_pos"]),
            "unknown": len(rounds) - len(known),
        },
    }
    for name, (unit, f) in METRICS.items():
        av = [f(a[r]) for r in rounds]
        cv = [f(c[r]) for r in rounds]
        if not rounds or any(v is None for v in av + cv):
            out["missing"].append(name)
            continue
        if not all(math.isfinite(v) for v in av + cv):
            out["non_finite"].append(name)
            continue
        deltas = [x - y for x, y in zip(av, cv, strict=True)]
        spread = max(cv) - min(cv)
        # Peak RSS repeats to within a page or two between loads, so its range can be 0; a fixed
        # materiality floor (MEM_FLOOR of the control) stops a few-MB difference reading as an effect.
        # Timings use the control's range alone.
        floor = MEM_FLOOR * statistics.median(cv) if unit == "MB" else 0.0
        threshold = max(spread, floor)
        med = statistics.median(deltas)
        same_sign = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
        outcome = "no change"
        if len(rounds) < MIN_ROUNDS:
            outcome = "unassessed"
        elif abs(med) > threshold and same_sign:
            outcome = "better" if med < 0 else "worse"
        out["metrics"][name] = {
            "unit": unit,
            "control_median": round(statistics.median(cv), 6),
            "arm_median": round(statistics.median(av), 6),
            "delta_median": round(med, 6),
            "delta_pct": round(100 * med / statistics.median(cv), 3)
            if statistics.median(cv)
            else None,
            "deltas": [round(x, 6) for x in deltas],
            "control_spread": round(spread, 6),
            "threshold": round(threshold, 6),
            "arm_spread": round(max(av) - min(av), 6),
            "outcome": outcome,
        }
    return out


def correctness(rows: list[dict], arm: str, ident: str) -> dict:
    """Answers vs P1's reference (same identity) and vs this run's P1 arm IN THE SAME ROUND AND MODE (every request,
    the 4096 forward). An arm row without a same-(run, round, mode) P1 row is unpaired and counts as not identical;
    a request either side lacks counts as compared and not identical."""
    own = [d for d in rows if d["_arm"] == arm and d["_identity"] == ident]
    ctl = {
        (d["_run"], d["_round"], d["_mode"]): d
        for d in rows
        if d["_arm"] == "p1" and d["_identity"] == ident
    }
    vs_ref = {"identical": 0, "differs": 0, "no_reference": 0}
    for d in own:
        for v in (d.get("vs_reference") or {}).values():
            key = {"identical": "identical", "DIFFERS": "differs"}.get(v, "no_reference")
            vs_ref[key] += 1
    same_p1 = total = long_same = unpaired = 0
    flips, max_abs = 0, 0.0
    for d in own:
        c = ctl.get((d["_run"], d["_round"], d["_mode"]))
        if c is None:
            unpaired += 1
        ca = (c or {}).get("answers") or {}
        rids = set(d["answers"]) | set(ca)
        total += len(rids)
        # Canonical JSON text: NaN compares equal to NaN (bit-identity of the published answer), never by identity.
        same_p1 += sum(
            1
            for k in rids
            if c is not None
            and k in d["answers"]
            and k in ca
            and canon(ca[k]) == canon(d["answers"][k])
        )
        if c is not None and canon(d["long_forward"]["answers"]) == canon(
            c["long_forward"]["answers"]
        ):
            long_same += 1
        for k, v in d["answers"].items():
            f, m = answer_delta(v, ca.get(k))
            flips, max_abs = flips + f, max(max_abs, m)
    # The long forward must actually fill the window (probe.long_request trims the state to exactly `window`).
    long_short = sum(
        1 for d in own if d["long_forward"].get("input_tokens") != d["long_forward"].get("window")
    )
    return {
        "vs_reference_requests": vs_ref,
        # Requests the probe answered; on an identity with a published reference every one must be compared with it.
        "requests_answered": sum(len(d["answers"]) for d in own),
        "vs_run_p1_requests": {"identical": same_p1, "compared": total},
        "vs_run_p1_forward4096": {"identical": long_same, "compared": len(own)},
        "unpaired_rows": unpaired,
        "forward4096_not_full_window": long_short,
        # probe.py compares repeats as canonical JSON text (repeat_compare), so NaN repeats compare equal.
        "repeat_identical": bool(own)
        and all(
            d.get("repeat_compare") == "canonical-json"
            and d["answers_repeat_identical"]
            and d["long_forward"]["repeat_identical"]
            for d in own
        ),
        "choice_flips_vs_p1": flips,
        "max_abs_delta_vs_p1": max_abs,
        "non_finite_fields": non_finite_fields(own),
    }


def non_finite_fields(own: list[dict]) -> list[str]:
    """Every answer field (requests and the 4096-token forward) holding NaN, +Infinity or -Infinity."""
    out = set()
    for d in own:
        answers = [*d["answers"].items(), ("forward-4096", d["long_forward"]["answers"])]
        for rid, a in answers:
            out |= {f"{rid}{k}" for k, x in _nums(a) if not math.isfinite(x)}
    return sorted(out)


def canon(x) -> str:
    return json.dumps(x, sort_keys=True)


def _nums(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _nums(v, f"{prefix}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _nums(v, f"{prefix}/{i}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def answer_delta(a: dict | None, b: dict | None) -> tuple[int, float]:
    """(choice flips, max |numeric difference|) between two answer sets of one request."""
    if a is None or b is None:
        return 0, 0.0
    flips = 0
    for q in set(a) & set(b):
        if a[q].get("choice") != b[q].get("choice"):
            flips += 1
    na, nb = dict(_nums(a)), dict(_nums(b))
    # A non-finite value on either side is not a numeric difference: it is listed in non_finite_fields instead.
    m = max(
        (
            abs(na[k] - nb[k])
            for k in set(na) & set(nb)
            if math.isfinite(na[k]) and math.isfinite(nb[k])
        ),
        default=0.0,
    )
    return flips, m


def summarize(rows: list[dict]) -> dict:
    idents = sorted({d["_identity"] for d in rows})
    out: dict = {"identities": idents, "rungs": {}, "dormant": {}, "control": {}}
    for ident in idents:
        for mode in ("local", "memfd"):
            c = cell(rows, "p1", mode, ident)
            if c:
                out["control"][f"{ident}|{mode}"] = {
                    name: {
                        "per_round": [round(f(c[r]), 6) for r in sorted(c)],
                        "spread": round(max(f(c[r]) for r in c) - min(f(c[r]) for r in c), 6),
                    }
                    for name, (_, f) in METRICS.items()
                }
        for rung, spec in RUNGS.items():
            for mode in ("local", "memfd"):
                if not cell(rows, spec["arm"], mode, ident):
                    continue
                out["rungs"].setdefault(rung, {})[f"{ident}|{mode}"] = paired(
                    rows, spec["arm"], mode, ident
                )
            if any(d["_arm"] == spec["arm"] and d["_identity"] == ident for d in rows):
                out["rungs"].setdefault(rung, {})[f"{ident}|correctness"] = correctness(
                    rows, spec["arm"], ident
                )
            if spec["dormant"] and cell(rows, spec["dormant"], "local", ident):
                out["dormant"].setdefault(rung, {})[ident] = {
                    "timing": paired(rows, spec["dormant"], "local", ident),
                    "correctness": correctness(rows, spec["dormant"], ident),
                }
    # E8 vs E9 head to head (the two alternatives for the same repack cost), same rounds.
    for ident in idents:
        for mode in ("local", "memfd"):
            if cell(rows, "e8", mode, ident) and cell(rows, "e9", mode, ident):
                out.setdefault("e8_vs_e9", {})[f"{ident}|{mode}"] = paired(
                    rows, "e9", mode, ident, ctl="e8"
                )
    return out


def fmt(m: dict | None, k: str) -> str:
    if not m or k not in m:
        return "-"
    x = m[k]
    if x["unit"] == "MB":
        return f"{x['control_median']:.0f} → {x['arm_median']:.0f} ({x['delta_median']:+.0f}, spread {x['control_spread']:.0f}) {x['outcome']}"
    return f"{x['control_median']:.3f} → {x['arm_median']:.3f} ({x['delta_median']:+.3f}, spread {x['control_spread']:.3f}) {x['outcome']}"


def table_md(s: dict) -> str:
    lines = [
        "| rung | host identity | mode | n rounds | " + " | ".join(HEADLINE) + " |",
        "|---|---|---|---|" + "---|" * len(HEADLINE),
    ]
    for rung, cells in s["rungs"].items():
        for key, v in cells.items():
            ident, mode = key.split("|")
            if mode == "correctness":
                continue
            lines.append(
                f"| {rung} | {ident} | {mode} | {v['n']} | "
                + " | ".join(fmt(v["metrics"], k) for k in HEADLINE)
                + " |"
            )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- P2 results
def _dist(xs: list[float], unit: str) -> dict | None:
    v = [x for x in xs if x is not None and math.isfinite(x)]
    if not v:
        return None
    s = sorted(v)
    pct = lambda p: s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]  # noqa: E731
    lo, hi = s[0], s[-1]
    bins = 12
    if hi == lo:
        hist = {"edges": [lo, hi], "counts": [len(v)]}
    else:
        w = (hi - lo) / bins
        counts = [0] * bins
        for x in v:
            counts[min(bins - 1, int((x - lo) / w))] += 1
        hist = {"edges": [round(lo + i * w, 6) for i in range(bins + 1)], "counts": counts}
    r = lambda x: round(x, 6)  # noqa: E731
    return {
        "unit": unit,
        "n": len(v),
        "p50": r(pct(50)),
        "p90": r(pct(90)),
        "p99": r(pct(99)),
        "min": r(lo),
        "max": r(hi),
        "mean": r(statistics.fmean(v)),
        "stdev": r(statistics.stdev(v)) if len(v) > 1 else 0.0,
        "histogram": hist,
    }


def _phases(rows: list[dict]) -> dict:
    """P2 start-class blocks from fixed-host loads. 'cold' = a new process loading the model from a local file
    (memfd copy for the memfd mode); there is no restore on a fixed host, so restore phases are null by design."""
    g = lambda f: [f(d) for d in rows]  # noqa: E731
    cold = {
        "n": len(rows),
        "weight_download": _dist(g(lambda d: d["phases_s"].get("weight_read")), "s"),
        "imports": _dist(g(lambda d: d["phases_s"].get("imports")), "s"),
        "ggml_backend_load": _dist(g(lambda d: d["phases_s"].get("ggml_backend_load")), "s"),
        "model_load_mmap": _dist(g(lambda d: d["phases_s"].get("model_load_mmap")), "s"),
        "context_init": _dist(g(lambda d: d["phases_s"].get("context_init")), "s"),
        "assets_load": _dist(g(lambda d: d["phases_s"].get("assets_load")), "s"),
        "warmup_inference": _dist(g(lambda d: d["phases_s"].get("warmup_inference")), "s"),
        "first_request_inference": _dist(
            g(lambda d: d["phases_s"].get("first_request_inference")), "s"
        ),
        "model_ready": _dist(g(lambda d: d["phases_s"].get("model_ready")), "s"),
        "first_request_total": _dist(g(lambda d: d["phases_s"].get("model_ready")), "s"),
        "report_max_memory_mb": _dist(g(lambda d: d["status_end_kb"]["VmHWM"] / 1024), "MB"),
    }
    warm_vals = [v for d in rows for vs in d["warm_s"].values() for v in vs]
    by_req: dict = {}
    for d in rows:
        for k, vs in d["warm_s"].items():
            by_req.setdefault(k, []).extend(vs)
    warm = {
        "n": len(warm_vals),
        "steady_state_inference": _dist(warm_vals, "s"),
        "steady_state_by_request": {k: _dist(v, "s") for k, v in sorted(by_req.items())},
        "forward_4096_tokens": _dist([x for d in rows for x in d["long_forward"]["seconds"]], "s"),
    }
    restore = {
        "n": 0,
        "report_restore_duration": None,
        "first_request_inference": None,
        "snapshot_init": None,
    }
    return {"cold": cold, "restore": restore, "warm": warm}


def uniform_conditions(rows: list[dict]) -> list[str]:
    """p2_result publishes one host, cpuset, thread count and request set per record from its first row. Refuse
    when the rows of one CPU identity disagree on any of them, so no first row stands for rows unlike it."""
    why = []
    for ident in sorted({d["_identity"] for d in rows}):
        own = [d for d in rows if d["_identity"] == ident]
        for name, f in (
            ("threads", lambda d: d.get("threads")),
            ("cpuset", lambda d: (d.get("host_before") or {}).get("sched_affinity")),
            ("machine", lambda d: (d.get("host_before") or {}).get("machine")),
            ("request ids", lambda d: sorted(d["answers"])),
        ):
            seen = {canon(f(d)) for d in own}
            if len(seen) != 1:
                why.append(f"{ident}: rows disagree on {name}: {sorted(seen)}")
    return why


def p2_result(
    rung: str,
    rows: list[dict],
    s: dict,
    verdict: dict,
    run_id: str,
    commit: str,
    mode: str,
    ident: str,
    x86: dict | None,
) -> dict:
    spec = RUNGS[rung]

    def sel(arm):
        return [
            d for d in rows if d["_arm"] == arm and d["_mode"] == mode and d["_identity"] == ident
        ]

    arms = {"p1": sel("p1"), spec["arm"]: sel(spec["arm"])}
    if spec["dormant"]:
        dorm = [d for d in rows if d["_arm"] == spec["dormant"] and d["_identity"] == ident]
        if dorm:
            arms[spec["dormant"]] = dorm
    cmp_ = s["rungs"][rung][f"{ident}|{mode}"]
    corr = s["rungs"][rung][f"{ident}|correctness"]
    host = rows[0]["host_before"]
    res = {
        "schema": "lkai-result/1",
        "experiment_id": f"G-general-{spec['experiment']}",
        "run_id": run_id,
        "status": verdict["status"],
        "primary_arm": spec["arm"],
        "hypothesis": verdict["hypothesis"],
        "changed_variable": {
            "name": spec["name"],
            "baseline": spec["baseline"],
            "experiment": spec["value"],
        },
        "commits": {
            "baseline": "70eab7be69f86ddeb8c64158d94df344b5180098",
            "experiment": commit,
            "repo_dirty_at_build": False,
        },
        "deployment": {
            "region": "fixed-host (not Lambda)",
            "az": None,
            "runtime": "python3.12 (P1's pinned Lambda image)",
            "runtime_version_arn": None,
            "architecture": "arm64" if host["machine"] == "aarch64" else "x86_64",
            "memory_mb": 10240,
            "snapstart": None,
            "functions": {
                "host": "fixed-host DevSpace (P1/H1 offline reference)",
                "identity": ident,
                "mode": mode,
                "cpuset": host.get("sched_affinity"),
                "threads": rows[0]["threads"],
                "note": "memory_mb is P1's Lambda setting, kept for the schema; the container has no "
                "memory limit (cgroup memory.max as recorded per load)",
            },
        },
        "model": {
            "id": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
            "revision": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
            "format": "GGUF Q8_0 (H512)",
        },
        "conditions": {
            "rounds": len(cmp_["rounds"]),
            "concurrency": 1,
            "request_ids": [*sorted(rows[0]["answers"]), "forward-4096"],
            "started": verdict["started"],
            "finished": verdict["finished"],
            "arms": sorted(arms),
            "mode": mode,
            "host_identity": ident,
            "note": "FIXED HOST, NOT LAMBDA LATENCY. One fresh container per load; arms interleaved in "
            "reversal-paired random orders (run_ladder.schedule: each arm before its control in exactly half the "
            "rounds); page cache primed before round 0; comparisons pair rounds (arm minus p1 in the same round).",
            "schedule_balance": cmp_.get("order"),
        },
        "classification": {
            "method": "fixed host: every load is a new process (cold); no restore exists; warm "
            "= the same process's later requests",
            "mismatches": 0,
        },
        "phases": {name: _phases(v) for name, v in arms.items() if v},
        "variance": {
            "paired_vs_p1": cmp_,
            "rule": "no change when |median paired delta| <= p1's per-round range "
            "or the paired deltas change sign",
        },
        "errors": {"count": verdict["errors"], "rows": verdict["error_rows"][:20]},
        "cost": {
            "billed_duration_s": 0.0,
            "total_usd": 0.0,
            "s3_request_usd": 0.0,
            "s3_gb_transferred": 0.0,
            "note": "fixed host already provisioned; no AWS calls",
        },
        "correctness": {
            "compared": corr["vs_run_p1_requests"]["compared"],
            "bit_identical": corr["vs_run_p1_requests"]["identical"],
            "choice_flips": corr["choice_flips_vs_p1"],
            "max_abs_delta": corr["max_abs_delta_vs_p1"],
            "vs_p1_reference": corr["vs_reference_requests"],
            "detail": corr,
        },
        "profiling_artifacts": [],
        "artifacts": {"ladder": "ladder.json", "probes": "probes.jsonl"},
        "reproduce": verdict["reproduce"],
        "conclusion": verdict["conclusion"],
        "general_verdict": verdict["general"],
    }
    if x86:
        res["x86_64"] = x86
    if spec["dormant"] and rung in s["dormant"]:
        res["dormant_check"] = s["dormant"][rung]
    return res


def guard(
    rung: str,
    s: dict,
    general: str,
    required: list[str] | None = None,
    min_rounds: int = MIN_ROUNDS,
    expected_failures: dict[str, dict] | None = None,
    control_nan: dict[str, list[str]] | None = None,
    reference_identities: list[str] | None = None,
) -> list[str]:
    """Why a 'general default' verdict is not allowed by the operator rule (empty when allowed). Fail-closed: the
    verdict needs complete, clean evidence on every required path, not merely the absence of a measured regression.

    required          "<identity>|<mode>" cells that must exist for this rung (e.g. arm64 local + memfd, x86 local)
    min_rounds        paired rounds every required cell must have
    expected_failures "<run>/<arm>" -> refusal spec (refusal_mismatch): by-design load failures; any other failure of
                      this rung's arm, dormant or p1 refuses
    control_nan       per identity, the non-finite answer fields (NaN or +/-Infinity) P1 itself is known to return
    reference_identities  identities with a published P1 reference: every answer there must be compared with it

    Refused when: a required cell is missing or short of rounds; any headline metric is missing, non-finite,
    unassessed or worse; the arm ran before its control more (or less) often than after; an unexpected load of the
    arm, its dormant check or p1 failed; any arm row has no same-round P1 row; any request or 4,096-token answer
    differs from the same round's P1, or the long forward did not fill the window, or repeats within a load were not
    identical; on a reference identity, any answer differs from the reference or was not compared with it; the arm
    returns a non-finite field P1 does not, or P1 returns one not declared in control_nan."""
    if general != "general default":
        return []
    why = []
    cells = s["rungs"].get(rung, {})
    for key in required or []:
        if key not in cells:
            why.append(f"{key}: required cell missing")
    timing_idents = {k.split("|")[0] for k in cells if not k.endswith("|correctness")}
    for ident in sorted(timing_idents):
        if f"{ident}|correctness" not in cells:
            why.append(f"{ident}: no correctness record")
    for key, v in sorted(cells.items()):
        if key.endswith("|correctness"):
            ident = key.split("|")[0]
            req, fwd = v["vs_run_p1_requests"], v["vs_run_p1_forward4096"]
            if v.get("unpaired_rows", 1):
                why.append(f"{key}: {v.get('unpaired_rows')} rows without a same-round P1 row")
            if not req["compared"] or req["identical"] != req["compared"]:
                why.append(f"{key}: answers differ from P1 ({req['identical']}/{req['compared']})")
            if not fwd["compared"] or fwd["identical"] != fwd["compared"]:
                why.append(
                    f"{key}: 4096-token answers differ from P1 ({fwd['identical']}/{fwd['compared']})"
                )
            if v.get("forward4096_not_full_window", 1):
                why.append(f"{key}: long forward did not fill the window")
            if not v["repeat_identical"]:
                why.append(f"{key}: repeats within a load were not identical")
            ref = v["vs_reference_requests"]
            if ref["differs"]:
                why.append(f"{key}: {ref['differs']} answers differ from P1's published reference")
            if ident in (reference_identities or []):
                answered = v.get("requests_answered", -1)
                if (
                    not answered
                    or ref["identical"] + ref["differs"] != answered
                    or ref["no_reference"]
                ):
                    why.append(
                        f"{key}: reference coverage {ref['identical'] + ref['differs']}/{answered} "
                        f"({ref['no_reference']} without a reference)"
                    )
            ctl = set((s.get("control_nan_fields") or {}).get(ident, []))
            extra = sorted(set(v["non_finite_fields"]) - ctl)
            if extra:
                why.append(f"{key}: non-finite answers P1 does not return: {extra}")
            undeclared = sorted(ctl - set((control_nan or {}).get(ident, [])))
            if undeclared:
                why.append(f"{key}: P1 returns undeclared non-finite answers: {undeclared}")
            continue
        if v["n"] < min_rounds:
            why.append(f"{key}: {v['n']} paired rounds, needs {min_rounds}")
        for name in v.get("missing", []):
            why.append(f"{key}: {name} missing")
        for name in v.get("non_finite", []):
            why.append(f"{key}: {name} non-finite")
        for name in HEADLINE:
            if name not in v["metrics"] and name not in v.get("missing", []) + v.get(
                "non_finite", []
            ):
                why.append(f"{key}: {name} not measured")
        for name, m in v["metrics"].items():
            if m["outcome"] == "worse":
                why.append(f"{key}: {name} worse ({m['delta_median']:+.4g} {m['unit']})")
            elif m["outcome"] == "unassessed":
                why.append(f"{key}: {name} unassessed")
        o = v.get("order") or {}
        if o.get("unknown", 1) or o.get("arm_first") != o.get("control_first"):
            why.append(f"{key}: order unbalanced vs p1 ({o})")
    arms = {RUNGS[rung]["arm"], RUNGS[rung]["dormant"], "p1"} - {None}
    for f in s.get("failed_loads", []):
        if f["arm"] in arms and refusal_mismatch(
            f, (expected_failures or {}).get(f"{f['run']}/{f['arm']}")
        ):
            why.append(
                f"{f['run']}/{f['arm']}: unexpected failed load (round {f.get('round')}, rc {f['rc']})"
            )
    return why


def refusal_mismatch(f: dict, spec: dict | None) -> str:
    """Why one failed load is not the declared refusal `spec` (empty when it is). A spec names the exit code and
    EITHER the probe's machine-readable `reason` (its LKAI_REFUSAL record) OR the exact final stderr line
    (`diagnostic`, for loads run before the probe recorded a reason)."""
    if not spec:
        return "not declared"
    if f.get("rc") != spec["rc"]:
        return f"rc {f.get('rc')} != {spec['rc']}"
    if f.get("mode") not in spec.get("modes", ["local"]):
        return f"mode {f.get('mode')!r} not declared"
    if "reason" in spec:
        got = (f.get("refusal") or {}).get("reason")
        if got != spec["reason"]:
            return f"reason {got!r} != {spec['reason']!r}"
        return ""
    last = ((f.get("stderr") or "").strip().splitlines() or [""])[-1]
    return (
        "" if last == spec["diagnostic"] else f"diagnostic {last[:120]!r} != {spec['diagnostic']!r}"
    )


def refusals_hold(s: dict, expected: dict[str, dict], rounds: int) -> list[str]:
    """An expected refusal is a claim too. For each declared "<run>/<arm>": exactly one failed load per declared mode
    in each round 0..rounds-1 (no duplicate, no missing round), every one matching the spec's rc and reason or exact
    diagnostic, and no successful row of that arm in that run."""
    why = []
    for key, spec in expected.items():
        run, arm = key.split("/")
        got = [f for f in s.get("failed_loads", []) if f["run"] == run and f["arm"] == arm]
        want = sorted((r, m) for r in range(rounds) for m in spec.get("modes", ["local"]))
        seen = sorted((f.get("round"), f.get("mode")) for f in got)
        if seen != want:
            why.append(f"{key}: refusals at (round, mode) {seen}, expected exactly {want}")
        for f in got:
            bad = refusal_mismatch(f, spec)
            if bad:
                why.append(f"{key}: round {f.get('round')} {f.get('mode')}: {bad}")
        ok = (s.get("rows_by_run_arm") or {}).get(key, 0)
        if ok:
            why.append(f"{key}: expected to refuse, but {ok} loads produced rows")
    return why


def failures_accounted(s: dict, expected: dict[str, dict]) -> list[str]:
    """Every failed load of every run is a declared refusal of its own kind; nothing fails silently."""
    return [
        f"{f['run']}/{f['arm']} round {f.get('round')} {f.get('mode')}: undeclared failed load ({bad})"
        for f in s.get("failed_loads", [])
        if (bad := refusal_mismatch(f, expected.get(f"{f['run']}/{f['arm']}")))
    ]


def status_holds(rung: str, s: dict, v: dict, primary: str) -> list[str]:
    """The P2 status a verdict publishes must follow from the primary identity's local-file cell."""
    m = (s["rungs"].get(rung, {}).get(f"{primary}|local") or {}).get("metrics", {})
    outcomes = {x["outcome"] for x in m.values()}
    want = (
        "regressed" if "worse" in outcomes else "improved" if "better" in outcomes else "no-change"
    )
    if not m:
        return [f"{primary}|local: no cell to derive status from"]
    return (
        []
        if v["status"] == want
        else [f"status {v['status']!r} but {primary}|local reads {want!r}"]
    )


def expected_refusals(vs: dict) -> dict[str, dict]:
    """Every rung's declared refusals, "<run>/<arm>" -> spec, from verdicts.json."""
    out: dict[str, dict] = {}
    for w in vs["rungs"].values():
        out |= w.get("expected_refusals", {})
    return out


def write_p2(out: str, rung: str, res: dict, rows: list[dict], s: dict) -> str:
    d = os.path.join(out, "p2", rung)
    os.makedirs(d, exist_ok=True)
    json.dump(s, open(os.path.join(d, "ladder.json"), "w"), indent=1, sort_keys=True)
    with open(os.path.join(d, "probes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()}, sort_keys=True) + "\n")
    json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1, sort_keys=True)
    return os.path.join(d, "results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--verdicts", help="verdicts.json: per rung status, general verdict, conclusion"
    )
    ap.add_argument("--commit", default="")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    rows = [d for r in a.runs for d in load_run(r)]
    s = summarize(rows)
    s["failed_loads"] = [f for r in a.runs for f in failed_loads(r)]
    s["loads"] = len(rows)
    s["rows_by_run_arm"] = {}
    for d in rows:
        k = f"{d['_run']}/{d['_arm']}"
        s["rows_by_run_arm"][k] = s["rows_by_run_arm"].get(k, 0) + 1
    s["control_nan_fields"] = {
        ident: correctness(rows, "p1", ident)["non_finite_fields"] for ident in s["identities"]
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, "ladder.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(a.out, "table.md"), "w").write(table_md(s))
    print(table_md(s))
    if not a.verdicts:
        return
    vs = json.load(open(a.verdicts))
    expected = expected_refusals(vs)
    bad = {
        r: guard(
            r,
            s,
            v["general"],
            required=v.get("required_cells", vs["required_cells"]),
            min_rounds=vs["min_rounds"],
            expected_failures=expected,
            control_nan=vs.get("control_nan_fields", {}),
            reference_identities=vs["reference_identities"],
        )
        + refusals_hold(s, v.get("expected_refusals", {}), vs["min_rounds"])
        + status_holds(r, s, v, vs["primary_identity"])
        for r, v in vs["rungs"].items()
    }
    bad["all runs"] = failures_accounted(s, expected) + uniform_conditions(rows)
    if any(bad.values()):
        raise SystemExit(
            f"verdicts refused: {json.dumps({k: w for k, w in bad.items() if w}, indent=1)}"
        )
    arm_rows = [d for d in rows if d["_machine"] == "aarch64"]
    for rung, v in vs["rungs"].items():
        x86 = {
            k: val
            for k, val in s["rungs"].get(rung, {}).items()
            if not k.startswith(vs["primary_identity"])
        }
        x86["failed_loads"] = [
            f for f in s["failed_loads"] if f["arm"] in (RUNGS[rung]["arm"], RUNGS[rung]["dormant"])
        ]
        mine = v.get("expected_refusals", {})
        errs = [
            f
            for f in x86["failed_loads"]
            if refusal_mismatch(f, mine.get(f"{f['run']}/{f['arm']}"))
        ]
        verdict = v | {
            "started": vs["started"],
            "finished": vs["finished"],
            "reproduce": vs["reproduce"],
            "errors": len(errs),
            "error_rows": errs,
        }
        res = p2_result(
            rung, arm_rows, s, verdict, a.run_id, v["commit"], "local", vs["primary_identity"], x86
        )
        res["commits"]["ladder"] = a.commit
        print("wrote", write_p2(a.out, rung, res, rows, s))


if __name__ == "__main__":
    main()
