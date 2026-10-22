"""E14 results: ctl vs rows on Lambda, plus the offline and Lambda-log evidence, as one lkai-result/1 file.

  python summarize_e14.py <run dir> --reference baseline/reference/outputs.json --meta e14/meta.json \
      --baseline-commit <P1 sha> --experiment-commit <this branch sha> \
      [--offline <offline out dir>] [--lambda-concat lambda-concat.json] --out <run dir>/results.json

<run dir> holds cold/ and, optionally, restore/ (each an e14bench.py output: conditions.json, invocations.jsonl,
logs/). Phases come from P1's baseline/summarize.py (arm_phases, costs, compare), merged over both legs per arm.

The primary metric is warm `steady_state_inference` (P1's engine timer, latency_ms), and warm REPORT Duration beside
it. Arms are compared only within one observed CPU part (MIDR part ID; no processor name is inferred from it), and
within each (round, CPU part, request) cell: warm passes are interleaved request by request, so each cell holds
both arms' invocations from the same minutes on environments created in the same round. A cell's delta is
median(rows) - median(ctl); its outcome is judged against the control's own spread in that cell.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import summarize as p1s  # noqa: E402

ARMS = ("ctl", "rows")
UNKNOWN = "unknown"


def _load_leg(d: str) -> tuple[dict, list[dict]]:
    """P1's load_run, also reading a committed invocations.jsonl.gz."""
    import gzip

    if os.path.exists(os.path.join(d, "invocations.jsonl")):
        return p1s.load_run(d)
    cond = json.load(open(os.path.join(d, "conditions.json")))
    with gzip.open(os.path.join(d, "invocations.jsonl.gz"), "rt") as f:
        return cond, [json.loads(line) for line in f if line.strip()]


def load(run: str) -> tuple[dict, list[dict], dict]:
    """Every leg's rows, the first leg's conditions, and {leg: conditions}."""
    legs, rows = {}, []
    for leg in ("cold", "restore"):
        d = os.path.join(run, leg)
        if os.path.isdir(d):
            cond, rs = _load_leg(d)
            legs[leg] = cond
            rows.extend(r | {"leg": leg} for r in rs)
    if not legs:
        raise SystemExit(f"{run}: no cold/ or restore/ leg")
    return next(iter(legs.values())), rows, legs


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def _engine(r: dict) -> float | None:
    return r["latency_ms"] / 1000 if r.get("latency_ms") is not None else None


def _report(r: dict) -> float | None:
    return r["report"]["Duration"] / 1000 if "Duration" in r["report"] else None


def _part(r: dict) -> str:
    return r.get("cpu_part") or UNKNOWN


def cell_table(rows: list[dict], metric, key) -> list[dict]:
    """Per cell (key(r)), both arms' n / median / range, delta and outcome against the control's spread."""
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {a: [] for a in ARMS})
    for r in rows:
        v = metric(r)
        if v is not None and r["arm"] in ARMS:
            cells[key(r)][r["arm"]].append(v)
    out = []
    for k in sorted(cells, key=str):
        c, w = cells[k]["ctl"], cells[k]["rows"]
        row: dict = {"cell": list(k), "n": [len(c), len(w)]}
        if c and w:
            mc, mw = statistics.median(c), statistics.median(w)
            spread = max(c) - min(c) if len(c) > 1 else None
            d = mw - mc
            row |= {
                "ctl_p50": round(mc, 6),
                "rows_p50": round(mw, 6),
                "delta_s": round(d, 6),
                "delta_pct": round(100 * d / mc, 3) if mc else None,
                "ctl_range_s": round(spread, 6) if spread is not None else None,
            }
            if spread is None:
                row["outcome"] = "unassessed"
            elif abs(d) <= spread:
                row["outcome"] = "within-control-variation"
            else:
                row["outcome"] = (
                    "negative-beyond-control-variation"
                    if d < 0
                    else "positive-beyond-control-variation"
                )
        else:
            row["outcome"] = "unmatched"
        out.append(row)
    return out


def part_summary(cells: list[dict]) -> dict:
    """Per CPU part (cell[1]): counts of cell outcomes, delta_pct range, and the median delta_pct."""
    by: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by[str(c["cell"][1])].append(c)
    out = {}
    for part, cs in sorted(by.items()):
        assessed = [c for c in cs if c.get("delta_pct") is not None]
        counts: dict[str, int] = defaultdict(int)
        for c in cs:
            counts[c["outcome"]] += 1
        ds = [c["delta_pct"] for c in assessed]
        out[part] = {
            "cells": len(cs),
            "outcomes": dict(sorted(counts.items())),
            "delta_pct_median": round(statistics.median(ds), 3) if ds else None,
            "delta_pct_range": [min(ds), max(ds)] if ds else None,
            "delta_signs": sorted({(d > 0) - (d < 0) for d in ds}),
        }
        if part == UNKNOWN:
            out[part]["note"] = "missing CPU part: counted, never a comparison stratum"
    return out


def per_part_request(rows: list[dict], metric) -> dict:
    """Pooled over rounds, per CPU part and request: both arms' p50 and the delta. Supporting, not primary."""
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {a: [] for a in ARMS})
    for r in rows:
        v = metric(r)
        if v is not None and r["arm"] in ARMS:
            acc[(_part(r), r["request_id"])][r["arm"]].append(v)
    out: dict = defaultdict(dict)
    for (part, rid), d in sorted(acc.items()):
        c, w = d["ctl"], d["rows"]
        e: dict = {"n": [len(c), len(w)]}
        if c and w:
            mc, mw = statistics.median(c), statistics.median(w)
            e |= {
                "ctl_p50": round(mc, 6),
                "rows_p50": round(mw, 6),
                "delta_pct": round(100 * (mw / mc - 1), 3),
            }
        out[part][rid] = e
    return dict(out)


def env_bootstrap(rows: list[dict], metric, part: str, reps: int = 2000, seed: int = 14) -> dict:
    """Per request on one CPU part: the mean over rounds of median_env(rows)/median_env(ctl) - 1, where each
    environment contributes the median of its own warm invocations. 95 % interval from resampling environments within
    each (round, arm). Environments, not invocations, are the independent units."""
    import random

    rng = random.Random(seed)
    out = {}
    for q in sorted({r["request_id"] for r in rows}):
        per: dict[str, dict[tuple, list[float]]] = {a: defaultdict(list) for a in ARMS}
        for r in rows:
            v = metric(r)
            if r["request_id"] == q and _part(r) == part and v is not None and r["arm"] in ARMS:
                per[r["arm"]][(r["round"], (r.get("lkai") or {}).get("env_id"))].append(v)
        med = {a: {k: statistics.median(v) for k, v in per[a].items()} for a in ARMS}
        rounds = sorted({k[0] for a in ARMS for k in med[a]})

        def stat(s: dict[str, dict[int, list[float]]], rounds: list[int] = rounds) -> float | None:
            ds = [
                statistics.median(s["rows"][rd]) / statistics.median(s["ctl"][rd]) - 1
                for rd in rounds
                if s["ctl"].get(rd) and s["rows"].get(rd)
            ]
            return statistics.fmean(ds) if ds else None

        base = {
            a: {rd: [v for k, v in med[a].items() if k[0] == rd] for rd in rounds} for a in ARMS
        }
        pt = stat(base)
        if pt is None:
            continue
        bs = sorted(
            x
            for x in (
                stat(
                    {
                        a: {
                            rd: rng.choices(vs, k=len(vs)) if vs else []
                            for rd, vs in base[a].items()
                        }
                        for a in ARMS
                    }
                )
                for _ in range(reps)
            )
            if x is not None
        )
        out[q] = {
            "environments": [len(med["ctl"]), len(med["rows"])],
            "delta_pct": round(100 * pt, 3),
            "ci95_pct": [
                round(100 * bs[int(0.025 * len(bs))], 3),
                round(100 * bs[int(0.975 * len(bs)) - 1], 3),
            ],
        }
    return out


def host_records(run: str, legs: dict) -> list[dict]:
    """The E14 first-invocation records from the saved logs: one per environment."""
    import gzip

    recs = []
    for leg in legs:
        d = os.path.join(run, leg, "logs")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            arm = f.removesuffix(".jsonl.gz").rsplit("-", 1)[-1]
            with gzip.open(os.path.join(d, f), "rt") as fh:
                for line in fh:
                    m = json.loads(line)["message"]
                    if m.startswith("E14 "):
                        recs.append(json.loads(m[4:]) | {"arm": arm, "leg": leg})
    return recs


def host_variability(rows: list[dict], recs: list[dict]) -> dict:
    ids: dict[tuple, int] = defaultdict(int)
    kernels, cpus, pythons, cg, multi = set(), set(), set(), set(), 0
    for rec in recs:
        h = rec["at_first_invoke"]
        ids[
            (
                rec["arm"],
                rec["leg"],
                h["cpu"]["part"],
                h["cpu"]["variant"],
                h["cpu"]["revision"],
                h["hwcap"],
                h["hwcap2"],
                ",".join(sorted(rec.get("ggml_cpu", {}))),
            )
        ] += 1
        kernels.add(h["kernel"])
        cpus.add(f"{h['os_cpu_count']}/{h['sched_affinity']}")
        pythons.add(h["python"])
        cg.add(json.dumps(h["cgroup"], sort_keys=True))
        if rec["at_init"]["cpu"]["part"] != h["cpu"]["part"]:
            multi += 1
    lib_sha: dict[str, set[str]] = defaultdict(set)
    for rec in recs:
        for lib, sha in rec.get("ggml_cpu", {}).items():
            lib_sha[f"{rec['arm']}:{lib}"].add(sha)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[f"{r['arm']}:{r['expected_class']}:{_part(r)}"] += 1
    return {
        "identity_label": "MIDR CPU part ID as read; no processor name or generation is inferred from it",
        "observed": {
            "environments": len(recs),
            "identities": [
                {
                    "arm": k[0],
                    "leg": k[1],
                    "cpu_part": k[2],
                    "cpu_variant": k[3],
                    "cpu_revision": k[4],
                    "hwcap": k[5],
                    "hwcap2": k[6],
                    "ggml_cpu_mapped": k[7],
                    "environments": n,
                }
                for k, n in sorted(ids.items())
            ],
            "kernels": sorted(kernels),
            "cpu_count_over_affinity": sorted(cpus),
            "python": sorted(pythons),
            "cgroup": [json.loads(x) for x in sorted(cg)],
            "restored_envs_whose_cpu_part_differs_from_snapshot_host": multi,
            "ggml_cpu_sha256_by_arm": {k: sorted(v) for k, v in sorted(lib_sha.items())},
            "row_counts_by_arm_class_part": dict(sorted(counts.items())),
        },
        "unknown": {
            "availability_zone": "not observable from inside the function",
            "physical_host": "not observable",
            "processor_generation": "not inferred: only the MIDR part ID, variant, revision and HWCAPs are reported",
        },
    }


def investigation(
    warm: list[dict], offline: dict, lam: dict, sigprof: list[dict], recs_p3: list[dict]
) -> dict:
    """The concat-share question, from data: Lambda's P3 share x CPU seconds per request, offline thread-0 concat
    time, both arms' wall time, the patch's Lambda effect, and SIGPROF thread delivery in Lambda."""
    per_req: dict = {}
    by_req: dict[str, list[dict]] = defaultdict(list)
    for r in recs_p3:
        if r["class"] == "warm" and r["cpu_part"] == "0xd40" and r["samples"]:
            by_req[r["request"]].append(r)
    pooled = per_part_request(warm, _engine).get("0xd40", {})
    for q, rs in sorted(by_req.items()):
        est = statistics.fmean(r["concat"] / r["samples"] * r["cpu_s"] for r in rs)
        eng = statistics.fmean(r["engine_s"] for r in rs)
        o = offline["per_request"].get(q, {})
        per_req[q] = {
            "p3_lambda_invocations": len(rs),
            "p3_lambda_concat_share_x_cpu_s_ms": round(1000 * est, 1),
            "p3_lambda_engine_ms": round(1000 * eng, 1),
            "offline_thread0_concat_ms": (o.get("concat_probe") or {}).get(
                "thread0_ms_per_request"
            ),
            "offline_ctl_p50_ms": round(1000 * o["ctl_p50_s"], 1) if o else None,
            "lambda_rows_vs_ctl_pct_0xd40": (pooled.get(q) or {}).get("delta_pct"),
        }
    sp_n = sum(x["samples"] for x in sigprof)
    sp_main = sum(x["samples"] * x["main_thread_share"] for x in sigprof)
    return {
        "per_request_0xd40": per_req,
        "lambda_concat_from_p3": {
            k: lam[k] for k in ("by_class_part", "warm_by_part_origin", "warm_by_part_n")
        },
        "sigprof_delivery_in_lambda": {
            "invocations": len(sigprof),
            "environments": len({x["log_stream"] for x in sigprof}),
            "cpu_parts": sorted({x["cpu_part"] for x in sigprof}),
            "kernels": sorted({x["kernel"] for x in sigprof}),
            "threads": sorted({x["threads"] for x in sigprof}),
            "samples": sp_n,
            "main_thread_share_pooled": round(sp_main / sp_n, 4) if sp_n else None,
            "main_thread_share_range": [
                min(x["main_thread_share"] for x in sigprof),
                max(x["main_thread_share"] for x in sigprof),
            ]
            if sigprof
            else None,
            "fair_share": sigprof[0]["fair_share"] if sigprof else None,
        },
    }


def summarize(
    run: str, ref: str, meta: dict, baseline: str, experiment: str, extra: dict | None = None
) -> dict:
    cond, rows, legs = load(run)
    m = cond["build_manifest"]
    arms = {}
    snaps = {}
    for leg in legs:
        snaps |= p1s.snapshot_inits(os.path.join(run, leg))
    for arm in ARMS:
        fn = cond["functions"][arm]["FunctionName"]
        arms[arm] = p1s.arm_phases([r for r in rows if r["arm"] == arm], snaps.get(fn, []))
    warm = ok_rows(rows, "warm")
    eng = cell_table(warm, _engine, lambda r: (r["round"], _part(r), r["request_id"], r["leg"]))
    rep = cell_table(warm, _report, lambda r: (r["round"], _part(r), r["request_id"], r["leg"]))
    recs = host_records(run, legs)
    errs = [
        {
            "arm": r["arm"],
            "leg": r["leg"],
            "round": r["round"],
            "wave": r["wave"],
            "error": r["error"],
        }
        for r in rows
        if r["error"]
    ]
    mism = [
        {
            "arm": r["arm"],
            "leg": r["leg"],
            "round": r["round"],
            "expected": r["expected_class"],
            "observed": r["observed_class"],
        }
        for r in rows
        if r["class_mismatch"]
    ]
    cost: dict = {}
    p1m = meta["p1_build_manifest"]
    for leg, c in legs.items():
        c = c | {"build_manifest": p1m}  # costs() reads the GGUF size from P1's build manifest
        part = p1s.costs(c, [r for r in rows if r["leg"] == leg], None)
        for k, v in part.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cost[k] = round(cost.get(k, 0) + v, 6)
    cost["prices"] = p1s.PRICES
    cost["note"] = (
        "sum over legs of baseline/summarize.costs; SnapStart cache at the 3-hour minimum per version"
    )
    ref_out = json.load(open(ref))["outputs"]
    corr = p1s.compare([r for r in rows if r["arm"] == "rows"], ref_out) | {
        "reference": "baseline/reference/outputs.json",
        "arm": "rows (the experiment arm) against P1's reference, same request and CPU part",
    }
    corr["ctl_vs_reference"] = p1s.compare([r for r in rows if r["arm"] == "ctl"], ref_out)
    corr["ctl_vs_rows_same_run"] = ctl_vs_rows(rows)
    f0 = cond["functions"]["ctl"]
    res = {
        "schema": "lkai-result/1",
        "experiment_id": meta["experiment_id"],
        "run_id": os.path.basename(os.path.normpath(run)),
        "status": meta["status"],
        "primary_arm": "rows",
        "hypothesis": meta["hypothesis"],
        "changed_variable": meta["changed_variable"],
        "commits": {"experiment": experiment, "baseline": baseline, "repo_dirty_at_build": False},
        "deployment": {
            "region": cond["region"],
            "functions": {a: f["FunctionName"] for a, f in cond["functions"].items()},
            "runtime": f0["Runtime"],
            "runtime_version_arn": cond.get("runtime_version_arn"),
            "architecture": f0["Architectures"][0],
            "memory_mb": f0["MemorySize"],
            "ephemeral_storage_mb": f0["EphemeralStorage"]["Size"],
            "timeout_s": f0["Timeout"],
            "snapstart": f0["SnapStart"],
            "handler": f0["Handler"],
            "code_sha256": {a: m["zips"][a]["sha256"] for a in ARMS},
            "p1_zip_sha256": m["p1_zip_sha256"],
            "environment": f0["environment"],
            "az": None,
        },
        "model": {
            k: p1m["model"][k] for k in ("id", "revision", "format", "gguf_bytes", "gguf_sha256")
        },
        "conditions": {
            "rounds": sum(c["rounds"] for c in legs.values()),
            "concurrency": cond["concurrency"],
            "legs": {
                leg: {
                    k: c.get(k)
                    for k in (
                        "rounds",
                        "warm_passes",
                        "warm_order",
                        "started",
                        "finished",
                        "read_timeout_s",
                    )
                }
                for leg, c in legs.items()
            },
            "warm_per_arm_per_round": cond["warm_per_arm_per_round"],
            "request_ids": [r["id"] for r in cond["requests"]["requests"]],
            "started": min(c["started"] for c in legs.values()),
            "finished": max(c.get("finished") or "" for c in legs.values()) or None,
            "client": cond["client"],
            "invoke_client": cond.get("invoke_client"),
        },
        "classification": {
            "method": "expected class from the harness (new $LATEST config -> cold; new published version -> "
            "restore; reused environment -> warm), checked against REPORT",
            **p1s.classify(rows),
            "mismatch_rows": mism,
        },
        "phases": arms,
        "variance": {a: p1s.round_medians([r for r in rows if r["arm"] == a]) for a in ARMS},
        "errors": {"count": len(errs), "rows": errs},
        "cost": cost,
        "correctness": corr,
        "profiling_artifacts": meta.get("profiling_artifacts", []),
        "artifacts": meta["artifacts"],
        "reproduce": meta["reproduce"],
        "conclusion": meta["conclusion"],
        "warm_comparison": {
            "method": __doc__.split("\n\n", 2)[2].strip(),
            "engine_cells": eng,
            "engine_by_part": part_summary(eng),
            "report_duration_cells": rep,
            "report_duration_by_part": part_summary(rep),
            "engine_pooled_by_part_request": per_part_request(warm, _engine),
            "engine_env_bootstrap": {
                f"{leg}:{p}": env_bootstrap([r for r in warm if r["leg"] == leg], _engine, p)
                for leg in legs
                for p in sorted({_part(r) for r in warm if r["leg"] == leg})
                if p != UNKNOWN
            },
        },
        "host_variability": host_variability(rows, recs),
    }
    for k in ("offline_not_lambda", "lambda_concat_from_p3", "investigation"):
        if k in meta:
            res[k] = meta[k]
    if extra:
        res["offline_not_lambda"] = extra["offline"]
        res["investigation"] = investigation(
            warm, extra["offline"], extra["lambda_concat"], extra["sigprof"], extra["p3_records"]
        )
    return res


def ctl_vs_rows(rows: list[dict]) -> dict:
    """Every rows answer against the ctl answers for the same request and CPU part in this run."""
    ctl: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["arm"] == "ctl" and not r["error"] and r.get("answers"):
            ctl[(r["request_id"], _part(r))].append(r["answers"])
    same = diff = missing = 0
    for r in rows:
        if r["arm"] != "rows" or r["error"] or not r.get("answers"):
            continue
        base = ctl.get((r["request_id"], _part(r)))
        if not base:
            missing += 1
        elif all(r["answers"] == b for b in base):
            same += 1
        else:
            diff += 1
    return {"bit_identical": same, "different": diff, "no_ctl_answer_for_request_and_part": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--experiment-commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--p1-manifest", required=True, help="P1's build/manifest.json (model size and identity)"
    )
    ap.add_argument("--offline-summary", help="offline_summary.py output (the probe run)")
    ap.add_argument("--lambda-concat", help="analyse_lambda.py output")
    ap.add_argument(
        "--p3-run", help="P3 run dir (invocations.jsonl.gz, logs/lkai-p3-sampler.jsonl.gz)"
    )
    ap.add_argument("--sigprof-dir", help="directory of sigprof_probe.py Lambda responses")
    a = ap.parse_args()
    extra = None
    if a.offline_summary and a.lambda_concat and a.p3_run and a.sigprof_dir:
        import glob
        from pathlib import Path

        sys.path.insert(0, HERE)
        import analyse_lambda

        rows3, logs3 = analyse_lambda.load(Path(a.p3_run))
        extra = {
            "offline": json.load(open(a.offline_summary)),
            "lambda_concat": json.load(open(a.lambda_concat)),
            "sigprof": [
                json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.sigprof_dir, "*.json")))
            ],
            "p3_records": analyse_lambda.join(rows3, logs3),
        }
    meta = json.load(open(a.meta))
    meta["p1_build_manifest"] = json.load(open(a.p1_manifest))
    res = summarize(a.run, a.reference, meta, a.baseline_commit, a.experiment_commit, extra)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
