"""Turn a bench.py run directory into results.json (and the correctness reference). Standard library only.

  python summarize.py results/baseline/<run-id>                    # writes <run-id>/results.json
  python summarize.py <run> --write-reference reference/outputs.json   # the baseline's answers become the reference
  python summarize.py <experiment-run> --reference reference/outputs.json --baseline-commit <sha>

results.json follows lkai-result/1 (results/schema.md). P2's dashboard ingests it; until P2 lands it is the record.
Every phase gets the same distribution block {n, p50, p90, p99, min, max, mean, stdev, histogram}. Percentiles are
nearest-rank on the sorted sample, so p99 of fewer than 100 samples is the maximum; n is always shown beside them.
Run-to-run variance is the spread of per-round medians (`rounds`), since each round is a fresh set of environments.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phases  # noqa: E402

# us-east-2 public prices (USD), 2026-10. arm64 duration per GB-s; SnapStart (Python) cache per GB-s and restore per GB;
# requests per million; S3 Standard GET per 1,000. Data transfer S3 -> Lambda in-region is free.
PRICES = {
    "arm64_gb_s": 0.0000133334,
    "request": 0.20 / 1e6,
    "snapstart_cache_gb_s": 0.0000015046,
    "snapstart_restore_gb": 0.0001397998,
    "s3_get_per_1000": 0.0004,
    "s3_head_per_1000": 0.0004,
    "s3_standard_gb_month": 0.023,
    "source": "aws.amazon.com/lambda/pricing, aws.amazon.com/s3/pricing (us-east-2)",
}
INIT_PHASES = (
    "runtime_to_import",
    "imports",
    "weight_download",
    "ggml_backend_load",
    "model_load_mmap",
    "context_init",
    "assets_load",
    "warmup_inference",
    "model_ready",
    "import_total",
)
CHUNK = 64 * 1024 * 1024  # examples/lambda/app/loader.py


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def histogram(xs: list[float], bins: int = 12) -> dict:
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return {"edges": [lo, hi], "counts": [len(xs)]}
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        counts[min(bins - 1, int((x - lo) / w))] += 1
    return {"edges": [round(lo + i * w, 6) for i in range(bins + 1)], "counts": counts}


def dist(xs: list[float], unit: str = "s") -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    r = lambda v: round(v, 6)  # noqa: E731
    return {
        "unit": unit,
        "n": len(xs),
        "p50": r(pct(xs, 50)),
        "p90": r(pct(xs, 90)),
        "p99": r(pct(xs, 99)),
        "min": r(min(xs)),
        "max": r(max(xs)),
        "mean": r(statistics.fmean(xs)),
        "stdev": r(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
        "histogram": histogram(xs),
    }


def load_run(run: str) -> tuple[dict, list[dict]]:
    cond = json.load(open(os.path.join(run, "conditions.json")))
    rows = [
        json.loads(line) for line in open(os.path.join(run, "invocations.jsonl")) if line.strip()
    ]
    return cond, rows


def snapshot_inits(run: str, versions: dict[str, set[str]] | None = None) -> dict[str, list[dict]]:
    """before_snapshot records from the saved CloudWatch logs: the init phases of each published version's
    snapshot (the init that ran inside publish_version, which no invocation's log tail shows).

    Lambda's log stream names carry the version (`2026/10/20/[7]...`); with `versions` ({function: {"3", ...}}) only
    the versions this run published count. Lambda may run more than one snapshot init per version (one per CPU
    generation it prepares), so each record carries its version and the count per version is reported."""
    out: dict[str, list[dict]] = defaultdict(list)
    d = os.path.join(run, "logs")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        fn = f.removesuffix(".jsonl.gz")
        seen = set()
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                e = json.loads(line)
                stream = e.get("logStreamName", "")
                ver = stream.split("[", 1)[1].split("]", 1)[0] if "[" in stream else None
                if versions is not None and ver not in versions.get(fn, set()):
                    continue
                for rec in phases.parse_lkai(e["message"]):
                    if rec.get("kind") == "before_snapshot" and rec.get("wall_ns") not in seen:
                        seen.add(rec.get("wall_ns"))
                        out[fn].append(rec | {"version": ver})
    return out


def classify(rows: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[f"{r['expected_class']}->{r['observed_class']}"] += 1
    return {
        "expected_vs_observed": dict(sorted(counts.items())),
        "mismatches": sum(r["class_mismatch"] for r in rows),
    }


def by(rows, **kv):
    return [r for r in rows if all(r.get(k) == v for k, v in kv.items())]


def arm_phases(rows: list[dict], snaps: list[dict]) -> dict:
    """Per start class, the per-phase distributions for one arm. Only invocations whose observed class matches the
    expected one count; mismatches are reported under errors and classification."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    out: dict = {}
    for cls in ("cold", "restore", "warm"):
        rs = [r for r in ok if r["expected_class"] == cls]
        if not rs:
            continue
        rep = lambda k, rs=rs: [r["report"].get(k, 0) / 1000 for r in rs if k in r["report"]]  # noqa: E731
        d = {
            "n": len(rs),
            "client_round_trip": dist([r["rt_s"] for r in rs]),
            "engine_evaluate": dist(
                [r["latency_ms"] / 1000 for r in rs if r.get("latency_ms") is not None]
            ),
            "report_duration": dist(rep("Duration")),
            "report_billed_duration": dist(rep("Billed Duration")),
            "report_max_memory_mb": dist(
                [r["report"].get("Max Memory Used") for r in rs], unit="MB"
            ),
        }
        inv = [r["lkai"]["inv"] for r in rs if r.get("lkai")]
        if inv:
            d["handler_total"] = dist([i["handler_s"] for i in inv])
            d["wait_for_load"] = dist([i["wait_load_s"] for i in inv])
        if cls == "cold":
            d["report_init_duration"] = dist(rep("Init Duration"))
            inits = [r["lkai"]["init"] for r in rs if r.get("lkai") and "init" in r["lkai"]]
            for ph in INIT_PHASES:
                d[ph] = dist([i[ph] for i in inits if ph in i])
            d["first_request_inference"] = d.pop("engine_evaluate")
            d["weight_download_mib_s"] = dist(
                [
                    r["lkai"]["counters"].get("weight_mib_s")
                    for r in rs
                    if r.get("lkai") and "counters" in r["lkai"]
                ],
                unit="MiB/s",
            )
        elif cls == "restore":
            d["report_restore_duration"] = dist(rep("Restore Duration"))
            d["report_billed_restore_duration"] = dist(rep("Billed Restore Duration"))
            d["restore_to_invoke"] = dist([i.get("restore_to_invoke_s") for i in inv])
            d["first_request_inference"] = d.pop("engine_evaluate")
            d["first_request_total"] = dist(
                [
                    (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
                    for r in rs
                ]
            )
            if snaps:  # the snapshot's own init, measured inside publish_version
                d["snapshot_init"] = {
                    ph: dist([s["init"][ph] for s in snaps if ph in s.get("init", {})])
                    for ph in INIT_PHASES
                }
        else:
            d["steady_state_inference"] = d.pop("engine_evaluate")
            per_req = defaultdict(list)
            for r in rs:
                per_req[r["request_id"]].append(r["latency_ms"] / 1000)
            d["steady_state_by_request"] = {k: dist(v) for k, v in sorted(per_req.items())}
        if cls == "cold":
            d["first_request_total"] = dist(
                [
                    (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
                    for r in rs
                ]
            )
        # Lambda picks the Graviton generation per environment; the engine phases differ ~1.7x between them.
        d["engine_by_cpu_part"] = by_cpu(
            rs, lambda r: r["latency_ms"] / 1000 if r.get("latency_ms") is not None else None
        )
        out[cls] = d
    return out


def round_medians(rows: list[dict]) -> dict:
    """Median per round of the headline numbers, and the spread of those medians (run-to-run variance)."""
    out: dict = {}
    keys = {
        "cold": lambda r: (
            (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
        ),
        "restore": lambda r: (
            (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
        ),
        "warm": lambda r: r["latency_ms"] / 1000,
    }
    for cls, f in keys.items():
        per = defaultdict(list)
        for r in rows:
            if r["expected_class"] == cls and not r["error"] and not r["class_mismatch"]:
                per[r["round"]].append(f(r))
        meds = [statistics.median(v) for _, v in sorted(per.items())]
        if meds:
            out[cls] = {
                "per_round_median_s": [round(m, 4) for m in meds],
                "median_of_medians_s": round(statistics.median(meds), 4),
                "range_s": round(max(meds) - min(meds), 4),
                "stdev_s": round(statistics.stdev(meds), 4) if len(meds) > 1 else 0.0,
                "cv": round(statistics.stdev(meds) / statistics.fmean(meds), 4)
                if len(meds) > 1
                else 0.0,
            }
    return out


def overhead(rows: list[dict]) -> dict:
    """Instrumented minus REPORT-only on Lambda's own numbers, matched by CPU part.

    Lambda picks the Graviton generation per environment and a forward is ~1.7x slower on Graviton2 than on
    Graviton3, so an unmatched comparison mostly measures the CPU mix. Each metric is compared within every CPU part
    both arms landed on; `pooled` weights those per-part deltas by the smaller arm's sample count."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    rep = lambda k: lambda r: r["report"].get(k, 0) / 1000 if k in r["report"] else None  # noqa: E731
    tot = lambda k: lambda r: (r["report"].get(k, 0) + r["report"].get("Duration", 0)) / 1000  # noqa: E731
    metrics = {
        "cold": {
            "report_init_duration": rep("Init Duration"),
            "report_duration": rep("Duration"),
            "first_request_total": tot("Init Duration"),
        },
        "restore": {
            "report_restore_duration": rep("Restore Duration"),
            "report_duration": rep("Duration"),
            "first_request_total": tot("Restore Duration"),
        },
        "warm": {
            "report_duration": rep("Duration"),
            "steady_state_inference": lambda r: (
                r["latency_ms"] / 1000 if r.get("latency_ms") else None
            ),
        },
    }
    out: dict = {}
    for cls, ms in metrics.items():
        rs = [r for r in ok if r["expected_class"] == cls]
        cpus = sorted(
            {r.get("cpu_part") for r in rs if r["arm"] == "inst"}
            & {r.get("cpu_part") for r in rs if r["arm"] == "rep"},
            key=str,
        )
        cls_out: dict = {}
        for name, f in ms.items():
            per, wsum, wdelta = {}, 0, 0.0
            for cpu in cpus:
                a = [
                    v
                    for r in rs
                    if r["arm"] == "inst" and r.get("cpu_part") == cpu
                    if (v := f(r)) is not None
                ]
                b = [
                    v
                    for r in rs
                    if r["arm"] == "rep" and r.get("cpu_part") == cpu
                    if (v := f(r)) is not None
                ]
                if not a or not b:
                    continue
                ma, mb = statistics.median(a), statistics.median(b)
                per[str(cpu)] = {
                    "inst_p50": round(ma, 6),
                    "rep_p50": round(mb, 6),
                    "delta_p50_s": round(ma - mb, 6),
                    "delta_pct": round(100 * (ma - mb) / mb, 2) if mb else None,
                    "n": [len(a), len(b)],
                }
                w = min(len(a), len(b))
                wsum += w
                wdelta += w * (ma - mb)
            if per:
                cls_out[name] = {
                    "by_cpu_part": per,
                    "pooled_delta_p50_s": round(wdelta / wsum, 6),
                    "pooled_weight": wsum,
                }
        if cls_out:
            out[cls] = cls_out
    unmatched = sorted({f"{r['arm']}:{r['expected_class']}:{r.get('cpu_part')}" for r in ok})
    out["method"] = (
        "median(inst) - median(rep) within each CPU part both arms ran on; Lambda REPORT numbers "
        "and the in-function engine latency, which both arms emit"
    )
    out["cells"] = unmatched
    return out


def by_cpu(rows: list[dict], f) -> dict:
    per = defaultdict(list)
    for r in rows:
        v = f(r)
        if v is not None:
            per[str(r.get("cpu_part"))].append(v)
    return {k: dist(v) for k, v in sorted(per.items())}


def _snap_inits_estimate(cond: dict, snaps: dict[str, list[dict]]) -> int | None:
    """Snapshot inits across both arms: observed on the instrumented arm, the same per version assumed for the
    REPORT-only arm (which records none). None when no record was found."""
    inst = cond["functions"].get("inst", {}).get("FunctionName")
    if not inst or not snaps.get(inst):
        return None
    return len(snaps[inst]) * len(cond["arms"])


def costs(cond: dict, rows: list[dict], snap_inits: int | None = None) -> dict:
    mem_gb = cond["functions"][cond["arms"][0]]["MemorySize"] / 1024
    billed_s = sum(r["report"].get("Billed Duration", 0) for r in rows) / 1000
    billed_restore_s = sum(r["report"].get("Billed Restore Duration", 0) for r in rows) / 1000
    restores = sum(1 for r in rows if "Restore Duration" in r["report"])
    colds = sum(1 for r in rows if "Init Duration" in r["report"])
    size = cond["build_manifest"]["model"]["gguf_bytes"]
    parts = math.ceil(size / CHUNK)
    versions = sum(len(m.get("published", {})) for m in cond.get("rounds_meta", []))
    cache_s = sum(
        max(3 * 3600, m["t_versions_deleted"] - p["t_publish"])
        for m in cond.get("rounds_meta", [])
        for p in m.get("published", {}).values()
        if "t_versions_deleted" in m
    )
    # Each snapshot init downloads the weights. The instrumented arm's before_snapshot records count them; the
    # REPORT-only arm emits none, so it is assumed to run as many per version as the instrumented arm.
    if snap_inits is None:
        snap_inits = versions
    p = PRICES
    c = {
        "invocations": len(rows),
        "cold_starts": colds,
        "restores": restores,
        "published_versions": versions,
        "billed_duration_s": round(billed_s, 3),
        "billed_restore_duration_s": round(billed_restore_s, 3),
        "lambda_duration_usd": round((billed_s + billed_restore_s) * mem_gb * p["arm64_gb_s"], 4),
        "lambda_requests_usd": round(len(rows) * p["request"], 6),
        "snapstart_restore_usd": round(restores * mem_gb * p["snapstart_restore_gb"], 4),
        "snapstart_cache_usd": round(cache_s * mem_gb * p["snapstart_cache_gb_s"], 4),
        "snapstart_cache_note": "3-hour minimum per version applied",
        "weight_downloads": colds + snap_inits,
        "snapshot_inits": snap_inits,
        "s3_get_requests_per_download": parts,
        "s3_head_requests_per_download": 1,
        "s3_request_usd": round(
            (colds + snap_inits) * (parts * p["s3_get_per_1000"] + p["s3_head_per_1000"]) / 1000, 6
        ),
        "s3_gb_transferred": round((colds + snap_inits) * size / 1e9, 2),
        "s3_transfer_usd": 0.0,
        "s3_transfer_note": "S3 to Lambda in the same region: no data transfer charge",
        "s3_storage_usd_month": round(size / 1e9 * p["s3_standard_gb_month"], 4),
        "prices": p,
    }
    c["total_usd"] = round(sum(v for k, v in c.items() if k.endswith("_usd")), 4)
    return c


def answers_by_request(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        if not r["error"] and r.get("answers"):
            out[r["request_id"]].append({"cpu_part": r.get("cpu_part"), "answers": r["answers"]})
    return out


def _flat(ans: dict) -> dict[str, float | str]:
    flat: dict[str, float | str] = {}
    for q, a in ans.items():
        for k in ("noul", "score", "confidence"):
            if k in a:
                flat[f"{q}.{k}"] = a[k]
        if "choice" in a:
            flat[f"{q}.choice"] = a["choice"]
        for o, pr in (a.get("probabilities") or {}).items():
            flat[f"{q}.p[{o}]"] = pr
    return flat


def _paths(v: object, prefix: str = "") -> set[str]:
    """Every leaf key path in an answer object, so an unknown field is part of its shape."""
    if isinstance(v, dict):
        return set().union(*(_paths(x, f"{prefix}.{k}") for k, x in v.items())) or {prefix}
    return {prefix}


def reference(rows: list[dict]) -> dict:
    """One answer set per (request, cpu_part): within a Graviton generation outputs repeat exactly (README)."""
    ref: dict = {}
    for rid, items in answers_by_request(rows).items():
        for it in items:
            ref.setdefault(rid, {}).setdefault(it["cpu_part"] or "?", it["answers"])
    return ref


def compare(rows: list[dict], ref: dict) -> dict:
    """Compare complete answer payloads vs the reference for the same request and CPU part.

    Numeric deltas are still reported for diagnosis, but bit-identical means the whole answer object matches.
    Missing or extra fields therefore cannot be silently certified as equal.
    """
    worst, flips, n, missing, exact, shape_mismatches = 0.0, 0, 0, 0, 0, 0
    for rid, items in answers_by_request(rows).items():
        for it in items:
            base = ref.get(rid, {}).get(it["cpu_part"] or "?")
            if base is None:
                missing += 1
                continue
            n += 1
            current = it["answers"]
            a, b = _flat(current), _flat(base)
            same = current == base
            if _paths(current) != _paths(base):
                shape_mismatches += 1
            for k in set(a) | set(b):
                v, w = b.get(k), a.get(k)
                if isinstance(v, str) or isinstance(w, str):
                    if v != w and k.endswith(".choice"):
                        flips += 1
                elif v is not None and w is not None:
                    worst = max(worst, abs(float(w) - float(v)))
            exact += same
    return {
        "compared": n,
        "bit_identical": exact,
        "choice_flips": flips,
        "max_abs_delta": worst,
        "answer_shape_mismatches": shape_mismatches,
        "no_reference_for_cpu_part": missing,
        "method": "complete answer object, same request id and same CPU part as the reference",
    }


def summarize(run: str, ref_path: str | None, baseline_commit: str | None, meta: dict) -> dict:
    cond, rows = load_run(run)
    versions = defaultdict(set)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            versions[cond["functions"][arm]["FunctionName"]].add(str(pub["version"]))
    snaps = snapshot_inits(run, versions)
    arms = {}
    # Counts, not distributions, so they sit under cost: lkai-result/1 allows only distributions under phases.
    snaps_per_version: dict[str, dict[str, int]] = {}
    for arm in cond["arms"]:
        fn = cond["functions"][arm]["FunctionName"]
        arms[arm] = arm_phases(by(rows, arm=arm), snaps.get(fn, []))
        per_ver: dict[str, int] = defaultdict(int)
        for s in snaps.get(fn, []):
            per_ver[str(s.get("version"))] += 1
        if per_ver:
            snaps_per_version[arm] = dict(sorted(per_ver.items()))
    errs = [
        {
            "arm": r["arm"],
            "round": r["round"],
            "wave": r["wave"],
            "error": r["error"],
            "detail": str(r.get("payload"))[:300],
        }
        for r in rows
        if r["error"]
    ]
    mism = [
        {
            "arm": r["arm"],
            "round": r["round"],
            "wave": r["wave"],
            "expected": r["expected_class"],
            "observed": r["observed_class"],
        }
        for r in rows
        if r["class_mismatch"]
    ]
    cpu = defaultdict(int)
    for r in rows:
        cpu[f"{r['arm']}:{r['expected_class']}:{r.get('cpu_part')}"] += 1
    m = cond["build_manifest"]
    f0 = cond["functions"][cond["arms"][0]]
    res = {
        "schema": "lkai-result/1",
        "experiment_id": meta.get("experiment_id", "P1-baseline"),
        "run_id": os.path.basename(os.path.normpath(run)),
        "status": meta.get("status", "baseline"),
        "hypothesis": meta.get(
            "hypothesis", "Baseline: no change. Pins the reference all experiments compare against."
        ),
        "changed_variable": meta.get("changed_variable"),
        "commits": {
            "experiment": m["repo_commit"],
            "baseline": baseline_commit or m["repo_commit"],
            "repo_dirty_at_build": m["repo_dirty"],
        },
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
            "code_sha256": m["function_zip"]["sha256"],
            "environment": f0["environment"],
            "az": "not exposed to the function; Lambda chooses per environment",
            "cpu_parts_observed": dict(sorted(cpu.items())),
        },
        "model": m["model"] | {"llama_cpp": m["llama_cpp"]},
        "build": {k: m[k] for k in ("images", "toolchain", "assets_sha256", "function_zip")},
        "conditions": {
            k: cond[k]
            for k in (
                "rounds",
                "concurrency",
                "warm_per_arm_per_round",
                "arms",
                "probe",
                "requests_file",
                "started",
            )
        }
        | {
            "finished": cond.get("finished"),
            "request_ids": [r["id"] for r in cond["requests"]["requests"]],
            "client": cond["client"],
            "weight_fetch": "HEAD + parallel 64 MiB ranged GETs (32 threads) into a memfd "
            "(examples/lambda/app/loader.py); llama.cpp mmaps /proc/self/fd/N",
        },
        "classification": {
            "method": "expected class from the harness (new $LATEST config -> cold; new published "
            "version -> restore; reused environment -> warm), checked against REPORT "
            "(Init Duration / Restore Duration / neither)",
            **classify(rows),
            "mismatch_rows": mism,
        },
        "phases": arms,
        "variance": {a: round_medians(by(rows, arm=a)) for a in cond["arms"]},
        "instrumentation_overhead": overhead(rows),
        "errors": {"count": len(errs), "rows": errs},
        "cost": costs(cond, rows, _snap_inits_estimate(cond, snaps))
        | {"snapshot_inits_per_version": snaps_per_version},
        "correctness": None,
        "profiling_artifacts": [],
        "artifacts": {
            "invocations": "invocations.jsonl",
            "logs": "logs/",
            "conditions": "conditions.json",
        },
        "reproduce": meta.get("reproduce", []),
        "conclusion": meta.get("conclusion"),
    }
    if ref_path:
        res["correctness"] = compare(rows, json.load(open(ref_path))["outputs"]) | {
            "reference": os.path.relpath(ref_path, HERE)
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference")
    ap.add_argument("--write-reference")
    ap.add_argument("--baseline-commit")
    ap.add_argument(
        "--meta", help="JSON file with hypothesis, changed_variable, conclusion, reproduce"
    )
    a = ap.parse_args()
    meta = json.load(open(a.meta)) if a.meta else {}
    if a.write_reference:
        cond, rows = load_run(a.run)
        json.dump(
            {
                "schema": "lkai-reference/1",
                "run_id": os.path.basename(os.path.normpath(a.run)),
                "model_gguf_sha256": cond["build_manifest"]["model"]["gguf_sha256"],
                "code_sha256": cond["build_manifest"]["function_zip"]["sha256"],
                "outputs": reference(rows),
            },
            open(a.write_reference, "w"),
            indent=1,
            sort_keys=True,
        )
        print("reference", a.write_reference)
    res = summarize(a.run, a.reference or a.write_reference, a.baseline_commit, meta)
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
