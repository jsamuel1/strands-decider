"""E9 results: baseline/summarize.py's lkai-result/1 plus the E9-specific parts.

  python summarize_e9.py <run-dir> <e9-build> --reference <outputs.json> --meta meta.json ...

On top of summarize.summarize (phases per arm, variance, classification, errors, cost):
  - correctness per arm, each against P1's reference/outputs.json (same request, same CPU part);
  - e9_vs_base: median(e9) - median(base) within each CPU part both arms drew, for every REPORT metric and init
    phase, pooled by sample count (summarize.overhead's method, which P1 uses for inst vs rep);
  - snapshot inits counted on both arms (both are instrumented, so neither needs P1's estimate);
  - each arm's zip and GGUF, the pack record, and load_paths (which KleidiAI path each environment took);
  - host_variability: host_variability.assess (observed CPU/environment metadata, per-CPU-part comparison and
    verdict), also written to host-variability.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import host_variability  # noqa: E402
import summarize as S  # noqa: E402

INIT_KEYS = (
    "weight_download",
    "ggml_backend_load",
    "model_load_mmap",
    "context_init",
    "assets_load",
    "warmup_inference",
    "model_ready",
    "imports",
)


def _metrics() -> dict:
    rep = lambda k: lambda r: r["report"][k] / 1000 if k in r["report"] else None  # noqa: E731
    tot = lambda k: lambda r: (r["report"].get(k, 0) + r["report"].get("Duration", 0)) / 1000  # noqa: E731
    init = lambda k: lambda r: (r.get("lkai") or {}).get("init", {}).get(k)  # noqa: E731
    mem = lambda r: r["report"].get("Max Memory Used")  # noqa: E731
    lat = lambda r: r["latency_ms"] / 1000 if r.get("latency_ms") is not None else None  # noqa: E731
    return {
        "cold": {
            "first_request_total": tot("Init Duration"),
            "report_init_duration": rep("Init Duration"),
            "report_duration": rep("Duration"),
            "first_request_inference": lat,
            "report_max_memory_mb": mem,
            **{k: init(k) for k in INIT_KEYS},
        },
        "restore": {
            "first_request_total": tot("Restore Duration"),
            "report_restore_duration": rep("Restore Duration"),
            "report_duration": rep("Duration"),
            "first_request_inference": lat,
            "report_max_memory_mb": mem,
        },
        "warm": {
            "report_duration": rep("Duration"),
            "steady_state_inference": lat,
            "report_max_memory_mb": mem,
        },
    }


def compare_arms(rows: list[dict], a: str = "e9", b: str = "base") -> dict:
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    out: dict = {}
    for cls, ms in _metrics().items():
        rs = [r for r in ok if r["expected_class"] == cls]
        cpus = sorted(
            (
                {r.get("cpu_part") for r in rs if r["arm"] == a}
                & {r.get("cpu_part") for r in rs if r["arm"] == b}
            )
            - {None},
            key=str,
        )
        cls_out: dict = {}
        for name, f in ms.items():
            per, wsum, wdelta = {}, 0, 0.0
            for cpu in cpus:
                xa = [
                    v
                    for r in rs
                    if r["arm"] == a and r.get("cpu_part") == cpu
                    if (v := f(r)) is not None
                ]
                xb = [
                    v
                    for r in rs
                    if r["arm"] == b and r.get("cpu_part") == cpu
                    if (v := f(r)) is not None
                ]
                if not xa or not xb:
                    continue
                ma, mb = statistics.median(xa), statistics.median(xb)
                per[str(cpu)] = {
                    f"{a}_p50": round(ma, 6),
                    f"{b}_p50": round(mb, 6),
                    "delta_p50": round(ma - mb, 6),
                    "delta_pct": round(100 * (ma - mb) / mb, 2) if mb else None,
                    "n": [len(xa), len(xb)],
                }
                w = min(len(xa), len(xb))
                wsum, wdelta = wsum + w, wdelta + w * (ma - mb)
            if per:
                cls_out[name] = {
                    "by_cpu_part": per,
                    "pooled_delta_p50": round(wdelta / wsum, 6),
                    "pooled_weight": wsum,
                }
        if cls_out:
            out[cls] = cls_out
    out["method"] = (
        f"median({a}) - median({b}) within each CPU part both arms drew (seconds; MB for "
        "report_max_memory_mb); rows with no CPU part enter no part; pooled_delta_p50 weights the per-part deltas "
        "by the smaller sample count"
    )
    return out


def load_paths(rows: list[dict], run: str) -> dict:
    """Which KleidiAI load path each environment took, from patch_e9.py's 'E9KAI <path> nr= kr=' stderr lines in the
    invocation log tails and the saved CloudWatch logs. The base arm runs P1's loader, which prints none."""
    import gzip
    import re

    pat = re.compile(r"E9KAI (adopt|convert|repack) nr=(\d+) kr=(\d+)")
    by_env: dict = defaultdict(set)
    part_of_env: dict = {}
    arm_of_env: dict = {}
    for r in rows:
        env = (r.get("lkai") or {}).get("env_id")
        part_of_env.setdefault(env, r.get("cpu_part"))
        arm_of_env.setdefault(env, r["arm"])
        for m in pat.finditer(r.get("log_tail") or ""):
            by_env[env].add(f"{m.group(1)} kr={m.group(3)}")
    streams: dict = defaultdict(lambda: defaultdict(int))
    d = os.path.join(run, "logs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".jsonl.gz"):
            continue
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                if line.strip() and (m := pat.search(json.loads(line)["message"])):
                    streams[f.removesuffix(".jsonl.gz")][f"{m.group(1)} kr={m.group(3)}"] += 1
    tally: dict = defaultdict(int)
    for env, paths in by_env.items():
        tally[
            f"{arm_of_env.get(env)}:{part_of_env.get(env) or 'unknown'}:{'+'.join(sorted(paths))}"
        ] += 1
    return {
        "method": "environments by arm, CPU part and E9KAI lines in their 4 KB log tails (a tail can miss the init "
        "lines; those environments are not counted here); cloudwatch_lines counts every E9KAI line in the saved logs",
        "environments": dict(sorted(tally.items())),
        "cloudwatch_lines": {k: dict(v) for k, v in sorted(streams.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("build")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--experiment-commit", required=True)
    ap.add_argument("--baseline-commit", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    res = S.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = S.load_run(a.run)
    ref = json.load(open(a.reference))["outputs"]
    st = json.load(open(os.path.join(a.build, "deployment.json")))
    res["primary_arm"] = "e9"
    res["commits"]["experiment"] = a.experiment_commit
    res["commits"]["baseline"] = a.baseline_commit
    res["commits"]["base_zip_and_weights_built_at"] = cond["build_manifest"]["repo_commit"]
    res["correctness"] = S.compare(S.by(rows, arm="e9"), ref) | {
        "reference": "baseline/reference/outputs.json (P1 run 20261020T080835Z)",
        "by_arm": {arm: S.compare(S.by(rows, arm=arm), ref) for arm in cond["arms"]},
        "tolerance": "bit-identical required: offline, the packed buffers the E9 loader holds hash identical to the "
        "runtime repack's on both kernel layouts (offline-verdict.json), so any difference would be a defect",
    }
    res["model"]["format_by_arm"] = {
        "base": {"format": "GGUF v3 Q8_0 (P1)", "sha256": st["gguf_sha256_by_arm"]["base"]},
        "e9": {
            "format": "GGUF v3 Q8_0 types, KleidiAI-prepacked payloads ("
            + st["e9_pack"]["layout"]
            + ")",
            "sha256": st["gguf_sha256_by_arm"]["e9"],
            "bytes": st["e9_pack"]["packed"]["bytes"],
        },
    }
    res["model"]["e9_pack"] = st["e9_pack"]
    res["code_by_arm"] = {
        arm: {"function_zip_sha256": z} for arm, z in st["zip_sha256_by_arm"].items()
    }
    res["deployment"]["code_sha256"] = st["zip_sha256_by_arm"]["e9"]
    res["deployment"]["code_sha256_by_arm"] = st["zip_sha256_by_arm"]
    res["e9_vs_base"] = compare_arms(rows)
    res["load_paths"] = load_paths(rows, a.run)
    res["instrumentation_overhead"] = {
        "method": "not measured in E9: both arms are instrumented; P1 measured it (within run-to-run spread)",
        "cells": [],
    }
    versions = defaultdict(set)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            versions[cond["functions"][arm]["FunctionName"]].add(str(pub["version"]))
    snaps = S.snapshot_inits(a.run, versions)
    n_snap = sum(len(v) for v in snaps.values()) or None
    res["cost"] = S.costs(cond, rows, n_snap) | {
        "snapshot_inits_per_version": res["cost"].get("snapshot_inits_per_version", {}),
        "snapshot_inits_note": "counted on both arms (both instrumented); the e9 GGUF is 128 bytes larger, "
        "which does not change the S3 request count or the transfer figure at this precision",
    }
    hv = host_variability.assess(a.run)
    res["host_variability"] = hv
    json.dump(hv, open(os.path.join(a.run, "host-variability.json"), "w"), indent=1)
    res["artifacts"]["host_variability"] = "host-variability.json"
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
