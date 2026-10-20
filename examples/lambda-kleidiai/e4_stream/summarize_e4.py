"""E4 results: baseline/summarize.py's lkai-result/1 plus the E4-specific parts.

  python summarize_e4.py <run-dir> --reference <outputs.json> --meta meta.json \
      --experiment-commit <sha> --baseline-commit <sha>

On top of summarize.summarize (phases per arm, variance, classification, errors, cost):
  - the streamed-load phases per arm and class (download_and_load, stream_header_wait, stream_load_blocked);
  - correctness of every arm against P1's reference/outputs.json (same request, same CPU part);
  - <arm>_vs_base for stream and streamfile: median(arm) - median(base) within each CPU part both drew, for the
    REPORT metrics and init phases, pooled by sample count (summarize.overhead's method);
  - snapshot inits counted on every arm (all are instrumented, so none needs P1's estimate);
  - snapstart_note: what the restore class can and cannot show for this variable.
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
import summarize as S  # noqa: E402

E4_PHASES = ("download_and_load", "stream_header_wait", "stream_load_blocked")
INIT_KEYS = (
    "weight_download",
    "model_load_mmap",
    "download_and_load",
    "stream_load_blocked",
    "context_init",
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


def compare_arms(rows: list[dict], a: str, b: str = "base") -> dict:
    """median(a) - median(b) within each CPU part both arms drew; pooled by the smaller sample count."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    out: dict = {}
    for cls, ms in _metrics().items():
        rs = [r for r in ok if r["expected_class"] == cls]
        cpus = sorted(
            {r.get("cpu_part") for r in rs if r["arm"] == a}
            & {r.get("cpu_part") for r in rs if r["arm"] == b},
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
        "report_max_memory_mb); pooled_delta_p50 weights the per-part deltas by the smaller sample count"
    )
    return out


def add_stream_phases(res: dict, rows: list[dict], snaps: dict) -> None:
    """The E4 init phases, which summarize.INIT_PHASES does not list, into phases.<arm>.cold/.restore."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    for arm, classes in res["phases"].items():
        cold = classes.get("cold")
        if cold is not None:
            inits = [
                r["lkai"]["init"]
                for r in ok
                if r["arm"] == arm
                and r["expected_class"] == "cold"
                and r.get("lkai")
                and "init" in r["lkai"]
            ]
            for ph in E4_PHASES:
                cold[ph] = S.dist([i[ph] for i in inits if ph in i])
        rest = classes.get("restore")
        fn = res["deployment"]["functions"][arm]
        if rest is not None and snaps.get(fn):
            si = rest.setdefault("snapshot_init", {})
            for ph in E4_PHASES:
                si[ph] = S.dist([s["init"][ph] for s in snaps[fn] if ph in s.get("init", {})])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--experiment-commit", required=True)
    ap.add_argument("--baseline-commit", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    res = S.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = S.load_run(a.run)
    ref = json.load(open(a.reference))["outputs"]
    res["primary_arm"] = "stream"
    res["commits"]["experiment"] = a.experiment_commit
    res["commits"]["baseline"] = a.baseline_commit
    res["commits"]["zip_built_at"] = cond["build_manifest"]["repo_commit"]
    res["correctness"] = S.compare(S.by(rows, arm="stream"), ref) | {
        "reference": "baseline/reference/outputs.json (P1 run 20261020T080835Z)",
        "by_arm": {arm: S.compare(S.by(rows, arm=arm), ref) for arm in cond["arms"]},
        "tolerance": "bit-identical required: every arm loads the same GGUF bytes with the same kernels; the "
        "streamed load only changes when each tensor is read, and waits for every byte before inference",
    }
    versions = defaultdict(set)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            versions[cond["functions"][arm]["FunctionName"]].add(str(pub["version"]))
    snaps = S.snapshot_inits(a.run, versions)
    add_stream_phases(res, rows, snaps)
    for arm in cond["arms"]:
        if arm != "base":
            res[f"{arm}_vs_base"] = compare_arms(rows, arm)
    res["instrumentation_overhead"] = {
        "method": "not measured in E4: every arm is instrumented; P1 measured it (within run-to-run spread)",
        "cells": [],
    }
    n_snap = sum(len(v) for v in snaps.values()) or None
    res["cost"] = S.costs(cond, rows, n_snap) | {
        "snapshot_inits_per_version": res["cost"].get("snapshot_inits_per_version", {}),
        "snapshot_inits_note": "counted on every arm (all instrumented)",
    }
    res["conditions"]["weight_fetch"] = (
        "every arm: PR #77's HEAD + 64 MiB ranged GETs on 32 threads into a memfd. base: then llama.cpp mmaps it. "
        "stream/streamfile: the load starts once the header has arrived and waits per tensor (e4_stream/)"
    )
    res["snapstart_note"] = (
        "Under SnapStart the download and load run inside publish_version's snapshot init, never in a restore. "
        "The streamed load can only change restore through what the snapshot holds (the same model in the same "
        "memory), so restore is the control class: any restore delta beyond run-to-run spread would be a defect "
        "or noise. The snapshot init itself (phases.<arm>.restore.snapshot_init) does show the overlap, and it "
        "bounds how long publish_version takes, not any request latency."
    )
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
