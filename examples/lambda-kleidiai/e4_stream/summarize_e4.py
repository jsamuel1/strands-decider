"""E4 results: baseline/summarize.py's lkai-result/1 plus the E4-specific parts.

  python summarize_e4.py <run-dir> --reference <outputs.json> --meta meta.json \
      --experiment-commit <sha> --baseline-commit <sha>

On top of summarize.summarize (phases per arm, variance, classification, errors, cost):
  - the streamed-load phases per arm and class (download_and_load, stream_header_wait, stream_load_blocked);
  - correctness of every arm against P1's reference/outputs.json (same request, same CPU part);
  - <arm>_vs_base for every arm but base: median(arm) - median(base) within each CPU part both drew, for the
    REPORT metrics and init phases, pooled by sample count (summarize.overhead's method);
  - snapshot inits counted on every arm (all are instrumented, so none needs P1's estimate);
  - snapstart_note: what the restore class can and cannot show for this variable;
  - host_variability: the CPU and environment metadata the run observed, what it did not record ("unknown"), and
    every arm's per-round delta against base within each CPU part, so a reader can see whether the host mix could
    change a conclusion.

  python summarize_e4.py <run-dir> --host-only --meta meta.json --summarized-at <sha>

adds or refreshes only host_variability (and the meta's conclusion) in an existing results.json, for runs that were
summarized before the block existed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

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


def _known_cpu_parts(rows: list[dict], *arms: str) -> list[str]:
    """CPU identities observed by every requested arm; unknown rows are never a shared identity."""
    sets = [
        {str(r["cpu_part"]) for r in rows if r["arm"] == arm and r.get("cpu_part") is not None}
        for arm in arms
    ]
    return sorted(set.intersection(*sets), key=str) if sets else []


def _unknown_cpu_parts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Unknown CPU-identity counts by start class and arm, retained as unassessed evidence."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (cls, arm), n in sorted(
        Counter((r["expected_class"], r["arm"]) for r in rows if r.get("cpu_part") is None).items()
    ):
        out[cls][arm] = n
    return {cls: dict(arms) for cls, arms in out.items()}


def compare_arms(rows: list[dict], a: str, b: str = "base") -> dict:
    """median(a) - median(b) within each known CPU part both arms drew; pooled by the smaller sample count."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    unknown = _unknown_cpu_parts(ok)
    out: dict = {}
    for cls, ms in _metrics().items():
        rs = [r for r in ok if r["expected_class"] == cls]
        cpus = _known_cpu_parts(rs, a, b)
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
    if unknown:
        out["unknown_cpu_part_rows"] = unknown
    out["method"] = (
        f"median({a}) - median({b}) within each known CPU part both arms drew (seconds; MB for "
        "report_max_memory_mb); pooled_delta_p50 weights the per-part deltas by the smaller sample count; "
        "rows with unknown CPU identity are excluded and counted in unknown_cpu_part_rows"
    )
    return out


# What /proc/cpuinfo, the auxiliary vector and the cgroup could have told us but no E4 run recorded. The harness logs
# only cpuinfo's "CPU part" line (P1's engine), the ggml CPU variants it loaded and its thread count.
HOST_UNRECORDED = (
    "cpuinfo CPU implementer, variant and revision",
    "cpuinfo Features flags",
    "AT_HWCAP / AT_HWCAP2",
    "cgroup cpu and memory limits",
    "kernel release",
    "availability zone and physical host",
)


def _log_fields(rows: list[dict], pattern: str) -> dict[str, Counter]:
    """{cpu_part: Counter(value)} for the last `pattern` match in each row's log tail (cold rows carry the init line).

    Rows without a CPU identity are counted under "unknown", an accounting bucket rather than a CPU part."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        m = re.findall(pattern, r.get("log_tail") or "")
        if m:
            cpu = r.get("cpu_part")
            out["unknown" if cpu is None else str(cpu)][m[-1]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _runtime_versions(run: str) -> dict[str, int]:
    """INIT_START runtime versions in the saved CloudWatch logs, counted per log stream (one per environment)."""
    seen: dict[str, str] = {}
    d = os.path.join(run, "logs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                e = json.loads(line)
                m = re.search(r"INIT_START Runtime Version: (\S+)", e.get("message", ""))
                if m:
                    seen[f"{f}:{e.get('logStreamName')}"] = m.group(1)
    return dict(sorted(Counter(seen.values()).items()))


def round_deltas(rows: list[dict], a: str, b: str = "base") -> dict:
    """Per round and CPU part, median(a) - median(b) of the first-request total and Max Memory Used.

    `sign` is "negative" or "positive" when every comparable round's delta has that sign, else "mixed". Each
    delta stays within one CPU part, but arms were run sequentially in alternating order within a round, so this
    controls CPU identity and round-level drift rather than simultaneous-host variation.
    """
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    ms = _metrics()
    out: dict = {}
    for cls in ("cold", "restore"):
        for name in ("first_request_total", "report_max_memory_mb"):
            f = ms[cls][name]
            per: dict = {}
            for cpu in _known_cpu_parts(ok, a, b):
                ds = []
                for rd in sorted({r["round"] for r in ok}):

                    def xs(arm, rd=rd, cpu=cpu, cls=cls, f=f):
                        return [
                            v
                            for r in ok
                            if r["arm"] == arm
                            and r["round"] == rd
                            and r["expected_class"] == cls
                            and r.get("cpu_part") == cpu
                            if (v := f(r)) is not None
                        ]

                    xa, xb = xs(a), xs(b)
                    if xa and xb:
                        ds.append(round(statistics.median(xa) - statistics.median(xb), 3))
                if ds:
                    sign = (
                        "negative"
                        if all(d < 0 for d in ds)
                        else "positive"
                        if all(d > 0 for d in ds)
                        else "mixed"
                    )
                    per[str(cpu)] = {
                        "per_round": ds,
                        "min": min(ds),
                        "max": max(ds),
                        "sign": sign,
                    }
            if per:
                out.setdefault(cls, {})[name] = per
    return out


def host_variability(run: str, cond: dict, rows: list[dict]) -> dict:
    """The initiative's HOST VARIABILITY assessment for this run, from what the run actually recorded."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    unknown = _unknown_cpu_parts(ok)
    mix: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (arm, cls, cpu), n in sorted(
        Counter(
            (r["arm"], r["expected_class"], str(r["cpu_part"]))
            for r in ok
            if r.get("cpu_part") is not None
        ).items()
    ):
        mix[cls][arm][cpu] = n
    envs = {(r["arm"], (r.get("lkai") or {}).get("env_id")) for r in ok if r.get("lkai")}
    return {
        "observed": {
            "cpu_part": "/proc/cpuinfo 'CPU part', read per request (rows' cpu_part)",
            "cpu_parts_by_class_and_arm": {c: dict(v) for c, v in mix.items()},
            "unknown_cpu_part_rows_by_class_and_arm": unknown,
            "ggml_cpu_variants_by_cpu_part": _log_fields(ok, r"ggml_cpu=(\S+)"),
            "os_cpu_count_by_cpu_part": _log_fields(ok, r"decider\.engine init .*? threads=(\d+)"),
            "runtime_version_arn": cond.get("runtime_version_arn"),
            "init_start_runtime_versions": _runtime_versions(run),
            "environments": len(envs),
            "log_streams": "logs/<function>.jsonl.gz, logStreamName per record",
            "memory_mb": sorted({f["MemorySize"] for f in cond["functions"].values()}),
            "architecture": sorted(
                {a for f in cond["functions"].values() for a in f["Architectures"]}
            ),
        },
        "unknown": list(HOST_UNRECORDED),
        "generation": "not claimed: the runs recorded the CPU part number only (no implementer), so no Graviton "
        "generation is named; parts are referred to by number",
        "method": "every arm vs base is compared only within one known CPU part ("
        "<arm>_vs_base.<class>.<metric>.by_cpu_part); rows with unknown CPU identity are excluded from "
        "within-part comparisons and counted under observed.unknown_cpu_part_rows_by_class_and_arm; arms run "
        "sequentially in alternating order each round, and per_round_deltas show each comparable round's "
        "within-part delta",
        "per_round_deltas": {arm: round_deltas(rows, arm) for arm in cond["arms"] if arm != "base"},
    }


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


def amend_host(run: str, meta: dict, summarized_at: str) -> dict:
    """Add host_variability to an existing results.json and take the meta's conclusion; nothing else changes."""
    path = os.path.join(run, "results.json")
    res = json.load(open(path))
    cond, rows = S.load_run(run)
    res["host_variability"] = host_variability(run, cond, rows) | {
        "summarized_at_commit": summarized_at,
        "assessment": meta.get("host_assessment"),
    }
    res["conclusion"] = meta["conclusion"]
    json.dump(res, open(path, "w"), indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--experiment-commit")
    ap.add_argument("--baseline-commit")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--summarized-at")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    if a.host_only:
        if not a.summarized_at:
            ap.error("--host-only needs --summarized-at")
        amend_host(a.run, meta, a.summarized_at)
        print("amended", os.path.join(a.run, "results.json"))
        return
    for req in ("reference", "experiment_commit", "baseline_commit"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} is required")
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
        "E4_STREAM=1 arms: the load starts once the header has arrived and waits per tensor (e4_stream/); "
        "E4_MMAP_THRESHOLD arms: glibc mmap threshold pinned (stream_loader.pin_mmap_threshold)"
    )
    res["snapstart_note"] = (
        "Under SnapStart the download and load run inside publish_version's snapshot init, never in a restore. "
        "The streamed load can only change restore through what the snapshot holds (the same model in the same "
        "memory), so restore is the control class: any restore delta beyond run-to-run spread would be a defect "
        "or noise. The snapshot init itself (phases.<arm>.restore.snapshot_init) does show the overlap, and it "
        "bounds how long publish_version takes, not any request latency."
    )
    res["host_variability"] = host_variability(a.run, cond, rows) | {
        "summarized_at_commit": a.experiment_commit,
        "assessment": meta.get("host_assessment"),
    }
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
