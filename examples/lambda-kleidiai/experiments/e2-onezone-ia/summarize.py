"""E2 results: P1's summary of a bench.py run, plus the std-vs-ozia comparison and the per-storage-class cost.

  python summarize.py <run-dir> --reference ../../baseline/reference/outputs.json --meta meta.json \
      --deployment build/deployment.json --baseline-commit <P1 sha>

The run directory is P1's bench.py output for arms `std` (STANDARD weights) and `ozia` (ONEZONE_IA weights). This
script calls baseline/summarize.py's `summarize()` unchanged for the distributions, classification, variance, errors
and correctness, then replaces the parts that assume P1's inst/rep pair:

- `comparison`: ozia minus std on every init phase, matched by start class (and, for engine-bound phases, by CPU
  part), with a seeded bootstrap 95% interval on the ratio of medians. `status` follows from a decision rule fixed
  before the run (`DECISION`), on cold `weight_download`.
- `cost`: per-arm S3 requests, retrieval and storage at the storage class each arm used, plus a per-download and
  per-1,000-cold-start model for Standard, One Zone-IA and S3 Express One Zone (E1) at list prices.
- `correctness`: per-arm comparison with P1's reference, the S3 full-object SHA-256 of each weight copy, and a
  SHA-256 over each arm's canonical answers per (request, CPU part).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "..", "baseline"))
sys.path.insert(0, BASE)
import summarize as base  # noqa: E402

ARMS = ("std", "ozia")
# us-east-2 list prices (USD), read from the AWS Price List API (service AmazonS3, regionCode us-east-2) 2026-10-20.
S3_PRICES = {
    "STANDARD": {
        "storage_gb_month": 0.023,
        "get_per_1000": 0.0004,
        "retrieval_per_gb": 0.0,
        "min_storage_days": 0,
        "usage_types": "USE2-TimedStorage-ByteHrs, USE2-Requests-Tier2",
    },
    "ONEZONE_IA": {
        "storage_gb_month": 0.01,
        "get_per_1000": 0.001,
        "retrieval_per_gb": 0.01,
        "min_storage_days": 30,
        "min_object_kb": 128,
        "usage_types": "USE2-TimedStorage-ZIA-ByteHrs, USE2-Requests-ZIA-Tier2, USE2-Retrieval-ZIA, "
        "USE2-EarlyDelete-ZIA",
    },
    "EXPRESS_ONEZONE": {
        "storage_gb_month": 0.11,
        "get_per_1000": 0.00003,
        "retrieval_per_gb": 0.0006,
        "retrieval_note": "charged on the part of each request above 512 KB; a 64 MiB range is ~99% billable",
        "min_storage_days": 0,
        "usage_types": "USE2-TimedStorage-XZ-ByteHrs, USE2-Requests-XZ-Tier2, USE2-Retrieval-XZ",
    },
    "source": "AWS Price List API, AmazonS3, regionCode us-east-2, read 2026-10-20",
}
# Fixed before the run. One Zone-IA is expected to change nothing; "no change" must be shown, not assumed.
DECISION = {
    "metric": "phases.<arm>.cold.weight_download p50",
    "statistic": "ratio ozia/std of medians, bootstrap 95% interval (10,000 resamples, seed 0)",
    "equivalence_band": [0.98, 1.02],
    "rule": "no-change if the interval lies inside the band; improved if its upper end is below 0.98; "
    "regressed if its lower end is above 1.02; otherwise inconclusive",
}
COMPARE = {
    "cold": (
        "weight_download",
        "weight_download_mib_s",
        "model_load_mmap",
        "model_ready",
        "report_init_duration",
        "first_request_total",
        "first_request_inference",
    ),
    "restore": (
        "report_restore_duration",
        "restore_to_invoke",
        "first_request_inference",
        "first_request_total",
    ),
    "warm": ("steady_state_inference", "report_duration"),
}


def boot_ratio(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> list[float] | None:
    """Bootstrap 95% interval of median(a) / median(b), resampling each arm independently."""
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    rs = []
    for _ in range(n):
        ma = statistics.median(rng.choices(a, k=len(a)))
        mb = statistics.median(rng.choices(b, k=len(b)))
        if mb:
            rs.append(ma / mb)
    rs.sort()
    return [round(rs[int(0.025 * len(rs))], 4), round(rs[int(0.975 * len(rs)) - 1], 4)]


def cluster_boot_ratio(
    a: dict[object, list[float]], b: dict[object, list[float]], n: int = 10000, seed: int = 0
) -> list[float] | None:
    """Paired round-cluster bootstrap 95% interval of median(a) / median(b).

    `a` and `b` map a round to that round's values. Each resample draws rounds with replacement and takes the SAME
    rounds for both arms, pooling every value of each drawn round, so samples that share a wave (and the time window
    that wave ran in) move together instead of being treated as independent.
    """
    rounds = sorted(set(a) & set(b), key=str)
    if len(rounds) < 2:
        return None
    rng = random.Random(seed)
    rs = []
    for _ in range(n):
        pick = rng.choices(rounds, k=len(rounds))
        va = [v for r in pick for v in a[r]]
        vb = [v for r in pick for v in b[r]]
        if not va or not vb:
            continue
        mb = statistics.median(vb)
        if mb:
            rs.append(statistics.median(va) / mb)
    if not rs:
        return None
    rs.sort()
    return [round(rs[int(0.025 * len(rs))], 4), round(rs[int(0.975 * len(rs)) - 1], 4)]


def decide(ci: list[float] | None) -> str:
    lo, hi = DECISION["equivalence_band"]
    if ci is None:
        return "inconclusive"
    if lo <= ci[0] and ci[1] <= hi:
        return "no-change"
    if ci[1] < lo:
        return "improved"
    if ci[0] > hi:
        return "regressed"
    return "inconclusive"


def _values(rows: list[dict], cls: str, phase: str, snaps: list[dict] | None = None) -> list[float]:
    ok = [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]
    rep = {
        "report_init_duration": "Init Duration",
        "report_restore_duration": "Restore Duration",
        "report_duration": "Duration",
    }
    out: list[float] = []
    for r in ok:
        if phase in rep:
            v = r["report"].get(rep[phase])
            v = None if v is None else v / 1000
        elif phase == "first_request_total":
            k = "Init Duration" if cls == "cold" else "Restore Duration"
            v = (r["report"].get(k, 0) + r["report"].get("Duration", 0)) / 1000
        elif phase in ("first_request_inference", "steady_state_inference"):
            v = None if r.get("latency_ms") is None else r["latency_ms"] / 1000
        elif phase == "restore_to_invoke":
            v = ((r.get("lkai") or {}).get("inv") or {}).get("restore_to_invoke_s")
        elif phase == "weight_download_mib_s":
            v = ((r.get("lkai") or {}).get("counters") or {}).get("weight_mib_s")
        else:
            v = ((r.get("lkai") or {}).get("init") or {}).get(phase)
        if v is not None:
            out.append(float(v))
    return out


def _cmp(a: list[float], b: list[float]) -> dict | None:
    if not a or not b:
        return None
    ma, mb = statistics.median(a), statistics.median(b)
    return {
        "std_p50": round(mb, 6),
        "ozia_p50": round(ma, 6),
        "delta_p50_s": round(ma - mb, 6),
        "ratio_p50": round(ma / mb, 4) if mb else None,
        "ratio_ci95": boot_ratio(a, b),
        "n": [len(b), len(a)],
    }


def _by_round(rows: list[dict], cls: str, phase: str) -> dict[object, list[float]]:
    out: dict[object, list[float]] = {}
    for rnd in sorted({r["round"] for r in rows}, key=str):
        v = _values([r for r in rows if r["round"] == rnd], cls, phase)
        if v:
            out[rnd] = v
    return out


SENSITIVITY = {
    "name": "paired round-cluster bootstrap",
    "why": "each round's cold starts run as one concurrent wave in a shared time window, and arm order alternates "
    "by round, so the samples within a round are not independent; the primary bootstrap treats them as if they were",
    "method": "resample rounds with replacement (10,000 resamples, seed 0), the same rounds for both arms, pool "
    "each drawn round's values, ratio of medians ozia/std",
    "role": "sensitivity only: the pre-registered decision rule above sets status",
}


def comparison(rows: list[dict], snaps: dict[str, list[dict]]) -> dict:
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    out: dict = {"decision": DECISION}
    for cls, names in COMPARE.items():
        c: dict = {}
        for ph in names:
            c[ph] = _cmp(_values(by_arm["ozia"], cls, ph), _values(by_arm["std"], cls, ph))
            if c[ph] is not None:
                c[ph]["ratio_ci95_round_cluster"] = cluster_boot_ratio(
                    _by_round(by_arm["ozia"], cls, ph), _by_round(by_arm["std"], cls, ph)
                )
        # Engine-bound phases depend on the host CPU (the cpuinfo `CPU part`); compare within a CPU part as well.
        per_cpu: dict = {}
        cpus = sorted(
            {r.get("cpu_part") for r in by_arm["std"] if r["expected_class"] == cls}
            & {r.get("cpu_part") for r in by_arm["ozia"] if r["expected_class"] == cls},
            key=str,
        )
        for cpu in cpus:
            sel = {a: [r for r in by_arm[a] if r.get("cpu_part") == cpu] for a in ARMS}
            per_cpu[str(cpu)] = {}
            for ph in names:
                d = _cmp(_values(sel["ozia"], cls, ph), _values(sel["std"], cls, ph))
                if d is not None:
                    d["ratio_ci95_round_cluster"] = cluster_boot_ratio(
                        _by_round(sel["ozia"], cls, ph), _by_round(sel["std"], cls, ph)
                    )
                per_cpu[str(cpu)][ph] = d
        c["by_cpu_part"] = per_cpu
        out[cls] = c
    # The snapshot init (inside publish_version) also downloads the weights, from the same object.
    snap: dict = {}
    for ph in ("weight_download", "model_ready"):
        a = [s["init"][ph] for s in snaps.get("ozia", []) if ph in s.get("init", {})]
        b = [s["init"][ph] for s in snaps.get("std", []) if ph in s.get("init", {})]
        snap[ph] = _cmp(a, b)
    out["snapshot_init"] = snap
    primary = out["cold"].get("weight_download") or {}
    out["outcome"] = decide(primary.get("ratio_ci95"))
    cl = primary.get("ratio_ci95_round_cluster")
    out["sensitivity"] = SENSITIVITY | {
        "rounds": len({r["round"] for r in rows}),
        "ci95": cl,
        "outcome_under_same_band": decide(cl),
        "agrees_with_primary": decide(cl) == out["outcome"],
    }
    return out


# What the run could and could not observe about the host. bench.py keeps the handler's `cpu_part` (the /proc/cpuinfo
# "CPU part" line) per invocation; the saved CloudWatch logs carry each environment's `init cpu_part=... ggml_cpu=...`
# line (the ggml CPU backend variants that ggml's own feature probe loaded). Nothing else about the host was logged.
HOST_UNOBSERVED = (
    "cpuinfo CPU implementer",
    "cpuinfo CPU variant and revision",
    "cpuinfo Features / AT_HWCAP / AT_HWCAP2",
    "core count of a restored environment (threads come from the snapshotting init)",
    "cgroup CPU and memory limits",
    "kernel release",
    "processor generation or product name (not inferred from CPU part or from timings)",
)


def _log_hosts(run: str) -> dict[str, dict]:
    """Per function: {cpu_part: set of ggml variants} from init lines, and the CPU part of every snapshot init."""
    out: dict[str, dict] = {}
    d = os.path.join(run, "logs")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        fn = f.removesuffix(".jsonl.gz")
        streams: dict[str, dict] = defaultdict(lambda: {"parts": [], "snap": 0})
        variants: dict[str, set[str]] = defaultdict(set)
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                e = json.loads(line)
                msg, st = e.get("message", ""), streams[e.get("logStreamName", "")]
                m = re.search(r"init cpu_part=(\S+) ggml_cpu=(\S+)", msg)
                if m:
                    st["parts"].append(m.group(1))
                    variants[m.group(1)].add(m.group(2))
                if msg.startswith("LKAI ") and '"before_snapshot"' in msg:
                    st["snap"] += 1
        snap_parts: dict[str, int] = defaultdict(int)
        for st in streams.values():
            if st["snap"]:
                for p in st["parts"] or ["unknown"]:
                    snap_parts[p] += 1
        out[fn] = {
            "ggml_cpu_by_cpu_part": {p: sorted(v) for p, v in sorted(variants.items())},
            "snapshot_init_cpu_parts": dict(sorted(snap_parts.items())),
        }
    return out


def _round_cv(rows: list[dict], cls: str, f) -> dict | None:
    per: dict[object, list[float]] = defaultdict(list)
    for r in rows:
        if r["expected_class"] == cls and not r["error"] and not r["class_mismatch"]:
            per[r["round"]].append(f(r))
    meds = [statistics.median(v) for _, v in sorted(per.items(), key=lambda kv: str(kv[0]))]
    if len(meds) < 2:
        return None
    return {
        "per_round_median_s": [round(m, 4) for m in meds],
        "cv": round(statistics.stdev(meds) / statistics.fmean(meds), 4),
    }


def _round_cpu_primary(rows: list[dict]) -> dict:
    """Cold weight-download comparison within each shared (round, CPU part) cell."""
    cells: list[dict] = []
    rounds = sorted({r["round"] for r in rows}, key=str)
    parts = sorted({str(r.get("cpu_part")) for r in rows}, key=str)
    for cpu in parts:
        for rnd in rounds:
            vals = {
                arm: _values(
                    [
                        r
                        for r in rows
                        if r["arm"] == arm and r["round"] == rnd and str(r.get("cpu_part")) == cpu
                    ],
                    "cold",
                    "weight_download",
                )
                for arm in ARMS
            }
            if not vals["std"] or not vals["ozia"]:
                continue
            std = statistics.median(vals["std"])
            ozia = statistics.median(vals["ozia"])
            ratio = ozia / std
            cells.append(
                {
                    "round": rnd,
                    "cpu_part": cpu,
                    "n": [len(vals["std"]), len(vals["ozia"])],
                    "std_median_s": round(std, 6),
                    "ozia_median_s": round(ozia, 6),
                    "delta_s": round(ozia - std, 6),
                    "ratio": round(ratio, 6),
                    "inside_equivalence_band": DECISION["equivalence_band"][0]
                    <= ratio
                    <= DECISION["equivalence_band"][1],
                }
            )
    ratios = [c["ratio"] for c in cells]
    deltas = [c["delta_s"] for c in cells]
    signs = sorted({"negative" if d < 0 else "positive" if d > 0 else "zero" for d in deltas})
    return {
        "metric": "cold weight_download median, ozia/std, within shared (round, CPU part)",
        "cells": cells,
        "shared_cells": len(cells),
        "ratio_range": [min(ratios), max(ratios)] if ratios else None,
        "delta_s_range": [min(deltas), max(deltas)] if deltas else None,
        "delta_signs": signs,
        "all_inside_equivalence_band": bool(cells)
        and all(c["inside_equivalence_band"] for c in cells),
        "missing": "CPU parts absent from either arm in a round have no shared-cell comparison",
    }


def _pooled_primary_by_cpu_part(rows: list[dict]) -> dict:
    """Cold weight-download median per CPU part, both arms pooled: how much the metric depends on the host."""
    parts = sorted({str(r.get("cpu_part")) for r in rows}, key=str)
    cells = {}
    for cpu in parts:
        v = _values([r for r in rows if str(r.get("cpu_part")) == cpu], "cold", "weight_download")
        if v:
            cells[cpu] = {"n": len(v), "median_s": round(statistics.median(v), 4)}
    meds = [c["median_s"] for c in cells.values()]
    return {
        "metric": "cold weight_download median, std and ozia pooled, per CPU part",
        "by_cpu_part": cells,
        "span_ratio": round(max(meds) / min(meds), 4) if meds else None,
    }


def _warm_by_round_and_cpu_part(rows: list[dict]) -> dict:
    """Warm request count and latency median per (arm, round, CPU part): where a pooled warm CV comes from."""
    out: dict = {}
    for arm in ARMS:
        per: dict = defaultdict(list)
        for r in rows:
            if (
                r["arm"] == arm
                and r["expected_class"] == "warm"
                and not r["error"]
                and not r["class_mismatch"]
            ):
                per[(r["round"], str(r.get("cpu_part")))].append(r["latency_ms"] / 1000)
        out[arm] = [
            {
                "round": rnd,
                "cpu_part": cpu,
                "n": len(v),
                "median_s": round(statistics.median(v), 4),
            }
            for (rnd, cpu), v in sorted(per.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))
        ]
    return out


def host_variability(run: str, cond: dict, rows: list[dict], cmp_: dict) -> dict:
    """Host identity observed per invocation, what was not observed, and whether the host mix can move the result."""
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    counts = {
        a: {
            cls: _count(r.get("cpu_part") for r in by_arm[a] if r["expected_class"] == cls)
            for cls in COMPARE
        }
        for a in ARMS
    }
    envs: dict[str, set] = defaultdict(set)
    for r in rows:
        env = (r.get("lkai") or {}).get("env_id")
        if env:
            envs[env].add(r.get("cpu_part"))
    logs = _log_hosts(run)
    fns = {a: cond["functions"][a]["FunctionName"] for a in ARMS}
    cold_parts = _count(r.get("cpu_part") for r in rows if r["expected_class"] == "cold")
    dominant = max(cold_parts, key=cold_parts.get) if cold_parts else None
    within = ((cmp_.get("cold") or {}).get("by_cpu_part") or {}).get(str(dominant)) or {}
    wd = within.get("weight_download") or {}
    # Warm round-to-round spread with and without the CPU-part mix (pooled vs one part only).
    lat = lambda r: r["latency_ms"] / 1000  # noqa: E731
    warm_cv = {
        a: {
            "all_cpu_parts": _round_cv(by_arm[a], "warm", lat),
            str(dominant): _round_cv(
                [r for r in by_arm[a] if r.get("cpu_part") == dominant], "warm", lat
            ),
        }
        for a in ARMS
    }
    round_cpu_primary = _round_cpu_primary(rows)
    rerun_needed = not (
        wd.get("n")
        and decide(wd.get("ratio_ci95")) == cmp_.get("outcome")
        and decide(wd.get("ratio_ci95_round_cluster")) == cmp_.get("outcome")
        and round_cpu_primary["all_inside_equivalence_band"]
    )
    return {
        "identity_field": "cpu_part: the /proc/cpuinfo 'CPU part' line read by the handler in each environment",
        "method": "compare the primary metric within CPU part and within shared (round, CPU part) cells; retain "
        "the pre-registered pooled result and paired round-cluster sensitivity",
        "observed": {
            "cpu_part_by_arm_and_class": counts,
            "execution_environments": len(envs),
            "environments_with_more_than_one_cpu_part": sum(len(v) > 1 for v in envs.values()),
            "ggml_cpu_variant_by_cpu_part": {
                a: logs.get(fns[a], {}).get("ggml_cpu_by_cpu_part") for a in ARMS
            },
            "snapshot_init_cpu_parts": {
                a: logs.get(fns[a], {}).get("snapshot_init_cpu_parts") for a in ARMS
            },
            "threads": sorted({(r.get("timing") or {}).get("threads") for r in rows}, key=str),
            "threads_source": "os.cpu_count() in the init that loaded the model (THREADS unset); a restored "
            "environment reports its snapshotting init's value",
            "memory_mb": sorted({r["report"].get("Memory Size") for r in rows}, key=str),
            "runtime_version_arn": cond.get("runtime_version_arn"),
            "log_streams": "logs/<function>.jsonl.gz, logStreamName per record; env_id per invocation",
        },
        "unobserved": list(HOST_UNOBSERVED),
        "generation": "unknown: not recorded, and not inferred from CPU part or timings",
        "primary_within_dominant_cpu_part": {
            "cpu_part": dominant,
            "n": wd.get("n"),
            "ratio_p50": wd.get("ratio_p50"),
            "ratio_ci95": wd.get("ratio_ci95"),
            "ratio_ci95_round_cluster": wd.get("ratio_ci95_round_cluster"),
            "outcome": decide(wd.get("ratio_ci95")),
            "outcome_round_cluster": decide(wd.get("ratio_ci95_round_cluster")),
            "agrees_with_primary": decide(wd.get("ratio_ci95")) == cmp_.get("outcome")
            and decide(wd.get("ratio_ci95_round_cluster")) == cmp_.get("outcome"),
        },
        "primary_pooled_by_cpu_part": _pooled_primary_by_cpu_part(rows),
        "primary_by_round_and_cpu_part": round_cpu_primary,
        "rerun_assessment": {
            "needed": rerun_needed,
            "decision": "no rerun" if not rerun_needed else "rerun affected comparison",
            "reason": "dominant-part primary and cluster intervals agree with the pooled no-change outcome, and "
            "every shared round/CPU-part cell is inside the +/-2% band"
            if not rerun_needed
            else "host-conditioned evidence does not preserve the pooled outcome and equivalence band",
            "limitation": "within-round signs may vary and sparse CPU parts have few or no shared cells",
        },
        "warm_round_cv": warm_cv,
        "warm_by_round_and_cpu_part": _warm_by_round_and_cpu_part(rows),
    }


def _count(xs) -> dict:
    out: dict = defaultdict(int)
    for x in xs:
        out[str(x) if x is not None else "unknown"] += 1
    return dict(sorted(out.items()))


# Every one-off charge of this run, by key in `cost()`'s dict. total_usd is the sum of exactly these. Monthly storage
# rates (`*_usd_month`) and P1's own total_usd (which base.costs() already put in the dict) are not charges to add.
CHARGE_KEYS = (
    "lambda_duration_usd",
    "lambda_requests_usd",
    "snapstart_restore_usd",
    "snapstart_cache_usd",
    "s3_request_usd",
    "s3_transfer_usd",
    "s3_retrieval_usd",
    "s3_onezone_ia_30_day_minimum_usd",
)


def download_cost(storage_class: str, size: int, gets: int, heads: int = 1) -> float:
    p = S3_PRICES[storage_class]
    return (gets + heads) * p["get_per_1000"] / 1000 + size / 1e9 * p["retrieval_per_gb"]


def cost(res: dict, cond: dict, rows: list[dict], snaps: dict[str, list[dict]], dep: dict) -> dict:
    """Lambda cost from P1's costs() over all rows; S3 cost per arm at that arm's storage class."""
    c = base.costs(cond, rows, sum(len(v) for v in snaps.values()) or None)
    size = cond["build_manifest"]["model"]["gguf_bytes"]
    gets = math.ceil(size / base.CHUNK)
    arms: dict = {}
    for arm in ARMS:
        sc = dep["weights"][arm]["storage_class"]
        colds = sum(1 for r in rows if r["arm"] == arm and "Init Duration" in r["report"])
        n_snap = len(snaps.get(arm, []))
        downloads = colds + n_snap
        arms[arm] = {
            "storage_class": sc,
            "cold_starts": colds,
            "snapshot_inits": n_snap,
            "weight_downloads": downloads,
            "restores_read_s3": False,
            "s3_get_requests": downloads * gets,
            "s3_head_requests": downloads,
            "s3_request_usd": round(
                downloads * (gets + 1) * S3_PRICES[sc]["get_per_1000"] / 1000, 6
            ),
            "s3_retrieval_usd": round(
                downloads * size / 1e9 * S3_PRICES[sc]["retrieval_per_gb"], 4
            ),
            "s3_gb_transferred": round(downloads * size / 1e9, 2),
            "s3_storage_usd_month": round(size / 1e9 * S3_PRICES[sc]["storage_gb_month"], 4),
        }
    per_dl = {sc: round(download_cost(sc, size, gets), 6) for sc in S3_PRICES if sc != "source"}
    std_store = size / 1e9 * S3_PRICES["STANDARD"]["storage_gb_month"]
    oz_store = size / 1e9 * S3_PRICES["ONEZONE_IA"]["storage_gb_month"]
    saving = std_store - oz_store
    extra = per_dl["ONEZONE_IA"] - per_dl["STANDARD"]
    model = {
        "object_gb": round(size / 1e9, 4),
        "requests_per_download": {"get": gets, "head": 1},
        "s3_usd_per_download": per_dl,
        "s3_usd_per_1000_cold_starts": {k: round(v * 1000, 3) for k, v in per_dl.items()},
        "storage_usd_month": {
            sc: round(size / 1e9 * S3_PRICES[sc]["storage_gb_month"], 4)
            for sc in S3_PRICES
            if sc != "source"
        },
        "onezone_ia_breakeven_downloads_per_month": round(saving / extra, 2) if extra > 0 else None,
        "onezone_ia_minimum_charge_usd": round(oz_store, 4),
        "note": "A restore reads no S3 bytes (the snapshot holds the memfd), so only cold starts and snapshot "
        "inits pay retrieval. Express One Zone figures are list prices; E1 measures its own run.",
    }
    s3_req = sum(a["s3_request_usd"] for a in arms.values())
    s3_ret = sum(a["s3_retrieval_usd"] for a in arms.values())
    c.update(
        s3_request_usd=round(s3_req, 6),
        s3_retrieval_usd=round(s3_ret, 4),
        s3_gb_transferred=round(sum(a["s3_gb_transferred"] for a in arms.values()), 2),
        s3_storage_usd_month=round(sum(a["s3_storage_usd_month"] for a in arms.values()), 4),
        weight_downloads=sum(a["weight_downloads"] for a in arms.values()),
        snapshot_inits=sum(a["snapshot_inits"] for a in arms.values()),
        by_arm=arms,
        per_storage_class=model,
        s3_prices=S3_PRICES,
    )
    # The One Zone-IA copy is billed for at least 30 days however soon it is deleted (USE2-EarlyDelete-ZIA).
    c["s3_onezone_ia_30_day_minimum_usd"] = round(oz_store, 4)
    c.pop("s3_get_requests_per_download", None)
    c.pop("s3_head_requests_per_download", None)
    # base.costs()'s total (P1's components plus a month of storage) is recomputed below.
    c.pop("total_usd", None)
    unknown = sorted(k for k in c if k.endswith("_usd") and k not in CHARGE_KEYS)
    if unknown:
        raise ValueError(f"cost keys not classified as charge or rate: {unknown}")
    c["total_components"] = list(CHARGE_KEYS)
    c["total_usd"] = round(sum(c[k] for k in CHARGE_KEYS), 6)
    return c


def answer_hash(rows: list[dict], arm: str) -> dict:
    """SHA-256 over the arm's distinct answers per (request, CPU part), canonical JSON; plus how many differed."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["arm"] == arm and not r["error"] and r.get("answers"):
            k = f"{r['request_id']}|{r.get('cpu_part')}"
            seen[k].add(json.dumps(r["answers"], sort_keys=True, separators=(",", ":")))
    canon = {k: sorted(v) for k, v in sorted(seen.items())}
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "cells": len(canon),
        "cells_with_more_than_one_answer": sum(len(v) > 1 for v in canon.values()),
    }


def correctness(rows: list[dict], ref: dict, dep: dict, gguf_sha: str) -> dict:
    per = {a: base.compare([r for r in rows if r["arm"] == a], ref) for a in ARMS}
    tot = {
        k: sum(p[k] for p in per.values())
        for k in (
            "compared",
            "bit_identical",
            "choice_flips",
            "answer_shape_mismatches",
            "no_reference_for_cpu_part",
        )
    }
    hashes = {a: answer_hash(rows, a) for a in ARMS}
    return {
        **tot,
        "max_abs_delta": max(p["max_abs_delta"] for p in per.values()),
        "method": per["std"]["method"],
        "by_arm": per,
        "answer_sha256": hashes,
        "weights": {
            a: {
                "storage_class": dep["weights"][a]["storage_class"],
                "s3_checksum_sha256_b64": dep["weights"][a]["checksum_sha256_b64"],
                "checksum_type": dep["weights"][a]["checksum_type"],
                "sha256_hex": gguf_sha,
                "matches_p1_manifest": dep["weights"][a]["sha256_matches_manifest"],
            }
            for a in ARMS
        },
    }


def summarize(
    run: str,
    ref_path: str,
    dep_path: str,
    baseline_commit: str,
    meta: dict,
    experiment_commit: str | None = None,
    experiment_dirty: bool | None = None,
) -> dict:
    res = base.summarize(run, ref_path, baseline_commit, meta)
    # base.summarize names the zip's build commit; E2 runs that zip unchanged, so the experiment commit is the
    # commit that ran the harness (this directory), and the zip's commit is kept beside it.
    zip_commit = res["commits"]["experiment"]
    if experiment_commit:
        res["commits"] = {
            "experiment": experiment_commit,
            "baseline": baseline_commit,
            "repo_dirty_at_build": res["commits"]["repo_dirty_at_build"],
            "experiment_dirty_at_run": bool(experiment_dirty),
            "function_zip_built_at": zip_commit,
        }
    cond, rows = base.load_run(run)
    dep = json.load(open(dep_path))
    versions: dict[str, set[str]] = defaultdict(set)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            versions[cond["functions"][arm]["FunctionName"]].add(str(pub["version"]))
    snaps_fn = base.snapshot_inits(run, versions)
    snaps = {a: snaps_fn.get(cond["functions"][a]["FunctionName"], []) for a in ARMS}
    ref = json.load(open(ref_path))["outputs"]
    gguf = cond["build_manifest"]["model"]["gguf_sha256"]
    cmp_ = comparison(rows, snaps)
    res["primary_arm"] = "ozia"
    res["comparison"] = cmp_
    res["host_variability"] = host_variability(run, cond, rows, cmp_)
    res["cost"] = cost(res, cond, rows, snaps, dep) | {
        "snapshot_inits_per_version": res["cost"].get("snapshot_inits_per_version", {})
    }
    res["correctness"] = correctness(rows, ref, dep, gguf) | {
        "reference": os.path.relpath(ref_path, BASE)
    }
    res["deployment"]["weights"] = {
        a: {"storage_class": dep["weights"][a]["storage_class"], "key": dep["weights"][a]["key"]}
        for a in ARMS
    }
    res["deployment"]["bucket_type"] = "general purpose (both arms, same bucket)"
    # P1's instrumentation-overhead block compares inst with rep; both E2 arms are instrumented.
    res["instrumentation_overhead"] = {
        "method": "not measured here: both arms run P1's instrumented configuration (LKAI_PHASES=1); "
        "P1 run 20261020T080835Z measured the overhead",
    }
    if meta.get("status") in (None, "auto"):
        res["status"] = cmp_["outcome"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference", default=os.path.join(BASE, "reference", "outputs.json"))
    ap.add_argument("--deployment", default=os.path.join(HERE, "build", "deployment.json"))
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--meta", default=os.path.join(HERE, "meta.json"))
    ap.add_argument("--experiment-commit")
    ap.add_argument("--experiment-dirty", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    res = summarize(
        a.run,
        a.reference,
        a.deployment,
        a.baseline_commit,
        meta,
        a.experiment_commit,
        a.experiment_dirty,
    )
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    w = res["comparison"]["cold"]["weight_download"] or {}
    print(
        "wrote",
        path,
        "status",
        res["status"],
        "cold weight_download ratio",
        w.get("ratio_p50"),
        w.get("ratio_ci95"),
    )


if __name__ == "__main__":
    main()
