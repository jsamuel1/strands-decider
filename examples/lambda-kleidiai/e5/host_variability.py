"""Host-variability assessment of an E5 run (initiative HOST VARIABILITY rules), from the records the run already holds.

  python host_variability.py <run-dir>            # prints the assessment as JSON

No new invocations. It answers one question: could the CPU an execution environment landed on change an E5
conclusion? It reports

  - metadata: every host/environment field the HOST VARIABILITY rules ask for, with how this run observed it, or
    "unknown" when the run did not record it. CPU part ids are reported as observed; no Graviton generation is
    inferred from a part id or from timings.
  - mix: invocations and distinct execution environments per (class, arm, CPU part), and whether any environment
    reported more than one part.
  - by_class: per class and metric, each CPU part both arms drew: n, p50, min, max per arm, the e5 - base p50
    delta, the per-round p50s and in how many rounds e5 was below base. Next to it, the unmatched (all parts
    pooled) delta, to show how much the CPU mix alone moves a comparison.
  - verdict: per class, whether the delta has the same sign in every part with at least MIN_N samples per arm.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

MIN_N = 3

# Fields the HOST VARIABILITY rules list, and how (whether) bench.py / the handler recorded them in this run.
METADATA = {
    "cpu_part": "observed: /proc/cpuinfo 'CPU part', read by the handler in every environment (row cpu_part)",
    "ggml_cpu_variant": "observed: the libggml-cpu variants mapped in /proc/self/maps after init (engine init log "
    "line). ggml picks the variant from the CPU features it detects; the record names the library only, not the "
    "features",
    "threads": "configured, not observed: 6 llama.cpp threads (row timing.threads)",
    "memory_size_mb": "observed: REPORT Memory Size",
    "runtime_version": "observed: INIT_START Runtime Version ARN, pinned (conditions.runtime_version_arn)",
    "architecture": "configured: arm64 (conditions.functions.*.Architectures)",
    "region": "configured: conditions.region",
    "execution_environment": "observed: row lkai.env_id and the CloudWatch log stream name (logs/*.jsonl.gz)",
    "cpu_implementer": "unknown (not recorded)",
    "cpu_variant": "unknown (not recorded)",
    "cpu_revision": "unknown (not recorded)",
    "midr": "unknown (not recorded)",
    "cpu_feature_flags": "unknown (not recorded; /proc/cpuinfo Features was not read)",
    "hwcap_hwcap2": "unknown (not recorded)",
    "online_core_count": "unknown (not recorded)",
    "cgroup_cpu_memory_limits": "unknown (not recorded)",
    "kernel": "unknown (not recorded)",
    "availability_zone": "unknown (not recorded)",
    "graviton_generation": "unknown: this run recorded nothing that names a generation; never inferred from a "
    "part id or from timings",
}


def _first_request(r: dict) -> float:
    rp = r["report"]
    return (rp.get("Init Duration", 0) + rp.get("Restore Duration", 0) + rp["Duration"]) / 1000


METRICS = {
    "cold": {
        "first_request_total": _first_request,
        "model_load_mmap": lambda r: (r.get("lkai") or {}).get("init", {}).get("model_load_mmap"),
        "max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
    },
    "restore": {
        "first_request_total": _first_request,
        "max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
    },
    "warm": {
        "steady_state_inference": lambda r: (
            r["latency_ms"] / 1000 if r.get("latency_ms") is not None else None
        ),
        "max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
    },
}


def _stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "p50": round(statistics.median(xs), 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
    }


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("error") and not r.get("class_mismatch")]


def mix(rows: list[dict]) -> dict:
    inv: dict = defaultdict(int)
    envs: dict = defaultdict(set)
    parts_by_env: dict = defaultdict(set)
    for r in rows:
        key = f"{r['expected_class']}:{r['arm']}:{r.get('cpu_part')}"
        inv[key] += 1
        env = (r.get("lkai") or {}).get("env_id")
        envs[key].add(env)
        parts_by_env[env].add(r.get("cpu_part"))
    return {
        "invocations": dict(sorted(inv.items())),
        "environments": {k: len(v) for k, v in sorted(envs.items())},
        "environments_total": len(parts_by_env),
        "environments_with_more_than_one_cpu_part": sum(
            1 for v in parts_by_env.values() if len(v) > 1
        ),
    }


def _vals(rs: list[dict], f, arm: str, part, rd=None) -> list[float]:
    return [
        v
        for r in rs
        if r["arm"] == arm and r.get("cpu_part") == part and (rd is None or r["round"] == rd)
        if (v := f(r)) is not None
    ]


def compare(rows: list[dict], a: str = "e5", b: str = "base") -> dict:
    ok = _ok(rows)
    rounds = sorted({r["round"] for r in ok})
    out: dict = {}
    for cls, metrics in METRICS.items():
        rs = [r for r in ok if r["expected_class"] == cls]
        parts = sorted(
            {r.get("cpu_part") for r in rs if r["arm"] == a}
            & {r.get("cpu_part") for r in rs if r["arm"] == b},
            key=str,
        )
        cls_out: dict = {}
        for name, f in metrics.items():
            per: dict = {}
            for part in parts:
                xa, xb = _vals(rs, f, a, part), _vals(rs, f, b, part)
                if not xa or not xb:
                    continue
                per_round, below = {}, 0
                for rd in rounds:
                    ra, rb = _vals(rs, f, a, part, rd), _vals(rs, f, b, part, rd)
                    if ra and rb:
                        ma, mb = statistics.median(ra), statistics.median(rb)
                        per_round[str(rd)] = {a: round(ma, 6), b: round(mb, 6)}
                        below += ma < mb
                sa, sb = _stats(xa), _stats(xb)
                per[str(part)] = {
                    a: sa,
                    b: sb,
                    "delta_p50": round(sa["p50"] - sb["p50"], 6),
                    "per_round_p50": per_round,
                    f"rounds_{a}_below_{b}": [below, len(per_round)],
                }
            ua = [v for r in rs if r["arm"] == a if (v := f(r)) is not None]
            ub = [v for r in rs if r["arm"] == b if (v := f(r)) is not None]
            if per:
                cls_out[name] = {
                    "by_cpu_part": per,
                    "unmatched_delta_p50": round(statistics.median(ua) - statistics.median(ub), 6)
                    if ua and ub
                    else None,
                }
        out[cls] = cls_out
    return out


def verdict(by_class: dict, a: str = "e5", b: str = "base") -> dict:
    """Per class and metric: the signs of the per-part deltas among parts with >= MIN_N samples on both arms."""
    out: dict = {}
    for cls, ms in by_class.items():
        for name, m in ms.items():
            signs = {
                part: (1 if c["delta_p50"] > 0 else -1 if c["delta_p50"] < 0 else 0)
                for part, c in m["by_cpu_part"].items()
                if c[a]["n"] >= MIN_N and c[b]["n"] >= MIN_N
            }
            small = sorted(p for p in m["by_cpu_part"] if p not in signs)
            out[f"{cls}.{name}"] = {
                "parts_with_n_ge_min": sorted(signs),
                "parts_below_min_n": small,
                "same_sign_in_every_part": len(set(signs.values())) <= 1 if signs else None,
            }
    return out


def log_streams(run: str) -> dict:
    out = {}
    for f in (
        sorted(os.listdir(os.path.join(run, "logs")))
        if os.path.isdir(os.path.join(run, "logs"))
        else []
    ):
        if f.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(run, "logs", f), "rt") as fh:
                out[f.split(".", 1)[0]] = len(
                    {json.loads(line)["logStreamName"] for line in fh if line.strip()}
                )
    return out


def assess(run: str) -> dict:
    rows = [
        json.loads(line) for line in open(os.path.join(run, "invocations.jsonl")) if line.strip()
    ]
    by_class = compare(rows)
    variants: dict = defaultdict(set)
    for r in rows:
        for line in (r.get("log_tail") or "").splitlines():
            if "decider.engine init" in line and "ggml_cpu=" in line:
                variants[str(r.get("cpu_part"))].add(line.split("ggml_cpu=", 1)[1].split()[0])
    return {
        "schema": "lkai-host-variability/1",
        "method": (
            "post-run, from this run's own records; no new invocations. Comparisons are within one observed CPU "
            f"part; a part counts toward the verdict when both arms have >= {MIN_N} samples in it. Arm order "
            "alternates between rounds (conditions.rounds_meta[].order). Within a round the cold and restore waves "
            "run one arm after the other and warm passes alternate arm order pass by pass, so time-within-round "
            "drift remains a limitation for cold and restore."
        ),
        "metadata": METADATA,
        "ggml_cpu_variant_by_cpu_part": {k: sorted(v) for k, v in sorted(variants.items())},
        "log_streams_per_function": log_streams(run),
        "mix": mix(_ok(rows)),
        "by_class": by_class,
        "verdict": verdict(by_class),
    }


if __name__ == "__main__":
    json.dump(assess(sys.argv[1]), sys.stdout, indent=1)
    print()
