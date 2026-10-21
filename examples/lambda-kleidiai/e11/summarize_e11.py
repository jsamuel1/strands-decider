"""E11 summary: one lkai-result/1 file for the `rel` arm (LKAI_RELEASE_MEMFD=1) against `off` from the same run.

  python summarize_e11.py <run-dir> --run-id <id> --baseline-commit <P1 sha> \
      --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

Every row-level computation is P1's summarize.py (load_run, arm_phases, compare, costs, round_medians), so the
distributions are the blocks P1 publishes. E11 adds:
- `e11.delta`: rel minus off per headline metric, matched by CPU part (/proc/cpuinfo part ID, an opaque label),
  with a bootstrap 95% CI over environments.
- `host_variability`: the same metrics per shared (round, CPU part) cell, each delta compared with the off arm's
  within-cell spread (E6's method). Rows without a CPU part are counted, never a stratum.
- `e11.memory`: the mechanism, per arm and stage (after load, before_snapshot, first invocation): VmRSS, RssAnon,
  RssShmem, the memfd's resident bytes and the loader's release stats, from e11_entry's records.
- `e11.host`: every CPU identity and environment field e11_entry read, with counts per arm.
- `e11.arm_agreement`: rel's answers against off's for the same request and CPU part.
Status: `improved` if a deployment headline's CI lies entirely below 0 on every CPU part both arms ran on and the
answers and errors are clean; `regressed` if any headline CI lies entirely above 0; `inconclusive` if answers are
not clean; `no-change` otherwise. Memory is reported as a resource observation, not a headline.
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

CONTROL, EXPERIMENT = "off", "rel"
UNKNOWN_CPU = (None, "", "unknown", "?")


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _total(r: dict, k: str) -> float | None:
    if k not in r["report"]:
        return None
    return (r["report"][k] + r["report"].get("Duration", 0)) / 1000


def _lat(r: dict) -> float | None:
    v = r.get("latency_ms")
    return v / 1000 if isinstance(v, (int, float)) else None


def _mem(r: dict) -> float | None:
    return r["report"].get("Max Memory Used")


KEYS = {
    "cold": {
        "first_request_total": lambda r: _total(r, "Init Duration"),
        "report_init_duration": lambda r: _rep(r, "Init Duration"),
        "model_load_mmap": lambda r: _init(r, "model_load_mmap"),
        "model_ready": lambda r: _init(r, "model_ready"),
        "first_request_inference": _lat,
        "report_max_memory_mb": _mem,
    },
    "restore": {
        "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
        "first_request_total": lambda r: _total(r, "Restore Duration"),
        "first_request_inference": _lat,
        "report_duration": lambda r: _rep(r, "Duration"),
        "report_max_memory_mb": _mem,
    },
    "warm": {"steady_state_inference": _lat, "report_max_memory_mb": _mem},
}
HEADLINES = (
    ("cold", "first_request_total"),
    ("restore", "first_request_total"),
    ("restore", "report_restore_duration"),
    ("warm", "steady_state_inference"),
)


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 11) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling each arm independently."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def _vals(rs, f, **kv) -> list[float]:
    return [v for r in rs if all(r.get(k) == x for k, x in kv.items()) if (v := f(r)) is not None]


def delta(rows: list[dict], cls: str, keys: dict) -> dict:
    """rel minus off per metric within each known CPU part both arms drew, plus descriptive per-round deltas."""
    rs = [r for r in ok_rows(rows, cls) if r.get("cpu_part") not in UNKNOWN_CPU]
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted({str(r["cpu_part"]) for r in rs}):
            a = _vals(rs, f, arm=CONTROL, cpu_part=cpu)
            b = _vals(rs, f, arm=EXPERIMENT, cpu_part=cpu)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[cpu] = {
                "off_p50": round(ma, 6),
                "rel_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        if per:
            out[name] = {"by_cpu_part": per}
    return out


def cells(rows: list[dict], cls: str, f) -> dict:
    """Per shared (round, CPU part) cell: per-arm n, median, range, off's within-cell spread, delta and sign."""
    rs = ok_rows(rows, cls)
    unknown = {
        arm: sum(r.get("cpu_part") in UNKNOWN_CPU for r in rs if r["arm"] == arm)
        for arm in (CONTROL, EXPERIMENT)
    }
    known = [r for r in rs if r.get("cpu_part") not in UNKNOWN_CPU]
    shared, missing = [], []
    for rnd, cpu in sorted({(r["round"], str(r["cpu_part"])) for r in known}):
        v = {
            arm: _vals(known, f, arm=arm, round=rnd, cpu_part=cpu) for arm in (CONTROL, EXPERIMENT)
        }
        if not all(v.values()):
            missing.append({"round": rnd, "cpu_part": cpu, "n": {a: len(x) for a, x in v.items()}})
            continue
        mo, mr = statistics.median(v[CONTROL]), statistics.median(v[EXPERIMENT])
        d = mr - mo
        q = statistics.quantiles(v[CONTROL], n=4) if len(v[CONTROL]) >= 4 else None
        shared.append(
            {
                "round": rnd,
                "cpu_part": cpu,
                "n": {a: len(x) for a, x in v.items()},
                "median": {CONTROL: round(mo, 6), EXPERIMENT: round(mr, 6)},
                "range": {a: [round(min(x), 6), round(max(x), 6)] for a, x in v.items()},
                "control_spread": round(max(v[CONTROL]) - min(v[CONTROL]), 6)
                if len(v[CONTROL]) >= 2
                else None,
                "control_iqr": round(q[2] - q[0], 6) if q else None,
                "delta": round(d, 6),
                "ratio": round(mr / mo, 6) if mo else None,
                "sign": "negative" if d < 0 else "positive" if d > 0 else "zero",
            }
        )
    return {
        "class": cls,
        "shared_cells": shared,
        "missing_cells": missing,
        "unknown_cpu_rows": unknown,
    }


def part_outcome(cs: list[dict]) -> dict:
    """One CPU part's outcome: every assessed cell's delta against off's within-cell spread."""
    assessed = [c for c in cs if c["control_spread"] is not None]
    outs = []
    for c in assessed:
        s, d = c["control_spread"], c["delta"]
        outs.append(
            "within-control-variation"
            if abs(d) <= s
            else "negative-beyond-control-variation"
            if d < 0
            else "positive-beyond-control-variation"
        )
    if not assessed:
        outcome = "insufficient-control-cell-spread"
    elif len(set(outs)) == 1:
        outcome = outs[0]
    else:
        outcome = "mixed-or-overlapping-control-variation"
    ds = [c["delta"] for c in cs]
    # Sensitivity, added after the first run was seen and labelled as such: the off arm's within-cell range is set
    # by its slowest restore (one outlier makes a 6 s range in a 10-sample cell), so the same test against the
    # interquartile range is reported beside it. The pre-registered outcome above stays the primary one.
    iq = [c for c in cs if c.get("control_iqr") is not None]
    iq_outs = {
        "within"
        if abs(c["delta"]) <= c["control_iqr"]
        else "negative"
        if c["delta"] < 0
        else "positive"
        for c in iq
    }
    return {
        "shared_cell_count": len(cs),
        "assessed_cell_count": len(assessed),
        "unassessed_rounds": [c["round"] for c in cs if c["control_spread"] is None],
        "delta_range": [min(ds), max(ds)] if ds else None,
        "delta_signs": sorted({c["sign"] for c in cs}),
        "outcome": outcome,
        "post_hoc_iqr_outcome": (
            "insufficient"
            if not iq
            else f"{next(iter(iq_outs))}-beyond-control-iqr"
            if len(iq_outs) == 1 and "within" not in iq_outs
            else "within-control-iqr"
            if iq_outs == {"within"}
            else "mixed"
        ),
    }


def sign_test(cs: list[dict]) -> dict:
    """Exact two-sided sign test over shared cells (each cell one paired observation; zero deltas dropped)."""
    from math import comb

    neg = sum(c["delta"] < 0 for c in cs)
    pos = sum(c["delta"] > 0 for c in cs)
    n = neg + pos
    if n == 0:
        return {"cells": 0, "negative": 0, "positive": 0, "p_two_sided": None}
    k = min(neg, pos)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"cells": n, "negative": neg, "positive": pos, "p_two_sided": round(p, 6)}


def host_variability(rows: list[dict], cond: dict) -> dict:
    specs = {f"{cls}_{name}": (cls, KEYS[cls][name]) for cls, name in HEADLINES}
    specs["restore_report_max_memory_mb"] = ("restore", KEYS["restore"]["report_max_memory_mb"])
    metrics = {}
    for name, (cls, f) in specs.items():
        m = cells(rows, cls, f)
        by_part = defaultdict(list)
        for c in m["shared_cells"]:
            by_part[c["cpu_part"]].append(c)
        m["by_cpu_part"] = {cpu: part_outcome(cs) for cpu, cs in sorted(by_part.items())}
        metrics[name] = m
    assessment = {}
    for name, m in metrics.items():
        outs = {cpu: p["outcome"] for cpu, p in m["by_cpu_part"].items()}
        assessed = {c: o for c, o in outs.items() if o != "insufficient-control-cell-spread"}
        consistent = (
            bool(assessed)
            and len(set(assessed.values())) == 1
            and not next(iter(set(assessed.values()))).startswith("mixed")
        )
        assessment[name] = {
            "by_cpu_part": outs,
            "consistent_across_parts": consistent,
            "outcome": next(iter(set(assessed.values())))
            if consistent
            else ("unassessed" if not assessed else "host-conditioned-or-mixed"),
            "unassessed_cpu_parts": sorted(set(outs) - set(assessed)),
            "unknown_cpu_rows": m["unknown_cpu_rows"],
            "post_hoc_iqr_by_cpu_part": {
                cpu: p["post_hoc_iqr_outcome"] for cpu, p in m["by_cpu_part"].items()
            },
            "sign_test_over_shared_cells": sign_test(m["shared_cells"]),
        }
    return {
        "schema": "lkai-host-variability/1",
        "observed": {"primary_metrics": metrics},
        "unknown": {
            "environment_fields": ["availability zone", "physical host", "snapshot-build host"],
            "note": "per-environment cpuinfo, HWCAPs, CPU count, affinity, cgroup limits and kernel are in "
            "e11.host where the sandbox exposed them; 'unknown' entries there were not readable",
        },
        "method": "Within each observed CPU part ID (an opaque label; no processor generation is inferred), compare "
        "arm medians in each round. "
        + cond.get("schedule", "")
        + ". A directional effect must exceed the off "
        "arm's within-cell range in every assessed shared cell; deltas inside it support a null outcome even if "
        "signs vary. Cells with fewer than two off observations are unassessed. Rows without a CPU part are "
        "counted, never a stratum.",
        "assessment": assessment,
    }


def _stage_records(rows: list[dict], snaps_raw: dict) -> dict:
    """Per arm: the first-invocation E11 records (and their after_load sub-record), and before_snapshot records."""
    out: dict = {}
    for arm in (CONTROL, EXPERIMENT):
        first = [
            ((r.get("lkai") or {}).get("e11")) for r in rows if r["arm"] == arm and not r["error"]
        ]
        first = [e for e in first if e]
        out[arm] = {"first_invoke": first, "before_snapshot": snaps_raw.get(arm, [])}
    return out


def _mem_summary(recs: list[dict]) -> dict | None:
    if not recs:
        return None

    def field(f):
        xs = [f(r) for r in recs]
        xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return p1.dist(xs, unit="MB") if xs else None

    def st(k, block="status_kb"):
        def get(r):
            v = (r["mem"].get(block) or {}).get(k)
            return v / 1024 if isinstance(v, int) else None

        return get

    def memfd_res(r):
        v = [x.get("resident_bytes") for x in (r.get("memfd") or {}).values()]
        v = [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
        return sum(v) / 2**20 if v else None

    rel = [r.get("release") for r in recs if isinstance(r.get("release"), dict)]
    return {
        "n": len(recs),
        "VmRSS": field(st("VmRSS")),
        "RssAnon": field(st("RssAnon")),
        "RssShmem": field(st("RssShmem")),
        "meminfo_Shmem": field(st("Shmem", "meminfo_kb")),
        "meminfo_MemAvailable": field(st("MemAvailable", "meminfo_kb")),
        "memfd_resident": field(memfd_res),
        "release_ranges_via_madvise_remove": sorted(
            Counter(x.get("ranges_via_madvise_remove") for x in rel).items(), key=str
        ),
        "release_released_bytes": sorted(Counter(x["released_bytes"] for x in rel).items()),
        "release_kept_bytes": sorted(Counter(x["kept_bytes"] for x in rel).items()),
        "release_errno": sorted(Counter(x["errno"] for x in rel).items()),
        "release_us": p1.dist([x["release_us"] / 1e6 for x in rel if x["release_us"]])
        if any(x["release_us"] for x in rel)
        else None,
    }


def memory(rows: list[dict], snaps_raw: dict) -> dict:
    recs = _stage_records(rows, snaps_raw)
    out = {}
    for arm, d in recs.items():
        fi = d["first_invoke"]
        out[arm] = {
            "before_snapshot": _mem_summary(d["before_snapshot"]),
            "first_invoke_cold": _mem_summary([e for e in fi if e.get("init_type") == "on-demand"]),
            "first_invoke_restored": _mem_summary(
                [e for e in fi if e.get("init_type") == "snap-start"]
            ),
        }
    out["note"] = (
        "MB = MiB of /proc/self/status, of the sandbox's /proc/meminfo, and (offline only; Lambda refuses mincore "
        "with EPERM) of the memfd's resident pages. RssShmem counts only mapped shmem; meminfo Shmem counts the "
        "memfd's pages whether mapped or not, so it is the gauge of what the snapshot holds. before_snapshot is "
        "the state SnapStart captures; first_invoke_restored is the restored image, measured after P1's handler "
        "returned."
    )
    return out


def e11_log_records(run: str, cond: dict) -> dict:
    """`E11 {json}` before_snapshot lines from the saved CloudWatch logs, per arm (only this run's versions)."""
    import gzip

    names = {cond["functions"][a]["FunctionName"]: a for a in cond["arms"]}
    versions = defaultdict(set)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            versions[arm].add(str(pub["version"]))
    out: dict = defaultdict(list)
    d = os.path.join(run, "logs")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        arm = names.get(f.removesuffix(".jsonl.gz"))
        if arm is None:
            continue
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                e = json.loads(line)
                stream = e.get("logStreamName", "")
                ver = stream.split("[", 1)[1].split("]", 1)[0] if "[" in stream else None
                if ver not in versions[arm]:
                    continue
                for part in e["message"].splitlines():
                    if part.startswith("E11 "):
                        try:
                            rec = json.loads(part[4:])
                        except ValueError:
                            continue
                        if rec.get("kind") == "before_snapshot":
                            out[arm].append(rec | {"version": ver})
    return out


def host(rows: list[dict]) -> dict:
    ids: dict = defaultdict(Counter)
    for r in rows:
        e = (r.get("lkai") or {}).get("e11")
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
                "cgroup": e.get("cgroup"),
                "kernel": e.get("kernel"),
                "python": e.get("python"),
                "execution_env": e.get("execution_env"),
            },
            sort_keys=True,
        )
        ids[r["arm"]][key] += 1
    return {
        "identities": {
            arm: [{"count": n, **json.loads(k)} for k, n in cnt.most_common()]
            for arm, cnt in ids.items()
        },
        "note": "One record per environment's first invocation, read inside the function from /proc/cpuinfo, "
        "/proc/self/auxv, os.sched_getaffinity and /sys/fs/cgroup. 'unknown' = not exposed. No processor "
        "generation is inferred from any field.",
    }


def arm_agreement(rows: list[dict]) -> dict:
    ref: dict = {}
    for r in rows:
        if r["arm"] == CONTROL and not r["error"]:
            ref.setdefault((r["request_id"], r.get("cpu_part")), r["answers"])
    off_self = sum(
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
        "no_off_answer_for_cpu_part": missing,
        "off_internal_disagreements": off_self,
        "method": "complete answer object of every successful rel invocation vs the off answer for the same "
        "request id and CPU part in this run",
    }


def read_timeouts(rows: list[dict]) -> dict:
    """client:ReadTimeoutError and other client errors by arm and wave (the first-restore-wave stall check)."""
    c = Counter(
        (r["arm"], r["wave"], r["error"])
        for r in rows
        if r["error"] and r["error"].startswith("client:")
    )
    return {f"{a}:{w}:{e}": n for (a, w, e), n in sorted(c.items())}


def status_of(res: dict) -> str:
    d = res["e11"]["delta"]
    heads = [d.get(cls, {}).get(name, {}).get("by_cpu_part", {}) for cls, name in HEADLINES]
    c = res["correctness"] or {}
    g = res["e11"]["arm_agreement"]
    clean = (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["no_off_answer_for_cpu_part"] == 0
        and g["off_internal_disagreements"] == 0
    )
    if not any(heads) or not clean:
        return "inconclusive"
    if any(v["ci95"][0] > 0 for h in heads for v in h.values()):
        return "regressed"
    if any(h and all(v["ci95"][1] < 0 for v in h.values()) for h in heads):
        return "improved"
    return "no-change"


def fmt(res: dict) -> dict:
    out = {}
    for cls, metrics in res["e11"]["delta"].items():
        if not isinstance(metrics, dict):
            continue
        for name, d in metrics.items():
            for cpu, v in d.get("by_cpu_part", {}).items():
                k = f"{cls}_{name}_{cpu}"
                out[k + "_off"], out[k + "_rel"], out[k + "_d"] = (
                    v["off_p50"],
                    v["rel_p50"],
                    v["delta_p50"],
                )
                out[k + "_lo"], out[k + "_hi"] = v["ci95"]
                out[k + "_n"] = v["n"]
    return out


def with_assets(run: str) -> None:
    """Runs whose manifest predates manifest_e11's assets_sha256 (20261021T103834Z): fill it from the deployed zip's
    assets, which are P1's (verified by sha256 when the run was summarized), so P1's summarize can read it."""
    path = os.path.join(run, "conditions.json")
    cond = json.load(open(path))
    m = cond["build_manifest"]
    if "assets_sha256" in m:
        return
    z = os.path.join(HERE, "build", "function.zip")
    import hashlib
    import zipfile

    h = hashlib.sha256(open(z, "rb").read()).hexdigest()
    if h != m["function_zip"]["sha256"]:
        sys.exit(
            f"{z} is not the deployed zip ({h} vs {m['function_zip']['sha256']}); cannot fill assets_sha256"
        )
    with zipfile.ZipFile(z) as zf:
        m["assets_sha256"] = {
            n.split("/", 1)[1]: hashlib.sha256(zf.read(n)).hexdigest()
            for n in sorted(zf.namelist())
            if n.startswith("assets/") and not n.endswith("/")
        }
    m["assets_sha256_note"] = (
        "filled by summarize_e11 from the deployed function.zip (sha256 checked)"
    )
    json.dump(cond, open(path, "w"), indent=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta))
    with_assets(a.run)
    p1.ARM_PHASES = {
        CONTROL: "1",
        EXPERIMENT: "1",
    }  # both arms are instrumented; the control is `off`, not rep
    res = p1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = p1.load_run(a.run)
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = EXPERIMENT
    res["commits"]["experiment"] = cond["build_manifest"]["repo_commit"]
    res["commits"]["p1_build"] = cond["build_manifest"].get("p1")
    if cond["build_manifest"].get("schema") == "lkai-e11-manifest/1":
        # P1's summarize labels any manifest other than its own /2 as /1 (tracked files only). E11's ran
        # buildcache.dirty (tracked, untracked and deleted inputs) plus the e11/ inputs (manifest_e11.py).
        res["commits"]["provenance_check"] = (
            "all executable inputs vs commit (buildcache.dirty) plus e11/ inputs (manifest_e11.e11_dirty)"
        )
    res["deployment"]["environment"] = {
        arm: cond["functions"][arm]["environment"] for arm in cond["arms"]
    }
    res["build"] = {
        k: cond["build_manifest"].get(k)
        for k in ("images", "toolchain", "function_zip", "patch_memfdrel_sha256", "p1")
    }
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["schedule"] = cond.get("schedule")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "both E11 arms run P1's phase timers and e11_entry's memory records; P1 measured the timer "
        "overhead (P1-baseline/20261020T080835Z)",
    }
    snaps_raw = e11_log_records(a.run, cond)
    res["host_variability"] = host_variability(rows, cond)
    res["e11"] = {
        "delta": {cls: delta(rows, cls, KEYS[cls]) for cls in KEYS}
        | {
            "method": "rel minus off, medians within each CPU part both arms ran on; ci95 = bootstrap over "
            "environments (4,000 resamples, seed 11)"
        },
        "memory": memory(rows, snaps_raw),
        "host": host(rows),
        "arm_agreement": arm_agreement(rows),
        "client_errors": read_timeouts(rows),
    }
    res["status"] = status_of(res)
    res["conclusion"] = (
        meta["conclusion_template"].format(**fmt(res))
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
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
