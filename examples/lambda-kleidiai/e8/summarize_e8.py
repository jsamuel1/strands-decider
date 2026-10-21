"""E8 summary: one lkai-result/1 file for the par arm, from one interleaved three-arm run (base, off, par).

  python summarize_e8.py <run-dir> --run-id <id> --baseline-commit <P1 sha> \
      --reference ../baseline/reference/outputs.json --meta meta.json [--hash-dir e8/build/hash]

Uses P1's summarize.py for every row-level computation (load_run, arm_phases, costs, compare, round_medians), so the
distributions are the blocks P1 publishes. E8 adds `e8`:
- `pairs`: for each comparison, B minus A per phase within each CPU part both arms ran on (cpuinfo part as read in
  the function; a row without one is never a stratum), with a bootstrap 95% CI over environments, and the same
  metric per (round, CPU part) cell with per-arm n, medians, delta, ratio and the delta sign set across rounds.
    effect    par - off    the one variable (same zip, one environment variable)
    dormant   off - base   the patched libraries with the switch unset, against P1's libraries
    total     par - base   the deployable change against P1's deployed code
- `mechanism`: per arm and class, the switch value, the loaded libggml-cpu variant and VmHWM, from e8_entry's
  first-invocation record, and whether the patched library logged its thread count.
- `host`: every CPU identity and environment field the wrapper read, with counts per arm and class. Anything the
  sandbox does not expose stays "unknown". No Graviton generation is inferred.
- `arm_agreement`: each arm's answers against base's for the same request and CPU part in this run.
- `offline_hash`: the hash_check.sh result (fixed host, not Lambda), when --hash-dir is given.
Status is judged on the `effect` pair's start-class headlines (cold and restore first request, warm inference):
`improved` only when a headline's CI lies below 0 on every CPU part both arms ran on, none lies above 0 anywhere, the
`dormant` pair shows no headline CI above or below 0, and answers and hashes are clean; `regressed` if any effect
headline CI lies above 0; `no-change` otherwise; `inconclusive` when the evidence is incomplete.
"""

from __future__ import annotations

import argparse
import glob
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

ARMS = ("base", "off", "par")
PRIMARY = "par"
PAIRS = {"effect": ("off", "par"), "dormant": ("base", "off"), "total": ("base", "par")}
UNKNOWN = "unknown"


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _mem(r: dict) -> float | None:
    return r["report"].get("Max Memory Used")


def _engine(r: dict) -> float | None:
    return (r.get("latency_ms") or 0) / 1000 or None


COLD_KEYS = {
    "model_load_mmap": lambda r: _init(r, "model_load_mmap"),
    "context_init": lambda r: _init(r, "context_init"),
    "weight_download": lambda r: _init(r, "weight_download"),
    "model_ready": lambda r: _init(r, "model_ready"),
    "warmup_inference": lambda r: _init(r, "warmup_inference"),
    "report_init_duration": lambda r: _rep(r, "Init Duration"),
    "first_request_inference": _engine,
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": _mem,
}
RESTORE_KEYS = {
    "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
    "first_request_inference": _engine,
    "first_request_total": lambda r: (
        (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": _mem,
}
WARM_KEYS = {"steady_state_inference": _engine, "report_max_memory_mb": _mem}
CLASS_KEYS = {"cold": COLD_KEYS, "restore": RESTORE_KEYS, "warm": WARM_KEYS}
HEADLINES = {
    "cold": "first_request_total",
    "restore": "first_request_total",
    "warm": "steady_state_inference",
}


def known_part(r: dict) -> str | None:
    p = r.get("cpu_part")
    return p if isinstance(p, str) and p and p not in ("?", UNKNOWN) else None


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 8) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling each arm independently."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def _vals(rs: list[dict], arm: str, f, **kv) -> list[float]:
    out = []
    for r in rs:
        if r["arm"] != arm or any(
            (known_part(r) if k == "cpu_part" else r.get(k)) != v for k, v in kv.items()
        ):
            continue
        v = f(r)
        if v is not None:
            out.append(v)
    return out


def delta(rows: list[dict], cls: str, keys: dict, a_arm: str, b_arm: str) -> dict:
    """b_arm minus a_arm per metric: within each known CPU part (bootstrap CI) and per (round, part) cell."""
    rs = ok_rows(rows, cls)
    parts = sorted({p for r in rs if (p := known_part(r))})
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in parts:
            a, b = _vals(rs, a_arm, f, cpu_part=cpu), _vals(rs, b_arm, f, cpu_part=cpu)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[cpu] = {
                "a_p50": round(ma, 6),
                "b_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        cells, missing = [], []
        for rnd in sorted({r["round"] for r in rs}):
            for cpu in parts:
                a = _vals(rs, a_arm, f, cpu_part=cpu, round=rnd)
                b = _vals(rs, b_arm, f, cpu_part=cpu, round=rnd)
                if not a or not b:
                    if a or b:
                        missing.append({"round": rnd, "cpu_part": cpu, "n": [len(a), len(b)]})
                    continue
                ma, mb = statistics.median(a), statistics.median(b)
                cells.append(
                    {
                        "round": rnd,
                        "cpu_part": cpu,
                        "n": [len(a), len(b)],
                        "a_p50": round(ma, 6),
                        "b_p50": round(mb, 6),
                        "delta_p50": round(mb - ma, 6),
                        "ratio": round(mb / ma, 4) if ma else None,
                    }
                )
        unassessed = Counter(
            r["arm"]
            for r in rs
            if r["arm"] in (a_arm, b_arm) and known_part(r) is None and f(r) is not None
        )
        if per or cells:
            ds = [c["delta_p50"] for c in cells]
            out[name] = {
                "by_cpu_part": per,
                "cells": cells,
                "cells_missing_an_arm": missing,
                "cell_delta_range": [min(ds), max(ds)] if ds else None,
                "cell_delta_signs": sorted({(d > 0) - (d < 0) for d in ds}),
                "rows_without_cpu_part": dict(unassessed),
            }
    return out


def pairs(rows: list[dict]) -> dict:
    return {
        pair: {"a": a, "b": b}
        | {cls: delta(rows, cls, keys, a, b) for cls, keys in CLASS_KEYS.items()}
        for pair, (a, b) in PAIRS.items()
    }


def e8_records(rows: list[dict]) -> list[tuple[dict, dict]]:
    return [(r, e) for r in rows if not r["error"] and (e := (r.get("lkai") or {}).get("e8"))]


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def thread_log_lines(run: str) -> dict:
    """Per function, how many 'kleidiai repack: threads=N' lines its saved logs hold (one per patched-path init)."""
    out: dict = {}
    for path in glob.glob(os.path.join(run, "logs", "*.jsonl.gz")):
        import gzip

        n: Counter = Counter()
        with gzip.open(path, "rt") as fh:
            for line in fh:
                msg = json.loads(line).get("message", "")
                if "kleidiai repack: threads=" in msg:
                    n[msg.strip().rsplit("=", 1)[-1]] += 1
        out[os.path.basename(path).removesuffix(".jsonl.gz")] = dict(n)
    return out


def mechanism(rows: list[dict], run: str) -> dict:
    """Per arm and class: the switch value, the loaded libggml-cpu variant and VmHWM, from e8_entry's record."""
    out: dict = defaultdict(dict)
    groups: dict = defaultdict(list)
    for r, e in e8_records(rows):
        groups[(r["arm"], r["expected_class"])].append(e)
    for (arm, cls), es in sorted(groups.items()):
        out[arm][cls] = {
            "environments": len(es),
            "repack_threads_env": sorted(
                Counter(str(x.get("repack_threads_env")) for x in es).items()
            ),
            "ggml_cpu_loaded": sorted(
                Counter(
                    ",".join(x["ggml_cpu_loaded"])
                    if isinstance(x.get("ggml_cpu_loaded"), list)
                    else str(x.get("ggml_cpu_loaded"))
                    for x in es
                ).items()
            ),
            "vm_hwm_kb_p50": _median(
                [
                    v
                    for x in es
                    if isinstance(v := (x.get("proc_status_kb") or {}).get("VmHWM"), int)
                ]
            ),
            "readout_s_max": max((x.get("readout_s") or 0) for x in es),
        }
    return {"by_arm": dict(out), "thread_log_lines": thread_log_lines(run)}


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm and class (counts), as e8_entry read it at the first invocation."""
    ids: dict = defaultdict(Counter)
    envs: dict = defaultdict(list)
    for r, e in e8_records(rows):
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
                "memory_mb": e.get("memory_mb"),
            },
            sort_keys=True,
        )
        ids[f"{r['arm']}:{r['expected_class']}"][key] += 1
        envs[r["arm"]].append(
            {
                "class": r["expected_class"],
                "round": r["round"],
                "cpu_part": c.get("part"),
                "init_cpu_part": e.get("init_cpu_part"),
                "init_type": e.get("init_type"),
                "log_stream": e.get("log_stream"),
                "ggml_cpu_loaded": e.get("ggml_cpu_loaded"),
            }
        )
    unknown = sorted(
        {
            f"cgroup {k}"
            for r, e in e8_records(rows)
            for k, v in (e.get("cgroup") or {}).items()
            if v == UNKNOWN
        }
    )
    return {
        "identities": {
            k: [{"count": n, **json.loads(s)} for s, n in cnt.most_common()]
            for k, cnt in sorted(ids.items())
        },
        "environments": dict(envs),
        "unknown_fields": [
            *unknown,
            "physical host",
            "AZ (not exposed to the function)",
            "CPU frequency",
            "co-tenancy",
        ],
        "note": "Read inside the function at each environment's first invocation, from /proc/cpuinfo, "
        "/proc/self/auxv, /proc/self/maps, os.sched_getaffinity, platform and /sys/fs/cgroup. init_cpu_part is the "
        "part of the host that ran the init (the snapshot init for a restore). 'unknown' = not exposed in the "
        "sandbox. No Graviton generation is inferred: parts and features are reported as read.",
    }


def arm_agreement(rows: list[dict]) -> dict:
    """Each arm's answers against base's in this run, for the same request id and known CPU part."""
    ref: dict = {}
    for r in rows:
        if r["arm"] == "base" and not r["error"] and known_part(r):
            ref.setdefault((r["request_id"], known_part(r)), r["answers"])
    base_self = sum(
        1
        for r in rows
        if r["arm"] == "base"
        and not r["error"]
        and known_part(r)
        and r["answers"] != ref[(r["request_id"], known_part(r))]
    )
    out: dict = {"base_internal_disagreements": base_self}
    for arm in ("off", "par"):
        n = same = missing = 0
        for r in rows:
            if r["arm"] != arm or r["error"]:
                continue
            want = ref.get((r["request_id"], known_part(r))) if known_part(r) else None
            if want is None:
                missing += 1
                continue
            n += 1
            same += r["answers"] == want
        out[arm] = {"compared": n, "bit_identical": same, "no_base_answer_for_cpu_part": missing}
    out["method"] = (
        "complete answer object of every successful off/par invocation vs the base answer for the same request id "
        "and CPU part in this run"
    )
    return out


def offline_hash(hash_dir: str) -> dict:
    """hash_check.sh's per-tensor hashes: every configuration and repeat against P1's libraries (repeat 0)."""
    runs: dict = {}
    for path in sorted(glob.glob(os.path.join(hash_dir, "*-*.txt"))):
        rep, cfg = os.path.basename(path)[:-4].split("-", 1)
        tensors, summary = {}, {}
        for line in open(path):
            parts = line.split()
            if parts and parts[0] == "T":
                tensors[parts[1]] = {
                    "type": parts[2],
                    "buft": parts[3],
                    "slots": int(parts[4]),
                    "bytes": int(parts[5]),
                    "hash": parts[6],
                }
            elif parts and parts[0] == "S":
                summary = dict(p.split("=", 1) for p in parts[1:])
        runs[(int(rep), cfg)] = {"tensors": tensors, "summary": summary}
    if not runs:
        return {"available": False}
    ref_key = min(k for k in runs if k[1] == "p1") if any(k[1] == "p1" for k in runs) else None
    ref = runs[ref_key]["tensors"] if ref_key else {}
    per_cfg: dict = defaultdict(
        lambda: {"runs": 0, "identical_runs": 0, "load_s": [], "all": set()}
    )
    mismatches = []
    for (rep, cfg), v in sorted(runs.items()):
        d = per_cfg[cfg]
        d["runs"] += 1
        d["load_s"].append(float(v["summary"].get("model_load_s", "nan")))
        d["all"].add(v["summary"].get("all"))
        same = v["tensors"].keys() == ref.keys() and all(
            v["tensors"][k]["hash"] == ref[k]["hash"]
            and v["tensors"][k]["bytes"] == ref[k]["bytes"]
            for k in ref
        )
        d["identical_runs"] += same
        if not same:
            bad = [k for k in ref if v["tensors"].get(k, {}).get("hash") != ref[k]["hash"]]
            mismatches.append({"repeat": rep, "config": cfg, "tensors": bad[:10], "n": len(bad)})
    packed = sum(1 for t in ref.values() if t["buft"] == "CPU_KLEIDIAI" and t["slots"] > 0)
    return {
        "available": True,
        "evidence": "OFFLINE, NOT LAMBDA: fixed host, P1's pinned Lambda image, P1's GGUF mmapped from local disk",
        "reference": "repeat 0 of p1 (P1's shipped libggml-cpu variants)",
        "tensors": len(ref),
        "kleidiai_packed_tensors": packed,
        "kleidiai_packed_bytes": sum(
            t["bytes"] for t in ref.values() if t["buft"] == "CPU_KLEIDIAI"
        ),
        "by_config": {
            c: {
                "runs": d["runs"],
                "identical_runs": d["identical_runs"],
                "distinct_model_hashes": len(d["all"]),
                "model_load_s": sorted(round(x, 4) for x in d["load_s"]),
                "model_load_s_p50": round(statistics.median(d["load_s"]), 4),
            }
            for c, d in sorted(per_cfg.items())
        },
        "mismatches": mismatches,
        "all_identical": not mismatches
        and all(d["identical_runs"] == d["runs"] for d in per_cfg.values()),
    }


def clean(res: dict) -> bool:
    c = res["correctness"] or {}
    g = res["e8"]["arm_agreement"]
    h = res["e8"].get("offline_hash") or {}
    return (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["base_internal_disagreements"] == 0
        and all(
            g[a]["compared"] > 0 and g[a]["bit_identical"] == g[a]["compared"]
            for a in ("off", "par")
        )
        and all(g[a]["no_base_answer_for_cpu_part"] == 0 for a in ("off", "par"))
        and h.get("all_identical") is True
    )


def _heads(res: dict, pair: str, min_n: int = 0) -> list[dict]:
    """The headline strata of a pair. With min_n, only strata where both arms have at least min_n samples."""
    p = res["e8"]["pairs"][pair]
    return [
        {
            cpu: v
            for cpu, v in p[cls].get(k, {}).get("by_cpu_part", {}).items()
            if min(v["n"]) >= min_n
        }
        for cls, k in HEADLINES.items()
    ]


# Added AFTER run 20261021T110412Z was seen (disclosed in results.json -> e8.status_rule): the dormant check counts
# only CPU-part strata with at least this many samples per arm. The pre-registered rule (min_n 0) is still computed
# and published as e8.status_rule.preregistered.
DORMANT_MIN_N = 5


def status_of(res: dict, dormant_min_n: int = DORMANT_MIN_N) -> str:
    heads = _heads(res, "effect")
    if not any(heads) or not clean(res):
        return "inconclusive"
    if any(v["ci95"][0] > 0 for h in heads for v in h.values()):
        return "regressed"
    dormant_moved = any(
        v["ci95"][0] > 0 or v["ci95"][1] < 0
        for h in _heads(res, "dormant", dormant_min_n)
        for v in h.values()
    )
    if dormant_moved:
        return "inconclusive"
    if any(h and all(v["ci95"][1] < 0 for v in h.values()) for h in heads):
        return "improved"
    return "no-change"


def fmt_values(res: dict) -> dict:
    """Numbers a conclusion template may cite: <pair>_<class>_<metric>_<part>_{a,b,d,lo,hi,n}."""
    out = {}
    for pair, p in res["e8"]["pairs"].items():
        for cls in ("cold", "restore", "warm"):
            for name, d in p[cls].items():
                for cpu, v in d["by_cpu_part"].items():
                    key = f"{pair}_{cls}_{name}_{cpu}"
                    out[key + "_a"], out[key + "_b"] = v["a_p50"], v["b_p50"]
                    out[key + "_d"] = v["delta_p50"]
                    out[key + "_lo"], out[key + "_hi"] = v["ci95"]
                    out[key + "_n"] = v["n"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--hash-dir")
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = dict.fromkeys(
        ARMS, "1"
    )  # every E8 arm runs P1's timers; E8's control is base/off, not rep
    res = p1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = p1.load_run(a.run)
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = PRIMARY
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["e8_arms"] = cond.get("e8_arms")
    res["deployment"]["code_sha256_by_arm"] = {
        arm: f.get("CodeSha256") for arm, f in cond["functions"].items()
    }
    res["deployment"]["zips"] = cond["build_manifest"].get("zips")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "every E8 arm runs P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["e8"] = {
        "pairs": pairs(rows),
        "pairs_method": "B minus A, medians within each CPU part (cpuinfo, read in the function) both arms ran on; "
        "ci95 = bootstrap over environments (4,000 resamples, seed 8); cells = per (round, CPU part) medians. "
        "Rows without a CPU part are counted, never a stratum. effect = par - off (the one variable), dormant = "
        "off - base (switch unset), total = par - base.",
        "mechanism": mechanism(rows, a.run),
        "host": host(rows),
        "arm_agreement": arm_agreement(rows),
        "offline_hash": offline_hash(a.hash_dir) if a.hash_dir else {"available": False},
    }
    res["status"] = status_of(res)
    pre = status_of(res, 0)
    res["e8"]["status_rule"] = {
        "applied": f"effect headlines (par - off) decide; the dormant pair (off - base) must not exclude 0 on any "
        f"headline stratum with n >= {DORMANT_MIN_N} per arm",
        "dormant_min_n": DORMANT_MIN_N,
        "status_applied": res["status"],
        "preregistered": {"dormant_min_n": 0, "status": pre},
        "disclosure": "The min-n condition was added after this run's results were seen, at the operator's "
        "direction. The pre-registered rule (any stratum) gives the status under preregistered.status.",
        "dormant_strata_excluded": [
            {"class": cls, "cpu_part": cpu, "n": v["n"], "ci95": v["ci95"]}
            for (cls, k) in HEADLINES.items()
            for cpu, v in res["e8"]["pairs"]["dormant"][cls]
            .get(k, {})
            .get("by_cpu_part", {})
            .items()
            if min(v["n"]) < DORMANT_MIN_N
        ],
    }
    tmpl = meta.get("conclusion_template")
    res["conclusion"] = (
        tmpl.format(**fmt_values(res), _pre=res["e8"]["status_rule"]["preregistered"]["status"])
        if tmpl
        else meta.get("conclusion")
    )
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
        "offline_hash": "offline-hash/",
        "offline_probe": "offline-probe/",
        "zips": "zips.json",
    }
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
