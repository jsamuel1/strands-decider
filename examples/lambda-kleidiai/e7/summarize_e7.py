"""E7 summary: one lkai-result/1 file for the skip arm, against the ctl arm from the same interleaved run.

  python summarize_e7.py <run-dir> --run-id <id> --baseline-commit <P1 sha> \
      --reference ../baseline/reference/outputs.json --meta meta.json

Uses P1's summarize.py for every row-level computation (load_run, arm_phases, costs, compare, round_medians), so the
distributions are the blocks P1 publishes. E7 adds `e7`:
- `delta`: skip minus ctl per phase, within each CPU identity both arms ran on (cpuinfo part as read in the
  function; a row without one is never a stratum), with a bootstrap 95% CI over environments.
- `cells`: the same metric per (round, CPU part) cell both arms have: per-arm n, medians, delta, ratio, and the set
  of delta signs across rounds, so the effect can be judged within host and round, not only pooled.
- `mechanism`: per arm, what the loaded model held (head skipped, CPU_KLEIDIAI / CPU buffer bytes) and VmHWM, from
  e7_entry's first-invocation record.
- `host`: every CPU identity and environment field the wrapper read, with counts per arm and class. Anything the
  sandbox does not expose stays "unknown". No Graviton generation is inferred.
- `arm_agreement`: skip's answers against ctl's for the same request and CPU part in this run.
Status follows E6's rule: judged on the start-class headlines (cold and restore first request, warm inference),
`improved` only when one headline's CI lies below 0 on every CPU part both arms ran on, none lies above 0 anywhere,
and the answers are clean.
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

CONTROL, EXPERIMENT = "ctl", "skip"
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


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 7) -> list[float]:
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


def delta(rows: list[dict], cls: str, keys: dict) -> dict:
    """skip minus ctl per metric: within each known CPU part (bootstrap CI) and per (round, part) cell."""
    rs = ok_rows(rows, cls)
    parts = sorted({p for r in rs if (p := known_part(r))})
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in parts:
            a, b = _vals(rs, CONTROL, f, cpu_part=cpu), _vals(rs, EXPERIMENT, f, cpu_part=cpu)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[cpu] = {
                "ctl_p50": round(ma, 6),
                "skip_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        cells, missing = [], []
        for rnd in sorted({r["round"] for r in rs}):
            for cpu in parts:
                a = _vals(rs, CONTROL, f, cpu_part=cpu, round=rnd)
                b = _vals(rs, EXPERIMENT, f, cpu_part=cpu, round=rnd)
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
                        "ctl_p50": round(ma, 6),
                        "skip_p50": round(mb, 6),
                        "delta_p50": round(mb - ma, 6),
                        "ratio": round(mb / ma, 4) if ma else None,
                    }
                )
        unassessed = Counter(r["arm"] for r in rs if known_part(r) is None and f(r) is not None)
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


def e7_records(rows: list[dict]) -> list[tuple[dict, dict]]:
    return [(r, e) for r in rows if not r["error"] and (e := (r.get("lkai") or {}).get("e7"))]


def mechanism(rows: list[dict]) -> dict:
    """Per arm and class: the loaded model's head flag and buffer bytes, and VmHWM after the first answer."""
    out: dict = defaultdict(dict)
    groups: dict = defaultdict(list)
    for r, e in e7_records(rows):
        groups[(r["arm"], r["expected_class"])].append(e)
    for (arm, cls), es in sorted(groups.items()):
        m = [x.get("mechanism") or {} for x in es]
        out[arm][cls] = {
            "environments": len(es),
            "tied_output_skipped": sorted(
                Counter(str(x.get("tied_output_skipped")) for x in m).items()
            ),
            "buft_bytes": {
                k: sorted(Counter((x.get("buft_bytes") or {}).get(k, UNKNOWN) for x in m).items())
                for k in ("CPU_KLEIDIAI", "CPU", "CPU_REPACK")
            },
            "vm_hwm_kb_p50": _median(
                [
                    v
                    for x in es
                    if isinstance(v := (x.get("proc_status_kb") or {}).get("VmHWM"), int)
                ]
            ),
            "readout_s_max": max((x.get("readout_s") or 0) for x in es),
            "errors": sorted({x["error"] for x in m if x.get("error")}),
        }
    return dict(out)


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm and class (counts), as e7_entry read it at the first invocation."""
    ids: dict = defaultdict(Counter)
    envs: dict = defaultdict(list)
    for r, e in e7_records(rows):
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
            }
        )
    unknown = sorted(
        {
            f"cgroup {k}"
            for r, e in e7_records(rows)
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
        "/proc/self/auxv, os.sched_getaffinity, platform and /sys/fs/cgroup. init_cpu_part is the part of the host "
        "that ran the init (the snapshot init for a restore). 'unknown' = not exposed in the sandbox. No Graviton "
        "generation is inferred: parts and features are reported as read.",
    }


def arm_agreement(rows: list[dict]) -> dict:
    """skip's answers against ctl's from the same run, for the same request and known CPU part."""
    ref: dict = {}
    for r in rows:
        if r["arm"] == CONTROL and not r["error"] and known_part(r):
            ref.setdefault((r["request_id"], known_part(r)), r["answers"])
    ctl_self = sum(
        1
        for r in rows
        if r["arm"] == CONTROL
        and not r["error"]
        and known_part(r)
        and r["answers"] != ref[(r["request_id"], known_part(r))]
    )
    n = same = missing = 0
    for r in rows:
        if r["arm"] != EXPERIMENT or r["error"]:
            continue
        base = ref.get((r["request_id"], known_part(r))) if known_part(r) else None
        if base is None:
            missing += 1
            continue
        n += 1
        same += r["answers"] == base
    return {
        "compared": n,
        "bit_identical": same,
        "no_ctl_answer_for_cpu_part": missing,
        "ctl_internal_disagreements": ctl_self,
        "method": "complete answer object of every successful skip invocation vs the ctl answer for the same "
        "request id and CPU part in this run",
    }


def clean(res: dict) -> bool:
    c = res["correctness"] or {}
    g = res["e7"]["arm_agreement"]
    return (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["no_ctl_answer_for_cpu_part"] == 0
        and g["ctl_internal_disagreements"] == 0
    )


def status_of(res: dict) -> str:
    d = res["e7"]["delta"]
    heads = [d[cls].get(k, {}).get("by_cpu_part", {}) for cls, k in HEADLINES.items()]
    if not any(heads) or not clean(res):
        return "inconclusive"
    if any(v["ci95"][0] > 0 for h in heads for v in h.values()):
        return "regressed"
    if any(h and all(v["ci95"][1] < 0 for v in h.values()) for h in heads):
        return "improved"
    return "no-change"


def fmt_values(res: dict) -> dict:
    """Numbers a conclusion template may cite: <class>_<metric>_<part>_{ctl,skip,d,lo,hi,n}."""
    out = {}
    for cls in ("cold", "restore", "warm"):
        for name, d in res["e7"]["delta"][cls].items():
            for cpu, v in d["by_cpu_part"].items():
                key = f"{cls}_{name}_{cpu}"
                out[key + "_ctl"], out[key + "_skip"] = v["ctl_p50"], v["skip_p50"]
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
    a = ap.parse_args(argv)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = {
        CONTROL: "1",
        EXPERIMENT: "1",
    }  # both arms instrumented: E7's control is ctl, not rep
    res = p1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = p1.load_run(a.run)
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = EXPERIMENT
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "both E7 arms run P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["e7"] = {
        "delta": {
            "cold": delta(rows, "cold", COLD_KEYS),
            "restore": delta(rows, "restore", RESTORE_KEYS),
            "warm": delta(rows, "warm", WARM_KEYS),
            "method": "skip minus ctl, medians within each CPU part (cpuinfo, read in the function) both arms "
            "ran on; ci95 = bootstrap over environments (4,000 resamples, seed 7); cells = per (round, CPU "
            "part) medians from the interleaved rounds. Rows without a CPU part are counted, never a stratum.",
        },
        "mechanism": mechanism(rows),
        "host": host(rows),
        "arm_agreement": arm_agreement(rows),
    }
    res["status"] = status_of(res)
    tmpl = meta.get("conclusion_template")
    res["conclusion"] = tmpl.format(**fmt_values(res)) if tmpl else meta.get("conclusion")
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
    }
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
