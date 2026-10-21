"""E10 summary: one lkai-result/1 file for the n_ctx sweep, every arm against the c16384 control (P1's n_ctx).

  python summarize_e10.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <P1 sha> --reference ../baseline/reference/outputs.json --meta meta.json \
      --offline offline/sizes.json --out <dir>

P1's summarize.py does every row-level computation (load_run, arm_phases, compare, costs, round_medians), so the
distributions are the blocks P1 publishes. E10 adds, under `e10`:
- `delta`: arm minus control per metric, within each CPU part both arms ran on, with a bootstrap 95% CI over
  environments (4,000 resamples, seed 10).
- `host_variability` (top level): every primary metric per shared (round, CPU part) cell, each arm against the
  control, with the control's within-cell range. Rows without a CPU part are counted, never a stratum.
- `memory`: per arm, REPORT Max Memory Used and the process VmHWM e10_entry read after the first response.
- `n_ctx_applied`: what the wrapper passed to Shim, per arm (mechanism evidence).
- `cache_bytes`: llama.cpp's own KV / recurrent / compute buffer sizes per n_ctx, from the offline probe (they
  depend only on n_ctx, the model and the context params, not on the host).
- `arm_agreement`: every arm's answers against the control's for the same request and CPU part.
Status reports `primary_arm` (meta): improved only if a headline's CI lies below 0 on every CPU part both ran on,
regressed if any headline CI lies above 0, no-change otherwise, inconclusive without clean answers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import summarize as p1  # noqa: E402

SIZES = (512, 1024, 2048, 4096, 8192, 16384)
ARMS = tuple(f"c{n}" for n in SIZES)
CONTROL = "c16384"
UNKNOWN_CPU = (None, "", "unknown", "?")


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def _lat(r: dict) -> float | None:
    v = r.get("latency_ms")
    return v / 1000 if v is not None else None


COLD_KEYS = {
    "context_init": lambda r: _init(r, "context_init"),
    "shim_init": lambda r: _init(r, "shim_init"),
    "warmup_inference": lambda r: _init(r, "warmup_inference"),
    "model_ready": lambda r: _init(r, "model_ready"),
    "first_request_inference": _lat,
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
RESTORE_KEYS = {
    "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
    "first_request_inference": _lat,
    "first_request_total": lambda r: (
        (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
WARM_KEYS = {
    "steady_state_inference": _lat,
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
HEADLINES = {
    "cold": "first_request_total",
    "restore": "first_request_total",
    "warm": "steady_state_inference",
}
PRIMARY = {  # the host-variability cells: the phases the memset hot spot sits in, plus the deployment headlines
    "cold_context_init": ("cold", COLD_KEYS["context_init"]),
    "cold_max_memory_mb": ("cold", COLD_KEYS["report_max_memory_mb"]),
    "cold_first_request_total": ("cold", COLD_KEYS["first_request_total"]),
    "restore_first_request_inference": ("restore", RESTORE_KEYS["first_request_inference"]),
    "restore_restore_duration": ("restore", RESTORE_KEYS["report_restore_duration"]),
    "restore_first_request_total": ("restore", RESTORE_KEYS["first_request_total"]),
    "warm_steady_state_inference": ("warm", WARM_KEYS["steady_state_inference"]),
}


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 10) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling each arm's environments independently."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def _vals(rs: list[dict], f, **kv) -> list[float]:
    return [v for r in rs if all(r.get(k) == x for k, x in kv.items()) if (v := f(r)) is not None]


def delta(rows: list[dict], arm: str, cls: str, keys: dict) -> dict:
    """arm minus control per metric, within each CPU part both ran on."""
    rs = ok_rows(rows, cls)
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted(
            {r.get("cpu_part") for r in rs if r.get("cpu_part") not in UNKNOWN_CPU}, key=str
        ):
            a = _vals(rs, f, arm=CONTROL, cpu_part=cpu)
            b = _vals(rs, f, arm=arm, cpu_part=cpu)
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[str(cpu)] = {
                "control_p50": round(ma, 6),
                "arm_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        if per:
            out[name] = {"by_cpu_part": per}
    return out


def cell_outcome(change: float, control_spread: float | None) -> str:
    if control_spread is None:
        return "unassessed"
    if abs(change) <= control_spread:
        return "within-control-variation"
    return (
        "negative-beyond-control-variation" if change < 0 else "positive-beyond-control-variation"
    )


def host_variability_analysis(rows: list[dict], arms: list[str]) -> dict:
    """Per primary metric and arm: arm vs control in every shared (round, CPU part) cell.

    Arms run as sequential waves; round r runs them rotated left by r (a cyclic Latin square), so over the rounds
    every arm takes every position once. That balances position-within-round drift across arms; it is still not
    simultaneous interleaving. A cell's outcome compares its delta with the control's within-cell range."""
    metrics: dict = {}
    unknown_rows: dict = {}
    for name, (cls, f) in PRIMARY.items():
        rs = ok_rows(rows, cls)
        unknown_rows[name] = {
            a: sum(r.get("cpu_part") in UNKNOWN_CPU for r in rs if r["arm"] == a) for a in arms
        }
        known = [r for r in rs if r.get("cpu_part") not in UNKNOWN_CPU]
        per_arm: dict = {}
        for arm in arms:
            if arm == CONTROL:
                continue
            cells, missing = [], []
            for rnd, cpu in sorted({(r["round"], str(r["cpu_part"])) for r in known}):
                c = [
                    v
                    for r in known
                    if r["arm"] == CONTROL and r["round"] == rnd and str(r["cpu_part"]) == cpu
                    if (v := f(r)) is not None
                ]
                x = [
                    v
                    for r in known
                    if r["arm"] == arm and r["round"] == rnd and str(r["cpu_part"]) == cpu
                    if (v := f(r)) is not None
                ]
                if not c or not x:
                    missing.append(
                        {"round": rnd, "cpu_part": cpu, "n": {CONTROL: len(c), arm: len(x)}}
                    )
                    continue
                mc, mx = statistics.median(c), statistics.median(x)
                spread = round(max(c) - min(c), 6) if len(c) >= 2 else None
                ch = round(mx - mc, 6)
                cells.append(
                    {
                        "round": rnd,
                        "cpu_part": cpu,
                        "n": {CONTROL: len(c), arm: len(x)},
                        "median": {CONTROL: round(mc, 6), arm: round(mx, 6)},
                        "control_spread": spread,
                        "delta": ch,
                        "ratio": round(mx / mc, 6) if mc else None,
                        "sign": "negative" if ch < 0 else "positive" if ch > 0 else "zero",
                        "outcome": cell_outcome(ch, spread),
                    }
                )
            by_part: dict = {}
            for cpu in sorted({c["cpu_part"] for c in cells}):
                pc = [c for c in cells if c["cpu_part"] == cpu]
                assessed = [c["outcome"] for c in pc if c["outcome"] != "unassessed"]
                by_part[cpu] = {
                    "shared_cells": len(pc),
                    "assessed_cells": len(assessed),
                    "delta_range": [min(c["delta"] for c in pc), max(c["delta"] for c in pc)],
                    "delta_signs": sorted({c["sign"] for c in pc}),
                    "cell_outcomes": dict(sorted(Counter(c["outcome"] for c in pc).items())),
                    "outcome": (
                        "insufficient-control-cell-spread"
                        if not assessed
                        else assessed[0]
                        if len(set(assessed)) == 1
                        else "mixed-or-overlapping-control-variation"
                    ),
                }
            per_arm[arm] = {"shared_cells": cells, "missing_cells": missing, "by_cpu_part": by_part}
        metrics[name] = {"class": cls, "arms": per_arm}
    return {
        "schema": "lkai-host-variability/1",
        "observed": {"primary_metrics": metrics},
        "unknown": {
            "cpu_identity_rows": unknown_rows,
            "environment_fields": ["availability zone", "physical host"],
        },
        "method": "Within each observed CPU part, compare each arm's median with the control's (c16384, P1's n_ctx) "
        "in each round. Arms are sequential waves; round r runs them rotated left by r (cyclic Latin square), "
        "which balances position-within-round across arms but is not simultaneous interleaving. A directional "
        "outcome needs the delta to exceed the control's within-cell range; a cell with fewer than two control "
        "observations is unassessed. Rows without a CPU part are counted, never a stratum. No Graviton "
        "generation is inferred from a CPU part.",
    }


def memory(rows: list[dict], arms: list[str]) -> dict:
    out: dict = {}
    for arm in arms:
        d: dict = {}
        for cls in ("cold", "restore", "warm"):
            rs = ok_rows(rows, cls)
            per = defaultdict(list)
            for r in rs:
                if r["arm"] == arm and r["report"].get("Max Memory Used") is not None:
                    per[str(r.get("cpu_part"))].append(r["report"]["Max Memory Used"])
            if per:
                d[f"{cls}_report_max_memory_mb"] = {
                    k: p1.dist(v, unit="MB") for k, v in sorted(per.items())
                }
        hwm = [
            e["vm_hwm_kib"] / 1024
            for r in rows
            if r["arm"] == arm and not r["error"]
            if isinstance((e := (r.get("lkai") or {}).get("e10") or {}).get("vm_hwm_kib"), int)
        ]
        if hwm:
            d["first_invoke_vm_hwm_mib"] = p1.dist(hwm, unit="MiB")
        out[arm] = d
    return out


def n_ctx_applied(rows: list[dict], arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        recs = [
            json.dumps(e["n_ctx"], sort_keys=True)
            for r in rows
            if r["arm"] == arm and not r["error"]
            if (e := (r.get("lkai") or {}).get("e10"))
        ]
        out[arm] = [{"count": n, **json.loads(k)} for k, n in Counter(recs).most_common()]
    return out


def host(rows: list[dict]) -> dict:
    ids: dict = defaultdict(Counter)
    for r in rows:
        e = (r.get("lkai") or {}).get("e10")
        if not e or r["error"]:
            continue
        c = e.get("cpu", {})
        key = {
            k: c.get(k)
            for k in ("implementer", "part", "variant", "revision", "processors", "features")
        } | {
            k: e.get(k)
            for k in (
                "hwcap",
                "hwcap2",
                "os_cpu_count",
                "sched_affinity",
                "cgroup",
                "kernel",
                "python",
                "execution_env",
            )
        }
        ids[r["arm"]][json.dumps(key, sort_keys=True)] += 1
    return {
        "identities": {
            a: [{"count": n, **json.loads(k)} for k, n in cnt.most_common()]
            for a, cnt in ids.items()
        },
        "note": "Read inside the function from /proc/cpuinfo, /proc/self/auxv, os.sched_getaffinity and "
        "/sys/fs/cgroup on each environment's first invocation. 'unknown' = not exposed. No Graviton generation is "
        "inferred: CPU part and features are reported as read.",
    }


def arm_agreement(rows: list[dict], arms: list[str]) -> dict:
    """Each arm's answers against the control's for the same request and CPU part, from this run."""
    ref: dict = {}
    for r in rows:
        if r["arm"] == CONTROL and not r["error"]:
            ref.setdefault((r["request_id"], r.get("cpu_part")), r["answers"])
    out = {}
    for arm in arms:
        n = same = missing = 0
        for r in rows:
            if r["arm"] != arm or r["error"]:
                continue
            base = ref.get((r["request_id"], r.get("cpu_part")))
            if base is None:
                missing += 1
                continue
            n += 1
            same += r["answers"] == base
        out[arm] = {"compared": n, "bit_identical": same, "no_control_answer_for_cpu_part": missing}
    return out | {
        "method": "complete answer object of every successful invocation vs the control arm's answer for the same "
        "request id and CPU part in this run"
    }


def merge_runs(cold: str, restore: str | None, out_dir: str) -> str:
    """One run directory holding both legs (cold first); restore rounds are renumbered 100+."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    legs = [cold, restore] if restore else [cold]
    with open(os.path.join(out_dir, "invocations.jsonl"), "w") as w:
        for leg in legs:
            for line in open(os.path.join(leg, "invocations.jsonl")):
                r = json.loads(line)
                r["leg"] = os.path.basename(os.path.normpath(leg)).split("-", 1)[0]
                if r["leg"] == "restore":
                    r["round"] = 100 + r["round"]
                w.write(json.dumps(r, separators=(",", ":")) + "\n")
    c = json.load(open(os.path.join(cold, "conditions.json")))
    if restore:
        r = json.load(open(os.path.join(restore, "conditions.json")))
        c["legs"] = {
            "cold": {k: c[k] for k in ("rounds", "started", "finished")},
            "restore": {k: r[k] for k in ("rounds", "started", "finished")},
        }
        c["rounds_meta"] = c["rounds_meta"] + [
            m | {"round": 100 + m["round"]} for m in r["rounds_meta"]
        ]
        c["finished"] = r.get("finished")
        c["read_timeout_s"] = {"cold": c.get("read_timeout_s"), "restore": r.get("read_timeout_s")}
    json.dump(c, open(os.path.join(out_dir, "conditions.json"), "w"), indent=1)
    for leg in legs:
        for f in sorted(os.listdir(os.path.join(leg, "logs"))):
            with (
                open(os.path.join(leg, "logs", f), "rb") as src,
                open(os.path.join(out_dir, "logs", f), "ab") as dst,
            ):
                shutil.copyfileobj(src, dst)
    return out_dir


def status_of(res: dict, arm: str) -> str:
    d = res["e10"]["delta"][arm]
    heads = [d.get(cls, {}).get(m, {}).get("by_cpu_part", {}) for cls, m in HEADLINES.items()]
    c = res["correctness"] or {}
    g = res["e10"]["arm_agreement"][arm]
    clean = (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["no_control_answer_for_cpu_part"] == 0
    )
    if not any(heads) or not clean:
        return "inconclusive"
    if any(v["ci95"][0] > 0 for h in heads for v in h.values()):
        return "regressed"
    if any(h and all(v["ci95"][1] < 0 for v in h.values()) for h in heads):
        return "improved"
    return "no-change"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", required=True)
    ap.add_argument("--restore")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--offline", required=True, help="offline/sizes.json from offline_sizes.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = merge_runs(a.cold, a.restore, a.out)
    meta = json.load(open(a.meta))
    cond, rows = p1.load_run(run)
    arms = list(cond["arms"])
    p1.ARM_PHASES = dict.fromkeys(
        arms, "1"
    )  # every E10 arm runs P1's timers; the control is c16384, not rep
    res = p1.summarize(run, a.reference, a.baseline_commit, meta)
    m = cond["build_manifest"]
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = meta["primary_arm"]
    res["commits"]["experiment"] = m["repo_commit"]
    res["build"] = {
        k: m.get(k)
        for k in ("images", "toolchain", "assets_sha256", "zips", "base", "wrapper_sha256")
    }
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["legs"] = cond.get("legs")
    res["conditions"]["n_ctx"] = cond.get("n_ctx")
    res["conditions"]["order"] = cond.get("order")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "every E10 arm runs P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["host_variability"] = host_variability_analysis(rows, arms)
    offline = json.load(open(a.offline))
    res["e10"] = {
        "control": CONTROL,
        "delta": {
            arm: {
                "cold": delta(rows, arm, "cold", COLD_KEYS),
                "restore": delta(rows, arm, "restore", RESTORE_KEYS),
                "warm": delta(rows, arm, "warm", WARM_KEYS),
            }
            for arm in arms
            if arm != CONTROL
        },
        "delta_method": "arm minus c16384, medians within each CPU part both ran on; ci95 = bootstrap over "
        "environments (4,000 resamples, seed 10)",
        "memory": memory(rows, arms),
        "n_ctx_applied": n_ctx_applied(rows, arms),
        "cache_bytes": offline.get("sizes"),
        "capacity": offline.get("capacity"),
        "host": host(rows),
        "arm_agreement": arm_agreement(rows, arms),
    }
    res["e10"]["status_by_arm"] = {arm: status_of(res, arm) for arm in arms if arm != CONTROL}
    res["status"] = res["e10"]["status_by_arm"][res["primary_arm"]]
    res["conclusion"] = meta.get("conclusion")
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
        "offline_sizes": "offline-sizes.json",
    }
    path = os.path.join(a.out, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"], res["e10"]["status_by_arm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
