"""E6 summary: one lkai-result/1 file for the pyc arm, against the ctl arm (P1's bytecode) from the same run.

  python summarize_e6.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <P1 sha> --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

Uses P1's summarize.py for every row-level computation (load_run, arm_phases, compare, costs, round_medians), so
the distributions are the same blocks P1 publishes. E6 adds:
- `e6.delta`: pyc minus ctl per phase. Each delta is matched by CPU part (P1's `overhead` method) and gets a
  bootstrap 95% CI over environments. Per-round paired medians come from the interleaved rounds.
- `e6.bytecode`: the mechanism, per arm. It records how many task-root modules each environment loaded from usable
  bytecode, from e6_entry's first-invocation record.
- `e6.host`: every CPU identity and environment field the wrapper read, with counts per arm. Anything the sandbox
  does not expose stays "unknown".
Status is `improved` only if the cold `imports` CI lies entirely below 0 within every CPU part both arms ran on,
and errors, class mismatches and correctness are clean. It is `regressed` if the CI lies entirely above 0, and
`no-change` otherwise.
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
import summarize as p1  # noqa: E402

CONTROL, EXPERIMENT = "ctl", "pyc"
COLD_KEYS = {
    "imports": lambda r: _init(r, "imports"),
    "import_total": lambda r: _init(r, "import_total"),
    "runtime_to_import": lambda r: _init(r, "runtime_to_import"),
    "report_init_duration": lambda r: _rep(r, "Init Duration"),
    "model_ready": lambda r: _init(r, "model_ready"),
    "weight_download": lambda r: _init(r, "weight_download"),
    "model_load_mmap": lambda r: _init(r, "model_load_mmap"),
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
WARM_KEYS = {"steady_state_inference": lambda r: (r.get("latency_ms") or 0) / 1000 or None}
RESTORE_KEYS = {
    "report_restore_duration": lambda r: _rep(r, "Restore Duration"),
    "first_request_total": lambda r: (
        (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "first_request_inference": lambda r: (r.get("latency_ms") or 0) / 1000 or None,
}


def _init(r: dict, k: str) -> float | None:
    return ((r.get("lkai") or {}).get("init") or {}).get(k)


def _rep(r: dict, k: str) -> float | None:
    return r["report"][k] / 1000 if k in r["report"] else None


def ok_rows(rows: list[dict], cls: str) -> list[dict]:
    return [
        r for r in rows if not r["error"] and not r["class_mismatch"] and r["expected_class"] == cls
    ]


def boot_ci(a: list[float], b: list[float], n: int = 4000, seed: int = 6) -> list[float]:
    """95% bootstrap CI of median(b) - median(a), resampling environments independently per arm."""
    rnd = random.Random(seed)
    ds = sorted(
        statistics.median(rnd.choices(b, k=len(b))) - statistics.median(rnd.choices(a, k=len(a)))
        for _ in range(n)
    )
    return [round(ds[int(0.025 * n)], 6), round(ds[int(0.975 * n) - 1], 6)]


def delta(rows: list[dict], cls: str, keys: dict) -> dict:
    """pyc minus ctl per metric, within each CPU part both arms ran on, plus the interleaved per-round deltas."""
    rs = ok_rows(rows, cls)
    out: dict = {}
    for name, f in keys.items():
        per: dict = {}
        for cpu in sorted({r.get("cpu_part") for r in rs}, key=str):
            a = [
                v
                for r in rs
                if r["arm"] == CONTROL and r.get("cpu_part") == cpu
                if (v := f(r)) is not None
            ]
            b = [
                v
                for r in rs
                if r["arm"] == EXPERIMENT and r.get("cpu_part") == cpu
                if (v := f(r)) is not None
            ]
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            per[str(cpu)] = {
                "ctl_p50": round(ma, 6),
                "pyc_p50": round(mb, 6),
                "delta_p50": round(mb - ma, 6),
                "delta_pct": round(100 * (mb - ma) / ma, 2) if ma else None,
                "ci95": boot_ci(a, b),
                "n": [len(a), len(b)],
            }
        rounds = []
        for rnd in sorted({r["round"] for r in rs}):
            a = [
                v
                for r in rs
                if r["arm"] == CONTROL and r["round"] == rnd
                if (v := f(r)) is not None
            ]
            b = [
                v
                for r in rs
                if r["arm"] == EXPERIMENT and r["round"] == rnd
                if (v := f(r)) is not None
            ]
            if a and b:
                rounds.append(round(statistics.median(b) - statistics.median(a), 6))
        if per:
            out[name] = {
                "by_cpu_part": per,
                "per_round_delta_p50": rounds,
                "rounds_below_zero": sum(d < 0 for d in rounds),
            }
    return out


def bytecode(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in (CONTROL, EXPERIMENT):
        recs = [
            r["lkai"]["e6"]["bytecode"]
            for r in rows
            if r["arm"] == arm
            and not r["error"]
            and ((r.get("lkai") or {}).get("e6") or {}).get("bytecode")
        ]
        if recs:
            keys = recs[0].keys()
            out[arm] = {
                "environments": len(recs),
                **{k: sorted(Counter(x[k] for x in recs).items()) for k in keys},
            }
    return out


def host(rows: list[dict]) -> dict:
    """Every distinct host identity per arm (counts), as e6_entry read it. Log streams are kept per environment."""
    ids: dict = defaultdict(Counter)
    streams: dict = defaultdict(list)
    for r in rows:
        e = (r.get("lkai") or {}).get("e6")
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
                "hwcap": e.get("hwcap"),
                "hwcap2": e.get("hwcap2"),
                "features": c.get("features"),
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
        streams[r["arm"]].append(
            {
                "class": r["expected_class"],
                "round": r["round"],
                "cpu_part": c.get("part"),
                "log_stream": e.get("log_stream"),
            }
        )
    return {
        "identities": {
            arm: [{"count": n, **json.loads(k)} for k, n in cnt.most_common()]
            for arm, cnt in ids.items()
        },
        "environments": dict(streams),
        "note": "Read inside the function from /proc/cpuinfo, /proc/self/auxv, os.sched_getaffinity and "
        "/sys/fs/cgroup. 'unknown' = not exposed in the sandbox. No Graviton generation is inferred: cpu part "
        "and features are reported as read.",
    }


def merge_runs(cold: str, restore: str | None, out_dir: str) -> str:
    """One run directory holding both legs' invocations, logs and conditions (cold first)."""
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    with open(os.path.join(out_dir, "invocations.jsonl"), "w") as w:
        for leg in [cold, restore] if restore else [cold]:
            for line in open(os.path.join(leg, "invocations.jsonl")):
                r = json.loads(line)
                r["leg"] = os.path.basename(os.path.normpath(leg)).split("-", 1)[0]
                if r["leg"] == "restore":
                    r["round"] = 100 + r["round"]  # keep per-round variance per leg
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
    # P1's snapshot_inits() maps logs/<function>.jsonl.gz to its function, so both legs go into one file each.
    for leg in [cold, restore] if restore else [cold]:
        for f in sorted(os.listdir(os.path.join(leg, "logs"))):
            with (
                open(os.path.join(leg, "logs", f), "rb") as src,
                open(os.path.join(out_dir, "logs", f), "ab") as dst,
            ):
                shutil.copyfileobj(src, dst)  # concatenated gzip members read back as one stream
    return out_dir


def arm_agreement(rows: list[dict]) -> dict:
    """pyc's answers against ctl's from the same run, for the same request and CPU part.

    Within a CPU part, outputs repeat exactly (P1's reference relies on this), so one ctl answer set per (request,
    cpu part) is the reference for every pyc answer there. This covers CPU parts P1's reference lacks."""
    ref: dict = {}
    for r in rows:
        if r["arm"] == CONTROL and not r["error"]:
            ref.setdefault((r["request_id"], r.get("cpu_part")), r["answers"])
    ctl_self = sum(
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
        "no_ctl_answer_for_cpu_part": missing,
        "ctl_internal_disagreements": ctl_self,
        "method": "complete answer object of every successful pyc invocation vs the ctl answer for the same "
        "request id and CPU part in this run",
    }


def status_of(res: dict) -> str:
    """The deployment outcome. It is judged on the start-class headlines (cold and restore first request total,
    warm inference), not on the phase the change targets. A phase that improves while the request it sits in
    does not is `no-change`, and the conclusion says so.

    improved: some headline's CI lies entirely below 0 on every CPU part both arms ran on, no headline's CI lies
    entirely above 0 on any part, and the answers are clean. regressed: any headline's CI lies entirely above 0.
    inconclusive: the answers are not clean, or there is no data."""
    d = res["e6"]["delta"]
    heads = [
        d["cold"].get("first_request_total", {}).get("by_cpu_part", {}),
        d["restore"].get("first_request_total", {}).get("by_cpu_part", {}),
        d["warm"].get("steady_state_inference", {}).get("by_cpu_part", {}),
    ]
    c = res["correctness"] or {}
    g = res["e6"]["arm_agreement"]
    # Answers: every pyc answer equals ctl's for the same request and CPU part (complete, from this run), and every
    # answer P1's reference covers equals it. Start-class mismatches are excluded from every distribution and listed
    # (P1's harness does the same); they are reported, not a correctness failure.
    clean = (
        not res["errors"]["count"]
        and c.get("bit_identical") == c.get("compared")
        and c.get("choice_flips") == 0
        and g["compared"] > 0
        and g["bit_identical"] == g["compared"]
        and g["no_ctl_answer_for_cpu_part"] == 0
        and g["ctl_internal_disagreements"] == 0
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = merge_runs(a.cold, a.restore, a.out)
    meta = json.load(open(a.meta))
    p1.ARM_PHASES = {
        CONTROL: "1",
        EXPERIMENT: "1",
    }  # both arms are instrumented: E6's control is ctl, not rep
    res = p1.summarize(run, a.reference, a.baseline_commit, meta)
    cond, rows = p1.load_run(run)
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = EXPERIMENT
    res["commits"]["experiment"] = cond["build_manifest"]["repo_commit"]
    res["deployment"]["code_sha256"] = {
        arm: cond["build_manifest"]["zips"][arm]["sha256"] for arm in (CONTROL, EXPERIMENT)
    }
    res["build"] = {
        k: cond["build_manifest"][k]
        for k in (
            "images",
            "toolchain",
            "assets_sha256",
            "zips",
            "base",
            "compiler",
            "bytecode",
            "pyc_unzipped_bytes",
        )
    }
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["legs"] = cond.get("legs")
    # Both arms are instrumented, so this run has no REPORT-only control. P1's run measured the overhead.
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "both E6 arms run P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["e6"] = {
        "delta": {
            "cold": delta(rows, "cold", COLD_KEYS),
            "warm": delta(rows, "warm", WARM_KEYS),
            "restore": delta(rows, "restore", RESTORE_KEYS),
            "method": "pyc minus ctl, medians within each CPU part both arms ran on; ci95 = bootstrap over "
            "environments (4,000 resamples, seed 6); per-round deltas from interleaved rounds",
        },
        "bytecode": bytecode(rows),
        "host": host(rows),
        "arm_agreement": arm_agreement(rows),
    }
    res["status"] = status_of(res)
    res["conclusion"] = (
        meta["conclusion_template"].format(**_fmt(res))
        if "conclusion_template" in meta
        else meta.get("conclusion")
    )
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    # The committed run directory (after redact.py and gzip of invocations.jsonl) holds exactly these.
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
    }
    path = os.path.join(a.out, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


def _fmt(res: dict) -> dict:
    """Numbers the conclusion template may cite."""
    out = {}
    for cls in ("cold", "restore", "warm"):
        for name, d in res["e6"]["delta"][cls].items():
            for cpu, v in d["by_cpu_part"].items():
                key = f"{cls}_{name}_{cpu}"
                out[key + "_ctl"] = v["ctl_p50"]
                out[key + "_pyc"] = v["pyc_p50"]
                out[key + "_d"] = v["delta_p50"]
                out[key + "_lo"], out[key + "_hi"] = v["ci95"]
                out[key + "_n"] = v["n"]
    return out


if __name__ == "__main__":
    sys.exit(main())
