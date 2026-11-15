"""Summarize an E16 run (e6 against lazy) into one lkai-result/1 record.

  python summarize_e16.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <E6 commit> --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

E6's summarizer (e6/summarize_e6.py) does the statistics: per-CPU-part medians and bootstrap CIs over environments, the
shared (round, CPU part) cells with their control-spread outcomes, the snapshot-init records bound to their own log
stream's CPU part, the in-run answer agreement and the headline status rule. This module points it at E16's arms
(control e6, experiment lazy) and E16's metrics, and adds E16's mechanism evidence: which model-stack packages the
init's main thread had imported, where the engine import ran, and the bytecode check.

Primary (targeted) cold metrics: REPORT Init Duration and the `imports` phase timer (the main thread's imports).
load_start (the download's start), engine_import and the first-request headline are reported beside them. No profiler
runs in either arm.
"""

# ruff: noqa: SLF001
# (Points E6's summarizer at E16's arms and metrics through its module-level helpers and globals.)
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KLEIDIAI = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(KLEIDIAI, "baseline"))
sys.path.insert(0, os.path.join(KLEIDIAI, "e6"))
import summarize_e6 as s6  # noqa: E402

CONTROL, EXPERIMENT = "e6", "lazy"
s6.CONTROL, s6.EXPERIMENT = CONTROL, EXPERIMENT
s6.COLD_KEYS = {
    "imports": lambda r: s6._init(r, "imports"),
    "report_init_duration": lambda r: s6._rep(r, "Init Duration"),
    "load_start": lambda r: s6._init(r, "load_start"),
    "engine_import": lambda r: s6._init(r, "engine_import"),
    "import_total": lambda r: s6._init(r, "import_total"),
    "runtime_to_import": lambda r: s6._init(r, "runtime_to_import"),
    "weight_download": lambda r: s6._init(r, "weight_download"),
    "model_ready": lambda r: s6._init(r, "model_ready"),
    "model_load_mmap": lambda r: s6._init(r, "model_load_mmap"),
    "first_request_total": lambda r: (
        (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000
    ),
    "report_max_memory_mb": lambda r: r["report"].get("Max Memory Used"),
}
s6.SNAPSHOT_METRICS = ("imports", "engine_import", "import_total", "model_ready")

# lkai_entry.py as deployed for run 20261115T215333Z (code acc1b94). Its SnapStart branch added the same model_ready
# span twice (fixed after the run), so every init record from a snapshot init carries 2x the span: the before_snapshot
# records in the logs, and the init each restored environment reports on its first invocation (the snapshot's own
# init, carried through the restore). An on-demand (cold) init records model_ready from handler._load's own span and
# is unaffected. The two adds read the clock microseconds apart, so half the recorded value is the span to ~1 us.
DOUBLED_MODEL_READY_HARNESS = "aa5498374931f91c5543c532056c0865506d8800df833b49ddf6e005995a78a0"


def _halve_model_ready(init: dict | None) -> bool:
    if not init or init.get("model_ready") is None:
        return False
    init["model_ready"] = round(init["model_ready"] / 2, 6)
    return True


def install_model_ready_correction(cond: dict) -> dict:
    """Correct snapshot-init model_ready at the two places init records are read (the log parser and the invocation
    rows), so every consumer sees the single span, when the run's harness is the one that doubled it. Returns the
    correction's record for results.json, with how many records each seam corrected (filled as they are read)."""
    harness = ((cond.get("build_manifest") or {}).get("harness_sha256") or {}).get("lkai_entry.py")
    rec = {
        "field": "model_ready",
        "applies": harness == DOUBLED_MODEL_READY_HARNESS,
        "harness_lkai_entry_sha256": harness,
        "before_snapshot_records": 0,
        "restore_invocation_records": 0,
        "method": "The deployed lkai_entry.py added the snapshot init's model_ready span twice (Recorder.add "
        "accumulates). Snapshot-init records (before_snapshot log records and each restored environment's first "
        "invocation record) are halved before any summary reads them. On-demand cold inits are unaffected. Raw "
        "logs and invocations keep the recorded values.",
    }
    if not rec["applies"]:
        return rec
    phases_mod = s6.p1.phases
    parse = phases_mod.parse_lkai
    seen: set = set()  # P1's and E6's readers both parse the logs: count each record once

    def parse_lkai(log: str) -> list[dict]:
        out = parse(log)  # fresh dicts on every parse, so each is halved exactly once
        for r in out:
            if r.get("kind") == "before_snapshot" and _halve_model_ready(r.get("init")):
                key = (r.get("wall_ns"), r.get("env_id"))
                if key not in seen:
                    seen.add(key)
                    rec["before_snapshot_records"] += 1
        return out

    phases_mod.parse_lkai = parse_lkai
    load = s6.p1.load_run

    def load_run(run: str):
        cond_, rows = load(run)
        n = 0
        for r in rows:
            lk = r.get("lkai") or {}
            inv = lk.get("inv") or {}
            if inv.get("class") == "restore" and _halve_model_ready(lk.get("init")):
                n += 1
        rec["restore_invocation_records"] = n
        return cond_, rows

    s6.p1.load_run = load_run
    return rec


def p1_summarize_arms() -> None:
    """Both E16 arms are instrumented (LKAI_PHASES=1): tell P1's summarize which arms to expect."""
    s6.p1.ARM_PHASES = {CONTROL: "1", EXPERIMENT: "1"}


def detection_floor(
    res: dict, metric: str = "report_init_duration", cpu: str = "0xd40"
) -> dict | None:
    """The smallest effect this run could have told from zero on one metric and CPU part: the half-width of its 95 %
    bootstrap CI for the e6-minus-lazy median difference, as measured. Not a power calculation."""
    v = ((res["e16"]["delta"]["cold"].get(metric) or {}).get("by_cpu_part") or {}).get(cpu)
    if not v:
        return None
    lo, hi = v["ci95"]
    return {
        "metric": metric,
        "cpu_part": cpu,
        "ci95_half_width_s": round((hi - lo) / 2, 6),
        "n": v["n"],
    }


def mechanism(rows: list[dict]) -> dict:
    """Per arm, from each environment's first-invocation e16 record: the main thread's model-stack modules at the
    end of the init, the thread that imported engine, and the bytecode check."""
    out: dict = {}
    for arm in (CONTROL, EXPERIMENT):
        recs = [
            r
            for r in rows
            if r["arm"] == arm and not r["error"] and ((r.get("lkai") or {}).get("e16"))
        ]
        if not recs:
            continue
        stack = Counter(
            json.dumps(r["lkai"]["e16"]["main_thread_stack_at_init_end"], sort_keys=True)
            for r in recs
        )
        threads = Counter(
            ((r["lkai"].get("counters") or {}).get("engine_import_thread") or "unknown")
            for r in recs
        )
        byte = Counter(
            json.dumps(
                {
                    k: r["lkai"]["e16"]["bytecode"][k]
                    for k in ("recompiled_from_source", "usable_unchecked_hash")
                }
            )
            for r in recs
        )
        out[arm] = {
            "environments": len(recs),
            "main_thread_stack_at_init_end": [
                {"count": n, **json.loads(k)} for k, n in stack.most_common()
            ],
            "engine_import_thread": dict(threads.most_common()),
            "bytecode": [{"count": n, **json.loads(k)} for k, n in byte.most_common()],
            "first_invoke_check_s_p50": statistics.median(
                r["lkai"]["e16"]["first_invoke_check_s"] for r in recs
            ),
        }
    return out


def host(rows: list[dict]) -> dict:
    """Distinct host identities per arm and class (H1's hostinfo record), with their counts. No generation."""
    ids: dict = defaultdict(Counter)
    detail: dict = {}
    for r in rows:
        e = (r.get("lkai") or {}).get("e16")
        if not e or r["error"]:
            continue
        h = e.get("host") or {}
        fp = h.get("fingerprint", "unknown")
        cpu = h.get("cpu") or {}
        ids[f"{r['arm']}:{r['expected_class']}"][fp] += 1
        detail.setdefault(
            fp,
            {
                "part": cpu.get("part"),
                "variant": cpu.get("variant"),
                "revision": cpu.get("revision"),
                "implementer": cpu.get("implementer"),
                "processors": cpu.get("processors"),
                "hwcap": h.get("hwcap"),
                "cores": h.get("cores"),
                "cgroup": h.get("cgroup"),
                "mem_total_kb": h.get("mem_total_kb"),
                "kernel": (h.get("kernel") or {}).get("release"),
                "python": h.get("python"),
                "unknown": sorted((h.get("unknown") or {}).keys()),
            },
        )
    return {
        "counts": {k: dict(v.most_common()) for k, v in sorted(ids.items())},
        "identities": detail,
        "note": "baseline/hostinfo.py (H1) read inside the function at init and again after a restore. A null field "
        "or an `unknown` key was not exposed. No processor generation is inferred from a CPU part or a timing.",
    }


def download_completion(rows: list[dict], seed: int = 16) -> dict:
    """When did the download end, and the model become ready, measured from the clock start (lkai_entry's import)?

    weight_download alone is not comparable between arms: lazy's download starts earlier. load_start + the load's
    pre-download time + weight_download is. Here, from each environment's init record:
      pre_download   model_ready - engine_load - warmup_inference - weight_download (the load's time before the
                     download returned that was not the download: on lazy, the wait for the engine import)
      download_done  load_start + model_ready - engine_load - warmup_inference
      ready          load_start + model_ready
    lazy minus e6 medians per CPU part, bootstrap 95 % CI over environments."""
    import random

    def ci(a: list[float], b: list[float], n: int = 4000) -> list[float]:
        rnd, d = random.Random(seed), []
        for _ in range(n):
            d.append(
                statistics.median(rnd.choices(b, k=len(b)))
                - statistics.median(rnd.choices(a, k=len(a)))
            )
        d.sort()
        return [round(d[int(0.025 * n)], 4), round(d[int(0.975 * n)], 4)]

    keys = ("load_start", "model_ready", "engine_load", "warmup_inference", "weight_download")
    out: dict = {}
    for cpu in sorted(
        {r.get("cpu_part") for r in s6.ok_rows(rows, "cold")} - set(s6.UNKNOWN_CPU), key=str
    ):
        vals: dict = {}
        for arm in (CONTROL, EXPERIMENT):
            inits = [
                i
                for r in s6.ok_rows(rows, "cold")
                if r["arm"] == arm and r.get("cpu_part") == cpu
                if all(k in (i := (r.get("lkai") or {}).get("init") or {}) for k in keys)
            ]
            vals[arm] = {
                "pre_download": [
                    i["model_ready"]
                    - i["engine_load"]
                    - i["warmup_inference"]
                    - i["weight_download"]
                    for i in inits
                ],
                "download_done": [
                    i["load_start"] + i["model_ready"] - i["engine_load"] - i["warmup_inference"]
                    for i in inits
                ],
                "ready": [i["load_start"] + i["model_ready"] for i in inits],
            }
        for k in ("pre_download", "download_done", "ready"):
            a, b = vals[CONTROL][k], vals[EXPERIMENT][k]
            if not a or not b:
                continue
            out.setdefault(k, {})[str(cpu)] = {
                CONTROL: round(statistics.median(a), 4),
                EXPERIMENT: round(statistics.median(b), 4),
                "delta": round(statistics.median(b) - statistics.median(a), 4),
                "ci95": ci(a, b) if min(len(a), len(b)) > 1 else None,
                "n": [len(a), len(b)],
            }
    out["method"] = download_completion.__doc__.split("\n\n", 1)[1].strip()
    return out


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
    p1_summarize_arms()
    run = s6.merge_runs(a.cold, a.restore, a.out)
    meta = json.load(open(a.meta))
    correction = install_model_ready_correction(
        json.load(open(os.path.join(run, "conditions.json")))
    )
    res = s6.p1.summarize(run, a.reference, a.baseline_commit, meta)
    cond, rows = s6.p1.load_run(run)
    m = cond["build_manifest"]
    res["experiment_id"] = meta["experiment_id"]
    res["run_id"] = a.run_id
    res["primary_arm"] = EXPERIMENT
    res["commits"]["experiment"] = m["repo_commit"]
    res["deployment"]["code_sha256"] = {
        arm: m["zips"][arm]["sha256"] for arm in (CONTROL, EXPERIMENT)
    }
    res["build"] = {
        k: m[k]
        for k in (
            "images",
            "toolchain",
            "assets_sha256",
            "zips",
            "base",
            "compiler",
            "harness_sha256",
            "placement_sha256",
        )
    }
    res["conditions"]["read_timeout_s"] = cond.get("read_timeout_s")
    res["conditions"]["legs"] = cond.get("legs")
    res["conditions"]["interleaving"] = cond.get("interleaving")
    res["instrumentation_overhead"] = {
        "measured": False,
        "note": "both E16 arms run P1's phase timers; overhead is P1's (P1-baseline/20261020T080835Z)",
    }
    res["host_variability"] = s6.host_variability_analysis(rows)
    res["host_variability"]["snapshot_init"] = s6.snapshot_host_analysis(run, cond)
    res["e16"] = {
        "delta": {
            "cold": s6.delta(rows, "cold", s6.COLD_KEYS),
            "warm": s6.delta(rows, "warm", s6.WARM_KEYS),
            "restore": s6.delta(rows, "restore", s6.RESTORE_KEYS),
            "method": "lazy minus e6, medians within each CPU part both arms ran on; ci95 = bootstrap over "
            "environments (4,000 resamples, seed 6); both arms ran in every wave (interleaved); pooled per-round "
            "deltas are descriptive only; host_variability contains the round-by-CPU-part cells",
        },
        "mechanism": mechanism(rows),
        "download_completion": download_completion(rows),
        "host": host(rows),
        "arm_agreement": s6.arm_agreement(rows),
    }
    res["e6"] = res["e16"]  # status_of and _fmt read res["e6"]
    res["e16"]["detection_floor"] = detection_floor(res)
    res["status"] = s6.status_of(res)
    del res["e6"]
    res["conclusion"] = meta.get("conclusion")
    res["corrections"] = [correction] if correction["applies"] else []
    res["profiling_artifacts"] = meta.get("profiling_artifacts", [])
    res["artifacts"] = {
        "invocations": "invocations.jsonl.gz",
        "logs": "logs/",
        "conditions": "conditions.json",
        "build_manifest": "build-manifest.json",
        "meta": "meta.json",
        "offline_importtime_e6": "offline/importtime_e6.json",
        "offline_importtime_lazy": "offline/importtime_lazy.json",
        "offline_pairs": "offline/pairs.json",
    }
    path = os.path.join(a.out, "results.json")
    json.dump(res, open(path, "w"), indent=1, allow_nan=False)
    print("wrote", path, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
