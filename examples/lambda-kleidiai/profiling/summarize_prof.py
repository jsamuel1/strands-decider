"""Summarise a P3 profiler-overhead run (baseline/bench.py over profiling/deploy_arms.py's arms) as lkai-result/1.

  python summarize_prof.py <run-dir> --reference baseline/reference/outputs.json --probe <probe.json>
                           [--offline <dir of offline results>] --meta profiling/meta.json

Per arm it reuses baseline/summarize.py (same phase distributions, variance, cost, correctness). On top:
  profiler_overhead.<arm>.<class>.<metric>   median(arm) - median(rep) within each CPU part both ran on, pooled by the
                                             smaller sample count (P1's instrumentation_overhead method, any arm pair)
  profiles.lambda.<arm>                       what each profiler recorded in Lambda, from its LKAI_PROF / LKAI_TEL log
                                             lines and X-Ray: merged top frames per start class
  applicability                               the matrix: tool -> works / denied / inapplicable, with evidence
  hot_spots                                   the ranked hot spots of the baseline by phase (for E6)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
import summarize as s1  # noqa: E402


def overhead_vs(rows: list[dict], arm: str, ctrl: str = "rep") -> dict:
    """P1's matched-by-CPU-part comparison for one profiler arm against the control."""
    sub = [dict(r, arm={arm: "inst", ctrl: "rep"}[r["arm"]]) for r in rows if r["arm"] in (arm, ctrl)]
    out = s1.overhead(sub)
    out["method"] = out["method"].replace("median(inst) - median(rep)", f"median({arm}) - median({ctrl})")
    return out


PAIRED_METRICS = (("cold", "Init Duration"), ("cold", "Duration"), ("restore", "Restore Duration"),
                  ("restore", "Duration"), ("warm", "Duration"))


def paired_rounds(rows: list[dict], arm: str, ctrl: str = "rep", cpu: str = "0xd40") -> dict:
    """Per round, median(arm) - median(ctrl) of a REPORT field, on one CPU part; then mean/min/max over rounds.

    The arms of one round run back to back, so a per-round difference cancels drift between rounds (the warm control's
    own per-round median moved by 70 ms, about 10%, across the five rounds of run 20261020T124451Z). Each warm arm's
    passes also all land in one execution environment per round, so a round is one environment, and the spread over
    rounds is the honest uncertainty."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"] and r.get("cpu_part") == cpu]
    out = {}
    for cls, field in PAIRED_METRICS:
        per: dict[tuple, list[float]] = defaultdict(list)
        for r in ok:
            if r["arm"] in (arm, ctrl) and r["expected_class"] == cls and field in (r["report"] or {}):
                per[(r["arm"], r["round"])].append(r["report"][field])
        rounds = sorted({k[1] for k in per})
        d = [statistics.median(per[(arm, k)]) - statistics.median(per[(ctrl, k)]) for k in rounds
             if per.get((arm, k)) and per.get((ctrl, k))]
        if d:
            base = statistics.median([v for k in rounds for v in per.get((ctrl, k), [])])
            out[f"{cls}.{field}"] = {"rounds": len(d), "delta_ms_by_round": [round(x, 1) for x in d],
                                     "mean_ms": round(statistics.fmean(d), 1), "min_ms": round(min(d), 1),
                                     "max_ms": round(max(d), 1), "control_p50_ms": round(base, 1),
                                     "mean_pct": round(100 * statistics.fmean(d) / base, 2) if base else None}
    return out


def paired_warm_by_request(rows: list[dict], arm: str, ctrl: str = "rep", cpu: str = "0xd40") -> dict:
    """Warm only: per (round, request id), median(arm) / median(ctrl) - 1 of REPORT Duration, in percent; then the
    median and range over all cells. The five requests take 0.25-1.75 s, so a pooled warm median moves with the mix
    of requests that landed in it; pairing by request removes that."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"] and r.get("cpu_part") == cpu
          and r["expected_class"] == "warm" and r["arm"] in (arm, ctrl)]
    per: dict[tuple, list[float]] = defaultdict(list)
    for r in ok:
        per[(r["arm"], r["round"], r.get("request_id"))].append(r["report"]["Duration"])
    cells = sorted({k[1:] for k in per})
    pct = [100 * (statistics.median(per[(arm, *c)]) / statistics.median(per[(ctrl, *c)]) - 1) for c in cells
           if per.get((arm, *c)) and per.get((ctrl, *c))]
    if not pct:
        return {}
    return {"cells": len(pct), "median_pct": round(statistics.median(pct), 2), "min_pct": round(min(pct), 2),
            "max_pct": round(max(pct), 2)}


def noise_floor(paired: dict, null_arm: str) -> dict:
    """The largest |per-round delta| of an arm with no in-function code change against the control: a difference
    smaller than this is not distinguishable from environment-to-environment noise."""
    return {k: max(abs(v["min_ms"]), abs(v["max_ms"])) for k, v in paired.get(null_arm, {}).items() if "min_ms" in v}


def _events(run: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    d = os.path.join(run, "logs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                out[f.removesuffix(".jsonl.gz")].append(json.loads(line))
    return out


def _records(events: list[dict], prefix: str) -> list[dict]:
    recs = []
    for e in events:
        for line in e["message"].splitlines():
            i = line.find(prefix + " {")
            if i >= 0:
                try:
                    recs.append(json.loads(line[i + len(prefix) + 1:]) | {"_stream": e.get("logStreamName")})
                except ValueError:
                    pass
    return recs


def _merge_top(lists: list[list[dict]], key: str, val: str, top: int = 20) -> list[dict]:
    c: Counter = Counter()
    for lst in lists:
        for x in lst or []:
            c[x[key]] += x[val]
    tot = sum(c.values()) or 1
    return [{key: k, val: round(v, 6), "share": round(v / tot, 4)} for k, v in c.most_common(top)]


def profile_extracts(run: str, cond: dict) -> dict:
    ev = _events(run)
    out: dict = {}
    for arm, f in cond["functions"].items():
        fn = f["FunctionName"]
        recs = _records(ev.get(fn, []), "LKAI_PROF")
        tel = _records(ev.get(fn, []), "LKAI_TEL")
        a: dict = {"init_records": sum(r["kind"] == "init" for r in recs),
                   "invoke_records": sum(r["kind"] == "invoke" for r in recs)}
        inits = [r for r in recs if r["kind"] == "init"]
        invs = [r for r in recs if r["kind"] == "invoke"]
        # The start class and CPU part of each invocation, from P1's phase record in the same log stream (the
        # lkai_prof arms keep LKAI_PHASES=1). A `warm` profile is the steady state; Graviton2 sandboxes (0xd0c) load
        # the dot-product variant, so their kernels differ and the tables are kept apart by CPU part.
        cls = {(r["_stream"], (r.get("inv") or {}).get("n")): ((r.get("inv") or {}).get("class"),
                                                                (r.get("inv") or {}).get("cpu_part"))
               for r in _records(ev.get(fn, []), "LKAI") if r.get("kind") == "invoke"}
        for r in invs:
            r["_class"], r["_cpu"] = cls.get((r["_stream"], r.get("n")), (None, None))
        warm_invs = [r for r in invs if r["_class"] == "warm"]
        warm_by_cpu = {c: [r for r in warm_invs if r["_cpu"] == c] for c in sorted({r["_cpu"] for r in warm_invs}, key=str)}
        if any("top" in r for r in recs):  # cProfile
            a["cprofile_init_top_cumtime"] = _merge_top(
                [r["top"].get("all_threads") for r in inits if isinstance(r.get("top"), dict)], "fn", "cumtime")
            a["cprofile_init_top_tottime"] = _merge_top(
                [r["top"].get("all_threads") for r in inits if isinstance(r.get("top"), dict)], "fn", "tottime")
            a["cprofile_invoke_top_tottime"] = _merge_top([r.get("top") for r in invs], "fn", "tottime")
            a["cprofile_invoke_warm_top_tottime"] = _merge_top([r.get("top") for r in warm_invs], "fn", "tottime")
        if any("pyspy" in r for r in recs):
            for kind, rs in (("init", inits), ("invoke", invs), ("invoke_warm", warm_invs)):
                ps = [r["pyspy"] for r in rs if isinstance(r.get("pyspy"), dict)]
                a[f"pyspy_{kind}"] = {
                    "records": len(ps),
                    "samples": sum(p.get("samples", 0) for p in ps),
                    "rc": dict(Counter(str(p.get("rc")) for p in ps)),
                    "popen_s": s1.dist([p.get("popen_s") for p in ps]),
                    "stderr_examples": sorted({p.get("stderr", "")[-200:] for p in ps if p.get("rc") not in (0, None)})[:3],
                    "top_leaf": _merge_top([p.get("top_leaf") for p in ps], "frame", "samples"),
                    "top_inclusive": _merge_top([p.get("top_inclusive") for p in ps], "frame", "samples"),
                }
        if any("sampler" in r for r in recs):
            groups = [("init", inits), ("invoke", invs)] + [(f"invoke_warm_{c}", rs) for c, rs in warm_by_cpu.items()]
            for kind, rs in groups:
                sm = [r["sampler"] for r in rs if isinstance(r.get("sampler"), dict)]
                ph = defaultdict(list)
                for x in sm:
                    for name, t in (x.get("phases") or {}).items():
                        ph[name].append(t)
                a[f"sampler_{kind}"] = {
                    "records": len(sm),
                    "samples": sum(x.get("n", 0) for x in sm),
                    "dropped": sum(x.get("dropped", 0) for x in sm),
                    "cpu_s": round(sum(r.get("cpu_s") or 0 for r in rs), 3),
                    "effective_hz_per_cpu_s": round(sum(x.get("n", 0) for x in sm)
                                                    / max(1e-9, sum(r.get("cpu_s") or 0 for r in rs)), 1),
                    "phases": {name: {"samples": sum(t["samples"] for t in ts),
                                      "top_self": _merge_top([t["top_self"] for t in ts], "symbol", "samples"),
                                      "by_library": _merge_top([t["by_library"] for t in ts], "library", "samples", 12),
                                      "top_caller_edges": _merge_top([t["top_caller_edges"] for t in ts], "edge",
                                                                     "samples", 12)}
                               for name, ts in ph.items()},
                }
        if tel:
            a["telemetry"] = telemetry_summary(tel)
        out[arm] = a
    return out


def telemetry_summary(tel: list[dict]) -> dict:
    """Lambda's platform events from the Telemetry API extension: per-phase durations and spans."""
    types = Counter(t["type"] for t in tel)
    reps: dict[str, list] = defaultdict(list)
    spans: dict[str, list] = defaultdict(list)
    for t in tel:
        rec = t.get("record") or {}
        if t["type"] in ("platform.initReport", "platform.restoreReport", "platform.report"):
            m = rec.get("metrics") or {}
            reps[f"{t['type']}.durationMs"].append(m.get("durationMs"))
        if t["type"] == "platform.initRuntimeDone":
            for sp in rec.get("spans") or []:
                spans[f"init.{sp.get('name')}"].append(sp.get("durationMs"))
        if t["type"] == "platform.runtimeDone":
            for sp in rec.get("spans") or []:
                spans[f"invoke.{sp.get('name')}"].append(sp.get("durationMs"))
        if t["type"] == "lkai.subscribed":
            reps["extension_register_ms"].append((t.get("register_s") or 0) * 1000)
    return {"event_types": dict(types),
            "durations_ms": {k: s1.dist(v, unit="ms") for k, v in sorted(reps.items())},
            "spans_ms": {k: s1.dist(v, unit="ms") for k, v in sorted(spans.items())}}


def init_phase_shares(phases: dict) -> list[dict]:
    """Rank the cold-start phases by p50 share of the first request (Init + Duration), the baseline's own numbers."""
    c = phases.get("cold") or {}
    tot = (c.get("first_request_total") or {}).get("p50")
    rows = []
    for ph in ("weight_download", "model_load_mmap", "imports", "assets_load", "context_init", "warmup_inference",
               "first_request_inference", "ggml_backend_load", "runtime_to_import"):
        d = c.get(ph)
        if d:
            rows.append({"phase": ph, "class": "cold", "p50_s": d["p50"], "share_of_first_request": round(d["p50"] / tot, 4) if tot else None})
    r = phases.get("restore") or {}
    rt = (r.get("first_request_total") or {}).get("p50")
    for ph in ("report_restore_duration", "first_request_inference", "restore_to_invoke"):
        d = r.get(ph)
        if d:
            rows.append({"phase": ph, "class": "restore", "p50_s": d["p50"], "share_of_first_request": round(d["p50"] / rt, 4) if rt else None})
    return sorted(rows, key=lambda x: (x["class"], -x["p50_s"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference")
    ap.add_argument("--baseline-commit")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--probe", help="lkai_probe.py output from inside Lambda")
    ap.add_argument("--offline", help="directory with offline/GPU results (offline-summary.json)")
    ap.add_argument("--applicability", required=True, help="profiling/applicability.json (the matrix, hand-checked)")
    ap.add_argument("--experiment-commit", help="commit whose profiling/ files the deployed prof.zip carries (checked "
                    "by sha256 against the build's prof-manifest.json when --prof-manifest is given)")
    ap.add_argument("--prof-manifest", help="build_prof_zip.sh's prof-manifest.json")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    res = s1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = s1.load_run(a.run)
    if a.experiment_commit:
        # The run's conditions carry P1's build manifest (P1's zip is every arm's base). The profiler arms' own code
        # is prof.zip = that zip + profiling/ files; record the commit that holds those exact files.
        res["commits"]["experiment"] = a.experiment_commit
        res["commits"]["p1_zip_built_from"] = cond["build_manifest"]["repo_commit"]
    if a.prof_manifest:
        pm = json.load(open(a.prof_manifest))
        if a.experiment_commit:
            import hashlib
            import subprocess

            for f, want in pm["added"].items():
                if f.endswith(".py"):
                    blob = subprocess.run(["git", "show", f"{a.experiment_commit}:examples/lambda-kleidiai/profiling/{f}"],
                                          capture_output=True, check=True).stdout
                    if hashlib.sha256(blob).hexdigest() != want:
                        sys.exit(f"{f} at {a.experiment_commit} is not the deployed file")
        res["deployment"]["prof_zip"] = pm
    res["primary_arm"] = "rep"
    res["profiler_overhead"] = {arm: overhead_vs(rows, arm) for arm in cond["arms"] if arm != "rep"}
    paired = {arm: paired_rounds(rows, arm) for arm in cond["arms"] if arm != "rep"}
    for arm in paired:
        paired[arm]["warm.Duration.by_request"] = paired_warm_by_request(rows, arm)
    res["profiler_overhead_paired"] = {
        "method": "per round: median(arm) - median(rep) of the REPORT field, Graviton3 (0xd40) rows only, no errors or "
                  "class mismatches; then mean/min/max over the rounds. noise_floor_ms: the largest |per-round delta| of "
                  "the xray arm, which runs P1's zip unchanged (Lambda's tracing is outside the function's Duration)",
        "cpu_part": "0xd40",
        "noise_floor_ms": noise_floor(paired, "xray"),
        "arms": paired,
    }
    res.pop("instrumentation_overhead", None)
    res["profiles"] = {"lambda": profile_extracts(a.run, cond)}
    if a.offline:
        p = os.path.join(a.offline, "offline-summary.json")
        res["profiles"]["offline_not_lambda"] = json.load(open(p))
    if a.probe:
        res["lambda_probe"] = json.load(open(a.probe))
    res["applicability"] = json.load(open(a.applicability))
    # Artifacts published beside result.json (run-relative paths; the publisher uploads and links each one).
    res["profiling_artifacts"] = [
        {"name": "applicability matrix", "path": "applicability.json"},
        {"name": "X-Ray segment summary", "path": "xray-summary.json"},
        {"name": "X-Ray traces (sampled)", "path": "xray.jsonl.gz"},
        {"name": "offline profiles summary (NOT Lambda)", "path": "offline-summary.json"},
        {"name": "README: matrix, overhead, hot spots by phase", "path": "README.md"},
    ]
    res["artifacts"].update({"xray": "xray.jsonl.gz", "xray_summary": "xray-summary.json",
                             "applicability": "applicability.json", "offline_summary": "offline-summary.json",
                             "readme": "README.md"})
    res["hot_spots"] = {"by_phase_baseline": init_phase_shares(res["phases"]["rep"])}
    # Correctness per arm: every arm must answer exactly as the baseline (a profiler must not change results).
    if a.reference:
        ref = json.load(open(a.reference))["outputs"]
        res["correctness_by_arm"] = {arm: s1.compare(s1.by(rows, arm=arm), ref) for arm in cond["arms"]}
    medians = {arm: statistics.median([r["rt_s"] for r in s1.by(rows, arm=arm) if not r["error"]] or [0])
               for arm in cond["arms"]}
    res["client_round_trip_p50_by_arm"] = {k: round(v, 4) for k, v in medians.items()}
    path = os.path.join(a.run, "results.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
