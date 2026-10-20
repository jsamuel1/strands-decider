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
        if any("top" in r for r in recs):  # cProfile
            a["cprofile_init_import_thread_top_cumtime"] = _merge_top(
                [r["top"].get("import_thread") for r in inits if isinstance(r.get("top"), dict)], "fn", "cumtime")
            a["cprofile_init_load_thread_top_cumtime"] = _merge_top(
                [r["top"].get("load_thread") for r in inits if isinstance(r.get("top"), dict)], "fn", "cumtime")
            a["cprofile_invoke_top_tottime"] = _merge_top([r.get("top") for r in invs], "fn", "tottime")
        if any("pyspy" in r for r in recs):
            for kind, rs in (("init", inits), ("invoke", invs)):
                ps = [r["pyspy"] for r in rs]
                a[f"pyspy_{kind}"] = {
                    "records": len(ps),
                    "samples": sum(p.get("samples", 0) for p in ps),
                    "rc": dict(Counter(str(p.get("rc")) for p in ps)),
                    "attach_s": s1.dist([p.get("attach_s") for p in ps]),
                    "stderr_examples": sorted({p.get("stderr", "")[-200:] for p in ps if p.get("rc") not in (0, None)})[:3],
                    "top_leaf": _merge_top([p.get("top_leaf") for p in ps], "frame", "samples"),
                    "top_inclusive": _merge_top([p.get("top_inclusive") for p in ps], "frame", "samples"),
                }
        if any("sampler" in r for r in recs):
            for kind, rs in (("init", inits), ("invoke", invs)):
                sm = [r["sampler"] for r in rs if isinstance(r.get("sampler"), dict)]
                ph = defaultdict(list)
                for x in sm:
                    for name, t in (x.get("phases") or {}).items():
                        ph[name].append(t)
                a[f"sampler_{kind}"] = {
                    "records": len(sm),
                    "samples": sum(x.get("n", 0) for x in sm),
                    "dropped": sum(x.get("dropped", 0) for x in sm),
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
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    res = s1.summarize(a.run, a.reference, a.baseline_commit, meta)
    cond, rows = s1.load_run(a.run)
    res["primary_arm"] = "rep"
    res["profiler_overhead"] = {arm: overhead_vs(rows, arm) for arm in cond["arms"] if arm != "rep"}
    res.pop("instrumentation_overhead", None)
    res["profiles"] = {"lambda": profile_extracts(a.run, cond)}
    if a.offline:
        p = os.path.join(a.offline, "offline-summary.json")
        res["profiles"]["offline_not_lambda"] = json.load(open(p))
    if a.probe:
        res["lambda_probe"] = json.load(open(a.probe))
    res["applicability"] = json.load(open(a.applicability))
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
