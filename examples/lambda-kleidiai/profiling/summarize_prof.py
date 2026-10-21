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
import re
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
NULL_ARM = "rep2"  # deploy_arms.py: a second function byte-identical to rep
MIN_CELL_N = 3  # a per-round median from fewer valid samples than this is not compared


def paired_rounds(rows: list[dict], arm: str, ctrl: str = "rep", cpu: str = "0xd40", min_n: int = MIN_CELL_N) -> dict:
    """Per round, median(arm) - median(ctrl) of a REPORT field, on one CPU part; then mean/min/max over rounds.

    The arms of one round run back to back, so a per-round difference cancels drift between rounds (the warm control's
    own per-round median moved by 70 ms, about 10%, across the five rounds of run 20261020T124451Z). Each warm arm's
    passes also all land in one execution environment per round, so a round is one environment, and the spread over
    rounds is the honest uncertainty. A round is compared only when both medians rest on at least `min_n` valid
    samples; skipped rounds are listed with their counts, so a sparse control can never stand in for a round."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"] and r.get("cpu_part") == cpu]
    out = {}
    for cls, field in PAIRED_METRICS:
        per: dict[tuple, list[float]] = defaultdict(list)
        for r in ok:
            if r["arm"] in (arm, ctrl) and r["expected_class"] == cls and field in (r["report"] or {}):
                per[(r["arm"], r["round"])].append(r["report"][field])
        rounds = sorted({r["round"] for r in rows})
        used, skipped = [], []
        for k in rounds:
            na, nc = len(per.get((arm, k), [])), len(per.get((ctrl, k), []))
            (used if na >= min_n and nc >= min_n else skipped).append((k, na, nc))
        d = [statistics.median(per[(arm, k)]) - statistics.median(per[(ctrl, k)]) for k, _, _ in used]
        cell: dict = {"rounds": len(d), "min_n_per_round": min_n,
                      "skipped_rounds": [{"round": k, "n_arm": na, "n_ctrl": nc} for k, na, nc in skipped]}
        if d:
            base = statistics.median([v for k, _, _ in used for v in per[(ctrl, k)]])
            cell.update({"delta_ms_by_round": [round(x, 1) for x in d], "mean_ms": round(statistics.fmean(d), 1),
                         "min_ms": round(min(d), 1), "max_ms": round(max(d), 1), "control_p50_ms": round(base, 1),
                         "mean_pct": round(100 * statistics.fmean(d) / base, 2) if base else None})
        out[f"{cls}.{field}"] = cell
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
    pc = [(c, 100 * (statistics.median(per[(arm, *c)]) / statistics.median(per[(ctrl, *c)]) - 1)) for c in cells
          if per.get((arm, *c)) and per.get((ctrl, *c))]
    if not pc:
        return {}
    pct = [x for _, x in pc]
    # The cells of one round share one environment per arm, so they are not independent: the round (the mean of its
    # cells) is the unit the verdict tests, as for every other metric.
    by_round: dict = defaultdict(list)
    for (rnd, _rid), x in pc:
        by_round[rnd].append(x)
    return {"cells": len(pct), "median_pct": round(statistics.median(pct), 2), "min_pct": round(min(pct), 2),
            "max_pct": round(max(pct), 2), "pct_by_cell": [round(x, 3) for x in pct],
            "pct_by_round": [round(statistics.fmean(by_round[k]), 3) for k in sorted(by_round)]}


def permutation_p(a: list[float], b: list[float], n_perm: int = 20000, seed: int = 0) -> float | None:
    """Two-sided p-value of |mean(a) - mean(b)| under exchangeability of a and b. Exact over every split when there
    are at most 50,000 of them (5 vs 5 rounds: 252), else `n_perm` seeded random splits."""
    import itertools
    import math
    import random

    if len(a) < 2 or len(b) < 2:
        return None
    pool, k = a + b, len(a)
    obs = abs(statistics.fmean(a) - statistics.fmean(b)) - 1e-9
    tot = sum(pool)

    def stat(idx) -> float:
        s = sum(pool[i] for i in idx)
        return abs(s / k - (tot - s) / (len(pool) - k))

    if math.comb(len(pool), k) <= 50000:
        splits = list(itertools.combinations(range(len(pool)), k))
        return round(sum(stat(c) >= obs for c in splits) / len(splits), 4)
    rng = random.Random(seed)
    hits = sum(stat(rng.sample(range(len(pool)), k)) >= obs for _ in range(n_perm))
    return round((hits + 1) / (n_perm + 1), 4)


def null_verdicts(paired: dict, null_arm: str = NULL_ARM, alpha: float = 0.05) -> dict:
    """Judge every arm (X-Ray and the phase timers included) against the same-code null: `null_arm` is a second
    function byte-identical to the control, so its per-round deltas against the control are what a zero-cost arm
    looks like. Per metric, an arm's per-round deltas (warm: its per-(round, request) percentages) are tested against
    the null's with an exact two-sided permutation test on the mean. Warm per-request percentages are tested as their
    per-round means, since a round's cells share its environments. `measurable` needs p < alpha AND |mean| beyond
    the null's largest |per-round value|; `not measurable` otherwise; `insufficient` when either side has under 2
    values. The null arm itself is never judged against itself."""
    nul = paired.get(null_arm) or {}
    out: dict = {}
    for arm, cells in paired.items():
        if arm == null_arm:
            continue
        v: dict = {}
        for key, cell in cells.items():
            if key == "warm.Duration.by_request":
                a, b, unit = cell.get("pct_by_round") or [], (nul.get(key) or {}).get("pct_by_round") or [], "pct"
            else:
                a, b, unit = cell.get("delta_ms_by_round") or [], (nul.get(key) or {}).get("delta_ms_by_round") or [], "ms"
            p = permutation_p(a, b)
            floor = max((abs(x) for x in b), default=None)
            if p is None:
                verdict = "insufficient"
            else:
                verdict = "measurable" if p < alpha and abs(statistics.fmean(a)) > floor else "not measurable"
            v[key] = {"verdict": verdict, "p": p, "n_arm": len(a), "n_null": len(b), "unit": unit,
                      "arm_mean": round(statistics.fmean(a), 2) if a else None,
                      "null_mean": round(statistics.fmean(b), 2) if b else None,
                      "null_max_abs": round(floor, 2) if floor is not None else None}
        out[arm] = v
    return out


def noise_floor(paired: dict, null_arm: str = NULL_ARM) -> dict:
    """The largest |per-round delta| of the null arm (identical code, a different function) against the control."""
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


def _merge_top(lists: list[list[dict]], key: str, val: str, totals: list, top: int = 20,
               inclusive: bool = False) -> list[dict]:
    """Merge per-record top-N tables. `totals` holds each record's FULL denominator (all its samples, or all its
    own-time), so `share` is a share of everything recorded, not of the rows that survived truncation. A record whose
    total is missing makes every share None rather than renormalising over retained rows. `inclusive` tables (a
    frame counted once per sample it appears in, or cProfile cumtime) may sum past the denominator; exclusive ones
    may not."""
    c: Counter = Counter()
    for lst in lists:
        for x in lst or []:
            c[x[key]] += x[val]
    tot = None if any(t is None for t in totals) else sum(totals)
    if tot is not None and not inclusive and sum(c.values()) > tot * (1 + 1e-6):
        raise ValueError(f"merged {key} rows exceed their denominator {tot}")
    return [{key: k, val: round(v, 6), "share": round(v / tot, 4) if tot else None} for k, v in c.most_common(top)]


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
        # The first invocation after a SnapStart restore, per CPU part (its forward differs from a warm one).
        rest_invs = [r for r in invs if r["_class"] == "restore"]
        rest_by_cpu = {c: [r for r in rest_invs if r["_cpu"] == c] for c in sorted({r["_cpu"] for r in rest_invs}, key=str)}
        if any("top" in r for r in recs):  # cProfile
            def cp(rs, sub=None):
                rs = [r for r in rs if isinstance(r.get("top"), (dict, list))]
                return [(r["top"].get(sub) if sub else r["top"]) for r in rs], [r.get("tottime_total") for r in rs]

            li, ti = cp(inits, "all_threads")
            a["cprofile_init_top_cumtime"] = _merge_top(li, "fn", "cumtime", ti, inclusive=True)
            a["cprofile_init_top_tottime"] = _merge_top(li, "fn", "tottime", ti)
            a["cprofile_init_tottime_total"] = None if None in ti else round(sum(ti), 3)
            for name, rs in (("cprofile_invoke_top_tottime", invs), ("cprofile_invoke_warm_top_tottime", warm_invs)):
                lv, tv = cp(rs)
                a[name] = _merge_top(lv, "fn", "tottime", tv)
        if any("pyspy" in r for r in recs):
            for kind, rs in (("init", inits), ("invoke", invs), ("invoke_warm", warm_invs)):
                ps = [r["pyspy"] for r in rs if isinstance(r.get("pyspy"), dict)]
                tot = [p.get("samples") for p in ps]
                a[f"pyspy_{kind}"] = {
                    "records": len(ps),
                    "samples": sum(p.get("samples", 0) for p in ps),
                    "rc": dict(Counter(str(p.get("rc")) for p in ps)),
                    "popen_s": s1.dist([p.get("popen_s") for p in ps]),
                    "stderr_examples": sorted({p.get("stderr", "")[-200:] for p in ps if p.get("rc") not in (0, None)})[:3],
                    "top_leaf": _merge_top([p.get("top_leaf") for p in ps], "frame", "samples", tot),
                    "top_inclusive": _merge_top([p.get("top_inclusive") for p in ps], "frame", "samples", tot,
                                                inclusive=True),
                }
        if any("sampler" in r for r in recs):
            groups = [("init", inits), ("invoke", invs)] + [(f"invoke_warm_{c}", rs) for c, rs in warm_by_cpu.items()]
            groups += [(f"invoke_restore_{c}", rs) for c, rs in rest_by_cpu.items()]
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
                                      "top_self": _merge_top([t["top_self"] for t in ts], "symbol", "samples",
                                                             [t["samples"] for t in ts]),
                                      "by_library": _merge_top([t["by_library"] for t in ts], "library", "samples",
                                                               [t["samples"] for t in ts], 12),
                                      "top_caller_edges": _merge_top([t["top_caller_edges"] for t in ts], "edge",
                                                                     "samples", [t["samples"] for t in ts], 12)}
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


def verify_prof_manifest(pm: dict, commit: str, repo: str = HERE) -> dict:
    """Check that `commit` holds every repository input build_prof_zip.sh packaged, byte for byte: each `sources`
    entry (the .py files, lkaisamp.c, the Telemetry extension) by sha256, and every added artifact traced to a
    source or to the pinned py-spy wheel through `built_from`. The packaged .py files are copied verbatim, so their
    `added` hash must also equal their source hash. Returns the check record; exits on any mismatch."""
    import hashlib
    import subprocess

    def die(msg: str):
        sys.exit(f"provenance: {msg}")

    for k in ("sources", "built_from", "source_commit", "pyspy_wheel_sha256"):
        if k not in pm:
            die(f"prof-manifest has no {k!r} (lkai-prof-build/1 manifests do not bind non-Python inputs)")
    full = subprocess.run(["git", "-C", repo, "rev-parse", f"{commit}^{{commit}}"], capture_output=True, text=True)
    if full.returncode:
        die(f"{commit} is not a commit here")
    want_commit = subprocess.run(["git", "-C", repo, "rev-parse", f"{pm['source_commit']}^{{commit}}"],
                                 capture_output=True, text=True).stdout.strip()
    for f, want in pm["sources"].items():
        blob = subprocess.run(["git", "-C", repo, "show", f"{commit}:./{f}"], capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != want:
            die(f"{f} at {commit} is not the file the deployed package was built from")
    for art in pm["added"]:
        if art not in pm["built_from"]:
            die(f"packaged {art} has no recorded source")
    for art, srcname in pm["built_from"].items():
        if srcname != "pyspy_wheel" and srcname not in pm["sources"]:
            die(f"{art} is built from {srcname}, which has no source hash")
        if art.endswith(".py") and pm["added"].get(art) != pm["sources"].get(srcname):
            die(f"packaged {art} is not a verbatim copy of {srcname}")
    pinned = re.search(r"^PYSPY_SHA256=([0-9a-f]{64})$", subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:./build_prof_zip.sh"], capture_output=True, text=True).stdout, re.M)
    if not pinned or pinned.group(1) != pm["pyspy_wheel_sha256"]:
        die("the py-spy wheel is not the one build_prof_zip.sh pins at that commit")
    return {"commit": full.stdout.strip(), "source_commit": want_commit, "sources_checked": sorted(pm["sources"]),
            "pyspy_wheel_pinned": True, "result": "match"}


def restore_evidence(rows: list[dict], arms: list[str], cpu: str = "0xd40", min_n: int = MIN_CELL_N) -> dict:
    """Restore-wave integrity: per arm and round, how many restore invokes were sent, how many failed client-side, and
    how many valid Graviton3 samples remain. `usable` is False when any arm has a round below `min_n`; restore
    conclusions then must not be drawn (paired_rounds skips those rounds either way)."""
    rs = [r for r in rows if r["expected_class"] == "restore"]
    by: dict = {}
    for arm in arms:
        per = {}
        for k in sorted({r["round"] for r in rows}):
            mine = [r for r in rs if r["arm"] == arm and r["round"] == k]
            per[str(k)] = {"sent": len(mine), "client_errors": sum(str(r["error"] or "").startswith("client:") for r in mine),
                           "valid_0xd40": sum(not r["error"] and not r["class_mismatch"] and r.get("cpu_part") == cpu
                                              for r in mine)}
        by[arm] = per
    sparse = [(a, k) for a, per in by.items() for k, c in per.items() if c["valid_0xd40"] < min_n]
    return {"min_n": min_n, "by_arm_round": by, "sparse_cells": [{"arm": a, "round": int(k)} for a, k in sparse],
            "client_errors": sum(c["client_errors"] for per in by.values() for c in per.values()),
            "usable": not sparse}


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
    ap.add_argument("--superseded-by", help="regenerate a withdrawn run (no rep2 null, a pre-provenance manifest): its "
                    "overhead verdicts are withheld and its provenance recorded as unverifiable, never passed")
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
        if not a.experiment_commit:
            sys.exit("--prof-manifest needs --experiment-commit to bind it")
        res["deployment"]["prof_zip"] = pm
        if a.superseded_by and "sources" not in pm:
            res["deployment"]["prof_zip_provenance"] = {
                "result": "unverifiable", "commit": a.experiment_commit,
                "reason": "lkai-prof-build/1 manifest: no source hashes for lkaisamp.c or the Telemetry extension, and "
                          "no pinned py-spy wheel; only the copied .py files can be compared"}
        else:
            res["deployment"]["prof_zip_provenance"] = verify_prof_manifest(pm, a.experiment_commit)
        # Every deployed function must run the bytes the manifest names (bench.py checked CodeSha256 at run start).
        for arm, f in cond["functions"].items():
            ok = {pm["prof_zip_sha256"], pm["base_zip_sha256"]}
            if f.get("expected_code_sha256") not in ok:
                sys.exit(f"{arm}: deployed code {f.get('expected_code_sha256')} is neither prof.zip nor P1's zip")
            if f.get("layers") and f.get("expected_layer_sha256") != pm["telemetry_layer_sha256"]:
                if a.superseded_by and "sources" not in pm:
                    continue  # recorded above as unverifiable: that run predates layer hash recording
                sys.exit(f"{arm}: its layer is not the manifest's telemetry-layer.zip")
    res["primary_arm"] = "rep"
    res["profiler_overhead"] = {arm: overhead_vs(rows, arm) for arm in cond["arms"] if arm != "rep"}
    paired = {arm: paired_rounds(rows, arm) for arm in cond["arms"] if arm != "rep"}
    for arm in paired:
        paired[arm]["warm.Duration.by_request"] = paired_warm_by_request(rows, arm)
    if NULL_ARM not in paired and not a.superseded_by:
        sys.exit(f"no {NULL_ARM} arm: overhead needs a same-code null (deploy_arms.py deploys it)")
    rest = restore_evidence(rows, cond["arms"])
    res["restore_evidence"] = rest
    verdicts = null_verdicts(paired) if NULL_ARM in paired else {
        arm: {k: {"verdict": "withheld", "reason": f"no {NULL_ARM} null in this run"} for k in cells}
        for arm, cells in paired.items()}
    if not rest["usable"]:
        for v in verdicts.values():
            for k in list(v):
                if k.startswith("restore."):
                    v[k] = {**v[k], "verdict": "withheld", "reason": "restore cells below min_n"}
    res["profiler_overhead_paired"] = {
        "method": "per round: median(arm) - median(rep) of the REPORT field, Graviton3 (0xd40) rows only, no errors or "
                  f"class mismatches, each median from >= {MIN_CELL_N} samples (else the round is skipped and listed); "
                  "then mean/min/max over the rounds. Warm also per (round, request) as a percentage. Null: "
                  f"{NULL_ARM}, a second function byte-identical to rep, so {NULL_ARM} - rep is what zero cost looks "
                  "like. Each arm (xray and none included) is judged against that null by an exact two-sided "
                  "permutation test on the mean of its per-round deltas (warm by request: per-round mean %), alpha "
                  "0.05, and must also exceed the null's largest |per-round value| to be called measurable",
        "cpu_part": "0xd40",
        "null_arm": NULL_ARM,
        "noise_floor_ms": noise_floor(paired),
        "verdicts": verdicts,
        "arms": paired,
    }
    if a.superseded_by:
        res["superseded_by"] = a.superseded_by
        res["profiler_overhead_paired"]["withdrawn"] = (
            f"superseded by run {a.superseded_by}: this run reused one Lambda client across SnapStart waits (restore "
            "cells contaminated by client read timeouts) and has no same-code null arm")
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
    # The phase ranking needs P1's phase records: `rep` runs with LKAI_PHASES=0 and has none, so it reads `none`
    # (P1's timers, which the null test above finds no measurable cost for).
    res["hot_spots"] = {"phase_source_arm": "none", "by_phase_baseline": init_phase_shares(res["phases"]["none"])}
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
