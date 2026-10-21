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
        cell: dict = {"rounds": len(d), "round_ids": [k for k, _, _ in used], "min_n_per_round": min_n,
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
    rounds = sorted(by_round)
    return {"cells": len(pct), "median_pct": round(statistics.median(pct), 2), "min_pct": round(min(pct), 2),
            "max_pct": round(max(pct), 2), "pct_by_cell": [round(x, 3) for x in pct], "round_ids": rounds,
            "pct_by_round": [round(statistics.fmean(by_round[k]), 3) for k in rounds]}


def sign_flip_p(d: list[float]) -> float | None:
    """Exact two-sided paired sign-flip (randomisation) p-value of |mean(d)| over all 2^n sign assignments of the
    per-pair differences `d`. Under the null each pair's difference is symmetric about 0, so flipping its sign is an
    equally likely relabelling. The smallest attainable value is 2 / 2^n (n = 5 pairs: 0.0625). None below 2 pairs."""
    import itertools

    if len(d) < 2:
        return None
    obs = abs(statistics.fmean(d)) - 1e-9
    flips = list(itertools.product((1, -1), repeat=len(d)))
    return round(sum(abs(statistics.fmean([s * x for s, x in zip(sg, d, strict=True)])) >= obs for sg in flips) / len(flips), 4)


def null_verdicts(paired: dict, null_arm: str = NULL_ARM, alpha: float = 0.05) -> dict:
    """Judge every arm (X-Ray and the phase timers included) against the same-code null. `null_arm` is a second
    function byte-identical to the control, run in the same rounds, so per round both `arm - rep` and `null - rep`
    share that round's `rep` median: they are PAIRED, not two independent groups. The per-round test statistic is
    their difference, (arm - rep) - (null - rep) = arm - null, over the rounds both arms kept (warm: per-round mean
    percentages, since a round's cells share its environments). Inference is an exact two-sided sign-flip test on
    the mean of those paired differences. With n pairs the smallest attainable p is 2 / 2^n, so 5 rounds cannot
    reach p < 0.05: `min_attainable_p` and `underpowered` say so on every cell.

    `measurable` needs p < alpha AND |mean(arm - rep)| beyond the null's largest |per-round value|; `not measurable`
    otherwise; `insufficient` under 2 paired rounds. `consistent_direction` (descriptive only, not a test) is True
    when every paired difference has the same sign and every per-round |arm - rep| exceeds the null's largest
    |per-round value|. The null arm itself is never judged."""
    nul = paired.get(null_arm) or {}
    out: dict = {}
    for arm, cells in paired.items():
        if arm == null_arm:
            continue
        v: dict = {}
        for key, cell in cells.items():
            vals, unit = ("pct_by_round", "pct") if key == "warm.Duration.by_request" else ("delta_ms_by_round", "ms")
            ncell = nul.get(key) or {}
            a = dict(zip(cell.get("round_ids") or [], cell.get(vals) or [], strict=True))
            b = dict(zip(ncell.get("round_ids") or [], ncell.get(vals) or [], strict=True))
            common = sorted(set(a) & set(b))
            d = [a[k] - b[k] for k in common]
            av, bv = [a[k] for k in common], [b[k] for k in common]
            p = sign_flip_p(d)
            floor = max((abs(x) for x in bv), default=None)
            if p is None:
                verdict = "insufficient"
            else:
                verdict = "measurable" if p < alpha and abs(statistics.fmean(av)) > floor else "not measurable"
            min_p = round(2 / 2 ** len(d), 4) if len(d) >= 2 else None
            v[key] = {"verdict": verdict, "p": p, "test": "exact two-sided paired sign-flip on arm - null per round",
                      "n_pairs": len(d), "rounds": common, "min_attainable_p": min_p,
                      "underpowered": min_p is None or min_p >= alpha, "unit": unit,
                      "paired_diff_by_round": [round(x, 3) for x in d],
                      "paired_diff_mean": round(statistics.fmean(d), 2) if d else None,
                      "consistent_direction": bool(len(d) >= 2 and (all(x > 0 for x in d) or all(x < 0 for x in d))
                                                   and floor is not None and all(abs(x) > floor for x in av)),
                      "arm_mean": round(statistics.fmean(av), 2) if av else None,
                      "null_mean": round(statistics.fmean(bv), 2) if bv else None,
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
        # lkai_prof arms keep LKAI_PHASES=1). A `warm` profile is the steady state. Each CPU part ID loads its own
        # libggml-cpu variant (logged at init), so kernels differ and tables are kept apart by CPU part ID. No
        # processor generation is inferred from a part ID.
        cls = {(r["_stream"], (r.get("inv") or {}).get("n")): ((r.get("inv") or {}).get("class"),
                                                                (r.get("inv") or {}).get("cpu_part"))
               for r in _records(ev.get(fn, []), "LKAI") if r.get("kind") == "invoke"}
        # A stream is one execution environment; its part ID never changed within a stream in the committed runs
        # (host_variability.observed.log_stream_linkage), so an init record takes its stream's part ID.
        stream_cpu: dict = defaultdict(set)
        for (stream, _n), (_c, cpu) in cls.items():
            stream_cpu[stream].add(cpu)
        for r in inits:
            parts = stream_cpu.get(r["_stream"]) or set()
            r["_cpu"] = next(iter(parts)) if len(parts) == 1 else None
        init_by_cpu = {c: [r for r in inits if r["_cpu"] == c] for c in sorted({r["_cpu"] for r in inits}, key=str)}
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
            groups = [("init", inits), ("invoke", invs)] + [(f"init_{c}", rs) for c, rs in init_by_cpu.items()]
            groups += [(f"invoke_warm_{c}", rs) for c, rs in warm_by_cpu.items()]
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
    how many valid samples on CPU part 0xd40 remain. `usable` is False when any arm has a round below `min_n`; restore
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


def _strip_lib(sym: str) -> str:
    return sym.split(":", 1)[1] if ":" in sym else sym


HOST_UNKNOWN = {
    "processor_generation_or_microarchitecture": "not observed: only the MIDR CPU part ID is recorded per invocation; "
    "no name is inferred from a part ID or from timings",
    "cpu_implementer_variant_revision_per_invocation": "not recorded per invocation (the probe read them, below, "
    "in 7 environments of a separate function, all part 0xd40)",
    "cpu_features_hwcaps_per_invocation": "not recorded (no /proc/cpuinfo Features or AT_HWCAP per invocation)",
    "hwcaps_for_0xd0c_0xd4f": "never observed: the probe never landed on those part IDs",
    "core_count_per_invocation": "not recorded; threads=6 is the configured thread count, not an observed nproc",
    "restored_core_count": "not recorded: nproc after a SnapStart restore was never read",
    "cgroup_cpu_memory_limits": "not recorded (cpu.max / memory.max were not read)",
    "kernel_per_invocation": "not recorded (the probe saw one kernel release in its 7 environments)",
    "availability_zone": "not observable from inside the function and not recorded",
    "physical_host_identity": "not observable",
    "snapshot_build_host_cpu": "not recorded: the CPU part of the environment that ran publish_version's init",
}


def host_variability(rows: list[dict], cond: dict, run: str, probe_dir: str | None = None,
                     profiles: dict | None = None) -> dict:
    """Machine-readable host identity for this run: what was observed, what was not (explicitly `unknown`), and how
    comparisons handle it. Reanalysis of the committed evidence only: no AWS call, no new measurement."""
    ok = [r for r in rows if not r["error"] and not r["class_mismatch"]]
    by_arm_class: dict = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        by_arm_class[r["arm"]][r["expected_class"]][str(r.get("cpu_part"))] += 1
    totals = Counter(str(r.get("cpu_part")) for r in rows)
    by_round: dict = defaultdict(Counter)
    for r in rows:
        by_round[str(r["round"])][str(r.get("cpu_part"))] += 1
    # Which libggml-cpu variants each part ID loaded (the engine logs them at init, cold rows only).
    variants: dict = defaultdict(Counter)
    for r in rows:
        m = re.search(r"init cpu_part=(\S+) ggml_cpu=(\S+)", r.get("log_tail") or "")
        if m:
            variants[m.group(1)][m.group(2)] += 1
    # Log-stream linkage: a row's START RequestId in its log tail -> the CloudWatch stream that logged it.
    req2stream: dict = {}
    runtimes: Counter = Counter()
    d = os.path.join(run, "logs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        with gzip.open(os.path.join(d, f), "rt") as fh:
            for line in fh:
                e = json.loads(line)
                for m in re.finditer(r"START RequestId: (\S+)", e["message"]):
                    req2stream[m.group(1)] = e.get("logStreamName")
                m = re.search(r"INIT_START Runtime Version: (\S+)", e["message"])
                if m:
                    runtimes[m.group(1)] += 1
    stream_parts: dict = defaultdict(set)
    linked = 0
    for r in rows:
        m = re.search(r"START RequestId: (\S+)", r.get("log_tail") or "")
        s = req2stream.get(m.group(1)) if m else None
        if s:
            linked += 1
            stream_parts[(r["arm"], s)].add(str(r.get("cpu_part")))
    streams_by_part = Counter(next(iter(p)) for p in stream_parts.values() if len(p) == 1)
    # REPORT Duration p50 per (arm, class, part): whether a figure differs by part ID.
    rep_by_part: dict = defaultdict(dict)
    groups: dict = defaultdict(list)
    for r in ok:
        if "Duration" in (r["report"] or {}):
            groups[(r["arm"], r["expected_class"], str(r.get("cpu_part")))].append(r["report"]["Duration"])
    for (arm, c, p), v in sorted(groups.items()):
        rep_by_part[f"{arm}.{c}"][p] = {"n": len(v), "duration_p50_ms": round(statistics.median(v), 1)}
    warm_req: dict = defaultdict(lambda: defaultdict(list))
    for r in ok:
        if r["arm"] == "rep" and r["expected_class"] == "warm" and r.get("latency_ms") is not None:
            warm_req[r["request_id"]][str(r.get("cpu_part"))].append(r["latency_ms"])
    engine_warm = {rid: {p: {"n": len(v), "engine_p50_ms": round(statistics.median(v), 1)} for p, v in sorted(ps.items())}
                   for rid, ps in sorted(warm_req.items())}
    # Native hot spots per part ID (lkaisamp), library prefix stripped so the same kernel in different variants lines up.
    kernels: dict = {}
    smp = ((profiles or {}).get("sampler") or {})
    for k, v in sorted(smp.items()):
        if k.startswith(("sampler_invoke_warm_", "sampler_invoke_restore_", "sampler_init_0x")):
            ph = (v.get("phases") or {})
            name = "init" if k.startswith("sampler_init_") else "invoke"
            top = (ph.get(name) or {}).get("top_self") or []
            merged: Counter = Counter()
            for x in top:  # the same kernel can appear once per loaded variant library; shares share a denominator
                if x["share"] is not None:
                    merged[_strip_lib(x["symbol"])] += x["share"]
            kernels[k.removeprefix("sampler_")] = {"records": v.get("records"), "samples": (ph.get(name) or {}).get("samples"),
                                                   "top_self": [{"symbol": s, "share": round(sh, 4)}
                                                                for s, sh in merged.most_common(10)]}
    probes = {}
    for p in sorted(os.listdir(probe_dir)) if probe_dir and os.path.isdir(probe_dir) else []:
        if p.endswith(".json"):
            j = json.load(open(os.path.join(probe_dir, p)))
            c = (j.get("cpuinfo") or {}).get("first") or {}
            probes[p] = {"implementer": c.get("CPU implementer"), "architecture": c.get("CPU architecture"),
                         "variant": c.get("CPU variant"), "part": c.get("CPU part"), "revision": c.get("CPU revision"),
                         "features": c.get("Features"), "processors": (j.get("cpuinfo") or {}).get("processors"),
                         "nproc": j.get("nproc"), "kernel": (j.get("uname") or {}).get("release"),
                         "python": j.get("python"), "mem_total": (j.get("meminfo_head") or [None])[0]}
    parts = sorted(totals)
    return {
        "schema": "lkai-host-variability/1",
        "new_measurements": False,
        "aws_calls": 0,
        "source": "reanalysis of this run's committed invocations.jsonl.gz, logs/ and conditions.json, plus the "
                  "committed lambda-probe/*.json; no rerun",
        "identity_label": "MIDR CPU part ID as read from /proc/cpuinfo per request (handler._cpu_part); used as an "
                          "opaque label. No processor generation is inferred from it.",
        "observed": {
            "cpu_part_ids": parts,
            "multiple_identities": len(parts) > 1,
            "cpu_part_counts_total": dict(sorted(totals.items())),
            "cpu_part_counts_by_arm_class": {a: {c: dict(sorted(v.items())) for c, v in sorted(cs.items())}
                                             for a, cs in sorted(by_arm_class.items())},
            "cpu_part_counts_by_round": {k: dict(sorted(v.items())) for k, v in sorted(by_round.items())},
            "ggml_cpu_variants_loaded_by_part": {p: dict(v) for p, v in sorted(variants.items())},
            "runtime_version_arn": cond.get("runtime_version_arn"),
            "runtime_versions_in_init_start": dict(runtimes),
            "configured": {"memory_mb": sorted({r["report"].get("Memory Size") for r in ok if r.get("report")}),
                           "region": cond.get("region"), "threads": 6},
            "log_stream_linkage": {"rows": len(rows), "rows_linked_to_stream": linked,
                                   "rows_unlinked": len(rows) - linked,
                                   "unlinked_reason": "the 4 KB log tail lost its START line",
                                   "streams": len(stream_parts),
                                   "streams_with_more_than_one_part": sum(len(p) > 1 for p in stream_parts.values()),
                                   "streams_by_part": dict(sorted(streams_by_part.items()))},
            "probe_environments": {"function": "lkai-p3-probe (separate function, same runtime ARN, arch, memory; "
                                               "not this run)", "by_file": probes,
                                   "part_ids_seen": sorted({v["part"] for v in probes.values()})},
            "report_duration_p50_by_arm_class_part": dict(rep_by_part),
            "rep_warm_engine_p50_by_request_part": engine_warm,
            "lkaisamp_top_self_by_part": kernels,
        },
        "unknown": HOST_UNKNOWN,
        "method": "Overhead verdicts use part 0xd40 rows only (paired by round against rep and rep2 on that part). "
                  "Other part IDs are counted, not compared: per (arm, class, round) they are too sparse for a "
                  "paired cell. Native hot-spot tables are split by part ID because each part ID loads a different "
                  "libggml-cpu variant. Phase hot spots (P1 timers) are pooled over parts unless split above.",
        "assessment": "More than one CPU part ID served this run; they are recorded, not ranked or named. Every "
                      "overhead verdict is within 0xd40, so a part mix cannot move it. Absolute figures differ by part "
                      "ID (report_duration_p50_by_arm_class_part, rep_warm_engine_p50_by_request_part); any pooled "
                      "figure depends on that run's part mix." if len(parts) > 1 else
                      "One CPU part ID served this run.",
    }


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
        "method": "per round: median(arm) - median(rep) of the REPORT field, CPU part 0xd40 rows only, no errors or "
                  f"class mismatches, each median from >= {MIN_CELL_N} samples (else the round is skipped and listed); "
                  "then mean/min/max over the rounds. Warm also per (round, request) as a percentage. Null: "
                  f"{NULL_ARM}, a second function byte-identical to rep, run in the same rounds, so {NULL_ARM} - rep "
                  "is what zero cost looks like. arm - rep and "
                  f"{NULL_ARM} - rep share each round's rep median, so they are paired: each arm (xray and none "
                  f"included) is tested on the per-round paired difference arm - {NULL_ARM} (warm by request: "
                  "per-round mean %) by an exact two-sided sign-flip test, alpha 0.05, and must also exceed the "
                  "null's largest |per-round value| to be called measurable. With 5 paired rounds the smallest "
                  "attainable p is 2/32 = 0.0625, so no cell can be called measurable at alpha 0.05 "
                  "(underpowered); consistent_direction is a descriptive flag, not a test",
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
    res["host_variability"] = host_variability(rows, cond, a.run, os.path.dirname(a.probe) if a.probe else None,
                                               res["profiles"]["lambda"])
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
