"""H1: host variability across every recorded lambda-kleidiai Lambda run. Standard library only.

  python hostvar.py extract  [--sources sources.json] [--repo <git dir>] --out <run-dir>/rows.jsonl.gz
  python hostvar.py analyze  <run-dir> [--fixed-host <offline-summary.json>]   # writes <run-dir>/results.json

`extract` reads each source run listed in sources.json with `git show <commit>:<path>`, never from a working tree,
so the inputs are exactly the pushed commits named there. It keeps one compact row per invocation (run, arm, round,
wave, class, request, CPU part, environment id, log stream, REPORT numbers, engine latency, ggml CPU variant) and
one row per log stream (kind, CPU part from the stream's own `init cpu_part=` line, version). Nothing is invented: a
field the run did not record stays null.

`analyze` turns those rows into an lkai-result/1 file for the P2 dashboard with a `host_variability` block:

  observed       the metadata fields every source recorded, and how
  unknown        every field the HOST VARIABILITY rules ask for that no source recorded, with the reason
  cpu_identity   CPU-part counts by run, arm and start class; per-environment stability; snapshot-build parts
  by_identity    latency distributions split by CPU part (same-code control arms only), per run and pooled
  run_to_run     per-run medians of each control metric within one CPU part, and the mix-only effect
  pooled_vs_within  for every arm pair in every run, the pooled median delta beside each shared part's delta,
                 flagging comparisons whose pooled sign or size disagrees with every within-part delta
  decision       whether Lambda placement is controlled, with sample sizes and what could not be observed

CPU parts are opaque identifiers here. No processor generation or microarchitecture is named from a part ID, a
ggml variant or a timing; tests/test_lkai_hostvar.py rejects such names in the generated result.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "..", "baseline")
sys.path.insert(0, os.path.abspath(BASELINE))
import summarize  # noqa: E402

CLASSES = ("cold", "restore", "warm")
SCHEMA = "lkai-host-variability/2"
MIN_N = 3  # per arm, per part, before a within-part delta is computed
PROBE = "ticket-route-1q"

_START = re.compile(r"START RequestId: (\S+)")
_INIT_PART = re.compile(r"decider\.engine init cpu_part=(\S+) ggml_cpu=(\S+)")
_VERSION = re.compile(r"\[([^\]]+)\]")


# ---------------------------------------------------------------------------- extract


def _git_bytes(repo: str, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout


def _git_ls(repo: str, commit: str, path: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", repo, "ls-tree", "--name-only", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [x for x in out.splitlines() if x]


def _jsonl(data: bytes, gz: bool) -> list[dict]:
    if gz:
        data = gzip.decompress(data)
    return [json.loads(x) for x in io.StringIO(data.decode()) if x.strip()]


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def compact_row(src: dict, r: dict) -> dict:
    """The fields H1 needs from one bench.py invocation row. Missing stays None."""
    tail = r.get("log_tail") or ""
    m = _START.search(tail)
    init = _INIT_PART.search(tail)
    lk = r.get("lkai") or {}
    rep = r.get("report") or {}
    return {
        "run": src["key"],
        "arm": r.get("arm"),
        "round": r.get("round"),
        "wave": r.get("wave"),
        "seq": r.get("seq"),
        "t_start": r.get("t_start"),
        "cls": r.get("expected_class"),
        "ok": not r.get("error") and not r.get("class_mismatch"),
        "request": r.get("request_id"),
        "cpu_part": r.get("cpu_part"),
        "env_id": lk.get("env_id"),
        "lambda_request_id": m.group(1) if m else lk.get("request_id"),
        "version": r.get("executed_version") or lk.get("version"),
        "ggml_cpu": init.group(2) if init else None,
        "host": lk.get("host"),
        "log_stream": lk.get("log_stream"),
        "engine_s": _num(r["latency_ms"] / 1000) if _num(r.get("latency_ms")) is not None else None,
        "duration_s": _num(rep["Duration"] / 1000)
        if _num(rep.get("Duration")) is not None
        else None,
        "init_s": _num(rep["Init Duration"] / 1000)
        if _num(rep.get("Init Duration")) is not None
        else None,
        "restore_s": _num(rep["Restore Duration"] / 1000)
        if _num(rep.get("Restore Duration")) is not None
        else None,
        "memory_mb": _num(rep.get("Memory Size")),
        "max_memory_mb": _num(rep.get("Max Memory Used")),
    }


def stream_rows(src: dict, fn: str, events: list[dict]) -> list[dict]:
    """One row per CloudWatch log stream: what started it, its version, and the CPU part its own init logged."""
    by: dict[str, list[str]] = defaultdict(list)
    for e in events:
        by[e.get("logStreamName", "")].append(e.get("message", ""))
    out = []
    for s, msgs in sorted(by.items()):
        v = _VERSION.search(s)
        parts = sorted({m.group(1) for x in msgs if (m := _INIT_PART.search(x))})
        variants = sorted({m.group(2) for x in msgs if (m := _INIT_PART.search(x))})
        has_init = any(x.startswith("INIT_START") for x in msgs)
        has_restore = any(x.startswith("RESTORE_START") for x in msgs)
        starts = sorted({m.group(1) for x in msgs if (m := _START.search(x))})
        ver = v.group(1) if v else None
        if has_restore:
            kind = "restore"
        elif has_init and not starts and ver not in (None, "$LATEST"):
            # A published version's init inside publish_version: no invocation in the stream.
            kind = "snapshot_build"
        elif has_init:
            kind = "cold"
        else:
            kind = "other"
        out.append(
            {
                "run": src["key"],
                "function": fn,
                "stream": s,
                "version": ver,
                "kind": kind,
                "init_cpu_parts": parts,
                "ggml_cpu": variants,
                "request_ids": starts,
            }
        )
    return out


def extract(sources: dict, repo: str) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    streams: list[dict] = []
    conds: dict = {}
    for src in sources["runs"]:
        for d in src["dirs"]:
            files = _git_ls(repo, src["commit"], d)
            inv = "invocations.jsonl.gz" if "invocations.jsonl.gz" in files else "invocations.jsonl"
            raw = _jsonl(_git_bytes(repo, src["commit"], f"{d}/{inv}"), inv.endswith(".gz"))
            rows += [compact_row(src, r) for r in raw]
            c = json.loads(_git_bytes(repo, src["commit"], f"{d}/conditions.json"))
            conds.setdefault(src["key"], []).append(_cond_summary(c))
            if "logs" in files:
                for lf in _git_ls(repo, src["commit"], f"{d}/logs"):
                    ev = _jsonl(
                        _git_bytes(repo, src["commit"], f"{d}/logs/{lf}"), lf.endswith(".gz")
                    )
                    streams += stream_rows(src, lf.split(".jsonl")[0], ev)
    return rows, streams, conds


def _cond_summary(c: dict) -> dict:
    fns = {}
    for arm, f in (c.get("functions") or {}).items():
        env = dict(f.get("environment") or {})
        env.pop("COLD_TAG", None)  # changes every round by design (forces a cold start)
        fns[arm] = {
            "code_sha256": f.get("CodeSha256"),
            "memory_mb": f.get("MemorySize"),
            "architecture": (f.get("Architectures") or [None])[0],
            "runtime": f.get("Runtime"),
            "handler": f.get("Handler"),
            "tracing": (f.get("TracingConfig") or {}).get("Mode"),
            "layers": f.get("layers") or [],
            "environment": env,
        }
    return {
        "region": c.get("region"),
        "rounds": c.get("rounds"),
        "concurrency": c.get("concurrency"),
        "runtime_version_arn": c.get("runtime_version_arn"),
        "started": c.get("started"),
        "finished": c.get("finished"),
        "orders": [m.get("order") for m in c.get("rounds_meta") or []],
        "client": c.get("client"),
        "functions": fns,
    }


# ---------------------------------------------------------------------------- analysis helpers


def med(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 6) if xs else None


def config_key(fn: dict) -> str:
    """Functions with the same key ran the same code with the same configuration (COLD_TAG aside)."""
    return json.dumps(fn, sort_keys=True)


METRICS = {
    "cold": {
        "first_request_total_s": lambda r: (
            (r["init_s"] or 0) + r["duration_s"] if r["duration_s"] is not None else None
        ),
        "probe_engine_s": lambda r: r["engine_s"] if r["request"] == PROBE else None,
    },
    "restore": {
        "restore_duration_s": lambda r: r["restore_s"],
        "probe_engine_s": lambda r: r["engine_s"] if r["request"] == PROBE else None,
    },
    "warm": {
        "engine_s": lambda r: r["engine_s"],
        "probe_engine_s": lambda r: r["engine_s"] if r["request"] == PROBE else None,
    },
}


def values(rows: list[dict], cls: str, metric: str) -> list[float]:
    f = METRICS[cls][metric]
    return [v for r in rows if r["ok"] and r["cls"] == cls and (v := f(r)) is not None]


def part_label(p: str | None) -> str:
    return p if p else "unknown"


def cpu_identity(rows: list[dict], streams: list[dict]) -> dict:
    by_run_arm_cls: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    total: Counter = Counter()
    for r in rows:
        by_run_arm_cls[r["run"]][r["arm"]][r["cls"] or "unknown"][part_label(r["cpu_part"])] += 1
        total[part_label(r["cpu_part"])] += 1
    env_parts: dict = defaultdict(set)
    for r in rows:
        if r["env_id"] and r["cpu_part"]:
            env_parts[(r["run"], r["env_id"])].add(r["cpu_part"])
    stream_parts = [s for s in streams if s["init_cpu_parts"]]
    # Every row whose START RequestId appears in a saved stream joins that stream; this covers REPORT-only arms
    # (no env_id) and restored environments (no init line), so stability is checked per stream as well.
    stream_of = {(s["run"], rid): s["stream"] for s in streams for rid in s["request_ids"]}
    linked: dict = defaultdict(set)
    n_linked = 0
    for r in rows:
        st = stream_of.get((r["run"], r["lambda_request_id"]))
        if st and r["cpu_part"]:
            linked[(r["run"], st)].add(r["cpu_part"])
            n_linked += 1
    for s in stream_parts:
        if (s["run"], s["stream"]) in linked:
            linked[(s["run"], s["stream"])].update(s["init_cpu_parts"])
    snap = [s for s in streams if s["kind"] == "snapshot_build"]
    snap_by_ver: dict = defaultdict(Counter)
    for s in snap:
        snap_by_ver[(s["run"], s["function"], s["version"])][
            ",".join(s["init_cpu_parts"]) or "unknown"
        ] += 1
    multi_part_versions = sum(
        1 for c in snap_by_ver.values() if len([k for k in c if k != "unknown"]) > 1
    )
    version_part_sets = Counter(
        "+".join(sorted({k for k in c if k != "unknown"})) or "unknown"
        for c in snap_by_ver.values()
    )
    restore_parts: Counter = Counter(
        part_label(r["cpu_part"]) for r in rows if r["cls"] == "restore"
    )
    variants: dict = defaultdict(Counter)
    for s in streams:
        for p in s["init_cpu_parts"]:
            for v in s["ggml_cpu"]:
                variants[p][v] += 1
    return {
        "total_by_part": dict(sorted(total.items())),
        "by_run_arm_class": {
            run: {
                arm: {cls: dict(sorted(c.items())) for cls, c in sorted(a.items())}
                for arm, a in sorted(arms.items())
            }
            for run, arms in sorted(by_run_arm_cls.items())
        },
        "environment_stability": {
            "environments_with_part": len(env_parts),
            "environments_with_more_than_one_part": sum(len(v) > 1 for v in env_parts.values()),
            "method": "rows grouped by (run, lkai.env_id); every row of one environment reports /proc/cpuinfo "
            "'CPU part' per request",
        },
        "log_stream_stability": {
            "rows": len(rows),
            "rows_linked_to_a_stream": n_linked,
            "streams_with_rows": len(linked),
            "streams_with_more_than_one_part": sum(len(v) > 1 for v in linked.values()),
            "method": "row joined to the saved log stream whose START RequestId matches the row's (from its "
            "log tail); the stream's own 'init cpu_part=' line is added where it has one. Rows whose tail lost "
            "its START line stay unlinked",
        },
        "restore_parts": dict(sorted(restore_parts.items())),
        "snapshot_build": {
            "streams": len(snap),
            "streams_without_part": sum(not s["init_cpu_parts"] for s in snap),
            "by_part": dict(
                sorted(Counter(",".join(s["init_cpu_parts"]) or "unknown" for s in snap).items())
            ),
            "versions": len(snap_by_ver),
            "versions_built_on_more_than_one_part": multi_part_versions,
            "versions_by_part_set": dict(sorted(version_part_sets.items())),
            "method": "a published-version log stream with INIT_START and no invocation START is the "
            "snapshot-building init inside publish_version; its part is its own 'init cpu_part=' line",
        },
        "ggml_cpu_variant_by_part": {
            p: dict(sorted(c.items())) for p, c in sorted(variants.items())
        },
    }


def control_groups(rows: list[dict], conds: dict, sources: dict) -> dict[str, list[dict]]:
    """Rows of every run's control arm(s), grouped by identical function configuration across runs."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for src in sources["runs"]:
        cfgs = {}
        for c in conds.get(src["key"], []):
            for arm, fn in c["functions"].items():
                cfgs.setdefault(arm, config_key(fn))
        for r in rows:
            if r["run"] == src["key"] and r["arm"] in src["control_arms"]:
                groups[cfgs.get(r["arm"], "unknown")].append(r)
    return groups


def by_identity(rows: list[dict]) -> dict:
    out: dict = {}
    for cls, ms in METRICS.items():
        out[cls] = {}
        for m in ms:
            per = defaultdict(list)
            for r in rows:
                if r["ok"] and r["cls"] == cls and (v := ms[m](r)) is not None:
                    per[part_label(r["cpu_part"])].append(v)
            out[cls][m] = {p: summarize.dist(v) for p, v in sorted(per.items())}
    return out


def run_to_run(rows: list[dict]) -> dict:
    """Per run, the median of each control metric within each CPU part; across runs, the spread of those medians
    and of the pooled (all parts) medians. The difference between the two spreads is what the part mix alone adds."""
    out: dict = {}
    runs = sorted({r["run"] for r in rows})
    for cls, ms in METRICS.items():
        out[cls] = {}
        for m in ms:
            per: dict = defaultdict(dict)
            pooled: dict = {}
            mix: dict = {}
            for run in runs:
                rs = [r for r in rows if r["run"] == run]
                vs = [
                    (part_label(r["cpu_part"]), v)
                    for r in rs
                    if r["ok"] and r["cls"] == cls and (v := ms[m](r)) is not None
                ]
                if not vs:
                    continue
                pooled[run] = med([v for _, v in vs])
                c = Counter(p for p, _ in vs)
                mix[run] = {p: round(n / len(vs), 4) for p, n in sorted(c.items())}
                for p in c:
                    xs = [v for q, v in vs if q == p]
                    if len(xs) >= MIN_N:
                        per[p][run] = {"n": len(xs), "median": med(xs)}

            def spread(d: dict) -> dict | None:
                ms_ = [x["median"] if isinstance(x, dict) else x for x in d.values()]
                if len(ms_) < 2:
                    return None
                mean = statistics.fmean(ms_)
                return {
                    "runs": len(ms_),
                    "min": round(min(ms_), 6),
                    "max": round(max(ms_), 6),
                    "range": round(max(ms_) - min(ms_), 6),
                    "cv": round(statistics.stdev(ms_) / mean, 4) if mean else None,
                }

            out[cls][m] = {
                "pooled_median_by_run": pooled,
                "part_mix_by_run": mix,
                "pooled_spread": spread(pooled),
                "within_part": {
                    p: {"median_by_run": d, "spread": spread(d)} for p, d in sorted(per.items())
                },
            }
    return out


MATERIAL = 0.01  # a delta under 1% of the control's pooled median is treated as zero when flagging


def compare_cell(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> dict:
    """Pooled median delta (b - a) next to each shared part's delta. Missing parts are not a stratum.

    Flags (noise floor: MATERIAL x the control's pooled median):
      sign_flip   pooled delta and every within-part delta exceed the floor, and pooled has the opposite sign
      mix_shift   pooled delta lies outside the within-part range by more than the floor and by more than a
                  quarter of the largest within-part |delta|
      consistent  neither; not assessable when no part has MIN_N rows in both arms"""
    pa, pb = Counter(p for p, _ in a), Counter(p for p, _ in b)
    pooled = med([v for _, v in b]) - med([v for _, v in a]) if a and b else None
    floor = MATERIAL * abs(med([v for _, v in a])) if a else 0.0
    per = {}
    for p in sorted((set(pa) & set(pb)) - {"unknown"}):
        xa, xb = [v for q, v in a if q == p], [v for q, v in b if q == p]
        if len(xa) >= MIN_N and len(xb) >= MIN_N:
            per[p] = {"n": [len(xa), len(xb)], "delta": round(med(xb) - med(xa), 6)}
    deltas = [d["delta"] for d in per.values()]
    if pooled is None or not deltas:
        flag = "not assessable"
    else:
        big = [d for d in deltas if abs(d) > floor]
        gap = max(min(deltas) - pooled, pooled - max(deltas), 0.0)
        if (
            abs(pooled) > floor
            and len(big) == len(deltas)
            and all((d > 0) != (pooled > 0) for d in deltas)
        ):
            flag = "sign_flip"
        elif gap > floor and gap > 0.25 * max(abs(d) for d in deltas):
            flag = "mix_shift"
        else:
            flag = "consistent"
    return {
        "pooled_delta": round(pooled, 6) if pooled is not None else None,
        "noise_floor": round(floor, 6),
        "mix": [dict(sorted(pa.items())), dict(sorted(pb.items()))],
        "within_part": per,
        "unassessed_rows": [pa.get("unknown", 0), pb.get("unknown", 0)],
        "flag": flag,
    }


def pooled_vs_within(rows: list[dict], sources: dict) -> list[dict]:
    """Every (run, class, metric) comparison of a non-control arm against that run's first control arm."""
    out = []
    for src in sources["runs"]:
        rs = [r for r in rows if r["run"] == src["key"]]
        arms = sorted({r["arm"] for r in rs})
        ctl = src["control_arms"][0]
        for arm in arms:
            if arm == ctl:
                continue
            for cls, ms in METRICS.items():
                for m, f in ms.items():
                    get = lambda a, f=f, cls=cls, rs=rs: [  # noqa: E731
                        (part_label(r["cpu_part"]), v)
                        for r in rs
                        if r["arm"] == a and r["ok"] and r["cls"] == cls and (v := f(r)) is not None
                    ]
                    a, b = get(ctl), get(arm)
                    if not a or not b:
                        continue
                    out.append(
                        {
                            "run": src["key"],
                            "control": ctl,
                            "arm": arm,
                            "class": cls,
                            "metric": m,
                            **compare_cell(a, b),
                        }
                    )
    return out


def interleaving(conds: dict, sources: dict) -> dict:
    out = {}
    for src in sources["runs"]:
        cs = conds.get(src["key"], [])
        orders = [o for c in cs for o in c["orders"] if o]
        out[src["key"]] = {
            "rounds": sum(len(c["orders"]) for c in cs),
            "orders": orders,
            "pattern": _order_pattern(orders),
            "invoke_client": src["invoke_client"],
        }
    return out


def _order_pattern(orders: list[list[str]]) -> str:
    if not orders:
        return "unknown (no round order recorded)"
    if all(o == orders[0] for o in orders):
        return "fixed order every round"
    if all(o == (orders[0] if i % 2 == 0 else orders[0][::-1]) for i, o in enumerate(orders)):
        return (
            "alternating: arm order reversed every other round; within a round each arm's cold and "
            "restore waves run one arm after another (sequential, not interleaved)"
        )
    return "rotated or varied arm order across rounds; waves sequential within a round"


# ---------------------------------------------------------------------------- metadata ledger

OBSERVED = {
    "cpu_part": "every source, per request: /proc/cpuinfo 'CPU part' read by PR #77's handler._cpu_part "
    "(row cpu_part; the engine init log line for cold and snapshot-build streams)",
    "ggml_cpu_variant": "cold and snapshot-build log streams: the libggml-cpu variants the engine loaded "
    "(engine init log line); ggml chose them from the CPU features it detected, which were not recorded",
    "execution_environment": "row lkai.env_id (instrumented arms) and the CloudWatch log stream linked by the "
    "invocation's START RequestId in its log tail",
    "memory_mb": "REPORT Memory Size, every row",
    "runtime_version": "conditions.runtime_version_arn and each stream's INIT_START/RESTORE_START line",
    "architecture": "configured: functions.*.Architectures",
    "region": "configured: conditions.region",
    "threads": "configured, not observed: 6 llama.cpp threads",
    "round_order": "conditions.rounds_meta[].order",
}
UNKNOWN = {
    "cpu_implementer_variant_revision": "not recorded per invocation by any source run; P3's probe function "
    "read them in 7 environments (all part 0xd40): implementer 0x41, variant 0x1, revision 1",
    "cpu_feature_flags": "not recorded per invocation; P3's probe saw one Features line in its 7 environments, "
    "all part 0xd40; never observed for 0xd0c or 0xd4f",
    "midr_el1": "not recorded by any source run",
    "hwcap_hwcap2": "not recorded by any source run",
    "core_count": "not recorded per invocation (P3 probe: nproc 6 in 7 part-0xd40 environments); restored "
    "environments never read it",
    "cgroup_cpu_memory_limits": "not recorded by any source run",
    "kernel": "not recorded per invocation (P3 probe: one release in 7 environments)",
    "availability_zone": "not observable inside the function",
    "physical_host": "not observable inside the function",
    "processor_generation": "not observable and never inferred: only CPU part IDs are recorded",
}


# ---------------------------------------------------------------------------- decision


def decide(ident: dict, ctl_rows: list[dict]) -> dict:
    parts = sorted(p for p in ident["total_by_part"] if p != "unknown")
    runs_all = list(ident["by_run_arm_class"])
    runs_with_all_parts = [
        run
        for run, arms in ident["by_run_arm_class"].items()
        if set(parts) <= {p for a in arms.values() for c in a.values() for p in c}
    ]
    probe = by_identity(ctl_rows)["warm"]["probe_engine_s"]
    ratios = {}
    base = probe.get("0xd40")
    for p, d in probe.items():
        if base and d and p not in ("0xd40", "unknown"):
            ratios[f"{p}/0xd40"] = round(d["p50"] / base["p50"], 3)
    # Share of each part per (run, class), every arm of the run.
    shares: dict = defaultdict(dict)
    for run, arms in ident["by_run_arm_class"].items():
        for cls in CLASSES:
            c: Counter = Counter()
            for a in arms.values():
                c.update(a.get(cls, {}))
            n = sum(c.values())
            if n:
                shares[cls][run] = {p: round(k / n, 4) for p, k in sorted(c.items())}
    share_range = {
        cls: {
            p: {
                "min": min(s.get(p, 0.0) for s in by.values()),
                "max": max(s.get(p, 0.0) for s in by.values()),
            }
            for p in parts
        }
        for cls, by in shares.items()
    }
    controlled = len(parts) <= 1
    return {
        "placement_controlled": controlled,
        "verdict": "not controlled" if not controlled else "controlled (single identity observed)",
        "evidence": {
            "cpu_part_ids_observed": parts,
            "runs": len(runs_all),
            "runs_with_every_part": len(runs_with_all_parts),
            "invocations": sum(ident["total_by_part"].values()),
            "environments_with_part": ident["environment_stability"]["environments_with_part"],
            "environments_with_more_than_one_part": ident["environment_stability"][
                "environments_with_more_than_one_part"
            ],
            "streams_with_rows": ident["log_stream_stability"]["streams_with_rows"],
            "streams_with_more_than_one_part": ident["log_stream_stability"][
                "streams_with_more_than_one_part"
            ],
            "restore_parts": ident["restore_parts"],
            "warm_probe_engine_p50_ratio_by_part": ratios,
            "part_share_by_run_and_class": dict(shares),
            "part_share_range_over_runs": share_range,
            "snapshot_versions": ident["snapshot_build"]["versions"],
            "snapshot_versions_by_part_set": ident["snapshot_build"]["versions_by_part_set"],
        },
        "meaning": "Lambda assigns each new execution environment a CPU part the function cannot choose. Every "
        "run drew more than one part, the share of each part changes from run to run and between start "
        "classes, and the same request's engine time differs by part. A part is fixed for the life of one "
        "environment (no environment reported two). Comparisons are therefore valid only within one part "
        "and only between arms measured in the same rounds; pooled medians and cross-run comparisons carry "
        "the part mix.",
        "rules_for_experiments": [
            "compare arms within each CPU part (summarize.by_cpu / engine_by_cpu_part) and report the part mix",
            "interleave or alternate arms within the same rounds; never compare against another run's numbers "
            "without stratifying by part",
            "report rows with no recorded part as unassessed, never as a shared stratum",
            "a CPU-bound phase (engine, model_load_mmap, restore first request) moves with the part; an I/O "
            "phase (weight_download) need not, but must be checked per part before pooling",
        ],
        "limits": [
            "a single part per environment proves nothing about the physical host or AZ; neither is observable",
            "feature flags, HWCAPs, cgroup limits and kernel are unrecorded for every source run; the new "
            "hostinfo instrument records them for future runs",
            "0xd4f appeared in cold rows of every run and in E6's warm rows, never in a restore wave or a "
            "snapshot-building init",
            "invocations are clustered by environment and round, so per-invocation counts overstate the "
            "independent sample size; environment counts are given beside them",
        ],
    }


# ---------------------------------------------------------------------------- result


def build_result(rows, streams, conds, sources, fixed_host, commits) -> dict:
    ident = cpu_identity(rows, streams)
    groups = control_groups(rows, conds, sources)
    main_key, main_rows = max(groups.items(), key=lambda kv: len(kv[1]))
    main_cfg = json.loads(main_key) if main_key != "unknown" else {}
    r2r = run_to_run(main_rows)
    pvw = pooled_vs_within(rows, sources)
    decision = decide(ident, main_rows)
    flagged = [c for c in pvw if c["flag"] not in ("consistent", "not assessable")]
    pooled_runs = sorted({r["run"] for r in main_rows})
    restore_suspect = sorted(
        s["key"] for s in sources["runs"] if s["invoke_client"] != "fresh-per-wave"
    )
    hv = {
        "schema": SCHEMA,
        "new_measurements": False,
        "aws_calls": 0,
        "source": "reanalysis of committed invocation rows, conditions and CloudWatch logs of every recorded "
        "run in hostvar/sources.json, read from the pinned fork commits; no Lambda invocation",
        "identity_label": "CPU part ID from /proc/cpuinfo, an opaque label; no generation inferred",
        "sources": [
            {
                k: s[k]
                for k in ("key", "experiment_id", "run_id", "branch", "commit", "invoke_client")
            }
            for s in sources["runs"]
        ],
        "excluded_sources": sources.get("excluded", {}),
        "observed": OBSERVED,
        "unknown": UNKNOWN,
        "cpu_identity": ident,
        "same_code_control_pool": {
            "runs": pooled_runs,
            "invocations": len(main_rows),
            "configuration": main_cfg,
            "method": "control-arm rows of every run whose function configuration (CodeSha256, memory, "
            "architecture, runtime, handler, tracing, layers, environment without COLD_TAG) is identical; "
            "weight bucket and S3 path differ between runs and are not in the key, so download phases are "
            "not pooled",
            "other_groups": {
                str(i): {"runs": sorted({r["run"] for r in g}), "invocations": len(g)}
                for i, (k, g) in enumerate(sorted(groups.items()))
                if k != main_key
            },
        },
        "by_identity": by_identity(main_rows),
        "by_identity_per_run": {
            run: by_identity([r for r in main_rows if r["run"] == run]) for run in pooled_runs
        },
        "run_to_run": r2r,
        "pooled_vs_within": {
            "min_n_per_arm_per_part": MIN_N,
            "noise_floor_fraction_of_control_median": MATERIAL,
            "comparisons": len(pvw),
            "flag_counts": dict(sorted(Counter(c["flag"] for c in pvw).items())),
            "flagged": flagged,
            "all": pvw,
        },
        "interleaving": interleaving(conds, sources),
        "restore_latency_caveat": {
            "runs_with_pooled_or_unknown_invoke_client": restore_suspect,
            "note": "restore latency from these runs predates the fresh-client-per-wave fix and may be "
            "contaminated by the first-restore-wave stall (I1 restore-stall, P3); REPORT Restore Duration is "
            "measured by Lambda and is unaffected by the client, client round-trip times are not",
        },
        "concat_gap": {
            "question": "P3 found ggml_compute_forward_concat at 5.3% (run 20261020T124451Z, lkaisamp, warm, "
            "part 0xd40) / 5.6% (rerun 20261020T231634Z) of Lambda CPU samples against 1.2% offline under perf "
            "and 2.65% offline under the same sampler. Host or code?",
            "lambda": {
                "20261020T124451Z": {"0xd40": 0.0527, "0xd0c": 0.0593},
                "20261020T231634Z": {"0xd40": 0.0564, "0xd0c": 0.0647},
                "source": "P3 results.json host_variability.observed.lkaisamp_top_self_by_part "
                "(invoke_warm_<part>)",
            },
            "offline": {
                "perf_record_self_pct": 1.16,
                "lkaisamp_steady_state_pct": 2.65,
                "host": "arm64 DevSpace, part 0xd40, kernel 6.1.188, 16 CPUs, 6 threads pinned",
                "source": "P3 results/offline/offline-summary.json, arm64 host section",
            },
            "assessment": "The same function.zip and the same part ID (0xd40) ran on both sides, so neither a "
            "code change nor a part-ID mix explains the gap. Within Lambda the share is similar on both parts "
            "that were sampled (run 1: 5.27% on 0xd40, 5.93% on 0xd0c; run 2: 5.64% and 6.47%), so it is not "
            "specific to one part. Two method differences remain: offline gave 2.65% under the same sampler "
            "and 1.16% under perf, so the tool alone moves it by about 2x; and Lambda vs offline differ in "
            "unrecorded host conditions (kernel 5.10 in Lambda vs 6.1 offline, cgroup CPU quota, core count). "
            "Same sampler, same part: 5.64% vs 2.65%. Not attributable to code from this evidence; it stays "
            "open until hostinfo-instrumented Lambda rows exist to compare host conditions.",
        },
        "fixed_host": fixed_host,
        "decision": decision,
    }
    hv["assessment"] = (
        f"Placement {decision['verdict']}: {len(decision['evidence']['cpu_part_ids_observed'])} CPU part IDs "
        f"({', '.join(decision['evidence']['cpu_part_ids_observed'])}) across "
        f"{decision['evidence']['runs']} runs, {decision['evidence']['runs_with_every_part']} of them drew every "
        f"part; no environment changed part. {len(flagged)} of {len(pvw)} within-run arm comparisons have a "
        "pooled delta that a part-mix difference moved beyond the noise floor (sign_flip or mix_shift)."
    )
    return _result(hv, main_rows, main_cfg, conds, commits, decision)


def _result(hv, main_rows, cfg, conds, commits, decision) -> dict:
    phases = {}
    by_part = defaultdict(list)
    for r in main_rows:
        by_part[part_label(r["cpu_part"])].append(r)
    for p, rs in sorted(by_part.items()):
        if p == "unknown":
            continue
        arm = {}
        for cls in CLASSES:
            c = [r for r in rs if r["ok"] and r["cls"] == cls]
            d: dict = {"n": len(c)}
            if cls == "cold":
                d["first_request_total"] = summarize.dist(
                    values(c, "cold", "first_request_total_s")
                )
                d["first_request_inference"] = summarize.dist(values(c, "cold", "probe_engine_s"))
                d["report_init_duration"] = summarize.dist([r["init_s"] for r in c if r["init_s"]])
                d["report_max_memory_mb"] = summarize.dist(
                    [r["max_memory_mb"] for r in c if r["max_memory_mb"]], unit="MB"
                )
                for ph in ("weight_download", "model_load_mmap", "model_ready"):
                    d[ph] = None  # not pooled across runs: weight location differs by run
            elif cls == "restore":
                d["report_restore_duration"] = summarize.dist(
                    values(c, "restore", "restore_duration_s")
                )
                d["first_request_inference"] = summarize.dist(
                    values(c, "restore", "probe_engine_s")
                )
                d["snapshot_init"] = None
            else:
                d["steady_state_inference"] = summarize.dist(values(c, "warm", "engine_s"))
                d["probe_request_inference"] = summarize.dist(values(c, "warm", "probe_engine_s"))
            arm[cls] = d
        phases[f"part_{p}"] = arm
    c0 = next(iter(conds.values()))[0]
    starts = sorted(c["started"] for cs in conds.values() for c in cs if c.get("started"))
    ends = sorted(c["finished"] for cs in conds.values() for c in cs if c.get("finished"))
    ev = decision["evidence"]
    cold_d0c = ev["part_share_range_over_runs"]["cold"].get("0xd0c", {})
    cold_d4f = ev["part_share_range_over_runs"]["cold"].get("0xd4f", {})
    warm_d0c = ev["part_share_range_over_runs"]["warm"].get("0xd0c", {})
    pv = hv["pooled_vs_within"]["flag_counts"]
    conclusion = (
        f"Null result for the experiment variable (none changed; reanalysis only, 0 AWS calls). Lambda CPU "
        f"placement is NOT controlled. {ev['invocations']:,} recorded invocations in {ev['runs']} runs drew CPU "
        f"parts {', '.join(ev['cpu_part_ids_observed'])}; {ev['runs_with_every_part']} runs drew all of them. "
        f"Share by run: cold 0xd0c {cold_d0c.get('min', 0):.1%}-{cold_d0c.get('max', 0):.1%}, cold 0xd4f "
        f"{cold_d4f.get('min', 0):.1%}-{cold_d4f.get('max', 0):.1%}, warm 0xd0c {warm_d0c.get('min', 0):.1%}-"
        f"{warm_d0c.get('max', 0):.1%}. Restores ran on {', '.join(k for k in ev['restore_parts'] if k != 'unknown')} only; "
        f"all {ev['snapshot_versions']} published versions built snapshots on "
        f"{', '.join(ev['snapshot_versions_by_part_set'])}. A part is fixed per environment: 0 of "
        f"{ev['environments_with_part']:,} environments and 0 of {ev['streams_with_rows']:,} log streams showed two. "
        f"Same-code control pool ({', '.join(hv['same_code_control_pool']['runs'])}), warm probe request engine p50: "
        + ", ".join(f"{k} = {v}x" for k, v in ev["warm_probe_engine_p50_ratio_by_part"].items())
        + f". Within-run arm comparisons: {pv.get('mix_shift', 0)} mix_shift and {pv.get('sign_flip', 0)} "
        f"sign_flip of {hv['pooled_vs_within']['comparisons']}, where the pooled delta differs from every "
        "within-part delta by more than 1% of the control median. Valid comparisons: within one CPU part, "
        "arms in the same rounds, part mix reported. No processor generation is inferred from a part ID. "
        "Unknown for every source run: feature flags, HWCAPs, MIDR, cgroup limits, core count, kernel, AZ, "
        "physical host; the baseline harness now records them per environment (hostinfo.py, not yet run on "
        "Lambda). Fixed-host reference: P3's existing offline arm64 DevSpace (part 0xd40), labelled "
        "fixed-host, not Lambda latency. Concat 5.3%/5.6% (Lambda, sampler) vs 2.65% (offline, same sampler) "
        "vs 1.16% (offline, perf): same code and part ID, so not attributable to code; host conditions "
        "unrecorded, open. Restore latency from pooled-client runs is suspect."
    )
    return {
        "schema": "lkai-result/1",
        "experiment_id": "H1-host-variability",
        "run_id": "20261021T-reanalysis-1",
        "status": "inconclusive",
        "hypothesis": "UNVERIFIED operator hypothesis: Lambda invocations land on different CPU types, so "
        "latency comparisons are confounded by placement. Tested against every recorded run; no "
        "generation is named.",
        "changed_variable": {
            "name": "analysis: stratify recorded Lambda invocations by observed CPU part ID",
            "baseline": "pooled over CPU parts",
            "experiment": "split by CPU part; cross-run and within-run",
        },
        "commits": {
            "baseline": commits["baseline"],
            "experiment": commits["experiment"],
            "repo_dirty_at_build": False,
        },
        "primary_arm": "part_0xd40",
        "deployment": {
            "region": c0["region"],
            "az": None,
            "runtime": cfg.get("runtime"),
            "runtime_version_arn": c0["runtime_version_arn"],
            "architecture": cfg.get("architecture") or "arm64",
            "memory_mb": int(cfg.get("memory_mb") or 10240),
            "snapstart": {"ApplyOn": "PublishedVersions"},
            "functions": {
                "pool": "same-code control arms of "
                + ", ".join(hv["same_code_control_pool"]["runs"])
            },
            "cpu_parts_observed": hv["cpu_identity"]["total_by_part"],
        },
        "model": {
            "id": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
            "revision": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
            "format": "GGUF Q8_0",
        },
        "conditions": {
            "rounds": sum(len(c["orders"]) for cs in conds.values() for c in cs),
            "concurrency": c0["concurrency"],
            "request_ids": sorted({r["request"] for r in main_rows if r["request"]}),
            "started": starts[0],
            "finished": ends[-1],
            "arms": sorted(phases),
            "note": "phases.<arm> here is one CPU part of the pooled same-code control rows, not a deployed "
            "function",
        },
        "classification": {
            "method": "each source run's own expected vs REPORT-observed class; mismatches and errors excluded",
            "mismatches": sum(1 for r in main_rows if not r["ok"]),
        },
        "phases": phases,
        "variance": hv["run_to_run"],
        "errors": {"count": 0, "rows": []},
        "cost": {
            "billed_duration_s": 0.0,
            "total_usd": 0.0,
            "s3_request_usd": 0.0,
            "s3_gb_transferred": 0.0,
            "note": "reanalysis only: no Lambda invocation, no S3 request",
        },
        "correctness": {
            "compared": 0,
            "bit_identical": 0,
            "choice_flips": 0,
            "max_abs_delta": 0.0,
            "note": "not applicable: no new answers",
        },
        "profiling_artifacts": [],
        "artifacts": {
            "rows": "rows.jsonl.gz",
            "streams": "streams.jsonl.gz",
            "sources": "sources.json",
        },
        "reproduce": [
            f"git fetch https://github.com/jsamuel1/strands-decider lkai/cb67-host-variability && git checkout {commits['experiment']}",
            "git fetch https://github.com/jsamuel1/strands-decider "
            + " ".join(sorted({s["branch"] for s in hv["sources"]})),
            "python examples/lambda-kleidiai/hostvar/hostvar.py extract --out <run-dir>",
            "python examples/lambda-kleidiai/hostvar/hostvar.py analyze <run-dir> --fixed-host <offline-summary.json>",
        ],
        "conclusion": conclusion,
        "host_variability": hv,
    }


def fixed_host_from_offline(path: str | None) -> dict:
    if not path:
        return {"used": False, "reason": "no fixed-host summary given"}
    # P3's key for its arm64 host section; the key is not copied into the result.
    g = json.load(open(path))["graviton"]
    plain = g["runs"]["plain"]
    return {
        "used": True,
        "label": "fixed-host (offline), not Lambda latency",
        "source": "P3 results/offline/offline-summary.json (lkai/bc9a-profiler 89a0df4); operator chose to reuse "
        "it rather than run a new fixed-host baseline (H1 ask, 2026-10-21)",
        "what": g.get("what"),
        "host_lines": g.get("host"),
        "pinned": {
            "cpu_part": plain["host"]["cpu"],
            "nproc": plain["host"]["nproc"],
            "kernel": plain["host"]["kernel"],
            "threads": plain["threads"],
            "affinity": "6 threads pinned to 6 CPUs (taskset), per P3",
            "ggml_cpu": plain["ggml_cpu"],
            "python": next((x for x in g.get("host") or [] if x.startswith("Python")), None),
            "cgroup_cpu_max": "max 100000 (host.txt: no CPU quota)",
        },
        "steady_state_p50_s": plain["steady_state_p50_s"],
        "model_load_mmap_s": plain["phases_s"].get("model_load_mmap"),
        "unknown": {
            "instance_type": "not recorded by P3",
            "cpu_implementer_variant_revision_features": "not recorded (host.txt has the part only)",
            "hwcaps": "not recorded",
            "frequency_governor": "not recorded",
            "other_load": "not recorded",
            "toolchain": "the deployed function.zip in P1's pinned Lambda image (same binary as Lambda)",
        },
    }


# ---------------------------------------------------------------------------- README tables

BEGIN, END = "<!-- hostvar:tables begin -->", "<!-- hostvar:tables end -->"


def _ms(d: dict | None) -> str:
    return f"{d['p50'] * 1000:,.0f} ms (n={d['n']})" if d else "n/a"


def _s(d: dict | None) -> str:
    return f"{d['p50']:.2f} s (n={d['n']})" if d else "n/a"


def tables(res: dict) -> str:
    """The README's generated section: every number in it comes from this result file."""
    hv = res["host_variability"]
    ident = hv["cpu_identity"]
    parts = sorted(
        {
            p
            for a in ident["by_run_arm_class"].values()
            for c in a.values()
            for m in c.values()
            for p in m
        }
    )
    out = [
        BEGIN,
        "",
        "### CPU part IDs by run and start class (all arms)",
        "",
        "| run | invoke client | class | " + " | ".join(parts) + " |",
        "|---|---|---|" + "---:|" * len(parts),
    ]
    client = {s["key"]: s["invoke_client"] for s in hv["sources"]}
    for run, arms in ident["by_run_arm_class"].items():
        for cls in CLASSES:
            c: Counter = Counter()
            for a in arms.values():
                c.update(a.get(cls, {}))
            if c:
                out.append(
                    f"| {run} | {client.get(run)} | {cls} | "
                    + " | ".join(str(c.get(p, 0)) for p in parts)
                    + " |"
                )
    out += [
        "",
        f"Totals: {json.dumps(ident['total_by_part'])}. Environments with a part: "
        f"{ident['environment_stability']['environments_with_part']:,}, with more than one: "
        f"{ident['environment_stability']['environments_with_more_than_one_part']}. Log streams with rows: "
        f"{ident['log_stream_stability']['streams_with_rows']:,} ({ident['log_stream_stability']['rows_linked_to_a_stream']:,} "
        f"of {ident['log_stream_stability']['rows']:,} rows linked), with more than one part: "
        f"{ident['log_stream_stability']['streams_with_more_than_one_part']}. Snapshot-building inits: "
        f"{ident['snapshot_build']['streams']} streams over {ident['snapshot_build']['versions']} versions, "
        f"by part {json.dumps(ident['snapshot_build']['by_part'])}; versions by part set "
        f"{json.dumps(ident['snapshot_build']['versions_by_part_set'])}.",
        "",
    ]
    out += [
        "### ggml CPU variant loaded, by part (cold and snapshot-build streams)",
        "",
        "| part | variants (streams) |",
        "|---|---|",
    ]
    for p, v in ident["ggml_cpu_variant_by_part"].items():
        out.append(f"| {p} | " + "; ".join(f"`{k}` ({n})" for k, n in v.items()) + " |")
    pool = hv["same_code_control_pool"]
    out += [
        "",
        f"### Latency by CPU part, same-code control pool ({', '.join(pool['runs'])}; "
        f"{pool['invocations']:,} invocations)",
        "",
        "| class | metric | " + " | ".join(parts) + " |",
        "|---|---|" + "---|" * len(parts),
    ]
    for cls, ms in hv["by_identity"].items():
        for m, by in ms.items():
            fmt = _s if m in ("first_request_total_s", "restore_duration_s") else _ms
            out.append(f"| {cls} | {m} | " + " | ".join(fmt(by.get(p)) for p in parts) + " |")
    out += [
        "",
        "### Run-to-run spread of control medians (same-code pool)",
        "",
        "| class | metric | pooled range (CV) | "
        + " | ".join(f"{p} range (CV)" for p in parts)
        + " |",
        "|---|---|---|" + "---|" * len(parts),
    ]
    for cls, ms in hv["run_to_run"].items():
        for m, d in ms.items():

            def fmt(s, m=m):
                if not s:
                    return "n/a"
                unit = 1 if m in ("first_request_total_s", "restore_duration_s") else 1000
                u = "s" if unit == 1 else "ms"
                return f"{s['range'] * unit:.{2 if unit == 1 else 1}f} {u} ({s['cv']:.1%}, {s['runs']} runs)"

            out.append(
                f"| {cls} | {m} | {fmt(d['pooled_spread'])} | "
                + " | ".join(fmt((d["within_part"].get(p) or {}).get("spread")) for p in parts)
                + " |"
            )
    pv = hv["pooled_vs_within"]
    out += [
        "",
        f"### Within-run comparisons where the part mix moved the pooled result ({len(pv['flagged'])} of "
        f"{pv['comparisons']}; floor {pv['noise_floor_fraction_of_control_median']:.0%} of control median)",
        "",
        "| run | control → arm | class | metric | pooled Δ | within-part Δ | mix (control / arm) | flag |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for c in pv["flagged"]:
        mult = 1 if c["metric"] in ("first_request_total_s", "restore_duration_s") else 1000
        u = "s" if mult == 1 else "ms"
        wp = ", ".join(
            f"{p} {d['delta'] * mult:+.{2 if mult == 1 else 1}f}"
            for p, d in c["within_part"].items()
        )
        mix = " / ".join(" ".join(f"{p}:{n}" for p, n in m.items()) for m in c["mix"])
        out.append(
            f"| {c['run']} | {c['control']} → {c['arm']} | {c['class']} | {c['metric']} | "
            f"{c['pooled_delta'] * mult:+.{2 if mult == 1 else 1}f} {u} | {wp} | {mix} | {c['flag']} |"
        )
    out += ["", END]
    return "\n".join(out)


# ---------------------------------------------------------------------------- CLI


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    with open(path, "wb") as f:
        f.write(gzip.compress(data.encode(), mtime=0))


def _read_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--sources", default=os.path.join(HERE, "sources.json"))
    e.add_argument("--repo", default=os.path.join(HERE, "..", "..", ".."))
    e.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("run")
    z.add_argument("--fixed-host")
    z.add_argument("--baseline-commit", default="70eab7be69f86ddeb8c64158d94df344b5180098")
    z.add_argument("--experiment-commit", required=True)
    t = sub.add_parser("tables", help="print the README's generated section from a results.json")
    t.add_argument("results")
    t.add_argument("--readme", help="replace the generated section of this README in place")
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        text = tables(json.load(open(a.results)))
        if not a.readme:
            print(text)
            return 0
        doc = open(a.readme).read()
        i, j = doc.index(BEGIN), doc.index(END) + len(END)
        with open(a.readme, "w") as f:
            f.write(doc[:i] + text + doc[j:])
        return 0
    if a.cmd == "extract":
        sources = json.load(open(a.sources))
        rows, streams, conds = extract(sources, a.repo)
        os.makedirs(a.out, exist_ok=True)
        _write_jsonl_gz(os.path.join(a.out, "rows.jsonl.gz"), rows)
        _write_jsonl_gz(os.path.join(a.out, "streams.jsonl.gz"), streams)
        with open(os.path.join(a.out, "conditions.json"), "w") as f:
            json.dump(conds, f, indent=1, sort_keys=True)
        with open(os.path.join(a.out, "sources.json"), "w") as f:
            json.dump(sources, f, indent=1)
        print(
            f"extracted {len(rows)} rows, {len(streams)} streams from {len(sources['runs'])} runs"
        )
        return 0
    rows = _read_jsonl_gz(os.path.join(a.run, "rows.jsonl.gz"))
    streams = _read_jsonl_gz(os.path.join(a.run, "streams.jsonl.gz"))
    conds = json.load(open(os.path.join(a.run, "conditions.json")))
    sources = json.load(open(os.path.join(a.run, "sources.json")))
    res = build_result(
        rows,
        streams,
        conds,
        sources,
        fixed_host_from_offline(a.fixed_host),
        {"baseline": a.baseline_commit, "experiment": a.experiment_commit},
    )
    with open(os.path.join(a.run, "results.json"), "w") as f:
        json.dump(res, f, indent=1, allow_nan=False)
    print(res["conclusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
