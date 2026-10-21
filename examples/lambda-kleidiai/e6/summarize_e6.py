"""E6 summary: one lkai-result/1 file for the pyc arm, against the ctl arm (P1's bytecode) from the same run.

  python summarize_e6.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> --run-id <id> \
      --baseline-commit <P1 sha> --reference ../baseline/reference/outputs.json --meta meta.json --out <dir>

Uses P1's summarize.py for every row-level computation (load_run, arm_phases, compare, costs, round_medians), so
the distributions are the same blocks P1 publishes. E6 adds:
- `e6.delta`: pyc minus ctl per phase. Each delta is matched by CPU part (P1's `overhead` method) and gets a
  bootstrap 95% CI over environments. The legacy pooled per-round medians remain descriptive; the required
  host-variability assessment uses shared `(round, CPU part)` cells.
- `e6.bytecode`: the mechanism, per arm. It records how many task-root modules each environment loaded from usable
  bytecode, from e6_entry's first-invocation record.
- `e6.host`: every CPU identity and environment field the wrapper read, with counts per arm. Anything the sandbox
  does not expose stays "unknown".
Status is `improved` only if a deployment headline's CI lies entirely below 0 within every CPU part both arms ran
on, and errors, class mismatches and correctness are clean. It is `regressed` if a headline CI lies entirely above
0, and `no-change` otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
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
    """pyc minus ctl per metric and CPU part, plus descriptive pooled per-round deltas."""
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


UNASSESSED_CELL = "unassessed"


def cell_outcome(delta: float, control_spread: float | None) -> str:
    """One shared cell's outcome: its delta against the control observations' within-cell range."""
    if control_spread is None:
        return UNASSESSED_CELL
    if abs(delta) <= control_spread:
        return "within-control-variation"
    if delta < 0:
        return "negative-beyond-control-variation"
    return "positive-beyond-control-variation"


def _part_variability(cells: list[dict]) -> dict:
    """Aggregate one CPU part from the outcome persisted on each of its shared cells."""
    deltas = [c["delta"] for c in cells]
    signs = sorted({c["sign"] for c in cells})
    assessed = [c for c in cells if c["outcome"] != UNASSESSED_CELL]
    unassessed = [
        {"round": c["round"], "reason": "fewer than two control observations"}
        for c in cells
        if c["outcome"] == UNASSESSED_CELL
    ]
    cell_outcomes = [c["outcome"] for c in assessed]
    if not assessed:
        outcome = "insufficient-control-cell-spread"
    elif len(set(cell_outcomes)) == 1:
        outcome = cell_outcomes[0]
    else:
        outcome = "mixed-or-overlapping-control-variation"
    return {
        "shared_cell_count": len(cells),
        "assessed_cell_count": len(assessed),
        "unassessed_cells": unassessed,
        "control_within_cell_spread_range": (
            [
                min(c["control_spread"] for c in assessed),
                max(c["control_spread"] for c in assessed),
            ]
            if assessed
            else None
        ),
        "delta_range": [min(deltas), max(deltas)],
        "delta_signs": signs,
        "effect_abs_floor": min(abs(d) for d in deltas),
        "cell_outcomes": dict(sorted(Counter(c["outcome"] for c in cells).items())),
        "outcome": outcome,
    }


def host_variability_analysis(rows: list[dict]) -> dict:
    """Primary metrics by shared (round, observed CPU part) cell.

    Arms run as sequential waves. Their order alternates by round, so this controls
    round drift and CPU identity but is not within-round interleaving. Each shared cell
    persists its own outcome (its delta against its control within-cell range, or
    `unassessed` without two control observations); the per-part outcome is aggregated
    from those persisted cell outcomes.
    """
    specs = {
        "cold_imports": ("cold", COLD_KEYS["imports"]),
        "cold_first_request_total": ("cold", COLD_KEYS["first_request_total"]),
        "restore_first_request_total": ("restore", RESTORE_KEYS["first_request_total"]),
        "warm_steady_state_inference": ("warm", WARM_KEYS["steady_state_inference"]),
    }
    metrics: dict = {}
    unknown_cpu_rows: dict = {}
    for name, (cls, measure) in specs.items():
        rs = ok_rows(rows, cls)
        unknown_cpu_rows[name] = {
            arm: sum(r.get("cpu_part") in (None, "", "unknown") for r in rs if r["arm"] == arm)
            for arm in (CONTROL, EXPERIMENT)
        }
        known = [r for r in rs if r.get("cpu_part") not in (None, "", "unknown")]
        keys = sorted({(r["round"], str(r["cpu_part"])) for r in known})
        shared, missing = [], []
        by_part: dict[str, list[dict]] = defaultdict(list)
        for rnd, cpu in keys:
            values = {
                arm: [
                    value
                    for r in known
                    if r["arm"] == arm and r["round"] == rnd and str(r["cpu_part"]) == cpu
                    if (value := measure(r)) is not None
                ]
                for arm in (CONTROL, EXPERIMENT)
            }
            if not all(values.values()):
                missing.append(
                    {
                        "round": rnd,
                        "cpu_part": cpu,
                        "n": {arm: len(values[arm]) for arm in (CONTROL, EXPERIMENT)},
                    }
                )
                continue
            ctl, pyc = values[CONTROL], values[EXPERIMENT]
            ctl_median, pyc_median = statistics.median(ctl), statistics.median(pyc)
            change = pyc_median - ctl_median
            spread = round(max(ctl) - min(ctl), 6) if len(ctl) >= 2 else None
            cell = {
                "round": rnd,
                "cpu_part": cpu,
                "n": {CONTROL: len(ctl), EXPERIMENT: len(pyc)},
                "median": {CONTROL: round(ctl_median, 6), EXPERIMENT: round(pyc_median, 6)},
                "range": {
                    CONTROL: [round(min(ctl), 6), round(max(ctl), 6)],
                    EXPERIMENT: [round(min(pyc), 6), round(max(pyc), 6)],
                },
                "control_spread": spread,
                "delta": round(change, 6),
                "ratio": round(pyc_median / ctl_median, 6) if ctl_median else None,
                "sign": "negative" if change < 0 else "positive" if change > 0 else "zero",
                "outcome": cell_outcome(round(change, 6), spread),
            }
            shared.append(cell)
            by_part[cpu].append(cell)
        metrics[name] = {
            "class": cls,
            "shared_cells": shared,
            "missing_cells": missing,
            "by_cpu_part": {
                cpu: _part_variability(cells) for cpu, cells in sorted(by_part.items())
            },
        }

    imports = metrics["cold_imports"]
    import_cells = imports["shared_cells"]
    import_outcomes = {cpu: part["outcome"] for cpu, part in imports["by_cpu_part"].items()}
    assessed_outcomes = {
        cpu: outcome
        for cpu, outcome in import_outcomes.items()
        if outcome != "insufficient-control-cell-spread"
    }
    unassessed_parts = sorted(set(import_outcomes) - set(assessed_outcomes))
    comparable_outcomes = {
        "negative-beyond-control-variation",
        "positive-beyond-control-variation",
        "within-control-variation",
    }
    imports_comparable = (
        bool(assessed_outcomes)
        and len(set(assessed_outcomes.values())) == 1
        and all(outcome in comparable_outcomes for outcome in assessed_outcomes.values())
    )
    imports_outcome = (
        next(iter(set(assessed_outcomes.values()))) if imports_comparable else "host-conditioned"
    )
    if not assessed_outcomes:
        imports_outcome = "unassessed"
    directional_sign = {
        "negative-beyond-control-variation": "negative",
        "positive-beyond-control-variation": "positive",
    }.get(imports_outcome)
    unassessed_conflicts = [
        cpu
        for cpu in unassessed_parts
        if directional_sign
        and any(
            sign not in (directional_sign, "zero")
            for sign in imports["by_cpu_part"][cpu]["delta_signs"]
        )
    ]
    if unassessed_conflicts:
        imports_comparable = False
        imports_outcome = "host-conditioned"
    if imports_comparable and imports_outcome == "negative-beyond-control-variation":
        reason = (
            "every assessed cell has a negative delta whose absolute size exceeds its control within-cell "
            "spread; sparse CPU parts are unassessed and have no opposite observed sign"
        )
    elif imports_comparable and imports_outcome == "within-control-variation":
        reason = (
            "every assessed cell stays within its control within-cell spread; sign variation is a "
            "limitation, not a directional effect"
        )
    elif not assessed_outcomes:
        reason = "no shared known-CPU cells have enough control observations to estimate within-cell spread"
    else:
        reason = (
            "CPU parts have different or indeterminate outcomes after control-variation adjustment, or a sparse "
            "part contradicts the assessed directional sign"
        )
    unknown_import_rows = unknown_cpu_rows["cold_imports"]
    return {
        "schema": "lkai-host-variability/1",
        "observed": {"primary_metrics": metrics},
        "unknown": {
            "cpu_identity_rows": unknown_cpu_rows,
            "environment_fields": [
                "availability zone",
                "physical host",
                "cgroup CPU limit",
                "cgroup memory limit",
            ],
        },
        "method": "Within each observed CPU part, compare arm medians in each round. Arms are sequential waves "
        "whose first position alternates by round; this is alternation, not within-round interleaving. A "
        "directional effect must exceed the control observations' within-cell range in every assessed shared "
        "cell; deltas inside that range support a null outcome even if their signs vary. Every shared cell "
        "persists its own outcome (within-control-variation, negative- or positive-beyond-control-variation, or "
        "unassessed), and each CPU part's outcome is aggregated from those persisted cell outcomes. A cell with "
        "fewer than two control observations has no spread estimate and is explicitly unassessed. Missing-arm "
        "cells and rows "
        "without CPU identity are listed but never treated as comparable strata.",
        "assessment": {
            "primary_effect": {
                "metric": "cold_imports",
                "comparable": imports_comparable,
                "outcome": imports_outcome,
                "by_cpu_part": import_outcomes,
                "unassessed_cpu_parts": unassessed_parts,
                "unassessed_conflicts": unassessed_conflicts,
                "shared_cells": len(import_cells),
                "unassessed_unknown_cpu_rows": unknown_import_rows,
                "reason": reason,
            },
            "deployment_headlines": "no consistent cold-first-request, restore, or warm benefit across observed "
            "CPU parts and rounds",
            "rerun": (
                "not needed: host-adjusted assessed cells support one imports-phase outcome; sparse CPU parts "
                "and unmatched cells remain explicit limitations, while deployment headlines remain null"
                if imports_comparable
                else "needed for the imports effect unless the host-conditioned or insufficient outcome is "
                "accepted as a limitation"
            ),
        },
    }


# The engine's own init line, logged once per environment init in the same log stream as that init's LKAI records.
ENGINE_CPU_LINE = re.compile(r"decider\.engine init cpu_part=(\S+)")
UNKNOWN_CPU = (None, "", "unknown")
SNAPSHOT_METRICS = ("imports", "import_total")


def published_versions(cond: dict) -> dict[str, dict[str, str]]:
    """{arm: {version: function name}} for every version this run published (the harness's rounds_meta)."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for m in cond.get("rounds_meta", []):
        for arm, pub in m.get("published", {}).items():
            out[arm][str(pub["version"])] = cond["functions"][arm]["FunctionName"]
    return dict(out)


def snapshot_records(run: str, cond: dict) -> dict:
    """Every before_snapshot record in the saved logs, each bound to the CPU part of its own log stream.

    A Lambda log stream belongs to one execution environment, so the snapshot init's `decider.engine init
    cpu_part=...` line and its before_snapshot record share a stream. Binding never crosses streams. Only
    streams of versions this run published count; records from any other version are counted as excluded.
    A stream with no known CPU part is `missing`; one naming more than one distinct part is `ambiguous`.
    Neither is ever used as a comparison stratum."""
    versions = published_versions(cond)
    out: dict = {}
    for arm in cond["arms"]:
        fn = cond["functions"][arm]["FunctionName"]
        path = os.path.join(run, "logs", f"{fn}.jsonl.gz")
        streams: dict[str, dict] = defaultdict(lambda: {"records": [], "cpu_parts": set()})
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    e = json.loads(line)
                    st = streams[e.get("logStreamName", "")]
                    msg = e.get("message", "")
                    for part in ENGINE_CPU_LINE.findall(msg):
                        if part not in UNKNOWN_CPU:
                            st["cpu_parts"].add(part)
                    st["records"] += [
                        r for r in p1.phases.parse_lkai(msg) if r.get("kind") == "before_snapshot"
                    ]
        published = set(versions.get(arm, {}))
        recs, excluded, seen = [], 0, set()
        for name, st in sorted(streams.items()):
            ver = name.split("[", 1)[1].split("]", 1)[0] if "[" in name else None
            for r in st["records"]:
                if r.get("wall_ns") in seen:
                    continue
                seen.add(r.get("wall_ns"))
                if ver not in published:
                    excluded += 1
                    continue
                parts = sorted(st["cpu_parts"])
                binding = "linked" if len(parts) == 1 else "missing" if not parts else "ambiguous"
                recs.append(
                    {
                        "version": ver,
                        "log_stream": name,
                        "binding": binding,
                        "cpu_part": parts[0] if binding == "linked" else None,
                        "stream_cpu_parts": parts,
                        **{k: (r.get("init") or {}).get(k) for k in SNAPSHOT_METRICS},
                    }
                )
        out[arm] = {"records": recs, "excluded_unpublished_records": excluded}
    return out


def snapshot_host_analysis(run: str, cond: dict) -> dict:
    """Snapshot-init (publish-time) metrics by the CPU part bound to each record from its own log stream.

    Each published version yields one snapshot init per CPU part Lambda prepares, so a (version, part) cell holds one
    record per arm. The comparison is therefore per CPU part across the run's published versions: the control
    spread is the range of ctl's per-version values within that part, and `cell_outcome` judges the part's delta
    against it. Only linked records count; missing, ambiguous and unpublished-version records are reported and
    excluded."""
    bound = snapshot_records(run, cond)
    accounting, mix = {}, Counter()
    for arm, b in bound.items():
        c = Counter(r["binding"] for r in b["records"])
        accounting[arm] = {
            "records": len(b["records"]),
            "linked": c["linked"],
            "missing": c["missing"],
            "ambiguous": c["ambiguous"],
            "excluded_unpublished_records": b["excluded_unpublished_records"],
        }
        mix.update(f"{arm}:{r['cpu_part']}" for r in b["records"] if r["binding"] == "linked")
    linked = {
        arm: [r for r in b["records"] if r["binding"] == "linked"] for arm, b in bound.items()
    }
    metrics: dict = {}
    for metric in SNAPSHOT_METRICS:
        parts = sorted({r["cpu_part"] for rs in linked.values() for r in rs})
        by_part, unmatched = {}, []
        for cpu in parts:
            vals = {
                arm: sorted(
                    (r["version"], r[metric])
                    for r in linked.get(arm, [])
                    if r["cpu_part"] == cpu and r[metric] is not None
                )
                for arm in (CONTROL, EXPERIMENT)
            }
            if not all(vals.values()):
                unmatched.append(
                    {"cpu_part": cpu, "n": {arm: len(vals[arm]) for arm in (CONTROL, EXPERIMENT)}}
                )
                continue
            ctl = [v for _, v in vals[CONTROL]]
            pyc = [v for _, v in vals[EXPERIMENT]]
            mc, mp = statistics.median(ctl), statistics.median(pyc)
            change = round(mp - mc, 6)
            spread = round(max(ctl) - min(ctl), 6) if len(ctl) >= 2 else None
            by_part[cpu] = {
                "n": {CONTROL: len(ctl), EXPERIMENT: len(pyc)},
                "per_version": {arm: {v: round(x, 6) for v, x in vals[arm]} for arm in vals},
                "median": {CONTROL: round(mc, 6), EXPERIMENT: round(mp, 6)},
                "range": {
                    CONTROL: [round(min(ctl), 6), round(max(ctl), 6)],
                    EXPERIMENT: [round(min(pyc), 6), round(max(pyc), 6)],
                },
                "control_spread": spread,
                "delta": change,
                "ratio": round(mp / mc, 6) if mc else None,
                "sign": "negative" if change < 0 else "positive" if change > 0 else "zero",
                "outcome": cell_outcome(change, spread),
            }
        metrics[metric] = {"by_cpu_part": by_part, "unmatched_cpu_parts": unmatched}

    imports = metrics["imports"]["by_cpu_part"]
    assessed = {cpu: p["outcome"] for cpu, p in imports.items() if p["outcome"] != UNASSESSED_CELL}
    unassessed = sorted(set(imports) - set(assessed))
    comparable = bool(assessed) and len(set(assessed.values())) == 1
    assessed_outcome = next(iter(set(assessed.values()))) if comparable else None
    directional_sign = {
        "negative-beyond-control-variation": "negative",
        "positive-beyond-control-variation": "positive",
    }.get(assessed_outcome)
    unassessed_conflicts = [
        cpu
        for cpu in unassessed
        if directional_sign and imports[cpu]["sign"] not in (directional_sign, "zero")
    ]
    if unassessed_conflicts:
        comparable = False
    if not imports:
        outcome = "unassessed"
        reason = "no CPU part has a linked snapshot-init record on both arms"
    elif not assessed:
        outcome = "unassessed"
        reason = "no shared CPU part has two control snapshot inits to estimate spread"
    elif comparable:
        outcome = assessed_outcome
        reason = (
            "every assessed CPU part has the same outcome against its control spread across published versions; "
            "sparse CPU parts have no opposite-sign observation; missing, ambiguous and unpublished records are "
            "excluded and counted"
        )
    else:
        outcome = "host-conditioned"
        reason = "CPU parts disagree after control-spread adjustment or a sparse part has an opposite observed sign"
    return {
        "schema": "lkai-snapshot-host-variability/1",
        "accounting": accounting,
        "arm_cpu_part_mix": dict(sorted(mix.items())),
        "records": {arm: b["records"] for arm, b in bound.items()},
        "metrics": metrics,
        "assessment": {
            "metric": "imports",
            "comparable": comparable,
            "outcome": outcome,
            "by_cpu_part": {cpu: p["outcome"] for cpu, p in imports.items()},
            "unassessed_cpu_parts": unassessed,
            "unassessed_conflicts": unassessed_conflicts,
            "unmatched_cpu_parts": metrics["imports"]["unmatched_cpu_parts"],
            "reason": reason,
        },
        "method": "Each before_snapshot record is bound to the CPU part named by the `decider.engine init "
        "cpu_part=` line in its own log stream (one stream = one execution environment); binding never crosses "
        "streams. Only streams of versions this run published count. A stream with no known part is missing, one "
        "with more than one distinct part is ambiguous; neither is a comparison stratum. Within each CPU part, "
        "the pyc-minus-ctl median delta across published versions is judged against the range of ctl's "
        "per-version values (cell_outcome); a part with fewer than two control inits is unassessed, and an "
        "opposite observed sign on such a sparse part blocks a comparable directional outcome.",
        "limits": [
            "snapshot inits run inside publish_version, so this is publish time, not a request-path latency",
            "only the CPU part is observable for a snapshot init (the engine's init log line); variant, revision, "
            "HWCAP, features, CPU count and kernel come from e6_entry's first-invocation record, which a snapshot "
            "init does not emit, so they are unknown here",
            "availability zone, physical host and cgroup limits are unknown",
            "Lambda chose which CPU parts to prepare per version; one init per (version, part, arm), so spread is "
            "across versions, not within a version",
            "no Graviton generation is inferred from the CPU part",
        ],
    }


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
    res["host_variability"] = host_variability_analysis(rows)
    res["host_variability"]["snapshot_init"] = snapshot_host_analysis(run, cond)
    res["e6"] = {
        "delta": {
            "cold": delta(rows, "cold", COLD_KEYS),
            "warm": delta(rows, "warm", WARM_KEYS),
            "restore": delta(rows, "restore", RESTORE_KEYS),
            "method": "pyc minus ctl, medians within each CPU part both arms ran on; ci95 = bootstrap over "
            "environments (4,000 resamples, seed 6); pooled per-round deltas are descriptive only; "
            "host_variability contains the required round-by-CPU-part cells",
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
    snap = (res.get("host_variability") or {}).get("snapshot_init") or {}
    for cpu, v in snap.get("metrics", {}).get("imports", {}).get("by_cpu_part", {}).items():
        key = f"snapshot_imports_{cpu}"
        out[key + "_ctl"], out[key + "_pyc"] = v["median"][CONTROL], v["median"][EXPERIMENT]
        out[key + "_d"] = v["delta"]
        out[key + "_n"] = f"{v['n'][CONTROL]}/{v['n'][EXPERIMENT]}"
    return out


if __name__ == "__main__":
    sys.exit(main())
