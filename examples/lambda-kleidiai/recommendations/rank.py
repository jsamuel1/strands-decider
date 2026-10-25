"""Ranked recommendations from the lambda-kleidiai experiments, computed from their published results.

    python examples/lambda-kleidiai/recommendations/rank.py            # writes ranking.json + charts/*.svg
    python examples/lambda-kleidiai/recommendations/rank.py --check    # fails if the committed outputs are stale

Inputs are the pinned copies in sources/ (sources.json records each object's S3 VersionId and SHA-256). Every
number in README.md comes from ranking.json, and every value in ranking.json is read from a result file. Nothing
here is typed in by hand except which field of which run answers which question (EFFECTS below).

Rules (initiative HOST VARIABILITY): deltas are candidate minus control within one CPU part, as read from
/proc/cpuinfo. The headline part is 0xd40, the only part with enough samples in every class. Other parts are listed
with their n and are never pooled into the headline. No CPU part is mapped to a processor generation. An effect
whose 95% interval includes 0 ranks as "no measurable change".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
PART = "0xd40"

# Each effect names the run that measured it, the metrics its own conclusion claims ("claimed": they passed that
# card's within-part host-spread rule; an interval that excludes 0 without that claim ranks as no measurable change),
# the JSON prefix of its per-phase/per-part deltas, and what kind of
# change it is. "kind" separates network-transfer from compute effects (initiative brief: keep them separate).
# "scope" says where the change is safe: "general" changes behaviour that every llama.cpp/decshim deployment shares;
# "lambda" depends on a Lambda-only condition (SnapStart, memfd, the S3 download, Lambda's CPU allotment).
EFFECTS: list[dict] = [
    {
        "id": "E8",
        "short": "6-thread repack (E8)",
        "claimed": ["cold.first_request_total", "cold.model_load_mmap"],
        "title": "Six-thread KleidiAI repack at load",
        "run": "E8-parallel-repack/20261021T110412Z",
        "prefix": ["e8", "pairs", "total"],
        "switch": "GGML_KLEIDIAI_REPACK_THREADS=6",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Load-time only, threads already idle during load. Same packed bytes at 1, 2, 3 and 6 threads.",
    },
    {
        "id": "E9",
        "short": "Pre-packed weights (E9)",
        "claimed": ["cold.first_request_total", "cold.model_load_mmap"],
        "title": "Pre-packed KleidiAI weights in the GGUF",
        "run": "E9-prepacked/20261021T101749Z",
        "prefix": ["e9_vs_base"],
        "switch": "E9 loader patch + s2b-h512-Q8_0-e9kr8.gguf",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Needs E9's loader and a KleidiAI-capable CPU. A file packed for one layout converts on the other.",
    },
    {
        "id": "E4",
        "short": "Stream into loader (E4)",
        "claimed": ["cold.first_request_total"],
        "title": "Stream weights into the loader during download",
        "run": "E4-stream-load/20261020T181753Z",
        "prefix": ["stream_vs_base"],
        "switch": "stream loader + mallopt M_MMAP_THRESHOLD=4 MiB",
        "kind": "overlap",
        "scope": "lambda",
        "scope_note": "Only pays when weights arrive over the network at load. Without the mallopt pin it costs +470 MB.",
    },
    {
        "id": "E7",
        "short": "Skip tied output (E7)",
        "claimed": [
            "cold.first_request_total",
            "cold.model_load_mmap",
            "restore.first_request_total",
            "cold.report_max_memory_mb",
        ],
        "title": "Skip the unused tied output head",
        "run": "E7-skip-tied-output/20261021T102524Z",
        "prefix": ["e7", "delta"],
        "switch": "LKAI_SKIP_TIED_OUTPUT=1",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Safe only while the deployment never asks llama.cpp for logits (the patched LM head aborts).",
    },
    {
        "id": "E5",
        "short": "No-vocab GGUF (E5)",
        "claimed": [
            "cold.first_request_total",
            "cold.model_load_mmap",
            "cold.report_max_memory_mb",
        ],
        "title": "GGUF without the embedded vocabulary",
        "run": "E5-mmap-format/20261020T114406Z",
        "prefix": ["e5_vs_base"],
        "switch": "s2b-h512-Q8_0-novocab.gguf",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Only for engines that tokenize outside llama.cpp, as decshim does.",
    },
    {
        "id": "E11",
        "short": "Release memfd (E11)",
        "claimed": ["restore.first_request_total", "cold.report_max_memory_mb"],
        "title": "Release unaliased memfd pages before the snapshot",
        "run": "E11-release-memfd/20261021T103834Z",
        "prefix": ["e11", "delta"],
        "switch": "LKAI_RELEASE_MEMFD=1",
        "kind": "compute",
        "scope": "lambda",
        "scope_note": "Only for a memfd-backed GGUF. Uses madvise(MADV_REMOVE) because Lambda refuses fallocate.",
    },
    {
        "id": "E12",
        "short": "Metadata-only clear (E12)",
        "claimed": ["restore.first_request_total"],
        "title": "Clear only cache metadata between requests",
        "run": "E12-cache-clear/20261021T105615Z",
        "prefix": ["e12", "delta"],
        "switch": "DEC_CLEAR_DATA=0",
        "kind": "compute",
        "scope": "lambda",
        "scope_note": "Gain is first touch of SnapStart-restored memory. Correctness proven only at llama.cpp f74e2fa6a.",
    },
    {
        "id": "E10",
        "short": "n_ctx 4096 (E10)",
        "claimed": ["restore.first_request_total", "cold.report_max_memory_mb"],
        "title": "n_ctx 4096 instead of 16384",
        "run": "E10-kv-size/20261021T110426Z",
        "prefix": ["e10", "delta", "c4096"],
        "switch": "n_ctx=4096",
        "kind": "compute",
        "scope": "general",
        "scope_note": "4096 is the model window; refuses none of the 231 JevBench tasks. Smaller sizes refuse requests.",
    },
    {
        "id": "E13",
        "short": "GOMP_SPINCOUNT (E13)",
        "claimed": [],
        "tentative": ["warm.steady_state_inference"],
        "title": "Shorter OpenMP spin before sleep",
        "run": "E13-omp-wait/20261021T103304Z-spin30k",
        "prefix": ["e13", "delta"],
        "switch": "GOMP_SPINCOUNT=30000",
        "kind": "compute",
        "scope": "lambda",
        "scope_note": "Measured only under Lambda's CPU allotment. A dedicated host may prefer spinning.",
    },
    {
        "id": "E6",
        "short": "Complete bytecode (E6)",
        "claimed": [],
        "title": "Ship complete bytecode",
        "run": "E6-bytecode/20261021T045838Z",
        "prefix": ["e6", "delta"],
        "switch": "unchecked-hash .pyc for every .py",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Packaging hygiene. Pays only when imports sit on the request path.",
    },
    {
        "id": "E15",
        "short": "F32 sgemm (E15)",
        "claimed": [],
        "title": "llamafile F32 sgemm in the SVE/i8mm variants",
        "run": "E15-f32-sgemm/20261021T2340Z-a",
        "prefix": ["e15", "delta"],
        "switch": "LKAI_E15_A=1",
        "kind": "compute",
        "scope": "general",
        "scope_note": "Answers not bit-identical (max |dp| 0.0148, 0 flips).",
    },
]

# Effects measured without per-part CI blocks in the shared shape; each is summarised from its own fields.
NULLS: list[dict] = [
    {
        "id": "E1",
        "title": "S3 Express One Zone bucket",
        "run": "E1-s3-express/20261020T1235Z-e1",
        "kind": "network",
    },
    {
        "id": "E2",
        "title": "S3 One Zone-IA storage class",
        "run": "E2-onezone-ia/20261020T1125Z-e2",
        "kind": "network",
    },
    {
        "id": "E3",
        "title": "Other ranged-GET thread/part sizes and CRT",
        "run": "E3-parallel-range-get/20261020T113449Z-t64x64",
        "kind": "network",
    },
    {
        "id": "E14",
        "title": "Row-split ggml concat",
        "run": "E14-concat/20261021T232302Z",
        "kind": "compute",
    },
    {
        "id": "E13a",
        "title": "OMP_WAIT_POLICY=active",
        "run": "E13-omp-wait/20261021T103304Z-active",
        "kind": "compute",
    },
]

# (class, phase, label, unit) rows the tables show, in order.
METRICS = [
    ("cold", "first_request_total", "Cold first request", "s"),
    ("restore", "first_request_total", "Restored first request", "s"),
    ("warm", "steady_state_inference", "Warm inference", "s"),
    ("cold", "report_max_memory_mb", "Max memory", "MB"),
    ("cold", "model_load_mmap", "Cold model load", "s"),
    ("cold", "weight_download", "Cold weight download", "s"),
]


def load(run: str) -> dict:
    return json.loads((SOURCES / run / "result.json").read_text())


def dig(o, path):
    for k in path:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def cell(res: dict, prefix: list[str], cls: str, phase: str) -> dict | None:
    """One phase's delta on every CPU part: {part: {delta, ci95, n}}. None when the run did not measure it."""
    parts = dig(res, [*prefix, cls, phase, "by_cpu_part"])
    if not parts:
        return None
    out = {}
    for part, v in sorted(parts.items()):
        out[part] = {
            "delta": v["delta_p50"],
            "ci95": v.get("ci95") or v.get("ci95_s"),
            "n": v.get("n"),
        }
    return out


def verdict(c: dict | None) -> str:
    """measured: the 0xd40 interval excludes 0. no-change: it includes 0. no-ci: the run published no interval."""
    if not c or PART not in c:
        return "not-measured"
    ci = c[PART]["ci95"]
    if not ci:
        return "no-ci"
    lo, hi = ci
    return "measured" if (hi < 0 or lo > 0) else "no-change"


def baseline() -> dict:
    p1 = load("P1-baseline/20261020T080835Z")
    inst = p1["phases"]["inst"]
    out = {}
    for cls, phase, label, unit in METRICS:
        d = inst.get(cls, {}).get(phase)
        if d:
            out[f"{cls}.{phase}"] = {
                "label": label,
                "unit": unit,
                "p50": d["p50"],
                "p90": d["p90"],
                "n": d["n"],
            }
    out["variance"] = {
        k: {"cv": v["cv"], "range_s": v["range_s"]} for k, v in p1["variance"]["inst"].items()
    }
    out["cold_phases"] = {
        k: inst["cold"][k]["p50"]
        for k in ("weight_download", "model_load_mmap", "imports", "context_init")
    }
    out["restore_phases"] = {
        "report_restore_duration": inst["restore"]["report_restore_duration"]["p50"],
        "first_request_inference": inst["restore"]["first_request_inference"]["p50"],
    }
    out["cost"] = {
        k: p1["cost"][k] for k in ("total_usd", "lambda_duration_usd", "s3_request_usd", "prices")
    }
    out["memory_gb"] = p1["deployment"]["memory_mb"] / 1024
    return out


def lambda_usd_per_1000(seconds: float, b: dict) -> float:
    """Lambda duration cost of `seconds` per 1,000 invocations at P1's memory and arm64 GB-s price."""
    return 1000 * seconds * b["memory_gb"] * b["cost"]["prices"]["arm64_gb_s"]


def effect_row(e: dict) -> dict:
    res = load(e["run"])
    row = {
        k: e[k]
        for k in ("id", "short", "title", "run", "switch", "kind", "scope", "scope_note", "claimed")
    }
    row["status"] = res["status"]
    row["commit"] = res["commits"].get("experiment")
    row["correctness"] = res.get("correctness")
    row["cost_usd_run"] = (res.get("cost") or {}).get("total_usd")
    row["metrics"] = {}
    for cls, phase, _label, _unit in METRICS:
        c = cell(res, e["prefix"], cls, phase)
        key = f"{cls}.{phase}"
        v = verdict(c)
        # An interval that excludes 0 is necessary, not sufficient: the experiment's own host-spread rule must also
        # stand behind the effect. Without that claim it ranks as no measurable change.
        if v in ("measured", "no-ci") and key in e.get("tentative", []):
            v = "inconclusive"
        elif v in ("measured", "no-ci") and key not in e["claimed"]:
            v = "not-claimed"
        row["metrics"][key] = {"parts": c, "verdict": v}
    return row


def null_row(e: dict) -> dict:
    res = load(e["run"])
    return {
        "id": e["id"],
        "title": e["title"],
        "run": e["run"],
        "kind": e["kind"],
        "status": res["status"],
        "commit": res["commits"].get("experiment"),
        "conclusion_head": res["conclusion"].split(". ")[0],
    }


def e13_warm() -> dict:
    """E13 and E15 publish warm results as paired per-round percentages; the bootstrap block above is per-request."""
    out = {}
    for key, run, block in (
        ("E13", "E13-omp-wait/20261021T103304Z-spin30k", "e13"),
        ("E13-passive", "E13-omp-wait/20261021T103304Z-passive", "e13"),
        ("E13-active", "E13-omp-wait/20261021T103304Z-active", "e13"),
        ("E15", "E15-f32-sgemm/20261021T2340Z-a", "e15"),
    ):
        w = load(run)[block]["warm_paired"]["report_duration"][PART]
        rc = load(run)[block]["round_cells"]["warm_report_duration"]["by_cpu_part"][PART]
        out[key] = {
            "mean_round_pct": w["mean_round_pct"],
            "round_pct_range": [min(w["pct_by_round"]), max(w["pct_by_round"])],
            "rounds": len(w["pct_by_round"]),
            "rounds_below_zero": w["rounds_below_zero"],
            "sign_flip_p": w["sign_flip_p"],
            "control_noise_pct": w["control_split_half"]["max_abs_round_pct"],
            "verdict": w["verdict"],
            "cells": rc["cells"],
            # E15's round_cells carry no in-band classification; absent stays None, never a synthetic 0.
            "cells_in_band": (rc.get("counts") or {}).get("in-band") if "counts" in rc else None,
            "cell_outcome": rc.get("outcome"),
        }
    return out


def e12_warm() -> dict:
    w = load("E12-cache-clear/20261021T105615Z")["e12"]["warm_paired"]["by_cpu_part"][PART]
    return {"median_ratio": w["median_ratio"], "ci95": w["ci95"], "cells": w["cells"]}


def network() -> dict:
    e1 = load("E1-s3-express/20261020T1235Z-e1")["e1"]["comparison"]
    e2 = load("E2-onezone-ia/20261020T1125Z-e2")["comparison"]["cold"]["weight_download"]
    e3_runs = sorted(
        p.parent.name for p in (SOURCES / "E3-parallel-range-get").glob("*/result.json")
    )
    e3 = []
    for r in e3_runs:
        res = load(f"E3-parallel-range-get/{r}")
        v = res["e3"]["download"]["vs_ctl"]
        e3.append(
            {
                "strategy": res["e3"]["strategy"],
                "status": res["status"],
                "delta_s": v.get("delta_p50_s"),
                "ci95": v.get("ci95_s"),
            }
        )
    return {
        "E1_no_vpc": {
            "download_delta_s": e1["bucket_type_no_vpc"]["cold.weight_download"]["delta_p50_s"],
            "rounds_faster": e1["bucket_type_no_vpc"]["cold.weight_download"]["rounds_a_faster"],
        },
        "E1_same_az": {
            "download_delta_s": e1["bucket_type_vpc_same_az"]["cold.weight_download"][
                "delta_p50_s"
            ],
        },
        "E2": {"ratio_p50": e2["ratio_p50"], "ratio_ci95": e2["ratio_ci95"]},
        "E3": e3,
    }


def tokenizer() -> dict:
    t0 = load("T0-tokenizer-baseline/20261021T102818Z")
    return {
        "status": t0["status"],
        "commit": t0["commits"]["experiment"],
        "lambda_share": t0.get("lambda_share"),
        "headroom": t0.get("headroom"),
    }


def build() -> dict:
    sources = json.loads((HERE / "sources.json").read_text())
    for s in sources["results"]:
        rel = s["key"].split("/", 1)[1]
        got = hashlib.sha256((SOURCES / rel).read_bytes()).hexdigest()
        if got != s["sha256"]:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {s['sha256']}")
    rk = {
        "schema": "lkai-ranking/1",
        "cpu_part": PART,
        "baseline": baseline(),
        "effects": [effect_row(e) for e in EFFECTS],
        "nulls": [null_row(e) for e in NULLS],
        "warm_paired": e13_warm(),
        "e12_warm": e12_warm(),
        "network": network(),
        "tokenizer": tokenizer(),
    }
    # Billing (P1's us-east-2 price table). Cold: billed duration covers Init + Duration, so a cold first-request
    # saving is GB-s at the configured memory. Restore: Lambda's Restore Duration is NOT billed as GB-s; SnapStart
    # bills per GB restored and per GB-s cached, both on snapshot size. So restore cost moves with (a) the restored
    # first request's REPORT Duration and (b) memory saved, used here as the snapshot-size proxy (not measured
    # directly). S3 cost is unchanged for every compute effect (same bytes, same GETs).
    b = rk["baseline"]
    pr = b["cost"]["prices"]
    for row in rk["effects"]:
        usd = {}
        m = row["metrics"]["cold.first_request_total"]
        if m["parts"] and PART in m["parts"] and m["verdict"] in ("measured", "no-ci"):
            usd["cold_duration_per_1000"] = round(
                lambda_usd_per_1000(m["parts"][PART]["delta"], b), 4
            )
        mem = row["metrics"]["cold.report_max_memory_mb"]
        if mem["parts"] and PART in mem["parts"] and mem["verdict"] in ("measured", "no-ci"):
            gb = mem["parts"][PART]["delta"] / 1024
            usd["restore_gb_per_1000_proxy"] = round(1000 * gb * pr["snapstart_restore_gb"], 4)
            usd["cache_per_version_month_proxy"] = round(
                gb * pr["snapstart_cache_gb_s"] * 86400 * 30, 4
            )
        row["cost_delta_usd"] = usd
    return rk


# ---------------------------------------------------------------------------- charts
# One message per chart. Plain SVG, no scripts: renders in GitHub, Artifactory's iframe and a browser.

W, BAR_H, GAP, LEFT, RIGHT = 640, 22, 8, 200, 80
INK, MUTED, GOOD, NEUTRAL, TENTATIVE = "#1a1a1a", "#666666", "#1f6f3f", "#9a9a9a", "#a8650a"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(
    title: str, items: list[tuple[str, float, str]], unit: str, note: str, digits: int = 2
) -> str:
    """Horizontal bars of seconds saved (positive = faster). items: (label, value, colour)."""
    top = 46
    h = top + len(items) * (BAR_H + GAP) + 34
    vmax = max(v for _, v, _ in items) or 1.0
    span = W - LEFT - RIGHT
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        f'role="img" aria-label="{_esc(title)}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{h}" fill="#ffffff"/>',
        f'<text x="12" y="24" font-size="15" font-weight="bold" fill="{INK}">{_esc(title)}</text>',
    ]
    for i, (label, v, colour) in enumerate(items):
        y = top + i * (BAR_H + GAP)
        bw = max(1.0, span * max(v, 0) / vmax)
        out.append(
            f'<text x="{LEFT - 8}" y="{y + 15}" font-size="12" text-anchor="end" fill="{INK}">{_esc(label)}</text>'
        )
        out.append(f'<rect x="{LEFT}" y="{y}" width="{bw:.1f}" height="{BAR_H}" fill="{colour}"/>')
        out.append(
            f'<text x="{LEFT + bw + 6:.1f}" y="{y + 15}" font-size="12" fill="{INK}">{v:.{digits}f} {_esc(unit)}</text>'
        )
    out.append(f'<text x="12" y="{h - 12}" font-size="11" fill="{MUTED}">{_esc(note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def stacked_chart(title: str, segments: list[tuple[str, float, str]], unit: str, note: str) -> str:
    """One horizontal bar split into phases: where the time goes. One legend row per phase keeps labels in bounds."""
    y = 46
    h = y + 40 + len(segments) * 18 + 28
    total = sum(v for _, v, _ in segments)
    span = W - 24
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        f'role="img" aria-label="{_esc(title)}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{h}" fill="#ffffff"/>',
        f'<text x="12" y="24" font-size="15" font-weight="bold" fill="{INK}">{_esc(title)}</text>',
    ]
    x = 12.0
    for _label, v, colour in segments:
        w = span * v / total
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="30" fill="{colour}" stroke="#ffffff"/>'
        )
        x += w
    for i, (label, v, colour) in enumerate(segments):
        ly = y + 48 + i * 18
        pct = 100 * v / total
        out.append(f'<rect x="12" y="{ly - 10}" width="12" height="12" fill="{colour}"/>')
        out.append(
            f'<text x="30" y="{ly}" font-size="12" fill="{INK}">{_esc(label)}: {v:.1f} {unit} ({pct:.0f}%)</text>'
        )
    out.append(f'<text x="12" y="{h - 10}" font-size="11" fill="{MUTED}">{_esc(note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def charts(rk: dict) -> dict[str, str]:
    b = rk["baseline"]
    eff = {e["id"]: e for e in rk["effects"]}

    def saved(eid: str, key: str) -> float:
        return -eff[eid]["metrics"][key]["parts"][PART]["delta"]

    def colour(eid: str, key: str) -> str:
        v = eff[eid]["metrics"][key]["verdict"]
        return (
            GOOD if v in ("measured", "no-ci") else (TENTATIVE if v == "inconclusive" else NEUTRAL)
        )

    cp = b["cold_phases"]
    total = b["cold.first_request_total"]["p50"]
    other = total - sum(cp.values())
    out = {
        "cold-time.svg": stacked_chart(
            f"Where a {total:.0f} s cold first request goes (P1 baseline p50)",
            [
                ("S3 weight download (network)", cp["weight_download"], "#3b6ea5"),
                ("Model load + KleidiAI repack (compute)", cp["model_load_mmap"], "#c47a2c"),
                ("Python imports", cp["imports"], "#6b6b6b"),
                ("Everything else", other, "#bdbdbd"),
            ],
            "s",
            "P1 inst arm, 50 cold starts. Phase p50s; 'everything else' is the remainder of the request p50.",
        )
    }
    for name, key, ids, title in (
        (
            "cold-saved.svg",
            "cold.first_request_total",
            ["E8", "E9", "E4", "E7", "E5", "E11", "E12", "E10", "E6"],
            "Cold first request: seconds saved per change",
        ),
        (
            "restore-saved.svg",
            "restore.first_request_total",
            ["E11", "E7", "E12", "E10", "E9", "E8", "E6", "E4", "E5"],
            "Restored first request: seconds saved per change",
        ),
    ):
        items = sorted(
            ((eff[i]["short"], saved(i, key), colour(i, key)) for i in ids), key=lambda t: -t[1]
        )
        out[name] = bar_chart(
            title + f" ({PART})",
            items,
            "s",
            "Green: the experiment claims the effect beyond host spread. Grey: no measurable change.",
        )
    mem = sorted(
        (
            (
                eff[i]["short"],
                saved(i, "cold.report_max_memory_mb"),
                colour(i, "cold.report_max_memory_mb"),
            )
            for i in ("E11", "E7", "E10", "E5", "E4")
        ),
        key=lambda t: -t[1],
    )
    out["memory-saved.svg"] = bar_chart(
        f"Max Memory Used: MB saved per change ({PART}, baseline {b['cold.report_max_memory_mb']['p50']:.0f} MB)",
        mem,
        "MB",
        "REPORT Max Memory Used, cold class. E4 is its mallopt pin's saving, not the overlap's.",
        digits=0,
    )
    wp = rk["warm_paired"]
    e13 = wp["E13"]
    out["warm-saved.svg"] = bar_chart(
        f"Warm request: % faster per change ({PART}, paired by round and request)",
        [
            ("GOMP_SPINCOUNT (E13)", -wp["E13"]["mean_round_pct"], TENTATIVE),
            ("OMP passive (E13)", -wp["E13-passive"]["mean_round_pct"], TENTATIVE),
            ("F32 sgemm (E15)", -wp["E15"]["mean_round_pct"], NEUTRAL),
            ("Metadata-only clear (E12)", (1 - rk["e12_warm"]["median_ratio"]) * 100, NEUTRAL),
        ],
        "%",
        f"Amber: {e13['rounds_below_zero']}/{e13['rounds']} rounds faster, but {e13['cells_in_band']} of "
        f"{e13['cells']} cells inside control spread (inconclusive). Grey: no measurable change.",
        digits=1,
    )
    return out


def render() -> dict[str, str]:
    rk = build()
    files = {"ranking.json": json.dumps(rk, indent=1, sort_keys=True) + "\n"}
    files.update({f"charts/{k}": v for k, v in charts(rk).items()})
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if committed outputs differ from regenerated ones",
    )
    a = ap.parse_args(argv)
    stale = []
    for rel, text in render().items():
        p = HERE / rel
        if a.check:
            if not p.exists() or p.read_text() != text:
                stale.append(rel)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    if stale:
        print("stale: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
