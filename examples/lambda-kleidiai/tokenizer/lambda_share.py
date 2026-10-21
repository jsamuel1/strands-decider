"""T0 Lambda share of tokenization, from already-published P1 and P3 runs (no new AWS spend).

    python lambda_share.py --p1 <P1 results.json> --p3-logs <P3 run>/logs --out lambda-share.json

Inputs (all committed on the fork):
  P1 run 20261020T080835Z (feat/lambda-kleidiai-baseline 70eab7b, examples/lambda-kleidiai/baseline/results/):
     light instrumentation only (phase timers; overhead within run-to-run spread). `assets_load` is
     tokenizer.json + head.safetensors + head_config.json (engine_load - shim_init), so it bounds tokenizer init
     from above.
  P3 run 20261020T231634Z (lkai/bc9a-profiler, profiling/results/20261020T231634Z/logs/), deployed from P1's zip
     (prof-manifest base_zip_sha256 26eb2651..., the P1 zip) with a profiler wrapper:
     cprofile arm   per-call cumulative time of Tokenizer.from_file (init) and HiddenStateEngine.fit /
                    Tokenizer.encode_batch / Tokenizer.encode (each invocation), joined to the invocation's start
                    class and CPU part by (log stream, invocation number). cProfile's measured overhead is +113 ms
                    Init (P3) and it inflates short Python calls, so these are upper bounds on the tokenizer's time.
     sampler arm    lkaisamp (SIGPROF at 1 kHz of process CPU time): samples in tokenizers.abi3.so per phase. An
                    independent, CPU-time view; blocking (I/O, waiting for restored pages) is not sampled.
Every row carries the CPU part the invocation reported; nothing is inferred about generations.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Any


def _pct(v: list[float], p: float) -> float:
    s = sorted(v)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    return {
        "n": len(v),
        "p50": _pct(v, 0.5),
        "p90": _pct(v, 0.9),
        "p99": _pct(v, 0.99),
        "min": min(v),
        "max": max(v),
        "mean": statistics.fmean(v),
    }


def _records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            yield o["logStreamName"], o["message"]


def cprofile_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(invocations, inits) from the cprofile arm's log."""
    prof: dict[tuple[str, int], dict[str, float]] = {}
    meta: dict[tuple[str, int], tuple[str, str]] = {}
    init_prof: dict[str, list[float]] = collections.defaultdict(list)
    init_meta: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for stream, m in _records(path):
        if m.startswith("LKAI_PROF "):
            r = json.loads(m[10:])
            if r["kind"] == "init":
                top = [x for v in r["top"].values() for x in v]
                init_prof[stream].append(
                    next((t["cumtime"] for t in top if "from_file" in t["fn"]), float("nan"))
                )
                continue
            d = {t["fn"]: t["cumtime"] for t in r["top"]}

            def g(key: str, d: dict[str, float] = d) -> float:
                return next((v for f, v in d.items() if key in f), 0.0)

            prof[(stream, r["n"])] = {
                "fit": g("hidden_engine.py:129:fit"),
                "encode_batch": g("'encode_batch'"),
                "encode": g("'encode' of 'tokenizers"),
                "handler": g("lkai_entry.py:137:handler"),
            }
        elif m.startswith("LKAI {"):
            r = json.loads(m[5:])
            if r["kind"] == "invoke":
                meta[(stream, r["inv"]["n"])] = (r["inv"]["class"], r["inv"]["cpu_part"])
                if r["inv"]["n"] == 1 and "init" in r:
                    init_meta[stream].append((r["inv"]["class"], r["inv"]["cpu_part"]))
            elif r["kind"] == "before_snapshot":
                init_meta[stream].append(("snapshot_init", "unknown"))
    inv = [
        {
            **v,
            "class": meta.get(k, ("unknown", "unknown"))[0],
            "cpu_part": meta.get(k, ("unknown", "unknown"))[1],
            "n": k[1],
        }
        for k, v in prof.items()
    ]
    inits = []
    for stream, ffs in init_prof.items():
        ms = init_meta.get(stream, [])
        for i, ff in enumerate(ffs):
            cls, cpu = ms[i] if i < len(ms) else ("unknown", "unknown")
            inits.append({"class": cls, "cpu_part": cpu, "from_file_s": ff})
    return inv, inits


def sampler_shares(path: Path) -> dict[str, Any]:
    meta: dict[tuple[str, int], str] = {}
    recs = []
    for stream, m in _records(path):
        if m.startswith("LKAI_PROF "):
            recs.append((stream, json.loads(m[10:])))
        elif m.startswith("LKAI {"):
            r = json.loads(m[5:])
            if r["kind"] == "invoke":
                meta[(stream, r["inv"]["n"])] = r["inv"]["class"]
    tot: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for stream, r in recs:
        cls = "init" if r["kind"] == "init" else meta.get((stream, r["n"]), "unknown")
        for pname, p in r["sampler"]["phases"].items():
            if pname == "untagged":
                continue
            tk = sum(x["samples"] for x in p.get("by_library", []) if "tokenizers" in x["library"])
            t = tot[cls]
            t[0] += tk
            t[1] += p["samples"]
            t[2] += 1
    return {
        c: {"tokenizers_samples": a, "all_samples": b, "records": n, "share": a / b if b else None}
        for c, (a, b, n) in tot.items()
    }


def p1_phase(p1: dict[str, Any], cls: str, name: str, arm: str = "inst") -> dict[str, Any] | None:
    ph = p1["phases"][arm][cls]
    if name in ph:
        return ph[name]
    return (ph.get("snapshot_init") or {}).get(name)


def analyse(p1: dict[str, Any], logs: Path) -> dict[str, Any]:
    inv, inits = cprofile_rows(logs / "lkai-p3-cprofile.jsonl.gz")
    out: dict[str, Any] = {
        "schema": "lkai-tok-lambda-share/1",
        "sources": {"p1_run": p1["run_id"], "p3_run": "20261020T231634Z"},
    }
    by = collections.defaultdict(list)
    for r in inv:
        by[(r["class"], r["cpu_part"])].append(r)
        by[(r["class"], "all")].append(r)
    out["per_request_cprofile"] = {
        f"{c}|{cpu}": {
            "fit_s": dist([x["fit"] for x in v]),
            "encode_calls_s": dist([x["encode_batch"] + x["encode"] for x in v]),
            "handler_s": dist([x["handler"] for x in v]),
            "fit_share_of_handler": dist([x["fit"] / x["handler"] for x in v if x["handler"] > 0]),
            "mean_fit_over_mean_handler": statistics.fmean(x["fit"] for x in v)
            / statistics.fmean(x["handler"] for x in v),
        }
        for (c, cpu), v in sorted(by.items())
    }
    iby = collections.defaultdict(list)
    for r in inits:
        iby[(r["class"], r["cpu_part"])].append(r["from_file_s"])
        iby[(r["class"], "all")].append(r["from_file_s"])
    out["init_from_file_cprofile"] = {f"{c}|{cpu}": dist(v) for (c, cpu), v in sorted(iby.items())}
    out["sampler_cpu_share"] = sampler_shares(logs / "lkai-p3-sampler.jsonl.gz")
    # P1 light-instrumentation context (instrumented arm, all CPU parts pooled, seconds).
    p1ctx = {
        "cold": {
            k: p1_phase(p1, "cold", k)
            for k in (
                "assets_load",
                "model_ready",
                "first_request_total",
                "report_init_duration",
                "first_request_inference",
            )
        },
        "restore": {
            k: p1_phase(p1, "restore", k)
            for k in ("first_request_total", "report_restore_duration", "first_request_inference")
        },
        "snapshot_init": {k: p1_phase(p1, "restore", k) for k in ("assets_load", "model_ready")},
        "warm": {k: p1_phase(p1, "warm", k) for k in ("steady_state_inference", "report_duration")},
    }
    out["p1_context"] = {
        c: {
            k: (None if v is None else {s: v.get(s) for s in ("n", "p50", "p90", "p99")})
            for k, v in d.items()
        }
        for c, d in p1ctx.items()
    }
    out["headroom"] = headroom(out)
    return out


def headroom(o: dict[str, Any]) -> dict[str, Any]:
    p = o["per_request_cprofile"]
    ctx = o["p1_context"]
    ff_cold = o["init_from_file_cprofile"]["cold|all"]["p50"]
    ff_snap = o["init_from_file_cprofile"].get("snapshot_init|all", {}) or {}
    return {
        "cold_first_request": {
            "tokenizer_init_s_p50_upper": ff_cold,
            "first_request_total_s_p50": ctx["cold"]["first_request_total"]["p50"],
            "share_upper": ff_cold / ctx["cold"]["first_request_total"]["p50"],
            "assets_load_s_p50_p1": ctx["cold"]["assets_load"]["p50"],
            "note": "on-demand init: PR #77 loads in a background thread after Init returns, serially after the "
            "weight download and model load, so tokenizer init is on the cold first request's critical "
            "path and outside REPORT Init Duration",
            "report_init_duration_s_p50": ctx["cold"]["report_init_duration"]["p50"],
            "share_of_report_init_duration": 0.0,
        },
        "snapshot_init": {
            "tokenizer_init_s_p50_upper": ff_snap.get("p50"),
            "note": "SnapStart: loaded at import before the snapshot; a restore does not re-run it",
        },
        "restore_first_request": {
            "fit_s": p["restore|all"]["fit_s"],
            "share_mean": p["restore|all"]["mean_fit_over_mean_handler"],
        },
        "warm": {
            "fit_s": p["warm|all"]["fit_s"],
            "share_mean": p["warm|all"]["mean_fit_over_mean_handler"],
            "share_p50": p["warm|all"]["fit_share_of_handler"]["p50"],
        },
        "cold_first_request_fit": {
            "fit_s": p["cold|all"]["fit_s"],
            "share_mean": p["cold|all"]["mean_fit_over_mean_handler"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p1", type=Path, required=True)
    ap.add_argument("--p3-logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = analyse(json.loads(a.p1.read_text(encoding="utf-8")), a.p3_logs)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["headroom"], indent=1))
    print(json.dumps(res["sampler_cpu_share"], indent=1))


if __name__ == "__main__":
    main()
