"""E14: what P3's Lambda samples say about the f32 CONCAT, from P3's published raw logs. No AWS calls.

  python analyse_lambda.py <P3 run dir> --out lambda-concat.json

<P3 run dir> is examples/lambda-kleidiai/profiling/results/<run-id>/ on lkai/bc9a-profiler: invocations.jsonl.gz and
logs/lkai-p3-sampler.jsonl.gz. Each sampler-arm invocation logged one `LKAI_PROF` record (lkaisamp's top-N leaf symbols
for that invocation) beside P1's `LKAI` invoke record (class, CPU part, invocation number). They are joined by
(logStreamName, invocation number). Shares are concat samples over ALL of that group's samples, and a symbol outside
an invocation's retained top-N counts as zero, so a share here is a lower bound (lkaisamp keeps 15 symbols, and concat
ranks 4th to 7th in every group below where it appears).

It splits the share by start class, CPU part (the MIDR part ID only; no processor name is inferred from it), the
environment's origin (the class of invocation 1 in that log stream), invocation index and request, and reports the
ratio of process CPU time to engine wall time (how many of the six threads were busy, on average).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

CONCAT = "ggml_compute_forward_concat"


def _json_tail(msg: str) -> dict:
    return json.loads(msg[msg.index("{") :])


def load(run: Path) -> tuple[list[dict], list[dict]]:
    with gzip.open(run / "invocations.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    with gzip.open(run / "logs" / "lkai-p3-sampler.jsonl.gz", "rt") as f:
        logs = [json.loads(line) for line in f]
    return rows, logs


def join(rows: list[dict], logs: list[dict]) -> list[dict]:
    """One record per sampler-arm invocation: class, part, n, request id, origin, samples, concat samples, cpu_s."""
    inv: dict[tuple[str, int], dict] = {}
    origin: dict[str, str] = {}
    for x in logs:
        m = x["message"]
        if m.startswith("LKAI ") and '"kind":"invoke"' in m:
            j = _json_tail(m)
            inv[(x["logStreamName"], j["inv"]["n"])] = j
            if j["inv"]["n"] == 1:
                origin[x["logStreamName"]] = j["inv"]["class"]
    by_rid = {
        r["lkai"]["request_id"]: r for r in rows if r.get("arm") == "sampler" and r.get("lkai")
    }
    out = []
    for x in logs:
        m = x["message"]
        if not (m.startswith("LKAI_PROF ") and '"kind":"invoke"' in m):
            continue
        p = _json_tail(m)
        i = inv.get((x["logStreamName"], p["n"]))
        if i is None:
            continue
        ph = p["sampler"]["phases"].get("invoke", {})
        top = ph.get("top_self", [])
        row = by_rid.get(i["request_id"])
        out.append(
            {
                "class": i["inv"]["class"],
                "cpu_part": i["inv"].get("cpu_part") or "unknown",
                "n": p["n"],
                "origin": origin.get(x["logStreamName"], "unknown"),
                "request": row["request_id"] if row else "unknown",
                "samples": int(ph.get("samples", 0)),
                "concat": sum(int(t["samples"]) for t in top if CONCAT in t["symbol"]),
                "cpu_s": float(p.get("cpu_s", 0.0)),
                "engine_s": float(i["inv"].get("engine_s") or 0.0),
                "duration_ms": (row or {}).get("report", {}).get("Duration"),
            }
        )
    return out


def group(recs: list[dict], key) -> dict[str, dict]:
    g: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
        g["|".join(str(k) for k in key(r))].append(r)
    res = {}
    for k in sorted(g):
        rs = g[k]
        s = sum(r["samples"] for r in rs)
        c = sum(r["concat"] for r in rs)
        util = [r["cpu_s"] / r["engine_s"] for r in rs if r["engine_s"] > 0]
        res[k] = {
            "invocations": len(rs),
            "samples": s,
            "concat_samples": c,
            "concat_share": round(c / s, 4) if s else None,
            "cpu_s_per_engine_s_p50": round(statistics.median(util), 3) if util else None,
        }
    return res


def analyse(run: Path) -> dict:
    rows, logs = load(run)
    recs = join(rows, logs)
    warm = [r for r in recs if r["class"] == "warm"]
    return {
        "schema": "lkai-e14-lambda-concat/1",
        "evidence": "LAMBDA, P3's published sampler-arm logs (profiler attribution, not latency)",
        "source_run": run.name,
        "records": len(recs),
        "unlinked_note": "records are LKAI_PROF lines joined to the LKAI invoke line of the same stream and n",
        "by_class_part": group(recs, lambda r: (r["class"], r["cpu_part"])),
        "warm_by_part_origin": group(warm, lambda r: (r["cpu_part"], "origin=" + r["origin"])),
        "warm_by_part_n": group(warm, lambda r: (r["cpu_part"], "n=2" if r["n"] == 2 else "n>=3")),
        "warm_by_part_request": group(warm, lambda r: (r["cpu_part"], r["request"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(json.dumps(analyse(a.run), indent=1) + "\n")


if __name__ == "__main__":
    main()
