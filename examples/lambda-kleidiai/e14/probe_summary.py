"""Summarise lkai-19b2-probe's run: thread-0 f32 CONCAT time per request IN Lambda, beside the offline probe's.

  python probe_summary.py <probe run dir> --offline <offline-summary.json> --out probe-summary.json

The probe function ran P1's concat kernel with patch_concat_probe.py's thread-0 timer. Each warm invocation logged
one `E14_PROBE` record (calls, thread-0 ns and bytes per concat shape for that invocation alone); P1's `LKAI` record
from the same invocation gives its CPU part and engine time. They are joined by Lambda request id. Per CPU part (MIDR
part ID only) and request: calls, thread-0 concat ms, its share of P1's engine time, and the offline values.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> list[dict]:
    probes: dict[str, dict] = {}
    with gzip.open(run / "logs.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)["message"]
            if m.startswith("E14_PROBE "):
                rec = json.loads(m[len("E14_PROBE ") :])
                probes[rec["request_id"]] = rec
    out = []
    with gzip.open(run / "invocations.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            lk = r.get("lkai") or {}
            p = probes.get(lk.get("request_id") or "")
            if r["wave"] != "warm" or r.get("error") or p is None:
                continue
            calls = sum(v[0] for v in p["concat"].values())
            ns = sum(v[1] for v in p["concat"].values())
            ns1 = sum(v[1] for k, v in p["concat"].items() if k.split("|")[1].split("x")[2] == "1")
            out.append(
                {
                    "request": r["request_id"],
                    "cpu_part": (lk.get("inv") or {}).get("cpu_part")
                    or lk.get("cpu_part")
                    or "unknown",
                    "engine_s": (lk.get("inv") or {}).get("engine_s"),
                    "calls": calls,
                    "thread0_ms": ns / 1e6,
                    "thread0_ms_one_sequence": ns1 / 1e6,
                    "lib": p["lib"],
                }
            )
    return out


def summarise(recs: list[dict], offline: dict, warm_rows: int) -> dict:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        g[(r["cpu_part"], r["request"])].append(r)
    table: dict = defaultdict(dict)
    for (part, q), rs in sorted(g.items()):
        eng = [r["engine_s"] for r in rs if r["engine_s"]]
        t0 = [r["thread0_ms"] for r in rs]
        o = offline["per_request"].get(q, {}).get("concat_probe") or {}
        table[part][q] = {
            "invocations": len(rs),
            "calls_per_request": statistics.median(r["calls"] for r in rs),
            "thread0_ms_p50": round(statistics.median(t0), 3),
            "thread0_ms_one_sequence_p50": round(
                statistics.median(r["thread0_ms_one_sequence"] for r in rs), 3
            ),
            "engine_ms_p50": round(1000 * statistics.median(eng), 1) if eng else None,
            "thread0_share_of_engine_p50": round(
                statistics.median(
                    r["thread0_ms"] / 1000 / r["engine_s"] for r in rs if r["engine_s"]
                ),
                4,
            )
            if eng
            else None,
            "offline_thread0_ms": o.get("thread0_ms_per_request"),
            "offline_calls_per_request": o.get("calls_per_request"),
            "offline_share_of_ctl_p50": o.get("thread0_share_of_ctl_p50"),
        }
    return {
        "schema": "lkai-e14-probe-summary/1",
        "evidence": "LAMBDA, diagnostic arm lkai-19b2-probe (P1 kernel + thread-0 timer); not a latency comparison",
        "warm_invocations_logged": len(recs),
        "warm_invocations_sent": warm_rows,
        "libs": sorted({r["lib"] for r in recs}),
        "by_part_request": dict(table),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--offline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load(a.run)
    with gzip.open(a.run / "invocations.jsonl.gz", "rt") as f:
        warm = sum(1 for line in f if json.loads(line)["wave"] == "warm")
    res = summarise(recs, json.load(open(a.offline)), warm)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    json.dump(res["by_part_request"], sys.stdout, indent=1)


if __name__ == "__main__":
    main()
