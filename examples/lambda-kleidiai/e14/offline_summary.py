"""Summarise an e14/offline.sh output directory (OFFLINE, NOT LAMBDA) into offline-summary.json.

  python offline_summary.py <offline out dir> --out offline-summary.json

Reads ctl<r>.json, rows<r>.json, probe<r>.json, host.txt, loadavg.txt, libs.sha256 and inputs.sha256. Reports:
- bit identity: every raw hidden-state SHA-256 of every pass and run against ctl round 1's;
- per request and round, median(rows) / median(ctl) - 1 (wall time of engine.evaluate);
- from the probe runs: f32 CONCAT calls, bytes and thread-0 nanoseconds per request, split by whether the ubatch had
  one sequence (dst ne2 = 1, where the unpatched kernel uses one thread) or several.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics


def _runs(d: str, arm: str) -> dict[int, dict]:
    out = {}
    for p in glob.glob(os.path.join(d, f"{arm}[0-9]*.json")):
        out[int(re.search(r"(\d+)\.json$", p).group(1))] = json.load(open(p))
    return dict(sorted(out.items()))


def summarise(d: str) -> dict:
    ctl, rows, probe = _runs(d, "ctl"), _runs(d, "rows"), _runs(d, "probe")
    if not ctl or set(ctl) != set(rows):
        raise SystemExit(f"{d}: ctl and rows rounds differ: {sorted(ctl)} vs {sorted(rows)}")
    first = ctl[min(ctl)]
    reqs = list(first["steady_state_s"])
    ref = {q: first["hidden_sha256"][q][0] for q in reqs}
    every = [*ctl.values(), *rows.values(), *probe.values()]
    bit = {
        "runs": len(every),
        "hidden_rows_hashed": sum(len(r["hidden_sha256"][q]) for r in every for q in reqs),
        "all_equal_to_ctl_round1": all(
            set(r["hidden_sha256"][q]) == {ref[q]} for r in every for q in reqs
        ),
        "answers_equal": all(r["answers"] == first["answers"] for r in every),
    }
    per_req = {}
    for q in reqs:
        rounds = {
            str(k): round(
                100
                * (
                    statistics.median(rows[k]["steady_state_s"][q])
                    / statistics.median(ctl[k]["steady_state_s"][q])
                    - 1
                ),
                3,
            )
            for k in ctl
        }
        c = [x for k in ctl for x in ctl[k]["steady_state_s"][q]]
        w = [x for k in rows for x in rows[k]["steady_state_s"][q]]
        e: dict = {
            "n": [len(c), len(w)],
            "ctl_p50_s": round(statistics.median(c), 6),
            "rows_p50_s": round(statistics.median(w), 6),
            "pooled_delta_pct": round(100 * (statistics.median(w) / statistics.median(c) - 1), 3),
            "per_round_delta_pct": rounds,
        }
        if probe:
            n = sum(len(p["steady_state_s"][q]) for p in probe.values())
            calls = ns = by = ns1 = calls1 = 0
            shapes: dict[str, int] = {}
            for p in probe.values():
                for k, (c_, t, b) in (p.get("concat_probe") or {}).get(q, {}).items():
                    calls += c_
                    ns += t
                    by += b
                    dst = re.search(r"dst=([0-9x]+)", k).group(1)
                    shapes[dst] = shapes.get(dst, 0) + c_
                    if dst.split("x")[2] == "1":
                        ns1 += t
                        calls1 += c_
            e["concat_probe"] = {
                "requests_probed": n,
                "calls_per_request": round(calls / n, 2),
                "one_sequence_calls_per_request": round(calls1 / n, 2),
                "dst_bytes_per_request": round(by / n),
                "thread0_ms_per_request": round(ns / n / 1e6, 3),
                "thread0_ms_per_request_one_sequence": round(ns1 / n / 1e6, 3),
                "thread0_share_of_ctl_p50": round(ns / n / 1e9 / e["ctl_p50_s"], 4),
                "dst_shapes_calls_per_request": {
                    k: round(v / n, 2) for k, v in sorted(shapes.items())
                },
            }
        per_req[q] = e
    tot = {
        str(k): {
            "ctl_s": ctl[k]["steady_state_total_s"],
            "rows_s": rows[k]["steady_state_total_s"],
            "ctl_cpu_over_wall": round(
                ctl[k]["steady_state_cpu_s"] / ctl[k]["steady_state_total_s"], 3
            ),
            "rows_cpu_over_wall": round(
                rows[k]["steady_state_cpu_s"] / rows[k]["steady_state_total_s"], 3
            ),
        }
        for k in ctl
    }

    def _txt(name: str) -> str | None:
        p = os.path.join(d, name)
        return open(p).read() if os.path.exists(p) else None

    return {
        "schema": "lkai-e14-offline-summary/1",
        "evidence": "OFFLINE, NOT LAMBDA",
        "host": _txt("host.txt"),
        "loadavg": _txt("loadavg.txt"),
        "libs_sha256": _txt("libs.sha256"),
        "inputs_sha256": _txt("inputs.sha256"),
        "toolchain": _txt("toolchain.txt"),
        "shim_init": first["shim_init"],
        "affinity": first["host"]["affinity"],
        "passes_per_run": len(first["steady_state_s"][reqs[0]]),
        "rounds": sorted(ctl),
        "bit_identity": bit,
        "per_request": per_req,
        "steady_state_totals": tot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(summarise(a.dir), f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
