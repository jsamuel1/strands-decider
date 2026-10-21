"""Compare E12's offline arms (offline_run.sh output). NOT Lambda.

  python offline_compare.py <dir> [--reference examples/lambda-kleidiai/baseline/reference/outputs.json] [--out f]

Checks, all of which must hold for "pass":
  1. every arm ran the same request sequence (same ids in the same order, same row count);
  2. position by position, every arm's answers and hidden-state hashes equal the p1 arm's (P1's own shim);
  3. within every arm, every row of one request id has the same answers and hidden-state hashes, whatever ran
     before it (shuffled orders, repeats, the wide-16q stress request): stale cache contents would break this;
  4. every answer to one of P1's fixed requests equals P1's reference/outputs.json entry for the same CPU part;
  5. the e12 arm's shim reports clear_data=0 and the dor arm's 1 (the switch is what the arm says it is).
Also reports the engine latency (PR #77's latency_ms) per request and arm, as fixed-host timing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load(path: str) -> tuple[dict, list[dict]]:
    init, rows = None, []
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == "init":
            init = r
        else:
            rows.append(r)
    if init is None:
        raise SystemExit(f"{path}: no init record")
    return init, rows


def key(r: dict) -> str:
    return json.dumps({"a": r["answers"], "h": r["hidden_sha256"]}, sort_keys=True)


def compare(runs: dict[str, tuple[dict, list[dict]]], reference: dict | None) -> dict:
    out: dict = {"runs": {}, "failures": []}
    fail = out["failures"]
    ids_seq = None
    p1_rows = [rows for name, (_, rows) in sorted(runs.items()) if name.startswith("p1-")]
    if not p1_rows:
        raise SystemExit("no p1 arm run")
    ref_rows = p1_rows[0]
    for name, (init, rows) in sorted(runs.items()):
        arm = name.split("-")[0]
        seq = [r["id"] for r in rows]
        if ids_seq is None:
            ids_seq = seq
        elif seq != ids_seq:
            fail.append(f"{name}: request sequence differs from the first run")
        cd = init.get("clear_data")
        want = {"p1": None, "dor": 1, "e12": 0}[arm]
        if cd != want:
            fail.append(f"{name}: clear_data={cd}, expected {want}")
        mism = sum(1 for a, b in zip(rows, ref_rows, strict=False) if key(a) != key(b))
        by_id: dict[str, set] = defaultdict(set)
        for r in rows:
            by_id[r["id"]].add(key(r))
        unstable = sorted(i for i, s in by_id.items() if len(s) != 1)
        if mism:
            fail.append(f"{name}: {mism} rows differ from the p1 arm at the same position")
        if unstable:
            fail.append(f"{name}: answers/hidden states vary across positions for {unstable}")
        ref_miss, ref_diff, ref_ok = 0, 0, 0
        if reference is not None:
            for r in rows:
                base = reference.get(r["id"], {})
                if not base:
                    continue
                exp = base.get(r["cpu"])
                if exp is None:
                    ref_miss += 1
                elif exp == r["answers"]:
                    ref_ok += 1
                else:
                    ref_diff += 1
            if ref_diff:
                fail.append(f"{name}: {ref_diff} answers differ from P1's reference")
        out["runs"][name] = {
            "arm": arm,
            "clear_data": cd,
            "rows": len(rows),
            "load_s": init.get("load_s"),
            "cpu": sorted({r["cpu"] for r in rows}),
            "init_info": init.get("init_info"),
            "rows_differing_from_p1": mism,
            "unstable_request_ids": unstable,
            "reference": {
                "bit_identical": ref_ok,
                "different": ref_diff,
                "no_entry_for_cpu": ref_miss,
            },
        }
    lat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, (_, rows) in runs.items():
        arm = name.split("-")[0]
        for r in rows:
            lat[r["id"]][arm].append(r["latency_ms"])
    out["latency_ms_p50"] = {
        rid: {
            arm: {"p50": round(statistics.median(v), 2), "n": len(v)}
            for arm, v in sorted(a.items())
        }
        for rid, a in sorted(lat.items())
    }
    out["requests"] = sorted(set(ids_seq or []))
    out["pass"] = not fail
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--reference")
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = {
        os.path.basename(p)[:-6]: load(p) for p in sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    }
    ref = json.load(open(a.reference))["outputs"] if a.reference else None
    res = compare(runs, ref)
    s = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
