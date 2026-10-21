"""E9 offline verdict from offline.sh's probe outputs: payload-hash parity, answer identity, fixed-host load timing.

python offline_verdict.py <offline out dir> <reference outputs.json> > offline-verdict.json
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def load(d: str, name: str) -> dict:
    return json.load(open(os.path.join(d, f"{name}.json")))


def parity(a: dict, b: dict) -> dict:
    pa, pb = a["payloads"]["per_tensor"], b["payloads"]["per_tensor"]
    diff = sorted(k for k in pa.keys() | pb.keys() if pa.get(k) != pb.get(k))
    return {
        "count": [len(pa), len(pb)],
        "digest": [a["payloads"]["digest"], b["payloads"]["digest"]],
        "identical": not diff and len(pa) > 0,
        "differing": diff[:20],
    }


def main(d: str, ref_path: str) -> None:
    ref = json.load(open(ref_path))["outputs"]
    out: dict = {"host": open(os.path.join(d, "host.txt")).read().splitlines()}
    out["parity"] = {
        "kr8_adopt_vs_runtime_repack": parity(
            load(d, "parity-repack-kr8"), load(d, "parity-adopt-kr8")
        ),
        "kr4_convert_vs_runtime_repack": parity(
            load(d, "parity-repack-kr4"), load(d, "parity-convert-kr4")
        ),
    }
    ans = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name == "offline-verdict":
            continue
        r = json.load(open(f))
        if "answers" not in r:
            continue
        cpu = r["cpu_part"]
        same = all(r["answers"][rid] == ref[rid][cpu] for rid in r["answers"])
        ans[name] = {
            "cpu_part": cpu,
            "no_i8mm": bool(r["env"]["E9_DIAG_NO_I8MM"]),
            "bit_identical_to_reference": same,
        }
    out["answers"] = ans
    t = {"base": [], "e9": []}
    for f in sorted(glob.glob(os.path.join(d, "time-*.json"))):
        arm = "e9" if "-b-e9" in f else "base"
        t[arm] += [x["model_load_mmap"] for x in json.load(open(f))["loads"]]
    out["fixed_host_model_load_mmap_s"] = {
        arm: {"n": len(v), "median": round(statistics.median(v), 3), "min": min(v), "max": max(v)}
        for arm, v in t.items()
        if v
    }
    if t["base"] and t["e9"]:
        out["fixed_host_model_load_mmap_s"]["delta_median"] = round(
            statistics.median(t["e9"]) - statistics.median(t["base"]), 3
        )
    out["note"] = (
        "fixed-host (arm64 DevSpace) numbers: one host, alternating arms, 3 loads per process; they isolate "
        "the load's CPU work and are not Lambda latency"
    )
    json.dump(out, sys.stdout, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
