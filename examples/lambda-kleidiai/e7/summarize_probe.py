"""Summarize E7's offline probe (probe_e7.sh): fixed-host load phases, memory, mechanism and answers per arm.

  python summarize_probe.py <probe-dir> [--reference ../baseline/reference/outputs.json] [--out probe.json]

Every pair is one ctl load and one skip load in fresh containers, alternating which goes first. This is a fixed-host
proxy for mechanism and correctness, NOT Lambda latency. Answers are compared as whole objects: skip against ctl on
the same host, and both against P1's reference for the CPU part the probe read.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ARMS = ("ctl", "skip")


def load(d: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {a: [] for a in ARMS}
    for f in sorted(os.listdir(d), key=lambda x: (int(x.split("-")[0]), x)):
        if not f.endswith(".json"):
            continue
        arm = f.rsplit("-", 1)[1][:-5]
        r = json.load(open(os.path.join(d, f)))
        r["pair"] = int(f.split("-")[0])
        out[arm].append(r)
    return out


def _stat(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "p50": round(statistics.median(xs), 4),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


def summarize(runs: dict[str, list[dict]], reference: dict | None) -> dict:
    metrics = {
        "model_load_mmap_s": lambda r: r["phases_s"]["model_load_mmap"],
        "dec_load_total_s": lambda r: r["phases_s"]["dec_load_total"],
        "context_init_s": lambda r: r["phases_s"]["context_init"],
        "first_inference_s": lambda r: r["first_inference_s"],
        "vm_hwm_after_load_mb": lambda r: r["status_after_load_kb"]["VmHWM"] / 1024,
        "vm_hwm_after_requests_mb": lambda r: r["status_after_requests_kb"]["VmHWM"] / 1024,
        "rss_anon_after_load_mb": lambda r: r["status_after_load_kb"]["RssAnon"] / 1024,
        "cpu_kleidiai_buffer_mb": lambda r: r["mechanism"]["buft_bytes"]["CPU_KLEIDIAI"] / 2**20,
    }
    by_pair = {a: {r["pair"]: r for r in runs[a]} for a in ARMS}
    pairs = sorted(set(by_pair["ctl"]) & set(by_pair["skip"]))
    res: dict = {"pairs": len(pairs), "metrics": {}}
    for name, f in metrics.items():
        a = [f(r) for r in runs["ctl"]]
        b = [f(r) for r in runs["skip"]]
        d = [f(by_pair["skip"][p]) - f(by_pair["ctl"][p]) for p in pairs]
        res["metrics"][name] = {
            "ctl": _stat(a),
            "skip": _stat(b),
            "paired_delta": _stat(d),
            "pairs_below_zero": sum(x < 0 for x in d),
        }
    res["mechanism"] = {
        a: sorted({json.dumps(r["mechanism"], sort_keys=True) for r in runs[a]}) for a in ARMS
    }
    res["host"] = sorted(
        {
            json.dumps({**r["host"]["cpu"], "ggml_cpu": r["ggml_cpu"]}, sort_keys=True)
            for a in ARMS
            for r in runs[a]
        }
    )
    base = runs["ctl"][0]["answers"]
    res["answers"] = {
        "skip_vs_ctl_bit_identical": all(r["answers"] == base for a in ARMS for r in runs[a]),
        "loads_compared": sum(len(runs[a]) for a in ARMS),
    }
    if reference is not None:
        part = runs["ctl"][0]["cpu_part"]
        ref = {rid: v.get(part) for rid, v in reference.items()}
        res["answers"]["reference_cpu_part"] = part
        res["answers"]["vs_reference_bit_identical"] = {
            rid: all(r["answers"][rid] == ref[rid] for a in ARMS for r in runs[a])
            if ref[rid]
            else None
            for rid in sorted(ref)
        }
    res["latency_s_p50"] = {
        a: {rid: round(statistics.median(r["latency_s"][rid] for r in runs[a]), 4) for rid in base}
        for a in ARMS
    }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--reference")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ref = json.load(open(a.reference))["outputs"] if a.reference else None
    res = summarize(load(a.probe), ref)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
