"""Collect the offline (NOT Lambda) profiles into one labelled summary: profiling/results/offline/offline-summary.json.

  python offline_summary.py --graviton <graviton out dir> --gpu <gpu out dir> --out <dir>

Inputs are the directories graviton_profile.sh and gpu_profile.sh write. Copies the small evidence files (JSON results,
perf report text, perf stat, nsys CSV summaries, nvidia-smi CSV, symbolised sampler tables) into --out, and writes:
  graviton.overhead_vs_plain   per tool, median steady-state forward per request vs the plain run, same host
  graviton.native_hot_spots    perf report self-time top symbols (whole run) and lkaisamp per phase
  x86 / gpu                    the same on the GPU host's EPYC CPU and its L4 GPU (llama.cpp CUDA backend, all layers)
Every block carries `evidence: "OFFLINE, NOT LAMBDA"` and its host.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics

LABEL = "OFFLINE, NOT LAMBDA"


def _j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def _med(d: dict) -> dict:
    return {k: round(statistics.median(v), 6) for k, v in sorted(d.items())}


def _run(d: dict | None) -> dict | None:
    if not d:
        return None
    return {"phases_s": d["phases_s"], "steady_state_p50_s": _med(d["steady_state_s"]),
            "passes": len(next(iter(d["steady_state_s"].values()))), "ggml_cpu": d["shim_init"]["ggml_cpu"],
            "threads": d["threads"], "ngl": d.get("ngl", 0), "host": d["host"]}


def _over(tool: dict | None, plain: dict | None) -> dict | None:
    if not tool or not plain:
        return None
    t, p = tool["steady_state_p50_s"], plain["steady_state_p50_s"]
    per = {k: round(100 * (t[k] - p[k]) / p[k], 2) for k in p if k in t}
    return {"steady_state_pct_by_request": per, "steady_state_pct_median": round(statistics.median(per.values()), 2),
            "model_load_mmap_pct": round(100 * (tool["phases_s"]["model_load_mmap"] / plain["phases_s"]["model_load_mmap"] - 1), 2)}


def _perf_self(path: str, top: int = 20) -> list[dict]:
    """Top-level lines of `perf report --no-children --sort dso,sym`: '  47.74%  lib.so  [.] symbol'."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"\s+([\d.]+)%\s+(\S+)\s+\[[.k]\]\s+(.+)$", line)
        if m:
            out.append({"pct": float(m.group(1)), "dso": m.group(2), "symbol": m.group(3).strip()[:160]})
        if len(out) >= top:
            break
    return out


def _perf_dso(path: str) -> list[dict]:
    out = []
    for line in open(path) if os.path.exists(path) else []:
        m = re.match(r"\s+([\d.]+)%\s+(\S+)", line)
        if m:
            out.append({"pct": float(m.group(1)), "dso": m.group(2)})
    return out[:10]


def _perf_stat(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    for row in csv.reader(open(path)):
        if len(row) > 2 and row[2] and not row[0].startswith("#"):
            out[row[2]] = None if row[0].startswith("<") else float(row[0])
    return out


def _csv_rows(path: str, n: int = 12) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    return [{k: (v[:120] if isinstance(v, str) else v) for k, v in r.items()} for r in rows[:n]]


def _smi(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    rows = list(csv.reader(open(path)))[1:]
    util = [float(r[1].strip(" %")) for r in rows if len(r) > 4 and r[1].strip(" %").replace(".", "").isdigit()]
    mem = [float(r[3].strip(" MiB")) for r in rows if len(r) > 4 and r[3].strip(" MiB").replace(".", "").isdigit()]
    pw = [float(r[4].strip(" W")) for r in rows if len(r) > 4 and r[4].strip(" W").replace(".", "").isdigit()]
    return {"samples_100ms": len(rows), "gpu_util_pct_max": max(util, default=None),
            "gpu_util_pct_mean": round(statistics.fmean(util), 1) if util else None,
            "samples_util_gt0": sum(u > 0 for u in util), "mem_used_mib_max": max(mem, default=None),
            "power_w_max": max(pw, default=None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graviton", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--graviton-symbolised", help="lkai_symbolise.py output for graviton sampler.samp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g, x = a.graviton, a.gpu
    keep = {g: ["host.txt", "perf-stat.csv", "perf-report-self.txt", "perf-report-self-flat.txt", "perf-report-dso.txt",
                "perf-report-children.txt", "plain.json",
                "plain-nomemfd.json", "perf-stat.json", "perf-record.json", "sampler.json", "cprofile.json",
                "pyspy.json", "pyspy-native.txt"],
            x: ["host.txt", "nvidia-smi-L.txt", "x86-cpu.json", "x86-cpu-sampled.json", "x86-cpu-pyspy.json",
                "x86-cpu.pyspy.txt", "gpu.json", "gpu-nsys.json", "gpu-ncu.json", "gpu.nvidia-smi.csv",
                "gpu-nsys_cuda_gpu_kern_sum.csv", "gpu-nsys_cuda_api_sum.csv", "gpu-nsys_cuda_gpu_mem_time_sum.csv",
                "gpu-ncu.csv"]}
    for src, files in keep.items():
        sub = os.path.join(a.out, "graviton" if src == g else "gpu-host")
        os.makedirs(sub, exist_ok=True)
        for f in files:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), sub)
    if a.graviton_symbolised:
        shutil.copy(a.graviton_symbolised, os.path.join(a.out, "graviton", "sampler-symbolised.json"))
    gr = {k: _run(_j(os.path.join(g, f"{k}.json"))) for k in
          ("plain", "plain-nomemfd", "perf-stat", "perf-record", "sampler", "cprofile", "pyspy")}
    sym = _j(a.graviton_symbolised) if a.graviton_symbolised else None
    res = {
        "schema": "lkai-offline-summary/1",
        "evidence": LABEL,
        "graviton": {
            "evidence": LABEL,
            "host": open(os.path.join(g, "host.txt")).read().strip().splitlines(),
            "what": "P1's deployed function.zip and GGUF in P1's pinned Lambda python3.12 image, 6 threads pinned to 6 "
                    "CPUs (taskset), GGUF read from local disk into a memfd (weight_read stands in for the S3 "
                    "download and is not comparable to it)",
            "runs": gr,
            "overhead_vs_plain": {k: _over(gr[k], gr["plain"]) for k in gr if k != "plain"},
            "perf_stat": _perf_stat(os.path.join(g, "perf-stat.csv")),
            "native_hot_spots_perf_self": _perf_self(os.path.join(g, "perf-report-self-flat.txt"), 25),
            "native_by_library_perf": _perf_dso(os.path.join(g, "perf-report-dso.txt")),
            "native_hot_spots_by_phase_lkaisamp": None if not sym else {
                k: {"samples": v["samples"], "top_self": v["top_self"][:12], "by_library": v["by_library"][:6]}
                for k, v in sym["phases"].items()},
        },
        "x86_cpu": {"evidence": LABEL, "host": open(os.path.join(x, "host.txt")).read().strip().splitlines()[:4],
                    "runs": {k: _run(_j(os.path.join(x, f"{k}.json"))) for k in
                             ("x86-cpu", "x86-cpu-sampled", "x86-cpu-pyspy")}},
        "gpu": {"evidence": LABEL + " (and not a GPU deployment of this example: the Lambda zip is CPU-only)",
                "host": open(os.path.join(x, "nvidia-smi-L.txt")).read().strip(),
                "what": "llama.cpp f74e2fa6a + the same patches built with GGML_CUDA (sm_89), decshim with DEC_NGL=99 "
                        "(every layer on the GPU), same GGUF, same request set",
                "runs": {k: _run(_j(os.path.join(x, f"{k}.json"))) for k in ("gpu", "gpu-nsys", "gpu-ncu")},
                "nvidia_smi_100ms": _smi(os.path.join(x, "gpu.nvidia-smi.csv")),
                "nsys_top_kernels": _csv_rows(os.path.join(x, "gpu-nsys_cuda_gpu_kern_sum.csv")),
                "nsys_cuda_api": _csv_rows(os.path.join(x, "gpu-nsys_cuda_api_sum.csv"), 8),
                "nsys_mem": _csv_rows(os.path.join(x, "gpu-nsys_cuda_gpu_mem_time_sum.csv"), 4)},
    }
    path = os.path.join(a.out, "offline-summary.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
