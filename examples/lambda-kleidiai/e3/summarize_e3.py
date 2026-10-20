"""Turn E3 runs into one lkai-result/1 file per download setting, each against the control. Standard library only.

  python summarize_e3.py --cold build/runs/<cold-run> [--restore build/runs/<restore-run>]
      [--verify build/runs/<verify-run>] --reference ../baseline/reference/outputs.json --out results/<run-id>

Each setting is reported separately, as the card asks: results/<run-id>/<arm>/results.json holds that setting's arm
(primary_arm) and the control arm `ctl` (P1's zip, run in the same rounds), the rows and logs of those two arms only,
and an `e3` block with:
  download     weight_download and MiB/s per arm, the delta of medians vs ctl with a bootstrap 95% interval, S3
               GET/HEAD requests per download (counted for the threads strategy, computed for CRT and P1's loader),
               retries, peak RSS right after the download (before llama.cpp maps the weights), /tmp in use
  compute      the phases after the download (model_load_mmap, context_init, assets_load, warmup_inference, cold
               first_request_inference, warm steady_state_inference), arm minus ctl within each CPU part both
               ran on, pooled, with the bootstrap interval of the pooled delta: the check that only transfer changed
  hash         the sha256 of the memfd each strategy filled (verify run), against the GGUF's sha256
  host         the host metadata the run recorded (CPU part, ggml CPU variant, threads, runtime version, log
               streams) and the fields it did not record, labelled unknown; the download delta vs ctl within each
               CPU part, and whether the status holds within the largest part alone (the host-variability check)
results/<run-id>/summary.json and summary.md tabulate every setting.

status: `improved` if the download interval is entirely below 0 with no errors, no class mismatches, bit-identical
answers and the right hash; `regressed` if entirely above 0; otherwise `no-change`. A failed arm is `failed`.
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
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import summarize as p1  # noqa: E402

EXPERIMENT_ID = "E3-parallel-range-get"
P1_STRATEGY = "threads:64:32"
P1_FETCH = "P1 loader.download_to_memfd: HEAD + 64 MiB ranged GETs on 32 threads into a memfd (threads:64:32)"
COMPUTE = (
    ("cold", "model_load_mmap"),
    ("cold", "context_init"),
    ("cold", "assets_load"),
    ("cold", "warmup_inference"),
    ("cold", "first_request_inference"),
    ("warm", "steady_state_inference"),
)
BOOT = 2000


def ok_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r["error"] and not r["class_mismatch"]]


def init(r: dict) -> dict:
    return (r.get("lkai") or {}).get("init") or {}


def ctr(r: dict) -> dict:
    return (r.get("lkai") or {}).get("counters") or {}


def metric(cls: str, name: str):
    """f(row) -> seconds for a compute phase, None when the row does not carry it."""
    if name in ("first_request_inference", "steady_state_inference"):
        return lambda r: r["latency_ms"] / 1000 if r.get("latency_ms") is not None else None
    return lambda r: init(r).get(name)


def boot_median_delta(a: list[float], b: list[float], seed: int = 0) -> dict:
    """median(a) - median(b) and a percentile-bootstrap 95% interval (resampling each side independently)."""
    rng = random.Random(seed)
    d = statistics.median(a) - statistics.median(b)
    ds = sorted(
        statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b)))
        for _ in range(BOOT)
    )
    return {
        "delta_p50_s": round(d, 6),
        "ci95_s": [round(ds[int(0.025 * BOOT)], 6), round(ds[int(0.975 * BOOT) - 1], 6)],
        "delta_pct": round(100 * d / statistics.median(b), 2) if statistics.median(b) else None,
        "n": [len(a), len(b)],
        "method": f"percentile bootstrap, {BOOT} resamples, seed {seed}",
    }


def matched(rows_a: list[dict], rows_b: list[dict], f) -> dict | None:
    """arm minus ctl within each CPU part both ran on (compute differs by CPU part), pooled by the smaller side's n.
    The pooled interval resamples within each part."""
    per_a, per_b = defaultdict(list), defaultdict(list)
    for r in rows_a:
        if (v := f(r)) is not None:
            per_a[str(r.get("cpu_part"))].append(v)
    for r in rows_b:
        if (v := f(r)) is not None:
            per_b[str(r.get("cpu_part"))].append(v)
    parts = sorted(set(per_a) & set(per_b))
    if not parts:
        return None
    out: dict = {"by_cpu_part": {}}
    wsum, wd = 0, 0.0
    for p in parts:
        a, b = per_a[p], per_b[p]
        out["by_cpu_part"][p] = boot_median_delta(a, b)
        w = min(len(a), len(b))
        wsum += w
        wd += w * out["by_cpu_part"][p]["delta_p50_s"]
    rng = random.Random(1)
    pooled = []
    for _ in range(BOOT):
        s = 0.0
        for p in parts:
            a, b = per_a[p], per_b[p]
            s += min(len(a), len(b)) * (
                statistics.median(rng.choices(a, k=len(a)))
                - statistics.median(rng.choices(b, k=len(b)))
            )
        pooled.append(s / wsum)
    pooled.sort()
    out["pooled_delta_p50_s"] = round(wd / wsum, 6)
    out["pooled_ci95_s"] = [
        round(pooled[int(0.025 * BOOT)], 6),
        round(pooled[int(0.975 * BOOT) - 1], 6),
    ]
    out["pooled_weight"] = wsum
    return out


def download_block(rows: list[dict], ctl_rows: list[dict]) -> dict:
    cold = [r for r in ok_rows(rows) if r["expected_class"] == "cold"]
    cold_c = [r for r in ok_rows(ctl_rows) if r["expected_class"] == "cold"]
    wd = [init(r)["weight_download"] for r in cold if "weight_download" in init(r)]
    wdc = [init(r)["weight_download"] for r in cold_c if "weight_download" in init(r)]
    gets = [ctr(r).get("e3_get_requests") for r in cold]
    heads = [ctr(r).get("e3_head_requests") for r in cold]
    retries = [ctr(r).get("e3_retries") for r in cold]
    per_round = defaultdict(list)
    for r in cold:
        if "weight_download" in init(r):
            per_round[r["round"]].append(init(r)["weight_download"])
    meds = [statistics.median(v) for _, v in sorted(per_round.items())]
    out = {
        "weight_download": p1.dist(wd),
        "weight_download_mib_s": p1.dist([ctr(r).get("weight_mib_s") for r in cold], unit="MiB/s"),
        "ctl_weight_download": p1.dist(wdc),
        "vs_ctl": boot_median_delta(wd, wdc) if wd and wdc else None,
        "per_round_median_s": [round(m, 4) for m in meds],
        "per_round_cv": round(statistics.stdev(meds) / statistics.fmean(meds), 4)
        if len(meds) > 1
        else None,
        "s3_get_requests_per_download": sorted({g for g in gets if g is not None}),
        "s3_head_requests_per_download": sorted({h for h in heads if h is not None}),
        "s3_requests_source": sorted({ctr(r).get("e3_requests_source") for r in cold} - {None}),
        "retries_total": sum(x for x in retries if x is not None)
        if any(x is not None for x in retries)
        else None,
        "peak_rss_mb_after_download": p1.dist(
            [ctr(r).get("e3_peak_rss_mb_after_download") for r in cold], unit="MB"
        ),
        "tmp_used_mb": p1.dist([ctr(r).get("e3_tmp_used_mb") for r in cold], unit="MB"),
        "crt_shutdown_s": p1.dist([ctr(r).get("e3_crt_shutdown_s") for r in cold]),
        "crt_shutdown_unclean": sum(
            1 for r in cold if ctr(r).get("e3_crt_shutdown_clean") is False
        ),
        "cpu_parts": dict(sorted(defaultdict(int, _count(cold)).items())),
    }
    return out


def _count(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[str(r.get("cpu_part"))] += 1
    return c


def end_to_end(rows: list[dict], ctl_rows: list[dict]) -> dict:
    """Cold first request (REPORT Init Duration + Duration) and the in-function model_ready, arm minus ctl, CPU-matched:
    the net of the download change and anything it moved downstream."""
    a = [r for r in ok_rows(rows) if r["expected_class"] == "cold"]
    b = [r for r in ok_rows(ctl_rows) if r["expected_class"] == "cold"]

    def total(r: dict) -> float:
        return (r["report"].get("Init Duration", 0) + r["report"].get("Duration", 0)) / 1000

    def restore_total(r: dict) -> float:
        return (r["report"].get("Restore Duration", 0) + r["report"].get("Duration", 0)) / 1000

    ra = [r for r in ok_rows(rows) if r["expected_class"] == "restore"]
    rb = [r for r in ok_rows(ctl_rows) if r["expected_class"] == "restore"]
    return {
        "cold.first_request_total": matched(a, b, total),
        "cold.model_ready": matched(a, b, lambda r: init(r).get("model_ready")),
        "restore.report_restore_duration": matched(
            ra, rb, lambda r: r["report"].get("Restore Duration", 0) / 1000
        )
        if ra and rb
        else None,
        "restore.first_request_total": matched(ra, rb, restore_total) if ra and rb else None,
    }


def compute_block(rows: list[dict], ctl_rows: list[dict]) -> dict:
    out = {}
    a, b = ok_rows(rows), ok_rows(ctl_rows)
    for cls, name in COMPUTE:
        m = matched(
            [r for r in a if r["expected_class"] == cls],
            [r for r in b if r["expected_class"] == cls],
            metric(cls, name),
        )
        out[f"{cls}.{name}"] = m
    out["method"] = (
        "median(arm) - median(ctl) within each CPU part both ran on, pooled by min(n); 95% bootstrap"
    )
    return out


# Host fields the initiative's HOST VARIABILITY rule asks for that this run's rows and logs do not carry. The handler
# logged only /proc/cpuinfo's "CPU part" line; nothing else about the host was captured.
HOST_UNRECORDED = (
    "cpuinfo CPU implementer",
    "cpuinfo CPU variant",
    "cpuinfo CPU revision",
    "cpuinfo Features / AT_HWCAP / AT_HWCAP2",
    "online core count (nproc)",
    "cgroup cpu.max / memory.max",
    "kernel release",
    "CPU generation / instance family",
)
_GGML_RE = re.compile(r"ggml_cpu=(\S+)")
_RUNTIME_RE = re.compile(r"INIT_START Runtime Version: (\S+)")


def _ggml_variant(r: dict) -> str:
    m = _GGML_RE.search(r.get("log_tail") or "")
    return m.group(1) if m else "unknown"


def _log_facts(log_paths: list[str]) -> dict:
    """Runtime versions and distinct log streams in the saved CloudWatch logs of the given functions."""
    versions: set[str] = set()
    streams: set[str] = set()
    for p in log_paths:
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                e = json.loads(line)
                streams.add(e.get("logStreamName"))
                if m := _RUNTIME_RE.search(e.get("message", "")):
                    versions.add(m.group(1))
    return {"runtime_versions": sorted(versions), "log_streams": len(streams - {None})}


def _cold_wd(rows: list[dict]) -> list[dict]:
    return [
        r for r in ok_rows(rows) if r["expected_class"] == "cold" and "weight_download" in init(r)
    ]


def _status_from_ci(ci: list[float]) -> str:
    return "regressed" if ci[0] > 0 else "improved" if ci[1] < 0 else "no-change"


def host_block(rows: list[dict], ctl_rows: list[dict], log_paths: list[str], status: str) -> dict:
    """What the run recorded about the hosts, and whether the download result depends on which CPU part served it.

    The download is a network transfer, so the CPU part should not move it; this checks that instead of assuming it.
    `by_cpu_part` is the arm-minus-ctl cold weight_download delta within each part both arms ran on. The status check
    recomputes the status from the largest shared part alone: if it matches the pooled status, the CPU mix did not
    decide the result. No CPU generation is inferred: a CPU part is Arm's MIDR part number, nothing more."""
    a, b = _cold_wd(rows), _cold_wd(ctl_rows)
    every = ok_rows(rows) + ok_rows(ctl_rows)

    def cells(f) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for r in every:
            c[f"{r['arm']}:{r['expected_class']}:{f(r)}"] += 1
        return dict(sorted(c.items()))

    by_part = matched(a, b, lambda r: init(r)["weight_download"])
    out: dict = {
        "recorded": {
            "cpu_part": cells(lambda r: r.get("cpu_part")),
            "ggml_cpu_variant": cells(_ggml_variant),
            "threads": sorted({(r.get("timing") or {}).get("threads") for r in every} - {None}),
            **_log_facts(log_paths),
        },
        "unknown": list(HOST_UNRECORDED),
        "cpu_part_source": "/proc/cpuinfo 'CPU part' line, logged by the handler at init (engine.py _cpu_part)",
        "download_by_cpu_part": by_part["by_cpu_part"] if by_part else None,
        "status_check": None,
    }
    if by_part:
        sizes = {p: min(v["n"]) for p, v in by_part["by_cpu_part"].items()}
        top = max(sizes, key=sizes.get)
        ci = by_part["by_cpu_part"][top]["ci95_s"]
        within = _status_from_ci(ci)
        pooled = status if status in ("improved", "regressed") else "no-change"
        out["status_check"] = {
            "largest_cpu_part": top,
            "n": by_part["by_cpu_part"][top]["n"],
            "status_within_part": within,
            "pooled_status": status,
            "holds": within == pooled,
            "method": "status rule applied to the delta within the largest CPU part both arms ran on",
        }
    return out


def hash_block(verify_rows: list[dict], arm: str, gguf_sha: str) -> dict:
    hs = [ctr(r).get("e3_weight_sha256") for r in verify_rows if r["arm"] == arm and not r["error"]]
    hs = [h for h in hs if h]
    return {
        "expected_sha256": gguf_sha,
        "observed_sha256": sorted(set(hs)),
        "samples": len(hs),
        "match": bool(hs) and all(h == gguf_sha for h in hs),
        "method": "sha256 of the filled memfd, in-function, on separate E3_VERIFY=1 cold starts",
    }


def cost_block(cond: dict, rows: list[dict], snaps: list[dict]) -> dict:
    """Lambda and S3 cost of this arm's rows, plus the per-1,000-cold-starts S3 request cost of the strategy."""
    pr = p1.PRICES
    mem_gb = cond["functions"]["ctl"]["MemorySize"] / 1024
    billed = sum(r["report"].get("Billed Duration", 0) for r in rows) / 1000
    billed_r = sum(r["report"].get("Billed Restore Duration", 0) for r in rows) / 1000
    restores = sum(1 for r in rows if "Restore Duration" in r["report"])
    colds = [r for r in rows if "Init Duration" in r["report"]]
    gets = [ctr(r).get("e3_get_requests") for r in colds if ctr(r).get("e3_get_requests")]
    heads = [ctr(r).get("e3_head_requests") or 0 for r in colds if ctr(r).get("e3_get_requests")]
    g = statistics.median(gets) if gets else None
    h = statistics.median(heads) if heads else 0
    downloads = len(colds) + len(snaps)
    size = cond["build_manifest"]["model"]["gguf_bytes"]
    c = {
        "invocations": len(rows),
        "cold_starts": len(colds),
        "restores": restores,
        "snapshot_inits": len(snaps),
        "billed_duration_s": round(billed, 3),
        "billed_restore_duration_s": round(billed_r, 3),
        "lambda_duration_usd": round((billed + billed_r) * mem_gb * pr["arm64_gb_s"], 4),
        "lambda_requests_usd": round(len(rows) * pr["request"], 6),
        "snapstart_restore_usd": round(restores * mem_gb * pr["snapstart_restore_gb"], 4),
        "s3_get_requests_per_download": g,
        "s3_head_requests_per_download": h,
        "weight_downloads": downloads,
        "s3_request_usd": round(
            downloads * ((g or 0) * pr["s3_get_per_1000"] + h * pr["s3_head_per_1000"]) / 1000, 6
        ),
        "s3_request_usd_per_1000_cold_starts": round(
            ((g or 0) * pr["s3_get_per_1000"] + h * pr["s3_head_per_1000"]), 6
        ),
        "s3_gb_transferred": round(downloads * size / 1e9, 2),
        "s3_transfer_usd": 0.0,
        "s3_transfer_note": "S3 to Lambda in the same region: no data transfer charge",
        "prices": pr,
    }
    c["total_usd"] = round(
        sum(v for k, v in c.items() if k.endswith("_usd") and "per_1000" not in k), 4
    )
    return c


def status_of(e3: dict, errors: int, mism: int, corr: dict | None) -> str:
    vs = e3["download"]["vs_ctl"]
    if vs is None:
        return "failed"
    clean = (
        errors == 0
        and mism == 0
        and corr is not None
        and corr["bit_identical"] == corr["compared"] > 0
        and e3["hash"]["match"]
    )
    lo, hi = vs["ci95_s"]
    if hi < 0 and clean:
        return "improved"
    if lo > 0:
        return "regressed"
    return "no-change" if clean else "inconclusive"


def subset_run(src: str, dst: str, arms: set[str], fn_names: set[str]) -> tuple[dict, list[dict]]:
    """Copy one run's evidence for `arms` only: invocations.jsonl rows, conditions.json, logs/<function>.jsonl.gz."""
    cond, rows = p1.load_run(src)
    os.makedirs(os.path.join(dst, "logs"), exist_ok=True)
    keep = [r for r in rows if r["arm"] in arms]
    with open(os.path.join(dst, "invocations.jsonl"), "a") as f:
        for r in keep:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    for name in fn_names:
        p = os.path.join(src, "logs", f"{name}.jsonl.gz")
        if os.path.exists(p):
            with (
                gzip.open(p, "rb") as i,
                gzip.open(os.path.join(dst, "logs", f"{name}.jsonl.gz"), "ab") as o,
            ):
                shutil.copyfileobj(i, o)
    return cond, keep


def summarize_arm(arm: str, a: argparse.Namespace, meta: dict) -> dict:
    out = os.path.join(a.out, arm)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    cond, _ = p1.load_run(a.cold)
    fns = {x: cond["functions"][x]["FunctionName"] for x in ("ctl", arm)}
    cold_cond, cold_rows = subset_run(a.cold, out, {"ctl", arm}, set(fns.values()))
    rows = list(cold_rows)
    runs = {"cold": os.path.basename(os.path.normpath(a.cold))}
    snaps: dict[str, list[dict]] = {}
    restore_cond = None
    if a.restore:
        rc, _ = p1.load_run(a.restore)
        if arm in rc["arms"] or arm == "ctl":
            restore_cond, rrows = subset_run(a.restore, out, {"ctl", arm}, set(fns.values()))
            rows += rrows
            runs["restore"] = os.path.basename(os.path.normpath(a.restore))
            versions = defaultdict(set)
            for m in rc.get("rounds_meta", []):
                for x, pub in m.get("published", {}).items():
                    versions[rc["functions"][x]["FunctionName"]].add(str(pub["version"]))
            snaps = p1.snapshot_inits(a.restore, versions)
    verify_rows = p1.load_run(a.verify)[1] if a.verify else []
    c2 = dict(cold_cond)
    c2["arms"] = ["ctl", arm] if arm != "ctl" else ["ctl"]
    c2["functions"] = {x: cold_cond["functions"][x] for x in c2["arms"]}
    c2["restore_run"] = (
        {
            k: restore_cond[k]
            for k in ("rounds", "concurrency", "arms", "started", "finished", "rounds_meta")
        }
        if restore_cond
        else None
    )
    json.dump(c2, open(os.path.join(out, "conditions.json"), "w"), indent=1)

    by_arm = {x: [r for r in rows if r["arm"] == x] for x in c2["arms"]}
    phases = {x: p1.arm_phases(by_arm[x], snaps.get(fns[x], [])) for x in c2["arms"]}
    m = cold_cond["build_manifest"]
    gguf = m["model"]["gguf_sha256"]
    e3 = {
        "strategy": cold_cond["e3_dl"].get(arm) or P1_FETCH,
        "download": download_block(by_arm[arm], by_arm["ctl"]),
        "compute": compute_block(by_arm[arm], by_arm["ctl"]) if arm != "ctl" else None,
        "end_to_end": end_to_end(by_arm[arm], by_arm["ctl"]) if arm != "ctl" else None,
        "hash": hash_block(verify_rows, arm, gguf)
        if arm != "ctl"
        else {
            "expected_sha256": gguf,
            "match": True,
            "method": "P1 measured weights (build manifest)",
        },
        "restore_measured": arm in (restore_cond or {}).get("arms", []),
        "runs": runs,
    }
    ref = json.load(open(a.reference))
    corr = p1.compare(by_arm[arm], ref["outputs"]) | {
        "reference": "baseline/reference/outputs.json"
    }
    errs = [
        {
            "arm": r["arm"],
            "round": r["round"],
            "wave": r["wave"],
            "error": r["error"],
            "detail": str(r.get("payload"))[:300],
        }
        for r in rows
        if r["error"]
    ]
    cls = p1.classify(by_arm[arm])
    f0 = cold_cond["functions"][arm]
    spec = cold_cond["e3_dl"].get(arm)
    st = status_of(e3, sum(1 for r in by_arm[arm] if r["error"]), cls["mismatches"], corr)
    e3["host"] = host_block(
        by_arm[arm],
        by_arm["ctl"],
        [os.path.join(out, "logs", f"{n}.jsonl.gz") for n in sorted(set(fns.values()))],
        st,
    )
    res = {
        "schema": "lkai-result/1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{a.run_id}-{arm}",
        "status": st,
        "primary_arm": arm,
        "hypothesis": meta["hypotheses"].get(arm, meta["hypothesis"]),
        "changed_variable": {
            "name": "weight_download_strategy",
            "baseline": P1_STRATEGY + " (" + P1_FETCH + ")",
            "experiment": spec or "P1 zip and loader (control)",
        },
        "commits": {
            "experiment": m["repo_commit"],
            "baseline": a.baseline_commit,
            "repo_dirty_at_build": m["repo_dirty"],
        },
        "deployment": {
            "region": cold_cond["region"],
            "functions": {x: cold_cond["functions"][x]["FunctionName"] for x in c2["arms"]},
            "runtime": f0["Runtime"],
            "runtime_version_arn": cold_cond.get("runtime_version_arn"),
            "architecture": f0["Architectures"][0],
            "memory_mb": f0["MemorySize"],
            "ephemeral_storage_mb": f0["EphemeralStorage"]["Size"],
            "timeout_s": f0["Timeout"],
            "snapstart": f0["SnapStart"],
            "handler": f0["Handler"],
            "code_sha256": {
                "ctl": cold_cond["p1_zip_sha256"],
                arm: m["function_zip"]["sha256"] if spec else cold_cond["p1_zip_sha256"],
            },
            "environment": f0["environment"],
            "az": "not exposed to the function; Lambda chooses per environment",
            "cpu_parts_observed": _cpu_cells(rows),
        },
        "model": m["model"] | {"llama_cpp": m["llama_cpp"]},
        "build": {
            "images": m["images"],
            "toolchain": m["toolchain"],
            "assets_sha256": m["assets_sha256"],
            "e3_zip": m["function_zip"],
            "awscrt_wheel": m["awscrt_wheel"],
            "e3_sources_sha256": m["e3_sources_sha256"],
        },
        "conditions": {
            "rounds": cold_cond["rounds"],
            "concurrency": cold_cond["concurrency"],
            "warm_per_arm_per_round": cold_cond["warm_per_arm_per_round"],
            "arms": c2["arms"],
            "all_arms_in_run": cold_cond["arms"],
            "probe": cold_cond["probe"],
            "requests_file": cold_cond["requests_file"],
            "started": cold_cond["started"],
            "finished": cold_cond.get("finished"),
            "request_ids": [r["id"] for r in cold_cond["requests"]["requests"]],
            "client": cold_cond["client"],
            "weight_fetch": spec or P1_FETCH,
            "interleaving": "every arm in every round; arm order rotated by one per round",
            "restore_run": c2["restore_run"],
        },
        "classification": {"method": "P1's (expected class vs REPORT)", **cls},
        "phases": phases,
        "variance": {x: p1.round_medians(by_arm[x]) for x in c2["arms"]},
        "instrumentation_overhead": {
            "method": "not re-measured in E3: every arm, ctl included, runs P1's instrumented entry "
            "(LKAI_PHASES=1). P1 measured the overhead at +0.46% cold first request."
        },
        "e3": e3,
        "errors": {"count": len(errs), "rows": errs},
        "cost": cost_block(cold_cond, by_arm[arm], snaps.get(fns[arm], [])),
        "correctness": corr,
        "profiling_artifacts": [],
        "artifacts": {
            "invocations": "invocations.jsonl",
            "logs": "logs/",
            "conditions": "conditions.json",
        },
        "reproduce": meta["reproduce"],
        "conclusion": None,
    }
    if not phases[arm].get("restore"):
        # The validator wants a restore class in every non-failed run. This arm's own restore was not measured
        # (the operator chose a restore leg for the best arm only); the control's restore is in phases.ctl if any.
        phases[arm]["restore"] = {
            "n": 0,
            "report_restore_duration": None,
            "first_request_total": None,
        }
        e3["restore_note"] = (
            "restore not measured for this setting (operator: restore leg for the best arm and ctl only). "
            "With SnapStart the weights download only in the snapshot init inside publish_version."
        )
    res["conclusion"] = conclude(arm, res)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    return res


def _cpu_cells(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[f"{r['arm']}:{r['expected_class']}:{r.get('cpu_part')}"] += 1
    return dict(c)


def moved_phases(comp: dict) -> dict[str, float]:
    """Compute phases whose pooled 95% interval excludes 0, with the pooled delta (seconds)."""
    return {
        k: v["pooled_delta_p50_s"]
        for k, v in comp.items()
        if isinstance(v, dict) and (v["pooled_ci95_s"][0] > 0 or v["pooled_ci95_s"][1] < 0)
    }


def conclude(arm: str, res: dict) -> str:
    e3 = res["e3"]
    d = e3["download"]
    vs = d["vs_ctl"]
    if vs is None:
        return f"{arm}: no successful cold download measured; see errors."
    s = (
        f"{e3['strategy']}: cold weight_download p50 {d['weight_download']['p50']:.2f} s "
        f"(n={d['weight_download']['n']}) vs control {d['ctl_weight_download']['p50']:.2f} s "
        f"(n={d['ctl_weight_download']['n']}), delta {vs['delta_p50_s']:+.2f} s "
        f"({vs['delta_pct']:+.1f}%), 95% CI [{vs['ci95_s'][0]:+.2f}, {vs['ci95_s'][1]:+.2f}] s. "
    )
    comp = e3.get("compute") or {}
    moved = moved_phases(comp)
    if arm != "ctl":
        s += (
            "Compute phases after the download: no interval excludes 0. "
            if not moved
            else "Compute phases whose 95% interval excludes 0 (pooled delta; 6 phases x 9 settings are "
            "tested, so about 3 such flags are expected by chance): "
            + ", ".join(f"{k} {v:+.3f} s" for k, v in moved.items())
            + ". "
        )
    chk = (e3.get("host") or {}).get("status_check")
    if chk:
        s += (
            f"Within CPU part {chk['largest_cpu_part']} alone (n={chk['n'][0]} vs {chk['n'][1]}) the status is "
            f"{chk['status_within_part']}, so the CPU mix {'does not decide' if chk['holds'] else 'changes'} "
            "this result. "
        )
    c = res["correctness"]
    s += (
        f"Answers {c['bit_identical']}/{c['compared']} bit-identical; weights hash "
        f"{'matches' if e3['hash']['match'] else 'NOT verified'}; S3 GETs per download "
        f"{d['s3_get_requests_per_download']}; peak RSS after download p50 "
        f"{(d['peak_rss_mb_after_download'] or {}).get('p50')} MB; /tmp used p50 "
        f"{(d['tmp_used_mb'] or {}).get('p50')} MB. Status {res['status']}."
    )
    return s


def _p50(d: dict | None) -> float | None:
    return d["p50"] if d else None


def _fmt(v: float | None, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def table(results: dict[str, dict]) -> str:
    lines = [
        "| arm | strategy | status | cold download p50 s (n) | vs ctl p50 s [95% CI] | MiB/s p50 | GETs | "
        "peak RSS after dl MB | /tmp MB | hash | answers identical | compute moved | status within largest CPU part |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, r in results.items():
        d, e3 = r["e3"]["download"], r["e3"]
        vs = d["vs_ctl"] or {}
        comp = e3.get("compute") or {}
        moved = [f"{k} {v:+.3f}s" for k, v in moved_phases(comp).items()]
        c = r["correctness"]
        chk = (e3.get("host") or {}).get("status_check")
        within = (
            f"{chk['status_within_part']} ({chk['largest_cpu_part']}, n={chk['n'][0]})"
            if chk
            else "—"
        )
        lines.append(
            f"| {arm} | `{e3['strategy'] if arm != 'ctl' else 'P1 loader (P1 zip)'}` | {r['status']} | "
            f"{_fmt(_p50(d['weight_download']))} ({(d['weight_download'] or {}).get('n', 0)}) | "
            + (
                f"{vs['delta_p50_s']:+.2f} [{vs['ci95_s'][0]:+.2f}, {vs['ci95_s'][1]:+.2f}]"
                if vs
                else "—"
            )
            + f" | {_fmt(_p50(d['weight_download_mib_s']), 1)} | {d['s3_get_requests_per_download']} | "
            f"{(d['peak_rss_mb_after_download'] or {}).get('p50')} | {(d['tmp_used_mb'] or {}).get('p50')} | "
            f"{'ok' if e3['hash']['match'] else 'NO'} | {c['bit_identical']}/{c['compared']} | "
            f"{', '.join(moved) if moved else ('none' if arm != 'ctl' else '—')} | {within} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", required=True)
    ap.add_argument("--restore")
    ap.add_argument("--verify")
    ap.add_argument("--reference", default=os.path.join(BASE, "reference", "outputs.json"))
    ap.add_argument("--meta", default=os.path.join(HERE, "meta.json"))
    ap.add_argument("--baseline-commit", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    cond, _ = p1.load_run(a.cold)
    results = {arm: summarize_arm(arm, a, meta) for arm in cond["arms"] if arm != "ctl"}
    summary = {
        "schema": "lkai-e3-summary/1",
        "run_id": a.run_id,
        "settings": {
            arm: {
                "status": r["status"],
                "strategy": r["e3"]["strategy"],
                "download": {
                    k: r["e3"]["download"][k] for k in ("vs_ctl", "s3_get_requests_per_download")
                }
                | {
                    "p50_s": _p50(r["e3"]["download"]["weight_download"]),
                    "ctl_p50_s": _p50(r["e3"]["download"]["ctl_weight_download"]),
                    "mib_s_p50": _p50(r["e3"]["download"]["weight_download_mib_s"]),
                    "peak_rss_mb_p50": (
                        r["e3"]["download"]["peak_rss_mb_after_download"] or {}
                    ).get("p50"),
                },
                "hash_match": r["e3"]["hash"]["match"],
                "host_status_check": r["e3"]["host"]["status_check"],
                "correctness": {
                    k: r["correctness"][k] for k in ("compared", "bit_identical", "choice_flips")
                },
                "result": f"{arm}/results.json",
                "conclusion": r["conclusion"],
            }
            for arm, r in results.items()
        },
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "summary.md"), "w").write(table(results) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()
