"""T0 tokenizer-only microbenchmark of the deployed tokenizer, run on a FIXED HOST (not Lambda latency).

    python tokbench.py --tokenizer tokenizer.json --corpus corpus.jsonl --out bench.json \\
        [--rounds 5] [--init-reps 10] [--iters 200] [--warmup 20] [--label <host label>]

Measures, in this order inside every round (rounds interleave init and steady-state work so a slow host period
lands on both):

  init       a fresh `python -c` child per repetition: `import tokenizers` (import), then Tokenizer.from_file
             (vocab load and construction, one call: tokenizers parses and builds in Rust), then the engine's
             no_truncation()/no_padding(). Wall and process CPU (resource.getrusage RUSAGE_SELF in the child), peak
             RSS (ru_maxrss) and the RSS delta from /proc/self/statm around each step.
  steady     in this process, for every corpus case of kind "request": HiddenStateEngine.fit on the rendered
             request (encode_batch of the questions plus encode of the state, as on Lambda) and, separately, the
             same encode calls without fit's bookkeeping; decode of the produced ids. Each case: `warmup` untimed
             calls, then `iters` timed calls with time.perf_counter_ns and time.thread_time_ns. tokens/s counts
             the ids fit returns; requests/s counts fit calls.
  memory     one tracemalloc pass (separate from the timed loop, because tracemalloc slows Python allocation) per
             case: peak traced bytes and allocated blocks for one fit; gc collections and collected objects across
             the timed loop (gc.get_stats deltas); RSS before/after.

Thread count: tokenizers parallelises encode_batch with rayon. Lambda gives the function 6 vCPUs at 10,240 MB and
the deployed engine does not set RAYON_NUM_THREADS or TOKENIZERS_PARALLELISM, so run this under `taskset -c 0-5`
(6 CPUs) to match; the CPU mask, nproc and the variables are recorded in `host`.

Run-to-run spread: each case's per-round p50s give the spread (min/max and coefficient of variation across rounds).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tokcorpus  # noqa: E402
import tokparity  # noqa: E402

INIT_CHILD = r"""
import json, os, resource, sys, time
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
out = {}
r0, c0, t0 = rss(), cpu(), time.perf_counter()
import tokenizers
t1, c1, r1 = time.perf_counter(), cpu(), rss()
tok = tokenizers.Tokenizer.from_file(sys.argv[1])
t2, c2, r2 = time.perf_counter(), cpu(), rss()
tok.no_truncation(); tok.no_padding()
t3 = time.perf_counter()
e = tok.encode("warm up", add_special_tokens=True)
t4 = time.perf_counter()
out = {"import_s": t1 - t0, "import_cpu_s": c1 - c0, "import_rss_b": r1 - r0,
       "from_file_s": t2 - t1, "from_file_cpu_s": c2 - c1, "from_file_rss_b": r2 - r1,
       "configure_s": t3 - t2, "first_encode_s": t4 - t3, "total_s": t4 - t0,
       "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "rss_after_b": rss(),
       "vocab_size": tok.get_vocab_size(with_added_tokens=True)}
print(json.dumps(out))
"""


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(v: list[float]) -> dict[str, float]:
    return {
        "n": len(v),
        "p50": pct(v, 0.5),
        "p90": pct(v, 0.9),
        "p99": pct(v, 0.99),
        "min": min(v),
        "max": max(v),
        "mean": statistics.fmean(v),
        "stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
    }


def rss_b() -> int:
    with open("/proc/self/statm", encoding="ascii") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def host_info(label: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "label": label,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "nproc_affinity": len(os.sched_getaffinity(0)),
        "cpus_allowed": sorted(os.sched_getaffinity(0)),
        "os_cpu_count": os.cpu_count(),
    }
    for k in ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS"):
        info[k] = os.environ.get(k, "unset")
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
        for key in (
            "CPU implementer",
            "CPU part",
            "CPU variant",
            "CPU revision",
            "Features",
            "model name",
        ):
            for line in cpuinfo.splitlines():
                if line.startswith(key):
                    info[key] = line.split(":", 1)[1].strip()
                    break
            else:
                info[key] = "unknown"
    except OSError:
        info["cpuinfo"] = "unknown"
    for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
        try:
            info[p] = Path(p).read_text(encoding="ascii").strip()
        except OSError:
            info[p] = "unknown"
    try:
        info["loadavg_start"] = Path("/proc/loadavg").read_text(encoding="ascii").strip()
    except OSError:
        info["loadavg_start"] = "unknown"
    import tokenizers

    info["tokenizers"] = tokenizers.__version__
    info["tokenizers_so_sha256"] = tokparity.so_sha(tokenizers)
    return info


def init_rep(tokenizer: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-c", INIT_CHILD, str(tokenizer)],
        capture_output=True,
        text=True,
        check=True,
    )
    r = json.loads(p.stdout)
    r["process_wall_s"] = time.perf_counter() - t0
    return r


def prepare(tok: Any, cases: list[dict[str, Any]]) -> list[tuple[str, str, list[str], list[str]]]:
    from strands_decider.prompting import render_question, render_state
    from strands_decider.schema import SystemOneRequest

    out = []
    for c in cases:
        if c["kind"] != "request":
            continue
        try:
            req = SystemOneRequest.model_validate(c["request"])
            st = render_state(req.state)
            qs = [render_question(req.questions[n]).text for n in req.questions]
            tok.encode(st)
            tok.encode_batch(qs)
        except Exception:
            continue
        out.append((c["id"], st, qs, c.get("tags", [])))
    return out


def steady_case(
    tok: Any, eng: Any, st: str, qs: list[str], iters: int, warmup: int
) -> dict[str, Any]:
    for _ in range(warmup):
        eng.fit(st, qs)
    g0 = [dict(x) for x in gc.get_stats()]
    wall, cpu, enc_wall, dec_wall = [], [], [], []
    ntok = 0
    for _ in range(iters):
        t0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        s, q, _ = eng.fit(st, qs)
        t1, c1 = time.perf_counter_ns(), time.thread_time_ns()
        wall.append((t1 - t0) / 1e9)
        cpu.append((c1 - c0) / 1e9)
        ntok = len(s) + sum(len(x) for x in q)
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        tok.encode_batch(qs, add_special_tokens=False)
        tok.encode(st, add_special_tokens=True)
        enc_wall.append((time.perf_counter_ns() - t0) / 1e9)
    ids = s + [i for x in q for i in x]
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        tok.decode(ids, skip_special_tokens=False)
        dec_wall.append((time.perf_counter_ns() - t0) / 1e9)
    g1 = gc.get_stats()
    return {
        "wall": wall,
        "cpu": cpu,
        "encode_only_wall": enc_wall,
        "decode_wall": dec_wall,
        "tokens": ntok,
        "gc_collections": sum(
            b["collections"] - a["collections"] for a, b in zip(g0, g1, strict=True)
        ),
        "gc_collected": sum(b["collected"] - a["collected"] for a, b in zip(g0, g1, strict=True)),
    }


def mem_case(eng: Any, st: str, qs: list[str]) -> dict[str, Any]:
    r0 = rss_b()
    tracemalloc.start()
    tracemalloc.reset_peak()
    s0 = tracemalloc.take_snapshot()
    eng.fit(st, qs)
    cur, peak = tracemalloc.get_traced_memory()
    s1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = s1.compare_to(s0, "filename")
    return {
        "traced_peak_b": peak,
        "traced_retained_b": cur,
        "alloc_blocks_delta": sum(x.count_diff for x in stats),
        "rss_delta_b": rss_b() - r0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--init-reps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument(
        "--tags",
        default="p1,length,batch",
        help="comma list: time only request cases carrying one of these tags (edge cases are parity-only)",
    )
    a = ap.parse_args()

    import tokenizers

    _head, cases = tokcorpus.load(a.corpus)
    want = set(a.tags.split(","))
    cases = [c for c in cases if want & set(c.get("tags", []))]
    tok = tokenizers.Tokenizer.from_file(str(a.tokenizer))
    eng = tokparity.engine_for(tok)
    work = prepare(tok, cases)
    res: dict[str, Any] = {
        "schema": "lkai-tok-bench/1",
        "label": "FIXED HOST, NOT LAMBDA",
        "host": host_info(a.label),
        "corpus_sha256": tokcorpus.sha256_file(a.corpus),
        "tokenizer_json_sha256": tokcorpus.sha256_file(a.tokenizer),
        "params": vars(a)
        | {"tokenizer": str(a.tokenizer), "corpus": str(a.corpus), "out": str(a.out)},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": [],
    }
    for rnd in range(a.rounds):
        r: dict[str, Any] = {"round": rnd, "init": [], "steady": {}}
        half = a.init_reps // 2
        r["init"] += [init_rep(a.tokenizer) for _ in range(half)]
        order = work if rnd % 2 == 0 else list(reversed(work))
        for cid, st, qs, _tags in order:
            r["steady"][cid] = steady_case(tok, eng, st, qs, a.iters, a.warmup)
        r["init"] += [init_rep(a.tokenizer) for _ in range(a.init_reps - half)]
        res["rounds"].append(r)
    res["memory"] = {cid: mem_case(eng, st, qs) for cid, st, qs, _ in work}
    res["process_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["summary"] = summarise(res, {cid: tags for cid, _, _, tags in work})
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"]["init"], indent=1))
    for cid, s in res["summary"]["steady"].items():
        print(
            f"{cid:34s} tok={s['tokens']:6d} p50={s['wall']['p50'] * 1e3:8.3f} ms "
            f"cv={s['round_p50_cv'] * 100:5.2f}% tok/s={s['tokens_per_s_p50']:,.0f}"
        )


def summarise(res: dict[str, Any], tags: dict[str, list[str]]) -> dict[str, Any]:
    init_keys = (
        "import_s",
        "import_cpu_s",
        "from_file_s",
        "from_file_cpu_s",
        "first_encode_s",
        "total_s",
        "process_wall_s",
        "from_file_rss_b",
        "import_rss_b",
        "maxrss_kb",
    )
    inits = [x for r in res["rounds"] for x in r["init"]]
    out: dict[str, Any] = {"init": {k: dist([x[k] for x in inits]) for k in init_keys}}
    out["init"]["round_p50_from_file_s"] = [
        pct([x["from_file_s"] for x in r["init"]], 0.5) for r in res["rounds"]
    ]
    out["steady"] = {}
    for cid in res["rounds"][0]["steady"]:
        per = [r["steady"][cid] for r in res["rounds"]]
        wall = [w for p in per for w in p["wall"]]
        rp50 = [pct(p["wall"], 0.5) for p in per]
        ntok = per[0]["tokens"]
        out["steady"][cid] = {
            "tags": tags.get(cid, []),
            "tokens": ntok,
            "wall": dist(wall),
            "cpu": dist([c for p in per for c in p["cpu"]]),
            "encode_only_wall": dist([w for p in per for w in p["encode_only_wall"]]),
            "decode_wall": dist([w for p in per for w in p["decode_wall"]]),
            "round_p50s": rp50,
            "round_p50_cv": statistics.stdev(rp50) / statistics.fmean(rp50)
            if len(rp50) > 1
            else 0.0,
            "round_p50_min": min(rp50),
            "round_p50_max": max(rp50),
            "tokens_per_s_p50": ntok / pct(wall, 0.5),
            "requests_per_s_p50": 1 / pct(wall, 0.5),
            "cpu_over_wall_p50": pct([c for p in per for c in p["cpu"]], 0.5) / pct(wall, 0.5),
            "gc_collections": sum(p["gc_collections"] for p in per),
            "gc_collected": sum(p["gc_collected"] for p in per),
            "memory": res["memory"].get(cid),
        }
    return out


if __name__ == "__main__":
    main()
