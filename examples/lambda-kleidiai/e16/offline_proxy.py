"""E16 offline proxy (NOT Lambda): what the main thread imports during an on-demand init, before and after E16.

  python3.12 offline_proxy.py importtime --root <task root>            # -X importtime of the main thread only
  python3.12 offline_proxy.py pairs --ctl <root> --lazy <root> --pairs 30 [--download-s 2.0]

A task root is an unzipped function.zip (handler.py, app.py, engine.py, strands_decider/, numpy/, ...). The proxy puts
a stand-in `loader` module first on sys.path, so nothing touches S3 and the model is never loaded:

- importtime: the stand-in's _client and download_to_memfd block forever. PR #77's background load therefore stops
  at its first call into loader, before E16's thread could import engine, and -X importtime lists only the modules
  the main thread imported. The child prints its sys.modules summary and exits with os._exit.
- pairs: the stand-in returns at once from _client and sleeps --download-s in download_to_memfd (the S3 download's
  place). Each child reports, from one CLOCK_MONOTONIC origin taken before `import app`:
    main_imports_s     `import app` (app.py + PR #77's handler.py, and whatever they import on this thread)
    load_start_s       the load's first call into loader (the download's start)
    engine_at_end      whether engine had been imported when the download ended (the import overlapped it)
  ctl and lazy children alternate, the first arm alternating every pair. Each child is a fresh interpreter on a
  read-only-in-practice tree with complete bytecode, so it measures imports, not compiles.

The x86 or arm64 box this runs on is not Lambda: its numbers show the direction and the module sets, not Lambda's
Init Duration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter

STUB = """
import os, sys, threading, time
_BLOCK = os.environ.get("E16_PROXY_MODE") == "importtime"
_never = threading.Event()
def _client():
    if _BLOCK:
        _never.wait()
    import builtins
    builtins._e16_load_start = time.monotonic_ns()
    return None
def download_to_memfd(bucket, key):
    import builtins
    if not hasattr(builtins, "_e16_load_start"):
        builtins._e16_load_start = time.monotonic_ns()
    if _BLOCK:
        _never.wait()
    time.sleep(float(os.environ.get("E16_PROXY_DOWNLOAD_S", "2.0")))
    t0 = builtins._e16_t0
    print("E16PROXY " + __import__("json").dumps({
        "main_imports_s": round((builtins._e16_main_done - t0) / 1e9, 6),
        "load_start_s": round((builtins._e16_load_start - t0) / 1e9, 6),
        "engine_at_end": "engine" in sys.modules,
    }), flush=True)
    os._exit(0)
"""

CHILD = """
import builtins, os, sys, time
builtins._e16_t0 = time.monotonic_ns()
import app
builtins._e16_main_done = time.monotonic_ns()
if os.environ.get("E16_PROXY_MODE") == "importtime":
    top = sorted({m.split(".")[0] for m in sys.modules})
    print("E16MODS " + __import__("json").dumps({k: sum(1 for m in sys.modules if m.split(".")[0] == k)
                                                 for k in ("numpy", "pydantic", "pydantic_core", "tokenizers",
                                                           "strands_decider", "engine", "safetensors")}),
          flush=True)
    os._exit(0)
time.sleep(60)  # the stand-in download exits the process
os._exit(2)
"""

LINE = re.compile(r"import time:\s+(\d+) \|\s+(\d+) \|( *)(\S+)")
PACKAGES = (
    "numpy",
    "pydantic",
    "pydantic_core",
    "tokenizers",
    "strands_decider",
    "engine",
    "safetensors",
)


def _env(mode: str, download_s: float) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    env.update(
        AWS_LAMBDA_INITIALIZATION_TYPE="on-demand",
        MODEL_BUCKET="proxy",
        MODEL_KEY="proxy.gguf",
        E16_PROXY_MODE=mode,
        E16_PROXY_DOWNLOAD_S=str(download_s),
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PYTHONPATH", None)
    return env


def _run(root: str, stub_dir: str, mode: str, download_s: float = 2.0, importtime: bool = False):
    args = [
        sys.executable,
        *(["-X", "importtime"] if importtime else []),
        "-c",
        f"import sys; sys.path[:0] = [{stub_dir!r}, {root!r}]\n" + CHILD,
    ]
    return subprocess.run(
        args, cwd=root, env=_env(mode, download_s), capture_output=True, text=True, timeout=120
    )


def parse_importtime(stderr: str) -> dict:
    """Cumulative and self microseconds per top-level package from -X importtime, and its module counts."""
    self_us: Counter = Counter()
    count: Counter = Counter()
    total = 0
    for line in stderr.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        own, cum, depth, name = int(m[1]), int(m[2]), len(m[3]) // 2, m[4]
        top = name.split(".")[0]
        self_us[top] += own
        count[top] += 1
        if depth == 0:
            total += cum
    return {
        "total_us": total,
        "modules": sum(count.values()),
        "packages": {k: {"self_us": self_us[k], "modules": count[k]} for k in PACKAGES if count[k]},
    }


def stub_dir() -> str:
    d = tempfile.mkdtemp(prefix="e16-proxy-")
    with open(os.path.join(d, "loader.py"), "w") as f:
        f.write(STUB)
    return d


def importtime(root: str, runs: int) -> dict:
    d = stub_dir()
    out = []
    for _ in range(runs):
        p = _run(root, d, "importtime", importtime=True)
        mods = next(
            (json.loads(x[8:]) for x in p.stdout.splitlines() if x.startswith("E16MODS ")), None
        )
        if mods is None:
            raise RuntimeError(f"importtime child failed: {p.stdout[-400:]} {p.stderr[-800:]}")
        out.append(parse_importtime(p.stderr) | {"sys_modules": mods})
    totals = [r["total_us"] for r in out]
    return {"root": root, "runs": out, "total_us_median": statistics.median(totals)}


def pairs(ctl: str, lazy: str, n: int, download_s: float) -> dict:
    d = stub_dir()
    rows = []
    for i in range(n):
        order = [("ctl", ctl), ("lazy", lazy)] if i % 2 == 0 else [("lazy", lazy), ("ctl", ctl)]
        for arm, root in order:
            p = _run(root, d, "pairs", download_s)
            rec = next(
                (json.loads(x[9:]) for x in p.stdout.splitlines() if x.startswith("E16PROXY ")),
                None,
            )
            if rec is None:
                raise RuntimeError(f"{arm} child failed: {p.stdout[-400:]} {p.stderr[-800:]}")
            rows.append({"pair": i, "arm": arm, **rec})
    summary = {}
    for key in ("main_imports_s", "load_start_s"):
        a = {r["pair"]: r[key] for r in rows if r["arm"] == "ctl"}
        b = {r["pair"]: r[key] for r in rows if r["arm"] == "lazy"}
        diffs = [b[i] - a[i] for i in sorted(a)]
        summary[key] = {
            "ctl_p50": round(statistics.median(a.values()), 6),
            "lazy_p50": round(statistics.median(b.values()), 6),
            "paired_delta_p50": round(statistics.median(diffs), 6),
            "pairs_below_zero": sum(x < 0 for x in diffs),
            "pairs": len(diffs),
        }
    summary["engine_at_download_end"] = {
        arm: sum(r["engine_at_end"] for r in rows if r["arm"] == arm) for arm in ("ctl", "lazy")
    }
    return {
        "download_s": download_s,
        "rows": rows,
        "summary": summary,
        "python": sys.version.split()[0],
        "machine": os.uname().machine,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    it = sub.add_parser("importtime")
    it.add_argument("--root", required=True)
    it.add_argument("--runs", type=int, default=5)
    pr = sub.add_parser("pairs")
    pr.add_argument("--ctl", required=True)
    pr.add_argument("--lazy", required=True)
    pr.add_argument("--pairs", type=int, default=30)
    pr.add_argument("--download-s", type=float, default=2.0)
    for p in (it, pr):
        p.add_argument("--out")
    a = ap.parse_args(argv)
    res = (
        importtime(a.root, a.runs)
        if a.cmd == "importtime"
        else pairs(a.ctl, a.lazy, a.pairs, a.download_s)
    )
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text if a.cmd == "importtime" else json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
