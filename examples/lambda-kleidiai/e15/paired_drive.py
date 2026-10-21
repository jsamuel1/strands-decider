"""E15 paired in-process probe, OFFLINE, NOT LAMBDA: candidate arms toggled per request inside one engine.

  python paired_drive.py --pkg <unzipped zip with candidate libs> --gguf <gguf> --variant <libggml-cpu-*.so>
                         [--passes 10] [--arms ctl,A,B] [--gdn-rows 16] [--out paired.json]

The arms share one process, one loaded model and one KV/recurrent state allocation; only the candidate switches in
the variant library change (lkai_e15_set_a / lkai_e15_set_b from patch_candidates.py). For every pass and request
the arms run back to back in an order that rotates with the pass, so slow host drift (other tenants, thermal) hits
every arm alike and cancels in the per-request paired differences. `ctl` is the candidate library with both switches
off, which is P1's arithmetic (its answers are checked bit-identical to P1's libraries elsewhere).

Output: per arm, per request, every latency; paired deltas (arm - ctl) per request with their median and a
bootstrap 95% interval of the median; answers per arm; loadavg before and after.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import statistics
import sys
import time


def memfd_copy(path: str) -> str:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.memfd_create(b"gguf", 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            os.write(fd, chunk)
    return f"/proc/self/fd/{fd}"


def boot_ci(xs: list[float], n: int = 2000, seed: int = 15) -> list[float]:
    rnd = random.Random(seed)
    meds = sorted(statistics.median(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return [meds[int(0.025 * n)], meds[int(0.975 * n) - 1]]


def set_arm(lib, arm: str, rows: int) -> None:
    lib.lkai_e15_set_a(1 if arm == "A" else 0)
    lib.lkai_e15_set_b(1 if arm == "B" else 0, rows)


def summarise(lat: dict[str, dict[str, list[float]]], arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        if arm == "ctl":
            continue
        per_req, rel_all = {}, []
        for rid, xs in lat["ctl"].items():
            d = [a - c for a, c in zip(lat[arm][rid], xs, strict=True)]
            rel = [a / c - 1 for a, c in zip(lat[arm][rid], xs, strict=True)]
            rel_all += rel
            per_req[rid] = {
                "median_delta_s": round(statistics.median(d), 5),
                "median_rel": round(statistics.median(rel), 5),
                "ci95_rel": [round(x, 5) for x in boot_ci(rel)],
            }
        out[arm] = {
            "per_request": per_req,
            "median_rel_all": round(statistics.median(rel_all), 5),
            "ci95_rel_all": [round(x, 5) for x in boot_ci(rel_all)],
            "n_pairs": len(rel_all),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument(
        "--variant", required=True, help="the libggml-cpu-*.so this host loads (holds the switches)"
    )
    ap.add_argument("--requests", required=True)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--arms", default="ctl,A,B")
    ap.add_argument("--gdn-rows", type=int, default=16)
    ap.add_argument("--host-label", default=platform.node())
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = a.arms.split(",")
    assert arms[0] == "ctl", "ctl must be an arm"

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("DEC_KV", "f32")
    os.environ.setdefault("DEC_UBATCH", "512")
    os.environ.setdefault("N_SEQ", "17")
    sys.path.insert(0, a.pkg)
    load0 = open("/proc/loadavg").read().strip()
    eng_mod = __import__("engine")
    gguf = memfd_copy(a.gguf)
    engine, shim = eng_mod.load_engine(
        os.path.join(a.pkg, "lib", "libdecider_hidden.so"),
        gguf,
        os.path.join(a.pkg, "assets"),
        a.threads,
        max_request_tokens=24576,
    )
    var = ctypes.CDLL(a.variant)  # the handle ggml already loaded
    var.lkai_e15_set_a.argtypes = [ctypes.c_int]
    var.lkai_e15_set_b.argtypes = [ctypes.c_int, ctypes.c_int]
    reqs = json.load(open(a.requests))["requests"]
    warm = {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
    for arm in arms:
        set_arm(var, arm, a.gdn_rows)
        engine.evaluate(warm)
    lat = {arm: {r["id"]: [] for r in reqs} for arm in arms}
    answers: dict[str, dict] = {arm: {} for arm in arms}
    t0 = time.monotonic()
    for p in range(a.passes):
        for ri, r in enumerate(reqs):
            k = (p * len(reqs) + ri) % len(arms)  # rotate which arm goes first
            for arm in arms[k:] + arms[:k]:
                set_arm(var, arm, a.gdn_rows)
                t = time.monotonic()
                out = engine.evaluate(r["request"])
                lat[arm][r["id"]].append(round(time.monotonic() - t, 6))
                ans = out.model_dump()["answers"]
                prev = answers[arm].setdefault(r["id"], ans)
                if prev != ans:
                    sys.exit(f"{arm} {r['id']}: answers differ between passes")
    set_arm(var, "ctl", a.gdn_rows)
    res = {
        "schema": "lkai-e15-paired/1",
        "evidence": "OFFLINE, NOT LAMBDA: arms toggled per request in one process on another host",
        "host": {
            "label": a.host_label,
            "machine": platform.machine(),
            "kernel": platform.release(),
            "cpu_part": next(
                (
                    ln.split(":")[1].strip()
                    for ln in open("/proc/cpuinfo")
                    if ln.startswith("CPU part")
                ),
                "?",
            ),
            "affinity": sorted(os.sched_getaffinity(0)),
            "loadavg_before": load0,
            "loadavg_after": open("/proc/loadavg").read().strip(),
        },
        "shim_init": shim.init_info,
        "threads": a.threads,
        "passes": a.passes,
        "arms": arms,
        "gdn_rows": a.gdn_rows,
        "wall_s": round(time.monotonic() - t0, 3),
        "latency_s": lat,
        "paired": summarise(lat, arms),
        "answers": answers,
    }
    js = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)
    print(json.dumps(res["paired"], indent=1))


if __name__ == "__main__":
    main()
