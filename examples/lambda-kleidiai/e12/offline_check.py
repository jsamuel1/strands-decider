"""E12 offline correctness and fixed-host timing check (NOT Lambda). Runs inside the pinned Lambda python3.12 image.

  python3.12 offline_check.py --task /pkg --model s2b-h512-Q8_0.gguf --orders 6 --repeats 3 --seed 12 --out rows.jsonl

/pkg is P1's unzipped function.zip, with lib/libdecshim.so replaced by E12's shim (built from shim/decshim.c). The
arm is chosen by the environment, exactly as on Lambda: DEC_CLEAR_DATA unset (or "1") is the baseline full clear,
DEC_CLEAR_DATA=0 the metadata-only clear. One process is one arm; run.sh compares the arms' rows afterwards.

The process imports PR #77's handler as Lambda does outside on-demand init (the model loads at import, then the
handler's warm-up request), then sends every request of the request set in `--orders` shuffled orders, each request
`--repeats` times in a row. The request set is P1's five fixed requests plus E12's stress requests (requests.json in
this directory): a long state with sixteen questions (the widest fork, 17 sequences), then short single-question
requests, so a short request runs right after a long one has written far more cache cells than it uses.

Each row records the full answer object, the SHA-256 of every hidden-state array the torso returned (the bytes the
pointer head reads, before any rounding), the engine's own latency, and the shim's clear mode.
"""

# ruff: noqa: SLF001
# (This driver reads PR #77's handler's module-private load future, as baseline/lkai_entry.py does.)
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time


def load_requests(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        doc = json.load(open(p))
        out.extend(doc["requests"])
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate request ids in {paths}")
    return out


def orders(ids: list[str], n: int, seed: int) -> list[list[str]]:
    """Order 0 is the request file order; orders 1..n-1 are seeded shuffles. Deterministic for a given seed."""
    rng = random.Random(seed)
    out = [list(ids)]
    for _ in range(n - 1):
        o = list(ids)
        rng.shuffle(o)
        out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--orders", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["MODEL_PATH"] = a.model
    sys.path.insert(0, a.task)
    t0 = time.monotonic()
    import app
    import engine as eng_mod
    import numpy as np

    load_s = time.monotonic() - t0
    engine, _ = app._pr77._ENGINE.result()
    shim = eng_mod.SHIM
    clear = (
        shim.lib.dec_clear_data(ctypes_handle(shim))
        if hasattr(shim.lib, "dec_clear_data")
        else None
    )

    captured: list[str] = []
    inner = engine.forward_shared

    def forward(state_ids, q_ids, rows=None):
        hs = inner(state_ids, q_ids, rows)
        captured.extend(hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest() for h in hs)
        return hs

    engine.forward_shared = forward
    reqs = {r["id"]: r["request"] for r in load_requests(a.requests)}
    ids = list(reqs)
    with open(a.out, "w") as f:
        f.write(
            json.dumps(
                {
                    "kind": "init",
                    "load_s": round(load_s, 3),
                    "clear_data": clear,
                    "env": {
                        k: os.environ.get(k)
                        for k in (
                            "DEC_CLEAR_DATA",
                            "DEC_KV",
                            "DEC_UBATCH",
                            "N_SEQ",
                            "THREADS",
                            "OMP_NUM_THREADS",
                        )
                    },
                    "init_info": shim.init_info,
                }
            )
            + "\n"
        )
        seq = 0
        for oi, order in enumerate(orders(ids, a.orders, a.seed)):
            for rid in order:
                for rep in range(a.repeats):
                    captured.clear()
                    out = app.handler(reqs[rid], None)
                    f.write(
                        json.dumps(
                            {
                                "kind": "row",
                                "seq": seq,
                                "order": oi,
                                "rep": rep,
                                "id": rid,
                                "answers": out.get("answers"),
                                "hidden_sha256": list(captured),
                                "latency_ms": out.get("latency_ms"),
                                "cpu": out["timing"]["cpu"],
                                "input_tokens": out["usage"]["input_tokens"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    seq += 1
    print("OFFLINE_CHECK_DONE", a.out, seq, "rows", flush=True)


def ctypes_handle(shim):
    import ctypes

    return ctypes.c_void_p(shim.h)


if __name__ == "__main__":
    main()
