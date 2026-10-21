"""Soak loop for profilers: repeat the tokenizer's steady-state work (or its init) so a sampling profiler or perf
stat sees enough of it. FIXED HOST profiling only; its timings are a profiler-overhead control, never latency.

    python toksoak.py --tokenizer tokenizer.json --corpus corpus.jsonl --seconds 20 --out soak.json
    python toksoak.py --tokenizer tokenizer.json --corpus corpus.jsonl --init-only 30 --out soak.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tokbench  # noqa: E402
import tokcorpus  # noqa: E402
import tokparity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--init-only", type=int, default=0)
    ap.add_argument("--tags", default="p1,length,batch")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tokenizers

    if a.init_only:
        t0 = time.perf_counter()
        for _ in range(a.init_only):
            t = tokenizers.Tokenizer.from_file(str(a.tokenizer))
            t.no_truncation()
            t.no_padding()
            del t
        dt = time.perf_counter() - t0
        a.out.write_text(
            json.dumps({"mode": "init", "n": a.init_only, "per_call_s": dt / a.init_only})
        )
        return
    want = set(a.tags.split(","))
    _h, cases = tokcorpus.load(a.corpus)
    tok = tokenizers.Tokenizer.from_file(str(a.tokenizer))
    eng = tokparity.engine_for(tok)
    work = tokbench.prepare(tok, [c for c in cases if want & set(c.get("tags", []))])
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        for _cid, st, qs, _ in work:
            eng.fit(st, qs)
        n += 1
    dt = time.perf_counter() - t0
    a.out.write_text(
        json.dumps({"mode": "steady", "passes": n, "cases": len(work), "per_pass_s": dt / n})
    )


if __name__ == "__main__":
    main()
