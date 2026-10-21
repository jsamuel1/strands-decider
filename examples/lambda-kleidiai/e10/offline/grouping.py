"""Fork grouping per n_ctx for a request set: snapstart/engine.py Shim.forward_shared's greedy rule, on token counts.

A request's answers can be bit-identical to n_ctx=16384's only if every n_ctx forms the same fork groups (different
groups change the batch, hence floating-point summation order). Prints, per request and n_ctx, the group sizes or
"refused" (state + longest question > n_ctx).

  python grouping.py TOKENIZER_DIR REQUESTS.json OUT.json   (REQUESTS: {id: request}, or P1's requests.json)
"""

import json
import sys

from tokenizers import Tokenizer

from strands_decider.hidden_engine import HiddenEngineConfig, HiddenStateEngine
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import SystemOneRequest

SIZES = (512, 1024, 2048, 4096, 8192, 16384)
N_SEQ = 17


def need(eng: HiddenStateEngine, body: dict) -> tuple[int, list[int]]:
    req = SystemOneRequest.model_validate(body)
    rq = [render_question(req.questions[n]) for n in req.questions]
    s, q, _ = eng.fit(render_state(req.state), [r.text for r in rq])
    return len(s), [len(x) for x in q]


def groups(n_s: int, ql: list[int], n_ctx: int) -> list[int] | None:
    if len(ql) == 1:
        return [1] if n_s + ql[0] <= n_ctx else None
    out, g, i = [], [], 0
    while i <= len(ql):
        fits = i < len(ql) and len(g) < N_SEQ - 1 and n_s + sum(ql[j] for j in g) + ql[i] <= n_ctx
        if fits:
            g.append(i)
            i += 1
            continue
        if not g:
            return None
        out.append(len(g))
        g = []
        if i == len(ql):
            break
    return out


if __name__ == "__main__":
    tok_dir, req_path, out_path = sys.argv[1:4]
    eng = HiddenStateEngine(
        Tokenizer.from_file(tok_dir + "/tokenizer.json"), None, HiddenEngineConfig(4096), None
    )
    src = json.load(open(req_path))
    reqs = {r["id"]: r["request"] for r in src["requests"]} if "requests" in src else src
    res = {}
    for rid, body in reqs.items():
        s, ql = need(eng, body)
        g = {str(n): groups(s, ql, n) for n in SIZES}
        res[rid] = {
            "state": s,
            "questions": len(ql),
            "groups": g,
            "same_groups_as_16384": [n for n in SIZES if g[str(n)] == g["16384"]],
        }
        print(rid, s, len(ql), g)
    json.dump(res, open(out_path, "w"), indent=1)
