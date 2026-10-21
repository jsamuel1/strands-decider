"""Is the multi-group fork path itself wrong, independent of n_ctx? At one n_ctx, compare one request's hidden rows
computed (a) as one fork group, (b) as two sequential fork groups, (c) one fork call per question, and (d) one
plain forward (dec_eval) per question over state + question. Offline, x86.

  python fork_groups_probe.py N_CTX APP LIB GGUF OUT.json
"""
import ctypes
import json
import os
import sys

import numpy as np

n_ctx, app, libdir, gguf, out_path = sys.argv[1:6]
n_ctx = int(n_ctx)
os.environ.update(DEC_KV="f32", DEC_UBATCH="512", N_SEQ="17", DEC_GGML_DIR=libdir)
sys.path.insert(0, app)
import engine as eng  # noqa: E402
from strands_decider.prompting import option_token_index, render_question, render_state  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402

reqs = {r["id"]: r["request"] for r in json.load(open(sys.argv[6]))["requests"]} if len(sys.argv) > 6 else None
shim = eng.Shim(os.path.join(libdir, "libdecshim.so"), gguf, 16, n_ctx, 17)
e = eng.HeadRowsEngine.from_assets(os.path.join(app, "assets"), shim.forward_shared)
res = {}
for rid, body in reqs.items():
    req = SystemOneRequest.model_validate(body)
    names = list(req.questions)
    if len(names) < 2:
        continue
    rq = [render_question(req.questions[n]) for n in names]
    s, q, offs = e.fit(render_state(req.state), [x.text for x in rq])
    rows = [list(option_token_index(o, r.option_spans, 0)) + [len(x) - 1] for r, o, x in zip(rq, offs, q, strict=True)]

    def split(h, idx):
        return np.split(h, np.cumsum([len(rows[j]) for j in idx])[:-1])

    one = split(shim._fork(s, q, rows), range(len(q)))
    half = len(q) // 2
    two = split(shim._fork(s, q[:half], rows[:half]), range(half)) + split(
        shim._fork(s, q[half:], rows[half:]), range(half, len(q))
    )
    each = [shim._fork(s, [q[i]], [rows[i]]) for i in range(len(q))]
    plain = [shim._eval(list(s) + list(q[i]), [len(s) + p for p in rows[i]]) for i in range(len(q))]
    def md(a, b):
        return float(max(np.nanmax(np.abs(x - y)) for x, y in zip(a, b, strict=True)))
    res[rid] = {
        "state": len(s),
        "q": [len(x) for x in q],
        "one_vs_two_groups": md(one, two),
        "one_vs_per_question_fork": md(one, each),
        "one_vs_plain_forward": md(one, plain),
        "plain_vs_per_question_fork": md(plain, each),
        "nan": {k: bool(any(np.isnan(x).any() for x in v)) for k, v in
                {"one": one, "two": two, "each": each, "plain": plain}.items()},
    }
    print(rid, json.dumps(res[rid]), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
