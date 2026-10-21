"""T0 tokenizer parity: record golden outputs of the deployed tokenizer, and check any implementation against them.

    python tokparity.py make  --corpus corpus.jsonl --tokenizer tokenizer.json --out golden.jsonl
    python tokparity.py check --corpus corpus.jsonl --tokenizer tokenizer.json --golden golden.jsonl \\
        [--impl module:factory]

`--impl` names a factory called as factory(tokenizer_json_path) that returns an object with the subset of the
`tokenizers.Tokenizer` API that strands_decider.hidden_engine uses: encode(text, add_special_tokens=...) and
encode_batch(texts, add_special_tokens=...) returning objects with `.ids` and `.offsets`, enable_truncation(
max_length=...), no_truncation(), no_padding() and decode(ids, skip_special_tokens=...). The default is the deployed
one, `tokenizers:Tokenizer.from_file`. T1-T3 run `check` with their candidate; it must report 0 mismatches.

What each case records (all from the deployed code path, nothing re-implemented):
  encode    ids, offsets, and decode(ids) with special tokens kept, plus whether that equals the input text
  batch     the same per text, from one encode_batch call
  request   SystemOneRequest validation, then exactly the engine's pre-forward path: render_question / render_state,
            HiddenStateEngine.fit (questions: encode_batch without special tokens; state: encode with special
            tokens, truncated to the window left after the question reserve), the max_request_tokens refusal
            (24576, handler.py's default) and option_token_index. The forward and the head are not run: their
            inputs are the token ids and option rows recorded here.
  error     an exception at any step is recorded as its type and message, so error behaviour is part of parity.

The engine is configured as handler.py configures it on Lambda: HiddenEngineConfig from head_config.json values
(max_length 4096, max_question_fraction 0.75, strict_window off), and the constructor's no_truncation() and
no_padding() applied to the tokenizer before any case runs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# This checkout's package (the one P1's zip copies), ahead of any other installed copy. Absent in the fixed-host
# container, where PYTHONPATH=/pkg supplies the deployed zip's copy.
_REPO_SRC = HERE.parent.parent.parent / "src"
if (_REPO_SRC / "strands_decider" / "hidden_engine.py").is_file():
    sys.path.insert(0, str(_REPO_SRC))
import tokcorpus  # noqa: E402

SCHEMA = "lkai-tok-golden/1"
MAX_LENGTH = 4096  # head_config.json max_length (deployed assets)
MAX_REQUEST_TOKENS = 24576  # examples/lambda/app/handler.py default


def load_impl(spec: str, tokenizer_path: Path) -> Any:
    mod, _, attr = spec.partition(":")
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj(str(tokenizer_path))


def _enc(e: Any) -> dict[str, Any]:
    return {"ids": list(e.ids), "offsets": [list(o) for o in e.offsets]}


def _err(exc: BaseException) -> dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "message": str(exc)[:500]}}


def _decode(tok: Any, ids: list[int], text: str) -> dict[str, Any]:
    try:
        d = tok.decode(ids, skip_special_tokens=False)
    except Exception as exc:
        return {"decode_error": type(exc).__name__}
    return {"decoded": d, "roundtrip_exact": d == text}


class _NoHead:
    """HiddenStateEngine.fit uses neither the head nor the forward."""


def engine_for(tok: Any) -> Any:
    from strands_decider.hidden_engine import HiddenEngineConfig, HiddenStateEngine

    cfg = HiddenEngineConfig(max_length=MAX_LENGTH, max_request_tokens=MAX_REQUEST_TOKENS)
    return HiddenStateEngine(tok, _NoHead(), cfg, forward_shared=None)  # type: ignore[arg-type]


def run_case(tok: Any, eng: Any, case: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": case["id"]}
    kind = case["kind"]
    try:
        if kind == "encode":
            e = tok.encode(case["text"], add_special_tokens=case["add_special_tokens"])
            out.update(_enc(e), **_decode(tok, list(e.ids), case["text"]))
        elif kind == "batch":
            es = tok.encode_batch(case["texts"], add_special_tokens=False)
            out["items"] = [
                {**_enc(e), **_decode(tok, list(e.ids), t)}
                for e, t in zip(es, case["texts"], strict=True)
            ]
        elif kind == "request":
            out.update(run_request(eng, case["request"]))
        else:
            raise ValueError(f"unknown case kind {kind!r}")
    except Exception as exc:
        out.update(_err(exc))
    return out


def run_request(eng: Any, body: dict[str, Any]) -> dict[str, Any]:
    from strands_decider.prompting import option_token_index, render_question, render_state
    from strands_decider.schema import SystemOneRequest

    out: dict[str, Any] = {}
    try:
        req = SystemOneRequest.model_validate(body)
    except Exception as exc:
        return {"stage": "validate", **_err(exc)}
    names = list(req.questions)
    rendered = [render_question(req.questions[n]) for n in names]
    try:
        s, q, offs = eng.fit(render_state(req.state), [rq.text for rq in rendered])
    except Exception as exc:
        return {"stage": "fit", **_err(exc)}
    n_input = len(s) + sum(len(x) for x in q)
    out.update(
        {
            "state_ids": list(s),
            "question_ids": [list(x) for x in q],
            "question_offsets": [[list(o) for o in oo] for oo in offs],
            "n_input": n_input,
        }
    )
    if n_input > MAX_REQUEST_TOKENS:
        out["stage"] = "refused-max-request-tokens"
        return out
    try:
        out["option_rows"] = [
            option_token_index(o, rq.option_spans, 0) for rq, o in zip(rendered, offs, strict=True)
        ]
    except Exception as exc:
        out.update({"stage": "option_index", **_err(exc)})
        return out
    out["stage"] = "ready-for-forward"
    return out


def make(corpus: Path, tokenizer: Path, impl: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head, cases = tokcorpus.load(corpus)
    tok = load_impl(impl, tokenizer)
    eng = engine_for(tok)
    rows = [run_case(tok, eng, c) for c in cases]
    import tokenizers

    meta = {
        "schema": SCHEMA,
        "corpus_sha256": tokcorpus.sha256_file(corpus),
        "tokenizer_json_sha256": tokcorpus.sha256_file(tokenizer),
        "impl": impl,
        "tokenizers_version": tokenizers.__version__,
        "tokenizers_so_sha256": so_sha(tokenizers),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "n_cases": len(rows),
    }
    if head.get("tokenizer_json_sha256") != meta["tokenizer_json_sha256"]:
        raise SystemExit("corpus was built from a different tokenizer.json")
    return meta, rows


def so_sha(mod: Any) -> str | None:
    for p in Path(mod.__file__).parent.glob("tokenizers*.so"):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return None


def _read(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def load_golden(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = _read(path).decode("ascii").splitlines()
    head = json.loads(lines[0])
    if head.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {head.get('schema')!r}, expected {SCHEMA}")
    return head, [json.loads(x) for x in lines[1:]]


def diff_rows(golden: list[dict[str, Any]], got: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every field that differs, per case. Error messages are compared by type only (wording is not semantics)."""

    def norm(r: dict[str, Any]) -> dict[str, Any]:
        r = json.loads(json.dumps(r))
        if "error" in r:
            r["error"] = {"type": r["error"]["type"]}
        return r

    out = []
    gi = {r["id"]: r for r in golden}
    if [r["id"] for r in golden] != [r["id"] for r in got]:
        out.append({"id": "*", "field": "case ids", "detail": "case list differs"})
    for r in got:
        g = gi.get(r["id"])
        if g is None:
            continue
        a, b = norm(g), norm(r)
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append({"id": r["id"], "field": k})
    return out


def check(corpus: Path, tokenizer: Path, golden: Path, impl: str) -> dict[str, Any]:
    gmeta, grows = load_golden(golden)
    if gmeta["corpus_sha256"] != tokcorpus.sha256_file(corpus):
        raise SystemExit("golden was recorded on a different corpus")
    meta, rows = make(corpus, tokenizer, impl)
    bad = diff_rows(grows, rows)
    ids_total = sum(_count_ids(r) for r in grows)
    return {
        "cases": len(rows),
        "token_ids_compared": ids_total,
        "mismatches": len(bad),
        "mismatch_rows": bad[:50],
        "impl": impl,
        "tokenizers_version": meta["tokenizers_version"],
        "tokenizers_so_sha256": meta["tokenizers_so_sha256"],
        "golden_tokenizers_version": gmeta["tokenizers_version"],
        "machine": meta["machine"],
    }


def _count_ids(r: dict[str, Any]) -> int:
    n = (
        len(r.get("ids", []))
        + len(r.get("state_ids", []))
        + sum(len(x) for x in r.get("question_ids", []))
    )
    return n + sum(len(i["ids"]) for i in r.get("items", []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("make", "check"))
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--impl", default="tokenizers:Tokenizer.from_file")
    a = ap.parse_args()
    if a.cmd == "make":
        meta, rows = make(a.corpus, a.tokenizer, a.impl)
        lines = [tokcorpus.dumps_line(meta)] + [tokcorpus.dumps_line(r) for r in rows]
        data = ("\n".join(lines) + "\n").encode("ascii")
        # sha256 is of the uncompressed bytes; the .gz is written with mtime 0 so it is stable too.
        a.out.write_bytes(gzip.compress(data, 9, mtime=0) if a.out.suffix == ".gz" else data)
        sha = hashlib.sha256(data).hexdigest()
        Path(str(a.out) + ".sha256").write_text(
            f"{sha}  {a.out.name} (uncompressed)\n", encoding="ascii"
        )
        print(
            json.dumps(
                {"cases": len(rows), "sha256": sha, "tokenizers": meta["tokenizers_version"]}
            )
        )
        return
    res = check(a.corpus, a.tokenizer, a.golden, a.impl)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["mismatches"] == 0 else 1)


if __name__ == "__main__":
    main()
