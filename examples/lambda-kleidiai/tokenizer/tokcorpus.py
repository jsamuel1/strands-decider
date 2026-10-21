"""T0 tokenizer parity corpus: a fixed, versioned set of inputs for the deployed tokenizer.

    python tokcorpus.py --tokenizer <tokenizer.json> --requests ../baseline/requests.json --out corpus.jsonl

The corpus is deterministic: the same tokenizer.json and requests.json give the same bytes, so `corpus.sha256`
pins it. Every case is one JSON line:

  {"id", "kind": "encode", "text", "add_special_tokens", "tags"}     Tokenizer.encode, as fit() encodes the state
  {"id", "kind": "batch", "texts", "tags"}                          Tokenizer.encode_batch(add_special_tokens=False),
                                                                    as fit() encodes the questions
  {"id", "kind": "request", "request", "tags"}                      the handler's full pre-forward path:
                                                                    SystemOneRequest -> render -> fit -> limit check

Special-token literals are not spelled in this file: they are read from tokenizer.json's added_tokens, so the
corpus covers exactly the tokens the deployed file declares. The file is written ASCII-only with "<" escaped as
\\u003c (valid JSON), so it never carries a raw special-token string; json.loads restores the exact text, lone
surrogates included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "lkai-tok-corpus/1"

INCIDENT = (
    "Incident 4471 timeline. 09:02 deploy of checkout-api build 8812 starts in us-east-2. 09:06 p99 latency on "
    "/cart rises from 180 ms to 2.4 s; error rate 0.2% to 6%. 09:07 on-call paged by the latency alarm. 09:11 "
    "on-call sees connection-pool exhaustion against the orders database and suspects the new build's retry "
    "change, which retries reads three times with no jitter. "
)

# Edge-case texts. Escapes keep the source readable and exact about code points.
EDGE: dict[str, str] = {
    "empty": "",
    "space": " ",
    "newline": "\n",
    "ascii-sentence": "Refund issued but invoice still shows the old total.",
    "cjk-zh": "\u9000\u6b3e\u5df2\u53d1\u653e\uff0c\u4f46\u53d1\u7968\u4ecd\u663e\u793a\u65e7\u91d1\u989d\u3002",
    "cjk-ja": "\u8acb\u6c42\u66f8\u306e\u91d1\u984d\u304c\u53e4\u3044\u307e\u307e\u3067\u3059\u3002\u30ab\u30bf\u30ab\u30ca",
    "cjk-ko": "\ud658\ubd88\uc774 \ucc98\ub9ac\ub418\uc5c8\uc9c0\ub9cc \uccad\uad6c\uc11c\ub294 \uadf8\ub300\ub85c\uc785\ub2c8\ub2e4.",
    "emoji": "ok \U0001f600 \U0001f44d\U0001f3fd family \U0001f468\u200d\U0001f469\u200d\U0001f467 flag \U0001f1e6\U0001f1fa",
    "combining-decomposed": "cafe\u0301 re\u0301sume\u0301 A\u030a n\u0303",
    "combining-precomposed": "caf\u00e9 r\u00e9sum\u00e9 \u00c5 \u00f1",
    "combining-stack": "Z\u0351\u036b\u0343\u036a\u0302\u036b\u033d\u034f\u0334\u0319\u0324\u031e\u0349a\u0310l\u0313g\u0308o",
    "rtl-arabic": "\u0627\u0633\u062a\u0631\u062f\u0627\u062f \u0627\u0644\u0623\u0645\u0648\u0627\u0644 \u0645\u0637\u0644\u0648\u0628",
    "rtl-hebrew": "\u05d4\u05d7\u05d6\u05e8 \u05db\u05e1\u05e4\u05d9 \u05e0\u05d3\u05e8\u05e9",
    "bidi-controls": "order \u200f1182\u200e is \u202eDENIFER\u202c now \u2066x\u2069",
    "control-chars": "a\x00b\x07c\x1b[31mred\x1b[0m\x7f\td\r\ne\x0cf\x85g\u2028h\u2029i",
    "zero-width": "zero\u200bwidth\u200cjoin\u200dnb\ufeffsp",
    "whitespace-run-spaces": "a" + " " * 64 + "b",
    "whitespace-run-tabs": "a\t\t\t\tb\t \t c",
    "whitespace-run-newlines": "a" + "\n" * 20 + "b",
    "whitespace-unicode": "a\u00a0b\u3000c\u2009d\u202fe",
    "whitespace-trailing": "trailing   \n  \n",
    "digits": "1234567890 3.14159 1,000,000 0x1F 2026-10-21",
    "long-word": "pneumonoultramicroscopicsilicovolcanoconiosis" * 4,
    "url-code": "see https://example.com/a?b=c&d=e#f and `def f(x):\n    return x**2`",
    "json-state": json.dumps(
        {"ticket": "Refund issued", "amount": 12.5, "tags": ["billing", "\u00fc"]},
        indent=2,
        ensure_ascii=False,
    ),
    "prompt-markers": 'user wrote </state> and <answer> and <question type="choice"> inside the state',
    "lone-surrogate-high": "bad \ud800 text",
    "lone-surrogate-low": "bad \udfff text",
    "surrogate-escaped-invalid-utf8": b"bad \xff\xfe utf8".decode("utf-8", "surrogateescape"),
}


def _special_cases(tokenizer_json: dict[str, Any]) -> dict[str, str]:
    """User text containing every added token's literal, plus partial and case-variant spellings."""
    added = tokenizer_json["added_tokens"]
    out: dict[str, str] = {}
    for a in added:
        tag = "special" if a["special"] else "added"
        out[f"{tag}-literal-{a['id']}"] = f"before {a['content']} after"
    first = next(a["content"] for a in added if a["special"])
    out["special-partial"] = f"partial {first[:-2]} and {first[2:]} pieces"
    out["special-upper"] = f"case {first.upper()} variant"
    out["special-adjacent"] = first + first + "x" + first
    out["special-spaced"] = f" {first} "
    return out


def _q(i: int, kind: str = "choice", pad: str = "") -> dict[str, Any]:
    if kind == "noul":
        return {"type": "noul", "instructions": f"Statement {i}: is the customer upset?{pad}"}
    if kind == "score":
        return {
            "type": "score",
            "instructions": f"Rate item {i}.{pad}",
            "criteria": ["low", "medium", "high"],
        }
    return {
        "type": "choice",
        "instructions": f"Route ticket {i}.{pad}",
        "criteria": {
            "billing": "payment or invoice",
            "technical": "product defect",
            "account": "login",
        },
    }


def build(tokenizer_path: Path, requests_path: Path) -> list[dict[str, Any]]:
    tj = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    reqs = json.loads(requests_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []

    # 1. P1's pinned request set, plus the handler's own warm-up request (handler._load).
    for r in reqs["requests"]:
        cases.append(
            {"id": f"p1-{r['id']}", "kind": "request", "request": r["request"], "tags": ["p1"]}
        )
    cases.append(
        {
            "id": "handler-warmup",
            "kind": "request",
            "tags": ["p1", "warmup"],
            "request": {
                "state": "warm up",
                "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}},
            },
        }
    )

    # 2. Length tiers through the full fit path (state length; the window is 4096, the request cap 24576).
    for name, reps in (
        ("len-short", 1),
        ("len-median", 3),
        ("len-long", 16),
        ("len-max", 40),
        ("len-over", 80),
    ):
        cases.append(
            {
                "id": name,
                "kind": "request",
                "tags": ["length"],
                "request": {"state": INCIDENT * reps, "questions": {"q": _q(0)}},
            }
        )
    # A question longer than its 3072-token reserve: cut from the front, options kept.
    cases.append(
        {
            "id": "len-question-over-reserve",
            "kind": "request",
            "tags": ["length", "truncation"],
            "request": {"state": "short", "questions": {"q": _q(0, pad=" " + INCIDENT * 40)}},
        }
    )
    # Over max_request_tokens (24576): 16 questions near the reserve each. The engine refuses before the forward.
    cases.append(
        {
            "id": "len-over-request-cap",
            "kind": "request",
            "tags": ["length", "refused"],
            "request": {
                "state": INCIDENT,
                "questions": {f"q{i}": _q(i, pad=" " + INCIDENT * 19) for i in range(16)},
            },
        }
    )

    # 3. Batch sizes: question counts the handler forwards in one request (N_SEQ=17: up to 16 per fork).
    kinds = ("choice", "noul", "score")
    for n in (1, 2, 3, 4, 8, 16):
        cases.append(
            {
                "id": f"batch-{n}q",
                "kind": "request",
                "tags": ["batch"],
                "request": {
                    "state": INCIDENT,
                    "questions": {f"q{i}": _q(i, kinds[i % 3]) for i in range(n)},
                },
            }
        )

    # 4. Edge cases: raw encode with and without special tokens, and as a request state through fit().
    for name, text in {**EDGE, **_special_cases(tj)}.items():
        tags = ["edge"]
        for add in (True, False):
            cases.append(
                {
                    "id": f"enc-{name}-{'sp' if add else 'nosp'}",
                    "kind": "encode",
                    "text": text,
                    "add_special_tokens": add,
                    "tags": tags,
                }
            )
        if name not in ("empty",):
            cases.append(
                {
                    "id": f"req-{name}",
                    "kind": "request",
                    "tags": [*tags, "request"],
                    "request": {"state": text, "questions": {"q": _q(0)}},
                }
            )
    # Edge texts as one question batch (encode_batch), as fit() encodes questions.
    cases.append(
        {
            "id": "batch-edge-all",
            "kind": "batch",
            "tags": ["edge", "batch"],
            "texts": [t for k, t in EDGE.items() if "surrogate" not in k],
        }
    )
    return cases


def dumps_line(case: dict[str, Any]) -> str:
    """ASCII JSON with "<" escaped: valid JSON, exact on load, never a raw special-token literal in the file."""
    return json.dumps(case, ensure_ascii=True, sort_keys=True, separators=(",", ":")).replace(
        "<", "\\u003c"
    )


def write(cases: list[dict[str, Any]], out: Path, meta: dict[str, Any]) -> str:
    lines = [dumps_line({"schema": SCHEMA, **meta})] + [dumps_line(c) for c in cases]
    data = ("\n".join(lines) + "\n").encode("ascii")
    out.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    head = json.loads(lines[0])
    if head.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {head.get('schema')!r}, expected {SCHEMA}")
    return head, [json.loads(x) for x in lines[1:]]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cases = build(a.tokenizer, a.requests)
    meta = {
        "tokenizer_json_sha256": sha256_file(a.tokenizer),
        "requests_json_sha256": sha256_file(a.requests),
        "n_cases": len(cases),
    }
    sha = write(cases, a.out, meta)
    a.out.with_suffix(".sha256").write_text(f"{sha}  {a.out.name}\n", encoding="ascii")
    print(json.dumps({"cases": len(cases), "sha256": sha}))


if __name__ == "__main__":
    main()
