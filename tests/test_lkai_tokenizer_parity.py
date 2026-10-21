"""T0 tokenizer parity suite (examples/lambda-kleidiai/tokenizer): the deployed tokenizer must reproduce the golden
token ids, offsets, decode round-trips, fit() outputs and error behaviour recorded from P1's deployed function.zip.

T1-T3 keep this suite passing unchanged. The golden was recorded on Graviton3 with tokenizers 0.23.2 (the deployed
wheel); the rows are identical on x86_64 with 0.23.3, so the suite runs on any host with the model's tokenizer.json.
Needs the Hub snapshot of StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 at 216d1ccb (skipped without it).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

TOK = Path(__file__).resolve().parent.parent / "examples" / "lambda-kleidiai" / "tokenizer"
DATA = TOK / "data"
sys.path.insert(0, str(TOK))

import stackcollapse  # noqa: E402
import tokcorpus  # noqa: E402
import tokparity  # noqa: E402


def test_stackcollapse_folds_root_first() -> None:
    text = (
        "python3.12 123\n"
        "\t     aaaa leaf_fn+0x10 (/pkg/tokenizers/tokenizers.abi3.so)\n"
        "\t     bbbb mid (/usr/lib/libc.so.6)\n"
        "\t     cccc [unknown] ([unknown])\n"
        "\n"
        "python3.12 123\n"
        "\t     aaaa leaf_fn+0x20 (/pkg/tokenizers/tokenizers.abi3.so)\n"
        "\t     bbbb mid (/usr/lib/libc.so.6)\n"
        "\t     cccc [unknown] ([unknown])\n"
    )
    got = stackcollapse.collapse(text.splitlines(keepends=True))
    assert dict(got) == {
        "python3.12;[unknown] ([unknown]);mid (libc.so.6);leaf_fn (tokenizers.abi3.so)": 2
    }


REV = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
TOKENIZER_SHA = "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86"


def _tokenizer_json() -> Path | None:
    env = os.environ.get("LKAI_TOKENIZER_JSON")
    cands = [Path(env)] if env else []
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cands.append(
        hub
        / "models--StrandsAgents--strands-decider-2B-qwen3.5-v1-2610"
        / "snapshots"
        / REV
        / "tokenizer.json"
    )
    for c in cands:
        if c.is_file():
            return c
    return None


TJ = _tokenizer_json()
needs_tok = pytest.mark.skipif(
    TJ is None, reason="model tokenizer.json not available (set LKAI_TOKENIZER_JSON)"
)


def _sha(p: Path, gz: bool = False) -> str:
    data = p.read_bytes()
    if gz:
        import gzip

        data = gzip.decompress(data)
    return hashlib.sha256(data).hexdigest()


def test_committed_corpus_and_golden_match_their_sha256() -> None:
    assert _sha(DATA / "corpus.jsonl") == (DATA / "corpus.sha256").read_text().split()[0]
    assert (
        _sha(DATA / "golden.jsonl.gz", gz=True)
        == (DATA / "golden.jsonl.gz.sha256").read_text().split()[0]
    )
    head, _ = tokparity.load_golden(DATA / "golden.jsonl.gz")
    assert head["corpus_sha256"] == _sha(DATA / "corpus.jsonl")
    assert head["tokenizer_json_sha256"] == TOKENIZER_SHA
    assert head["tokenizers_version"] == "0.23.2" and head["machine"] == "aarch64"


def test_corpus_files_hold_no_raw_special_token_literal() -> None:
    for name in ("corpus.jsonl",):
        assert "<|" not in (DATA / name).read_text(encoding="ascii")


def test_corpus_covers_the_required_categories() -> None:
    _, cases = tokcorpus.load(DATA / "corpus.jsonl")
    ids = {c["id"] for c in cases}
    tags = {t for c in cases for t in c["tags"]}
    assert {"p1", "length", "batch", "edge", "truncation", "refused", "warmup"} <= tags
    for rid in (
        "ticket-route-1q",
        "payouts-3q",
        "cracked-screen-2q",
        "agent-action-guard-2q",
        "incident-review-4q",
    ):
        assert f"p1-{rid}" in ids
    for n in (1, 2, 3, 4, 8, 16):
        assert f"batch-{n}q" in ids
    for edge in (
        "cjk-zh",
        "emoji",
        "combining-decomposed",
        "rtl-arabic",
        "control-chars",
        "whitespace-run-spaces",
        "lone-surrogate-high",
        "special-partial",
        "empty",
    ):
        assert f"enc-{edge}-sp" in ids
    assert (
        sum(i.startswith("enc-special-literal-") for i in ids) == 2 * 21
    )  # every special token, both modes


@needs_tok
def test_corpus_rebuilds_byte_identically() -> None:
    assert TJ is not None
    cases = tokcorpus.build(TJ, TOK.parent / "baseline" / "requests.json")
    head, committed = tokcorpus.load(DATA / "corpus.jsonl")
    assert head["tokenizer_json_sha256"] == tokcorpus.sha256_file(TJ) == TOKENIZER_SHA
    assert [tokcorpus.dumps_line(c) for c in cases] == [tokcorpus.dumps_line(c) for c in committed]


@needs_tok
def test_deployed_tokenizer_matches_every_golden_row() -> None:
    assert TJ is not None
    res = tokparity.check(
        DATA / "corpus.jsonl", TJ, DATA / "golden.jsonl.gz", "tokenizers:Tokenizer.from_file"
    )
    assert res["mismatches"] == 0, res["mismatch_rows"]
    assert res["cases"] == 217 and res["token_ids_compared"] > 60000


def _golden_rows() -> list[dict]:
    return tokparity.load_golden(DATA / "golden.jsonl.gz")[1]


def test_one_changed_token_id_fails_parity() -> None:
    """Deliberate mutation: one id in one request's state -> exactly that case is reported."""
    golden = _golden_rows()
    got = copy.deepcopy(golden)
    row = next(r for r in got if r["id"] == "p1-incident-review-4q")
    row["state_ids"][17] += 1
    bad = tokparity.diff_rows(golden, got)
    assert bad == [{"id": "p1-incident-review-4q", "field": "state_ids"}]


@pytest.mark.parametrize("field", ["ids", "offsets", "decoded"])
def test_one_changed_encode_field_fails_parity(field: str) -> None:
    golden = _golden_rows()
    got = copy.deepcopy(golden)
    row = next(r for r in got if r["id"] == "enc-emoji-sp")
    if field == "decoded":
        row[field] += "x"
    else:
        row[field][0] = row[field][0] + 1 if field == "ids" else [9, 9]
    assert tokparity.diff_rows(golden, got) == [{"id": "enc-emoji-sp", "field": field}]


def test_changed_error_type_and_dropped_case_fail_parity() -> None:
    golden = _golden_rows()
    got = copy.deepcopy(golden)
    row = next(r for r in got if r["id"] == "enc-lone-surrogate-high-sp")
    row["error"]["type"] = "ValueError"
    assert tokparity.diff_rows(golden, got) == [
        {"id": "enc-lone-surrogate-high-sp", "field": "error"}
    ]
    # A different message with the same type is not a semantic change.
    got = copy.deepcopy(golden)
    next(r for r in got if r["id"] == "enc-lone-surrogate-high-sp")["error"]["message"] = (
        "other words"
    )
    assert tokparity.diff_rows(golden, got) == []
    assert tokparity.diff_rows(golden, got[:-1])[0]["field"] == "case ids"


@needs_tok
def test_mutated_implementation_is_caught_end_to_end(tmp_path: Path) -> None:
    """A candidate tokenizer that differs by one id on one input fails `check` (the path T1-T3 use)."""
    assert TJ is not None
    mod = tmp_path / "t0_mutant.py"
    mod.write_text(
        "import tokenizers\n"
        "class _E:\n"
        "    def __init__(self, e, bump):\n"
        "        self.ids = list(e.ids); self.offsets = e.offsets\n"
        "        if bump and self.ids: self.ids[-1] += 1\n"
        "class Mutant:\n"
        "    def __init__(self, p): self.t = tokenizers.Tokenizer.from_file(p)\n"
        "    def __getattr__(self, k): return getattr(self.t, k)\n"
        "    def encode(self, text, add_special_tokens=True):\n"
        "        return _E(self.t.encode(text, add_special_tokens=add_special_tokens), 'Z\\u0351' in text)\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        res = tokparity.check(
            DATA / "corpus.jsonl", TJ, DATA / "golden.jsonl.gz", "t0_mutant:Mutant"
        )
    finally:
        sys.path.remove(str(tmp_path))
    ids = {r["id"] for r in res["mismatch_rows"]}
    assert res["mismatches"] >= 2 and {"enc-combining-stack-sp", "enc-combining-stack-nosp"} <= ids


def test_semantics_recorded_in_golden() -> None:
    """Facts the README states, pinned to the golden so a candidate cannot change them silently."""
    g = {r["id"]: r for r in _golden_rows()}
    assert (
        g["enc-empty-sp"]["ids"] == [] and g["enc-empty-sp"]["roundtrip_exact"] is True
    )  # no BOS for Qwen
    assert (
        g["enc-ascii-sentence-sp"]["ids"] == g["enc-ascii-sentence-nosp"]["ids"]
    )  # add_special adds nothing
    lit = g["enc-special-literal-248044-sp"]
    assert 248044 in lit["ids"]  # a special literal in user text becomes the special id (not split)
    assert g["enc-special-partial-sp"]["roundtrip_exact"] is True
    for k in ("lone-surrogate-high", "lone-surrogate-low", "surrogate-escaped-invalid-utf8"):
        assert g[f"enc-{k}-sp"]["error"]["type"] == "TypeError"
        assert g[f"req-{k}"]["stage"] == "fit"
    assert g["enc-combining-decomposed-sp"]["roundtrip_exact"] is False  # NFC normalisation
    assert (
        g["len-max"]["n_input"] == 4096 and g["len-over"]["n_input"] == 4096
    )  # state trimmed to the window
    assert g["len-over-request-cap"]["stage"] == "refused-max-request-tokens"
    assert g["len-question-over-reserve"]["stage"] == "ready-for-forward"
    assert (
        len(g["len-question-over-reserve"]["question_ids"][0]) == 3072
    )  # 0.75 x 4096 reserve, cut from the front
    json.dumps(g["batch-16q"])  # 16 questions, the most one fork group takes (N_SEQ=17)
