"""R (ce4d) ranked recommendations: outputs are derived from the pinned results, and the ranking rules hold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples" / "lambda-kleidiai" / "recommendations"


def _rank():
    spec = importlib.util.spec_from_file_location("lkai_rank", ROOT / "rank.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_committed_outputs_are_current():
    rank = _rank()
    assert rank.main(["--check"]) == 0


def test_every_pinned_source_hashes_to_its_record():
    # build() refuses a source whose bytes differ from sources.json; it must succeed on the committed tree.
    rk = _rank().build()
    assert rk["cpu_part"] == "0xd40"


def test_interval_excluding_zero_is_not_enough_without_the_experiments_claim():
    rank = _rank()
    rows = {r["id"]: r for r in rank.build()["effects"]}
    # E8's restore interval excludes 0 ([-0.072, -0.006]) but E8 claims no restore effect.
    e8_restore = rows["E8"]["metrics"]["restore.first_request_total"]
    _lo, hi = e8_restore["parts"]["0xd40"]["ci95"]
    assert hi < 0
    assert e8_restore["verdict"] == "not-claimed"
    # E13's own host-cell assessment is mixed: its warm effect stays inconclusive, never "measured".
    assert rows["E13"]["metrics"]["warm.steady_state_inference"]["verdict"] == "inconclusive"


def test_verdict_rules():
    rank = _rank()
    assert rank.verdict(None) == "not-measured"
    assert rank.verdict({"0xd40": {"delta": -1.0, "ci95": [-2.0, -0.5], "n": [5, 5]}}) == "measured"
    assert rank.verdict({"0xd40": {"delta": -1.0, "ci95": [-2.0, 0.1], "n": [5, 5]}}) == "no-change"
    assert rank.verdict({"0xd40": {"delta": -1.0, "ci95": None, "n": [5, 5]}}) == "no-ci"
    assert (
        rank.verdict({"0xd0c": {"delta": -1.0, "ci95": [-2.0, -0.5], "n": [5, 5]}})
        == "not-measured"
    )


def test_headline_never_pools_cpu_parts():
    rows = _rank().build()["effects"]
    for r in rows:
        for m in r["metrics"].values():
            if m["parts"]:
                assert "pooled" not in json.dumps(m["parts"])


def test_charts_are_self_contained_svg():
    for svg in (ROOT / "charts").glob("*.svg"):
        text = svg.read_text()
        assert text.startswith("<svg ")
        assert "<script" not in text and "href=" not in text
        assert 'role="img"' in text and "aria-label=" in text


def test_no_generation_named_from_a_cpu_part():
    text = (ROOT / "README.md").read_text() + (ROOT / "ranking.json").read_text()
    for name in ("Graviton2", "Graviton3", "Graviton4", "Graviton 2", "Graviton 3", "Graviton 4"):
        assert name not in text
