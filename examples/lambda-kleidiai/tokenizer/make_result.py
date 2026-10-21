"""Assemble T0's P2 result file (lkai-result/1) from the Lambda share analysis and the fixed-host benchmarks.

    python make_result.py --p3-results <P3 20261020T231634Z results.json> --share data/lambda-share.json \\
        --bench data/bench-r1.json --bench data/bench-r2.json --parity data/parity-check.json \\
        --run-id <id> --commit <T0 commit> --out <run-dir>/results.json

Lambda numbers are reused from P3's run 20261020T231634Z (P1's deployed zip, us-east-2, no new spend):
  phases.rep        REPORT-only control (LKAI_PHASES=0), P3's arm as published: the light-instrumentation view.
  phases.cprofile   P3's cProfile arm, plus the tokenizer rows this card extracts from its logs:
                    cold.tokenizer_init_from_file, restore.snapshot_init.tokenizer_init_from_file (as published by
                    P3 under snapshot_init), and tokenizer_fit for cold / restore / warm. cProfile overhead
                    (+113 ms Init, P3) makes these upper bounds.
Fixed-host numbers (Graviton3 DevSpace, pinned Lambda image) are in `fixed_host`, labelled, never in `phases`.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any

P1_BASELINE = "70eab7be69f86ddeb8c64158d94df344b5180098"
P3_COMMIT = "0eabd60fd11f6183faa8ac908d138724b63d1eca"


def _d(x: dict[str, Any] | None, unit: str = "s") -> dict[str, Any] | None:
    if not x:
        return None
    out = {"unit": unit, **{k: x[k] for k in ("n", "p50", "p90", "p99", "min", "max", "mean")}}
    out["stdev"] = x.get("stdev", 0.0)
    return out


def fixed_host_block(benches: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    for b in benches:
        s = b["summary"]
        runs.append(
            {
                "label": b["label"],
                "run": b["host"]["label"],
                "host": b["host"],
                "started": b["started"],
                "finished": b["finished"],
                "params": {
                    k: b["params"][k] for k in ("rounds", "init_reps", "iters", "warmup", "tags")
                },
                "init": s["init"],
                "steady": {
                    cid: {
                        k: v[k]
                        for k in (
                            "tags",
                            "tokens",
                            "wall",
                            "cpu",
                            "encode_only_wall",
                            "decode_wall",
                            "round_p50s",
                            "round_p50_cv",
                            "round_p50_min",
                            "round_p50_max",
                            "tokens_per_s_p50",
                            "requests_per_s_p50",
                            "cpu_over_wall_p50",
                            "gc_collections",
                            "gc_collected",
                            "memory",
                        )
                    }
                    for cid, v in s["steady"].items()
                },
                "process_maxrss_kb": b["process_maxrss_kb"],
            }
        )
    rep = None
    if len(runs) >= 2:
        a, b = runs[0], runs[1]
        rows = {}
        for cid in a["steady"]:
            if cid not in b["steady"]:
                continue
            pa, pb = a["steady"][cid]["wall"]["p50"], b["steady"][cid]["wall"]["p50"]
            spread = max(
                a["steady"][cid]["round_p50_max"], b["steady"][cid]["round_p50_max"]
            ) - min(a["steady"][cid]["round_p50_min"], b["steady"][cid]["round_p50_min"])
            within = max(a["steady"][cid]["round_p50s"]) >= min(
                b["steady"][cid]["round_p50s"]
            ) and max(b["steady"][cid]["round_p50s"]) >= min(a["steady"][cid]["round_p50s"])
            rows[cid] = {
                "p50_a_s": pa,
                "p50_b_s": pb,
                "delta_pct": (pb - pa) / pa * 100,
                "round_p50_range_s": spread,
                "round_ranges_overlap": within,
            }
        fa, fb = a["init"]["from_file_s"]["p50"], b["init"]["from_file_s"]["p50"]
        rows["init:from_file"] = {
            "p50_a_s": fa,
            "p50_b_s": fb,
            "delta_pct": (fb - fa) / fa * 100,
            "round_ranges_overlap": max(a["init"]["round_p50_from_file_s"])
            >= min(b["init"]["round_p50_from_file_s"])
            and max(b["init"]["round_p50_from_file_s"]) >= min(a["init"]["round_p50_from_file_s"]),
        }
        rep = {
            "runs": [runs[0]["run"], runs[1]["run"]],
            "rows": rows,
            "all_overlap": all(r["round_ranges_overlap"] for r in rows.values()),
            "max_abs_delta_pct": max(abs(r["delta_pct"]) for r in rows.values()),
        }
    return {"label": "FIXED HOST, NOT LAMBDA", "runs": runs, "repeatability": rep}


def build(
    p3: dict[str, Any],
    share: dict[str, Any],
    benches: list[dict[str, Any]],
    parity: dict[str, Any],
    prof: dict[str, Any] | None,
    run_id: str,
    commit: str,
) -> dict[str, Any]:
    ph = {
        "rep": copy.deepcopy(p3["phases"]["rep"]),
        "cprofile": copy.deepcopy(p3["phases"]["cprofile"]),
    }
    pc = share["per_request_cprofile"]
    ph["cprofile"]["cold"]["tokenizer_init_from_file"] = _d(
        share["init_from_file_cprofile"]["cold|all"]
    )
    ph["cprofile"]["restore"]["snapshot_init"]["tokenizer_init_from_file"] = _d(
        share["init_from_file_cprofile"].get("snapshot_init|all")
    )
    for cls in ("cold", "restore", "warm"):
        ph["cprofile"][cls]["tokenizer_fit"] = _d(pc[f"{cls}|all"]["fit_s"])
        ph["cprofile"][cls]["tokenizer_fit_by_cpu_part"] = {
            k.split("|")[1]: _d(v["fit_s"])
            for k, v in pc.items()
            if k.startswith(cls + "|") and not k.endswith("|all")
        }
    h = share["headroom"]
    fh = fixed_host_block(benches)
    res = {
        "schema": "lkai-result/1",
        "experiment_id": "T0-tokenizer-baseline",
        "run_id": run_id,
        "status": "inconclusive",
        "primary_arm": "cprofile",
        "hypothesis": (
            "Characterisation, not an experiment: identify the deployed tokenizer, pin its semantics with a "
            "golden parity suite, and measure its share of Lambda cold, restore and warm latency. The "
            "operator's report that the tokenizer is implemented in Python is tested by binary "
            "inspection."
        ),
        "changed_variable": {
            "name": "measurement scope (no deployed code change)",
            "baseline": "P1 deployed zip 26eb2651..., P3 run 20261020T231634Z arms rep and cprofile",
            "experiment": "tokenizer-only extraction of those Lambda runs, plus a fixed-host "
            "microbenchmark and perf profile of the same tokenizer bytes",
        },
        "commits": {
            "baseline": P1_BASELINE,
            "experiment": commit,
            "repo_dirty_at_build": False,
            "p1_zip_built_from": p3["commits"].get("p1_zip_built_from"),
            "p3_lambda_run_commit": P3_COMMIT,
        },
        "deployment": p3["deployment"],
        "model": p3["model"],
        "conditions": {
            **p3["conditions"],
            "t0_note": "Lambda conditions are P3 run 20261020T231634Z's; T0 made "
            "no Lambda invocations",
        },
        "classification": p3["classification"],
        "phases": ph,
        "variance": p3.get("variance"),
        "errors": {"count": 0, "rows": []},
        "cost": {
            **{
                k: p3["cost"][k]
                for k in ("billed_duration_s", "total_usd", "s3_request_usd", "s3_gb_transferred")
            },
            "t0_new_aws_usd": 0.0,
            "note": "costs are P3's run, already paid and published; T0 added no AWS spend",
        },
        "correctness": {**p3["correctness"], "tokenizer_parity": parity},
        "tokenizer": tokenizer_identity(),
        "lambda_share": share,
        "headroom": h,
        "fixed_host": fh,
        "profile": prof,
        "profiling_artifacts": [],
        "artifacts": {},
        "reproduce": [
            f"git fetch https://github.com/jsamuel1/strands-decider lkai/951d-tok-baseline && git checkout {commit}",
            "python -m pytest tests/test_lkai_tokenizer_parity.py   # parity suite (needs the model's tokenizer.json)",
            "examples/lambda-kleidiai/tokenizer/fixed_host.sh <P1 function.zip> out golden g1",
            "examples/lambda-kleidiai/tokenizer/fixed_host.sh <P1 function.zip> out bench r1   # and r2, interleaved",
            "examples/lambda-kleidiai/tokenizer/fixed_host.sh <P1 function.zip> out profile p1",
            "git show origin/lkai/bc9a-profiler:examples/lambda-kleidiai/profiling/results/20261020T231634Z/logs/"
            "lkai-p3-{cprofile,sampler}.jsonl.gz > logs/...  # P3's published logs",
            "python examples/lambda-kleidiai/tokenizer/lambda_share.py --p1 examples/lambda-kleidiai/baseline/"
            "results/20261020T080835Z/results.json --p3-logs logs --out lambda-share.json",
            "python examples/lambda-kleidiai/tokenizer/make_result.py ...   # this file",
        ],
        "conclusion": "",
    }
    return res


def tokenizer_identity() -> dict[str, Any]:
    return {
        "implementation": "Hugging Face tokenizers 0.23.2 (Rust core, PyO3 binding): tokenizers/tokenizers.abi3.so "
        "in function.zip, sha256 999ab0b786c4c73eeadda6420e6048b22e0c465fc06587aa8eb0d341a287c1f3",
        "native_evidence": [
            "ELF 64-bit aarch64 shared object, not stripped",
            ".comment: rustc version 1.98.0 (88d9e12ae 2026-08-18)",
            "572 dynamic symbols including PYO3_INTROSPECTION_*",
            "statically linked crates: tokenizers' onig 6.5.3, regex 1.13.1, rayon 1.12.0, "
            "unicode-normalization-alignments 0.1.12, serde_json 1.0.151, esaxx-rs 0.1.10",
        ],
        "pure_python": False,
        "call_path": [
            "lkai_entry.handler",
            "app.handler",
            "handler.handler (PR #77)",
            "handler._evaluate",
            "engine.HeadRowsEngine.evaluate",
            "strands_decider.hidden_engine.HiddenStateEngine.fit",
            "Tokenizer.encode_batch(questions, add_special_tokens=False) and "
            "Tokenizer.encode(state, add_special_tokens=True)",
        ],
        "construction": "strands_decider.hidden_engine.HiddenStateEngine.from_assets -> "
        "Tokenizer.from_file(assets/tokenizer.json), then no_truncation() and no_padding()",
        "init_placement": {
            "on-demand": "PR #77 _load in a ThreadPoolExecutor thread, after the weight download and "
            "llama.cpp load; outside REPORT Init Duration, inside the cold first request",
            "snapstart": "at module import, before the snapshot (before_snapshot hook); not re-run "
            "on restore",
        },
        "model": {
            "id": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
            "revision": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
            "base": "Qwen/Qwen3.5-2B-Base",
            "tokenizer_class": "Qwen2Tokenizer",
        },
        "vocab_file": {
            "path": "assets/tokenizer.json",
            "bytes": 19989592,
            "sha256": "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86",
            "same_as_hub_revision": True,
        },
        "algorithm": {
            "model": "BPE",
            "vocab": 248044,
            "merges": 247587,
            "added_tokens": 33,
            "special_tokens": 21,
            "vocab_with_added": 248077,
            "byte_fallback": False,
            "unk_token": None,
            "dropout": None,
        },
        "normalizer": "NFC",
        "pre_tokenizer": "Split (Qwen regex: contractions, letters with optional leading non-letter, single "
        "digits, punctuation runs, newline runs, whitespace) Isolated, then ByteLevel "
        "(add_prefix_space false, use_regex false)",
        "post_processor": "ByteLevel (no BOS/EOS added: add_special_tokens adds nothing for this file)",
        "decoder": "ByteLevel",
        "special_tokens_in_text": "split_special_tokens false: a special-token literal in user text becomes that "
        "token's id (recorded per token in the golden)",
        "chat_template": "chat_template.jinja exists on the Hub but is not shipped or used: the engine renders its "
        "own <state>/<question>/<answer> prompt (strands_decider.prompting)",
        "truncation": "tokenizer.json sets Right/LongestFirst max_length 4096; the engine calls no_truncation(), "
        "then fit() reserves min(longest question, 0.75 x 4096 = 3072) tokens, cuts questions from "
        "the front to the reserve, and truncates the state from the right to 4096 - reserve",
        "padding": "tokenizer.json sets BatchLongest; the engine calls no_padding()",
        "max_length": {
            "window": 4096,
            "max_request_tokens": 24576,
            "over_cap": "ValueError before the forward (handler: HTTP 400 / error response)",
        },
        "errors": {
            "lone_surrogate": "tokenizer: TypeError 'TextInputSequence must be str' from encode / encode_batch "
            "(PyO3 cannot convert to UTF-8). Function URL path: pydantic's JSON parser "
            "rejects the \\ud800 escape first (HTTP 422, tokenizer never called). Direct "
            "Invoke path: the runtime's json.loads yields the surrogate, validation "
            "accepts it, and fit raises TypeError, which handler.handler does not catch "
            "(invocation error)",
            "invalid_utf8": "cannot reach the tokenizer as bytes: decoding the event body fails first "
            "(UnicodeDecodeError / json_invalid); surrogate-escaped text behaves as a lone "
            "surrogate (TypeError)",
            "empty": "encode('') -> [] (no special tokens added); an empty state renders to "
            "'<state>\\n\\n</state>\\n'",
            "oversize": "state truncated to the window; request over 24576 tokens refused",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p3-results", type=Path, required=True)
    ap.add_argument("--share", type=Path, required=True)
    ap.add_argument("--bench", type=Path, action="append", default=[])
    ap.add_argument("--parity", type=Path, required=True)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--conclusion", type=Path, required=True)
    ap.add_argument("--artifact", action="append", default=[], help="name=relpath")
    ap.add_argument("--profiling-artifact", action="append", default=[], help="name=tool=relpath")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rd = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    res = build(
        rd(a.p3_results),
        rd(a.share),
        [rd(b) for b in a.bench],
        rd(a.parity),
        rd(a.profile) if a.profile else None,
        a.run_id,
        a.commit,
    )
    res["conclusion"] = a.conclusion.read_text(encoding="utf-8").strip()
    for s in a.artifact:
        k, v = s.split("=", 1)
        res["artifacts"][k] = v
    for s in a.profiling_artifact:
        n, t, p = s.split("=", 2)
        res["profiling_artifacts"].append({"name": n, "tool": t, "path": p})
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(statistics.fmean([1.0]) and f"wrote {a.out}")


if __name__ == "__main__":
    main()
