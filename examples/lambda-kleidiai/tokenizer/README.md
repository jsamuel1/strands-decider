# lambda-kleidiai T0: tokenizer identity, parity suite, baseline and profile

T0 measures the tokenizer that P1's deployed Lambda uses. It changes no code on the request path. Its results
gate T1 (optimise the Python implementation), T2 (compiled hot paths) and T3 (an off-the-shelf tokenizer).

**Finding.** Tokenization is negligible end to end, and the deployed tokenizer is not pure Python. It is Hugging Face
`tokenizers` 0.23.2, a Rust core behind a PyO3 binding. On a warm request it costs 0.83 ms (p50), against a 654 ms
handler. Its only material cost is `Tokenizer.from_file`, about 0.6–0.8 s, which runs once per on-demand
environment. Details, limitations and the T1–T3 assessment are in
[`results/20261021T102818Z/conclusion.txt`](results/20261021T102818Z/conclusion.txt).

Base: P1's pushed tip `70eab7b` (`feat/lambda-kleidiai-baseline`). The deployed artifact is P1's `function.zip`,
sha256 `26eb2651…`.

## Identity and semantics

| | |
|---|---|
| Implementation | `tokenizers` 0.23.2 wheel in `function.zip`: `tokenizers/tokenizers.abi3.so`, sha256 `999ab0b7…`. The `.so` is an aarch64 ELF (not stripped). Its `.comment` is `rustc version 1.98.0`. It exports `PYO3_INTROSPECTION_*` symbols and links the onig 6.5.3, regex 1.13.1, rayon 1.12.0, serde_json 1.0.151 and unicode-normalization-alignments crates. Python code calls `from_file`, `encode`, `encode_batch` and `decode`. Everything else runs in Rust |
| Call path | `lkai_entry.handler` → `app.handler` → PR #77 `handler.handler` → `_evaluate` → `engine.HeadRowsEngine.evaluate` → `strands_decider.hidden_engine.HiddenStateEngine.fit` → `Tokenizer.encode_batch(questions, add_special_tokens=False)` and `Tokenizer.encode(state, add_special_tokens=True)` |
| Construction | `HiddenStateEngine.from_assets` → `Tokenizer.from_file(assets/tokenizer.json)`, then `no_truncation()` and `no_padding()` |
| Where init runs | On demand: inside PR #77's `_load` in a background thread. The order is weight download, llama.cpp load, then assets. That is after REPORT Init Duration ends and before the cold first request is answered. With SnapStart: at import, before the snapshot. A restore does not run it again |
| Model | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` at `216d1ccb`. Base model `Qwen/Qwen3.5-2B-Base`. `tokenizer_class` is `Qwen2Tokenizer` |
| Vocabulary file | `assets/tokenizer.json`, 19,989,592 bytes, sha256 `a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86`. This is byte-identical to the Hub file at that revision |
| Algorithm | Byte-level BPE. Vocab 248,044, merges 247,587, 33 added tokens (21 special), 248,077 ids in total. No `byte_fallback`, no `unk_token`, no dropout |
| Normalizer | NFC |
| Pre-tokenizer | Qwen's split regex (contractions, letter runs with an optional leading non-letter, single digits, punctuation runs, newline runs, whitespace), `Isolated`, then ByteLevel (`add_prefix_space` false) |
| Post-processor / decoder | ByteLevel. `add_special_tokens=True` adds no BOS/EOS for this file |
| Special tokens in user text | `split_special_tokens` is false, so a special-token literal in the state or a question becomes that token's id. The golden records this for every special and added token, and for partial and case variants |
| Chat template | `chat_template.jinja` exists on the Hub, but it is neither shipped nor used. The engine renders its own `<state>`/`<question>`/`<answer>` prompt (`strands_decider.prompting`) |
| Truncation | The file sets Right/LongestFirst at 4096, and the engine turns it off. `fit()` then reserves `min(longest question, 0.75 × 4096 = 3072)` tokens. It cuts each question from the front down to the reserve and truncates the state from the right to `4096 − reserve` |
| Padding | The file sets BatchLongest. The engine calls `no_padding()` |
| Max length | Window 4096. `MAX_REQUEST_TOKENS` 24,576 (`handler.py`). A request above that limit raises ValueError before the forward pass (HTTP 422 on the Function URL) |
| Errors | **Lone surrogate**: `encode` raises `TypeError: TextInputSequence must be str`. On the Function URL path, pydantic's JSON parser rejects the `\ud800` escape first (422), so the tokenizer is never called. On the direct Invoke path, the runtime's `json.loads` yields the surrogate, validation accepts it, and `fit` raises TypeError, which `handler.handler` does not catch (an invocation error). **Invalid UTF-8**: the bytes never reach the tokenizer, because body decoding fails first. **Empty**: `encode("")` returns `[]`. **Oversize**: the state is trimmed to the window, and a request over the cap is refused |

## Parity suite (for T1–T3)

- `data/corpus.jsonl`: sha256 `e436930a70d4cdc8e1e14eb77ba72ac6b1e2b13b2e54af6bddef4dfb6746e0a4`, 217 cases. It is built deterministically by `tokcorpus.py` and covers:
  - the P1 request set, plus the handler's warm-up request;
  - five state-length tiers, up to 2× over the window;
  - a question over its reserve, and a request over the 24,576-token cap;
  - batches of 1, 2, 3, 4, 8 and 16 questions (16 is the most one fork group takes at `N_SEQ=17`);
  - edge cases: CJK, emoji (ZWJ sequences and flags), decomposed and stacked combining marks, Arabic and Hebrew with bidi controls, control characters, zero-width characters, whitespace runs, lone surrogates, prompt-marker literals, and every special and added token literal.

  Special-token strings are read from `tokenizer.json`, never written in source. The file is ASCII with `<` escaped.
- `data/golden.jsonl.gz`: sha256 of the uncompressed data is `ea093b07…`. It was recorded with the deployed bytes in P1's pinned Lambda image on Graviton3. For each case it holds:
  - the ids and offsets;
  - `decode(ids)` and whether the decode round-trips exactly;
  - the `fit()` outputs (state ids, question ids and offsets, and the option rows the head reads);
  - the outcome stage and the error type.
- `python tokparity.py check --corpus data/corpus.jsonl --tokenizer <tokenizer.json> --golden data/golden.jsonl.gz [--impl module:factory]` exits non-zero on any difference. `tests/test_lkai_tokenizer_parity.py` runs it against the installed tokenizer and proves that a single changed id, offset, decode result, error type or missing case fails it. Rows are identical on x86_64 with tokenizers 0.23.3.

## Reproduce

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/951d-tok-baseline && git checkout <commit>
python -m pytest tests/test_lkai_tokenizer_parity.py
# Fixed host: an arm64 box with Docker and P1's function.zip (sha256 26eb2651...). T0_CPUS = 4-6 CPUs no other job uses.
T0_CPUS=6-9 examples/lambda-kleidiai/tokenizer/fixed_host.sh <function.zip> out golden g1
T0_CPUS=6-9 examples/lambda-kleidiai/tokenizer/fixed_host.sh <function.zip> out bench r2
T0_CPUS=6-9 examples/lambda-kleidiai/tokenizer/fixed_host.sh <function.zip> out profile p1
T0_CPUS=6-9 examples/lambda-kleidiai/tokenizer/fixed_host.sh <function.zip> out bench r3
python examples/lambda-kleidiai/tokenizer/perf_summary.py out/perf-steady.folded out/perf-init.folded --out perf-summary.json
# Lambda share, from P3's published logs (no AWS calls):
for a in cprofile sampler; do git show fork/lkai/bc9a-profiler:examples/lambda-kleidiai/profiling/results/20261020T231634Z/logs/lkai-p3-$a.jsonl.gz > logs/lkai-p3-$a.jsonl.gz; done
python examples/lambda-kleidiai/tokenizer/lambda_share.py --p1 examples/lambda-kleidiai/baseline/results/20261020T080835Z/results.json --p3-logs logs --out lambda-share.json
```

## Files

| File | What it does |
|---|---|
| `tokcorpus.py` | Builds the versioned corpus |
| `tokparity.py` | Records the golden and checks an implementation against it |
| `tokbench.py` | Fixed-host microbenchmark. Init (import, `from_file`) runs in fresh child processes. Steady-state `fit`, encode-only and decode use warmup and per-round repeats. Also records `tracemalloc`, GC and RSS |
| `toksoak.py` | Loop for profilers, and the profiler-overhead control |
| `fixed_host.sh`, `tok_container.sh` | Run golden, bench or profile in P1's pinned Lambda image with the deployed zip |
| `stackcollapse.py`, `perf_summary.py` | Fold `perf script` output and compute hot-spot shares over all samples |
| `lambda_share.py` | Lambda tokenizer share from P1 and P3 runs |
| `make_result.py` | Writes the P2 `lkai-result/1` file |
| `results/20261021T102818Z/` | Published run: conclusion, Lambda share, fixed-host benches, perf data and flame-graph stacks (`*.folded.gz`) |
