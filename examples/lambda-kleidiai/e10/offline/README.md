# E10 offline probes (x86, NOT Lambda)

Host: Intel Xeon 6975P-C, 16 threads, llama.cpp `f74e2fa6a` with `llama/patch_llama.py`, built `GGML_NATIVE=ON`
(no KleidiAI: x86). Weights: P1's GGUF `8dbb8872…`. App code: P1's zip (`26eb2651…`) `engine.py`, `handler.py`,
`strands_decider/`, `assets/`, on the host's numpy/tokenizers (the zip ships arm64 wheels).

| file | what |
| --- | --- |
| `offline_probe.py`, `probe-<n_ctx>.json` | one process per n_ctx: llama.cpp's buffer sizes (its load log), `context_init`, `llama_memory_clear(mem, true)` time (30 calls), RSS, answers for `requests.json` |
| `requests.json` | P1's five requests, `evaluation/device_parity.py`'s 18, and three documented long inputs (a ~3,300-token state with 1 and 64 questions, a ~2,100-token state with 16) |
| `sizes.json` (`offline_sizes.py`) | the buffer sizes per n_ctx, checked against the closed form, and what each n_ctx refuses for P1's requests and JevBench |
| `jevbench-w4096-tokens.json` | `usage.input_tokens` of all 231 JevBench public tasks, from the release's own `eval/jevbench-w4096/results.jsonl` |
| `p1-context-need.json`, `ctx_need.py` | the context each P1 request needs |
| `fork_groups_probe.py`, `fork-groups-16384*.json` | the multi-question finding below |

## Buffer sizes (they depend on n_ctx and the context parameters, not on the CPU)

| n_ctx | attention K+V | recurrent state | compute buffer | cleared per request (K/V + recurrent) |
| --- | --- | --- | --- | --- |
| 512 | 12 MiB | 327.5 MiB | 96.0 MiB | 339.5 MiB |
| 1024 | 24 MiB | 327.5 MiB | 97.0 MiB | 351.5 MiB |
| 2048 | 48 MiB | 327.5 MiB | 99.0 MiB | 375.5 MiB |
| 4096 | 96 MiB | 327.5 MiB | 105.0 MiB | 423.5 MiB |
| 8192 | 192 MiB | 327.5 MiB | 179.1 MiB | 519.5 MiB |
| 16384 (P1) | 384 MiB | 327.5 MiB | 327.3 MiB | 711.5 MiB |

Attention K/V is 24,576 bytes per cell (6 attention layers × 2 KV heads × 256 × f32 × K and V). The recurrent
state is 18 Gated DeltaNet layers × 17 sequences, so n_ctx does not change it. `dec_eval` and `dec_eval_fork` call
`llama_memory_clear(mem, true)` before every forward, which memsets both buffers. Going from 16384 to 4096 removes
288 MiB of attention K/V (40% of the cleared bytes) and 222 MiB of compute buffer.

## The multi-question path is wrong on x86, independent of n_ctx

At P1's n_ctx 16384, every P1 request with more than one question gives answers on x86 that differ from P1's
arm64 reference by up to 0.67 in probability, and `incident-review-4q` returns NaN. `fork_groups_probe.py` shows the
shared-state fork's hidden rows for several questions in one `dec_eval_fork` call differ (max |Δ| 53 to 142) from
the same questions decoded one fork call each, or one plain `dec_eval` each. Those two agree with each other
(≤ 0.08), and P1's `gdn` patch is not the cause (`-nogdn`: same result, and then per-question and plain agree
exactly). P1's arm64 reference answers for those same multi-question requests (Lambda, one fork call) match x86
single-question forwards to ≤ 0.013. So the discrepancy appears only on this x86 build; n_ctx does not cause it
(it is present at P1's 16384), and the cause inside llama.cpp's multi-sequence path is not identified. Consequence
for E10: offline answer comparisons are read only for single-question requests; Lambda's answers decide
correctness.
