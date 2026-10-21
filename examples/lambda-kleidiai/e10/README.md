# E10: right-size the KV cache (n_ctx)

One variable: the llama.cpp context size `n_ctx` that `snapstart/engine.py` passes to `dec_load_seq`. P1 uses 16384
(with `N_SEQ=17`). Everything else is P1's: zip, weights, runtime version, memory, `N_SEQ`, `DEC_KV=f32`,
`DEC_UBATCH=512`, region. Baseline commit `70eab7be69f86ddeb8c64158d94df344b5180098` (P1's pushed
`feat/lambda-kleidiai-baseline` HEAD when this card started; the card brief named `aac59d4`, P1 has since added
harness fixes on top, and E10 uses the newer HEAD so it gets I1's fresh-client-per-wave fix).

**Result (run `20261021T110426Z`).** n_ctx=4096 cuts what P3 found: on CPU part 0xd40 the restored first forward
falls from 0.669 to 0.560 s (−0.109 s, 95% CI [−0.114, −0.099]), `context_init` from 0.323 to 0.198 s, and Max Memory
Used by 277 MB. The restored request gets 0.200 s faster (−5.1%). Answers are bit-identical, and 4096 refuses nothing
the model's 4,096-token window admits. The cold first request barely changes (−0.08 s of 36 s; the S3 download
dominates), and warm inference does not change. Status `no-change` for the primary arm only because the small 0xd0c
and 0xd4f samples' headline CIs overlap 0; their direction matches. **Recommendation: n_ctx = 4096.**

## Capacity first (the operator decision)

n_ctx is a capacity limit. A request needs `n_ctx ≥ state + its longest question`: `Shim.forward_shared` cuts fork
groups so each fits, and the window fit (`max_length` 4096) has already bounded state + longest question. So any
n_ctx ≥ 4096 refuses nothing that P1 accepts.

| n_ctx | P1's 5 requests | JevBench public (231 tasks, max 3,936 tokens) | documented inputs |
| --- | --- | --- | --- |
| 512 | all answered; incident-review-4q in 2 fork groups (answers change, below) | 64 refused | 1,038- and 3,272-token states refused |
| 1024 | all | 41 refused | both refused |
| 2048 | all | 36 refused | 3,272-token state refused; 1,038-token state + 16 questions in 2 groups |
| 4096 | all | none | all (64 questions in 6 groups) |
| 8192, 16384 | all | none | all |

P1's requests need at most 379 tokens (incident-review-4q: state 301 + longest question 78; all four questions in
one group need 598). JevBench's figures are the release's own `eval/jevbench-w4096/results.jsonl`
(`input_tokens`, one question per task; `evaluation/jevbench.md` says no task is cut at 4096). See
`offline/sizes.json` → `capacity`.

**Over the limit.** P1's code raises `RuntimeError` from decshim's `rc=-2` (or "question 0 does not fit"), which
the HTTP handler turns into a 500. `e10_entry.py` adds one check before any decode: if state + longest question
> n_ctx it raises `ContextTooSmall(ValueError)`, so the Function URL answers 422 with the token counts and how many
to cut, and a direct invocation fails with that message. Nothing is truncated. With n_ctx ≥ 4096 the check can never
fire. Tested in `tests/test_lkai_e10.py` and end to end through P1's `handler.py` offline (422 at n_ctx 1024 for a
1,673-token request).

## Bytes (llama.cpp's own load log; they depend on n_ctx and the context parameters, not the CPU)

| n_ctx | attention K+V | recurrent state | compute buffer | memset per forward (K/V + recurrent) |
| --- | --- | --- | --- | --- |
| 512 | 12 MiB | 327.5 MiB | 96.0 MiB | 339.5 MiB |
| 1024 | 24 MiB | 327.5 MiB | 97.0 MiB | 351.5 MiB |
| 2048 | 48 MiB | 327.5 MiB | 99.0 MiB | 375.5 MiB |
| 4096 | 96 MiB | 327.5 MiB | 105.0 MiB | 423.5 MiB |
| 8192 | 192 MiB | 327.5 MiB | 179.1 MiB | 519.5 MiB |
| 16384 (P1) | 384 MiB | 327.5 MiB | 327.3 MiB | 711.5 MiB |

K/V is 24,576 B per cell (6 attention layers × 2 KV heads × 256 × f32 × K and V). The recurrent state is 18 Gated
DeltaNet layers × 17 sequences: N_SEQ sets it, so E10 does not shrink it. At 4096 the remaining 327.5 MiB of
recurrent state is 77% of what `llama_memory_clear` still memsets per forward; E12 (clear only used cells) owns that.

## Lambda measurements

Six functions `lkai-fbb1-c<n_ctx>` from one zip (P1's `26eb2651…` plus `e10_entry.py`, `2273a5ff…`), differing only in
`E10_N_CTX`. Control `c16384` runs P1's exact arguments. Cold: 6 rounds × 10 environments per arm, then a warm pass
of each of P1's 5 requests. Restore: 6 rounds; each round publishes all six versions together, then 10 restores per
arm and one warm pass. Arm order is a cyclic Latin square (round r = arms rotated by r): every arm takes every
position once, which balances position-in-round drift. Arms are still sequential waves, not simultaneous.
2,880 invocations, 0 errors, 0 start-class mismatches, no `client:ReadTimeoutError` (fresh client per wave).
Cost $8.45 at list prices (operator ceiling $25); the SnapStart cache 3-hour minimum on 36 versions is $5.85 of it.
All functions, the role, the zip and the log groups were deleted afterwards.

Medians on CPU part 0xd40 (n cold ≈ 50, restore ≈ 42–56 per arm), delta vs c16384 with bootstrap 95% CI:

| | 512 | 1024 | 2048 | **4096** | 8192 | 16384 |
| --- | --- | --- | --- | --- | --- | --- |
| context_init (s) | 0.169 | 0.173 | 0.177 | **0.198** | 0.241 | 0.323 |
| Max Memory Used, cold (MB) | 4,773 | 4,791 | 4,818 | **4,883** | 4,984 | 5,160 |
| restored first forward (s) | 0.528 | 0.534 | 0.543 | **0.560** | 0.601 | 0.669 |
| Δ restored first forward | −0.141 | −0.135 | −0.125 | **−0.109 [−0.114, −0.099]** | −0.068 | — |
| Δ Restore Duration | −0.127 | −0.131 | −0.097 | **−0.098 [−0.124, −0.074]** | −0.048 | — |
| Δ restore first-request total | −0.263 | −0.268 | −0.229 | **−0.200 [−0.233, −0.168]** | −0.127 | — |
| Δ cold first-request total | −0.199 | −0.159 | −0.043 | **−0.082 [−0.150, −0.023]** | −0.019 | — |
| Δ warm steady state | −0.008 | −0.006 | −0.009 | **−0.006 [−0.018, +0.010]** | −0.003 | — |
| status (all-parts rule) | inconclusive (answers) | improved | improved | **no-change** | no-change | control |

The effect follows the bytes: memory falls by ~1 MB per MiB of K/V plus compute buffer removed, and the restored
first forward by 0.07 to 0.14 s as K/V drops from 384 to 12 MiB. On 0xd0c (6–12 cold and 4–18 restore environments
per arm) the restored first forward falls 0.17–0.19 s and memory by the same amounts; its first-request CIs are
wider. 0xd4f appeared on 3–5 cold environments per arm and on no restore.
`results.json` → `e10.delta` has every metric per arm and part.

**Host variability.** `e10_entry.py` read, on each environment's first invocation: cpuinfo implementer/part/variant/
revision/features, HWCAP/HWCAP2, CPU count and affinity (6), kernel `5.10.255-268-309.1092.amzn2.aarch64`, Python
3.12.14. cgroup CPU/memory limits are not exposed in the sandbox (`unknown`), nor are AZ and physical host. Three CPU
parts appeared (0xd40, 0xd0c, 0xd4f) with the same kernel. Every comparison above is within one part; no Graviton
generation is inferred from a part. `results.json` → `host_variability` compares each arm with the control in every
shared (round, part) cell. Max Memory Used is negative beyond the control's within-cell range in every assessed cell
for every arm. `context_init` is too, except c8192 on 0xd40 (all six cells negative, some inside the control's
range). For c512 to c4096 the restore headlines are negative in every shared cell, but some cells overlap the
control's spread (the c16384 control's round-3 restore median was 5.1 s against ~3.9 s in other rounds), so they
read `mixed-or-overlapping`; c8192 has one positive 0xd0c restore cell. Warm is within control variation. No rerun
is needed: no cell has the opposite sign for memory or `context_init` on any part, and the 4096 recommendation
rests on 0xd40, where the restore CIs exclude 0. H1's fixed-host EC2
box was not used; the bytes are host-independent and the latency effect was measurable on Lambda directly.

## Correctness

Against P1's reference (`baseline/reference/outputs.json`, same request and CPU part): c1024, c2048, c4096, c8192 and
c16384 are bit-identical on every compared answer (460–468 per arm; the reference has no 0xd4f multi-question answers,
and those rows match the in-run control). c512 changes `incident-review-4q` on every part (68 of 472, max |Δp|
0.0035, no choice flip): at 512 its 4 questions (598 tokens with the state) are split into two fork groups, which
changes the batch and so the floating-point order. `offline/grouping-p1.json` shows c1024 to c16384 form the same
groups for every P1 request. For larger inputs, 4096 splits a 3,272-token state with 64 questions into 6 groups
(16384: 4), so answers there can differ by rounding; P1's set does not exercise it.

## Offline (x86, NOT Lambda)

`offline/README.md`. Besides the sizes, it records a finding outside E10's variable: on this x86 build the
multi-question fork path disagrees with per-question forwards at every n_ctx including 16384 (and gives NaN for
incident-review-4q), while Lambda's arm64 answers do not. It is not caused by n_ctx; its cause is not identified.

## Reproduce

`meta.json` → `reproduce`. Tests: `tests/test_lkai_e10.py`.

## Limits

- One region (us-east-2) and one memory size (10,240 MB). One run; within-run variance only.
- Restore ran on 0xd40 and 0xd0c only; 0xd0c and 0xd4f cold cells are small.
- Sequential waves inside a round (Latin-square position balance, not simultaneous interleaving).
- The JevBench sizes come from the release's GPU run at window 4096, not a Lambda run.
- The recurrent-state memset (327.5 MiB per forward) is untouched: that is E12's.
