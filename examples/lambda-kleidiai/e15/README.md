# E15: per-op timing of the warm graph, then one change to the top non-matmul op

Card `task-1792534589-243bd4ed95ec8d96a22f72-89cf` (initiative `lambda-kleidiai`). Branch
`jsamuel1/strands-decider` `lkai/89cf-nonmatmul-ops`, created from P1's pinned baseline
`aac59d45555ebded9357c23f6f8e4f939edf25d7`.

## Phase 1: per-op wall time of the warm forward (OFFLINE, NOT LAMBDA)

Everything in this section is measured off Lambda. Latency conclusions come only from Lambda REPORT lines and P1's
phase timers. This table is for attribution.

### Host and conditions

| | Value |
|---|---|
| Host | P1's arm64 DevSpace `ds-lkai-p1-rs5ghbzv`. `/proc/cpuinfo`: implementer `0x41`, variant `0x1`, part `0xd40`, revision `1`; features include `sve svei8mm i8mm bf16 asimddp`; 16 vCPUs; kernel `6.1.188-233.386.amzn2023.aarch64`. Generation is not inferred from the part ID |
| Image | P1's pinned Lambda build image `public.ecr.aws/lambda/python@sha256:37fe9ca4…` (Python 3.12.15) |
| CPUs | `taskset -c 0-5`, 6 engine threads (Lambda's 10,240 MB function has 6 vCPUs). Container cgroup `cpu.max` = `max 100000` (no CPU quota) |
| Code and model | P1's deployed `function.zip` (sha256 `26eb2651…`, the `code_sha256` in `baseline/reference/outputs.json`), GGUF sha256 `8dbb8872…`, read from local disk into a memfd |
| CPU variant loaded | `libggml-cpu-armv8.6_1.so` (the variant P3 recorded for part `0xd40` in Lambda) |
| Requests | `baseline/requests.json`, 5 requests, 10 passes each after P1's warm-up request: 160 graph computes per run |
| Environment | `DEC_KV=f32 DEC_UBATCH=512 N_SEQ=17`, flash attention off, as deployed |

Unknown, not observed: physical host identity and its other tenants, memory bandwidth, the DevSpace's
hypervisor scheduling.

### Method

[`patch_nodeprof.py`](patch_nodeprof.py) adds an environment-gated timer to `ggml_graph_compute_thread` in
llama.cpp `f74e2fa6a`'s CPU backend. With `LKAI_NODEPROF=1`, thread 0 reads `CLOCK_MONOTONIC` when each graph node
starts and again when the barrier after it releases. Every thread meets that barrier, so each interval is the node's
wall time across all 6 threads, load imbalance included, and consecutive intervals tile the graph: node time sums to
99.95% of graph time in both rounds. Rows are keyed by op, node name, source types, source-0 buffer type and name,
with layer numbers removed. Bookkeeping runs after the graph's final barrier, never inside a timed node.

[`nodeprof.sh`](nodeprof.sh) rebuilds only the `libggml-cpu-*.so` variants with the patch, using P1's flags and
image, and swaps them into a copy of the unzipped `function.zip`. It then runs `profiling/offline_drive.py` (P3's
driver, plus a reset/dump around steady state) interleaved as ctl, off, on, ctl, off, on, and then
`perf record --call-graph dwarf` on P1's own libraries.

| Run | Libraries | Steady-state total, 50 requests (s) |
|---|---|---|
| ctl1 / ctl2 | P1's | 38.515 / 38.354 |
| off1 / off2 | patched, `LKAI_NODEPROF=0` | 38.658 / 38.499 |
| on1 / on2 | patched, `LKAI_NODEPROF=1` | 38.623 / 38.763 |

ctl1 and ctl2 differ by 0.4%, which is this host's run-to-run spread. off is within 0.4% of ctl, and on is within
0.6% of off. The profiler does not move totals beyond the drift between identical runs. Answers from all six runs are
bit-identical to each other and to P1's `0xd40` reference (`baseline/reference/outputs.json`).

### Result: share of warm graph wall time by op

Round 1 / round 2. Shares are of the summed wall time of all 160 graph computes. "ms per forward" is the mean per
graph compute.

| Category | Share of warm graph wall time | ms per forward | Nodes per forward |
|---|---|---|---|
| KleidiAI matmul (Q8_0 weights, i8mm) | 60.47% / 60.46% | 141.19 / 141.57 | 186 |
| **GATED_DELTA_NET** | **9.62% / 9.59%** | 22.46 / 22.46 | 18 |
| **attention KQV (F32 MUL_MAT, `ggml_vec_dot_f32`)** | **5.40% / 5.39%** | 12.60 / 12.62 | 6 |
| **attention KQ (F32 MUL_MAT, `ggml_vec_dot_f32`)** | **5.21% / 5.20%** | 12.16 / 12.17 | 6 |
| CONCAT (E14's) | 4.93% / 5.01% | 11.52 / 11.73 | 18 |
| Hadamard rotation (MUL_MAT with the Hadamard hint, which calls `ggml_compute_forward_fwht`) | 4.39% / 4.39% | 10.26 / 10.28 | 96 |
| SSM_CONV | 2.65% / 2.63% | 6.18 / 6.17 | 18 |
| RMS_NORM (+ fused MUL) | 1.97% / 1.96% | 4.60 / 4.59 | 115 |
| SWIGLU | 1.12% / 1.13% | 2.63 / 2.64 | 24 |
| SILU | 1.05% / 1.04% | 2.46 / 2.45 | 36 |
| every other op (MUL, SOFT_MAX, GET_ROWS, CPY, SCALE, ADD, SIGMOID, SET_ROWS, ROPE, CONT, SOFTPLUS) | 3.2% | 7.4 | |

Full rows, per round: [`results/offline-20261021T0950Z/table.md`](results/offline-20261021T0950Z/table.md),
`table.json`, and the raw per-key tables `nodeprof1.tsv` / `nodeprof2.tsv`.

The KleidiAI share here (60%) is larger than its CPU-sample share in P3 (about 50%). Wall time per node includes
threads waiting at the barrier, and CPU samples include libgomp spinning, so the two measures are not expected to
match.

### `ggml_vec_dot_f32` is attention

- perf (dwarf call graph, P1's libraries, [`perf-vec_dot_f32-callers.txt`](results/offline-20261021T0950Z/perf-vec_dot_f32-callers.txt)):
  every `ggml_vec_dot_f32` sample is under `ggml_compute_forward_mul_mat`. That is P3's `?` caller.
- Node table: the only `MUL_MAT` nodes that are neither KleidiAI nor Hadamard are the 6 full-attention layers' `kq`
  (K·Q, `cache_k` view × permuted, non-contiguous Q) and `kqv` (V·softmax). Both are F32 × F32. Together they take
  10.6% of graph wall time.
- Why they take the scalar-per-row path: `ggml-cpu.c` at this commit has
  `#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)` → `#undef GGML_USE_LLAMAFILE`. The armv8.6_1
  variant (SVE + i8mm), the one `0xd40` loads, therefore never calls llamafile's F32 `tinyBLAS` sgemm, although
  `sgemm.cpp` is compiled into it. `mul_mat` falls through to the per-row `ggml_vec_dot_f32` loop. The armv8.2 variants
  (no SVE, no i8mm, e.g. `0xd0c`'s `armv8.2_2`) keep llamafile. On those parts `kqv`, whose src1 is contiguous, would
  take tinyBLAS, while `kq` (src1 non-contiguous) still would not. This is a source reading; it was not measured on
  `0xd0c`.

### Phase 1 conclusion

On this host, by node wall time, the largest costs outside the KleidiAI Q8_0 matmul are:

1. full attention's two F32 matmuls, KQ + KQV, 10.6% (on `ggml_vec_dot_f32`, because llamafile is compiled out of
   the SVE/i8mm variants);
2. GATED_DELTA_NET, 9.6% (already the baseline's single-pass NEON kernel, `llama/patch_llama.py` `gdn`);
3. CONCAT, 5.0% (E14 owns it);
4. the Hadamard rotation, 4.4%.

Offline is not Lambda. P3 found warm Lambda concat at 5.6% of CPU samples against 1.2–2.65% offline on the same part
ID, so Lambda's ranking near the 5% level can differ. The top two are 1 point apart here, and both are well above the
next op.

## Phase 2

Operator decisions, 2026-10-21:
- Probe candidates A and B offline, then pick one for Lambda.
- Tolerance against P1's reference for the same CPU part: max |Δprobability| ≤ 0.02 and 0 choice flips.
- Lambda runs go in P1's account, us-east-2, with function names `lkai-89cf-*`.

### Offline probe of candidates A and B (OFFLINE, NOT LAMBDA)

[`patch_candidates.py`](patch_candidates.py) builds both candidates into one set of `libggml-cpu-*.so`. Each is off
unless its runtime switch is set:

- **A**: llamafile tinyBLAS F32 sgemm for F32 × F32 `MUL_MAT` in the SVE / i8mm variants, the path upstream compiles
  out. Quantised types keep their own path, and non-SVE variants are unchanged.
- **B**: a row-blocked GATED_DELTA_NET for the baseline's single-pass case. Each block of 16 state rows runs over all
  tokens while resident in L1, and threads get `H × n_seqs × S_v/16` work items instead of `H × n_seqs`. Every element
  sees the same operations in the same order.

The DevSpace is shared. Other cards' offline jobs pinned CPUs 0–5 (94–98% busy) during this probe. Two
sequential-arm attempts (`probe.sh`) gave ctl totals of 241 s and 41–66 s against Phase 1's 38.4 s. Their timings
are discarded; their answers are still used, because the arithmetic does not depend on timing.

[`paired_drive.py`](paired_drive.py) (run by [`paired.sh`](paired.sh)) loads the engine once on CPUs 6–11 (2–6%
busy before the run, system loadavg about 11 from the other cards). It switches arms per request (ctl, A, B) in a
rotating order, so drift from other tenants hits every arm alike and cancels in the per-request paired difference.
The result has 10 passes × 5 requests = 50 pairs per arm, with bootstrap 95% intervals of the median.

| Arm | Median paired change per warm request vs ctl | 95% CI | Correctness vs P1 `0xd40` reference |
|---|---|---|---|
| A (F32 sgemm) | **−1.88%** | −2.10% to −1.64% | max \|Δ\| 0.0101 over 29 values, 0 flips (within tolerance); same in all 5 runs of A |
| B (row-blocked GDN) | −0.19% | −0.52% to −0.09% | bit-identical |

Per request, A's median is −1.6% to −2.4% on all five, and its interval excludes 0 on four (`cracked-screen-2q`:
−2.1% to +0.6%). ctl with the candidate library (switches off) gives answers bit-identical to P1's libraries
(`bitcheck-*.json`). Results: [`results/offline-paired-20261021T1105Z/`](results/offline-paired-20261021T1105Z/).

Why B gives so little: the per-node table times GATED_DELTA_NET at 22.5 ms of a 233 ms forward, but the kernel is
already the single-pass NEON one. Row blocking changes cache residency and the thread split. It does not change the
work, and on this host the state does not appear to be bandwidth-bound.

Why A is smaller than the 5.4% KQV share: tinyBLAS takes KQV only. KQ's src1 (Q, permuted) is not contiguous, so KQ
stays on `ggml_vec_dot_f32`. The gain is about a third of KQV's time. The arithmetic now goes through tinyBLAS's
FMA tiles instead of a per-row dot product, so outputs change within the tolerance.

Scope note: the shipped zip's portable libraries also serve other part IDs. On `0xd0c` (`armv8.2_2`, no SVE),
llamafile is already on, so A changes nothing there. On `0xd4f` (`armv8.6_2`), A applies as on `0xd40`. It is not
measured on either.

### Lambda run of candidate A (run `20261021T2340Z`)

The operator chose A, with a $50 ceiling, in P1's account, us-east-2.

**Result: no deployment gain shown.** On Lambda, candidate A is about 1% faster per warm request on part `0xd40`,
not the 1.9% measured offline, and it does not pass the pre-declared verdict rule. Answers stay within the accepted
tolerance. Status is `no-change` for both arms.

| Arm | What differs | Warm REPORT Duration vs ctl, part `0xd40`, 9 paired rounds | Status |
|---|---|---|---|
| ctl | P1's zip `26eb2651` plus `e15_entry.py` (zip `02c39f7f`) | median 613 ms | baseline |
| off | zip `4a1286cd`: only the six `lib/libggml-cpu-*.so` rebuilt with `patch_a.py`, switch unset | −0.69% mean, 8/9 rounds below 0, p = 0.027 | no-change |
| a | the same zip with `LKAI_E15_A=1` | −1.17% mean, 8/9 rounds below 0, p = 0.0078; median 607 ms | no-change |
| a vs off | the switch alone | −0.48% mean, 7/9 rounds below 0, p = 0.043 | no-change |

E13's rule calls an arm `faster` only if every round is below 0, the sign-flip p is under 0.05, and the mean exceeds
the control's own split-half noise (1.04% on `0xd40`). `a` fails on two counts: one round of 9 is +0.36%, and its
1.17% mean is close to that noise. The engine's own timer (−1.18%) and process CPU per request (−1.14%) agree with
REPORT.

The dormant-build control (`off`, switch unset) is not the same as `ctl` in timing: −0.69%, 8/9 rounds below 0.
Its answers are bit-identical, so its arithmetic is P1's. That rebuilding the variant libraries alone moves warm
time by a fraction of a percent (code layout or alignment) is an inference, not a measurement. It means about 0.7 of
`a`'s 1.17 points vs ctl are not attributable to the switch. The switch alone is −0.48%.

Other CPU parts are too few rounds for p < 0.05 (`underpowered`):
- `0xd4f` (`armv8.6_2`, changed): `a` −2.5%, 4/4 rounds below 0.
- `0xd0c` (`armv8.2_2`, unchanged by A): `a` −0.9%, 3/4 rounds below 0. Its answers are bit-identical to ctl
  (144/144), as the negative control predicts.

Side effects on `0xd40`, `a` − ctl, bootstrap 95% CI:
- cold first request +0.10 s [−0.01, +0.20], n = 43/46
- restored first request +0.03 s [−0.01, +0.07], n = 26/30
- cold max memory +0 MB

**Correctness** (tolerance accepted by the operator: max |Δprobability| ≤ 0.02, 0 choice flips):
- `a` against ctl, same request and CPU part: 1,440 answers, max |Δ| 0.0148 (on `0xd4f`), 0 flips. On `0xd40`
  the maximum is 0.0101.
- `a` against P1's `reference/outputs.json`: 1,380 answers, max |Δ| 0.0101, 0 flips. Within tolerance. Only the
  `0xd0c` answers are bit-identical.
- `off`: 1,440/1,440 bit-identical to ctl, and 1,344/1,344 bit-identical to P1's reference.

A changes floating-point summation order, from per-row dot products to tinyBLAS FMA tiles, which explains the
non-identical answers.

**Errors:** 0 function errors, 0 client errors, and 0 restore rows with `client:ReadTimeoutError` across 4,320
invocations. Every wave used a new invoke client (I1).

**Cost:** $2.97 at list prices ($1.46 of it the SnapStart cache 3-hour minimum), under the $50 ceiling. Teardown
removed every `lkai-89cf-*` function and version, the log groups, the role and the `e15/` S3 prefix. A list of each
afterwards returned nothing.

**Host variability.** CPU identity was read in every environment from `/proc/cpuinfo` and `AT_HWCAP`/`AT_HWCAP2`:

| CPU part | Variant | HWCAP / HWCAP2 | ggml variants mapped | Changed by A |
|---|---|---|---|---|
| `0xd40` | `0x1` | `0xfffffff` / `0x1f201` | `armv8.0_1`, `armv8.6_1` | yes |
| `0xd4f` | `0x0` | `0x2ff3ffff` / `0x3f3bf` | `armv8.0_1`, `armv8.6_2` | yes |
| `0xd0c` | `0x3` | `0x119fff` / `0x0` | `armv8.0_1`, `armv8.2_2` | no |

Environments, counted from the first-invoke record over all arms and classes: 216 `0xd40`, 32 `0xd0c` and 22
`0xd4f`. Every environment reported:
- 6 CPUs, `sched_getaffinity` 6
- kernel 5.10.255, Python 3.12.14
- 12 threads after the first invocation

The cgroup limit files are not exposed and are recorded as `unknown`. The availability zone, physical host and CPU
frequency are also `unknown`. No CPU generation is inferred.

Comparisons are within one part, and the warm arms are interleaved pass by pass within each round.
`host_variability.observed.primary_metric_cells` holds every shared (round, part) cell. The pooled warm medians
(`delta.warm`) are not used for the verdict.

The offline-to-Lambda gap (1.9% offline vs 1.2% on Lambda, same part ID) is not attributed to code. The offline
host runs kernel 6.1 and Lambda runs 5.10, and P3 saw the same kind of gap for `concat`.

**Recommendation for R:** do not adopt A on its own; the measured gain is about 1% and not distinguishable from the
control's split-half noise. Its correctness cost is within tolerance. More rounds would not change the verdict under
this rule. Per-round deltas on `0xd40` have SD 0.94 points, so with a true −1.2% mean each round has about an 11%
chance of landing above 0 (normal approximation). Requiring every round below 0 gets harder as rounds are added:
the probability is about 25% at 9 rounds and 5% at 20. A ladder rung that wants to resolve A needs a different
pre-declared test, such as the sign-flip p plus the split-half noise floor without the every-round condition,
stated before it measures.

## Reproduce (Phase 1)

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/89cf-nonmatmul-ops && git checkout <commit>
# P1's build/ on an arm64 host with Docker (baseline/build.sh): function.zip, model/s2b-h512-Q8_0.gguf, llama.cpp
# with llama/patch_llama.py applied
E15_WORK=/tmp/e15work bash examples/lambda-kleidiai/e15/nodeprof.sh examples/lambda-kleidiai/baseline/build out/
python examples/lambda-kleidiai/e15/summarize_nodeprof.py out/ --json out/table.json --md out/table.md
# Phase 2 offline probe (pick 6 idle CPUs on a shared host)
E15_CPUS=6-11 E15_WORK=/tmp/e15work bash examples/lambda-kleidiai/e15/paired.sh examples/lambda-kleidiai/baseline/build paired/ 10
```
