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

Not started: the candidate, the correctness tolerance and Lambda spend need the operator's decision (see the card).

## Reproduce (Phase 1)

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/89cf-nonmatmul-ops && git checkout <commit>
# P1's build/ on an arm64 host with Docker (baseline/build.sh): function.zip, model/s2b-h512-Q8_0.gguf, llama.cpp
# with llama/patch_llama.py applied
E15_WORK=/tmp/e15work bash examples/lambda-kleidiai/e15/nodeprof.sh examples/lambda-kleidiai/baseline/build out/
python examples/lambda-kleidiai/e15/summarize_nodeprof.py out/ --json out/table.json --md out/table.md
```
