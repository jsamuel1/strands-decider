# E8: multi-threaded KleidiAI weight repack at model load (cold load)

**Result (run `20261021T110412Z`, status `improved`, cold start only).** Six repack threads cut cold
`model_load_mmap` on CPU part 0xd40 from 8.42 s to 2.88 s. The cold first request (Init + Duration) falls from
36.00 s to 30.60 s, a change of −5.41 s (95 % CI [−5.52, −5.34], n 44/40). Every round on every CPU part has the
same sign. Restore, warm, Init Duration and Max Memory Used do not change. Answers are bit-identical to P1's
reference (1,800 / 1,800), and every loaded weight is byte-identical to P1's libraries at 1, 2, 3 and 6 threads.
The run cost $3.53 at list prices.

The status rule was amended after the results were seen, at the operator's direction. The rule written before the
run gives `inconclusive`: one sparse stratum of the switch-off check (restore on 0xd0c, n 2/4) excluded 0, with
the patched libraries *faster*. The amended rule counts only strata with n ≥ 5 per arm. Both verdicts are in
`results.json` → `e8.status_rule`. See "Status rule" below.

## The variable

At model load, KleidiAI repacks every Q8_0 weight in two loops, one tensor at a time on the loading thread
(`tensor_traits::repack` in `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, called from
`ggml_backend_cpu_kleidiai_buffer_set_tensor` inside `llama_model_load_from_file`):

1. Requantisation turns each Q8_0 row into int8 values and one per-row scale.
2. `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` interleaves groups of `nr` rows into the kernel's layout.

P3's offline profile found `model_load` on one thread. The download finishes before the load starts, so 5 of the
6 vCPUs are idle during the repack.

`patch_parallel_repack.py` (applied after P1's `llama/patch_llama.py`) adds one switch,
`GGML_KLEIDIAI_REPACK_THREADS=N`:
- With N > 1, each loop splits one tensor into N contiguous row blocks on N `std::thread`s. The pack's blocks
  start on multiples of `nr`, so each packed row group is written by exactly one thread, from the same rows and
  scales the serial pack reads.
- Unset, empty, 0 or 1 runs upstream's code. The serial loops are kept byte for byte as the `else` branches.
- Q6_K and every other weight type stay serial. The packing routine and its parameters do not change.

KleidiAI is compiled into each `libggml-cpu-*` variant, so those six libraries are the only files the patch
changes.

| Arm | Zip | Switch | Purpose |
|---|---|---|---|
| `base` | `base.zip` 9cbe63c4: P1's deployed zip 26eb2651 byte for byte, plus `e8_entry.py` and `e8_host.py` | unset | P1's libraries |
| `off` | `off.zip` 0d7f8a51: `base.zip` with only the six `lib/libggml-cpu-*.so` replaced (`zips.json`) | unset | the dormant patch |
| `par` | `off.zip` | `6` | the one variable |

`par − off` is the effect (same zip, one environment variable). `off − base` checks that the patch is dormant.
All three functions copy P1's deployed configuration:
- python3.12, arm64, 10,240 MB, 512 MB ephemeral storage, 900 s timeout, SnapStart on published versions
- P1's environment with `LKAI_PHASES=1`, P1's pinned runtime ARN, and P1's weights object (read-only)

They live in their own functions (`lkai-5a61-*`), role, log groups and `e8/` S3 prefix. All of these were deleted
after the run (functions, role, log groups and zips verified gone).

## Offline checks (fixed host, NOT Lambda)

Fixed host: arm64 DevSpace, CPU part 0xd40 (implementer 0x41, variant 0x1, revision 1), kernel 6.1. P1's pinned
Lambda image, `--cpuset-cpus` 10-15 (6 CPUs). Other cards' containers shared the host: load average was about 13 on
16 CPUs.

**Byte identity** (`hash_check.sh`, `kaihash.cpp`). kaihash loads P1's GGUF (sha256 8dbb8872) with one set of
libraries and reads every weight back through `llama_internal_get_tensor_map`. For a KleidiAI tensor it hashes
exactly what repack writes: the header fields and each packed slot, never the alignment padding. It needs no
change to the libraries, so the same binary checks P1's unpatched libraries.

- Configurations: P1's libraries, `off`, and 1, 2, 3 and 6 threads, each 3 times, interleaved and rotated.
- **All 18 runs match P1's libraries on all 320 tensors** (187 KleidiAI-packed, 1,891,124,360 bytes). The
  whole-model hash is `954cb356a270b4c9` in every run (`results/…/offline-hash/`).

Load times are a fixed-host proxy only:

| Threads | P1 libs | 1 (`off`) | 1 | 2 | 3 | 6 |
|---|---|---|---|---|---|---|
| `model_load_s` median of 3 | 7.94 | 8.05 | 7.95 | 4.89 | 3.62 | 3.28 |

An earlier build used a rewritten serial loop instead of upstream's text. It was also 18 / 18 identical, with
6 threads at 2.26 s. The 2.26 s against 3.28 s gap tracks the co-tenant load and is not attributed to the code.
Scaling flattens after 3 threads on this host, and the serial share (I/O setup, buffer allocation, the zero-fill)
stays serial.

**Answers** (`probe_e8.sh`). The probe used P1's engine environment (`DEC_KV=f32`, `DEC_UBATCH=512`, `N_SEQ=17`)
and the GGUF in a memfd, 2 rounds × 3 arms. All 5 requests in all 6 loads are bit-identical to
`baseline/reference/outputs.json` for 0xd40. Fixed-host `model_load_mmap`: base 8.03 / 8.14 s, off 8.12 / 8.08 s,
par 2.48 / 2.43 s. A first probe without P1's engine environment differed from the reference on every arm,
`base` included. That was a probe setup error and is superseded.

## Lambda measurement (run `20261021T110412Z`, us-east-2)

The harness is P1's `baseline/bench.py` at 70eab7b, driven by `bench_e8.py`:
- 5 rounds × 3 arms. Each round runs a cold wave of 10, publishes a version, runs a restore wave of 10, then 10
  warm passes of 10 per arm.
- Each wave uses a new Lambda client (I1's finding) and `--read-timeout 300`.
- The arm order reverses every round, and warm passes alternate. This is alternation, not simultaneous
  interleaving, so drift within a round is a limitation.

1,800 invocations. 0 errors, 0 class mismatches, 0 `client:ReadTimeoutError`, 0 rows without a CPU part.

| class | metric | CPU part | off p50 | par p50 | Δ | 95 % CI | n off/par |
|---|---|---|---|---|---|---|---|
| cold | model_load_mmap (s) | 0xd40 | 8.42 | 2.88 | −5.540 | [−5.592, −5.493] | 44/40 |
| cold | model_load_mmap (s) | 0xd0c | 19.61 | 5.32 | −14.293 | [−14.365, −14.205] | 3/6 |
| cold | model_load_mmap (s) | 0xd4f | 7.09 | 2.37 | −4.714 | [−4.779, −4.641] | 3/4 |
| cold | first_request_total (s) | 0xd40 | 36.00 | 30.60 | −5.406 | [−5.523, −5.340] | 44/40 |
| cold | first_request_total (s) | 0xd0c | 48.61 | 34.21 | −14.403 | [−14.966, −14.135] | 3/6 |
| cold | first_request_total (s) | 0xd4f | 34.20 | 29.87 | −4.322 | [−4.815, −3.970] | 3/4 |
| cold | report_init_duration (s) | 0xd40 | 0.80 | 0.80 | +0.001 | [−0.011, +0.016] | 44/40 |
| cold | report_max_memory_mb | 0xd40 | 5159.5 | 5158.0 | −1.5 | [−9.5, +0.5] | 44/40 |
| restore | report_restore_duration (s) | 0xd40 | 3.25 | 3.23 | −0.024 | [−0.052, +0.012] | 46/46 |
| restore | first_request_total (s) | 0xd40 | 3.92 | 3.90 | −0.018 | [−0.047, +0.014] | 46/46 |
| restore | first_request_total (s) | 0xd0c | 5.09 | 5.07 | −0.024 | [−0.126, +0.060] | 4/4 |
| warm | steady_state_inference (s) | 0xd40 | 0.66 | 0.64 | −0.016 | [−0.073, +0.073] | 460/460 |
| warm | steady_state_inference (s) | 0xd0c | 1.16 | 1.16 | +0.001 | [−0.543, +0.514] | 40/40 |

The cold `model_load_mmap` and first-request deltas are negative in every (round, CPU part) cell. On 0xd40 the five
per-round first-request deltas run from −5.58 to −5.32 s.

Per-round medians of the cold first request (all CPU parts) have a range of 0.20 s (base), 0.20 s (off) and
0.22 s (par), coefficient of variation 0.2–0.3 %.

Full distributions per arm and phase (p50/p90/p99, min/max, histogram) are in `results.json` → `phases`, and
per-cell deltas in `e8.pairs`.

**Why the request gains less than the load phase.** The on-demand load runs in PR #77's background thread after
Init, so Init Duration does not change. The 25 s S3 download (`weight_download` 25.0 s p50, unchanged) runs before
the load. The first request therefore saves exactly the load's saving (`model_ready` −5.42 s on 0xd40).

**Restore.** The SnapStart snapshot init also loads faster: snapshot-init `model_load_mmap` 8.23 s (off) against
2.72 s (par), so `publish_version` is about 10 s shorter. A restore replays the snapshot either way, and restore
latency does not change.

**Switch off.** `off − base` includes 0 on every headline stratum with n ≥ 5 per arm. For example, on 0xd40:
- cold first request −0.014 s [−0.076, +0.130]
- restore first request −0.014 s [−0.056, +0.018]
- warm +0.002 s [−0.094, +0.072]

`model_load_mmap` alone shows +0.066 s [+0.007, +0.111] on 0xd40. That is under 1 % of the phase and inside the
spread between rounds (cell deltas from −0.31 to +0.20 s).

**Cost** (list prices, `results.json` → `cost`):

| Lambda duration | SnapStart cache (3-hour minimum on 15 versions) | SnapStart restore | S3 | Total |
|---|---|---|---|---|
| $0.88 | $2.44 | $0.21 | under $0.01 | **$3.53** |

The operator's ceiling was $10. Billed duration falls in `par` cold rows by the load saving. The S3 bytes do not
change.

## Status rule

Written before the run: the effect pair (`par − off`) decides on the cold, restore and warm headlines. The switch-off
pair (`off − base`) must not exclude 0 on any headline stratum.

One switch-off stratum did exclude 0: restore first request on 0xd0c, n 2 (base) and 4 (off), −0.121 s
[−0.194, −0.020], with the patched libraries faster. So the original rule gives `inconclusive`.

After seeing that, the operator chose to publish with a minimum of 5 samples per arm for a switch-off stratum to
count. The excluded strata and both verdicts are listed in `e8.status_rule`. Every effect headline that excludes 0
favours `par`, on every CPU part, including strata with n ≥ 40.

## Host variability

`e8_entry.py` records the host at each environment's first invocation, after P1's handler has answered. That
placement keeps it out of every P1 phase and reads the restore host, not the snapshot host. It records:
- `/proc/cpuinfo` implementer, part, variant, revision, features and processor count
- AT_HWCAP / AT_HWCAP2
- `sched_getaffinity`, cgroup v2 limits, kernel, Python and the execution environment
- the `libggml-cpu-*` variant mapped into the process
- VmHWM, the log stream, and the init host's CPU part

| CPU part | variant | loaded variant | cold base/off/par | restore base/off/par |
|---|---|---|---|---|
| 0xd40 | 0x1 | armv8.6_1 | 41/44/40 | 48/46/46 |
| 0xd0c | 0x3 | armv8.2_2 | 4/3/6 | 2/4/4 |
| 0xd4f | 0x0 | armv8.6_2 | 5/3/4 | 0/0/0 |

Every environment reported 6 processors, `sched_affinity` 6 and kernel 5.10.255 (amzn2).

`unknown` (not exposed in the sandbox): cgroup `cpu.max`, `memory.max`, `cpuset.cpus.effective`, the physical
host, the AZ, CPU frequency and co-tenancy.

Every comparison is within one CPU part. No Graviton generation is inferred from a part ID. Each part runs a
different kernel variant, so its absolute times differ: 0xd0c's repack is the slowest, and it gains the most. Those
differences are not attributed to the code change.

The 0xd0c and 0xd4f strata are small, and 0xd4f had no restore environment. Thread scaling is bounded by the 6
vCPUs at 10,240 MB. A smaller memory size gives fewer vCPUs and a smaller gain; that was not measured.

## Limitations

- One region (us-east-2), one memory size (10,240 MB), one model (Q8_0), 5 rounds.
- The arms run as sequential waves within a round, with the order alternated between rounds.
- Only Q8_0 takes the parallel path. Every KleidiAI weight in this model is Q8_0 (187 tensors). The other 134
  tensors (133 f32 and one Q8_0 `token_embd`, used by GET_ROWS) stay in the plain CPU buffer and are never repacked.
- The threads are created per tensor (std::thread). A pool would cut thread creation further but was not tried.
- The offline load times come from a co-tenanted DevSpace and are not Lambda latency.
- The status rule was amended after the results were seen (see above).
- E4 (overlapped download) and E9 (pre-pack at build time) touch the same load path. This run measures against P1
  only. Once a download-overlap rung lands, the saving should show up as cold first-request latency.

## Reproduce

See `meta.json` → `reproduce`. `build.sh`, `hash_check.sh` and `probe_e8.sh` need an arm64 Linux host with Docker
and P1's `baseline/build/` (llama.cpp at `f74e2fa6a`, the GGUF). `deploy_e8.py` and `bench_e8.py` need
`AWS_PROFILE`. Tests: `pytest tests/test_lkai_e8.py`.
