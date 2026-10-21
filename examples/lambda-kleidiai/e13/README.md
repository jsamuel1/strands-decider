# E13: OpenMP wait policy for the warm forward

P3's profile (`profiling/README.md` on `lkai/bc9a-profiler`) put 5.7% of warm CPU samples on Lambda in libgomp's
spin/barrier code, called from `ggml_graph_compute_thread`. In the pinned llama.cpp (f74e2fa), `ggml_barrier` is
`#pragma omp barrier`. libgomp 11, with `OMP_WAIT_POLICY` unset, spins 300,000 times before it sleeps on a futex
(`libgomp/env.c`, `config/linux/wait.h`). The engine runs 6 OpenMP threads on Lambda's 6 vCPUs.

**Result (run `20261021T103304Z`).**
- A short spin or no spin makes warm requests about 5% faster.
- A spin that never sleeps makes them 55% slower.
- Answers are bit-identical on every arm.
- Thread count was not varied.

| Arm | Environment | libgomp spin count (logged) | Warm REPORT Duration, paired, CPU part 0xd40 | Status |
|---|---|---|---|---|
| ctl | none (P1) | 300,000 | baseline (median 605 ms) | — |
| passive | `OMP_WAIT_POLICY=passive` | 0 | **−5.15%**, 9/9 rounds below 0, p = 0.0039 (587 ms) | improved |
| spin30k | `GOMP_SPINCOUNT=30000` | 30,000 | **−5.40%**, 9/9 rounds below 0, p = 0.0039 (577 ms) | improved |
| active | `OMP_WAIT_POLICY=active` | 30,000,000,000 | **+55.3%**, 9/9 rounds above 0, p = 0.0039 (935 ms) | regressed |

passive and spin30k cannot be told apart: spin30k − passive = −0.24 percentage points per round, 6 of 9 rounds
below 0, p = 0.32. The gain comes from not spinning for the default 300,000 iterations, not from any particular
short count.

**Recommendation:** set `OMP_WAIT_POLICY=passive` in the function environment. It costs nothing, needs no code
change, and does not change answers. Since the wait policy has a clear effect, the thread-count follow-up
(5 against 6) is optional. If it is filed, it should be measured with `passive` set.

## The one variable

Every arm deploys the same zip, `e5f10006…`. That zip is P1's published `function.zip` `26eb2651…` (the zip behind
`baseline/results/20261020T080835Z`), byte for byte, plus `e13_entry.py`. Every arm uses P1's configuration:
- python3.12 and P1's pinned runtime ARN
- arm64, 10,240 MB, SnapStart on published versions
- P1's weights object, read-only
- P1's environment

The arms differ only in `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT`. `tests/test_lkai_e13.py` pins that, and also pins
that no arm sets `THREADS` or `OMP_NUM_THREADS`. Every arm sets `OMP_DISPLAY_ENV=VERBOSE`. libgomp then prints the
spin count it is actually using, and the saved logs show the expected value on all 72 starts of each arm
(`e13.spincount`).

`e13_entry.py` is identical on every arm and only observes. Before P1's clock starts, it records the host metadata
(`/proc/cpuinfo`, `AT_HWCAP`/`AT_HWCAP2`, affinity, cgroup files, kernel). Around every invocation, it records
`getrusage` for the whole process: user and system CPU seconds and voluntary and involuntary context switches. Those
counters are the mechanism evidence.

## Measurements

Two legs, 5,760 invocations in total. Every wave runs on a new invoke client (I1's fix).
- **Cold leg:** 6 rounds. Each round runs 10 cold invokes per arm, then 15 warm passes (P1's 5 requests × 3).
- **Restore leg:** 3 rounds. Each round publishes one SnapStart version per arm, runs 10 restore invokes per arm,
  then 15 warm passes on the restored environments.

**Warm (primary metric).** The arms are interleaved within each round. Every warm pass sends one request to each of
the four arms in turn, and the arm order rotates every pass. The metric is REPORT Duration, paired the way P3's
`paired_warm_by_request` does it:
- For each (round, request) cell, take median(arm) / median(ctl) − 1, within one CPU part.
- The round is the unit (the mean of its cells), because a round's cells share its environments.
- The test is an exact two-sided sign flip over the rounds. With 9 rounds, the smallest attainable p is 0.0039.

A split-half null within ctl measures the noise floor: no ctl round mean exceeds 3.05%. The engine's own timer
(`latency_ms`) agrees with REPORT on every arm.

| CPU part 0xd40, per warm request (median) | ctl | passive | spin30k | active |
|---|---|---|---|---|
| REPORT Duration | 605 ms | 587 ms | 577 ms | 935 ms |
| process CPU (user + system) | 2.78 s | 2.74 s | 2.75 s | 4.58 s |
| voluntary context switches | 370 | 17,453 | 3,821 | 7 |

The voluntary-switch counts confirm the change. Idle workers that spun in the barrier now sleep: 370 → 17,453
switches per request with no spin, and 3,821 with a 30,000 spin. Paired process CPU per request changes −1.93%
(passive) and −1.05% (spin30k) while wall time falls about 5%: the request finishes sooner with almost the same CPU
work. The extra CPU that `active` burns (+67.6% paired) comes with more involuntary switches (30 → 381):
threads get preempted more often. The process has 12 threads on 6 vCPUs after its first invocation. That spinning
workers delay the other runnable threads is an inference from these counters, not a direct measurement.

**Side effects (arm − ctl, CPU part 0xd40, bootstrap 95% CI over environments).**

| | passive | spin30k | active |
|---|---|---|---|
| cold first request (Init + Duration) | −0.057 s [−0.145, +0.015] | −0.081 s [−0.190, +0.001] | **+0.265 s [+0.194, +0.362]** |
| restored first request | −0.029 s [−0.071, +0.039] | +0.003 s [−0.070, +0.107] | **+0.521 s [+0.440, +0.603]** |
| restored first inference | −27 ms [−38, −6] | −17 ms [−31, −2] | +502 ms |
| cold max memory | −2 MB | 0 MB | −1 MB |

Neither short-spin arm makes cold or restore slower. `active` makes both slower, through the first inference.

**Errors.**
- 0 function errors.
- 1 client-side `SSLError` (EOF after 0.42 s) on a passive warm invoke in restore-leg round 2. The arm's saved logs
  hold no REPORT line that a successful row does not account for, so Lambda never ran that invoke. It is published
  as an error and carries no answer or latency. `summarize_e13.error_kinds` classifies it as `client-transport`.
- 0 restore rows with `client:ReadTimeoutError`.

**Cost:** $3.98 at list prices, under the operator's $8 ceiling. The SnapStart cache 3-hour minimum on 12 versions
is $1.95 of that. Teardown removed every function, version, role, log group and the zip, and was verified.

## Correctness

- Every arm's answers are bit-identical to ctl's for the same request and CPU part: 1,439/1,439 on passive (the
  missing one is the transport error), 1,440/1,440 on spin30k, 1,440/1,440 on active.
- All 5,579 answers that P1's `reference/outputs.json` covers are bit-identical to it. The reference has no 0xd4f
  entry for four of the five requests, which leaves 180 warm answers on 0xd4f uncovered; the run's own ctl
  comparison covers those.
- The wait policy changes only how idle threads wait, not how the work is split into rows. ggml's per-row maths is
  the same, so identical answers are what that predicts.

## Host variability

Three CPU identities appeared, read from `/proc/cpuinfo` and the auxiliary vector. No CPU generation is inferred
from them.

| CPU part | Variant | HWCAP | HWCAP2 | ggml variant loaded | cold envs (ctl/passive/spin30k/active) | restore envs |
|---|---|---|---|---|---|---|
| 0xd40 | 0x1 | 0xfffffff | 0x1f201 | armv8.6_1 | 52 / 53 / 50 / 49 | 29 / 29 / 29 / 27 |
| 0xd0c | 0x3 | 0x119fff | 0x0 | armv8.2_2 | 2 / 5 / 5 / 9 | 1 / 1 / 1 / 3 |
| 0xd4f | 0x0 | 0x2ff3ffff | 0x3f3bf | armv8.6_2 | 6 / 2 / 5 / 2 | 0 |

Every environment reported:
- 6 CPUs and `sched_getaffinity` 6
- kernel 5.10.255, Python 3.12.14
- 12 threads after the first invocation
- cgroup v1 controllers, all at `/`

The cgroup v2 and v1 limit files (`cpu.max`, `cpu.cfs_quota_us`, `memory.limit_in_bytes`, …) are not exposed:
`unknown`. The availability zone, physical host and CPU frequency are also `unknown`.

Every comparison is made within one CPU part, and rows without a CPU part are never compared. The verdicts rest on
0xd40, which has 9 paired rounds. 0xd0c and 0xd4f have too few rounds for any p < 0.05 (`underpowered` or
`insufficient`), but every one of their rounds has the same sign as on 0xd40:
- passive: 0xd0c −6.3% (1 round), 0xd4f −5.5% (2 rounds)
- spin30k: 0xd0c −7.7% (3), 0xd4f −4.8% (4)
- active: 0xd0c +14.6% (3), 0xd4f +55.7% (1)

`results.json` → `host_variability` lists every shared (round, CPU part) cell. Host variation therefore cannot
explain the result:
- The arms are interleaved within each round.
- Each cell compares one CPU part.
- The effect has the same sign in every round and on every part.

The effect is also larger than the control's own split-half noise. No rerun is needed, and H1's fixed-host box was
not needed: this is a Lambda deployment setting, and it was measured on Lambda.

## Limits

- One region (us-east-2), one memory size (10,240 MB, 6 vCPUs) and one thread count (6). On a different vCPU count
  or thread count, the size of the effect may differ.
- Cold and restore waves ran sequentially within a round, with the arm order rotating across rounds (a Latin square
  over 4 rounds). That controls order and round drift, but not drift within a round. Warm is interleaved within the
  round.
- The 0xd0c and 0xd4f strata are small. Their direction agrees with 0xd40, but they are not tested on their own.
- libgomp is the gcc 11 build that P1's zip ships. The default spin count is that version's.
- Thread count (5 against 6) was not run here, as the card specified.

## Files

- `e13_entry.py`: Lambda entry point (metadata and rusage), wraps `baseline/lkai_entry.py` unchanged.
- `e13_zip.py`: P1's zip plus the wrapper.
- `e13deploy.py`: the four arms, `up` / `show` / `teardown --all`.
- `e13bench.py`: cold/restore legs, with the warm passes interleaved.
- `summarize_e13.py`: one `lkai-result/1` file per experiment arm, and `--stage` for P2 publication.
- `results/20261021T103304Z/`:
  - `<arm>/results.json` for each experiment arm
  - shared `invocations.jsonl.gz`, `logs/`, `conditions.json`, `build-manifest.json` and `meta.json`
  - redacted, with no account or bucket

## Reproduce

See `meta.json` → `reproduce`.
