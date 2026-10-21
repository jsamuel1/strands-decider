# lambda-kleidiai P3: profiler applicability and overhead on the P1 baseline

Which profilers work inside the P1 Lambda (python3.12, arm64, 10,240 MB, SnapStart), what each one costs, and where
the baseline's time goes, by phase. The results feed E6 (profile-led hot spots).

The operator asked for "detailed profiling using 'nvidia-spi' as a path and other tools". No NVIDIA tool is called
`nvidia-spi`. This study reads it as NVIDIA's GPU tooling: `nvidia-smi` (the driver's System Management Interface, the
closest name) and Nsight Systems / Nsight Compute. **Lambda has no GPU, so all three are inapplicable there.** This
was checked from inside the function, not assumed (below). They were run, with operator approval, on a separate GPU
host, which is labelled separately.

Everything is in `results/` and is published to the P2 dashboard as experiment `P3-profilers`
(`results/<run-id>/results.json`, `lkai-result/1`).

## Evidence labels

| Label | Meaning |
|---|---|
| **Lambda** | Measured inside the deployed function: same zip bytes as P1 (plus added files on profiler arms), same runtime version ARN, arm64, 10,240 MB, SnapStart, us-east-2 |
| **OFFLINE, NOT LAMBDA (arm64 DevSpace, part 0xd40)** | P1's `function.zip` and GGUF in P1's pinned Lambda python3.12 image on an arm64 DevSpace (CPU part 0xd40, kernel 6.1), 6 threads pinned to 6 CPUs. The GGUF is read from local disk, so `weight_read` is not comparable to Lambda's S3 download |
| **OFFLINE, NOT LAMBDA (x86 / GPU)** | The same llama.cpp commit and patches rebuilt for x86_64, with the CUDA backend for the NVIDIA L4 of a g6.4xlarge. These are not the deployed binaries: there is no KleidiAI on x86, and the Lambda zip is CPU-only |

## What the function runs on (Lambda)

`lkai_probe.py`, inside `lkai-p3-probe` (P1's runtime version, arm64, 10,240 MB), across 7 execution environments
(`results/lambda-probe/*.json`):

- `uname`: `5.10.255-268-309.1092.amzn2.aarch64`. `/proc/cpuinfo`: 6 processors, implementer `0x41`, variant `0x1`,
  part `0xd40`, revision `1` in all 7, with `sve i8mm bf16 asimddp` among the features. The probe never landed on
  another part ID, but the profiler run did (see Host variability): the probe is not a census of Lambda hosts.
- **No GPU:** `/dev/nvidia*` absent, no PCI vendor `0x10de` (no PCI devices are listed at all), `/proc/driver/nvidia`
  absent, `libcuda.so.1` and `libnvidia-ml.so.1` not loadable, `nvidia-smi`/`nsys`/`ncu` not on `PATH`.
- Sandbox: uid 993, every capability set empty (`CapEff=CapBnd=0`), `NoNewPrivs=1`, seccomp mode 2 with 6 filters,
  `perf_event_paranoid=2`, `kptr_restrict=1`, `unprivileged_bpf_disabled=1`, no PMU devices, no tracefs/debugfs/bpffs.
- Syscalls tried: `perf_event_open` (hardware cycles, software cpu-clock, task-clock, user space only, own
  thread) → EPERM for all three. `bpf(BPF_MAP_CREATE)` → EPERM. `ptrace(PTRACE_SEIZE)` and `ptrace(PTRACE_ATTACH)` of our
  own child → EPERM. `process_vm_readv` of self → allowed; of the parent from a child → allowed.
  `setitimer(ITIMER_PROF)` → SIGPROF delivered.

## Applicability matrix

`applicability.json` holds the same rows with the full evidence for each.

| Tool | In Lambda | Evidence | Offline |
|---|---|---|---|
| nvidia-smi | **inapplicable** | no GPU device, driver or library (above) | works on the L4 (100 ms sampling) |
| Nsight Systems (nsys) | **inapplicable** | no GPU; it would also need perf/ptrace, both denied | works on the L4 (kernel, API and memcpy summaries) |
| Nsight Compute (ncu) | **inapplicable** | no GPU | works on the L4; kernel replay makes it heavy (warm-up 16.2 s vs 0.09 s) |
| P1 phase timers | **works** | `LKAI` records; arm `none` | — |
| Lambda REPORT / CloudWatch | **works** | every invocation; arm `rep` is the zero-instrumentation control | — |
| X-Ray active tracing | **works** | arm `xray`: `TracingConfig: Active`, no code change; Lambda's own segments | — |
| Telemetry API extension | **works** | arm `telemetry`: `extensions/lkai-telemetry` subscribed (`platform.telemetrySubscription: Subscribed`) | — |
| cProfile | **works** | arm `cprofile`: on Python 3.12 one profile covers every thread; a second concurrent one is refused | — |
| py-spy, Python frames (`--nonblocking`) | **works** | arm `pyspy`: run as the function's own child, reading the parent with `process_vm_readv` | — |
| py-spy `--native` / blocking | **denied** | `Failed to suspend process ... EPERM` (needs ptrace) | works on the arm64 DevSpace, about +29% per forward |
| perf | **denied** | `perf_event_open` EPERM ×3, no PMU, not in the image | works on the arm64 DevSpace |
| eBPF (bpftrace, BCC) | **denied** | `bpf()` EPERM, `unprivileged_bpf_disabled=1`, no bpffs/tracefs | — |
| ptrace tools (gdb, strace, ltrace) | **denied** | `PTRACE_SEIZE`/`PTRACE_ATTACH` EPERM; none on `PATH` | — |
| lkaisamp (this directory's native sampler) | **works** | arm `sampler`: SIGPROF delivered; PCs symbolised in the function against the shipped libraries' `.symtab` | agrees with perf on the arm64 DevSpace |
| async-profiler | **inapplicable** | JVM-only; the runtime is Python | — |

### lkaisamp: native profiling where perf is denied

`lkaisamp.c` sets `setitimer(ITIMER_PROF)` (1 ms of process CPU time). The `SIGPROF` handler records the interrupted
thread's PC and LR (aarch64's return address, so the leaf's caller), its tid and a phase tag. It does nothing else: no
allocation, no locks, no unwinding. `lkai_symbolise.py` resolves the samples in the function from `/proc/self/maps`
and the ELF symbol tables of the shipped libraries, using the standard library only. `libggml-cpu-armv8.6_1.so`, the
variant part 0xd40 loads, keeps 2,459 function symbols, 733 of them KleidiAI `kai_*`. Its limits:

- It records a PC and its caller only, not a full stack.
- The kernel keeps one SIGPROF pending per process, so with 6 busy threads the effective rate is well under 1 kHz per
  CPU-second (`effective_hz_per_cpu_s` in the results).
- It measures CPU time only, so S3 reads and futex waits do not appear.

On the arm64 DevSpace (part 0xd40) its per-phase ranking matches `perf record` (results below).

## Overhead (Lambda)

Each profiler arm is its own function (`lkai-p3-<arm>`, deployed by `deploy_arms.py`). Each has P1's configuration,
runtime version, GGUF object and zip bytes; the `lkai_prof.py` arms also carry the added files. The arms run through
P1's harness (`baseline/bench.py`), interleaved, 5 rounds × 10 cold + 10 restore + 10 warm passes of 10. `rep` is
P1's REPORT-only arm, rebuilt from P1's zip as P3's own function so that no other card shares it. `rep2` is a second
function with the same zip, handler and configuration as `rep`: it changes nothing, so `rep2 − rep` is what a
zero-cost arm looks like. That is the null every arm is judged against, X-Ray and the phase timers included.

Run `20261020T231634Z` (us-east-2, 8 arms, 5 rounds, 4,800 invocations, $9.61 per its `cost` block, most of it the
SnapStart cache minimum on 40 published versions). `bench.py` opened a new Lambda client for every invocation wave
(480 clients), so no pooled connection outlived a publish or SnapStart wait. All 4,800 invocations succeeded, with 0
client errors and 0 start-class mismatches, and every arm/round restore cell has 6 to 10 valid samples on CPU part
0xd40 (`restore_evidence`). All eight arms returned answers bit-identical to P1's reference (4,800 compared, 0
choice flips).

Each cell is the mean over the 5 rounds of `median(arm) − median(rep)` within the round, CPU part 0xd40 rows only,
then the [min, max] over rounds. A round is compared only when both medians rest on at least 3 valid samples (none
was skipped in this run). Warm is paired by request as well (25 cells of round × request, since the five requests
take 0.25 to 1.75 s), shown as the median percentage.

**Inference.** `rep2` runs in the same rounds as every arm, so `arm − rep` and `rep2 − rep` share each round's `rep`
median: they are paired, not two independent groups. The test statistic is the per-round paired difference
`(arm − rep) − (rep2 − rep) = arm − rep2` (warm: each round's mean percentage, since a round's cells share its
environments), and `p` is an exact two-sided sign-flip test over its 2⁵ = 32 sign assignments. **With 5 pairs the
smallest attainable p is 2/32 = 0.0625, so no cell can reach p < 0.05: this study is underpowered to call any
profiler cost statistically measurable.** An earlier version of this table tested the two sets of deltas as
independent 5-vs-5 groups (252 splits) and reported p = 0.008 for four cells; that discarded the pairing and
overstated the evidence, and is withdrawn. The estimates themselves are unchanged.

Bold with † marks a **consistent direction** (descriptive, not a test): all 5 paired differences have the same sign,
and every round's `arm − rep` is beyond `rep2`'s largest |per-round value|.

| Arm | Cold Init Duration | Cold Duration (first request) | Restore Duration | First request after restore | Warm Duration, per request |
|---|---|---|---|---|---|
| `rep2` (null: same code as `rep`) | −0 ms [−28, +26] | +53 ms [−63, +155] | +48 ms [+5, +112] | +12 ms [−8, +35] | +0.3% [−4.6, +4.8] |
| `none` (phase timers) | +9 ms [−5, +38], p=0.5 | +69 ms [−67, +199], p=0.75 | +5 ms [−53, +42], p=0.125 | +0 ms [−14, +18], p=0.25 | −0.3% [−4.1, +2.5], p=0.125 |
| `xray` | +5 ms [−21, +24], p=0.25 | +77 ms [−27, +280], p=0.5625 | +24 ms [−21, +52], p=0.0625 | +10 ms [+5, +18], p=0.6875 | −0.6% [−6.9, +2.1], p=0.1875 |
| `cprofile` | **+113 ms [+101, +120], p=0.0625 †** | +186 ms [+21, +394], p=0.0625 | +70 ms [+47, +105], p=0.125 | +12 ms [−1, +40], p=1 | −0.2% [−1.5, +4.7], p=1 |
| `telemetry` | **+93 ms [+57, +134], p=0.0625 †** | −102 ms [−308, +46], p=0.0625 | +50 ms [−3, +127], p=1 | +6 ms [−10, +17], p=0.25 | −0.3% [−5.1, +4.1], p=0.6875 |
| `pyspy` | +20 ms [−5, +38], p=0.0625 | +134 ms [+16, +302], p=0.1875 | +18 ms [−32, +123], p=0.4375 | +25 ms [+6, +49], p=0.0625 | **+4.7% [+0.9, +12.8], p=0.0625 †** |
| `sampler` (lkaisamp) | +16 ms [−1, +40], p=0.0625 | **+457 ms [+382, +646], p=0.0625 †** | +107 ms [+41, +180], p=0.375 | +1,011 ms [−9, +1,772], p=0.25 | −0.3% [−7.4, +3.9], p=0.1875 |

Several cells sit at p = 0.0625 without a consistent direction: their 5 paired differences share a sign, but at
least one round's `arm − rep` falls inside the null's spread.

Control medians (`rep`, part 0xd40): Init Duration 812 ms, cold Duration 35.2 s, Restore Duration 3.21 s, first
request after restore 678 ms, warm 650 ms (pooled over the five requests). Full per-round values, the paired
differences, the verdicts and their p-values are in `results.json` → `profiler_overhead_paired`.

What this supports, and what it does not:

- **No profiler cost is statistically established at α = 0.05** (5 paired rounds cannot get below p = 0.0625). A
  6th round could reach p = 0.031; 7 rounds 0.016.
- **Phase timers (`none`) and X-Ray: no consistent direction** on any phase. This is not proof of zero cost: with 5
  rounds, a cost smaller than the null's spread (up to 28 ms Init, 155 ms cold Duration, 112 ms Restore Duration,
  35 ms first restored request, about 1.8% per-round warm) cannot be resolved.
- **cProfile: +113 ms of Init Duration** (14%), consistent direction, every round above the null (p = 0.0625). Its
  +186 ms on the cold first request does not have a consistent direction. No warm cost is visible: its
  per-invocation profile spans only the Python layer, and the forward runs in C.
- **Telemetry API extension: +93 ms of Init Duration** (11%), consistent direction (p = 0.0625). Telemetry's own
  `extension_register_ms` (p50 30 ms, p90 89 ms) is a plausible part of it; nothing else has a consistent direction.
- **py-spy (non-native): +4.7% median per warm request**, consistent direction (per-round paired differences +2.6%
  to +5.7%; every round above the null's largest 1.8%; p = 0.0625).
- **lkaisamp: +457 ms on the cold first request** (Duration), consistent direction (p = 0.0625). On the first request
  after a restore it added about +1.5 to +1.8 s in rounds 2 to 4 but nothing in rounds 0 and 1 (per-round deltas
  +24, −9, +1,512, +1,756, +1,772 ms; p = 0.25). Run `20261020T124451Z` saw +1.8 to +2.3 s in all rounds; why two
  rounds of this run did not is not established. No warm cost is visible. These deltas appear in Lambda's
  `Duration`, while the handler's internal `latency_ms` and phase records are unchanged. The sampler dumps and
  symbolises before `lkai_prof.handler` returns, so the cost is outside the baseline's internal timing but still
  before Lambda receives the response. Attribution within that hook remains open: parsing the ELF tables takes 26 ms
  offline, which does not explain it.

The offline overheads below (arm64 DevSpace) are not Lambda numbers, but they rank the same way for the warm forward.

**Final latency comparisons use light instrumentation only:** the REPORT-only arm or P1's phase timers. Profiler arms
are for finding hot spots, not for timing them.

### Superseded: run 20261020T124451Z

The previous run (7 arms, no `rep2`) is superseded and its overhead numbers are withdrawn. Two defects, both found in
review, invalidated it:

- `bench.py` kept one Lambda client for the whole run. In every round, the first arm's restore wave (after the
  longest publish/SnapStart wait) lost 9 of 10 invokes to client read timeouts with no `RESTORE_START`, leaving the
  `rep` restore control with one valid sample in rounds 0, 2 and 4. Its restore overhead numbers compared against
  those single samples. That run's README reported this as a Lambda platform finding ("5 of 5 rounds"); it was not.
  With a fresh client per wave, run `20261020T231634Z` had 0 client errors in 400 restore invokes, including every
  first restore wave after a 19 to 22 minute wait. E1 saw the same failure and fix (`e1-s3-express/bench_e1.py`).
- Its noise floor was the X-Ray arm's own deltas, which were then used to conclude X-Ray had no measurable cost.
  `rep2` replaces that.

Its raw evidence stays in `results/20261020T124451Z/` for audit. Its `results.json` was regenerated with the corrected
summariser, so the restore verdicts there are `withheld` (sparse cells) and every share uses full denominators.

## Host variability (Lambda, run 20261020T231634Z)

The initiative's host-variability rules apply: CPU identity is recorded as observed, anything not observed is marked
unknown, and no processor generation is inferred from a part ID or from timings. This section is a reanalysis of
the committed evidence only (no AWS calls, no new measurement). The machine-readable form is `results.json` →
`host_variability` (`lkai-host-variability/1`).

**Observed.** Every invocation row records the MIDR CPU part ID that `handler._cpu_part()` read from
`/proc/cpuinfo` during that request, so a restored environment reports the host it was restored on. Three part IDs
served this run: `0xd40` 4,225 rows, `0xd0c` 528, `0xd4f` 47. They appear in every arm:

| Arm | cold (0xd40 / 0xd0c / 0xd4f) | restore (0xd40 / 0xd0c) | warm (0xd40 / 0xd0c) |
|---|---|---|---|
| `rep` | 35 / 8 / 7 | 42 / 8 | 421 / 79 |
| `rep2` | 43 / 5 / 2 | 45 / 5 | 451 / 49 |
| `none` | 37 / 9 / 4 | 46 / 4 | 459 / 41 |
| `xray` | 33 / 9 / 8 | 45 / 5 | 451 / 49 |
| `cprofile` | 36 / 7 / 7 | 48 / 2 | 481 / 19 |
| `telemetry` | 31 / 12 / 7 | 40 / 10 | 398 / 102 |
| `pyspy` | 36 / 6 / 8 | 46 / 4 | 460 / 40 |
| `sampler` | 36 / 10 / 4 | 46 / 4 | 459 / 41 |

No restore or warm row ran on `0xd4f`. Each part ID loaded its own ggml CPU variant (from the engine's init log,
250 cold rows): `0xd40` → `libggml-cpu-armv8.6_1.so`, `0xd0c` → `libggml-cpu-armv8.2_2.so`, `0xd4f` →
`libggml-cpu-armv8.6_2.so`, so the KleidiAI kernels differ by part ID (warm matmul on 0xd40
`kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm`, on 0xd0c
`kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod`). 4,472 of 4,800 rows link to their CloudWatch log
stream through the START RequestId in the log tail (the other 328 tails lost that line); across the 650 linked
streams no stream changed part ID. Every INIT_START names `python:3.12.mainlinev2.v43`. The in-function probe
(`lambda-probe/*.json`, a separate function, 7 environments) read implementer `0x41`, variant `0x1`, part `0xd40`,
revision `1`, 6 processors, kernel `5.10.255-268-309.1092.amzn2.aarch64`, Python 3.12.14 and the full feature string;
it never landed on `0xd0c` or `0xd4f`.

**Unknown** (not recorded in this run): implementer, variant, revision and feature flags per invocation; HWCAPs;
the core count per invocation and after a restore (`threads=6` is configuration, not an observed nproc); cgroup CPU
and memory limits; the kernel per invocation; availability zone; physical host; the part ID of the environment that
built each SnapStart snapshot; and any processor name or generation for any part ID.

**How comparisons handle it.** Every overhead cell above uses part `0xd40` rows only, paired by round against `rep`
and `rep2` on the same part, so the part mix cannot move a verdict. Other part IDs are counted but not compared:
per (arm, class, round) they are too sparse for a paired cell. Figures do differ by part ID, so any pooled figure
depends on its run's mix. `rep` warm REPORT Duration p50 is 650 ms on `0xd40` vs 1,188 ms on `0xd0c`; `rep` warm
engine p50 for `ticket-route-1q` 245 ms vs 440 ms, for `incident-review-4q` 1,763 ms vs 3,449 ms; `rep` cold Duration
p50 35.2 s (`0xd40`, n=35), 47.4 s (`0xd0c`, n=8), 33.9 s (`0xd4f`, n=7). The native hot-spot tables are split by part
ID (`host_variability.observed.lkaisamp_top_self_by_part`): init `buffer_set_tensor` is 26.4% of init CPU samples on
`0xd40` (36 inits), 37.6% on `0xd0c` (10), 30.3% on `0xd4f` (4); warm `concat` is 5.6% on `0xd40` and 6.5% on `0xd0c`.

**The concat gap.** Lambda warm `concat` (lkaisamp, `0xd40`, 459 invocations) is 5.6% of CPU samples. Offline, on
the arm64 DevSpace that also reads part `0xd40`, it is 2.65% under the same sampler (`steady_state_inference`, 6,008
samples) and 1.2% under `perf record` (whole run). The same part ID on both sides rules out a part-ID mix as the
cause of this gap, but not other host differences that were not recorded (implementer/variant/revision per
invocation, cgroup limits, core count, kernel 5.10 vs 6.1, sampler rate). It is not attributed to code.

H1 (host-variability card) reuses these tables rather than re-profiling.

## Baseline hot spots by phase (feeds E6)

Phases are ranked by wall time from P1's own timers, read from the `none` arm (`rep` runs with `LKAI_PHASES=0` and
has no phase records; `none` runs P1's timers, which show no consistent-direction cost against the null above). Each
phase's contents come from the profiler best placed to see them. **Lambda** rows are from run `20261020T231634Z`,
CPU part 0xd40 where split, pooled over part IDs where not (see Host variability). **Offline** rows are the arm64
DevSpace (part 0xd40) and are not Lambda.

Every percentage below is a share of **all** samples (or all own-time) the profiler recorded for that phase, not of
the rows kept in its top-N table: `summarize_prof._merge_top` divides by each record's full total. The previous
version of this section divided by the retained rows only and overstated every share (for example
`buffer_set_tensor` 35.2% where the full-denominator figure for that run was 30.1%).

### Cold start: 36.0 s first request (Init 0.82 s + Duration 35.2 s), p50, Lambda

An on-demand cold start loads the model inside the first invocation (PR #77's background load), so `Duration` holds it.

| Rank | Phase (P1 timer) | p50 | Share of first request | What the time is (evidence) |
|---|---|---|---|---|
| 1 | `weight_download` | 24.99 s | 69.4% | **Waiting on S3 over TLS.** py-spy (Lambda): 58.3% of init leaf samples in `ssl.py:1103 read`. cProfile (Lambda): `_ssl._SSLSocket.read` has 58.5% of init own-time (1,487 s of 2,540 s summed over all threads, the 32 download threads included). lkaisamp sees little here because waiting uses no CPU; `libc read` + `pwrite` (copies into the memfd) are 7.7% of init CPU samples (5.1% + 2.6%). 79 MiB/s. Network-bound: the E1 to E4 experiments target it |
| 2 | `model_load_mmap` | 8.38 s | 23.3% | **The KleidiAI weight repack, not I/O.** lkaisamp (Lambda, init, 124,272 samples, pooled over part IDs): `ggml_backend_cpu_kleidiai_buffer_set_tensor` 30.8% (16.4% + 14.4%, the copies in the two CPU variants) + `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` 12.2% (6.2% + 6.0%) + `memset` 7.5% + `lroundf` 2.6% of init CPU samples. Offline arm64 DevSpace lkaisamp, `model_load` phase only: `buffer_set_tensor` 46.8%, `rhs_pack` 22.8%, the variant's unsymbolised `+0x32000` 9.0%, `memset` 6.1%, all on **one thread** (1 sampled thread over the 7.9 s phase). py-spy (Lambda): `engine.py:114 __init__` (the native `dec_load`) holds 31.3% of init leaf samples. E5 (mmap-friendly format) targets it; a repacked-at-build-time format would remove it |
| 3 | `assets_load` | 0.95 s | 2.6% | Tokenizer and numpy head (`tokenizers.abi3.so` 2.1% of init CPU samples) |
| 4 | `imports` | 0.73 s | 2.0% | Python imports and bytecode compile (cProfile: `builtins.compile` 21.8 s own-time summed over threads, 0.9%) |
| 5 | `context_init` | 0.33 s | 0.9% | llama context creation |
| 6 | `warmup_inference` | 0.25 s | 0.7% | One forward, same kernels as steady state |
| 7 | `first_request_inference` | 0.24 s | 0.7% | The request itself |

### SnapStart restore: 3.90 s first request (Restore 3.21 s + Duration 0.68 s), p50, Lambda

| Rank | Phase | p50 | Share | What the time is |
|---|---|---|---|---|
| 1 | `report_restore_duration` | 3.21 s | 82.2% | Lambda's snapshot restore, before any function code runs (X-Ray `Restore` segment p50 3.23 s, 50 traces). No profiler inside the function can see it: it happens before the runtime resumes. Telemetry reports the same span (`platform.restoreReport`) |
| 2 | `first_request_inference` | 0.67 s | 17.3% | **2.8× the warm forward (0.24 s) for the same request, and half of it is `memset`.** Over all 46 restored first invocations of the sampler arm on part 0xd40 (`sampler_invoke_restore_0xd40`, 2,670 samples): `memset` 50.7%, the i8mm matmul 34.9%, then the warm kernels at 1 to 2% each. Warm, `memset` is 2.2%. Its callers, from the sampler's caller edges: `llama_kv_cache::clear` (690 samples) and `llama_memory_recurrent::clear` (661), the per-request cache clears. Warm, the same clears are cheap, so the likely extra cost is first-touch page faults on the restored cache memory (a fault's kernel time is charged to the instruction that touched the page). That part is an inference: Lambda allows no perf to count faults directly |
| 3 | `restore_to_invoke` | 0.011 s | 0.3% | after-restore hooks |

### Warm steady state: 0.25 to 1.77 s per request, Lambda

| Rank | Native symbol (Lambda lkaisamp, warm, part 0xd40; 27,899 samples over 459 invocations) | Share of CPU samples | Offline arm64 DevSpace (part 0xd40) perf record (whole run) |
|---|---|---|---|
| 1 | `kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` (KleidiAI int8 matmul) | 48.8% | 47.7% |
| 2 | `ggml_vec_dot_f32` (f32 dot products: attention / recurrent state) | 8.8% | 9.4% |
| 3 | `ggml_compute_forward_gated_delta_net_one_chunk` (the linear-attention recurrence) | 7.8% | 8.3% |
| 4 | `ggml_compute_forward_concat` | 5.6% | 1.2% |
| 5 | libgomp spin/barrier (`+0x27000`) | 5.4% | 5.6% (all libgomp) |
| 6 | `ggml_compute_forward_fwht` (the runtime Hadamard rotation) | 5.1% | 4.0% |
| 7 | `ggml_compute_forward_ssm_conv` | 2.6% | 2.5% |
| 8 | `memset` | 2.2% | — |
| 9 | `kai_run_lhs_quant_pack_qai8dxp_f32` (activation quantisation) | 2.0% | 1.3% |
| 10 | `rms_norm_mul_fused`, `swiglu`, `silu` | 1.5%, 1.3%, 1.1% | 0.9%, 1.0%, 0.9% |

By library (Lambda, warm, part 0xd40): `libggml-cpu-armv8.6_1.so` 88.8%, libgomp 5.5%, libc 4.0%, numpy's OpenBLAS
0.6%. Python is negligible. py-spy (Lambda, warm): 93.1% of leaf samples sit in `engine.py:156 _fork`, the ctypes call
into the forward. cProfile (Lambda, warm): `_fork` 92.0% of own-time, `_eval` 6.0%, the numpy head (`cumsum`,
`logits`) under 2%. Part 0xd0c environments (10.5% of warm rows in this run) load the dot-product variant
(`kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod`), which is kept in its own table
(`sampler_invoke_warm_0xd0c`).

### Pointers for E6

- Warm: about half the CPU time is the int8 matmul kernel, already the fastest KleidiAI path for this CPU. The
  non-matmul half (f32 dot products, the gated-delta-net recurrence, `concat`, `fwht`, OpenMP barriers: about 33%
  together) has more room. `concat` is 5.6% in Lambda (part 0xd40) against 2.65% offline under the same sampler and
  1.2% under perf, also on part 0xd40; the gap is not attributed to code (see Host variability), but worth one look.
- Restore: half of the first forward after a restore is `memset` inside `llama_kv_cache::clear` and
  `llama_memory_recurrent::clear`. Clearing (touching) those caches once in an after-restore hook, before the first
  request, is a candidate one-change experiment. So is clearing only the cells that were used.
- Cold: after the S3 download (E1 to E4), the 8.4 s load is the KleidiAI repack (`buffer_set_tensor` +
  `rhs_pack`). Offline it runs on one thread for the whole phase, so threading it or doing it at build time (E5)
  are the candidates.

### Offline comparison (NOT LAMBDA)

| Host | Steady forward, `ticket-route-1q` / `incident-review-4q` | Notes |
|---|---|---|
| Lambda, part 0xd40, 6 vCPU (`rep` warm, REPORT Duration p50, run 20261020T231634Z) | 0.247 / 1.765 s | the deployed function |
| arm64 DevSpace, part 0xd40, same image, 6 CPUs | 0.243 / 1.739 s | within 2% of Lambda; perf available |
| AMD EPYC 7R13 (g6.4xlarge host CPU), 6 threads, `haswell` variant, no KleidiAI | 0.670 / 4.488 s | x86 is not a deployment target of this example |
| NVIDIA L4, all layers offloaded (CUDA, sm_89) | 0.023 / 0.106 s | top kernels `mul_mat_q` (Q8_0) at 61.5% of GPU kernel time over all its tile sizes; nvidia-smi peak memory and utilisation in `results/offline/gpu-host/` |

## Reproduce

```bash
# 0. P1's build/ (function.zip, GGUF, deployment.json) on an arm64 host with Docker: baseline/build.sh, deploy.py up
B=examples/lambda-kleidiai/baseline/build
P=examples/lambda-kleidiai/profiling
# 1. Profiler package: P1's zip plus lkai_prof.py, lkai_probe.py, lkai_symbolise.py, liblkaisamp.so, py-spy.
#    Refuses a dirty source; pins the py-spy wheel sha256; records every source hash and the commit in prof-manifest.json
$P/build_prof_zip.sh $B/function.zip build/p3
cp $B/deployment.json build/p3/p1-deployment.json
# 2. Deploy the arms (lkai-p3-*, own role) and run P1's harness over them
AWS_PROFILE=<p> python $P/deploy_arms.py up --p1-state build/p3/p1-deployment.json --build build/p3
AWS_PROFILE=<p> python examples/lambda-kleidiai/baseline/bench.py --build build/p3 --region us-east-2 --rounds 5 \
  --concurrency 10 --warm 10 --read-timeout 120 --arms rep,rep2,none,cprofile,pyspy,sampler,xray,telemetry --out build/p3/runs/<run-id>
AWS_PROFILE=<p> python $P/deploy_arms.py teardown --build build/p3          # published versions (SnapStart cache)
AWS_PROFILE=<p> python $P/fetch_xray.py build/p3/runs/<run-id>
# 3. In-Lambda probe: invoke any lkai_prof arm with {"mode": "lkai_probe"}, or deploy lkai_probe.py alone
# 4. Offline, NOT Lambda
$P/graviton_profile.sh $B/function.zip $B/model/s2b-h512-Q8_0.gguf build/p3/graviton                 # arm64 + Docker
GGUF_URL=... ZIP_URL=... GGUF_SHA256=... REPO_URL=... REPO_REF=<commit> bash $P/gpu_profile.sh /opt/lkai-p3   # GPU host
python $P/offline_summary.py --graviton build/p3/graviton --gpu <gpu out> --out $P/results/offline
# 5. Summary and publication (P2)
#    --experiment-commit is the manifest's source_commit; every packaged input is re-checked against it
python $P/summarize_prof.py build/p3/runs/<run-id> --reference examples/lambda-kleidiai/baseline/reference/outputs.json \
  --baseline-commit aac59d45555ebded9357c23f6f8e4f939edf25d7 --meta $P/meta.json --applicability $P/applicability.json \
  --offline $P/results/offline --probe $P/results/lambda-probe/q1.json \
  --experiment-commit "$(python3 -c 'import json;print(json.load(open("build/p3/prof-manifest.json"))["source_commit"])')" \
  --prof-manifest build/p3/prof-manifest.json
```

## Files

| File | What it is |
|---|---|
| `lkai_probe.py` | The hardware and permission probe (Lambda or any Linux host) |
| `lkai_prof.py` | Handler `lkai_prof.handler`: wraps `lkai_entry.handler`. `LKAI_PROF` picks `none`, `cprofile`, `pyspy` or `sampler` |
| `stage_run.sh` | Summarise, redact and stage a run under `results/<run-id>/` |
| `lkaisamp.c`, `lkai_symbolise.py` | The in-process native sampler and its standard-library ELF symboliser |
| `extensions/lkai-telemetry` | The Telemetry API extension, deployed as a layer |
| `build_prof_zip.sh`, `deploy_arms.py` | Build the profiler package; deploy and tear down the `lkai-p3-*` arms |
| `summarize_prof.py`, `fetch_xray.py` | Build `results.json` (lkai-result/1 plus `profiler_overhead`, `profiles`, `applicability`, `hot_spots`) |
| `offline_drive.py` | Runs the engine outside Lambda with `PHASE` markers; used by every offline profile |
| `graviton_profile.sh`, `graviton_container.sh` | The offline arm64 DevSpace profile (perf, py-spy --native, cProfile, lkaisamp) |
| `gpu_profile.sh`, `gpu_container.sh`, `gpu_rerun.sh` | The offline x86 and NVIDIA L4 profile (nvidia-smi, nsys, ncu) |
| `applicability.json` | The matrix, with evidence |
| `results/` | Probe outputs, offline evidence, and the Lambda run (redacted: no account ID or bucket name) |

## Cost and resources

- Lambda run: computed from REPORT lines in `results.json` → `cost`.
- Run `20261020T231634Z`: $9.61. Run `20261020T124451Z` (superseded): $8.36. The aborted first attempt is under
  `results/aborted-20261020T115546Z/`.
- The GPU host was already running (g6.4xlarge, IsaacLabDcvStack). This study used about 15 minutes of it. The only
  change to that host was installing the kernel module package matching its running kernel
  (`linux-modules-nvidia-580-open-6.8.0-1066-aws`; the operator approved, no reboot). Scratch files are in `/opt/lkai-p3`.
- Teardown: `deploy_arms.py teardown` deletes every published version, which stops SnapStart cache billing.
  `teardown --all` also deletes the functions, role and layer.
