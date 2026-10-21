> **SUPERSEDED by run `20261020T231634Z` (see `../../README.md`). The overhead table, the restore conclusions and
> the "first restore wave" section below are withdrawn.** This run kept one Lambda client across the SnapStart
> waits: its 45 restore "stalls" were client read timeouts that never reached Lambda, not a platform finding, and
> they left the `rep` restore control with one valid part-0xd40 sample in rounds 0, 2 and 4. Its noise floor was the
> X-Ray arm itself. The hot-spot shares below were computed over the retained top-N rows and overstate each share;
> `results.json` here was regenerated with full denominators (for example `buffer_set_tensor` 29.9%, `rhs_pack`
> 11.7% of init CPU samples) and with every overhead verdict withheld. The applicability matrix is unchanged.
> CPU part IDs (0xd40, 0xd0c, 0xd4f) are opaque labels: no processor generation is inferred from them (the
> current run's README, "Host variability"). `results.json` here carries the same `host_variability` block.

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

- `uname`: `5.10.255-268-309.1092.amzn2.aarch64`. `/proc/cpuinfo`: 6 processors, CPU part `0xd40` in
  all 7, with `sve i8mm bf16 asimddp`.
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
| py-spy `--native` / blocking | **denied** | `Failed to suspend process ... EPERM` (needs ptrace) | works on part 0xd40, about +29% per forward |
| perf | **denied** | `perf_event_open` EPERM ×3, no PMU, not in the image | works on part 0xd40 |
| eBPF (bpftrace, BCC) | **denied** | `bpf()` EPERM, `unprivileged_bpf_disabled=1`, no bpffs/tracefs | — |
| ptrace tools (gdb, strace, ltrace) | **denied** | `PTRACE_SEIZE`/`PTRACE_ATTACH` EPERM; none on `PATH` | — |
| lkaisamp (this directory's native sampler) | **works** | arm `sampler`: SIGPROF delivered; PCs symbolised in the function against the shipped libraries' `.symtab` | agrees with perf on part 0xd40 |
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

On part 0xd40 its per-phase ranking matches `perf record` (results below).

## Overhead (Lambda)

Each profiler arm is its own function (`lkai-p3-<arm>`, deployed by `deploy_arms.py`). Each has P1's configuration,
runtime version, GGUF object and zip bytes; the `lkai_prof.py` arms also carry the added files. The arms run through
P1's unmodified harness (`baseline/bench.py`), interleaved, 5 rounds × 10 cold + 10 restore + 10 warm passes of 10.
Overhead is `median(arm) − median(rep)` on Lambda's own REPORT numbers, compared within the same CPU part, the method P1 used.
`rep` is P1's REPORT-only arm, rebuilt from P1's zip as P3's own function so that no other card shares it.

Run `20261020T124451Z` (us-east-2, 5 rounds, 4,200 invocations, about $8.36 per its `cost` block, most of it the
SnapStart cache minimum on 35 published versions). Each cell is the mean over the 5 rounds of `median(arm) −
median(rep)` within the round, CPU part 0xd40 rows only, followed by the [min, max] over rounds. Warm cells pair
by request as well (25 cells of round × request; the five requests take 0.25 to 1.75 s), shown as a percentage. All
seven arms returned answers identical to P1's reference (4,154 answers compared, all bit-identical, 0 choice flips).
46 invocations errored and are excluded: 45 restore invokes that never started (below) and 1 cold `pyspy` request
failed by the profiler itself (py-spy's output was cut mid-character; `lkai_prof.py` now isolates every profiler
hook so that this cannot fail a request, with a regression test).

| Arm | Cold Init Duration | Cold Duration (first request) | Restore Duration | First request after restore | Warm Duration, per request |
|---|---|---|---|---|---|
| `none` (phase timers) | +8 ms [+4, +12] | +40 ms [−8, +114] | −110 ms [−267, +52] | −10 ms [−17, −2] | −0.5% [−3.6, +8.4] |
| `cprofile` | **+107 ms** [+99, +117] | **+165 ms** [+101, +219] | +55 ms [−72, +130] | −4 ms [−18, +7] | +0.1% [−2.2, +4.2] |
| `pyspy` | +11 ms [−5, +21] | +39 ms [−178, +247] | −32 ms [−201, +90] | **+24 ms** [+16, +34] | **+4.8%** [+0.0, +13.3] |
| `sampler` (lkaisamp) | +9 ms [−5, +26] | **+377 ms** [+133, +543] | −21 ms [−192, +64] | **+2,084 ms** [+1,831, +2,344] | +0.9% [−8.1, +8.5] |
| `xray` | +5 ms [+2, +11] | −10 ms [−145, +170] | −79 ms [−166, +52] | −12 ms [−30, −1] | −0.1% [−4.2, +4.8] |
| `telemetry` | **+98 ms** [+79, +130] | −85 ms [−171, +33] | +13 ms [−172, +172] | −12 ms [−25, +3] | +0.3% [−5.5, +6.3] |

Control medians (`rep`, part 0xd40): Init Duration 803 ms, cold Duration 35.1 s, Restore Duration 3.28 s, first
request after restore 682 ms, warm 711 ms (pooled over the five requests).

`xray` runs P1's zip unchanged; Lambda records the trace outside the function. Its row is the noise floor: a
same-code, different-environment difference of up to 11 ms on Init Duration, 170 ms on cold Duration, 166 ms on
Restore Duration, 30 ms on the first restored request and ±4.8% per warm request. Read against it:

- **Phase timers (`none`) and X-Ray: no measurable cost** on any phase.
- **cProfile: about +107 ms of Init Duration** (13%; consistent in every round) and +165 ms on the cold first
  request (0.5%). It has no measurable warm cost, because its per-invocation profile spans only the Python layer and
  the forward runs in C.
- **Telemetry API extension: about +98 ms of Init Duration** (12%, every round). This is the extension's own
  registration and subscription; Telemetry's `extension_register_ms` p50 is 30 ms, p90 99 ms. It has no measurable
  cost elsewhere.
- **py-spy (non-native): +4.8% median per warm request**, and above zero in every one of the 25 cells, plus +24 ms
  on the first restored request. The cost is the sampler process reading the interpreter at 100 Hz while 6 threads
  compute.
- **lkaisamp: about +0.9% per warm request (inside the noise), but +377 ms on the cold first request and +2.1 s on
  the first request after a restore.** Both deltas appear in Lambda's `Duration`, while the handler's internal
  `latency_ms` and phase records are unchanged. The sampler dumps and symbolises before `handler` returns, so the
  measured delta is outside the baseline's internal timing but still before Lambda receives the response. The
  summary record is logged about 2 s after the baseline handler returns. Attribution within that hook remains open:
  parsing the ELF tables takes 26 ms offline, which does not explain the Lambda delta.

The offline arm64 DevSpace (part 0xd40) overheads below are not Lambda numbers, but they rank the same way for the warm forward.

**Final latency comparisons use light instrumentation only:** the REPORT-only arm or P1's phase timers. Profiler arms
are for finding hot spots, not for timing them.

### First restore wave after a long publish

In every round (5 of 5), the first arm's restore wave had 9 of 10 concurrent invokes that never started: no
`RESTORE_START`, no throttle, a client read timeout at 120 s. One invoke succeeded normally. The affected arm was
`rep` in rounds 0, 2 and 4 and `telemetry` in rounds 1 and 3. The arm order alternates, so the stall follows wave
position, not code. Every later arm's 10 restores in the same round were clean. The first, aborted run
(`20261020T115546Z`) hit the same stall in its round 0 with a 900 s timeout. The cause is not established. One factor
these waves share: they hit the version published longest ago, about 18 minutes after its publish (seven versions are
published one after another at about 157 s each before any restore wave). This matters to every experiment using
`baseline/bench.py` with several arms: check restore rows for `client:ReadTimeoutError` before reading restore
latency. Errored rows are excluded from the distributions and listed under `errors`. `bench.py --read-timeout`
(new; default 900, recorded in `conditions.json`) bounds what a stall costs. As a result the `rep` restore control
has fewer samples (21 against 44 to 48 for the other arms).

## Baseline hot spots by phase (feeds E6)

Phases are ranked by wall time from P1's own timers. The `none` arm runs P1's phase records unchanged and costs
nothing measurable (above). Each phase's contents come from the profiler best placed to see them. **Lambda** rows are
from run `20261020T124451Z`, CPU part 0xd40 where split. **Offline** rows are the arm64 DevSpace (part 0xd40) and are not
Lambda.

### Cold start: 35.9 s first request (Init 0.81 s + Duration 35.1 s), p50, Lambda

An on-demand cold start loads the model inside the first invocation (PR #77's background load), so `Duration` holds it.

| Rank | Phase (P1 timer) | p50 | Share of first request | What the time is (evidence) |
|---|---|---|---|---|
| 1 | `weight_download` | 24.97 s | 69.5% | **Waiting on S3 over TLS.** py-spy (Lambda): 57% of init leaf samples in `ssl.py:1103 read`. cProfile (Lambda): `_ssl._SSLSocket.read` has 59% of init own-time (1,535 s summed over the 32 download threads). lkaisamp sees little here because waiting uses no CPU; `libc read` + `pwrite` (copies into the memfd) are 9.6% of init CPU samples (6.4% + 3.2%). 79 MiB/s. Network-bound: the E1 to E4 experiments target it |
| 2 | `model_load_mmap` | 8.39 s | 23.4% | **The KleidiAI weight repack, not I/O.** lkaisamp (Lambda, init): `ggml_backend_cpu_kleidiai_buffer_set_tensor` 35.2% (the copies in two CPU variants combined) + `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` 13.7% + `memset` 9.2% + `lroundf` 3.2% of init CPU samples. offline arm64 DevSpace (part 0xd40) lkaisamp, `model_load` phase only: `buffer_set_tensor` 46.8%, `rhs_pack` 22.8%, the variant's unsymbolised `+0x32000` 9.0%, `memset` 6.1%, all on **one thread** (1 sampled thread over the 7.9 s phase). py-spy (Lambda): `engine.py:114 __init__` (the native `dec_load`) holds 33% of init leaf samples. E5 (mmap-friendly format) targets it; a repacked-at-build-time format would remove it |
| 3 | `assets_load` | 0.95 s | 2.6% | Tokenizer and numpy head (`tokenizers.abi3.so` 2.3% of init CPU samples) |
| 4 | `imports` | 0.73 s | 2.0% | Python imports and bytecode compile (cProfile: `builtins.compile` 22.6 s own-time summed over threads) |
| 5 | `context_init` | 0.33 s | 0.9% | llama context creation |
| 6 | `warmup_inference` | 0.25 s | 0.7% | One forward, same kernels as steady state |
| 7 | `first_request_inference` | 0.24 s | 0.7% | The request itself |

### SnapStart restore: 3.99 s first request (Restore 3.30 s + Duration 0.68 s), p50, Lambda

| Rank | Phase | p50 | Share | What the time is |
|---|---|---|---|---|
| 1 | `report_restore_duration` | 3.30 s | 82.7% | Lambda's snapshot restore, before any function code runs (X-Ray `Restore` segment p50 3.25 s, 48 traces). No profiler inside the function can see it: it happens before the runtime resumes. Telemetry reports the same span (`platform.restoreReport`) |
| 2 | `first_request_inference` | 0.68 s | 17.0% | **2.8× the warm forward (0.24 s) for the same request, and half of it is `memset`.** Over all 44 restored first invocations on part 0xd40 (lkaisamp, 2,538 samples): `memset` 50.4%, the i8mm matmul 35.0%, then the warm kernels at 1 to 2% each. Warm, `memset` is 2.2%. Its callers, from the sampler's caller edges: `llama_kv_cache::clear` (651 samples) and `llama_memory_recurrent::clear` (628), the per-request cache clears. Warm, the same clears are cheap, so the likely extra cost is first-touch page faults on the restored cache memory (a fault's kernel time is charged to the instruction that touched the page). That part is an inference: Lambda allows no perf to count faults directly |
| 3 | `restore_to_invoke` | 0.011 s | 0.3% | after-restore hooks |

### Warm steady state: 0.25 to 1.75 s per request, Lambda

| Rank | Native symbol (Lambda lkaisamp, warm, part 0xd40; 26,653 samples over 440 invocations) | Share of CPU samples | offline arm64 DevSpace (part 0xd40) perf record (whole run) |
|---|---|---|---|
| 1 | `kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` (KleidiAI int8 matmul) | 49.7% | 47.7% |
| 2 | `ggml_vec_dot_f32` (f32 dot products: attention / recurrent state) | 8.2% | 9.4% |
| 3 | `ggml_compute_forward_gated_delta_net_one_chunk` (the linear-attention recurrence) | 7.8% | 8.3% |
| 4 | libgomp spin/barrier (`+0x27000`) | 5.7% | 5.6% (all libgomp) |
| 5 | `ggml_compute_forward_concat` | 5.3% | 1.2% |
| 6 | `ggml_compute_forward_fwht` (the runtime Hadamard rotation) | 5.2% | 4.0% |
| 7 | `ggml_compute_forward_ssm_conv` | 2.5% | 2.5% |
| 8 | `memset` | 2.2% | — |
| 9 | `kai_run_lhs_quant_pack_qai8dxp_f32` (activation quantisation) | 1.9% | 1.3% |
| 10 | `rms_norm_mul_fused`, `silu`, `swiglu` | 1.6%, 1.1%, 1.0% | 0.9%, 0.9%, 1.0% |

By library (Lambda, warm, part 0xd40): `libggml-cpu-armv8.6_1.so` 88.5%, libgomp 5.8%, libc 4.0%, numpy's OpenBLAS
0.6%. Python is negligible. py-spy (Lambda, warm): 93% of leaf samples sit in `engine.py:156 _fork`, the ctypes call
into the forward. cProfile (Lambda, warm): `_fork` 92% of own-time, `_eval` 6%, the numpy head (`cumsum`, `logits`)
under 2%. Part 0xd0c environments (6.3% of warm rows) load the dot-product variant
(`kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod`), which is kept in its own table
(`sampler_invoke_warm_0xd0c`).

### Pointers for E6

- Warm: about half the CPU time is the int8 matmul kernel, already the fastest KleidiAI path for this CPU. The
  non-matmul half (f32 dot products, the gated-delta-net recurrence, `concat`, `fwht`, OpenMP barriers: about 33%
  together) has more room. `concat` is 5.3% in Lambda against 1.2% offline, which is worth one look.
- Restore: half of the first forward after a restore is `memset` inside `llama_kv_cache::clear` and
  `llama_memory_recurrent::clear`. Clearing (touching) those caches once in an after-restore hook, before the first
  request, is a candidate one-change experiment. So is clearing only the cells that were used.
- Cold: after the S3 download (E1 to E4), the 8.4 s load is the KleidiAI repack (`buffer_set_tensor` +
  `rhs_pack`). Offline it runs on one thread for the whole phase, so threading it or doing it at build time (E5)
  are the candidates.

### Offline comparison (NOT LAMBDA)

| Host | Steady forward, `ticket-route-1q` / `incident-review-4q` | Notes |
|---|---|---|
| Lambda, part 0xd40, 6 vCPU (`rep` warm, REPORT Duration p50) | 0.247 / 1.749 s | the deployed function |
| arm64 DevSpace (part 0xd40), same image, 6 CPUs | 0.243 / 1.739 s | within 2% of Lambda; perf available |
| AMD EPYC 7R13 (g6.4xlarge host CPU), 6 threads, `haswell` variant, no KleidiAI | 0.670 / 4.488 s | x86 is not a deployment target of this example |
| NVIDIA L4, all layers offloaded (CUDA, sm_89) | 0.023 / 0.106 s | top kernels `mul_mat_q` (Q8_0) at 61.5% of GPU kernel time over all its tile sizes; nvidia-smi peak memory and utilisation in `results/offline/gpu-host/` |

## Reproduce

```bash
# 0. P1's build/ (function.zip, GGUF, deployment.json) on an arm64 host with Docker: baseline/build.sh, deploy.py up
B=examples/lambda-kleidiai/baseline/build
P=examples/lambda-kleidiai/profiling
# 1. Profiler package: P1's zip plus lkai_prof.py, lkai_probe.py, lkai_symbolise.py, liblkaisamp.so, py-spy
$P/build_prof_zip.sh $B/function.zip build/p3
cp $B/deployment.json build/p3/p1-deployment.json
# 2. Deploy the arms (lkai-p3-*, own role) and run P1's harness over them
AWS_PROFILE=<p> python $P/deploy_arms.py up --p1-state build/p3/p1-deployment.json --build build/p3
AWS_PROFILE=<p> python examples/lambda-kleidiai/baseline/bench.py --build build/p3 --region us-east-2 --rounds 5 \
  --concurrency 10 --warm 10 --read-timeout 120 --arms rep,none,cprofile,pyspy,sampler,xray,telemetry --out build/p3/runs/<run-id>
AWS_PROFILE=<p> python $P/deploy_arms.py teardown --build build/p3          # published versions (SnapStart cache)
AWS_PROFILE=<p> python $P/fetch_xray.py build/p3/runs/<run-id>
# 3. In-Lambda probe: invoke any lkai_prof arm with {"mode": "lkai_probe"}, or deploy lkai_probe.py alone
# 4. Offline, NOT Lambda
$P/graviton_profile.sh $B/function.zip $B/model/s2b-h512-Q8_0.gguf build/p3/graviton                 # arm64 + Docker
GGUF_URL=... ZIP_URL=... GGUF_SHA256=... REPO_URL=... REPO_REF=<commit> bash $P/gpu_profile.sh /opt/lkai-p3   # GPU host
python $P/offline_summary.py --graviton build/p3/graviton --gpu <gpu out> --out $P/results/offline
# 5. Summary and publication (P2)
python $P/summarize_prof.py build/p3/runs/<run-id> --reference examples/lambda-kleidiai/baseline/reference/outputs.json \
  --meta $P/meta.json --applicability $P/applicability.json --offline $P/results/offline --probe $P/results/lambda-probe/q1.json
```

## Files

| File | What it is |
|---|---|
| `lkai_probe.py` | The hardware and permission probe (Lambda or any Linux host) |
| `lkai_prof.py` | Handler `lkai_prof.handler`: wraps `lkai_entry.handler`. `LKAI_PROF` picks `none`, `cprofile`, `pyspy` or `sampler` |
| `lkaisamp.c`, `lkai_symbolise.py` | The in-process native sampler and its standard-library ELF symboliser |
| `extensions/lkai-telemetry` | The Telemetry API extension, deployed as a layer |
| `build_prof_zip.sh`, `deploy_arms.py` | Build the profiler package; deploy and tear down the `lkai-p3-*` arms |
| `summarize_prof.py`, `fetch_xray.py` | Build `results.json` (lkai-result/1 plus `profiler_overhead`, `profiles`, `applicability`, `hot_spots`) |
| `offline_drive.py` | Runs the engine outside Lambda with `PHASE` markers; used by every offline profile |
| `graviton_profile.sh`, `graviton_container.sh` | The offline arm64 DevSpace (part 0xd40) profile (perf, py-spy --native, cProfile, lkaisamp) |
| `gpu_profile.sh`, `gpu_container.sh`, `gpu_rerun.sh` | The offline x86 and NVIDIA L4 profile (nvidia-smi, nsys, ncu) |
| `applicability.json` | The matrix, with evidence |
| `results/` | Probe outputs, offline evidence, and the Lambda run (redacted: no account ID or bucket name) |

## Cost and resources

- Lambda run: computed from REPORT lines in `results.json` → `cost`.
- The GPU host was already running (g6.4xlarge, IsaacLabDcvStack). This study used about 15 minutes of it. The only
  change to that host was installing the kernel module package matching its running kernel
  (`linux-modules-nvidia-580-open-6.8.0-1066-aws`; the operator approved, no reboot). Scratch files are in `/opt/lkai-p3`.
- Teardown: `deploy_arms.py teardown` deletes every published version, which stops SnapStart cache billing.
  `teardown --all` also deletes the functions, role and layer.
