# E14: the ggml f32 CONCAT in the warm forward

E14 owns `ggml_compute_forward_concat` from P3's profile. That profile gave it 5.3–5.6 % of Lambda warm CPU samples
and 1.2 % offline (P3 run `20261020T231634Z`, `profiling/README.md` on `lkai/bc9a-profiler`). E14 found what the
concat is, why one copy runs on one thread, and a one-change fix. It measured the fix offline and on Lambda, and it
explained the share difference. Branch `lkai/19b2-concat`, created from P1 `aac59d45555ebded9357c23f6f8e4f939edf25d7`.

**Result: no change on Lambda. Answers are bit-identical. No deployment change is recommended.** The full result is
`results/20261021T232302Z/results.json` (lkai-result/1), published to P2 as `E14-concat/20261021T232302Z`.

| | Offline, arm64 host, part 0xd40, NOT LAMBDA | Lambda, us-east-2, 10,240 MB |
|---|---|---|
| concat calls per request | 18–90 | 18–90 (same) |
| thread-0 concat time per request, P1 kernel (0xd40) | 15.7–88.0 ms | 15.8–117.5 ms |
| that time as a share of the forward | 2.7–6.6 % | 2.6–6.9 % |
| rows vs ctl, warm `steady_state_inference` | −1.0 % to −6.0 % (two runs, 4 interleaved rounds each) | no consistent change (below) |
| answers | 600 hidden-state hashes identical | 1126/1126 bit-identical to P1's reference |

## What the concat is

The qwen35 graph has exactly one f32 CONCAT: `build_conv_state` in `src/models/delta-net-base.cpp` (llama.cpp
`f74e2fa6a`, called from `qwen35.cpp`). There is one per Gated DeltaNet layer: 18 of the 24 trunk layers (GGUF
`block_count` 25 includes one MTP layer; every 4th trunk layer is full attention, and the probe saw nodes
`conv_input-0..22` skipping 3, 7, 11, …):

```
conv_input = concat(conv_states [3, 6144, n_seqs], transpose(qkv_mixed) [n_tokens, 6144, n_seqs], dim 0)
```

`d_conv - 1 = 3` (GGUF `ssm.conv_kernel = 4`). The channel count is 6144, which is `ssm.inner_size` 2048 plus 2 ×
16 groups × 128 state. `src1` is a transposed view, so it is not contiguous and the kernel is a strided element
gather. `ggml_compute_forward_concat_f32` splits work only by dst's dim 2 (`for i2 = ith; i2 < ne2; i2 += nth`,
"TODO: smarter multi-threading"), and dim 2 is the ubatch's sequence count. The shim decodes the state prefix as one
sequence and the questions as sequences 1..nq. Every state-prefix ubatch, and every forward of a one-question
request, therefore copies on thread 0 while the other 5 threads wait at the next barrier.

The offline probe (`patch_concat_probe.py`: thread-0 CLOCK_MONOTONIC around each f32 CONCAT) measured per request:

| request | calls | one-sequence calls | dst MB | thread-0 ms | of which one-sequence | share of ctl p50 |
|---|---|---|---|---|---|---|
| ticket-route-1q | 18 | 18 | 38.0 | 16.8 | 16.8 | 6.6 % |
| payouts-3q | 72 | 36 | 96.0 | 15.7 | 4.0 | 2.7 % |
| cracked-screen-2q | 54 | 36 | 70.3 | 17.6 | 6.6 | 3.9 % |
| agent-action-guard-2q | 54 | 36 | 140.7 | 51.8 | 37.9 | 5.8 % |
| incident-review-4q | 90 | 36 | 279.1 | 88.0 | 72.3 | 4.9 % |

(`results/20261021T232302Z/offline-20261021b/offline-summary.json`; 40 steady-state requests each.)

## The change

`patch_concat.py` replaces only the f32 kernel's loop nest. It splits dst's rows `(i1, i2, i3)`, flattened, into one
contiguous block per thread. The element loop, the src0/src1 selection and the store are the same statements, and
every dst element is written exactly once from the same source element, so values cannot change.
`tests/test_lkai_e14.py` pins that property: every row is written once for 1–64 threads. The f16, i8 and generic
paths are untouched. Of P1's zip, only the six `lib/libggml-cpu-*.so` differ (`concat_zip.py`, `zips.json`). They
were rebuilt from P1's patched llama.cpp tree in P1's pinned build image, with gcc 11.5.0, the same compiler as P1.

## Offline (NOT LAMBDA)

This used P1's DevSpace `ds-lkai-p1-rs5ghbzv` (part 0xd40, implementer 0x41, variant 0x1, revision 1, kernel
6.1.188, nproc 16). The container ran in P1's pinned Lambda python3.12 image, pinned to CPUs 10–15 with 6 threads and
P1's environment (DEC_KV=f32, DEC_UBATCH=512, N_SEQ=17). The ggml loader picked `armv8.6_1`. ctl (P1's libs), rows
and probe were interleaved, and the order alternated each round. Each run was 10 passes over P1's five requests.
`offline_e14.py` hashed every raw hidden-state row the shim returned.

- Bit identity: all 600 (run b) and 400 (run a) hidden-state SHA-256s matched ctl round 1's, and so did the answers.
- Run a (`results/20261021T232302Z/offline-20261021a/`): rows was 2.0–6.0 % faster at pooled p50, and the sign was the same in all 20
  request × round cells. Its probe arm had a table-reset bug (fixed in `patch_concat_probe.py`), so its probe JSON
  is not kept.
- Run b (`results/20261021T232302Z/offline-20261021b/`), with load average 8–22 from other cards' builds on the shared 16-CPU host: rows was
  −1.0 % to −4.4 % on four requests. On incident-review-4q it was +0.8 % pooled, with a +37.9 % round when the host
  was busiest. The host load is recorded per run in `loadavg.txt`.

## Lambda

`e14deploy.py` and `e14bench.py` follow E6's pattern. They deploy `lkai-19b2-ctl` (P1's zip plus `e14_entry.py`)
and `lkai-19b2-rows` (the same zip with the rows libraries), both with P1's configuration and pinned runtime version
ARN, and both read P1's weights object. The benchmark is P1's `Bench` with I1's fresh client per wave (5764b9e,
brought in unchanged as `baseline/bench.py`). The cold leg ran 5 rounds; in each, a cold wave of 10 per arm (order
alternating), then 4 warm passes over the five requests 10 wide. The warm passes were **interleaved request by
request** (ctl r1, rows r1, ctl r2, …), with the leading arm alternating per pass. The restore leg ran 2 rounds: a
published version per arm, 10 restores, then 2 interleaved warm passes. Totals: 2540 invocations, 0 errors, 0
class mismatches (cold 100, restore 40, warm 2400). Lambda cost $1.48 (`cost` in results.json, SnapStart cache at
the 3-hour minimum). The probe and SIGPROF functions together cost less than $0.25 more.

Arms are compared only within one CPU part and one (round, request, leg) cell, against the control's own spread in
that cell. `engine_env_bootstrap` gives environment-level 95 % intervals, treating environments rather than
invocations as the independent units.

| CPU part | environments (ctl/rows), cold leg | cells within control spread | median cell Δ | env-level Δ by request (95 % CI) |
|---|---|---|---|---|
| 0xd40 | 36 / 38 | 35 / 35 | +0.70 % | −1.26 % [−1.68, −0.65] to +3.12 % [+2.05, +3.86] |
| 0xd4f | 9 / 9 | 15 / 15 matched | +1.15 % | −1.01 % [−2.85, +0.89] to +2.49 % [+0.75, +5.95] |
| 0xd0c | 5 / 3 | 9 / 15 matched | −1.62 % | −3.42 % [−3.81, −3.04] to +3.03 % [+2.66, +3.41] |

On each part some requests are faster and some slower, and the signs do not follow a request's offline gain. On
0xd40, for example, ticket-route-1q is +1.2 % on Lambda but −6.0 % offline. REPORT Duration agrees with P1's
engine timer. Cold and restore phases are unaffected, as expected, since the concat runs only in forwards. Every
warm-after-restore cell on 0xd40 (10/10) is within the control spread.

## Why the shares differed, and why the fix did not carry over

1. **The 1.2 % was not the steady-state number.** P3's offline perf share is over the whole run, including about 9
   s of model load. On the same samples, lkaisamp's steady-state share is 2.65 %. That is still CPU share, though,
   and the concat runs on one thread while the others wait (most of the wait is OpenMP barrier spinning, which is
   counted in `libgomp`). Offline, the thread-0 timer puts it at 0.9 % of CPU seconds but 4.8 % of wall time.
2. **The Lambda share is real.** The diagnostic function `lkai-19b2-probe` ran P1's kernel with the thread-0 timer
   on Lambda (`probe_lambda.py`, `probe_summary.py`; 190 of 200 warm invocations logged, 10 cold waves failed on a
   payload bug that was then fixed). On 0xd40 it shows the same calls per request and the same thread-0 time as
   offline: ticket-route-1q 16.8 vs 16.8 ms, payouts-3q 15.8 vs 15.7, cracked-screen-2q 17.7 vs 17.6,
   agent-action-guard-2q 52.5 vs 51.8, and incident-review-4q 117.5 vs 88.0. That is 2.6–6.9 % of engine time, in
   line with P3's 5.6 % sample share. On 0xd0c the same copy takes 1.6–2.5× as long.
3. **The sampler is not biased toward the main thread on Lambda.** `sigprof_probe.py` ran P3's deployed arm64
   `liblkaisamp.so` with 6 equal native threads in 4 fresh environments on 0xd40 (kernel 5.10.255). The main thread
   received 15.8 % of 2906 signals, against a fair share of 16.7 % (range 8.5–21.9 % per invocation). On the local
   x86 6.1 host the same probe was also fair. Delivery bias does not explain the gap.
4. **What is unexplained.** On Lambda, removing about 17–118 ms of single-threaded copy per request does not shorten
   the forward. Offline, the same libraries shorten it by roughly the copy's duration. P3's records show warm
   environments using 4.5 CPU-seconds per engine-second on 0xd40, against 5.2–5.7 offline. One candidate, untested:
   while thread 0 copies, the 5 waiting threads spin in libgomp's barrier, and under Lambda's CPU allocation that
   spinning may cost the copying thread less than it does on a dedicated host, so there is less to win back. Testing
   it needs a change to OpenMP's wait policy (`OMP_WAIT_POLICY` / `GOMP_SPINCOUNT`), which is a different variable
   from E14's. H1's fixed-host EC2 baseline (task-…-cb67) is the place to isolate the host.

## Host variability

- Observed per environment, at both import and first invocation (`results.json` → `host_variability`): MIDR part,
  variant, revision, AT_HWCAP/HWCAP2, kernel, CPU count/affinity, Python, cgroup files, and the SHA-256 of the mapped
  ggml variant.
  - Parts: 0xd40 (variant 0x1, revision 1, hwcap 0xfffffff, hwcap2 0x1f201) → `armv8.6_1`; 0xd4f (0x0/1,
    0x2ff3ffff/0x3f3bf) → `armv8.6_2`; 0xd0c (0x3/1, 0x119fff/0x0) → `armv8.2_2`.
  - Every environment ran kernel 5.10.255-268-309.1092.amzn2.aarch64 with 6 CPUs and affinity 6.
  - The cgroup v2 files are not exposed and are recorded as `unknown`.
  - Each arm mapped its own library build. ctl's `armv8.6_1` is P1's (7aadff86…) and rows' is the patched build
    (058b4805…).
  - No restored environment reported a CPU part different from its snapshot host's.
- Unknown: availability zone, physical host, the snapshot-builder's host, and processor generation. No generation
  is inferred from a part ID.
- P3's existing Lambda data (`results/20261021T232302Z/lambda-concat-p3.json`, `analyse_lambda.py`): every P3 warm sampler row came from an
  environment whose first invocation was a restore. P3 therefore cannot separate warm-after-cold from
  warm-after-restore. E14's own run has both, and they agree (above). The restore-memory first-touch hypothesis is
  not supported: P3's invocation 2 and invocation ≥ 3 shares overlap (6.5 % vs 5.6 % on 0xd40), and cold first
  invocations show the concat as well.

## Limitations

- The offline run b host was shared and busy. Run a was quieter. Both are offline and decide nothing about
  deployment.
- 0xd0c and 0xd4f each had 3–9 environments per arm in the cold leg, and one restore cell on 0xd0c had 1 ctl
  environment.
- The probe arm's latency is not comparable with either arm's, because its timer runs on every concat.
- `e14_entry.py` in the measured zips has SHA-256 439b84bf…. The committed file differs only in one comment line,
  changed for lint, and has SHA-256 84c243d5….
- The three functions, the role, the log groups and the zips under `e14/code/` in P1's bucket are still deployed,
  at the operator's choice. None has a published version, so none incurs SnapStart cost. Remove them with
  `e14deploy.py teardown --all` and by deleting `lkai-19b2-probe` and `lkai-19b2-sigprof`.

## Files

| file | what |
|---|---|
| `patch_concat.py` | the change (f32 concat, rows across threads) |
| `patch_concat_probe.py` | offline/diagnostic thread-0 timer; never in a measured zip |
| `offline.sh`, `offline_e14.py`, `offline_summary.py` | offline A/B with hidden-state hashing |
| `concat_zip.py`, `e14_entry.py`, `e14deploy.py`, `e14bench.py`, `summarize_e14.py`, `meta.json` | Lambda A/B and P2 result |
| `probe_lambda.py`, `e14_probe_entry.py`, `probe_summary.py` | Lambda diagnostic concat timer |
| `sigprof_probe.py`, `sigprof_bias.c` | SIGPROF thread-delivery checks (Lambda, local) |
| `analyse_lambda.py` | P3's sampler logs, re-split by class, part, origin, invocation and request |
| `results/20261021T232302Z/` | results.json; raw Lambda logs and invocations (`cold/`, `restore/`), `probe/`, `sigprof-probe/`, both offline runs, P3 re-split, P1 build manifest |
