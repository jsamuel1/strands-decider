# H1: host variability of the lambda-kleidiai Lambda runs

Lambda picks a host for every new execution environment. This directory answers three questions with the runs the
initiative has already paid for, without a new invocation:

1. Which CPU identities did those runs land on, by arm and start class, and what could not be observed?
2. How much does latency depend on that identity, and how much does a run's identity mix move a comparison?
3. Is placement controlled well enough to compare arms, and under what rules?

It also adds the instrument the old runs lacked: [`baseline/hostinfo.py`](../baseline/hostinfo.py) records each
execution environment's full CPU identity, features, HWCAPs, cores, cgroup limits and kernel (see
[Harness change](#harness-change)).

**CPU part IDs are opaque labels here.** A part ID is the value `/proc/cpuinfo` printed. No processor generation or
microarchitecture is named from a part ID, a ggml variant or a timing, and `tests/test_lkai_hostvar.py` rejects such a
name in the generated result.

## Decision

**Lambda placement is not controlled.** Every one of the 10 runs drew more than one CPU part (all 10 drew 0xd0c,
0xd40 and 0xd4f), and the share of each part changes from run to run and between start classes. The same request
on the same code takes about 1.8x as long on 0xd0c as on 0xd40. One environment keeps one part for its whole life
(0 of 3,138 environments and 0 of 3,939 log streams reported two), so the part is a property of the environment, not
of the request.

What that means for every experiment card (E1 to E15, T0 to T3, R):

- Compare arms **within one CPU part**, measured **in the same rounds**. Report the part mix next to every pooled
  number. P1's `summarize.by_cpu` / `engine_by_cpu_part` already do the split.
- A row with no recorded part is **unassessed**. Never treat "unknown" as a shared stratum.
- Never compare against another run's numbers without the split: the part mix differs by run.
- CPU-bound phases (engine, `model_load_mmap`, restore first request) move with the part. An I/O phase
  (`weight_download`) need not, but check it per part before pooling.
- With P1's harness, warm passes reuse the environments the restore wave created, so the warm part mix is the
  restore mix (no 0xd4f). E3 and E6 run warm passes on cold environments and do see 0xd4f warm.

Limits of this decision:

- One part per environment says nothing about the physical host or the AZ. Neither is observable in a function.
- Feature flags, HWCAPs, MIDR, cgroup limits, core count and kernel were not recorded by any source run. P3's probe
  function read implementer `0x41`, variant `0x1`, revision `1`, one Features line, 6 CPUs and one kernel release, in
  7 environments, all part 0xd40. Nothing is known about those fields on 0xd0c or 0xd4f.
- 0xd4f appeared in cold rows of every run and in E6's warm rows. It never served a restore and never built a
  snapshot: all 174 published versions built their snapshots on 0xd0c and 0xd40.
- Invocations cluster by environment and round, so per-invocation counts overstate the independent sample size.
  Environment counts are given beside them.
- Interleaving: every source run alternated or rotated arm order between rounds. Within a round, each arm's cold and
  restore waves ran one arm after another (sequential, not interleaved); P1's warm passes alternate arm order pass by
  pass. Time-within-round drift is therefore not controlled for cold and restore.
- Restore latency from the runs that used one pooled invoke client (P1, E2, E3, E5, P3a; E4a unknown) is suspect:
  the first restore wave could stall on dead pooled connections (I1 restore-stall, P3). REPORT `Restore Duration`
  is Lambda's own number and is unaffected; client round-trip times are not.

## The concat question: host or code?

P3 found `ggml_compute_forward_concat` at 5.3% (run 20261020T124451Z) and 5.6% (rerun 20261020T231634Z) of Lambda warm
CPU samples on part 0xd40, against 1.2% offline under `perf`. The same `function.zip` and the same part ID ran on both
sides, so neither a code change nor a part-ID mix explains it. Inside Lambda the share is similar on both parts that
were sampled (5.27% on 0xd40 and 5.93% on 0xd0c in run 1; 5.64% and 6.47% in run 2). Two differences remain. The
measuring tool: offline, the same sampler gave 2.65% and `perf` gave 1.16%, so the tool alone moves the share about
2x. And the host conditions nobody recorded: kernel 5.10 in Lambda against 6.1 offline, cgroup CPU quota, core count.
Same sampler and same part, the gap is 5.64% against 2.65%. **It is not attributable to code from this evidence, and
it stays open** until hostinfo-instrumented Lambda rows exist to compare host conditions.

## Fixed-host reference

The operator chose to reuse P3's existing offline arm64 DevSpace rather than run a new fixed-host baseline (H1 ask,
2026-10-21). Its numbers are **fixed-host, not Lambda latency**. Recorded by P3: CPU part 0xd40, 16 CPUs, kernel
`6.1.188-233.386.amzn2023.aarch64`, `cpu.max` `max 100000` (no CPU quota), Python 3.12.15, P1's deployed
`function.zip` and GGUF in P1's pinned Lambda image, 6 threads pinned to 6 CPUs with taskset. Not recorded:
instance type, implementer/variant/revision, Features, HWCAPs, frequency governor, other load. The block is
`host_variability.fixed_host` in the result. Lambda cold, warm and restore validation and cost still decide
deployment conclusions.

## Data

Sources are listed in [`sources.json`](sources.json): each run's fork branch and the exact commit it is read from
(`git show <commit>:<path>`, never a working tree), its control arm(s), and whether its harness used one pooled
invoke client or a fresh one per wave, with the evidence for that. Three runs are excluded there with the reason.

The latency tables below use the **same-code control pool**: the control-arm rows of every run whose function
configuration is byte-identical (CodeSha256, memory, architecture, runtime, handler, tracing, layers, environment
without `COLD_TAG`). That is P1 (`inst`, `rep`), E1 (`gp`) and E5 (`base`). The weights' bucket differs between runs
and is not in the key, so download-dominated phases are not pooled across runs (they are `null` in `phases`).

<!-- hostvar:tables begin -->

### CPU part IDs by run and start class (all arms)

| run | invoke client | class | 0xd0c | 0xd40 | 0xd4f | unknown |
|---|---|---|---:|---:|---:|---:|
| E1 | fresh-per-wave | cold | 12 | 171 | 17 | 0 |
| E1 | fresh-per-wave | restore | 16 | 184 | 0 | 0 |
| E1 | fresh-per-wave | warm | 173 | 1827 | 0 | 0 |
| E2 | pooled | cold | 4 | 84 | 12 | 0 |
| E2 | pooled | restore | 5 | 95 | 0 | 0 |
| E2 | pooled | warm | 50 | 950 | 0 | 0 |
| E3 | pooled | cold | 51 | 423 | 26 | 0 |
| E3 | pooled | restore | 0 | 60 | 0 | 0 |
| E3 | pooled | warm | 51 | 483 | 26 | 0 |
| E4a | unknown | cold | 15 | 124 | 11 | 0 |
| E4a | unknown | restore | 18 | 132 | 0 | 0 |
| E4a | unknown | warm | 180 | 1320 | 0 | 0 |
| E4b | fresh-per-wave | cold | 37 | 134 | 29 | 0 |
| E4b | fresh-per-wave | restore | 23 | 177 | 0 | 0 |
| E4b | fresh-per-wave | warm | 230 | 1770 | 0 | 0 |
| E5 | pooled | cold | 5 | 84 | 11 | 0 |
| E5 | pooled | restore | 8 | 92 | 0 | 0 |
| E5 | pooled | warm | 80 | 920 | 0 | 0 |
| E6 | fresh-per-wave | cold | 8 | 101 | 11 | 0 |
| E6 | fresh-per-wave | restore | 2 | 58 | 0 | 0 |
| E6 | fresh-per-wave | warm | 43 | 562 | 55 | 0 |
| P1 | pooled | cold | 11 | 84 | 5 | 0 |
| P1 | pooled | restore | 1 | 99 | 0 | 0 |
| P1 | pooled | warm | 10 | 990 | 0 | 0 |
| P3a | pooled | cold | 33 | 295 | 22 | 0 |
| P3a | pooled | restore | 20 | 285 | 0 | 45 |
| P3a | pooled | warm | 220 | 3280 | 0 | 0 |
| P3b | fresh-per-wave | cold | 66 | 287 | 47 | 0 |
| P3b | fresh-per-wave | restore | 42 | 358 | 0 | 0 |
| P3b | fresh-per-wave | warm | 420 | 3580 | 0 | 0 |

Totals: {"0xd0c": 1834, "0xd40": 19009, "0xd4f": 272, "unknown": 45}. Environments with a part: 3,138, with more than one: 0. Log streams with rows: 3,939 (21,115 of 21,160 rows linked), with more than one part: 0. Snapshot-building inits: 488 streams over 174 versions, by part {"0xd0c": 240, "0xd40": 238, "unknown": 10}; versions by part set {"0xd0c+0xd40": 174}.

### ggml CPU variant loaded, by part (cold and snapshot-build streams)

| part | variants (streams) |
|---|---|
| 0xd0c | `libggml-cpu-armv8.0_1.so,libggml-cpu-armv8.2_2.so` (482) |
| 0xd40 | `libggml-cpu-armv8.0_1.so,libggml-cpu-armv8.6_1.so` (2026) |
| 0xd4f | `libggml-cpu-armv8.0_1.so,libggml-cpu-armv8.6_2.so` (191) |

### Latency by CPU part, same-code control pool (E1, E5, P1; 1,800 invocations)

| class | metric | 0xd0c | 0xd40 | 0xd4f | unknown |
|---|---|---|---|---|---|
| cold | first_request_total_s | 48.52 s (n=10) | 35.95 s (n=128) | 34.34 s (n=12) | n/a |
| cold | probe_engine_s | 439 ms (n=10) | 239 ms (n=128) | 210 ms (n=12) | n/a |
| restore | restore_duration_s | 4.12 s (n=7) | 3.26 s (n=143) | n/a | n/a |
| restore | probe_engine_s | 1,063 ms (n=7) | 672 ms (n=143) | n/a | n/a |
| warm | engine_s | 1,114 ms (n=73) | 627 ms (n=1427) | n/a | n/a |
| warm | probe_engine_s | 435 ms (n=15) | 244 ms (n=285) | n/a | n/a |

### Run-to-run spread of control medians (same-code pool)

| class | metric | pooled range (CV) | 0xd0c range (CV) | 0xd40 range (CV) | 0xd4f range (CV) | unknown range (CV) |
|---|---|---|---|---|---|---|
| cold | first_request_total_s | 0.11 s (0.2%, 3 runs) | 0.12 s (0.2%, 2 runs) | 0.14 s (0.2%, 3 runs) | 0.37 s (0.8%, 2 runs) | n/a |
| cold | probe_engine_s | 2.3 ms (0.5%, 3 runs) | 1.5 ms (0.2%, 2 runs) | 3.1 ms (0.7%, 3 runs) | 6.2 ms (2.1%, 2 runs) | n/a |
| restore | restore_duration_s | 0.03 s (0.5%, 3 runs) | 0.04 s (0.7%, 2 runs) | 0.04 s (0.6%, 3 runs) | n/a | n/a |
| restore | probe_engine_s | 10.2 ms (0.8%, 3 runs) | 9.3 ms (0.6%, 2 runs) | 9.8 ms (0.7%, 3 runs) | n/a | n/a |
| warm | engine_s | 20.8 ms (1.6%, 3 runs) | 62.3 ms (3.0%, 3 runs) | 20.8 ms (1.7%, 3 runs) | n/a | n/a |
| warm | probe_engine_s | 2.4 ms (0.5%, 3 runs) | 0.6 ms (0.1%, 2 runs) | 2.3 ms (0.5%, 3 runs) | n/a | n/a |

### Within-run comparisons where the part mix moved the pooled result (17 of 188; floor 1% of control median)

| run | control → arm | class | metric | pooled Δ | within-part Δ | mix (control / arm) | flag |
|---|---|---|---|---:|---|---|---|
| P1 | inst → rep | warm | engine_s | -33.7 ms | 0xd40 -8.1 | 0xd0c:10 0xd40:490 / 0xd40:500 | mix_shift |
| E1 | gp → xz | warm | engine_s | +35.7 ms | 0xd0c +18.5, 0xd40 +7.9 | 0xd0c:33 0xd40:467 / 0xd0c:94 0xd40:406 | mix_shift |
| E1 | gp → xzv | warm | engine_s | -63.3 ms | 0xd40 +3.8 | 0xd0c:33 0xd40:467 / 0xd40:500 | mix_shift |
| E3 | ctl → aa | warm | probe_engine_s | -3.7 ms | 0xd40 -1.1 | 0xd0c:3 0xd40:17 / 0xd40:10 | mix_shift |
| E3 | ctl → crt8 | warm | engine_s | +20.2 ms | 0xd0c +1.3, 0xd40 +14.2 | 0xd0c:6 0xd40:72 0xd4f:2 / 0xd0c:4 0xd40:44 0xd4f:2 | mix_shift |
| E3 | ctl → crtf16 | cold | first_request_total_s | +1.01 s | 0xd0c -0.22, 0xd40 +0.11 | 0xd0c:6 0xd40:42 0xd4f:2 / 0xd0c:8 0xd40:39 0xd4f:3 | mix_shift |
| E3 | ctl → crtf16 | warm | probe_engine_s | +3.7 ms | 0xd40 +0.7 | 0xd0c:3 0xd40:17 / 0xd0c:2 0xd40:8 | mix_shift |
| E3 | ctl → ldr | warm | probe_engine_s | -1.4 ms | 0xd40 +1.2 | 0xd0c:3 0xd40:17 / 0xd40:10 | mix_shift |
| E3 | ctl → t16x64 | warm | engine_s | +14.7 ms | 0xd40 +8.7 | 0xd0c:6 0xd40:72 0xd4f:2 / 0xd40:48 0xd4f:2 | mix_shift |
| E3 | ctl → t16x64 | warm | probe_engine_s | +0.6 ms | 0xd40 +3.2 | 0xd0c:3 0xd40:17 / 0xd40:10 | mix_shift |
| E3 | ctl → t64x64 | warm | probe_engine_s | +2.3 ms | 0xd40 +4.9 | 0xd0c:3 0xd40:17 / 0xd40:20 | mix_shift |
| E3 | ctl → t8x128 | warm | engine_s | +6.0 ms | 0xd40 -6.7 | 0xd0c:6 0xd40:72 0xd4f:2 / 0xd0c:2 0xd40:46 0xd4f:2 | mix_shift |
| E3 | ctl → t8x128 | warm | probe_engine_s | +0.1 ms | 0xd40 +2.8 | 0xd0c:3 0xd40:17 / 0xd40:10 | mix_shift |
| E4a | base → streamfile | warm | engine_s | +17.0 ms | 0xd0c -6.9, 0xd40 +2.0 | 0xd0c:60 0xd40:440 / 0xd0c:70 0xd40:430 | mix_shift |
| P3a | rep → cprofile | warm | engine_s | -25.9 ms | 0xd0c -70.9, 0xd40 -58.5 | 0xd0c:29 0xd40:444 / 0xd0c:30 0xd40:470 | mix_shift |
| P3b | rep → cprofile | warm | engine_s | -17.0 ms | 0xd0c -9.4, 0xd40 +5.3 | 0xd0c:79 0xd40:421 / 0xd0c:19 0xd40:481 | mix_shift |
| P3b | rep → telemetry | warm | engine_s | +13.6 ms | 0xd0c -39.5, 0xd40 -20.0 | 0xd0c:79 0xd40:421 / 0xd0c:102 0xd40:398 | sign_flip |

<!-- hostvar:tables end -->

The full record, including every comparison (not only the flagged ones), per-run distributions and the part mix of
every run and class, is `results/<run>/results.json` → `host_variability`.

## Harness change

`baseline/hostinfo.py` is new. `baseline/lkai_entry.py` calls `hostinfo.snapshot()` once at init and again in the
after-restore hook (a restored copy can run on a different host from the one that built its snapshot). The full
record rides on the environment's first invoke **response** as `lkai.host`, with the invocation context's log stream
name, so bench.py stores it in the invocation row with no bench.py or summarize.py change. Log lines carry only a
16-hex `host_fingerprint`: bench.py keeps a 4 KB log tail, and a full record there could push the START line out.
The snapshot-building init's full record rides on `before_snapshot`. The REPORT-only arm does not read it.

| | |
|---|---|
| Old harness | `70eab7be69f86ddeb8c64158d94df344b5180098` (P1 fork tip, `feat/lambda-kleidiai-baseline`) |
| New harness | the first commit on `lkai/cb67-host-variability` (see the result's `commits.experiment`) |
| Why | the host-variability rules ask for implementer/variant/revision, features, MIDR, HWCAPs, cores, cgroup limits, kernel and log stream per environment; no run recorded them |
| Files | `baseline/hostinfo.py` (new), `baseline/lkai_entry.py`, `baseline/container_build.sh` (packages and import-checks it), `baseline/buildcache.py` (`hostinfo.py` keys the zip stage) |
| Pinned baseline | `baseline/results/20261020T080835Z/` is unchanged. A new build produces a new `function.zip` (new CodeSha256), so a run with the new harness is a new run, not the pinned baseline |
| Cost | `snapshot()` reads a few `/proc` and `/sys` files once per environment; `probe_s` records its time |

Nothing has run on Lambda with the new harness yet: this card grants no spend.

## Reproduce

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/cb67-host-variability \
  feat/lambda-kleidiai-baseline lkai/d654-s3-express lkai/7a13-onezone-ia lkai/cab0-ranged-get \
  lkai/7cb8-stream-load lkai/3259-mmap-format lkai/fada-hotspot lkai/bc9a-profiler
R=examples/lambda-kleidiai/hostvar/results/<run-id>
python examples/lambda-kleidiai/hostvar/hostvar.py extract --out $R      # reads the commits in sources.json
git show 89a0df4bb677eb1ab8f993046328230fdb189e38:examples/lambda-kleidiai/profiling/results/offline/offline-summary.json \
  > $R/offline-summary.json
python examples/lambda-kleidiai/hostvar/hostvar.py analyze $R --fixed-host $R/offline-summary.json \
  --experiment-commit <commit>
python examples/lambda-kleidiai/hostvar/hostvar.py tables $R/results.json   # the section above
```

`extract` needs every source commit in the local object store (the `git fetch` above). `analyze` and `tables` read
only the run directory. Publication: `results/publish_result.py` from P2 (`feat/lambda-kleidiai-results`).
