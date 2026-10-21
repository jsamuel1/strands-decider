# lambda-kleidiai E11: release the weight memfd's repacked pages after load

One variable against the [P1 baseline](../baseline/README.md) (pinned at `70eab7b`): whether llama.cpp releases the
pages of the weight memfd that no tensor reads any more, after loading and before the SnapStart snapshot. Result:
[`results/20261021T103834Z/results.json`](results/20261021T103834Z/results.json) (`lkai-result/1`, status
`improved`), published to the [P2 dashboard](https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results)
as `E11-release-memfd/20261021T103834Z`.

**Result.** Releasing 1.53 GB of the 2.08 GB memfd cuts REPORT Restore Duration by 0.72 s (−22%) and the restored
first request by 0.73 s (−18%) on CPU part `0xd40`, with 95% CIs well below zero. Every shared round/CPU-part cell
moves the same way. Max Memory Used falls about 1,005 MB in every class. All 1,200 answers are bit-identical to P1's
reference. The cold first request and warm inference show no change beyond noise.

## Premise check

The card asked first whether the memfd is a large share of resident memory after load. Offline, on the arm64 DevSpace
(CPU part `0xd40`), with P1's `function.zip` (`26eb2651…`) and GGUF (`8dbb8872…`), `premise_probe.py` loads the model
the way the Lambda does (GGUF copied into a memfd, `decshim` mmapping `/proc/self/fd/N`):

| After load + warm-up | P1 behaviour (`off`) | E11 (`rel`) |
|---|---|---|
| VmRSS | 4,896 MB | 3,472 MB |
| RssAnon (llama's repacked weights, KV cache, Python) | 2,911 MB | 2,911 MB |
| RssShmem (mapped memfd pages) | 1,954 MB | 530 MB |
| memfd resident (mincore over the whole file) | 1,980.5 MiB, all of it | 517.7 MiB |

The memfd is 40% of resident memory, and every byte of it stays resident. That is the premise, so the experiment
went ahead. Probe outputs: `results/offline/premise-p1zip.json` (P1's zip), and `results/offline/v3-*.json` (the
deployable E11 zip, `off`/`rel`, with and without `lambda_like_seccomp.json`).

Why llama.cpp keeps it: in `f74e2fa6a`, `load_all_data` maps the file, aliases the tensors that stay in the mmap CPU
buffer (`token_embd` and the F32 norms), and copies everything else into other buffers. Every Q8_0 matmul weight is
copied into `CPU_KLEIDIAI` and repacked. At the end, llama.cpp munmaps the mapping outside `[first, last)` of the
aliased tensors (`llama-model-loader.cpp`, final cleanup). For a file on disk that frees page cache. For a memfd it
frees nothing, because shmem pages belong to the file, not to the mapping, and the PR #77 loader keeps the memfd open
for the life of the process.

## The change

`patch_memfdrel.py` is applied on top of P1's patched llama.cpp. P1's `llama/patch_llama.py` and build stage keys are
unchanged. The patch is dormant unless `LKAI_RELEASE_MEMFD=1`, so one zip serves both arms.

1. In `load_all_data`, it records the `[offs, offs + nbytes)` range of every tensor aliased into the mapping.
2. At the final cleanup, after llama.cpp's own unmap, it computes per file the page-aligned complement of those
   ranges. A page shared by an aliased tensor and a released neighbour is kept.
3. For each released range, it first munmaps llama's view, so a stray read faults instead of reading zeros. It then
   frees the pages through a read-write reopen of `/proc/self/fd/<fd>`: with `fallocate(PUNCH_HOLE | KEEP_SIZE)`, or,
   where the sandbox refuses that, with `madvise(MADV_REMOVE)` on a temporary writable shared view, unmapped at once.
4. It exports `lkai_memfd_release_stats`: released bytes, kept bytes, microseconds, the first errno, ranges, files,
   and the ranges released by the `MADV_REMOVE` fallback.

**Lambda refuses `fallocate` and `mincore`.** The first deployed build released nothing (`errno 1`). `syscall_probe.py`
(a 1 GB function, deleted afterwards; output in `results/syscall-probe/`) shows that in the sandbox `fallocate`
returns EPERM, `madvise(MADV_REMOVE)` and `ftruncate` work, and `mincore` returns EPERM. The fallback was added in
`e4325aa`. `lambda_like_seccomp.json` reproduces both EPERMs under Docker, and the offline runs below use it. Because
`mincore` is refused and the released memfd is no longer mapped, `RssShmem` cannot see it. The function therefore
records the sandbox's own `/proc/meminfo` `Shmem`, which counts the memfd's pages whether they are mapped or not.

On every `rel` environment of the run (errno 0), it released 1,533,873,632 B in 74 ranges, all through
`MADV_REMOVE`, and kept 542,838,784 B (26% of the file). The release itself took 0.178 s p50.

## Correctness

- **Offline**, on the DevSpace, `0xd40`: 7 loads of the deployable zip, including `off`, `rel`, and `rel` and `off`
  under `lambda_like_seccomp.json`. Each answered all five fixed requests bit-identically to P1's
  `reference/outputs.json`.
- **Lambda**, 1,200 invocations: every answer was bit-identical to P1's reference for the same request and CPU part,
  with 0 choice flips and a maximum delta of 0.0. All 600 `rel` answers were identical to `off`'s from the same run.
  Cold, restore and warm all ran the full request set (warm passes cycle all five), so the released ranges were
  exercised after a restore as well as after a cold load.

## The benchmark

`e11deploy.py up` creates `lkai-f989-off` and `lkai-f989-rel`. They share one zip (`7aaa1ff0…`, built at `e4325aa`
from P1's llama.cpp tree plus the patch, with the manifest's dirty list empty). Both copy P1's configuration from the
deployed `lkai-p1-inst`: python3.12, arm64, 10,240 MB, SnapStart on published versions, P1's pinned runtime version
ARN and P1's weights object. They are instrumented (`LKAI_PHASES=1`) and differ only in `LKAI_RELEASE_MEMFD`. The
zip sits under P1's bucket at `e11/`, and the role reads only the weights key.

`e11bench.py` runs P1's lifecycle per round (new `COLD_TAG`, cold, publish and wait for the snapshot, restore, 10 warm
passes, delete versions). Unlike P1 and E6, it submits both arms' waves **together**, interleaved request by request
through one fresh Lambda client per wave. The arm order inside a wave flips every round. Each start class of the two
arms is therefore sampled over the same seconds, not in sequential arm waves. Read timeout is 300 s; there were
0 client errors.

Run `20261021T103834Z`: harness `7d08acc`, zip `e4325aa`, baseline `70eab7b`. us-east-2, 2026-10-21 10:38–11:20Z.
5 rounds, concurrency 10, 1,200 invocations, 0 errors, 0 class mismatches.

## Results: rel minus off, p50, matched by CPU part

| Class | Metric | `0xd40` off → rel | Δ (95% CI) | `0xd0c` Δ |
|---|---|---|---|---|
| restore | **REPORT Restore Duration** | 3.259 → 2.542 s | **−0.717 s (−22.0%)** [−0.759, −0.657] | −0.921 s (n 3/5) |
| restore | **first request (Restore + Duration)** | 3.971 → 3.236 s | **−0.734 s (−18.5%)** [−0.783, −0.653] | −0.885 s |
| restore | first request inference | 0.672 → 0.679 s | +0.007 s [−0.011, +0.023] | +0.033 s |
| restore | Max Memory Used | 5,148 → 4,143 MB | −1,005 MB [−1,005, −1,005] | −1,007 MB |
| cold | first request (Init + Duration) | 36.06 → 35.84 s | −0.221 s (−0.6%) [−0.313, −0.158] | −0.644 s |
| cold | `model_load_mmap` | 8.364 → 8.545 s | +0.181 s [+0.148, +0.209] | −0.009 s |
| cold | `context_init` | 0.323 → 0.044 s | −0.280 s | |
| cold | Max Memory Used | 5,159 → 4,153 MB | −1,006 MB | −986 MB |
| warm | steady-state inference | 0.620 → 0.644 s | +0.024 s [−0.070, +0.082] | −0.038 s |
| warm | Max Memory Used | 5,164 → 4,143 MB | −1,021 MB | −1,031 MB |

n on `0xd40`: restore 47/45, cold 42/42, warm 470/450. The 3 cold environments per arm on part `0xd4f` are in the
result file, not here.

- **Restore.** The gain is in Lambda's own restore phase. The first forward after a restore is unchanged, so the
  smaller snapshot restores faster and runs no slower. Restore Duration p90 on `0xd40` is 6.02 → 2.75 s. p99 is
  9.31 → 9.44 s: each arm has a 9 s restore in round 0. One configuration gives one point (about 0.5 s per GiB
  released). It is not a measured scaling law.
- **Memory.** Before the snapshot, the sandbox's `Shmem` falls from 1,983 to 520 MiB and VmRSS from 4,958 to 3,566 MiB
  (n 20 snapshot inits per arm). REPORT Max Memory Used falls about 1,005 MB. That is less than the 1,463 MiB `Shmem`
  drop, and this run does not establish how Lambda computes Max Memory Used.
- **Cold.** The first request is 0.22 s faster on `0xd40`, but that is inside `off`'s within-cell range in every
  assessed round, so no effect is claimed. Its phases move in opposite directions. `model_load_mmap` is +0.18 s, the
  release's own cost (0.178 s in-loader). `context_init` is −0.28 s (0.323 → 0.044 s); its cause was not tested.
- **Warm.** No change: 4 of 8 shared cells per sign, every cell inside `off`'s within-cell range.
- **Run-to-run variance** (CV of the per-round medians): restore first request 1.3% (off) and 4.5% (rel; one round's
  median is 3.55 s against 3.19–3.24 s for the rest). Cold 0.2% and 0.3%. Warm 7.0% and 4.0%.
- **Cost** of the run: $2.39, of which the SnapStart cache was $1.63 (10 versions at the 3-hour minimum). It also
  included one cold invoke per arm before the run and the syscall probe. All published versions were deleted; the
  two functions and the zip remain until teardown.

## Host variability

The assessment follows the initiative's HOST VARIABILITY rules (`results.json` → `host_variability`, `e11.host`).

**Observed per environment** (first invocation; 100 per arm). `/proc/cpuinfo` implementer, part, variant and
revision, the processor count, `AT_HWCAP` / `AT_HWCAP2`, `os.cpu_count`, `sched_getaffinity`, the kernel, the runtime
and the log stream. Three identities appeared, each on both arms with the same counts within a few:

| CPU part | implementer / variant / revision | HWCAP / HWCAP2 | CPUs | ggml variant | environments off / rel |
|---|---|---|---|---|---|
| `0xd40` | 0x41 / 0x1 / 1 | 0xfffffff / 0x1f201 | 6 | armv8.6_1 | 89 / 87 |
| `0xd0c` | 0x41 / 0x3 / 1 | 0x119fff / 0x0 | 6 | armv8.2_2 | 8 / 10 |
| `0xd4f` | 0x41 / 0x0 / 1 | 0x2ff3ffff / 0x3f3bf | 6 | armv8.6_2 | 3 / 3 |

Kernel `5.10.255-268-309.1092.amzn2.aarch64` everywhere. **Unknown**: cgroup `cpu.max`, `memory.max` and
`cpuset.cpus.effective` (not exposed), the AZ, the physical host and the snapshot-build host. No processor
generation is inferred from any of these fields.

**Method.** For each shared (round, CPU part) cell, the arm medians are compared. The pre-registered test (E6's)
counts a cell as an effect only when its delta exceeds `off`'s within-cell range.

- **Restore Duration.** All 8 shared cells are negative, −0.63 to −1.38 s (exact sign test over cells, p = 0.008).
  On `0xd40`, 2 of 5 cells pass the range test. In the other 3, one or two `off` restores of 6–9 s widen `off`'s range
  to 2.8–6.2 s, so the outcome is `mixed-or-overlapping-control-variation`. Against `off`'s interquartile range,
  a sensitivity test added after the run was seen, every `0xd40` cell is beyond it except rounds 0 and 1, where the
  IQR is 1.6–1.8 s. `0xd0c` cells hold one `off` sample each, so they are unassessed, but their deltas have the same
  sign.
- **Cold first request.** 8 of 9 cells are negative (p = 0.039). On `0xd40` they are inside `off`'s range (no
  effect claimed). On `0xd0c`, one assessed cell (n 3/1) is beyond it.
- **Warm.** Within range on both parts.

**Assessment.** The restore effect does not change sign across CPU part or round, and it is about five times the
restore per-round range of `off` (0.14 s). Host variation could change its size estimate, not its conclusion, so it
was not rerun. The tail outliers that defeat the strict range test are a stated limitation. The interleaved schedule
removes the time-within-round confound of sequential arm waves. H1's fixed-host EC2 baseline was not needed: the
offline DevSpace runs already isolate memory (identical RssAnon, 1.42 GB less RssShmem), and the restore effect exists
only on Lambda.

## Limitations

- Restore Duration is Lambda's own measurement, and SnapStart's snapshot format is not visible from inside the
  function. That the smaller restore comes from the smaller shmem footprint is inferred from one variable and
  unchanged everything else. It was not observed directly.
- The released ranges are those of this GGUF on this loader. E7 (skipping the tied output repack) and E9 (pre-packed
  weights) change what the memfd must keep, so R must re-measure E11 on top of them.
- `0xd0c` and `0xd4f` are sparse (3 to 10 environments per arm).
- The REPORT Max Memory Used drop (about 1,005 MB) is smaller than the `Shmem` drop (1,463 MiB). Why is not known.

## Reproduce

See `meta.json` → `reproduce`. In short: P1's `baseline/build.sh` on an arm64 host, then `e11/container_build.sh`
and `e11/manifest_container.sh` in P1's pinned Lambda image, `premise_probe.py` per arm offline, then `e11deploy.py
up`, `e11bench.py`, `e11deploy.py teardown`, `summarize_e11.py`, `../baseline/redact.py`, and P2's
`publish_result.py`. `results/<id>` regenerates `results.json` exactly with `summarize_e11.py` (checked).
