# E6: profile-led in-Lambda hot spots, and the top one measured (complete Python bytecode)

E6 started from P3's corrected profile (`profiling/README.md`, run `20261020T231634Z`; the earlier
`20261020T124451Z` run is superseded). It proposed one-change experiments for
the hot spots that no other card owns, and it measured the top one on Lambda. The other cards already own:
- the tokenizer: T0 to T3
- the KleidiAI repack: E7, E8, E9
- the KV and recurrent cache clears: E10, E12
- memfd pages: E11
- per-op and concat: E14, E15
- the OpenMP wait policy: E13
- the restore stall: I1

That left the cold `imports` phase and the allocator for E6.

**Result (run `20261021T045838Z`, status `no-change`).**
- Complete unchecked-hash bytecode halves the cold `imports` phase: 0.719 to 0.346 s on CPU part 0xd40, Δ −0.373 s
  [−0.380, −0.367]. Init Duration falls by 0.380 s.
- The same direction appears in every shared round/CPU-part cell. The 0xd0c and 0xd4f cells remain sparse.
- Answers are bit-identical.
- The cold first request does not get faster. The on-demand load runs in PR #77's background thread, and the
  request waits for the 25 s S3 download either way. Starting the download 0.37 s earlier only means waiting longer
  for it to finish.
- Restore and warm latency are unchanged.
- Adopt it as hygiene: it costs nothing. It pays off once imports are on the critical path, which happens after an
  experiment that overlaps or removes the download.

## Hot spots and the ranked list

| # | Hot spot (P3 evidence) | One-change experiment | Phase | Expected | Disposition |
|---|---|---|---|---|---|
| 1 | `imports` 0.715 s p50 cold (P1). cProfile: `builtins.compile` 22.6 s own-time over 68 inits, ≈0.33 s per init. py-spy init leaves in `importlib._compile_bytecode` / `get_data` | Ship complete, valid bytecode (the only variable) | cold Init / `imports`; SnapStart snapshot init | −0.2 to −0.4 s on `imports` | **Run here.** Measured −0.373 s on imports, null on the first request |
| 2 | Residual imports after #1: numpy (86 modules), pydantic (38), tokenizers (14). Offline 0.12 s | Lazy-import what is not needed before the load starts | cold `imports` | < 50 ms | Filed for Chief as a separate card (operator choice), measured after #1 |
| 3 | Allocator: `malloc` 0.9 % of init CPU samples (811 / 106,648); malloc and free absent from the warm top list (26,599 samples) | jemalloc / tcmalloc / glibc tunables | — | none | **Not proposed: no profile support.** A null finding |

The tokenizer is in T0's scope. P3 attributes 2.3 % of init CPU samples to `tokenizers.abi3.so`, and E6 did not
measure it again.

## Root cause

`baseline/container_build.sh` takes four steps:
1. It installs the wheels with `pip --no-compile`.
2. It deletes every `__pycache__`.
3. It runs an import check, which writes 149 `.pyc` stamped with the build time.
4. It touches every file to 2026-01-01.

The source-mtime header in each of those 149 `.pyc` therefore no longer matches its `.py`, and Python treats all of
them as stale. 886 of the 1,035 `.py` have no `.pyc` at all. `/var/task` is read-only, so a cold start compiles
every imported module from source, every time. `e6_entry.py` confirms this inside Lambda, in every environment:
- **ctl**: 153 to 157 task-root modules loaded, 0 from usable bytecode.
- **pyc**: 152 to 156 of them from bytecode. The one compiled from source is `e6_entry.py` itself, which has no
  bytecode in either zip.

## The one variable

`pyc_zip.py` builds both zips from P1's `function.zip` `d148d706…`, the zip P1's published baseline is built from:
- `ctl.zip` is P1's zip, every entry byte for byte with its metadata, plus `e6_entry.py`.
- `pyc.zip` is `ctl.zip`, plus `py_compile` `UNCHECKED_HASH` bytecode for all 1,035 `.py`. This bytecode replaces
  P1's 149 stale `.pyc`.

The tests pin that every non-`.pyc` entry is identical between the two zips. The compiler is CPython 3.12.13; Lambda
reports 3.12.14. Both have bytecode magic `cb0d0d0a`. Lambda's 3.12 loads unchecked-hash bytecode without reading
the source.

Both functions (`lkai-e6-ctl`, `lkai-e6-pyc`) use P1's configuration:
- python3.12, arm64, 10,240 MB, SnapStart on published versions
- P1's environment, pinned runtime ARN and weights object

The zips sit under `e6/` in P1's bucket. The role may read only the weights key.

## Measurements

Run `results/20261021T045838Z/`:
- Cold: 6 rounds × 10 on each arm, followed by 5 warm passes (one per fixed request). Each arm is one sequential
  wave; the first arm alternates every round. This is alternation, not within-round interleaving.
- Restore: 3 rounds × 10 per arm, followed by one warm pass.
- 840 invocations. 0 errors. 1 class mismatch (a ctl restore invoke landed warm), excluded as P1's harness does.
- Cost: $1.73 at list prices (the operator's ceiling was $20). The SnapStart cache 3-hour minimum is most of it.

| Phase (CPU part 0xd40) | ctl p50 | pyc p50 | Δ p50, 95 % CI | n |
|---|---|---|---|---|
| cold `imports` | 0.719 s | 0.346 s | **−0.373 [−0.380, −0.367]** | 49 / 52 |
| cold Init Duration (REPORT) | 0.814 s | 0.434 s | **−0.380 [−0.393, −0.376]** | 49 / 52 |
| cold `weight_download` | 25.02 s | 25.32 s | +0.30 [+0.26, +0.38] | 49 / 52 |
| cold `import_total` + `weight_download` | 25.75 s | 25.68 s | (the download ends at the same time) | 49 / 52 |
| cold `model_load_mmap` | 8.37 s | 8.38 s | +0.005 [−0.035, +0.033] | 49 / 52 |
| **cold first request total** | 36.01 s | 36.00 s | −0.012 [−0.098, +0.079] | 49 / 52 |
| cold max memory | 5,159 MB | 5,150 MB | −9 [−10, −6] MB | 49 / 52 |
| SnapStart snapshot init `imports` (publish), see below | — | — | — | — |
| **restore first request total** | 3.94 s | 3.90 s | −0.045 [−0.097, +0.014] | 27 / 30 |
| **warm inference** | 0.605 s | 0.613 s | +0.008 [−0.010, +0.025] | 272 / 290 |

SnapStart snapshot inits ran on two CPU parts, so they get their own per-part table. Each `before_snapshot` record
is bound to the CPU part named in its own log stream (`results.json` → `host_variability.snapshot_init`). Only the
versions this run published count. All 12 records (6 per arm) are linked: 0 missing, 0 ambiguous, 0 excluded.

| Snapshot init `imports` (publish) | ctl median | pyc median | Δ median | ctl spread across versions | n |
|---|---|---|---|---|---|
| 0xd40 | 0.660 s | 0.326 s | −0.334 s | 0.005 s | 3 / 3 |
| 0xd0c | 0.838 s | 0.408 s | −0.429 s | 0.022 s | 3 / 3 |

Both deltas exceed the control spread (`negative-beyond-control-variation`), so the snapshot-phase outcome is
comparable across parts. `unassessed_conflicts` is empty; if a sparse known part had the opposite observed sign,
the outcome would be `host-conditioned`, not comparable. The earlier pooled figure (0.665 → 0.332 s) mixed both
parts; P1's `phases.<arm>.restore.snapshot_init` block is still that pooled distribution and is not the
host-attributed result.

- The 0xd0c and 0xd4f parts show the same `imports` halving: −0.479 s (n = 5 / 3) and −0.291 s (n = 6 / 5).
- On 0xd0c the cold first request is −0.69 s [−1.17, −0.32], but the sample is small, and on 0xd4f it is +0.05 s.
  That does not add up to a consistent first-request gain.
- Every aggregate delta is matched by CPU part. The CIs bootstrap over environments. For the targeted `imports`
  phase, all 6 shared 0xd40 round/part cells are below zero. Sparse 0xd0c and 0xd4f rows form 6 more shared cells
  (2 and 4), also below zero, plus 5 unmatched cells that are listed but not compared in `host_variability`.

Why the S3 download looks slower on pyc:
- The wrapper's bytecode check adds 9 ms (pyc) to 12 ms (ctl) after P1's timers have stopped, so it does not explain
  the gap.
- `weight_download` is longer on pyc because it starts earlier and ends at the same time: `import_total +
  weight_download` is the same on both arms.
- An environment's S3 bandwidth is the binding constraint, which E3 also concluded. This is an inference from the
  timings. E6 did not test it separately.

## Correctness

- 420 of 420 pyc answers are bit-identical to ctl's for the same request and CPU part in this run.
- All 796 answers that P1's `reference/outputs.json` covers are bit-identical to it.
- The reference has no 0xd4f entry for the four warm-only requests, so 44 answers were left to the run's own ctl
  comparison.

## Host variability

`e6_entry.py` records, per environment:
- `/proc/cpuinfo` implementer, part, variant, revision and Features
- `AT_HWCAP` and `AT_HWCAP2`
- the CPU count and `sched_getaffinity`
- the kernel, the Python version and the log stream

The cgroup v2 files are not exposed in the sandbox, so they are recorded as `unknown`.

Three CPU identities appeared on both arms:

| CPU part | Variant | HWCAP | HWCAP2 | ctl envs | pyc envs |
|---|---|---|---|---|---|
| 0xd40 | 0x1 | 0xfffffff | 0x1f201 | 76 | 82 |
| 0xd0c | 0x3 | 0x119fff | 0x0 | 7 | 3 |
| 0xd4f | 0x0 | 0x2ff3ffff | 0x3f3bf | 6 | 5 |

All environments reported 6 CPUs, kernel 5.10.255 and Python 3.12.14. Aggregate comparisons are made within one
part. `results.json` → `host_variability` records every primary metric per shared `(round, CPU part)` cell,
including per-arm counts, medians, ranges, delta, ratio and sign. It compares `|delta|` with the control
observations' within-cell range and persists that cell's outcome (`within-control-variation`,
`negative-beyond-control-variation`, `positive-beyond-control-variation`, or `unassessed` with fewer than two
control observations). Each CPU part's outcome is aggregated from those stored cell outcomes, and `cell_outcomes`
counts them, so a mixed part shows which cells drove it. On 0xd40, all 6 cold-import cells are
`negative-beyond-control-variation`. The 0xd0c/0xd4f cells have one control observation each, so they are stored as
`unassessed` supporting rows; their observed deltas have the same sign. `unassessed_conflicts` is therefore empty;
an opposite observed sign on either sparse part would block comparability as `host-conditioned`. Unmatched cells
are also explicit. No Graviton generation is claimed:
the parts and features are reported as read.

The change removes CPU work (bytecode compilation), so a fixed-host run would only repeat the offline proxy below. The
H1 box was not needed.

Offline proxy (NOT Lambda): Intel Xeon 6975P-C, read-only trees, 30 interleaved pairs. `imports` went from 0.326 to
0.121 s, Δ p50 −0.203 s. All 30 pairs were below 0.

## Reproduce

See `meta.json` → `reproduce`. `pyc_zip.py` needs CPython 3.12 and needs no arm64 host.

## Limits

- The run covers one region (us-east-2) and one memory size (10,240 MB).
- The restore leg ran only on 0xd40.
- Arms were sequential waves, alternating first position by round. This controls round-level drift and CPU identity,
  but not time-within-round drift.
- The 0xd0c and 0xd4f cells are small (3 to 7 environments), and 5 cold `imports` round/part cells lack one arm.
- Lambda's Python 3.12.14 differs from the compiler's 3.12.13 in patch level only (same magic number). Lambda used
  the bytecode, as the in-function check shows.
- The SnapStart publish-time saving (the snapshot init's `imports`) is not on any request path.
- For snapshot inits only the CPU part is observable, from the engine's `init cpu_part=` log line in the same
  stream. Variant, revision, HWCAP, features, CPU count and kernel come from `e6_entry`'s first-invocation record,
  which a snapshot init does not emit, so they are unknown there. Lambda chose the parts; there is one init per
  (version, part, arm), so the control spread is across the 3 published versions, not within one.
