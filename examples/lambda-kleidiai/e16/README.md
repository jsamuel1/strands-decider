# E16: import the model stack inside the load, beside the download (lazy imports, cold Init)

**Result (run `20261115T215333Z`, status `no-change`).** Moving `engine` and what it imports (numpy, pydantic,
tokenizers, strands_decider) off the init's main thread and into the load:

- cuts cold `imports` from 0.367 to 0.004 s on CPU part 0xd40 (Δ −0.364 s, 95 % CI [−0.371, −0.356]);
- cuts REPORT Init Duration from 0.477 to 0.123 s (Δ −0.354 s [−0.367, −0.338]) and starts the S3 download 0.36 s
  earlier. The same direction appears on 0xd4f and 0xd0c.

This is about 7× the "< 50 ms" the card expected. On Lambda, importing the stack costs 0.28 to 0.56 s even with E6's
bytecode. On the offline x86 box it costs 0.12 s.

The cold first request does **not** get faster (0xd40: −0.023 s [−0.067, +0.044]). The download is still the binding
constraint, as E6 found. Starting it earlier only moves the import's cost onto the download: the import now runs
beside the download, competing with it, and the download ends at the same time. Restore and warm are unchanged.
Answers are bit-identical. The run made 1,080 invocations with 0 errors and cost $1.94.

## The one variable

Import placement. Both functions run the same code at the same bytecode mode, model, requests, memory, architecture,
region, SnapStart configuration and runtime. `lazy_zip.py` builds both zips from P1's `function.zip` `d148d706…`:

| | e6 (`lkai-e16-e6`, control) | lazy (`lkai-e16-lazy`) |
|---|---|---|
| `handler.py`, `app.py` | P1's, byte for byte: `engine` imported at module level, on the init's main thread, before `_load` starts | this branch's: `_load` imports `engine` in a thread started beside `loader.download_to_memfd`; request handling imports pydantic and the schema after the load; `app.py` imports `engine` only in the eval mode, after the load |
| bytecode | E6's: unchecked-hash `.pyc` for every `.py` (0 recompiled in every environment, both arms) | same |
| harness | `lkai_entry.py` (this branch), H1's `hostinfo.py`, `e16_entry.py` | same bytes |
| zip SHA-256 | `d75807e3…` | `b266cff4…` |

The tests (`tests/test_lkai_e16.py`) pin that the two zips differ in exactly `handler.py`, `app.py` and their two
`.pyc`, and that every other entry is P1's. `src/strands_decider` is unchanged. Its pydantic models are classes
defined at module import, and `engine` needs them, so they stay eager inside the deferred import.

`handler.py` builds the S3 client (`loader._client()`, boto3's import and setup) on the load thread before it
submits the engine import, so the import does not delay the first ranged GET.

### Harness change (both arms)

P1's `lkai_entry.py` imported `engine` itself, inside its `imports` timer, before `app`. That would have forced the
control's placement onto both arms. It now patches `engine` through a one-shot `sys.meta_path` hook when `engine` is
first imported, on whatever thread, so it times each deployment's own placement and leaves it alone. It adds three
fields. Every P1 phase name keeps its place in the order:

- `engine_import`: the time `engine`'s own module took to execute;
- `engine_import_thread`: the counter naming the thread that ran the import;
- `load_start`: seconds from the module clock's start to the load's first call into `loader`.

On demand, `imports` is the main thread's imports (`loader` + `app` + `handler`). On e6 that includes `engine`, as
P1's did. On a snapshot init only `loader` is timed before `app`, because there `app`'s import is the whole load
(`model_ready`). `engine_import` is reported on its own. P1's and E6's tests pass unchanged.

## Offline `-X importtime` (NOT Lambda)

This is a proxy: Intel Xeon 6975P-C, CPython 3.12.13, the deployed zips' own sources with this host's wheels
(numpy 2.5.3, pydantic 2.14.0, tokenizers 0.23.2), unchecked-hash bytecode. A stand-in `loader` blocks the
download, so `-X importtime` lists only the main thread's imports. Medians of 11 runs
(`results/…/offline/importtime_*.json`):

| package (self time on the main thread) | e6 | lazy |
|---|---|---|
| numpy (86 modules) | 29.3 ms | not imported |
| pydantic (38) | 24.1 ms | not imported |
| pydantic_core (3) | 8.1 ms | not imported |
| strands_decider (4) | 9.6 ms | not imported |
| tokenizers (14) | 3.5 ms | not imported |
| **main-thread import total** (341 / 81 modules) | **136.5 ms** | **21.7 ms** |

30 interleaved pairs with a 1 s stand-in download (`offline/pairs.json`): the main thread's imports and the download
start both fall from 0.130 to 0.017 s (paired Δ −0.112 s, 30/30 pairs below 0). In both arms `engine` was imported
before the download ended (30/30).

## Measurements

Run `results/20261115T215333Z/`, us-east-2, 10,240 MB, arm64, python3.12 (P1's runtime ARN), SnapStart on
published versions.

- **Interleaved:** every wave runs both arms at once. Invokes are submitted alternately into one pool, with the
  first arm alternating by round, and each arm gets its own fresh invoke client (I1).
- Cold: 8 rounds × 10 per arm, then 5 warm passes (one per fixed request).
- Restore: 3 rounds × 10 per arm, then one warm pass.
- 1,080 invocations (160 cold, 60 restore, 860 warm). 0 errors. 0 class mismatches.
- Cost: $1.94 at list prices, mostly the SnapStart cache minimum and cold Duration (ceiling $10).
- Only phase timers and REPORT are used. No profiler ran.

Deltas are lazy minus e6. Medians are taken within one CPU part. CIs bootstrap over environments.

| cold, by CPU part | 0xd40 (n 51/44) | 0xd4f (n 26/30) | 0xd0c (n 3/6) |
|---|---|---|---|
| `imports` (main thread) | 0.367 → 0.004 s, **−0.364 [−0.371, −0.356]** | 0.284 → 0.003, **−0.281 [−0.284, −0.275]** | 0.565 → 0.007, −0.558 |
| REPORT Init Duration | 0.477 → 0.123 s, **−0.354 [−0.367, −0.338]** | 0.370 → 0.086, **−0.283 [−0.303, −0.274]** | 0.832 → 0.272, −0.559 |
| `load_start` (download start) | 0.369 → 0.006 s, −0.363 [−0.370, −0.356] | 0.285 → 0.004, −0.281 | 0.569 → 0.011, −0.558 |
| `engine_import` (its own thread on lazy) | 0.364 → 0.736 s, +0.372 [+0.365, +0.387] | 0.281 → 0.534, +0.253 | 0.561 → 1.041, +0.480 |
| download end, from clock start | 25.804 → 25.799 s, −0.004 [−0.027, +0.018] | 25.828 → 25.816, −0.012 [−0.047, +0.032] | 26.36 → 25.96, −0.41 |
| model ready, from clock start | 35.786 → 35.766 s, −0.021 [−0.070, +0.040] | 34.264 → 34.229, −0.035 [−0.099, +0.025] | 48.66 → 48.03, −0.63 |
| `model_load_mmap` | 8.377 → 8.401 s, +0.024 [−0.016, +0.050] | 7.139 → 7.112, −0.027 | 19.92 → 19.73 |
| **first request total** (Init + Duration) | 36.141 → 36.118 s, **−0.023 [−0.067, +0.044]** | 34.568 → 34.513, **−0.055 [−0.123, +0.035]** | 49.36 → 48.70, −0.66 [−0.84, −0.37] |
| Billed Duration (p50) | 36.142 → 36.119 s | | |
| Max Memory Used | 5,132 → 5,133 MB | 5,134 → 5,137 | 5,130 → 5,132 |

| restore and warm | e6 p50 | lazy p50 | Δ, 95 % CI | n |
|---|---|---|---|---|
| restore first request total (0xd40) | 3.284 s | 3.360 s | +0.076 [−0.011, +0.143] | 29 / 21 |
| REPORT Restore Duration (0xd40) | 2.599 s | 2.647 s | +0.048 [−0.006, +0.111] | 29 / 21 |
| warm inference (0xd40) | 0.609 s | 0.614 s | +0.005 [−0.020, +0.021] | 284 / 241 |
| warm inference (0xd4f) | 0.530 s | 0.540 s | +0.010 [−0.024, +0.028] | 130 / 150 |

**Detection floor.** On 0xd40, the 95 % CI half-width for the Init Duration median difference was 14 ms (n 51/44).
For the cold first request it was about 55 ms. Effects smaller than that on the first request are not ruled out.

### Why the first request does not move

`download_completion` in `results.json` measures from one clock start on both arms, because `weight_download` alone
is not comparable: lazy's download starts earlier.

- e6 imports first (0.37 s on the main thread) and then downloads.
- lazy starts the download at 6 ms. Its load then waits 0.45 s past the download's end for the engine import. That
  import took 0.74 s on its own thread, against 0.36 s on the main thread, because the two threads share the GIL and
  six cores.
- The download ends at the same moment on both arms (Δ −4 ms on 0xd40), and so does model ready.

The Init that lazy saves is spent waiting for the load inside Duration, so billed cold time does not change either.
This is the E6 picture again. Import work leaves the first request only when it can finish inside a download that is
still running, and here the download hides it on both arms.

### SnapStart snapshot init (publish time, not on a request path)

Each `before_snapshot` record is bound to its own log stream's CPU part. All 12 records (6 per arm) are linked: 0
missing, 0 excluded. On 0xd40 the load's median finish was 0.37 s sooner (`model_ready` 34.97 → 34.61 s, 3 versions
per arm, a median delta beyond the control spread of 0.05 s). One lazy version (35.22 s) was slower than every e6
version, so this is a median shift, not a per-version one. On 0xd0c the median was 0.20 s sooner (46.84 → 46.64 s,
control spread 0.13 s; one lazy version, 46.74 s, was slower than the fastest e6 version). Only publish time changes.
The restore leg above shows no restore benefit.

**Correction.** The harness deployed in this run (`lkai_entry.py` at `acc1b94`) added the snapshot init's
`model_ready` span twice, so every recorded snapshot-init `model_ready` is 2× the span. It is fixed on this branch,
with a regression test that runs the SnapStart path. `summarize_e16.py` halves the recorded value in every
snapshot-init record (12 `before_snapshot` records, 60 restored-environment records) before any summary reads it,
when the run's harness hash is the one that doubled it, and `results.json` → `corrections` records that. The two
adds read the clock microseconds apart, so half is the span to about 1 µs. The raw logs and invocations keep the
recorded values. No cold, restore or warm REPORT figure, and no other phase, depended on this field.

## Correctness

- 540 of 540 lazy answers are bit-identical to e6's for the same request and CPU part in this run.
- All 856 answers that P1's `reference/outputs.json` covers are bit-identical to it (0 choice flips).
- The other 224 are on CPU parts or requests the reference does not cover. They are covered by the in-run comparison.

## Host variability

`hostinfo.py` (H1) runs inside the function at init and again after a restore. Three identities appeared, each the
same on both arms:

| fingerprint | CPU part | variant | HWCAP / HWCAP2 | CPUs | kernel | e6 envs (cold/restore) | lazy envs |
|---|---|---|---|---|---|---|---|
| `1f5c7b54…` | 0xd40 | 0x1 | 0xfffffff / 0x1f201 | 6 | 5.10.255 | 51 / 29 | 44 / 21 |
| `5681f533…` | 0xd4f | 0x0 | 0x2ff3ffff / 0x3f3bf | 6 | 5.10.255 | 26 / 0 | 30 / 0 |
| `8e9db0ba…` | 0xd0c | 0x3 | 0x119fff / 0x0 | 6 | 5.10.255 | 3 / 1 | 6 / 9 |

- cgroup limits, the availability zone and the physical host are not exposed. They are recorded as unknown.
- No processor generation is inferred from a part.
- Every comparison is made within one part.
- 0xd0c is sparse: 3 e6 cold environments and 1 e6 restore environment. Its cells are reported but unassessed.
- 0xd4f had no restores, so its restore result is unassessed.
- `host_variability` persists every shared (round, part) cell. For cold `imports`, all 8 cells on 0xd40 and all 7
  on 0xd4f are `negative-beyond-control-variation`. No sparse cell has the opposite sign.
- The cold first request is `within-control-variation` on 0xd40 (8/8 cells). On 0xd4f it is mixed (2 negative,
  5 within).

## Recommendation

Adopt as a candidate ladder rung for R, S and G. It costs nothing and is a measured 0.28 to 0.56 s cut in Init
Duration and in the time to download start, with identical answers. On the current baseline it does not shorten the
first request, the restore or warm inference. It can pay off only on a rung where the download is no longer the
critical path (E4 streaming, a faster fetch, or prepacked or smaller weights). Re-measure it on that rung, because
there the off-thread import competes with the load itself.

## Reproduce

See `meta.json` → `reproduce`. `lazy_zip.py` needs CPython 3.12 and no arm64 host.

## Limits

- One region (us-east-2) and one memory size (10,240 MB).
- The restore leg ran on 0xd40 (29/21 environments) and 0xd0c (1/9) only.
- 0xd0c samples are small.
- Arms are interleaved inside each wave. The first-submitted arm alternates by round, which balances any
  submission-order effect but does not remove it.
- The `imports` timer has a different scope on the two arms by design: it is the main thread's imports, and the
  variable is what runs there. `engine_import` and REPORT Init Duration are the arm-independent measures.
- Lambda's 3.12.14 vs the compiler's 3.12.13: same bytecode magic, and 0 modules recompiled from source in every
  environment.
