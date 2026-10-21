# E12: clear only the cache metadata between requests (restored first request, warm)

P3's profile (`profiling/README.md`, "Baseline hot spots by phase") found that half of the first forward after a
SnapStart restore is `memset`, called from `llama_kv_cache::clear` and `llama_memory_recurrent::clear`. The first
request after a restore took 0.68 s against 0.24 s warm. E12 owns that hot spot.

**Result (run `20261021T105615Z`, status `improved`).**
- The first request after a restore gets 0.29 s faster: REPORT `Duration` 0.687 → 0.394 s on CPU part 0xd40
  (−42.6 %, 95 % CI of the delta [−0.301, −0.282] s, n = 54 / 51), and 1.064 → 0.678 s on 0xd0c (−36.3 %, n = 6 / 9).
  In every (round, CPU part) cell, every e12 row is faster than every dor row.
- Restore first request total (`Restore Duration` + `Duration`): 3.956 → 3.651 s on 0xd40 (−7.7 %). Lambda's own
  `Restore Duration`, which runs before function code, does not move (+0.005 s, CI [−0.046, +0.032]).
- Warm: −1.6 % on 0xd40, but the CI includes 0 (paired median ratio 0.984, CI [0.969, 1.002]). The ~2 % warm
  hypothesis is not established. See "Warm" below for why the effect differs by request.
- Cold first request: unchanged; the 25 s S3 download still binds.
- Answers: bit-identical to P1's reference and to the control arm. Offline on arm64, hidden states are byte-identical
  in every order tried. 0 errors, 0 class mismatches, 0 `client:ReadTimeoutError` in 1,320 invocations.
- Cost of the run: $2.99 at list prices (ceiling $10). Most of it is the SnapStart cache 3-hour minimum.

**Scope: Lambda only.** The result applies to Lambda with SnapStart, not to SageMaker-hosted models (operator scope
note, 2026-10-21). The saving is the first touch of cache memory restored from a SnapStart snapshot. The warm effect
on its own is not established. No SageMaker deployment was measured here, and none of these numbers should be used
for one.

## What the clear does, and why the zeroing is not needed

`shim/decshim.c` calls `llama_memory_clear(mem, true)` before every request (`dec_eval` and `dec_eval_fork`). At
llama.cpp `f74e2fa6a` (the hybrid memory for qwen35 calls both halves):
- `llama_kv_cache::clear(data)` resets the cell metadata (`v_cells[s].reset()`, heads to 0) and, with `data`, calls
  `ggml_backend_buffer_clear(buf, 0)` on every K/V buffer.
- `llama_memory_recurrent::clear(data)` resets every cell (`pos`, `seq_id`, `src`, `tail`), `head`, `used` and
  `rs_idx`, and, with `data`, zeroes every recurrent (conv `r` and SSM `s`) buffer.

The buffer zeroing is not needed for correct results:
- **Attention K/V.** `set_input_kq_mask` gives every cell that holds no position of the token's sequence the value
  `-INFINITY`, so a stale K row has softmax weight exactly 0. A stale V row is multiplied by that 0. The cells a request
  uses are written by the request before it reads them. The shim runs with flash attention off and `DEC_KV=f32`.
- **Recurrent state.** In `find_slot`, a sequence without a tail cell gets a fresh cell with `src < 0`, whose `src0`
  is set to `rs_z`. `build_rs` then zeroes that one state row in the graph (`ggml_scale_inplace(state_zero, 0)`, which
  writes `x * 0`, and a stale NaN/Inf would survive that: see "Limits") and copies it into every fresh cell. The
  conv and SSM states of a new sequence therefore start at zero whatever the buffer held.
- `n_rs_seq` is 0 in the shim (the default), so no rollback snapshot rows are read.

**The one variable:** `DEC_CLEAR_DATA`. The shim reads it once at load. Unset, or anything but exactly `"0"`, keeps
P1's `clear(true)` (the dormant switch). `"0"` calls `llama_memory_clear(mem, false)`. `dec_clear_data()` reports
the mode, and the Lambda wrapper records it from inside every environment.

## Deployment

- One zip for both arms (`build_zip.py`): P1's `function.zip` `26eb2651…` (P1's published run), every entry copied
  byte for byte, except `lib/libdecshim.so` (E12's shim, `1793f24c…`, built by `build_shim.sh` with P1's command in
  P1's pinned build image against P1's own llama.cpp build) and an added `e12_entry.py`. 1,898 entries are unchanged.
- `lkai-e12-dor`: `DEC_CLEAR_DATA` unset. `lkai-e12-e12`: `DEC_CLEAR_DATA=0`. Nothing else differs (`e12bench.py`
  checks the deployed environments before it starts).
- P1's configuration: python3.12, arm64, 10,240 MB, SnapStart on published versions, P1's pinned runtime version ARN,
  P1's weights object (read-only; the role may `GetObject` only that key), `us-east-2`.
- `e12_entry.py` wraps P1's `lkai_entry.py` unchanged. On each environment's first invocation, after P1's handler has
  returned, it reads the host metadata (0.2 to 0.5 ms) and the shim's mode. It reads per environment, not at import,
  because a SnapStart environment imports on the snapshot host.

## Measurements

Run `results/20261021T105615Z/`: 6 rounds. In every round, both arms' cold waves (10 each) start at the same
moment, then a version of each arm is published, then both restore waves (10 each) start at the same moment, then 9
warm passes, each sending the same request to both arms at the same moment (P1's five requests, then E12's four
stress requests). The publish order alternates by round. This is within-round interleaving: both arms share the
round, the minute and the CPU mix. Each wave gets a new Lambda client (I1's fix).

| Phase | CPU part | dor p50 | e12 p50 | Δ p50, 95 % CI | n |
|---|---|---|---|---|---|
| restore `Duration` (first request) | 0xd40 | 0.687 s | 0.394 s | **−0.293 [−0.301, −0.282]** | 54 / 51 |
| restore `Duration` (first request) | 0xd0c | 1.064 s | 0.678 s | −0.386 [−0.436, −0.354] | 6 / 9 |
| restore first request total | 0xd40 | 3.956 s | 3.651 s | −0.305 [−0.345, −0.240] | 54 / 51 |
| restore `Restore Duration` (not targeted) | 0xd40 | 3.259 s | 3.264 s | +0.005 [−0.046, +0.032] | 54 / 51 |
| restore billed duration | 0xd40 | 0.700 s | 0.403 s | −0.297 [−0.305, −0.285] | 54 / 51 |
| warm inference | 0xd40 | 0.467 s | 0.460 s | −0.007 [−0.033, +0.011] | 486 / 459 |
| cold first request total | 0xd40 | 35.98 s | 35.88 s | −0.103 [−0.175, +0.002] | 50 / 54 |
| cold `warmup_inference` (first forward ever) | 0xd40 | 0.249 s | 0.236 s | −0.012 [−0.018, −0.009] | 50 / 54 |
| max memory used (cold / restore) | 0xd40 | 5,158 / 5,150 MB | 5,160 / 5,150 MB | +2.5 / 0 MB | |

Per-round restore `Duration` deltas (pooled parts): −0.304, −0.296, −0.281, −0.278, −0.275, −0.313 s.

### Warm

Paired by (round, request, CPU part), e12 / dor has median ratio 0.984 on 0xd40 (54 cells, CI [0.969, 1.002]). By
request it is not uniform:

| Request (input tokens) | e12 / dor, 0xd40 |
|---|---|
| short-1q (78), ticket-route-1q (83), short-1q-b (85), pair-2q (134), cracked-screen-2q (147) | 0.95 to 0.99 |
| payouts-3q (196), agent-action-guard-2q (306), incident-review-4q (598), wide-16q (2,818) | 1.00 to 1.03 |

The four slower ones are, in this run's fixed warm order, each the longest request its environment had run so far
(`results.json` → `e12.warm_by_reach`). The likely mechanism: with a metadata-only clear, cache memory an environment
has never written is first touched inside the forward of the first request that reaches it, whereas P1's clear
touches every buffer before every request. That is an inference. Lambda allows no fault counting, and request size
and reach are confounded because the warm order is fixed on Lambda. The offline check shuffles the order for
correctness, not timing.

## Correctness

- 840 of 840 answers that P1's `reference/outputs.json` covers are bit-identical to it (P1's five requests; the
  reference has no entries for E12's stress requests).
- 660 of 660 e12 answers equal the dor answer for the same request and CPU part in this run, which covers the stress
  requests too. dor's own answers agree with each other.
- Offline on arm64 (`offline_run.sh`, NOT Lambda; DevSpace, CPU part 0xd40, P1's pinned Lambda image, 6 threads):
  three arms (P1's own shim, the dormant switch, `DEC_CLEAR_DATA=0`), each run twice, alternating, over the nine
  requests in 6 seeded shuffled orders with every request repeated 3 times in a row. All 972 rows have identical
  answers and identical SHA-256 of every hidden-state array the torso returned, at the same position across arms
  and for the same request at any position within an arm. A short request run right after `wide-16q` (a 16-question
  fork over a 2.8k-token state, the widest fork `N_SEQ=17` allows) matches the same request run first.
  `results/<id>/offline_compare.json`.

## Host variability

`e12_entry.py` records per environment: `/proc/cpuinfo` implementer, part, variant, revision and Features;
`AT_HWCAP`/`AT_HWCAP2`; CPU count and affinity; kernel; Python; memory; region; log stream. The cgroup v2 files
(`cpu.max`, `memory.max`, `cpuset.cpus.effective`) are not readable in the sandbox and are recorded as `unknown`, as
are the AZ and the physical host.

| CPU part | Variant | HWCAP | HWCAP2 | dor envs (cold / restore) | e12 envs (cold / restore) |
|---|---|---|---|---|---|
| 0xd40 | 0x1 | 0xfffffff | 0x1f201 | 50 / 54 | 54 / 51 |
| 0xd0c | 0x3 | 0x119fff | 0x0 | 7 / 6 | 6 / 9 |
| 0xd4f | 0x0 | 0x2ff3ffff | 0x3f3bf | 3 / 0 | 0 / 0 |

All environments reported 6 CPUs, kernel 5.10.255 and Python 3.12.14. No Graviton generation is claimed: parts and
features are reported as read. Every delta is computed within one CPU part.

`results.json` → `host_variability` holds the (round, CPU part) cells. For the targeted phase (restore `Duration`),
all 8 shared cells have a negative delta and every e12 row is below every dor row (`p_e12_below_dor` = 1.0). Three
0xd40 cells fail the "delta larger than the control range" rule only because one dor row is slow (e.g. 3.38 s
against 0.66 to 0.80 s for the rest). Rows without a CPU identity: 0. Three cells have only one arm (0xd0c drew
unevenly) and are listed, not compared. A rerun is not needed: in every cell the slowest e12 row is faster than the
fastest dor row, and on 0xd40 the delta is 2 to 7 times the range of the dor rows once each cell's slowest dor row
is set aside. A fixed host (H1) was not used: the change removes memory writes
on SnapStart-restored memory, which a fixed EC2 host cannot reproduce.

Fixed-host timing from the offline run is not reported. The DevSpace was shared with other cards' jobs (load
average 12 to 18) during the run.

## Reproduce

See `meta.json` → `reproduce`. `offline_run.sh` and `build_shim.sh` need an arm64 host with Docker and P1's
`build/`. `build_zip.py`, the deploy, the bench and the summary need only Python and boto3.

## Limits

- Lambda only: not applicable to SageMaker-hosted models, and not measured there (see Scope above).
- One region (us-east-2), one memory size (10,240 MB). The 0xd0c restore cells are small (6 / 9 environments) and
  0xd4f drew no restores.
- The warm first-touch explanation is an inference. The warm order on Lambda is fixed, so "first to reach" and
  "longer request" cannot be separated in this run.
- `build_rs` zeroes the fresh state row with `x * 0`. A NaN or Inf left in that row by an earlier request would
  survive. Correct requests do not write NaN/Inf into the cache (the offline run checks bytes over every order tried,
  and P1's answers are finite), but a request that did would contaminate the next one in this mode, where P1's
  memset would have cleaned it. Not observed; recorded as a risk.
- A snapshot taken by the e12 arm also has the warm-up request's caches already touched, as dor's has. The saving
  comes from not re-zeroing restored pages; the deferred first touch on longer requests is the cost. Touching the
  caches once before the snapshot (a `before_snapshot` hook) is a follow-on, not measured here.
- Interactions to measure on a ladder rung: E10 (a smaller `n_ctx` shrinks the same buffers and the clear) and E11
  (memfd page release before the snapshot).
