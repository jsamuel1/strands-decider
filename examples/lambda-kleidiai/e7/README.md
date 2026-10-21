# E7: skip the KleidiAI repack of the unused tied output head (cold load)

**Variable.** Whether llama.cpp creates, and so KleidiAI repacks, the tied output head at model load. Everything
else is P1's baseline (`../baseline/`): the same GGUF bytes (`s2b-h512-Q8_0.gguf`, sha256 `8dbb8872…`, P1's weights
object read in place), the same download path, the same llama.cpp `f74e2fa6a` with P1's patches, the same function
configuration, request set and harness.

**Result (run `20261021T102524Z`, status `improved`).** Cold first request −2.11 s (36.03 → 33.92 s) on CPU part
0xd40, restore first request −0.35 s, Max Memory Used about −0.5 GB in every class, warm inference unchanged. All
1,200 answers are bit-identical to P1's reference. The P2 dashboard has the run as `E7-skip-tied-output`.

## Why the head is unused (step 1 of the card)

Read from llama.cpp `f74e2fa6a` after P1's `llama/patch_llama.py`, and from the engine:

- The deployed GGUF has no `output.weight`. `src/models/qwen35.cpp`, `load_arch_tensors`, then creates
  `output = create_tensor(token_embd.weight, …, TENSOR_DUPLICATED)`. The loader gives that duplicate the output
  layer's buffer list, so it becomes its own tensor in the `CPU_KLEIDIAI` buffer and is repacked at load. Meanwhile
  `token_embd` stays in the `CPU` buffer for `GET_ROWS`.
- `shim/decshim.c` always loads with `embeddings = true` and `pooling_type = NONE`. P1's prefill patch returns from
  the qwen35 graph before the LM head (`build_lora_mm(model.output, …)`), so `model.output` is in no graph.
- The other reader of `model.output` is the MTP draft graph (`graph_mtp`). It needs `load_mtp`, which is false by
  default, and decshim never sets it. `prism.hadamard.version` in the GGUF is 1 with no `tied_output`, so the
  Hadamard setup does not reference `output` either.
- The engine (`snapstart/engine.py`) reads only `llama_get_embeddings_ith`. The answer comes from the numpy pointer
  head (`strands_decider/hidden_engine.py`), not from llama.cpp logits.

No deployed request path reads llama's logits or the `output` tensor.

## The change

`patch_skip_output.py` is applied after P1's patches. It is a dormant switch:

- With `LKAI_SKIP_TIED_OUTPUT=1` set at load time, qwen35 does not create the duplicate, and `model.output` stays
  NULL.
- Unset, or any other value, the code is upstream's, plus one `getenv` per load.
- If anything ever builds the LM head with the head skipped (embeddings off, or an MTP graph), qwen35 aborts with a
  message instead of dereferencing NULL.
- Two exported queries let the function record that the switch took effect:
  - `llama_model_lkai_tied_output_skipped`
  - `llama_model_lkai_buft_bytes(model, "CPU_KLEIDIAI")`, the bytes per weight buffer type from
    `llama_model::memory_breakdown()`

One zip (`function.zip` sha256 `f6150b19…`, built at `719111e` by `build.sh` in P1's pinned image) serves both
functions. Handler `e7_entry.handler` imports P1's `lkai_entry` unchanged, so P1's phase timers time every phase:

| Function | Environment |
|---|---|
| `lkai-f768-ctl` | `LKAI_SKIP_TIED_OUTPUT` unset (P1's load) |
| `lkai-f768-skip` | `LKAI_SKIP_TIED_OUTPUT=1` |

On each environment's first invocation, after P1's handler has answered, `e7_entry` reads the host metadata and the
mechanism (`e7_host.py`). This readout is outside every P1 phase and takes about 1 ms.

## Measurements (Lambda)

Run `results/20261021T102524Z/`: python3.12, arm64, 10,240 MB, SnapStart on published versions, us-east-2, P1's
pinned runtime version. P1's harness at its pushed HEAD (`bench_e7.py` over `baseline/bench.py`) ran 5 rounds. Each
round had 10 cold and 10 restore invokes per arm, then 10 warm passes of 10. The arm order alternated every round,
and warm passes alternated the arm pass by pass. Each wave used a fresh Lambda client, with a 300 s read timeout. The
run had 1,200 invocations, 0 errors and 0 class mismatches.

Deltas are skip minus ctl within one CPU part, as read from `/proc/cpuinfo` inside the function: p50, with a 95%
bootstrap CI over environments.

| Phase | CPU part | ctl p50 | skip p50 | Δ p50 [95% CI] | n ctl / skip |
|---|---|---|---|---|---|
| cold `model_load_mmap` | 0xd40 | 8.38 s | 6.33 s | **−2.05 [−2.09, −2.00]** | 42 / 36 |
| cold `model_load_mmap` | 0xd0c | 19.65 s | 14.51 s | −5.14 [−5.57, −4.95] | 3 / 7 |
| cold `model_load_mmap` | 0xd4f | 7.12 s | 5.41 s | −1.71 [−1.85, −1.60] | 5 / 7 |
| **cold first request** (Init + Duration) | 0xd40 | 36.03 s | 33.92 s | **−2.11 [−2.20, −2.01]** | 42 / 36 |
| cold first request | 0xd0c | 48.34 s | 43.49 s | −4.85 [−5.19, −4.68] | 3 / 7 |
| cold first request | 0xd4f | 34.49 s | 32.69 s | −1.80 [−2.05, −1.33] | 5 / 7 |
| cold `weight_download` | 0xd40 | 25.03 s | 24.97 s | −0.06 [−0.11, +0.01] | 42 / 36 |
| cold Init Duration (REPORT) | 0xd40 | 0.801 s | 0.801 s | −0.001 [−0.009, +0.010] | 42 / 36 |
| REPORT Restore Duration | 0xd40 | 3.27 s | 2.92 s | **−0.34 [−0.39, −0.31]** | 46 / 48 |
| **restore first request** | 0xd40 | 3.95 s | 3.60 s | **−0.35 [−0.39, −0.30]** | 46 / 48 |
| restore first request | 0xd0c | 5.13 s | 4.66 s | −0.47 [−0.57, −0.38] | 4 / 2 |
| restore first forward | 0xd40 | 0.671 s | 0.673 s | +0.002 [−0.010, +0.014] | 46 / 48 |
| **warm inference** | 0xd40 | 0.643 s | 0.656 s | +0.013 [−0.084, +0.090] | 460 / 480 |
| Max Memory Used, cold | 0xd40 | 5,158 MB | 4,666 MB | −492 MB | 42 / 36 |
| Max Memory Used, restore | 0xd40 | 5,150 MB | 4,656 MB | −494 MB | 46 / 48 |
| Max Memory Used, warm | 0xd40 | 5,172 MB | 4,669 MB | −503 MB | 460 / 480 |

What this shows:

- **Mechanism, in every environment.** All 100 skip environments (50 cold, 50 restore) report the head skipped.
  None of ctl's 100 do. The `CPU_KLEIDIAI` weight buffer is 1,998,856,192 B on ctl and 1,458,511,872 B on skip
  (−540.3 MB), which is the repacked copy of `token_embd`.
- **Cold.** The saving is all in `model_load_mmap`. The download and Init Duration are unchanged, so this is load
  compute, not network. It is larger than E5's −1.7 s offline estimate on every CPU part. On 0xd0c it is −5.1 s,
  because that part's repack is slower.
- **Restore.** The saving is in Restore Duration, and the restored first forward is unchanged. This is consistent
  with a snapshot about 0.5 GB smaller, but the run did not measure snapshot size directly.
- **Memory.** Max Memory Used falls by about the size of the repacked copy in every class and on every part.
- **Warm.** Warm inference is unchanged. The CI includes 0 on both parts that warm passes ran on.
- **Run-to-run variance.** The coefficient of variation of the per-round medians is 0.24% (ctl) and 0.20% (skip)
  for cold, 2.4% and 1.0% for restore, and 11% and 4.0% for warm. One ctl warm round (round 3) had a 0.815 s median
  against 0.63 to 0.69 s in the other rounds; it is in the distributions.
- **Cost.** The run cost $2.37 at list prices, and the pilot (2 rounds × 3) cost $0.73. The operator's ceiling was
  $10. Most of it is the SnapStart 3-hour cache minimum on 10 versions.

## Correctness

- All 1,200 answers are bit-identical to P1's `reference/outputs.json` for their request and CPU part (complete
  answer objects, `choice_flips` 0, `max_abs_delta` 0).
- All 600 skip answers are identical to ctl's for the same request and CPU part in this run.

## Host variability

`e7_entry` records the following at each environment's first invocation, on the host that serves it:

- `/proc/cpuinfo` implementer, part, variant, revision and Features
- `AT_HWCAP` and `AT_HWCAP2`
- the CPU count and `sched_getaffinity`
- the kernel, the Python version and the log stream
- the CPU part of the host that ran the init (`init_cpu_part`, which is the snapshot init for a restore)

The cgroup v2 files are not exposed in the sandbox, so they are `unknown`. The physical host, AZ, CPU frequency and
co-tenancy are also `unknown`.

| CPU part | Variant | HWCAP / HWCAP2 | cold ctl / skip | restore ctl / skip | warm ctl / skip |
|---|---|---|---|---|---|
| 0xd40 | 0x1 | 0xfffffff / 0x1f201 | 42 / 36 | 46 / 48 | 460 / 480 |
| 0xd0c | 0x3 | 0x119fff / 0x0 | 3 / 7 | 4 / 2 | 40 / 20 |
| 0xd4f | 0x0 | 0x2ff3ffff / 0x3f3bf | 5 / 7 | 0 / 0 | 0 / 0 |

All 200 environments reported 6 CPUs, kernel 5.10.255 and Python 3.12.14. In all 100 restored environments, the CPU
part at the first invocation equalled the part recorded when the snapshot init imported the module. No Graviton
generation is claimed: the parts and features are reported as read.

`results.json` → `host_variability` checks each headline and target metric within each part and within each
(round, part) cell. The following all have the same sign (lower on skip) on every part's CI and in every cell:

- cold `model_load_mmap` (cell Δ −5.16 to −1.66 s)
- cold first request
- Restore Duration
- restore first request (cell Δ −0.41 to −0.29 s)
- Max Memory Used

Warm inference is null on every part. Host variation therefore cannot change the conclusion, and no rerun was needed
(`rerun_needed: false`).

Limitations:

- Some (round, part) cells exist for one arm only. They are listed in `cells_missing_an_arm`.
- The 0xd0c and 0xd4f samples are small (2 to 7 environments).
- The restore and warm legs drew no 0xd4f.
- The arms ran sequentially within a round. Alternating the arm order controls round-level drift, but time within a
  round remains a limitation.

H1's fixed-host EC2 baseline was not needed. The change removes load work, and the offline probe below already
isolates it on a fixed host.

## Offline check (fixed host, NOT Lambda)

`probe_e7.sh` ran on the arm64 DevSpace (CPU part 0xd40, 16 CPUs) in P1's pinned Lambda image. It used 6 threads
pinned to CPUs 0 to 5, a fresh container per load, and 6 interleaved pairs with the first arm alternating. Other
builds were running on the host at the time. The raw files are in `offline/`.

| | ctl p50 | skip p50 | paired Δ p50 (6 / 6 pairs below 0) |
|---|---|---|---|
| `model_load_mmap` | 8.00 s | 6.01 s | −1.99 s |
| `CPU_KLEIDIAI` buffer | 1,906.3 MiB | 1,390.9 MiB | −515.3 MiB |
| VmHWM after the fixed requests | 4,832 MiB | 4,328 MiB | −504 MiB |

Both arms' answers are bit-identical to each other and to P1's reference for 0xd40.

## Reproduce

See `meta.json` → `reproduce`. `build.sh` and `probe_e7.sh` need an arm64 Linux host with Docker and P1's `build/`.
`deploy_e7.py` and `bench_e7.py` need boto3, an `AWS_PROFILE`, and P1's deployed `lkai-p1-inst` (for its bucket,
weights key, environment and runtime ARN).

## Limits

- The run covers one region (us-east-2) and one memory size (10,240 MB).
- The 0xd0c and 0xd4f cells are small, and the restore leg drew no 0xd4f.
- The restore saving is attributed to a smaller snapshot by inference. Snapshot size is not exposed.
- The switch is safe only while nothing asks llama.cpp for logits. The patch makes that case abort loudly instead
  of running without the head.
- The baseline commit is P1's pushed HEAD `70eab7b`, not the card's `aac59d4`. P1 moved on after the card was
  filed. `70eab7b` already opens a fresh Lambda client per wave, which is I1's fix for the restore stall. Its zip
  inputs equal `aac59d4`'s except `baseline/weights.sh`. The GGUF is the same object, sha256 `8dbb8872…`.
