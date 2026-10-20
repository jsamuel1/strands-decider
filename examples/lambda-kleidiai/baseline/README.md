# lambda-kleidiai P1 baseline: pinned build, phase timers, benchmark harness

The reference every lambda-kleidiai experiment compares against. It deploys the recipe in `snapstart/` exactly as
measured (same libraries, weights, flags and handler code), adds phase timers that do not change the request path,
and benchmarks cold, SnapStart-restore and warm starts with enough samples to report run-to-run variance.

```bash
AWS_PROFILE=<profile> examples/lambda-kleidiai/baseline/run.sh <run-id>
```

`run.sh` needs an arm64 Linux host with Docker (a Graviton instance or an arm64 DevSpace) and about 60 GB of disk.
It runs `build.sh`, `deploy.py up`, `bench.py` and `summarize.py`, then deletes the published versions so the
SnapStart cache stops billing. The results are in `results/<run-id>/results.json`. Knobs: `REGION` (default
`us-east-2`, the only Region with a pinned price table; `deploy.py` and `bench.py` refuse any other before touching
AWS, until a dated table is added to `summarize.PRICES_BY_REGION`), `MEMORY` (10240 MB; 128 to 10,240), `ROUNDS` (5;
at least 2, for run-to-run variance), `CONCURRENCY` (10; at least 1), `WARM` (10 passes; at least 1), `PREFIX`
(`lkai-p1`), `TEARDOWN` (`versions`, `all` or `none`). `MEMORY` is set on both functions, checked against the
deployed functions before the benchmark starts, and recorded in `conditions.memory_mb` and `deployment.memory_mb`.

`run.sh` exits non-zero when any invocation failed or returned no answer (including a non-finite latency or answer
value), when any invocation was not compared with the reference, when the run has no invocations, when an arm is
missing a cold, restore or warm sample, or when cleanup could not delete a published version. The failed run's
`results.json` is still written, as strict JSON, but its answers never become the reference. Cleanup attempts every
version even after one delete fails, and the benchmark's own failure is the exit status that is kept. Each wave
invokes through a new Lambda client, so no measured invoke reuses a connection pooled across the publish/snapshot
wait; that affects client round-trip timing only.

`build.sh` reuses a stage only when its content-addressed key matches (`buildcache.py`). The `llama` stage is
keyed on `LLAMA_REV` and the bytes of `llama/patch_llama.py`. The `weights` stage is keyed on the checkpoint and
revision, the Python image digest, the `llama` key, `weights.sh`, `weights/`, `prepare_model.py`, the package
sources and `pyproject.toml`. The `zip` stage is keyed on the Lambda image digest, the `llama` key and the bytes of
every file the zip copies or compiles, assets included. The `quant` stage is keyed on the quant type, the `weights`
key and the `zip` key. A changed key deletes that stage's outputs and every downstream stage's outputs, then rebuilds
them. `manifest.json` records all four keys in `stage_keys`, and `function_zip.inputs_sha256` is the `zip` key.

## What the example is (read from `7147a30`)

| | |
|---|---|
| Runtime | `python3.12` managed runtime, zip package. SnapStart supports Python 3.12 and later, so this runtime qualifies. The runtime version is pinned per function (`RuntimeManagementConfig` Manual, set by `deploy.py` from the first invocation's `Runtime Version ARN`) |
| Architecture | `arm64`. Lambda places it on Graviton2, 3 or 4. The portable llama.cpp build loads the best `libggml-cpu-*` variant per host |
| Memory | 10,240 MB (6 vCPUs) by default (`MEMORY`), 512 MB ephemeral storage (SnapStart's maximum), 900 s timeout |
| SnapStart | `ApplyOn: PublishedVersions`. Loading and warm-up run at import (`examples/lambda/app/handler.py`), so the snapshot holds a loaded model |
| Weight fetch | `examples/lambda/app/loader.py`: one `HeadObject`, then 64 MiB ranged `GetObject`s on 32 threads (a ~2 GiB object is 32 ranges), each resumed from the byte where it broke off. The bytes go into a `memfd`, not `/tmp`, and llama.cpp mmaps `/proc/self/fd/<n>` |
| Model | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` at Hub revision `216d1ccb`. The base is `Qwen/Qwen3.5-2B-Base` at the revision the checkpoint's `provenance.json` names. The LoRA is merged in fp32, every matmul weight is H512 Hadamard-folded, and the result is converted to GGUF f32 and quantized to Q8_0 (about 2 GB). The pointer head, tokenizer and temperatures ship in the zip (`assets/`) |
| Loader | llama.cpp `f74e2fa6a` with `llama/patch_llama.py`, built with `GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS` and KleidiAI. It is reached through `shim/decshim.c` by ctypes. It needs a readable, mmappable path to the GGUF, the `libggml-cpu-*` variants in `lib/`, and `libgomp`, `libstdc++` and `libgcc_s` bundled |
| Deploy tooling | The source used boto3 scripts (`snapstart/deploy.py`) against hand-staged S3 objects. `examples/lambda/` has a CDK stack for the PR #77 build. This baseline uses boto3 (`deploy.py`) so that versions and configuration can be driven per sample |

## What this directory changes

Nothing in the request path. `snapstart/app.py`, `engine.py` and PR #77's `handler.py` and `loader.py` ship
unchanged. The function's handler is `lkai_entry.handler`, which wraps them:

- `shim/decshim.c` gains `dec_load_phases_ns`. It reads `CLOCK_MONOTONIC` four times inside `dec_load_seq`
  (backend load, `llama_model_load_from_file`, `llama_init_from_model`) and changes nothing else.
- `phases.py` holds the standard-library timers and the parsers for the `LKAI {json}` stdout records and Lambda's
  `REPORT` line.
- `lkai_entry.py` times each wrapped call: the imports, `loader.download_to_memfd`, the shim's load phases, the
  asset load, the warm-up inference (the first inference) and PR #77's `_load`. It registers the SnapStart
  `before_snapshot` and `after_restore` hooks, and for every invocation it records the start class, the handler
  time and PR #77's `latency_ms` (`engine.evaluate`). The `REPORT` line is left alone.
- With `LKAI_PHASES=0` the module imports `app` and calls `app.handler` directly, with no timers. This is the
  REPORT-only arm.

`deploy.py` creates two functions from the same zip with the same configuration. `<prefix>-inst` has
`LKAI_PHASES=1` and `<prefix>-rep` has `LKAI_PHASES=0`. The instrumentation overhead is the difference in Lambda's
own numbers between the two (`results.json` → `instrumentation_overhead`).

## Phases

| Phase | Where it is measured | Start classes |
|---|---|---|
| `runtime_to_import` | process start (`/proc/self/stat`) to `lkai_entry` import | cold, snapshot |
| `imports` | numpy, pydantic, tokenizers, `strands_decider`, engine, loader | cold, snapshot |
| `weight_download` (+ `weight_mib_s`) | `loader.download_to_memfd` | cold, snapshot |
| `ggml_backend_load` | decshim: `ggml_backend_load_all_from_path` + `llama_backend_init` | cold, snapshot |
| `model_load_mmap` | decshim: `llama_model_load_from_file` (mmap, tensor setup, KleidiAI repack) | cold, snapshot |
| `context_init` | decshim: `llama_init_from_model` | cold, snapshot |
| `assets_load` | tokenizer, head and config (`engine_load` − `shim_init`) | cold, snapshot |
| `warmup_inference` | the handler's warm-up `evaluate` (the first inference) | cold, snapshot |
| `model_ready` | PR #77's `_load`, end to end | cold, snapshot |
| `report_init_duration` | REPORT `Init Duration` | cold |
| `report_restore_duration` | REPORT `Restore Duration` (and the billed value) | restore |
| `restore_to_invoke` | `after_restore` hook to the first invocation | restore |
| `first_request_inference` | `engine.evaluate` on the probe request, in a new environment | cold, restore |
| `first_request_total` | Init or Restore Duration + Duration | cold, restore |
| `steady_state_inference` | `engine.evaluate`, overall and per request | warm |
| `report_duration`, `report_billed_duration`, `report_max_memory_mb`, `client_round_trip` | REPORT and client | all |

"snapshot" means the init that runs inside `publish_version`. Its phases come from the `before_snapshot` record in
CloudWatch Logs (`results.json` → `phases.<arm>.restore.snapshot_init`), because no invocation sees that init.

Durations are never computed across a snapshot boundary. A restored environment's monotonic clock need not
continue from the snapshot's, so `restore_to_invoke` uses two monotonic reads taken after the restore.

## How the harness forces each start class

Each round runs these steps for both arms. The arm order alternates between rounds.

1. Set a new `COLD_TAG` on `$LATEST`. A configuration change retires every `$LATEST` environment.
2. **cold**: `CONCURRENCY` simultaneous invokes of `$LATEST`. Each needs a new on-demand environment.
3. Publish a version from that configuration and wait until its state is `Active` and
   `SnapStart.OptimizationStatus` is `On`.
4. **restore**: `CONCURRENCY` simultaneous invokes of that version. Each restores a new environment.
5. **warm**: `WARM` passes per arm on that version. A pass is `CONCURRENCY` simultaneous invokes of one request,
   so it reuses every environment the restore wave created, on whatever Graviton generations it drew. The arms
   alternate pass by pass, and the request cycles through all five.
6. Delete both versions. A version's SnapStart cache is billed as long as it exists.

The harness checks each invocation's expected class against its REPORT line: `Init Duration` means cold,
`Restore Duration` means restore, and neither means warm. Mismatches are counted and excluded from the phase
distributions; they are never relabelled. Cold and restore waves send `requests.json`'s probe request.

A success that is not correctness evidence is an error (`malformed:<why>` in `errors.rows`). That covers an HTTP
response shape, an absent or empty `answers` object, an answer that is not an object, answer ids that differ from
the request's question ids, and a missing `latency_ms`. Such a row is excluded from the distributions and counted
in `correctness.without_answers`.

Lambda chooses the Graviton generation per environment, and a forward on Graviton2 takes about 1.7 times as long as
on Graviton3. Every class therefore also reports `engine_by_cpu_part`, and the instrumentation overhead compares the
two arms only within a CPU part both landed on (`instrumentation_overhead.<class>.<metric>.by_cpu_part`, pooled by
sample count). An unmatched comparison would mostly measure the CPU mix.

## Correctness reference

The first run writes `reference/outputs.json`: one answer set per request and CPU part. Kernels differ by Graviton
generation (README, "Notes and caveats"), and within one generation the outputs repeat exactly. Later runs, and
every experiment, are compared against it on the same request and CPU part (`results.json` → `correctness`:
`bit_identical`, `choice_flips`, `max_abs_delta`). `correctness.complete` is true only when every invocation was
compared. `without_answers` and `no_reference_for_cpu_part` count the invocations that were not.

## Results format

`results/<run-id>/results.json` is `lkai-result/1` (`results/schema.md`). It covers every field the initiative brief
asks each experiment to publish. P2's dashboard ingests it. Raw evidence sits beside it: `invocations.jsonl` (one
row per invocation, with the log tail), `logs/` (CloudWatch events for the run's window) and `conditions.json`.

## Baseline results: run `20261020T080835Z`

The run's harness and result generation used commit `4412b31`, us-east-2, 5 rounds × 2 arms × (10 cold + 10 restore
+ 10 warm passes of 10). Deployed `function.zip` sha256 is `26eb2651…`, recorded in
`results/20261020T080835Z/build-manifest.json`; that zip was built from commit `fd96d7f`. Between `fd96d7f` and
`4412b31`, the only zip-input change was a comment in `container_build.sh`, so deployed executable inputs were
unchanged. This produced 1,200 invocations, with 0 errors and 0 class mismatches. Every answer was bit-identical to
the reference for its request and CPU part. The table shows the instrumented arm, p50 / p90 / p99 in seconds. Full
distributions, histograms and the REPORT-only arm are in `results/20261020T080835Z/results.json`.

| Class | Phase | p50 / p90 / p99 | n |
|---|---|---|---|
| cold | first request (Init Duration + Duration) | 35.99 / 36.41 / 48.90 | 50 |
| cold | REPORT Init Duration (the load runs in a thread, after init) | 0.80 / 0.82 / 1.04 | 50 |
| cold | weight download (79 MiB/s p50) | 25.04 / 25.19 / 25.27 | 50 |
| cold | model load / mmap (`llama_model_load_from_file`) | 8.37 / 8.67 / 19.86 | 50 |
| cold | imports, backend load, context init, assets | 0.72, 0.03, 0.32, 0.93 | 50 |
| cold | warm-up inference (first inference) | 0.25 / 0.26 / 0.42 | 50 |
| cold | first request inference (probe) | 0.24 / 0.25 / 0.44 | 50 |
| restore | first request (Restore Duration + Duration) | 3.95 / 4.20 / 7.50 | 50 |
| restore | REPORT Restore Duration | 3.27 / 3.50 / 6.72 | 50 |
| restore | after-restore hook to first invoke | 0.011 / 0.020 / 0.025 | 50 |
| restore | first request inference (probe) | 0.67 / 0.70 / 1.02 | 50 |
| warm | steady-state inference, all five requests | 0.66 / 1.74 / 1.82 | 500 |
| warm | max memory used | 5,170 MB of 10,240 | 500 |

- **CPU mix.** The cold waves drew Graviton3 (0xd40) 84 times, Graviton2 (0xd0c) 11 times and Graviton4 (0xd4f)
  5 times across both arms. The restore waves were Graviton3 99 times out of 100. The probe forward took 0.24 s on
  Graviton3 and 0.44 s on Graviton2. Compare experiments per CPU part (`engine_by_cpu_part`).
- **Warm latency by request** (p50): ticket-route-1q 0.24 s, cracked-screen-2q 0.46 s, payouts-3q 0.65 s,
  agent-action-guard-2q 0.85 s, incident-review-4q 1.74 s. The warm p90 and p99 reflect this mix of requests.
- **A restored environment's first forward is 2.8 times a cold one's** on the same request and CPU part (0.67 s
  against 0.24 s). Page faults on memory restored from the snapshot are a likely cause, but this run did not test
  that; P3's profiling can. Lambda bills that forward as Duration, not Restore Duration.
- **Snapshot init.** Publishing ran two snapshot inits per version (10 observed on the instrumented arm). Each
  downloaded the weights in 24.6 s (p50).
- **Run-to-run variance** is the coefficient of variation of the per-round medians, instrumented arm: cold 0.35%,
  restore 2.6%, warm 5.4%. The REPORT-only arm: 0.23%, 3.1% and 2.4%.
- **Instrumentation overhead** is instrumented minus REPORT-only on Lambda's numbers, Graviton3 only (the one CPU
  part both arms drew in every class), at p50. Cold first request: +0.17 s (+0.46%). Restore Duration: −0.019 s.
  Restore first request: −0.004 s. Warm Duration: +0.008 s (+1.2%). Each of these is within that class's
  run-to-run spread.
- **Cost** of the run, computed from the REPORT lines and list prices: $2.38. SnapStart cache was $1.63 (10
  versions × the 3-hour minimum), duration $0.61 and restores $0.14. The run moved 249 GB from S3 to Lambda in 120
  weight downloads (each one HEAD and 31 ranged GETs), for $0.0015 in requests and no transfer charge.

## Security and cost

- The bucket blocks public access and denies non-TLS requests, and objects use SSE-S3.
- The execution role may `GetObject` the one GGUF key and write its own two log groups. Nothing else.
- Both functions are reachable only by direct `Invoke` with IAM credentials. They have no Function URL.
- Committed results never include account IDs or bucket names. `run.sh` writes raw output to `build/runs/`
  (git-ignored with `build/`, as is `build/deployment.json`) and commits only the copy that `redact.py` makes.
- At us-east-2 prices, one default run (5 rounds × 2 arms × (10 cold + 10 restore + 100 warm)) cost $2.38.
  Most of that is the 3-hour SnapStart cache minimum on 10 versions. `results.json` → `cost` gives the computed
  figure for each run. `deploy.py teardown --all` removes functions, published versions, their CloudWatch log
  groups, role and bucket; cleanup fans out through failures and reports every failed step.
