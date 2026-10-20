# `lkai-result/1`: one benchmark run

`summarize.py` writes one `results.json` per run directory. Durations are seconds and memory is MB unless a block's
`unit` says otherwise. P2's dashboard ingests these files. Each experiment writes only its own
`results/<run-id>/` directory, so no two experiments edit the same file.

Every distribution block is `{unit, n, p50, p90, p99, min, max, mean, stdev, histogram: {edges, counts}}`.
Percentiles are nearest-rank, so with fewer than 100 samples `p99` equals `max`. Always read a percentile next to its `n`.

| Key | Brief field | Contents |
|---|---|---|
| `experiment_id`, `run_id` | | `P1-baseline` for the baseline; the run directory name |
| `status` | outcome | `baseline` for P1; experiments use P2's `improved`, `no-change`, `regressed`, `inconclusive` or `failed`. From `meta.json`, except that a run `summarize.py` cannot certify as a baseline is written as `failed`, with the reasons in `baseline_invalid` and `conclusion` |
| `hypothesis` | hypothesis | from `meta.json` |
| `changed_variable` | the single changed variable | `null` for the baseline |
| `commits.experiment`, `commits.baseline` | baseline and experiment SHAs | experiment harness/result commit from the run manifest (with `repo_dirty_at_build`), and `--baseline-commit`; `build.function_zip.source_commit` separately pins the commit that produced deployed zip bytes. `commits.provenance_check` names the dirty check the build ran (`lkai-build-manifest/2`: every executable input, tracked, untracked or deleted, against the commit; `/1`: tracked files only), `commits.dirty_inputs` its list, and `commits.harness` the harness commit and its dirty scripts |
| `deployment` | deployment, runtime, architecture, region/AZ | functions, runtime and pinned runtime-version ARN, arch, memory, ephemeral storage, timeout, SnapStart, handler, code sha256, environment (no bucket name), and `cpu_parts_observed` per arm and class. Lambda does not expose the AZ to the function, and the record says so |
| `model` | model identity and format | Hub id and revision, GGUF Q8_0, size and sha256, the transform (LoRA merge, H512 fold), llama.cpp commit, patches and flags |
| `build` | | immutable container image digests, toolchain, sha256 of each asset, and `function_zip` bytes/hash/source commit/input key (the `zip` stage key from `buildcache.py`; runs built before it carry the earlier git-index key) |
| `conditions` | invocation conditions, sample count | memory (MB), rounds, concurrency, warm passes, arms, probe request, request ids, client, weight-fetch method, start and finish times |
| `classification` | cold/warm/restore classification | the method, an `expected->observed` count matrix, mismatch rows |
| `phases.<arm>.<cold\|restore\|warm>` | latency distributions per phase | one distribution per phase (see the README's phase table). Also `engine_by_cpu_part`, `steady_state_by_request` (warm) and `snapshot_init` (restore: the init that ran inside `publish_version`) |
| `variance.<arm>.<class>` | run-to-run variance | per-round medians of the class's headline number, with their median, range, stdev and CV |
| `instrumentation_overhead` | | instrumented minus REPORT-only, within each CPU part both arms ran on, pooled |
| `errors` | errors | count and rows (arm, round, wave, error, detail). A success with no usable answer or latency is an error `malformed:<why>` |
| `cost` | cost and resource observations | invocations, cold starts, restores, versions, billed duration and billed restore duration, Lambda duration and request cost, SnapStart cache (3-hour minimum per version) and restore cost, S3 GET/HEAD count and cost per download, GB moved, storage per month, `snapshot_inits_per_version.<arm>` (observed `before_snapshot` records per published version), and the price table used. `report_max_memory_mb` sits under `phases` |
| `correctness` | correctness vs baseline outputs | `invocations`, `compared`, `bit_identical` (complete answer-object equality), `choice_flips`, `max_abs_delta`, `answer_shape_mismatches`, `without_answers` (errors and malformed successes), `no_reference_for_cpu_part`, and `complete` (every invocation compared), against `reference/outputs.json` |
| `profiling_artifacts` | profiling artifact links | empty for P1 (P3 adds profilers) |
| `artifacts` | | relative paths of the raw evidence: `invocations.jsonl`, `logs/`, `conditions.json` |
| `reproduce` | exact reproducible commands | from `meta.json` |
| `conclusion` | conclusion | from `meta.json`. A failed or null result is still a result: it keeps its errors and is published |
| `request_coverage.<arm>` | | `answered` and `missing` fixed request ids: a baseline needs every id answered on both arms |
| `arm_phases_env.<arm>` | | the `LKAI_PHASES` value each arm was deployed with (`inst`: `1`, `rep`: `0`) |

`results/<run-id>/meta.json` is the meta that run was summarized with, so `summarize.py results/<run-id> --reference
reference/outputs.json --meta results/<run-id>/meta.json --baseline-commit <sha>` regenerates its `results.json` from the
committed evidence. When P2's `examples/lambda-kleidiai/results/publish_result.py` is present, `run.sh` validates the new
file with `--validate-only`; publish it to the store and dashboard with that script (see its README).
