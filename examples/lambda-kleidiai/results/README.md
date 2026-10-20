# lambda-kleidiai results: schema, store and dashboard

Every lambda-kleidiai experiment (P1's baseline, E1–E6, the final ranking) publishes its measurements here. Each run
is one JSON file in `lkai-result/1` format. A generator builds a static dashboard from all of those files: an
overview of every experiment against the baseline, plus one detail page per run.

| | Where |
|---|---|
| Dashboard (durable URL) | <https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results>: Harmony Artifactory artifact `aece9ccf-c1d5-4217-b31e-e5b3c78ee759`, in the cluster [lambda-kleidiai experiments](https://artifactory.amazon.dev/clusters/907e0057-b2bb-46c9-abd4-6e934dbf63d4). Midway sign-in; PRIVATE until the owner shares it |
| Results store | `s3://strands-decider-lkai-results-apse2/` in the `playground1` account, region `ap-southeast-2`: private, versioned, SSE-S3, TLS-only |
| Run file | `runs/<experiment_id>/<run_id>/result.json` |
| Raw artifacts | `runs/<experiment_id>/<run_id>/artifacts/<path>` (`invocations.jsonl`, `logs/`, `conditions.json`, profiles) |
| Generated pages | `site/index.html`, `site/<experiment_id>/<run_id>.html`, `site/dashboard.html` (a single page with everything) |
| Schema | [`schema.json`](schema.json) |

## Publishing a run

```bash
export LKAI_RESULTS_STORE=s3://strands-decider-lkai-results-apse2
export LKAI_RESULTS_PROFILE=sauhsoj+playground1-Admin LKAI_RESULTS_REGION=ap-southeast-2
# Requires installed/configured artifactory-mcp; LKAI_DASHBOARD_ARTIFACT defaults to the artifact above.

python examples/lambda-kleidiai/results/publish_result.py <run-dir>/results.json [extra files in <run-dir>...]
```

The command:

1. validates the file (exit 2 lists every problem);
2. uploads the artifacts the file lists (`artifacts{}` and `profiling_artifacts[].path`, relative to the file's
   directory) and any extra paths, then `result.json`, all under `runs/<experiment_id>/<run_id>/`;
3. regenerates `site/` from every `result.json` in the store, and writes the pages to `--out`
   (default `build/lkai-dashboard/`);
4. uploads `dashboard.html` to the stable Artifactory artifact with a SHA captured before any result/store write,
   refuses if another dashboard version appears during the run, and verifies the READY artifact SHA-256 equals the
   generated file.

The command completes store publication, dashboard regeneration, and Artifactory upload in one invocation. It
starts the installed `artifactory-mcp` server directly, fetches the current artifact SHA before writing the run,
uploads `dashboard.html` only if that generation is still current, waits for `READY`, and re-fetches the artifact to
verify its SHA-256 equals the generated file. This keeps the single command idempotent and prevents stale publishers
from overwriting a newer dashboard. A stale-SHA refusal means another experiment published first: rerun
`publish_result.py --regenerate-only` to include that run and retry. Use
`--skip-dashboard-upload` only for local previews and tests.

P1's `baseline/run.sh` writes `results.json` in this format already (`baseline/summarize.py`). An experiment that has
its own harness can build the file with `lkai_results.dist()`, which computes the same distribution block.

The publisher needs `boto3` for an `s3://` store (it is the only dependency, and only that store imports it). On a
human-review card each Artifactory call asks the operator for approval; the gate prints the exact question.

A missing listed artifact is a publication refusal (exit 3): result JSON is never written until every listed raw
artifact and profile is readable. Symlinks that resolve outside the run directory are refused. This prevents a run
from claiming stable raw-artifact links when upload was incomplete.

Pass `--validate-only` to check a file without uploading anything. Pass `--store <dir>` for a local dry run with the
same layout.

## Why parallel experiments cannot clobber each other

- A publisher writes only under its own `runs/<experiment_id>/<run_id>/` prefix. The IDs are restricted to
  `[A-Za-z0-9._-]`, so a run cannot write outside its prefix.
- Republishing the same bytes is a no-op. Republishing different content under an existing run ID is refused
  (exit 3) unless you pass `--replace`. A new measurement gets a new `run_id`.
- No experiment edits a shared file. The pages in `site/` and the Artifactory dashboard are derived: every
  regeneration rebuilds them from all the run files. If two publishers race, the later regeneration includes both
  runs. After rendering, the publisher re-lists the store and renders again if a run appeared in the meantime.
- Artifactory uploads are compare-and-swap (`expectedCurrentSha256`), so a stale page cannot replace a newer one.

## What a run file holds

Everything the initiative brief asks each experiment to publish. `schema.json` is authoritative.

| Brief item | Field |
|---|---|
| outcome | `status`: `baseline`, `improved`, `no-change`, `regressed`, `inconclusive` or `failed` |
| hypothesis, single changed variable | `hypothesis`, `changed_variable` (`null` only for the baseline; otherwise `{name, baseline, experiment}`) |
| baseline and experiment commits | `commits.baseline`, `commits.experiment`, `commits.repo_dirty_at_build` |
| deployment, runtime, architecture, region/AZ | `deployment.{region, az, runtime, runtime_version_arn, architecture, memory_mb, snapstart, functions}` |
| model identity and format | `model.{id, revision, format, gguf_bytes, gguf_sha256}` |
| invocation conditions and sample count | `conditions.{rounds, concurrency, warm_per_arm_per_round, request_ids, started, finished, ...}`; each distribution's `n` |
| cold/warm/restore | `classification` and `phases.<arm>.<cold\|restore\|warm>` |
| per-phase distributions | `phases.<arm>.<class>.<phase>` = `{unit, n, p50, p90, p99, min, max, mean, stdev, histogram{edges, counts}}`. Phases include `weight_download`, `model_load_mmap` (deserialize/mmap), `context_init` and `model_ready` (model init), `first_request_inference`, `steady_state_inference`, `report_restore_duration` and `snapshot_init` |
| run-to-run variance | `variance` |
| errors | `errors.{count, rows}` |
| cost and resources | `cost.{billed_duration_s, s3_request_usd, s3_gb_transferred, total_usd, ...}`; a `report_max_memory_mb` distribution |
| correctness vs baseline | `correctness.{compared, bit_identical, choice_flips, max_abs_delta}` (`null` = not compared) |
| profiling artifacts | `profiling_artifacts[] = {name, tool, path \| url}` |
| raw logs | `artifacts{name: run-relative path}`; after publishing, `published.artifacts` holds each object's URI and SHA-256 |
| conclusion | `conclusion`. It is required for every experiment, including failures and null results |
| reproducible commands | `reproduce[]` |

The validator also requires each arm to carry cold, restore and warm classes; each class to carry its applicable
requested phase families (`weight_download`, `model_load_mmap`, `model_ready`, and first inference for cold;
restore duration and first inference for restore; steady-state inference for warm); fixed request IDs and
started/finished timestamps; cost totals; a cold max-memory distribution per arm; at least one raw artifact; and
complete baseline/experiment values for the single changed variable. A run that cannot measure one required phase
records that phase as `null` instead of silently omitting it.

The schema relaxes latency and deployment requirements for a `failed` run, but it still requires at least one raw
artifact, a hypothesis, the changed variable, commits, errors, a conclusion and the reproduce commands. Every other
status needs the full set. Account IDs are rejected anywhere in the file (`deployment.functions` holds names, not
ARNs).

## How results are shown

- The overview lists each run's status, changed variable, and p50 (with `n`) for four measures: cold first request,
  restore first request, warm inference and weight download. It also shows the percentage change against the
  baseline on the same phase and start class, plus errors, correctness, cost and conclusion. Failed runs have red
  rows. No-change and inconclusive runs have grey rows.
- Each detail page shows every phase for every arm and start class, with p50, p90 and p99 against the baseline. An
  inline SVG histogram marks p50, p90 and p99, with the baseline's p50 as a dashed line. A phase that was not
  measured is shown as "not measured (null)". The page also shows the classification, variance, errors,
  correctness, cost, artifact and profile links, and the reproduce commands.
- The baseline is the newest `status: baseline` run that is not a fixture.
- The pages are self-contained: inline CSS and SVG, no scripts and no external fetches. They render inside
  Artifactory's sandboxed iframe and from S3.

## Fixtures

`fixtures/` holds four synthetic runs: a baseline, an improvement, a null result with one unmeasured phase, and a
failure. They show the format and drive the tests. `fixtures/make_fixtures.py` regenerates them. Each is marked
`"fixture": true` and shown with a FIXTURE badge. Fixtures are hidden as soon as the store holds any real run;
pass `--include-fixtures` to show them anyway.

P1's real baseline is published as `P1-baseline/20261020T080835Z`, so the durable dashboard hides fixtures by
default. Before any real run exists in a fresh/local store, the dashboard shows these fixtures under a FIXTURE DATA
banner.

## Access and cost

- The bucket blocks public access and denies non-TLS requests. Links on the pages point at the S3 console, so
  opening them needs `playground1` access. The dashboard itself needs only Artifactory (Midway).
- Storage is result JSON plus logs: well under 1 GB at S3 Standard rates. Every request is a single PUT or GET.

## Tests

`tests/test_lkai_results.py`: schema and validator, publish idempotency and conflict refusal, parallel publishers,
the publish command's exit codes, self-contained rendering, failed and null rendering, fixture labelling and
escaping.
