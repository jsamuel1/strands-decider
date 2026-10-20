# E2: weights in S3 One Zone-IA (vs S3 Standard)

**Outcome: no change in latency. One Zone-IA adds cost. Keep the weights in S3 Standard.**

One variable against the P1 baseline (`aac59d4`, branch `feat/lambda-kleidiai-baseline`): the storage class of the
GGUF weight object. Everything else is P1's pinned deployment.

```bash
AWS_PROFILE=<profile> SOURCE_BUCKET=<bucket P1's deploy.py upload wrote> \
  examples/lambda-kleidiai/experiments/e2-onezone-ia/run.sh <run-id>
```

`run.sh` needs no arm64 host or Docker, because it builds nothing. It runs `deploy.py up`, then P1's `bench.py`
unchanged with `--arms std,ozia`, then `summarize.py` and P1's `redact.py`. The default run (5 rounds × 2 arms × (10
cold + 10 restore + 10 warm passes of 10)) cost $6.01 at list prices, plus the $0.02 30-day minimum on the One Zone-IA
copy.

## What is held fixed and what changes

| | `std` arm (baseline) | `ozia` arm (experiment) |
|---|---|---|
| Storage class | `STANDARD` | `ONEZONE_IA` |
| Bucket | the same new general-purpose bucket, `us-east-2` | same |
| Bytes | server-side copy of P1's GGUF; S3 full-object SHA-256 = `8dbb8872…` (checked by `deploy.py`) | same |
| Fetch path | `examples/lambda/app/loader.py`: `HeadObject` + 31 ranged 64 MiB `GetObject`s on 32 threads into a memfd | same |
| Function | P1's published-run zip `26eb2651…`, `python3.12` arm64, 10,240 MB, SnapStart on published versions, `LKAI_PHASES=1`, runtime pinned to `…runtime:0fec05b8…` | same; only `MODEL_KEY` differs |

`deploy.py up` refuses to continue unless each copy reports the intended storage class, the manifest's size, and a
`FULL_OBJECT` SHA-256 equal to the manifest's GGUF hash. It then smoke-invokes both functions and checks that the
runtime pin holds. Both arms are instrumented, so the instrumentation overhead does not enter the comparison
(P1 measured it).

## Decision rule (fixed before the run)

The rule uses the ratio of the One Zone-IA median to the Standard median for cold `weight_download`. The 95% interval
comes from a bootstrap of 10,000 resamples with seed 0. The result is **no change** if the interval lies inside
0.98–1.02, **improved** if it lies below 0.98, **regressed** if it lies above 1.02, and **inconclusive** otherwise.
`summarize.py` sets `status` from this rule.

## Results: run `20261020T1125Z-e2`

The run used commit `311ca2d`. There were 1,200 invocations, 0 errors and 0 class mismatches (100 cold, 100 restore,
1,000 warm). The table gives p50 values in seconds. Ratios are One Zone-IA over Standard, with bootstrap 95% intervals.
Full distributions are in `results/20261020T1125Z-e2/results.json` (`phases.<arm>`, `comparison`).

| Class | Phase | Standard | One Zone-IA | ratio [95% CI] | n |
|---|---|---|---|---|---|
| cold | weight download | 24.99 | 25.03 | 1.0016 [0.9978, 1.0054] | 50 / 50 |
| cold | download throughput (MiB/s) | 79.3 | 79.1 | 0.9975 [0.995, 1.0025] | 50 / 50 |
| cold | model load / mmap | 8.34 | 8.36 | 1.0024 [0.9964, 1.0065] | 50 / 50 |
| cold | model ready (PR #77 `_load`) | 34.91 | 34.94 | 1.0009 [0.9981, 1.0034] | 50 / 50 |
| cold | first request (Init + Duration) | 35.93 | 35.95 | 1.0007 [0.9985, 1.0034] | 50 / 50 |
| snapshot init | weight download | 24.85 | 24.85 | 0.9998 [0.9954, 1.0106] | 10 / 10 |
| restore | Restore Duration | 3.22 | 3.21 | 0.9966 [0.9852, 1.0085] | 50 / 50 |
| restore | first request (Restore + Duration) | 3.90 | 3.91 | 1.0029 [0.9889, 1.0115] | 50 / 50 |
| warm | steady-state inference | 0.660 | 0.665 | 1.0063 [0.9206, 1.1393] | 500 / 500 |

- **Download.** The two storage classes give the same download time. The p99 is 28.7 s for Standard and 25.2 s for
  One Zone-IA, a single slow Standard sample that is not a class effect. Both arms match P1's 25.04 s and 79 MiB/s.
- **CPU mix.** Lambda chose Graviton generations independently per arm (cold: Standard 43/1/6 and One Zone-IA 41/3/6
  for Graviton3/2/4). Within Graviton3 alone, the cold download ratio is 0.9995 [0.997, 1.005] and Restore Duration is
  0.9961 [0.985, 1.005].
- **Only interval outside 1.0.** The cold probe forward on Graviton3 is 0.2374 s against 0.2393 s (+0.8%, CI
  [1.002, 1.013]). That forward runs on weights already in memory and never touches S3, so it is not a storage-class
  effect. Read it as the noise floor of a 2 ms quantity.
- **Run-to-run variance** (CV of per-round medians, Standard / One Zone-IA): cold 0.09% / 0.15%, restore 0.73% /
  0.94%, warm 1.1% / 10.3%. The warm figure reflects the request mix per pass, not storage.
- **Correctness.** 1,200 of 1,200 answers are bit-identical to P1's `reference/outputs.json` for the same request and
  CPU part, with 0 choice flips and a maximum delta of 0. The SHA-256 over each arm's canonical answers per (request,
  CPU part) is `0977393b…` on both arms. Both weight copies have S3 full-object SHA-256 `8dbb8872…`.
- **Memory and billing.** Max memory is 5,159 / 5,160 MB. Billed duration across both arms is 4,535 s, against
  4,603 s in P1.

## Cost: the only effect

`us-east-2` list prices, read from the AWS Price List API on 2026-10-20 (`summarize.py` → `S3_PRICES`). The object is
2.0767 GB, and one download is 1 HEAD + 31 GETs. A SnapStart restore reads no S3 bytes, because the snapshot holds the
memfd. Only cold starts and snapshot inits pay (2 snapshot inits per published version were observed on both arms).

| | Standard | One Zone-IA | Express One Zone (E1, list) |
|---|---|---|---|
| Storage $/GB-month | 0.023 | 0.01 (30-day minimum, 128 KB minimum object) | 0.11 |
| GET $/1,000 | 0.0004 | 0.001 | 0.00003 |
| Retrieval $/GB | 0 | 0.01 | 0.0006 (above 512 KB per request) |
| **S3 $ per weight download** | 0.000013 | **0.0208** | 0.0012 |
| **S3 $ per 1,000 cold starts or snapshot inits** | 0.013 | **20.80** | 1.25 |
| Storage $/month for this object | 0.048 | 0.021 | 0.228 |

- This run: One Zone-IA retrieval was $1.246 (60 downloads, 124.6 GB). The Standard arm's S3 requests cost $0.0008.
- One Zone-IA saves $0.027 per month in storage and pays $0.0208 more per download. It breaks even at 1.3 downloads
  per month, and this function performs 2 snapshot inits per published version. One Zone-IA also keeps a single AZ
  copy, which Standard does not.
- E1 measures Express One Zone in its own run. The column above is list price only.

## Files

- `deploy.py`: bucket, checksum-verified copies, role (`GetObject` on the two weight keys only), functions, runtime
  pin, smoke test, teardown (`--all` removes everything).
- `summarize.py`: P1's `summarize()` plus `comparison`, the decision rule, per-arm and per-class `cost`, and
  per-arm `correctness` with answer hashes.
- `results/20261020T1125Z-e2/`: `results.json` (`lkai-result/1`, validated by P2's `publish_result.py`), `meta.json`,
  `invocations.jsonl`, `logs/`, `conditions.json`. Bucket name and account ID are redacted.
- Resources kept after the run, at the operator's request: bucket `lkai-e2-us-east-2-*`, role `lkai-e2-fn`, functions
  `lkai-e2-std` and `lkai-e2-ozia` (`$LATEST` only, so no SnapStart cache billing).
