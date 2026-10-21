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
cold + 10 restore + 10 warm passes of 10)) cost $3.64 at list prices (`cost.total_usd`, the sum of the keys in
`cost.total_components`), including the $0.02 30-day minimum on the One Zone-IA copy.

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

That bootstrap treats the 50 cold starts per arm as independent samples. They are not: each round's 10 cold starts
run as one concurrent wave in a shared time window, and the arm order alternates by round. `summarize.py` therefore
also reports a sensitivity analysis, a paired round-cluster bootstrap (`comparison.sensitivity` and
`ratio_ci95_round_cluster` on every comparison). It resamples whole rounds with replacement, takes the same rounds
for both arms, and pools each drawn round's values. It is reported beside the primary interval and does not set
`status`. With only 5 rounds, it is coarse.

## Results: run `20261020T1125Z-e2`

The run used commit `311ca2d`. There were 1,200 invocations, 0 errors and 0 class mismatches (100 cold, 100 restore,
1,000 warm). The table gives p50 values in seconds. Ratios are One Zone-IA over Standard, with bootstrap 95% intervals.
Full distributions are in `results/20261020T1125Z-e2/results.json` (`phases.<arm>`, `comparison`).

| Class | Phase | Standard | One Zone-IA | ratio [95% CI] | round-cluster CI | n |
|---|---|---|---|---|---|---|
| cold | weight download | 24.99 | 25.03 | 1.0016 [0.9978, 1.0054] | [0.9972, 1.0067] | 50 / 50 |
| cold | download throughput (MiB/s) | 79.3 | 79.1 | 0.9975 [0.995, 1.0025] | [0.9937, 1.0025] | 50 / 50 |
| cold | model load / mmap | 8.34 | 8.36 | 1.0024 [0.9964, 1.0065] | [0.9981, 1.0039] | 50 / 50 |
| cold | model ready (PR #77 `_load`) | 34.91 | 34.94 | 1.0009 [0.9981, 1.0034] | [0.9993, 1.0027] | 50 / 50 |
| cold | first request (Init + Duration) | 35.93 | 35.95 | 1.0007 [0.9985, 1.0034] | [0.9993, 1.0032] | 50 / 50 |
| cold | probe forward (first request inference) | 0.2366 | 0.2392 | 1.0109 [1.0012, 1.0136] | [1.0066, 1.012] | 50 / 50 |
| snapshot init | weight download | 24.85 | 24.85 | 0.9998 [0.9954, 1.0106] | n/a | 10 / 10 |
| restore | Restore Duration | 3.22 | 3.21 | 0.9966 [0.9852, 1.0085] | [0.982, 1.0152] | 50 / 50 |
| restore | first request (Restore + Duration) | 3.90 | 3.91 | 1.0029 [0.9889, 1.0115] | [0.9842, 1.0133] | 50 / 50 |
| warm | steady-state inference | 0.660 | 0.665 | 1.0063 [0.9206, 1.1393] | [0.9739, 1.0926] | 500 / 500 |

The round-cluster interval for the primary metric is [0.9972, 1.0067]. It also lies inside 0.98–1.02, so the
outcome does not depend on treating concurrent cold starts as independent.

- **Download.** The two storage classes give the same download time. The p99 is 28.7 s for Standard and 25.2 s for
  One Zone-IA, a single slow Standard sample that is not a class effect. Both arms match P1's 25.04 s and 79 MiB/s.
- **CPU mix.** Lambda placed each execution environment on a host independently per arm. The handler records the
  `/proc/cpuinfo` `CPU part` (see [Host variability](#host-variability)). Cold starts: Standard 43 `0xd40`, 1
  `0xd0c`, 6 `0xd4f`; One Zone-IA 41 / 3 / 6. Within `0xd40` alone, the cold download ratio is 0.9995 [0.997, 1.005]
  and Restore Duration is 0.9961 [0.985, 1.005].
- **Intervals that exclude 1.0.** Every interval in the table above contains 1.0 except the cold probe forward.
  In `comparison.<class>.by_cpu_part`, these primary intervals also exclude 1.0:
  - Cold probe forward on CPU part `0xd40`: 0.2374 s against 0.2393 s, ratio 1.008 [1.0019, 1.0131], n=43/41.
    The aggregate probe-forward interval [1.0012, 1.0136] is this effect plus the CPU mix (One Zone-IA drew 3
    `0xd0c` cold starts, whose probe forward is about 0.43 s, and Standard drew 1). The forward runs on weights already in memory and never
    touches S3, so a storage-class cause is not plausible. Read it as run-to-run drift in a 2 ms quantity.
  - Cold weight download on CPU part `0xd4f`: 25.06 s against 25.18 s, ratio 1.0049 [1.0017, 1.009], and
    throughput 79.05 against 78.65 MiB/s, 0.9949 [0.9912, 0.9987]. This is n=6 per arm, drawn from different rounds
    (Standard rounds 0, 1, 2, 4; One Zone-IA rounds 1, 2, 3, 4). The paired round-cluster interval uses only the
    3 shared rounds. The point estimate is +0.5%, inside the ±2% band, and the `0xd40` subset (n=43/41) shows no
    difference. This is a small, low-n signal that the run cannot separate from round timing. It is not a
    performance effect of the storage class at the scale the decision rule tests.
  - Under the round-cluster sensitivity, cold model load on `0xd40` also excludes 1.0 by a hair, [1.0001,
    1.0033] (+0.16%), and `0xd4f` probe forward is lower for One Zone-IA, [0.9461, 0.983] (n=6). Neither is a
    download phase.
- **Run-to-run variance** (CV of per-round medians, Standard / One Zone-IA): cold 0.09% / 0.15%, restore 0.73% /
  0.94%, warm 1.1% / 10.3%. The warm One Zone-IA figure is the host mix, not storage: its round 4 served 30 of 100
  warm requests from `0xd0c` environments (median 1.12 s against 0.65 s for the other 70 on `0xd40`). Restricted
  to `0xd40`, the warm CV is 2.2% / 2.0% (`host_variability.warm_round_cv`). The warm count and median for every
  (arm, round, CPU part) are in `host_variability.warm_by_round_and_cpu_part`.
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

- This run: $3.639 total. One Zone-IA retrieval was $1.246 (60 downloads, 124.6 GB), the SnapStart cache $1.625,
  Lambda duration $0.605, SnapStart restore $0.140, the One Zone-IA 30-day minimum $0.021, S3 requests $0.0027
  (the Standard arm's share $0.0008) and Lambda requests $0.0002.
- One Zone-IA saves $0.027 per month in storage and pays $0.0208 more per download. It breaks even at 1.3 downloads
  per month, and this function performs 2 snapshot inits per published version (one on a `0xd0c` host and one on a
  `0xd40` host for every version on both arms). One Zone-IA also keeps a single AZ copy, which Standard does not.
- E1 measures Express One Zone in its own run. The column above is list price only.

## Host variability

Added for the initiative's host-variability rules (operator request item-mvhbiqd5-1degq-099). This is a re-analysis
of the committed evidence of run `20261020T1125Z-e2`. It made no new invocations and used no AWS resources.
`summarize.py` writes it to `results.json` → `host_variability`.

**Observed.** Each invocation records `cpu_part`, the `CPU part` line of `/proc/cpuinfo` read by the handler in that
execution environment. The saved CloudWatch logs record, per environment, which ggml CPU backend variants ggml's
own feature probe loaded, the log stream, and the runtime version. Three CPU parts appeared: `0xd0c` (ggml loaded
`armv8.2_2`), `0xd40` (`armv8.6_1`) and `0xd4f` (`armv8.6_2`). Each of the 200 environments reported a single CPU
part. Every environment ran 6 threads (`os.cpu_count()` in the init that loaded the model), 10,240 MB, and runtime
`…runtime:0fec05b8…`. The snapshot init for each published version ran twice, once on `0xd0c` and once on `0xd40`,
on both arms (5 + 5 each).

**Not observed, so unknown.** The `CPU implementer`, variant and revision lines, the `Features` flags and
`AT_HWCAP`/`AT_HWCAP2`, cgroup CPU and memory limits, the kernel release, and the core count of a restored
environment (its thread count comes from the snapshotting init). This README names CPU parts only. It makes no claim
about which processor generation or product a part is. The earlier wording that named generations was an inference
from the part number and timings, and was removed.

**Is the conclusion exposed to host mix?** No, and no rerun was made:

- The arms were interleaved by round, and both drew the same part mix within one or two cold starts per part (table
  above). The snapshot inits that seeded every restore ran on the same parts on both arms.
- The primary metric, cold `weight_download`, does not depend materially on the CPU part in this run. Pooled over both arms,
  its median is 24.84 s on `0xd0c` (n=4), 24.97 s on `0xd40` (n=84) and 25.15 s on `0xd4f` (n=12). Those medians
  span 1.2% end to end (`host_variability.primary_pooled_by_cpu_part.span_ratio` 1.0123), against a ±2% band on
  the arm ratio.
- Within each shared (round, CPU part) cell, the eight available arm comparisons have ratios from 0.994925 to
  1.009692 and deltas from -0.127197 s to +0.241319 s. Every ratio is inside the ±2% band; signs vary across rounds,
  so this is run timing around a null result rather than a stable storage-class direction. CPU parts absent from
  either arm in a round have no shared-cell comparison. The complete cells are in
  `host_variability.primary_by_round_and_cpu_part`.
- Restricted to the dominant part `0xd40` (84 of 100 cold starts), the primary ratio is 0.9995, primary interval
  [0.997, 1.0051], round-cluster [0.9973, 1.0072]. Both lie inside the band, so the outcome is **no-change** with
  the host held fixed (`host_variability.primary_within_dominant_cpu_part`).
- `host_variability.rerun_assessment` records **no rerun**: the dominant-part primary and cluster intervals preserve
  the pooled no-change outcome, and every shared round/CPU-part ratio remains inside the decision band. Sparse
  parts and changing within-round signs remain explicit limitations.
- Correctness is compared per (request, CPU part) against P1's reference: 1,200 / 1,200 bit-identical with 0
  missing references.
- The phases where the part mix does show (probe forward, warm inference) are compute on weights already in memory.
  They are reported per part above and are not storage-class effects. The `0xd0c` and `0xd4f` subsets are small
  (n ≤ 6 per arm for cold), and their arms were drawn from different rounds, so they decide nothing.

A fixed-host EC2 baseline (H1) isolates CPU and memory behavior. It cannot measure S3 storage-class effects inside
Lambda, and this experiment does not need it.

## Files

- `deploy.py`: bucket, checksum-verified copies, role (`GetObject` on the two weight keys only), functions, runtime
  pin, smoke test, teardown (`--all` removes everything).
- `summarize.py`: P1's `summarize()` plus `comparison`, the decision rule, per-arm and per-class `cost`, and
  per-arm `correctness` with answer hashes.
- `results/20261020T1125Z-e2/`: `results.json` (`lkai-result/1`, validated by P2's `publish_result.py`), `meta.json`,
  `invocations.jsonl`, `logs/`, `conditions.json`. Bucket name and account ID are redacted.
- Resources kept after the run, at the operator's request: bucket `lkai-e2-us-east-2-*`, role `lkai-e2-fn`, functions
  `lkai-e2-std` and `lkai-e2-ozia` (`$LATEST` only, so no SnapStart cache billing).
