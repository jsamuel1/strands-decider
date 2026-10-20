# E3: download strategy for the weights (parallel ranged GETs into memory)

E3 changes one thing: how the 1,980 MiB Q8_0 GGUF gets from S3 into the memfd that llama.cpp mmaps. The bucket,
object, format, loader and every phase after the download stay as P1 measured them. Every strategy writes straight
into the same anonymous memfd as P1, so no weight byte touches `/tmp` or disk.

**The card's premise did not hold.** The card assumed P1 downloads with one sequential `GetObject`. P1 already uses
parallel ranged GETs: `examples/lambda/app/loader.py` sends one `HeadObject`, then 64 MiB ranges on 32 threads. P1
measured 79 MiB/s, a 25.0 s cold download. On the operator's instruction ("download in parallel directly into RAM,
avoid disk"), E3 sweeps the parallel strategy against P1's setting. It does not test a single-GET arm.

## Result (run `20261020T113449Z`, playground1 us-east-2)

**No setting beats P1's 64 MiB × 32 threads.** The cold weight download sits at about 79 MiB/s (25.0 s) for P1's
setting and for 64 threads at P1's part size. Every other setting is slower:
- smaller parts or more requests: up to +11.4 s
- awscrt's S3Client: +0.2 to +2.3 s

Every strategy filled the memfd with exactly the GGUF's bytes. Every answer was bit-identical to P1's reference. The
compute phases after the download did not move, with one exception noted below.

Each row below is that setting against the control `ctl`. `ctl` is P1's exact zip and handler, run in the same rounds.
Values are the cold `weight_download` p50 with n = 50, 5 rounds × 10 new environments. Δ is the arm's median minus
`ctl`'s median (24.95 s), with a 95 % bootstrap interval. GETs counts S3 GETs per download. Peak RSS is VmHWM after
the download, before llama.cpp maps the weights.

| Setting | `E3_DL` | Status | p50 s | Δ vs ctl, s [95 % CI] | MiB/s | GETs | Peak RSS MB | `/tmp` MB | Hash | Answers identical |
|---|---|---|---|---|---|---|---|---|---|---|
| ldr (A/A) | `loader` (P1's loader through E3's zip) | no-change | 25.03 | +0.08 [−0.00, +0.16] | 79.1 | 31 + 1 HEAD | 599 | 0 | ok | 96/96 |
| aa | `threads:64:32` (P1's setting, re-implemented) | no-change | 25.03 | +0.07 [−0.00, +0.16] | 79.1 | 31 + 1 | 598 | 0 | ok | 96/96 |
| t64x64 | `threads:64:64` | no-change | 24.99 | +0.03 [−0.07, +0.11] | 79.3 | 31 + 1 | 597 | 0 | ok | 159/159 |
| t32x64 | `threads:32:64` | regressed | 25.13 | +0.17 [+0.08, +0.33] | 78.8 | 62 + 1 | 962 | 0 | ok | 98/98 |
| t16x64 | `threads:16:64` | regressed | 25.31 | +0.56 [+0.19, +1.73] | 77.0 | 124 + 1 | 861 | 0 | ok | 98/98 |
| t8x128 | `threads:8:128` | regressed | 36.31 | +11.40 [+10.15, +11.86] | 54.4 | 248 + 1 | 850 | 0 | ok | 98/98 |
| crtf16 | `crtfile:16:25` (CRT writes `/proc/self/fd/N`) | regressed | 25.14 | +0.18 [+0.07, +0.69] | 78.8 | 124 (computed) | 1,079 | 0 | ok | 97/97 |
| crt16 | `crt:16:25` (CRT, `on_body` pwrite) | regressed | 25.70 | +0.74 [+0.54, +1.48] | 77.0 | 124 (computed) | 1,875 | 0 | ok | 96/96 |
| crt8 | `crt:8:25` | regressed | 27.21 | +2.27 [+1.48, +4.70] | 72.7 | 248 (computed) | 1,475 | 0 | ok | 99/99 |

- **The ceiling is per environment.** The A/A arm, P1's setting and doubling the threads all land on the same
  79 MiB/s ≈ 663 Mbit/s. Their spread is small: the per-round medians of ctl and t64x64 range from 24.9 to 25.1 s,
  and p99 is ≤ 25.4 s. Adding connections does not raise the rate, so the limit is the environment's network
  bandwidth. The client's parallelism is not the limit.
- **More, smaller requests lose.** At ≤ 32 MiB parts the median moves a little, but a tail appears. In 6 to 46 of
  the 50 colds per setting, the download took 27 to 78 s. At 8 MiB × 128 threads, all 50 colds were slow. The
  tail did not come from one Graviton generation: it appeared on 0xd40, which is most samples, as well as on 0xd0c
  and 0xd4f. Its likely cause is per-request latency and throttling at more concurrent connections, but this run
  does not isolate it.
- **CRT adds nothing at this bandwidth.** CRT also uses about 0.5 to 1.3 GB more peak RSS before the mmap:
  - `on_body` buffers parts in Python and native memory.
  - `recv_filepath` (crtfile) is the leanest CRT path, at 1,079 MB.
  The function has 10,240 MB and the run peaks at about 5.1 GB, so this is not a constraint here.
- **CRT shutdown is safe.** Every CRT client shut down cleanly before the download returned (0.01 s). No CRT thread
  or socket can enter a SnapStart snapshot.
- **Compute is unchanged.** This is arm minus ctl within each CPU part, pooled, for model_load_mmap, context_init,
  assets_load, warm-up, first-request inference and warm steady-state inference. 51 of the 54 intervals include 0.
  The 3 that exclude 0 are all `model_load_mmap` on the CRT arms, at −0.19 to −0.34 s. This is the only phase after
  the download that moved. A plausible cause is that CRT has already touched more of the memfd's pages, but this
  run does not test that. It does not change the ranking: on the crt arms the net cold first request is still
  +0.06 to +2.3 s (`e3.end_to_end`).
- **Restore: no change that the download can explain.** For SnapStart restores, the weights download only during
  the snapshot init inside `publish_version`, which no request waits for. The restore leg ran the best arm
  (t64x64) and ctl, 3 rounds × 10 restores each, all on Graviton3:
  - Snapshot-init download: 24.82 s against 24.86 s (n = 6 each).
  - Restore Duration p50: 3.26 s against 3.20 s, Δ +0.07 s (+1.7 %), 95 % CI [+0.01, +0.10].
  - First request after restore: Δ +0.07 s, CI [−0.00, +0.11].
  The +0.07 s Restore Duration gap cannot come from the download strategy, because a restore downloads nothing.
  The two functions also run different zips: t64x64 runs the E3 zip, which is 4 MB larger with awscrt, though the
  threads arm never imports awscrt. With n = 30 per arm, the gap may also be noise. This run does not separate those
  causes.
- **S3 request cost** per 1,000 cold starts is $0.0128 at P1's 31 GETs + 1 HEAD, up to $0.0996 at 248 GETs. That
  is negligible next to Lambda duration: one cold start bills about 36 s × 10 GB, $0.0048.

**Recommendation.** Keep P1's loader. Below 64 MiB parts, transfer gets slower and the tail gets worse. Above 32
threads, nothing improves. awscrt adds 4 MB of zip and up to 1.3 GB of peak RSS for no gain. A faster cold start has
to come from a different variable:
- a smaller or differently laid-out model (E5)
- overlapping the download with the load (E4)
- a bucket type with more bandwidth per connection (E1)

The ~663 Mbit/s per-environment ceiling is the number those experiments need to beat.

## Control identity

`ctl` runs P1 zip `d148d706…`, and the E3 zip is built from it. P1's published run `20261020T080835Z` deployed
`26eb2651…` (its `conditions.json` and `build-manifest.json`), a rebuild of the same inputs. The two zips have the
same 1,899 entries. They differ only in the 4-byte source-mtime field of the `.pyc` headers; every `.pyc` body and
every other file is byte-identical. The control also reproduces P1's published cold numbers within 0.08 s:

| | P1 run, p50 | E3 ctl, p50 |
|---|---|---|
| imports | 0.715 | 0.719 |
| weight_download | 25.04 | 24.96 |
| model_load_mmap | 8.37 | 8.37 |
| model_ready | 34.97 | 34.91 |
| first request | 35.99 | 35.96 |

## How it is measured

- `e3dl.py` holds the strategies, selected by `E3_DL`:
  - `loader`: P1's function.
  - `threads:<MiB>:<n>`: P1's algorithm, parameterised and counting its GETs and retries.
  - `crt:<MiB>:<Gbps>`: awscrt `S3Client`, pwriting each body chunk at its offset.
  - `crtfile:...`: the same client with `recv_filepath=/proc/self/fd/N`.
  `E3_VERIFY=1` makes it sha256 the memfd. That runs only on the separate verify invocations, because hashing adds
  ~3 s.
- `e3_entry.py` installs the strategy as `loader.download_to_memfd`, then imports P1's `lkai_entry`. P1's own timers
  therefore time the download (`weight_download`, `weight_mib_s`) and every later phase. The `e3_*` counters ride in
  P1's `LKAI` records.
- `e3deploy.py build` makes the E3 zip from P1's zip (`d148d706…`) byte for byte, plus `e3dl.py`, `e3_entry.py` and
  the awscrt 0.38.0 aarch64 wheel (sha256 `f54a804f…`). The result is `4a81582c…`. `up` creates one function per
  setting (`lkai-e3-<arm>`) and the control `lkai-e3-ctl`, which runs P1's zip and handler. All of them use P1's
  configuration:
  - python3.12, arm64, 10,240 MB, 512 MB ephemeral, SnapStart on published versions
  - P1's environment and runtime-version ARN
  - P1's weights object, read-only
  Only `E3_DL` differs. The role may `GetObject` only that key.
- `e3bench.py` reuses P1's `Bench`: the same invoke, REPORT parsing, start-class check, rows and log capture.
  - Cold mode: every round sets a new `COLD_TAG` on every arm. Then each arm in turn runs 10 simultaneous cold
    invokes of P1's probe and one warm pass on those environments. The arm order rotates by one each round, so no
    setting always runs first, last or next to the same neighbour.
  - Restore mode: publish, 10 restores and one warm pass per arm. The version is deleted at once.
  - Verify mode: one hashed cold per arm.
- `summarize_e3.py` writes one `lkai-result/1` file per setting, `results/<run>/<arm>/results.json`, with
  `primary_arm` = the setting. That file holds the setting's and ctl's phases, rows and logs, and an `e3` block:
  - download distribution and Δ vs ctl with a bootstrap CI
  - requests, retries, peak RSS, `/tmp`, CRT shutdown
  - CPU-matched compute deltas
  - end-to-end deltas
  - the hash check
  Status is `improved` only if the download CI is entirely below 0 and errors, class mismatches, answers and hash
  are all clean. It is `regressed` if the CI is entirely above 0, and `no-change` otherwise.
- `results/20261020T113449Z/` holds the 9 per-setting directories, `summary.md` and `summary.json`, and the raw runs
  (`raw-cold/`, `raw-restore/`, `raw-verify/`). The bucket name and account id are redacted (`baseline/redact.py`).
  The run had 1,000 cold-mode invocations (cold and warm), 120 restore-leg invocations (restore and warm) and 9
  verify invocations, with 0 errors and 0 class mismatches.
- Cost: about $4 at list prices. That is $2.8 in Lambda duration for the sweep and $1.0 in SnapStart cache for 6
  versions × the 3-hour minimum.

## Reproduce

See `meta.json` → `reproduce`. In short, with `AWS_PROFILE` set:

```bash
python examples/lambda-kleidiai/e3/e3deploy.py build --p1-zip p1.zip --p1-manifest p1-manifest.json --wheel awscrt-0.38.0-cp311-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
python examples/lambda-kleidiai/e3/e3deploy.py up
python examples/lambda-kleidiai/e3/e3bench.py --mode verify --out build/runs/verify-<id>
python examples/lambda-kleidiai/e3/e3bench.py --mode cold --rounds 5 --concurrency 10 --out build/runs/cold-<id>
python examples/lambda-kleidiai/e3/e3bench.py --mode restore --arms ctl,t64x64 --rounds 3 --out build/runs/restore-<id>
python examples/lambda-kleidiai/e3/summarize_e3.py --cold build/runs/cold-<id> --restore build/runs/restore-<id> \
  --verify build/runs/verify-<id> --baseline-commit 7f066aa03e99e0d3ef2efb5453ff6bf35db3f5dd --run-id <id> --out build/runs/summary-<id>
python examples/lambda-kleidiai/e3/e3deploy.py teardown        # versions; --all removes functions, role and the E3 zip
```

`p1.zip` and `p1-manifest.json` are P1's `code/d148d706…/` objects. The build needs no arm64 host, because it only
adds files to P1's zip.

## Limits

- One region (us-east-2) and one memory size (10,240 MB). Lambda's network share may scale with memory, and this
  run did not test that.
- Lambda chose the Graviton generation: about 85 % 0xd40. The download tail appears on all three generations.
- CRT request counts are computed as ceil(size / part size), not observed. CRT does not expose them, and they are
  labelled as computed.
- The restore leg covers t64x64 only, by operator choice. Restores never download, so no strategy can change them.
