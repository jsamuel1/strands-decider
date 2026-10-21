# E1: weights in an S3 Express One Zone directory bucket

One variable against the P1 baseline (`../baseline`): the type of bucket that holds the GGUF. Everything else is
P1's: the same `function.zip` (SHA-256 `26eb2651…`), the same Q8_0 GGUF (`8dbb8872…`), the same loader
(`examples/lambda/app/loader.py`, one HEAD then 64 MiB ranged GETs on 32 threads into a memfd), python3.12 arm64 at
10,240 MB, SnapStart on published versions, P1's pinned runtime version, request set, concurrency, rounds and
warm passes, all driven by P1's `bench.py`.

```bash
AWS_PROFILE=<profile> SOURCE=s3://<bucket holding P1's weights/ and code/> \
  examples/lambda-kleidiai/e1-s3-express/run_e1.sh <run-id>
```

No build step: `deploy_e1.py` copies P1's zip and GGUF server-side (SOURCE is only read) and reads each staged GGUF
back to check its SHA-256 before any function exists. `FROM_BUILD=<baseline/build>` uploads a local P1 build
instead. Knobs: `REGION` (us-east-2), `AZ_ID` (use2-az1), `ROUNDS` (5), `CONCURRENCY` (10), `WARM` (10), `PREFIX`
(lkai-e1), `TEARDOWN` (`versions`, `all` or `none`). `deploy_e1.py teardown --all` removes every E1 resource.

## Arms

A Lambda function outside a VPC has no fixed Availability Zone, so a directory bucket in one AZ is same-AZ for only
some execution environments, and which ones is not observable. A VPC-attached function fixes the AZ but adds its own
network setup to the cold start. E1 therefore runs both factors, all four arms in every round. Within a round the
arms run as sequential waves (each arm's cold wave, then each arm's restore wave, then warm passes), and the arm
order is reversed on odd rounds (`../baseline/bench.py`). That alternating round order balances arm order and
round-level drift, but drift within a round (time-within-round) is not controlled and is a limitation:

| Arm | Weights bucket | Network |
|---|---|---|
| `gp` | E1's own S3 general-purpose bucket (Standard) | no VPC, as P1 |
| `xz` | S3 Express One Zone directory bucket in `use2-az1` | no VPC, as P1 |
| `gpv` | the general-purpose bucket | private subnet in `use2-az1`, S3 gateway endpoint |
| `xzv` | the directory bucket | the same subnet, S3 Express gateway endpoint |

Contrasts in `results.json` under `e1.comparison` (each is a minus b, pooled medians, p95, per-round deltas):

- `bucket_type_no_vpc` = xz − gp: the deployable answer, and the headline.
- `bucket_type_vpc_same_az` = xzv − gpv: bucket type with the AZ held fixed.
- `vpc_confound_gp` = gpv − gp and `vpc_confound_xz` = xzv − xz: what the VPC itself costs.

The status rule (`summarize_e1.decide`) reads the headline's cold `weight_download`. It reports `improved` or
`regressed` only when every round moves the same way and the pooled change exceeds both 2% and the control arm's
range of per-round medians. Otherwise it reports `no-change`.

## What changed for directory-bucket access

Nothing in the function. The runtime-included SDK on the pinned runtime is boto3/botocore 1.42.97 (probed by
`deploy_e1.py`), which recognises the `--<az-id>--x-s3` bucket name, calls `CreateSession` and signs the HEAD and
ranged GETs with the session credentials against the zonal endpoint. CreateSession credentials last 5 minutes and
the SDK refreshes them. Each arm has its own execution role holding only what that arm needs. A general-purpose arm's
role is P1's: `s3:GetObject` on the weights object and log writes to its own group. A directory-bucket arm's role
replaces `s3:GetObject` with the one E1-specific grant, `s3express:CreateSession` on the directory bucket,
conditioned on `s3express:SessionMode = ReadOnly` (reads on a directory bucket are authorised by the session).
Only the two VPC arms' roles (`gpv`, `xzv`) carry `AWSLambdaVPCAccessExecutionRole` for their network interfaces;
`gp` and `xz` have no ENI permissions. Run 20261020T1235Z-e1 was measured with an earlier revision that gave all
four arms one shared role with that policy attached. IAM grants do not enter the measured path, so the numbers
stand; `deploy_e1.py up` moves the functions to the per-arm roles and deletes the shared one.

The VPC network is new and E1-only. It is a /24 subnet with its own route table, gateway endpoints for `s3` and
`s3express` attached only to that table, and a security group whose only egress is TCP 443 to the S3 and S3 Express
prefix lists. Gateway endpoints are free.

## Checks before any claim (verified 2026-10-20)

| | |
|---|---|
| Region / AZ | S3 Express One Zone in us-east-2 is in `use2-az1` and `use2-az2` (AWS docs, "Endpoints for directory buckets in Availability Zones") |
| SDK | runtime boto3/botocore 1.42.97; the loader's code path is unchanged |
| IAM | `s3express:CreateSession` (ReadOnly) on the bucket ARN `arn:aws:s3express:<region>:<account>:bucket/<name>` |
| Prices (Price List API, us-east-2) | Express: storage $0.11/GB-month, GET $0.00003 per 1,000, retrieval $0.0006/GB, upload $0.0032/GB. Standard: storage $0.023/GB-month, GET $0.0004 per 1,000, no retrieval charge. In-region S3 to Lambda transfer is free for both |
| Correctness | the GGUF's SHA-256 is checked in both buckets after staging; answers are compared, as complete objects, with P1's `reference/outputs.json` per request and CPU part |

Per 2 GB download: Standard costs 32 requests × $0.0004/1000 ≈ $0.0000128. Express costs 32 × $0.00003/1000 +
2.08 GB × $0.0006 ≈ $0.00125. At these sizes, Express's per-GB retrieval charge dominates S3 cost. Lambda duration
(about $0.0045 per cold start of ~34 s at 10 GB) dominates both.

## Host variability

Lambda chooses the host for each execution environment, and E1 cannot control that. Each invocation records the
`CPU part` from `/proc/cpuinfo` (read per request in P1's handler) and logs the ggml CPU variant it loaded.
`summarize_e1.py` writes `e1.host_variability`:

- `cpu_part_mix`: invocations per arm, class and CPU part.
- `cells`, `assessment`, `rerun` and `verdict`: the verdict basis. For every contrast, cold `weight_download` is
  compared inside each shared `(round, CPU part)` cell with per-arm `n`, medians, delta, ratio, sign, and the
  control arm's within-cell range. A cell is `directional` when |delta| exceeds that range, `in-band` when it does
  not, and `unassessed` with fewer than 2 control samples. Rows without a CPU part form no cell. The assessment is
  `comparable` (every assessed cell directional with the pooled sign), `null` (every assessed cell in-band),
  `mixed` (directional and in-band cells, no flip), `confounded` (directional cells of both signs, or against the
  pooled sign, or a CPU part with only unassessed cells showing the opposite sign) or `unknown` (no assessable cell).
- `method`: the harness order (alternating round order, not within-round interleaving) and the verdict basis.
- `within_cpu_part` and `headline_status_by_cpu_part`: descriptive only. Every contrast recomputed with rounds
  pooled inside each CPU part both arms ran on (at least 3 samples per arm), and the status rule on that pool.
- `pooled_mix_confounded`: pooled measures that pass the status rule while no CPU part does, meaning the arms'
  different CPU mixes produced the change.
- `unknown`: what the run did not record. That covers CPU implementer, variant and revision, feature flags and
  HWCAPs, core count, cgroup limits, kernel release, and the AZ of non-VPC environments.
- `snapshot_init`: the same check for the SnapStart snapshot inits, whose weight download sits behind every
  restore. Each `before_snapshot` record takes its CPU part from the engine init line in its own log stream, its
  `env_id` from the record and its version from the stream name. Per arm it reports the CPU-part mix, missing
  `env_id`s, streams or environments showing more than one part, and records per version. Per contrast it reports
  each arm's CPU-part shares and the snapshot `weight_download` median delta within each part with at least 3
  records per arm. Records whose stream shows no part (`unknown`) or several (`ambiguous`) are counted, never
  compared. No status rule is applied, because snapshot inits are not paired by round.

CPU part values are reported as recorded and are not mapped to a processor generation.

Run 20261020T1235Z-e1 saw three CPU parts: 0xd40 on 171 of 200 cold starts, 0xd4f on 17 and 0xd0c on 12. The
headline (xz − gp) has 12 shared cells. Six are directional and faster: rounds 0, 1, 2 and 4 on 0xd40 (−0.66 to
−0.77 s against gp ranges of 0.30 to 0.47 s), round 1 on 0xd4f and round 4 on 0xd0c (−1.02 s against 0.54 s).
One is in-band: round 3 on 0xd40, −0.64 s inside a 3.92 s gp range. Five have a single gp sample and are
unassessed; all five are also faster. No cell flips sign, so the assessment is `mixed`: the direction holds
within hosts, but the margin does not hold in every cell. The same-AZ contrast (xzv − gpv) is in-band in all 6
assessed cells (`null`). `pooled_mix_confounded` is empty.

No rerun was performed. The mixed headline is a stated limitation of the `improved` status. It does not affect the
"not recommended" conclusion, which rests on the same-AZ null and the cost increase. H1's fixed-host EC2 baseline
isolates CPU and memory effects, but E1's variable is the network path, so it does not apply.

The 56 snapshot inits all recorded an `env_id` and exactly one CPU part per stream. Every arm drew 0xd40 and
0xd0c in equal numbers (gp and xz 5 + 5, gpv and xzv 9 + 9), so the snapshot contrasts compare identical CPU
mixes. Within each part, xz − gp snapshot `weight_download` is −0.70 s on 0xd40 and −1.21 s on 0xd0c (5 vs 5
each), the same sign as the pooled −1.02 s. The same-AZ contrast stays within ±0.07 s on both parts. With 5 to 9
records per arm and part, these figures support the direction only.

## Files

- `deploy_e1.py`: buckets, network, per-arm roles, the four functions, SDK probe, smoke test, teardown.
- `run_e1.sh`: deploy → `../baseline/bench.py --arms gp,xz,gpv,xzv` → `summarize_e1.py` → redaction → P2 validation.
- `summarize_e1.py`: P1's `summarize.py` output plus `e1.comparison`, `e1.weight_download_pcts` (cold and
  snapshot-init p50/p95/p99 per arm), `e1.s3_costs_by_arm` (including cost per 1,000 cold starts),
  `e1.host_variability` and the status.
- `meta.json`: hypothesis, changed variable, AZ note, conclusion and reproduce commands.
- `results/<run-id>/`: committed, redacted run output (bucket names and the account ID replaced).
