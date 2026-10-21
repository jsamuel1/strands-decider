# I1: the first restore wave after a long publish (restore measurement integrity)

**Cause found.** It is client-side and specific to this harness's network path; it is not a Lambda or SnapStart
effect. `baseline/bench.py` at the pinned P1 baseline (`aac59d45555ebded9357c23f6f8e4f939edf25d7`) uses one boto3
Lambda client for every lifecycle call and every invoke. Its pooled HTTPS keep-alive connections sit idle through
the publish phase. The benchmark host reaches us-east-2 through a NAT gateway (`ip-10-0-3-99`, private subnet,
ap-southeast-2). Once a connection has been idle on that path for longer than somewhere between 200 s and 400 s, the
flow is dropped silently somewhere on the path, while the client socket still reads `ESTABLISHED`. urllib3's reuse
check sees nothing wrong and writes the next invoke into it. The request bytes are never acknowledged: the
kernel retransmits them until the client read timeout fires. Lambda never sees the request. It writes no `START`
line and its `Invocations` metric falls short by exactly the number of stalled requests. A fresh client opens new
connections and is never affected.

Ruled out: (a) the version's age since publish, which had no effect from 0 to 1,102 s, and (b) SnapStart cache
warm-up for a version never invoked before. Both were tested at every idle length through a fresh client, and
neither produced a single error.

**Fix:** `bench.py` on this branch invokes every cold, restore and warm wave through a new client
(`Bench.new_invoke_client`). Lifecycle calls go through a separate client with a 60 s read timeout and standard
retries, so a dead pooled connection there costs one retry instead of a hang. The fix is the same design as P1's
own later fix (`feat/lambda-kleidiai-baseline` e262455) and P3's `invoke_client()`, so experiment branches converge
on one shape. The tests are `tests/test_lkai_baseline.py::test_every_wave_invokes_through_its_own_new_client` and
`::test_lifecycle_client_retries_and_a_lone_invoke_still_gets_a_new_client`. Both fail against aac59d4's
`bench.py`.

## The repro

`repro.py` (this directory) uses a 128 MB python3.13 arm64 SnapStart function that sleeps 0.3 s, so the model is not
involved. Each arm has its own function and its own clients, and all 20 arms run in parallel:

1. Publish v1 and run a warm-up wave of 10 concurrent invokes through client R. This leaves 10 pooled connections,
   as bench.py's cold wave does.
2. Publish v2 and wait for its snapshot. Then idle until `idle` seconds have passed since the warm-up wave ended.
3. Run a measured wave of 10 concurrent invokes: no retries, 120 s read timeout.

| Arm | Measured wave goes through | Target | Tests |
|---|---|---|---|
| `reuse_poll` | R, which also made every publish and readiness poll (every 3 s). This is aac59d4 bench.py | v2 | the P3 shape |
| `reuse_nopoll` | R, whose pool sat untouched; lifecycle calls used another client | v2 | pure pool idle |
| `fresh_new` | a new client | v2 (published during the idle, never invoked) | (a) and (b) |
| `fresh_old` | a new client | v1 (invoked once, then idle for `idle` s) | (a) on an old version |

Run `20261021T044949Z`, us-east-2, from the same client host as P3. Raw data is in `results/20261021T044949Z/`:
`conditions.json` (with `lambda_side`), `invocations.jsonl.gz`, `tcp_samples.jsonl.gz` and `urllib3.log.gz`.
It is published to the P2 dashboard as `I1-restore-stall/20261021T044949Z`
(<https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results>, version 28; store
`s3://strands-decider-lkai-results-apse2/runs/I1-restore-stall/20261021T044949Z/`). Its status is `failed`
because the record has no model phases by design (P2's schema requires them for every other status); the
investigation itself found the cause. The record's experiment commit is `5764b9e`, the tested code.

```bash
cd examples/lambda-kleidiai/baseline/restore_stall
AWS_PROFILE=<p> python repro.py up       --build build/ --idle 30,200,400,700,1100
AWS_PROFILE=<p> python repro.py run      --build build/ --out build/runs/<run-id> --idle 30,200,400,700,1100
AWS_PROFILE=<p> python repro.py teardown --build build/
```

### Results (measured wave, 10 invokes per cell)

ok/10 is the client-side result. `Inv` is CloudWatch `Invocations` for the target version over the whole run, and
`START` is the count of `START ... Version: N` lines in the function's log group. Both include the warm-up wave's 10
for `fresh_old` (target v1).

| Idle (s) | `reuse_poll` ok / Inv / START | `reuse_nopoll` ok / Inv / START | `fresh_new` ok / Inv / START | `fresh_old` ok / Inv / START |
|---|---|---|---|---|
| 30 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 20 / 20 |
| 200 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 20 / 20 |
| 400 | **1** / **1** / **1** | **0** / **0** / **0** | 10 / 10 / 10 | 10 / 20 / 20 |
| 700 | **1** / **1** / **1** | **0** / **0** / **0** | 10 / 10 / 10 | 10 / 20 / 20 |
| 1100 | **1** / **1** / **1** | **0** / **0** / **0** | 10 / 10 / 10 | 10 / 20 / 20 |

- **The P3 shape reproduces exactly.** At 400 s idle and above, `reuse_poll` gets 1 of 10, with 9
  `client:ReadTimeoutError` at the 120 s timeout. The one that succeeds goes out on the connection the 3 s
  readiness polls kept active (the urllib3 log shows 9 new connections at warm-up, not 10, because the poll
  connection was reused). `reuse_nopoll`, which made no polls, gets 0 of 10. P3 saw 9 of 10 in 5 of 5 rounds, and
  E1's first attempt saw 9 of 10 after 604 s. E4's pilot saw 2 of 3 `ConnectionClosedError` after 455 s; that is a
  RST instead of a silent drop, but the cause is the same.
- **Lambda never received the stalled requests.** `Invocations` and `START` equal the client's ok count in every
  cell. That matches P3's aborted run (`Invocations` = 1 for `lkai-p3-rep:1`).
- **The bytes were never acknowledged.** `tcp_samples.jsonl.gz` holds snapshots, taken 5 to 108 s into the wave, of
  every TCP socket on the stalled connections' local ports. Each stalled socket reads `ESTABLISHED`, with about
  1,785 B unacknowledged (`tx_queue`, one invoke request) and its retransmit counter climbing from 3 to 7. The
  surviving `reuse_poll` connection shows `tx_queue 0` and no retransmits.
- **urllib3 cannot see it.** Just before each measured wave, every pooled socket reads `ESTABLISHED` and
  `wait_for_read(sock, 0)` returns false (`pool_before` in `conditions.json`). urllib3 relies on that check to
  decide whether to reconnect, so it logged no "Resetting dropped connection" and wrote each request into the dead
  flow.
- **Threshold:** clean at 200 s, fully stalled from 400 s. That brackets the AWS NAT gateway's documented 350 s
  idle timeout, and the client host's egress is a public NAT gateway in ap-southeast-2. **Which element drops the
  flow is not established.** AWS documents that a NAT gateway sends an RST to the instance when it uses a
  timed-out connection ([NAT troubleshooting](https://docs.aws.amazon.com/vpc/latest/userguide/nat-gateway-troubleshooting.html),
  "Internet connection drops after 350 seconds"). No RST arrived here: the sockets stayed `ESTABLISHED` and kept
  retransmitting for 108 s. The NAT's `IdleTimeoutCount` metric is shared with the host's other traffic and has
  no datapoint aligned with the 15:56 (+11:00) idle expiry, so it neither confirms nor rules out the NAT. The
  threshold was not bisected below 400 s.
- **Not version age, not cache warm-up:** `fresh_new` restored a never-invoked v2 that was 0 to 1,048 s old, and
  `fresh_old` invoked v1 after 30 to 1,102 s, all through a new client: 100 of 100 succeeded.
- Unrelated side observation, not part of the finding: `fresh_old` at 400 s or more restored instead of hitting the
  warm-up's environments (6 of 10 at 400 s, 10 of 10 at 700 s and 1100 s). Lambda had already retired those idle
  environments.

### Cost

About 400 invokes and 16.7 GB-s, plus snapshot cache for 40 tiny versions for about 20 minutes: well under US$0.10.
Teardown removed every function, version, log group and the role, and the post-teardown listing shows none left.

## Which published restore numbers are affected

The failure mode drops rows. It does not bias the latency of rows that succeeded. Every stalled request ends in
`client:ReadTimeoutError` or `client:ConnectionClosedError`, and `summarize.py` excludes errored rows. There is one
second-order effect: in each stalled P3 round, the first warm pass on that arm hit 9 environments that had not been
created yet, so 9 rows labelled warm were really restores (45 in P3 `20261020T124451Z`). `summarize.py` flags them
as `class_mismatch` and excludes them from both classes, so they are not mixed into warm numbers either. The affected
arm's restore distribution has fewer samples, and the stalled wave's surviving row is the only first-arm restore in
that round.

| Run | Client | Long idle before a wave? | Client errors | Effect |
|---|---|---|---|---|
| P1 baseline `20261020T080835Z` | shared (aac59d4) | 152 to 175 s | 0 | none |
| E2 `20261020T1125Z-e2` | shared | 291 to 317 s | 0 | none (under threshold) |
| E5 regen run | shared | 291 to 320 s | 0 | none (under threshold) |
| E3 cold/restore `20261020T113449Z` | shared, its own driver | up to 549 s | 0 | none. A different client host (python 3.12.13, boto3 1.43.115) and a different egress path; not tested here |
| E1 first attempt `20261020T1130Z-e1` | shared | 604 s | 9 + 1 | run discarded by E1 |
| E1 `20261020T1235Z-e1` | fresh per wave (93577a5) | 605 to 627 s | 0 | none |
| E4 pilot | shared | 455 s | 2 of 3 | pilot only |
| E4 `20261020T131645Z`, `20261020T181753Z` | fresh per wave (c1e161a) | 459 to 651 s | 0 | none |
| P3 aborted `20261020T115546Z` | shared | 1,109 s | 9 | aborted |
| P3 `20261020T124451Z` | shared | 1,074 to 1,112 s | 45 restore rows, 45 warm rows mislabelled | the affected restore arms have n = 23 (`rep`) and n = 32 (`telemetry`) against n = 50 for unaffected arms; latency rows exclude client errors and class mismatches |
| P3 rerun `20261020T231634Z` | fresh per wave | 1,252 to 1,347 s | 0 | none |

R: no published restore latency needs correcting. Published restore sample counts for P3's first-wave arms are
lower than nominal, and their uncertainty should reflect the n each one actually has. Any future run on a shared
client with more than about 200 s between waves on this host risks losing rows. Use this `bench.py` or an
equivalent fresh client per wave.

## Limitations

- One client host and one egress path, the one P3, E1, E2, E4 and E5 used. The threshold depends on the path.
  TCP keepalive below the idle timeout would probably also have prevented the stall; AWS recommends it for NAT
  idle drops, and `tcp_keepalive_time` here is 7,200 s with no `SO_KEEPALIVE` from botocore by default. It was not
  tested. A fresh client per wave does not depend on the path.
- The threshold is bracketed at 200 to 400 s, not bisected, and the dropping element on the path (NAT gateway,
  cross-region path, or the service front end) is not identified. The harness fix does not depend on which one it
  is.
- One run of 20 cells. Every cell was all-or-nothing and matched the mechanism, so more rounds were not needed to
  decide the cause.
