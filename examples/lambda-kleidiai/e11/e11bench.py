"""E11 benchmark: `off` against `rel`, P1's lifecycle (cold, publish, restore, warm) with both arms' waves interleaved.

  python e11bench.py --rounds 5 --concurrency 10 --warm 10 --out build/runs/<run-id>

Per round:
  1. set a new COLD_TAG on both arms' $LATEST together (this retires every environment)
  2. cold     both arms' `concurrency`-wide waves of P1's probe, submitted together and interleaved request by
              request (off, rel, off, rel, ... in an order that flips each round), on new on-demand environments
  3. publish  both versions (sequentially, order flipping each round) and wait for each snapshot (P1's Bench.publish)
  4. restore  both arms' restore waves, interleaved the same way, on new restored environments
  5. warm     `warm` passes; a pass sends one request `concurrency` wide to both arms at once, interleaved, cycling
              through every fixed request
  6. delete both versions (SnapStart cache billing)
So each start class of the two arms is sampled over the same seconds rather than in sequential arm waves; the
remaining time-within-round confounder is the arm order inside a wave, which flips each round. Every invoke wave uses
a fresh Lambda client (P1's Bench), and --read-timeout bounds one invoke. Rows are P1's invocations.jsonl shape.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import e11deploy  # noqa: E402


def order_for(arms: list[str], rnd: int) -> list[str]:
    """Alternate which arm goes first, round by round."""
    return list(arms) if rnd % 2 == 0 else list(arms)[::-1]


def interleave(order: list[str], concurrency: int) -> list[tuple[str, int]]:
    """(arm, seq) submission order: one invoke per arm in turn, `concurrency` per arm."""
    return [(arm, i) for i in range(concurrency) for arm in order]


def make_bench(sess, st, reqs, out, concurrency, read_timeout):
    import bench as p1bench
    from botocore.config import Config

    class Bench(p1bench.Bench):
        def new_invoke_client(self):
            self.invoke_clients += 1
            return self.sess.client(
                "lambda",
                config=Config(
                    read_timeout=read_timeout,
                    connect_timeout=10,
                    retries={"max_attempts": 0},
                    max_pool_connections=max(32, self.concurrency * 4),
                ),
            )

        def paired_wave(self, order, quals, rnd, wave, rid=None, seq0=0):
            """Both arms' waves at once, through one fresh client, submitted interleaved."""
            lam = self.new_invoke_client()
            jobs = interleave(order, self.concurrency)
            with ThreadPoolExecutor(len(jobs)) as ex:
                futs = [
                    ex.submit(
                        self.invoke, arm, quals[arm], rid or self.probe, rnd, wave, seq0 + i, lam
                    )
                    for arm, i in jobs
                ]
                return [f.result() for f in futs]

    return Bench(sess, st, reqs, out, concurrency)


def conditions(b, a, arms, reqs) -> dict:
    import boto3
    import deploy as p1deploy

    fns = {}
    for arm in arms:
        f = b.st["functions"][arm]
        c = b.lam.get_function_configuration(FunctionName=f["name"])
        rm = b.lam.get_runtime_management_config(FunctionName=f["name"])
        fns[arm] = {
            k: c.get(k)
            for k in (
                "FunctionName",
                "Runtime",
                "Architectures",
                "MemorySize",
                "Timeout",
                "EphemeralStorage",
                "SnapStart",
                "Handler",
                "CodeSha256",
                "CodeSize",
            )
        }
        env = c["Environment"]["Variables"]
        fns[arm]["environment"] = {k: v for k, v in env.items() if k != "MODEL_BUCKET"}
        fns[arm]["runtime_management"] = {
            k: rm.get(k) for k in ("UpdateRuntimeOn", "RuntimeVersionArn")
        }
        if p1deploy.b64_to_hex(c["CodeSha256"]) != f["code_sha256"]:
            sys.exit(f"{arm}: deployed code {c['CodeSha256']} is not {f['code_sha256']}")
        if rm.get("RuntimeVersionArn") != b.st["runtime_version_arn"]:
            sys.exit(f"{arm}: runtime {rm.get('RuntimeVersionArn')} is not P1's")
        if env.get("LKAI_RELEASE_MEMFD") != e11deploy.ARMS[arm] or env.get("LKAI_PHASES") != "1":
            sys.exit(
                f"{arm}: LKAI_RELEASE_MEMFD={env.get('LKAI_RELEASE_MEMFD')!r} LKAI_PHASES={env.get('LKAI_PHASES')!r}"
            )
    others = [
        {
            k: v
            for k, v in fns[arm]["environment"].items()
            if k not in ("LKAI_RELEASE_MEMFD", "E11_ARM", "COLD_TAG")
        }
        for arm in arms
    ]
    if any(o != others[0] for o in others):
        sys.exit("arms' environments differ beyond LKAI_RELEASE_MEMFD")
    return {
        "schema": "lkai-conditions/1",
        "experiment": "E11",
        "changed_variable": "LKAI_RELEASE_MEMFD (unset vs 1)",
        "region": a.region,
        "memory_mb": fns[arms[0]]["MemorySize"],
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": a.warm,
        "read_timeout_s": a.read_timeout,
        "schedule": "paired interleaved waves: both arms' cold, restore and warm waves submitted together, "
        "alternating request by request; arm order flips each round",
        "arms": arms,
        "requests_file": "baseline/requests.json",
        "requests": reqs,
        "probe": reqs["probe"],
        "functions": fns,
        "build_manifest": b.st["manifest"],
        "runtime_version_arn": b.st["runtime_version_arn"],
        "client": {
            "python": platform.python_version(),
            "boto3": boto3.__version__,
            "host": platform.machine(),
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds_meta": [],
    }


def main() -> None:
    import bench as p1bench
    import boto3
    import summarize as p1sum

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(e11deploy.ARMS))
    ap.add_argument("--rounds", type=p1bench.at_least(2), default=5)
    ap.add_argument("--concurrency", type=p1bench.positive_int, default=10)
    ap.add_argument("--warm", type=p1bench.positive_int, default=10)
    ap.add_argument("--read-timeout", type=p1bench.positive_int, default=900)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=p1bench.nonnegative_int, default=20)
    a = ap.parse_args()
    arms = a.arms.split(",")
    if sorted(arms) != sorted(e11deploy.ARMS):
        ap.error(f"--arms must be exactly {sorted(e11deploy.ARMS)}")
    reqs = json.load(open(a.requests))
    if p := p1sum.request_set_problem(reqs):
        ap.error(p)
    if p := p1sum.warm_problem(a.warm, reqs):
        ap.error(p)
    if not a.profile:
        ap.error("--profile or AWS_PROFILE is required: no implicit credentials")
    st = e11deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E11 deployment; run e11deploy.py up")
    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(os.path.join(a.out, f)) for f in ("invocations.jsonl", "logs")):
        sys.exit(f"{a.out} already holds a run; use a new --out")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    b = make_bench(sess, st, reqs, a.out, a.concurrency, a.read_timeout)
    cond = conditions(b, a, arms, reqs)
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)

    def show(rnd, wave, recs):
        for arm in arms:
            rs = [r for r in recs if r["arm"] == arm]
            ok = [r for r in rs if not r["error"]]
            print(
                f"r{rnd} {arm} {wave}: ok {len(ok)}/{len(rs)} rt median "
                f"{sorted(r['rt_s'] for r in rs)[len(rs) // 2]:.2f}s",
                flush=True,
            )

    last_end = 0.0
    for rnd in range(a.rounds):
        order = order_for(arms, rnd)
        tag = f"e11-r{rnd}-{time.time():.3f}"
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        failed = True
        try:
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, t=tag: b.set_cold_tag(arm, t), order))
            show(rnd, "cold", b.paired_wave(order, dict.fromkeys(order, "$LATEST"), rnd, "cold"))
            for arm in order:
                published[arm] = b.publish(arm, tag)
            meta["published"] = published
            quals = {arm: published[arm]["version"] for arm in order}
            show(rnd, "restore", b.paired_wave(order, quals, rnd, "restore"))
            for i in range(a.warm):
                rid = b.order[i % len(b.order)]
                b.paired_wave(
                    order if i % 2 == 0 else order[::-1],
                    quals,
                    rnd,
                    "warm",
                    rid=rid,
                    seq0=i * a.concurrency,
                )
            print(f"r{rnd} warm: {a.warm} passes x {a.concurrency} per arm", flush=True)
            failed = False
        finally:
            failures = p1bench.delete_versions(b, published)
            if published:
                meta["t_versions_deleted"] = time.time()
            if failures:
                meta["cleanup_failures"] = failures
                cond["rounds_meta"].append(meta)
                json.dump(cond, open(path, "w"), indent=1)
                if not failed:
                    raise p1bench.CleanupError(f"round {rnd}: versions not deleted: {failures}")
        time.sleep(a.log_lag)
        t_end = time.time()
        meta["log_window"] = [max(t0, last_end), t_end]
        meta["log_events"] = b.save_logs(max(t0, last_end), t_end)
        last_end = t_end
        cond["rounds_meta"].append(meta)
        json.dump(cond, open(path, "w"), indent=1)
    cond["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(cond, open(path, "w"), indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
