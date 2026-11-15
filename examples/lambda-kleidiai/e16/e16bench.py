"""E16 benchmark: e6 against lazy, interleaved within every wave, on new environments.

  python e16bench.py --mode cold    --rounds 8 --concurrency 10 --out build/runs/cold-<id>
  python e16bench.py --mode restore --rounds 3 --concurrency 10 --out build/runs/restore-<id>

Every wave runs BOTH arms at once: `concurrency` invokes per arm, submitted alternately (e6, lazy, e6, ... with the
first arm alternating by round) into one pool, each arm through its own fresh invoke client (I1: no connection pooled
before a wait). So the two arms share each round's time window, S3 and Lambda conditions, rather than running as
sequential waves (E6's design).

cold mode, per round: set a new COLD_TAG on both arms' $LATEST (this retires every environment), then one cold wave of
P1's probe, then one warm wave per fixed request (the five requests in P1's order) on those environments.
restore mode, per round: set COLD_TAG, publish a version of each arm and wait for both snapshots (P1's Bench.publish),
one restore wave, one warm wave, then delete both versions. Only the snapshot init imports the code, so its phases come
from the saved logs (before_snapshot records).

--read-timeout bounds one invoke (default 900 s, P1's). It is recorded in conditions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
KLEIDIAI = os.path.dirname(HERE)
BASE = os.path.join(KLEIDIAI, "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(KLEIDIAI, "e6"))
sys.path.insert(0, HERE)
import bench as p1bench  # noqa: E402  (baseline/bench.py)
import e6bench  # noqa: E402  (its Bench subclass with --read-timeout, set_tag, conditions)
import e16deploy  # noqa: E402


def order_for(arms: list[str], rnd: int) -> list[str]:
    """The arm submitted first alternates round by round."""
    return list(arms) if rnd % 2 == 0 else list(arms)[::-1]


def submission(arms: list[str], concurrency: int, rnd: int) -> list[tuple[str, int]]:
    """(arm, i) in submission order: the arms alternate invoke by invoke, the first arm alternating by round."""
    order = order_for(arms, rnd)
    return [(arm, i) for i in range(concurrency) for arm in order]


def joint_wave(
    b: e6bench.Bench,
    arms: list[str],
    qualifiers: dict[str, str],
    rnd: int,
    wave: str,
    rid: str | None = None,
    seq0: int = 0,
) -> list[dict]:
    """One wave of both arms at once. Each arm has its own fresh invoke client."""
    clients = {arm: b.new_invoke_client() for arm in arms}
    plan = submission(arms, b.concurrency, rnd)
    with ThreadPoolExecutor(len(plan)) as ex:
        futs = [
            ex.submit(
                b.invoke, arm, qualifiers[arm], rid or b.probe, rnd, wave, seq0 + i, clients[arm]
            )
            for arm, i in plan
        ]
        return [f.result() for f in futs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "restore"], default="cold")
    ap.add_argument("--arms", default=",".join(e16deploy.ARMS))
    ap.add_argument("--rounds", type=p1bench.at_least(2), default=8)
    ap.add_argument("--concurrency", type=p1bench.positive_int, default=10)
    ap.add_argument("--read-timeout", type=p1bench.positive_int, default=900)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=p1bench.nonnegative_int, default=20)
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required")
    st = e16deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E16 deployment; run e16deploy.py up")
    arms = a.arms.split(",")
    reqs = json.load(open(a.requests))
    os.makedirs(a.out, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    acct = sess.client("sts").get_caller_identity()["Account"]
    if acct != st.get("account"):
        sys.exit(f"profile reaches {acct}, the deployment is in {st.get('account')}")
    b = e6bench.Bench(sess, st, reqs, a.out, a.concurrency)
    b.read_timeout = a.read_timeout
    b.ctl = sess.client(
        "lambda", config=Config(retries={"mode": "standard", "max_attempts": 8}, read_timeout=60)
    )
    cond = e6bench.conditions(b, a, arms, reqs)
    cond["experiment"] = "E16"
    cond["interleaving"] = (
        "both arms in every wave: invokes submitted alternately into one pool, first arm alternating by round"
    )
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)
    last_end = 0.0
    for rnd in range(a.rounds):
        tag = f"e16-{a.mode}-r{rnd}-{time.time():.3f}"
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order_for(arms, rnd), "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(len(arms)) as ex:
                list(ex.map(lambda arm, t=tag: e6bench.set_tag(b, arm, t), arms))
            if a.mode == "restore":
                with ThreadPoolExecutor(len(arms)) as ex:
                    for arm, pub in zip(
                        arms, ex.map(lambda arm, t=tag: b.publish(arm, t), arms), strict=True
                    ):
                        published[arm] = pub
                qual, wave = {arm: published[arm]["version"] for arm in arms}, "restore"
            else:
                qual, wave = dict.fromkeys(arms, "$LATEST"), "cold"
            recs = joint_wave(b, arms, qual, rnd, wave)
            for arm in arms:
                rs = [r for r in recs if r["arm"] == arm]
                ok = [r for r in rs if not r["error"]]
                print(
                    f"r{rnd} {arm} {wave}: ok {len(ok)}/{len(rs)} rt median "
                    f"{sorted(r['rt_s'] for r in rs)[len(rs) // 2]:.2f}s",
                    flush=True,
                )
            warm_ids = b.order if a.mode == "cold" else [b.order[rnd % len(b.order)]]
            for i, rid in enumerate(warm_ids):
                joint_wave(b, arms, qual, rnd, "warm", rid=rid, seq0=(i + 1) * a.concurrency)
            for arm in list(published):
                b.delete_version(arm, published[arm]["version"])
                published[arm]["t_deleted"] = time.time()
        finally:
            failures = p1bench.delete_versions(
                b, {k: v for k, v in published.items() if "t_deleted" not in v}
            )
            for k in published:
                published[k].setdefault("t_deleted", time.time())
            if failures:
                print("VERSION CLEANUP FAILED:", failures, file=sys.stderr)
        if published:
            meta["published"] = published
            meta["t_versions_deleted"] = max(p["t_deleted"] for p in published.values())
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
