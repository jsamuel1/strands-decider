"""E6 benchmark: ctl against pyc, interleaved, on new environments. Rows are P1's (baseline/bench.py Bench).

  python e6bench.py --mode cold    --rounds 6 --concurrency 10 --out build/runs/cold-<id>
  python e6bench.py --mode restore --rounds 3 --concurrency 10 --out build/runs/restore-<id>

cold mode, per round:
  1. set a new COLD_TAG on both arms' $LATEST (this retires every environment). Both arms are updated together.
  2. for each arm, in an order that alternates every round: `concurrency` simultaneous cold invokes of P1's probe,
     then one warm pass per fixed request (`concurrency` wide, the five requests in P1's order) on those
     environments. The warm passes answer every request on every arm, for the correctness check.
restore mode, per round, for each arm in alternating order: set COLD_TAG, publish a version and wait for its
  snapshot (P1's Bench.publish), `concurrency` restore invokes, one warm pass, then delete the version. Only the
  snapshot init imports the code, so its `imports` phase comes from the saved logs (before_snapshot records).

--read-timeout bounds one invoke (default 900 s, P1's). P3 saw first restore waves stall with no RESTORE_START, so
that a stall costs N seconds rather than 15 minutes. The value is recorded in conditions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import bench as p1bench  # noqa: E402  (baseline/bench.py)
import deploy as p1deploy  # noqa: E402
import e6deploy  # noqa: E402


class Bench(p1bench.Bench):
    """P1's Bench with a configurable per-invoke read timeout (its other settings unchanged)."""

    read_timeout = 900

    def new_invoke_client(self):
        self.invoke_clients += 1
        return self.sess.client(
            "lambda",
            config=Config(
                read_timeout=self.read_timeout,
                connect_timeout=10,
                retries={"max_attempts": 0},
                max_pool_connections=max(16, self.concurrency * 2),
            ),
        )


def order_for(arms: list[str], rnd: int) -> list[str]:
    """Alternate which arm goes first, round by round."""
    return list(arms) if rnd % 2 == 0 else list(arms)[::-1]


def set_tag(b: Bench, arm: str, tag: str) -> None:
    name = b.st["functions"][arm]["name"]
    c = b.ctl.get_function_configuration(FunctionName=name)
    env = dict(c["Environment"]["Variables"], COLD_TAG=tag)
    b.ctl.update_function_configuration(FunctionName=name, Environment={"Variables": env})
    p1deploy.wait_updated(b.ctl, name)


def conditions(b: Bench, a, arms: list[str], reqs: dict) -> dict:
    fns = {}
    for arm in arms:
        f = b.st["functions"][arm]
        c = b.ctl.get_function_configuration(FunctionName=f["name"])
        rm = b.ctl.get_runtime_management_config(FunctionName=f["name"])
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
        fns[arm]["environment"] = {
            k: v for k, v in c["Environment"]["Variables"].items() if k != "MODEL_BUCKET"
        }
        fns[arm]["runtime_management"] = {
            k: rm.get(k) for k in ("UpdateRuntimeOn", "RuntimeVersionArn")
        }
        if p1deploy.b64_to_hex(c["CodeSha256"]) != f["code_sha256"]:
            sys.exit(f"{arm}: deployed code {c['CodeSha256']} is not {f['code_sha256']}")
        if rm.get("RuntimeVersionArn") != b.st["runtime_version_arn"]:
            sys.exit(f"{arm}: runtime {rm.get('RuntimeVersionArn')} is not P1's")
    return {
        "schema": "lkai-conditions/1",
        "experiment": "E6",
        "mode": a.mode,
        "region": a.region,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": len(b.order) if a.mode == "cold" else 1,
        "read_timeout_s": a.read_timeout,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "restore"], default="cold")
    ap.add_argument("--arms", default=",".join(e6deploy.ARMS))
    ap.add_argument("--rounds", type=p1bench.at_least(2), default=6)
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
    st = e6deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E6 deployment; run e6deploy.py up")
    arms = a.arms.split(",")
    reqs = json.load(open(a.requests))
    os.makedirs(a.out, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    b = Bench(sess, st, reqs, a.out, a.concurrency)
    b.read_timeout = a.read_timeout
    b.ctl = sess.client(
        "lambda", config=Config(retries={"mode": "standard", "max_attempts": 8}, read_timeout=60)
    )
    cond = conditions(b, a, arms, reqs)
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)
    last_end = 0.0
    for rnd in range(a.rounds):
        tag = f"e6-{a.mode}-r{rnd}-{time.time():.3f}"
        order = order_for(arms, rnd)
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, t=tag: set_tag(b, arm, t), order))
            for arm in order:
                if a.mode == "restore":
                    published[arm] = b.publish(arm, tag)
                    q, wave = published[arm]["version"], "restore"
                else:
                    q, wave = "$LATEST", "cold"
                recs = b.wave(arm, q, rnd, wave)
                ok = [r for r in recs if not r["error"]]
                print(
                    f"r{rnd} {arm} {wave}: ok {len(ok)}/{len(recs)} rt median "
                    f"{sorted(r['rt_s'] for r in recs)[len(recs) // 2]:.2f}s",
                    flush=True,
                )
                warm_ids = b.order if a.mode == "cold" else [b.order[rnd % len(b.order)]]
                for i, rid in enumerate(warm_ids):
                    b.wave(arm, q, rnd, "warm", rid=rid, seq0=(i + 1) * a.concurrency)
                if arm in published:
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
