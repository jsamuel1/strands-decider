"""E13 benchmark: four OpenMP wait-policy arms, arm order rotated by round and, for warm, by pass.

  python e13bench.py --mode cold    --rounds 6 --concurrency 10 --warm-reps 3 --out build/runs/cold-<id>
  python e13bench.py --mode restore --rounds 3 --concurrency 10 --warm-reps 1 --out build/runs/restore-<id>

cold mode, per round:
  1. set a new COLD_TAG on every arm's $LATEST (this retires every environment). All arms are updated together.
  2. cold: for each arm, in the round's order, `concurrency` simultaneous cold invokes of P1's probe request.
  3. warm: `warm-reps` x 5 passes (P1's five fixed requests, in P1's order, repeated). Each pass sends one request to
     every arm, one arm-wave of `concurrency` simultaneous invokes after another, so they reuse that arm's
     environments. The arm order rotates every pass. This interleaves the arms within the round, which is what the
     primary (warm) comparison needs.
restore mode, per round: set COLD_TAG on every arm and publish one version each (in parallel), wait for every
  snapshot, then the restore waves in the round's order, then `warm-reps` x 5 interleaved warm passes on the restored
  environments, then delete the versions.

Round order: a cyclic rotation of the arm list by round number (every arm in every position once per len(arms)
rounds), reversed in alternate cycles. The cold and restore waves of a round
are sequential waves (alternated first position across rounds), not interleaved within the round; that is stated as
a limitation of those side-effect metrics. Every wave invokes through a new client (P1/I1), and --read-timeout
bounds a stalled invoke.
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
import e13deploy  # noqa: E402


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


def round_order(arms: list[str], rnd: int) -> list[str]:
    """The arms rotated by the round number (a cyclic Latin square: over any len(arms) consecutive rounds every arm
    takes every position once). Each later cycle of len(arms) rounds runs reversed, so who-follows-whom also changes."""
    n = len(arms)
    base = list(arms)[::-1] if (rnd // n) % 2 else list(arms)
    k = rnd % n
    return base[k:] + base[:k]


def pass_order(arms: list[str], rnd: int, i: int) -> list[str]:
    """The round's order rotated by the warm pass index: within one round each arm is first equally often."""
    base = round_order(arms, rnd)
    k = i % len(base)
    return base[k:] + base[:k]


def warm_plan(order_ids: list[str], reps: int) -> list[str]:
    """The warm passes of one round: P1's fixed requests in P1's order, `reps` times."""
    return [rid for _ in range(reps) for rid in order_ids]


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
        env = {k: v for k, v in c["Environment"]["Variables"].items() if k != "MODEL_BUCKET"}
        fns[arm]["environment"] = env
        fns[arm]["runtime_management"] = {
            k: rm.get(k) for k in ("UpdateRuntimeOn", "RuntimeVersionArn")
        }
        if p1deploy.b64_to_hex(c["CodeSha256"]) != f["code_sha256"]:
            sys.exit(f"{arm}: deployed code {c['CodeSha256']} is not {f['code_sha256']}")
        if rm.get("RuntimeVersionArn") != b.st["runtime_version_arn"]:
            sys.exit(f"{arm}: runtime {rm.get('RuntimeVersionArn')} is not P1's")
        want = e13deploy.arm_environment(b.st["p1_environment"], arm)
        got = {k: v for k, v in c["Environment"]["Variables"].items() if k != "COLD_TAG"}
        if got != {k: v for k, v in want.items() if k != "COLD_TAG"}:
            sys.exit(f"{arm}: deployed environment differs from e13deploy.arm_environment")
    return {
        "schema": "lkai-conditions/1",
        "experiment": "E13",
        "mode": a.mode,
        "region": a.region,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": len(warm_plan(b.order, a.warm_reps)),
        "warm_reps": a.warm_reps,
        "read_timeout_s": a.read_timeout,
        "arms": arms,
        "arm_variables": {arm: e13deploy.ARMS[arm] for arm in arms},
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
    ap.add_argument("--arms", default=",".join(e13deploy.ARMS))
    ap.add_argument("--rounds", type=p1bench.at_least(2), default=6)
    ap.add_argument("--concurrency", type=p1bench.positive_int, default=10)
    ap.add_argument("--warm-reps", type=p1bench.positive_int, default=3)
    ap.add_argument("--read-timeout", type=p1bench.positive_int, default=900)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=p1bench.nonnegative_int, default=20)
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required")
    st = e13deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E13 deployment; run e13deploy.py up")
    arms = a.arms.split(",")
    if unknown := sorted(set(arms) - set(e13deploy.ARMS)):
        sys.exit(f"unknown arms {unknown}")
    reqs = json.load(open(a.requests))
    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(os.path.join(a.out, f)) for f in ("invocations.jsonl", "logs")):
        sys.exit(f"{a.out} already holds a run; use a new --out")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    b = Bench(sess, st, reqs, a.out, a.concurrency)
    b.read_timeout = a.read_timeout
    b.ctl = sess.client(
        "lambda", config=Config(retries={"mode": "standard", "max_attempts": 8}, read_timeout=60)
    )
    cond = conditions(b, a, arms, reqs)
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)
    plan = warm_plan(b.order, a.warm_reps)
    last_end = 0.0
    for rnd in range(a.rounds):
        tag = f"e13-{a.mode}-r{rnd}-{time.time():.3f}"
        order = round_order(arms, rnd)
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0, "warm_orders": []}
        published: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, t=tag: set_tag(b, arm, t), order))
            if a.mode == "restore":
                with ThreadPoolExecutor(len(order)) as ex:
                    for arm, pub in zip(
                        order, ex.map(lambda arm, t=tag: b.publish(arm, t), order), strict=True
                    ):
                        published[arm] = pub
                meta["published"] = published
            wave = "restore" if a.mode == "restore" else "cold"
            for arm in order:
                q = published[arm]["version"] if arm in published else "$LATEST"
                recs = b.wave(arm, q, rnd, wave)
                ok = [r for r in recs if not r["error"]]
                print(
                    f"r{rnd} {arm} {wave}: ok {len(ok)}/{len(recs)} rt median "
                    f"{sorted(r['rt_s'] for r in recs)[len(recs) // 2]:.2f}s",
                    flush=True,
                )
            for i, rid in enumerate(plan):
                po = pass_order(arms, rnd, i)
                meta["warm_orders"].append(po)
                for arm in po:
                    q = published[arm]["version"] if arm in published else "$LATEST"
                    b.wave(arm, q, rnd, "warm", rid=rid, seq0=(i + 1) * a.concurrency)
            print(
                f"r{rnd} warm: {len(plan)} passes x {a.concurrency} per arm, interleaved",
                flush=True,
            )
        finally:
            failures = p1bench.delete_versions(b, published)
            if published:
                meta["t_versions_deleted"] = time.time()
            if failures:
                meta["cleanup_failures"] = failures
                print("VERSION CLEANUP FAILED:", failures, file=sys.stderr)
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
