"""E14 benchmark: ctl against rows on Lambda. Warm inference is the targeted phase.

  python e14bench.py --mode cold    --rounds 5 --concurrency 10 --warm-passes 4 --out build/runs/<id>/cold
  python e14bench.py --mode restore --rounds 2 --concurrency 10 --warm-passes 2 --out build/runs/<id>/restore

Per round:
  1. set a new COLD_TAG on both arms' $LATEST (retires every environment), both arms together.
  2. restore mode only: publish a version of each arm and wait for its snapshot (P1's Bench.publish).
  3. one cold (or restore) wave of `concurrency` probe invokes per arm, the arm order alternating every round.
  4. `warm-passes` passes over P1's five fixed requests on those environments. Within a pass the arms are
     INTERLEAVED request by request (ctl r1, rows r1, ctl r2, rows r2, ... with the leading arm alternating per pass),
     so each warm comparison is made seconds apart on environments created in the same round.
  5. restore mode: delete both versions (also in `finally`).
Warm rows therefore come from cold-origin environments in cold mode and restore-origin ones in restore mode.

Every wave uses a new invoke client (baseline/bench.py at I1's 5764b9e). --read-timeout bounds one invoke.
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
import e14deploy  # noqa: E402


def at_least(n: int):
    def f(v: str) -> int:
        i = int(v)
        if i < n:
            raise argparse.ArgumentTypeError(f"must be >= {n}")
        return i

    return f


def delete_versions(b, published: dict[str, dict]) -> list[str]:
    """Delete every listed version, attempting each even when an earlier one fails; return the failures."""
    failures = []
    for arm, pub in published.items():
        try:
            b.delete_version(arm, pub["version"])
        except Exception as e:  # the next version must still be deleted
            failures.append(f"{arm} v{pub['version']}: {e}")
    return failures


class Bench(p1bench.Bench):
    """P1's Bench with a configurable per-invoke read timeout."""

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


def order_for(arms: list[str], i: int) -> list[str]:
    """Alternate which arm goes first."""
    return list(arms) if i % 2 == 0 else list(arms)[::-1]


def warm_schedule(arms: list[str], requests: list[str], passes: int) -> list[tuple[int, str, str]]:
    """(pass, arm, request) in run order: request by request, arms interleaved, leading arm alternating per pass."""
    out = []
    for p in range(passes):
        for rid in requests:
            for arm in order_for(arms, p):
                out.append((p, arm, rid))
    return out


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
        "experiment": "E14",
        "mode": a.mode,
        "region": a.region,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_passes": a.warm_passes,
        "warm_per_arm_per_round": a.warm_passes * len(b.order),
        "warm_order": "per request, arms interleaved, leading arm alternating per pass",
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
        "invoke_client": "new client per wave (baseline/bench.py at I1 5764b9e)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds_meta": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "restore"], default="cold")
    ap.add_argument("--arms", default=",".join(e14deploy.ARMS))
    ap.add_argument("--rounds", type=at_least(2), default=5)
    ap.add_argument("--concurrency", type=at_least(1), default=10)
    ap.add_argument("--warm-passes", type=at_least(1), default=4)
    ap.add_argument("--read-timeout", type=at_least(1), default=900)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=at_least(0), default=20)
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required")
    st = e14deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E14 deployment; run e14deploy.py up")
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
        tag = f"e14-{a.mode}-r{rnd}-{time.time():.3f}"
        order = order_for(arms, rnd)
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, t=tag: set_tag(b, arm, t), order))
            quals: dict[str, str] = {}
            for arm in order:
                if a.mode == "restore":
                    # Each version is recorded before the next operation can fail, so finally deletes it.
                    published[arm] = b.publish(arm, tag)
                    quals[arm], wave = published[arm]["version"], "restore"
                else:
                    quals[arm], wave = "$LATEST", "cold"
            for arm in order:
                wave = "restore" if a.mode == "restore" else "cold"
                recs = b.wave(arm, quals[arm], rnd, wave)
                ok = [r for r in recs if not r["error"]]
                print(
                    f"r{rnd} {arm} {wave}: ok {len(ok)}/{len(recs)} rt median "
                    f"{sorted(r['rt_s'] for r in recs)[len(recs) // 2]:.2f}s",
                    flush=True,
                )
            sched = warm_schedule(arms, b.order, a.warm_passes)
            meta["warm_schedule"] = [list(s) for s in sched]
            for i, (_p, arm, rid) in enumerate(sched):
                b.wave(arm, quals[arm], rnd, "warm", rid=rid, seq0=(i + 1) * a.concurrency)
            for arm in list(published):
                b.delete_version(arm, published[arm]["version"])
                published[arm]["t_deleted"] = time.time()
        finally:
            failures = delete_versions(
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
    cond["invoke_clients_made"] = b.invoke_clients
    json.dump(cond, open(path, "w"), indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
