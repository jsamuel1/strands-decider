"""E3 benchmark: every download strategy against the control, interleaved, on new environments.

  python e3bench.py --rounds 5 --concurrency 10 --out build/runs/<run-id>                 # cold sweep, all arms
  python e3bench.py --mode restore --arms ctl,<best> --rounds 3 --out build/runs/<run-id>   # small restore leg
  python e3bench.py --mode verify --out build/runs/<run-id>                               # one hashed cold per arm

Reads build/deployment.json (e3deploy.py) and P1's requests.json, and invokes through P1's harness (baseline/bench.py
Bench: same invoke, REPORT parsing, start-class check, row format and log capture), so rows are comparable with P1's.

cold mode, per round:
  1. set a new COLD_TAG on every arm's $LATEST (retires its environments), all arms at once
  2. for each arm, in an order rotated by one every round (so no arm always goes first or last): `concurrency`
     simultaneous cold invokes with P1's probe request, then one warm pass of `concurrency` invokes on the same
     environments (one request per round, cycling P1's five), the steady-state check that compute is unchanged
restore mode, per round, for each arm in alternating order: set COLD_TAG, publish a version and wait for its snapshot
  (P1's Bench.publish), `concurrency` restore invokes, one warm pass, then delete the version. The weights are
  downloaded only by the snapshot init inside publish_version; its phases come from the saved CloudWatch logs.
verify mode: COLD_TAG and E3_VERIFY=1 on every E3 arm, one cold invoke each; the LKAI record then carries the sha256
  of the memfd the strategy filled (not a measured invocation: the hash adds seconds). Then E3_VERIFY is removed.
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

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import bench as p1bench  # noqa: E402  (baseline/bench.py)
import deploy as p1deploy  # noqa: E402
import e3deploy  # noqa: E402


def set_env(b: p1bench.Bench, arm: str, **kv: str | None) -> None:
    """Update $LATEST's environment (a None value removes the key) and wait for the update."""
    name = b.st["functions"][arm]["name"]
    c = b.lam.get_function_configuration(FunctionName=name)
    env = dict(c["Environment"]["Variables"])
    for k, v in kv.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    b.lam.update_function_configuration(FunctionName=name, Environment={"Variables": env})
    p1deploy.wait_updated(b.lam, name)


def rotated(arms: list[str], rnd: int) -> list[str]:
    k = rnd % len(arms)
    return arms[k:] + arms[:k]


def conditions(b, sess, a, arms, reqs) -> dict:
    lam = b.lam
    fns = {}
    for arm in arms:
        f = b.st["functions"][arm]
        c = lam.get_function_configuration(FunctionName=f["name"])
        rm = lam.get_runtime_management_config(FunctionName=f["name"])
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
            sys.exit(
                f"{arm}: runtime {rm.get('RuntimeVersionArn')} is not P1's {b.st['runtime_version_arn']}"
            )
    return {
        "schema": "lkai-conditions/1",
        "experiment": "E3",
        "mode": a.mode,
        "region": a.region,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": 1 if a.mode != "verify" else 0,
        "arms": arms,
        "e3_dl": {arm: b.st["functions"][arm]["e3_dl"] for arm in arms},
        "requests_file": "baseline/requests.json",
        "requests": reqs,
        "probe": reqs["probe"],
        "functions": fns,
        "build_manifest": b.st["manifest"],
        "p1_zip_sha256": b.st["p1_zip_sha256"],
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
    ap.add_argument("--mode", choices=["cold", "restore", "verify"], default="cold")
    ap.add_argument("--arms", default=",".join(e3deploy.ARMS))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=int, default=20)
    a = ap.parse_args()

    st = e3deploy.load_state()
    if st.get("region") != a.region or not st.get("functions"):
        sys.exit("no E3 deployment; run e3deploy.py up")
    arms = a.arms.split(",")
    if a.mode == "verify":
        arms = [x for x in arms if st["functions"][x]["e3_dl"]]
        a.rounds, a.concurrency = 1, 1
    reqs = json.load(open(a.requests))
    os.makedirs(a.out, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    b = p1bench.Bench(sess, st, reqs, a.out, max(a.concurrency, 1))
    cond = conditions(b, sess, a, arms, reqs)
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)
    last_end = 0.0
    for rnd in range(a.rounds):
        tag = f"e3-{a.mode}-r{rnd}-{time.time():.3f}"
        order = (
            rotated(arms, rnd) if a.mode != "restore" else (arms if rnd % 2 == 0 else arms[::-1])
        )
        t0 = time.time()
        meta: dict = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        warm_rid = b.order[rnd % len(b.order)]
        try:
            extra = {"E3_VERIFY": "1"} if a.mode == "verify" else {}
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, t=tag, e=extra: set_env(b, arm, COLD_TAG=t, **e), order))
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
                if a.mode != "verify":
                    b.wave(arm, q, rnd, "warm", rid=warm_rid)
                if arm in published:
                    b.delete_version(arm, published[arm]["version"])
                    published[arm]["t_deleted"] = time.time()
        finally:
            for arm, pub in published.items():
                if "t_deleted" not in pub:
                    b.delete_version(arm, pub["version"])
            if a.mode == "verify":
                with ThreadPoolExecutor(len(order)) as ex:
                    list(ex.map(lambda arm: set_env(b, arm, E3_VERIFY=None), order))
        if published:
            meta["published"] = published
            meta["t_versions_deleted"] = max(
                p.get("t_deleted", time.time()) for p in published.values()
            )
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
