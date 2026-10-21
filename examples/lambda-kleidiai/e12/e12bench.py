"""E12 benchmark: the dor arm (P1's full cache clear) against the e12 arm (metadata-only clear), interleaved.

  python e12bench.py --rounds 6 --concurrency 10 --warm 10 --read-timeout 300 --out build/runs/<id>

Each round, on new environments:
  1. set a new COLD_TAG on both arms' $LATEST (this retires every environment);
  2. cold: both arms' waves (`concurrency` simultaneous invokes of P1's probe each) start at the same moment;
  3. publish a version of each arm and wait for its SnapStart snapshot (P1's Bench.publish, sequential, in the
     round's arm order, which alternates every round);
  4. restore: both arms' restore waves start at the same moment, on the new versions;
  5. warm: `--warm` passes on the restored environments; in each pass both arms run the same request at the same
     moment (`concurrency` wide each), cycling through P1's fixed requests and then E12's stress requests;
  6. delete both versions (ending SnapStart cache billing), then save the round's logs.
Running the arms' waves together, rather than one arm after the other, puts both arms in the same minute of the
same round: time-within-round drift and the arms' CPU mix are shared. Each wave still gets its own new Lambda client
(P1's fix for the pooled-connection stall, I1), and every invoke has a `--read-timeout` bound.

Rows are P1's invocation rows (baseline/bench.py), so P1's summarize.py reads them.
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
import e12deploy  # noqa: E402
import summarize as p1sum  # noqa: E402


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
    """Alternate which arm is listed (and published) first, round by round."""
    return list(arms) if rnd % 2 == 0 else list(arms)[::-1]


def warm_ids(order: list[str], extra: list[str], passes: int) -> list[str]:
    """The request of each warm pass: P1's fixed requests first, then E12's stress requests, cycling."""
    ids = list(order) + [i for i in extra if i not in order]
    return [ids[i % len(ids)] for i in range(passes)]


def together(b: Bench, arms: list[str], fn) -> dict[str, list[dict]]:
    """Run fn(arm) for every arm at the same moment; one result list per arm."""
    with ThreadPoolExecutor(len(arms)) as ex:
        futs = {arm: ex.submit(fn, arm) for arm in arms}
        return {arm: f.result() for arm, f in futs.items()}


def merged_requests(p1_reqs: dict, extra: dict | None) -> dict:
    """P1's request set (with its probe) plus E12's stress requests. Ids must not collide."""
    out = {"probe": p1_reqs["probe"], "requests": list(p1_reqs["requests"])}
    if extra:
        seen = {r["id"] for r in out["requests"]}
        for r in extra["requests"]:
            if r["id"] in seen:
                raise SystemExit(f"duplicate request id {r['id']}")
            out["requests"].append(r)
    return out


def conditions(b: Bench, a, arms: list[str], reqs: dict, st: dict) -> dict:
    fns = {}
    for arm in arms:
        f = st["functions"][arm]
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
        env = dict(c["Environment"]["Variables"])
        fns[arm]["environment"] = {k: v for k, v in env.items() if k != "MODEL_BUCKET"}
        fns[arm]["runtime_management"] = {
            k: rm.get(k) for k in ("UpdateRuntimeOn", "RuntimeVersionArn")
        }
        if p1deploy.b64_to_hex(c["CodeSha256"]) != f["code_sha256"]:
            sys.exit(f"{arm}: deployed code {c['CodeSha256']} is not {f['code_sha256']}")
        if rm.get("RuntimeVersionArn") != st["runtime_version_arn"]:
            sys.exit(f"{arm}: runtime {rm.get('RuntimeVersionArn')} is not P1's")
        want = e12deploy.ARM_ENV[arm].get("DEC_CLEAR_DATA")
        if env.get("DEC_CLEAR_DATA") != want:
            sys.exit(f"{arm}: DEC_CLEAR_DATA={env.get('DEC_CLEAR_DATA')!r}, the arm needs {want!r}")
    # Apart from the arm label, the switch and COLD_TAG, the arms' environments must be identical.
    strip = ("COLD_TAG", "E12_ARM", "DEC_CLEAR_DATA")
    envs = {
        arm: {k: v for k, v in fns[arm]["environment"].items() if k not in strip} for arm in arms
    }
    if len({json.dumps(e, sort_keys=True) for e in envs.values()}) != 1:
        sys.exit(f"arm environments differ beyond DEC_CLEAR_DATA: {envs}")
    return {
        "schema": "lkai-conditions/1",
        "experiment": "E12",
        "region": a.region,
        "memory_mb": fns[arms[0]]["MemorySize"],
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": a.warm,
        "read_timeout_s": a.read_timeout,
        "arms": arms,
        "interleaving": "within round: both arms' cold waves start together, both restore waves start together, "
        "and each warm pass sends the same request to both arms at the same moment; the publish order (and the "
        "listed arm order) alternates by round",
        "requests_file": "baseline/requests.json + e12/requests.json",
        "requests": reqs,
        "probe": reqs["probe"],
        "functions": fns,
        "build_manifest": st["manifest"],
        "runtime_version_arn": st["runtime_version_arn"],
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
    ap.add_argument("--arms", default=",".join(e12deploy.ARMS))
    ap.add_argument("--rounds", type=p1bench.at_least(2), default=6)
    ap.add_argument("--concurrency", type=p1bench.positive_int, default=10)
    ap.add_argument("--warm", type=p1bench.positive_int, default=10)
    ap.add_argument("--read-timeout", type=p1bench.positive_int, default=900)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--requests", default=os.path.join(BASE, "requests.json"))
    ap.add_argument("--extra-requests", default=os.path.join(HERE, "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-lag", type=p1bench.nonnegative_int, default=20)
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required")
    arms = a.arms.split(",")
    if sorted(arms) != sorted(e12deploy.ARMS):
        sys.exit(f"arms must be exactly {list(e12deploy.ARMS)}")
    p1_reqs = json.load(open(a.requests))
    if p := p1sum.request_set_problem(p1_reqs):
        sys.exit(f"{a.requests}: {p}")
    reqs = merged_requests(p1_reqs, json.load(open(a.extra_requests)) if a.extra_requests else None)
    if p := p1sum.warm_problem(a.warm, p1_reqs):
        sys.exit(p)
    if a.region not in p1sum.PRICES_BY_REGION:
        sys.exit(f"no pinned prices for {a.region}")
    st = e12deploy.load_state()
    if st.get("region") != a.region or sorted(st.get("functions", {})) != sorted(arms):
        sys.exit("no E12 deployment; run e12deploy.py up")
    if st["manifest"].get("dirty_inputs"):
        sys.exit(
            f"refusing to benchmark: build inputs differ from the commit: {st['manifest']['dirty_inputs']}"
        )
    os.makedirs(a.out, exist_ok=True)
    if any(os.path.exists(os.path.join(a.out, f)) for f in ("invocations.jsonl", "logs")):
        sys.exit(f"{a.out} already holds a run; use a new --out")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    b = Bench(sess, st, reqs, a.out, a.concurrency)
    b.read_timeout = a.read_timeout
    cond = conditions(b, a, arms, reqs, st)
    path = os.path.join(a.out, "conditions.json")
    json.dump(cond, open(path, "w"), indent=1)
    p1_order = [r["id"] for r in p1_reqs["requests"]]
    passes = warm_ids(p1_order, [r["id"] for r in reqs["requests"]], a.warm)
    last_end = 0.0
    for rnd in range(a.rounds):
        tag = f"e12-r{rnd}-{time.time():.3f}"
        order = order_for(arms, rnd)
        t0 = time.time()
        meta: dict = {
            "round": rnd,
            "order": order,
            "cold_tag": tag,
            "t0": t0,
            "warm_requests": passes,
        }
        published: dict[str, dict] = {}
        failed = True
        try:
            together(b, order, lambda arm, tag=tag: b.set_cold_tag(arm, tag))
            cold = together(b, order, lambda arm, rnd=rnd: b.wave(arm, "$LATEST", rnd, "cold"))
            for arm in order:
                recs = cold[arm]
                print(
                    f"r{rnd} {arm} cold: ok {sum(not r['error'] for r in recs)}/{len(recs)}",
                    flush=True,
                )
            for arm in order:
                published[arm] = b.publish(arm, tag)
            meta["published"] = published
            rest = together(
                b,
                order,
                lambda arm, p=published, rnd=rnd: b.wave(arm, p[arm]["version"], rnd, "restore"),
            )
            for arm in order:
                recs = rest[arm]
                print(
                    f"r{rnd} {arm} restore v{published[arm]['version']}: ok "
                    f"{sum(not r['error'] for r in recs)}/{len(recs)}",
                    flush=True,
                )
            for i, rid in enumerate(passes):
                together(
                    b,
                    order,
                    lambda arm, rid=rid, i=i, p=published, rnd=rnd: b.wave(
                        arm,
                        p[arm]["version"],
                        rnd,
                        "warm",
                        rid=rid,
                        seq0=(i + 1) * a.concurrency,
                    ),
                )
            print(f"r{rnd} warm: {len(passes)} passes x {a.concurrency} per arm", flush=True)
            failed = False
        finally:
            failures = p1bench.delete_versions(b, published)
            if published:
                meta["t_versions_deleted"] = time.time()
            if failures:
                meta["cleanup_failures"] = failures
                print("VERSION CLEANUP FAILED:", failures, file=sys.stderr)
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
