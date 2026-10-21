"""Benchmark harness for the lambda-kleidiai baseline. Runs from any machine with credentials for the account.

  python bench.py --build build/ --rounds 5 --concurrency 10 --warm 10 --out results/baseline/<run-id>

Reads build/deployment.json (deploy.py) and requests.json. Each round, for both functions (the instrumented arm and
the REPORT-only arm), in an order that alternates between rounds:
  1. set COLD_TAG on $LATEST (a configuration change retires every $LATEST execution environment)
  2. cold     `concurrency` simultaneous invokes of $LATEST: each needs a new on-demand environment
  3. publish  a version from that configuration and wait until its SnapStart snapshot is ready
  4. restore  `concurrency` simultaneous invokes of the new version: each restores a new environment
  5. warm     `warm` passes per arm on that version, each `concurrency` simultaneous invokes of one request (so they
              reuse every restored environment), the arms alternating pass by pass and the request cycling
  6. delete both versions (a SnapStart version's cache is billed while it exists)
Cold and restore waves send requests.json's probe request, so first-inference latency is comparable; warm cycles every
request. Every invocation's expected start class is checked against its REPORT line (Init Duration = cold, Restore
Duration = restore, neither = warm) and a mismatch is recorded, never silently relabelled. After each round the
function log groups for its time window are saved verbatim.

Writes to --out: invocations.jsonl (one row per invocation: timing, REPORT fields, LKAI phase record, answers,
log tail), logs/<function>.jsonl.gz (CloudWatch events), conditions.json (everything pinned). summarize.py turns them
into results.json.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import platform
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deploy  # noqa: E402
import phases  # noqa: E402

_lock = threading.Lock()
observed_class = phases.observed_class


class Bench:
    """Two kinds of Lambda client (I1, examples/lambda-kleidiai/baseline/restore_stall/README.md).

    `lam` makes the lifecycle calls: configuration, publish, readiness polls, deletes. They are idempotent, so it
    has a short read timeout and standard retries. Every cold, restore and warm wave invokes through a NEW client
    from new_invoke_client(), so no measured invoke goes out on a connection pooled before a publish/SnapStart wait
    or by an earlier wave. Reusing one client for everything let a request be written into a pooled connection
    that had idled through the publish phase (P3: 1074 to 1112 s): the bytes were never acknowledged, the request
    never reached Lambda (no START, Invocations short by the same count) and it ended in a client read timeout.
    A fresh client changes only the client round trip (one TLS handshake per connection, equal on every arm), never
    Lambda's REPORT or the function's own timings."""

    def __init__(self, sess, st: dict, reqs: dict, out: str, concurrency: int):
        self.st, self.out, self.sess = st, out, sess
        self.concurrency = concurrency
        self.invoke_clients = 0
        self.lam = sess.client(
            "lambda",
            config=Config(
                read_timeout=60,
                connect_timeout=10,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        self.logs = sess.client("logs")
        self.reqs = {r["id"]: r["request"] for r in reqs["requests"]}
        self.order = [r["id"] for r in reqs["requests"]]
        self.probe = reqs["probe"]
        self.fh = open(os.path.join(out, "invocations.jsonl"), "a")

    def new_invoke_client(self):
        """A fresh client for one wave: no retries (a retry would hide a failed start), the function's 900 s
        timeout, and a pool as wide as the wave."""
        self.invoke_clients += 1
        return self.sess.client(
            "lambda",
            config=Config(
                read_timeout=900,
                connect_timeout=10,
                retries={"max_attempts": 0},
                max_pool_connections=max(16, self.concurrency * 2),
            ),
        )

    def write(self, rec: dict) -> None:
        with _lock:
            self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.fh.flush()

    def invoke(
        self, arm: str, qualifier: str, rid: str, rnd: int, wave: str, seq: int, client=None
    ) -> dict:
        name = self.st["functions"][arm]["name"]
        t_wall = time.time()
        t = time.perf_counter()
        try:
            r = (client or self.new_invoke_client()).invoke(
                FunctionName=name,
                Qualifier=qualifier,
                LogType="Tail",
                Payload=json.dumps(self.reqs[rid]).encode(),
            )
            body = r["Payload"].read()
            err, status = r.get("FunctionError"), r["StatusCode"]
            tail = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
            exec_version = r.get("ExecutedVersion")
        except Exception as e:  # a transport failure is a result too
            body, err, status, tail, exec_version = (
                str(e).encode(),
                f"client:{type(e).__name__}",
                None,
                "",
                None,
            )
        rt = time.perf_counter() - t
        rep = phases.parse_report(tail)
        rec = {
            "arm": arm,
            "round": rnd,
            "wave": wave,
            "seq": seq,
            "request_id": rid,
            "qualifier": qualifier,
            "executed_version": exec_version,
            "t_start": t_wall,
            "rt_s": round(rt, 6),
            "status": status,
            "error": err,
            "report": rep,
            "expected_class": "warm" if wave == "warm" else wave,
            "observed_class": observed_class(rep),
            "log_tail": tail,
        }
        m = re.findall(r"cpu_part=(\S+)", tail)
        rec["cpu_part"] = m[-1] if m else None
        try:
            p = json.loads(body)
        except ValueError:
            p = {"raw": body[:2000].decode(errors="replace")}
        if err:
            rec["payload"] = p
        else:
            rec["answers"] = p.get("answers")
            rec["latency_ms"] = p.get("latency_ms")
            rec["usage"] = p.get("usage")
            rec["timing"] = p.get("timing")
            if "lkai" in p:
                rec["lkai"] = p["lkai"]
                rec["cpu_part"] = p["lkai"]["inv"].get("cpu_part") or rec["cpu_part"]
            elif p.get("timing"):
                rec["cpu_part"] = p["timing"].get("cpu") or rec["cpu_part"]
        rec["class_mismatch"] = rec["observed_class"] != rec["expected_class"]
        self.write(rec)
        return rec

    def wave(
        self, arm: str, qualifier: str, rnd: int, wave: str, rid: str | None = None, seq0: int = 0
    ) -> list[dict]:
        # One fresh pool per cold/restore/warm wave (see the class docstring).
        lam = self.new_invoke_client()
        with ThreadPoolExecutor(self.concurrency) as ex:
            futs = [
                ex.submit(self.invoke, arm, qualifier, rid or self.probe, rnd, wave, seq0 + i, lam)
                for i in range(self.concurrency)
            ]
            return [f.result() for f in futs]

    def set_cold_tag(self, arm: str, tag: str) -> None:
        name = self.st["functions"][arm]["name"]
        c = self.lam.get_function_configuration(FunctionName=name)
        env = dict(c["Environment"]["Variables"], COLD_TAG=tag)
        self.lam.update_function_configuration(FunctionName=name, Environment={"Variables": env})
        deploy.wait_updated(self.lam, name)

    def publish(self, arm: str, tag: str) -> dict:
        name = self.st["functions"][arm]["name"]
        t = time.time()
        v = self.lam.publish_version(FunctionName=name, Description=f"COLD_TAG {tag}")["Version"]
        # wait_updated can sys.exit, and the caller never learns v, so delete the version here.
        try:
            c = deploy.wait_updated(self.lam, name, v)
        except BaseException:
            self.delete_version(arm, v)
            raise
        return {
            "arm": arm,
            "version": v,
            "publish_s": round(time.time() - t, 1),
            "t_publish": t,
            "snapstart": c.get("SnapStart"),
            "code_sha256_b64": c["CodeSha256"],
            "runtime_version": (c.get("RuntimeVersionConfig") or {}).get("RuntimeVersionArn"),
        }

    def delete_version(self, arm: str, v: str) -> None:
        self.lam.delete_function(FunctionName=self.st["functions"][arm]["name"], Qualifier=v)

    def save_logs(self, t0: float, t1: float) -> int:
        n = 0
        os.makedirs(os.path.join(self.out, "logs"), exist_ok=True)
        for f in self.st["functions"].values():
            path = os.path.join(self.out, "logs", f"{f['name']}.jsonl.gz")
            with gzip.open(path, "at") as fh:
                for page in self.logs.get_paginator("filter_log_events").paginate(
                    logGroupName=f"/aws/lambda/{f['name']}",
                    startTime=int(t0 * 1000),
                    endTime=int(t1 * 1000),
                ):
                    for e in page["events"]:
                        fh.write(
                            json.dumps({k: e[k] for k in ("timestamp", "logStreamName", "message")})
                            + "\n"
                        )
                        n += 1
        return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--requests", default=os.path.join(HERE, "requests.json"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument(
        "--warm",
        type=int,
        default=10,
        help="warm passes per arm per round (each `concurrency` wide)",
    )
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--arms", default="inst,rep")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--log-lag", type=int, default=20, help="seconds to wait for CloudWatch before fetching"
    )
    a = ap.parse_args()

    st = deploy.load_state(a.build)
    if st.get("region") != a.region or "functions" not in st:
        sys.exit(f"no deployment for {a.region} in {a.build}; run deploy.py up")
    reqs = json.load(open(a.requests))
    os.makedirs(a.out, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    arms = a.arms.split(",")
    b = Bench(sess, st, reqs, a.out, a.concurrency)
    lam = b.lam
    fns = {}
    for arm in arms:
        c = lam.get_function_configuration(FunctionName=st["functions"][arm]["name"])
        rm = lam.get_runtime_management_config(FunctionName=st["functions"][arm]["name"])
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
        if deploy.b64_to_hex(c["CodeSha256"]) != st["manifest"]["function_zip"]["sha256"]:
            sys.exit(f"{arm}: deployed code {c['CodeSha256']} is not build/function.zip")
    cond = {
        "schema": "lkai-conditions/1",
        "region": a.region,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": a.warm,
        "arms": arms,
        "requests_file": os.path.relpath(a.requests, HERE),
        "requests": reqs,
        "probe": reqs["probe"],
        "functions": fns,
        "build_manifest": st["manifest"],
        "runtime_version_arn": st.get("runtime_version_arn"),
        "client": {
            "python": platform.python_version(),
            "boto3": boto3.__version__,
            "host": platform.machine(),
            "invoke_client": "fresh per wave",
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds_meta": [],
    }
    json.dump(cond, open(os.path.join(a.out, "conditions.json"), "w"), indent=1)

    last_end = 0.0
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else arms[::-1]
        tag = f"r{rnd}-{time.time():.3f}"
        t0 = time.time()
        meta = {"round": rnd, "order": order, "cold_tag": tag, "t0": t0}
        published: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(len(order)) as ex:
                list(ex.map(lambda arm, tag=tag: b.set_cold_tag(arm, tag), order))
            for arm in order:
                recs = b.wave(arm, "$LATEST", rnd, "cold")
                print(
                    f"r{rnd} {arm} cold: ok {sum(not r['error'] for r in recs)}/{len(recs)} "
                    f"rt median {sorted(r['rt_s'] for r in recs)[len(recs) // 2]:.2f}s",
                    flush=True,
                )
            # Publish sequentially so each created version is recorded before another operation can fail.
            # The outer run.sh trap is the final cleanup guard; this finally also makes bench.py safe standalone.
            for arm in order:
                published[arm] = b.publish(arm, tag)
            meta["published"] = published
            for arm in order:
                recs = b.wave(arm, published[arm]["version"], rnd, "restore")
                print(
                    f"r{rnd} {arm} restore (v{published[arm]['version']}, "
                    f"publish {published[arm]['publish_s']}s): ok "
                    f"{sum(not r['error'] for r in recs)}/{len(recs)} "
                    f"rt median {sorted(r['rt_s'] for r in recs)[len(recs) // 2]:.2f}s",
                    flush=True,
                )
            for i in range(a.warm):
                # One pass = `concurrency` simultaneous invokes per arm, so they land on every environment the
                # restore wave created (and so on every Graviton generation it drew), not on one reused environment.
                rid = b.order[i % len(b.order)]
                for arm in order if i % 2 == 0 else order[::-1]:
                    b.wave(
                        arm,
                        published[arm]["version"],
                        rnd,
                        "warm",
                        rid=rid,
                        seq0=i * a.concurrency,
                    )
            print(f"r{rnd} warm: {a.warm} passes x {a.concurrency} per arm", flush=True)
        finally:
            for arm, pub in published.items():
                b.delete_version(arm, pub["version"])
            if published:
                meta["t_versions_deleted"] = time.time()
        time.sleep(a.log_lag)
        # Each round's window starts where the last one ended, so no event is saved twice. The snapshot init runs
        # inside publish_version, after t0, so the round's own window holds it.
        t_end = time.time()
        meta["log_window"] = [max(t0, last_end), t_end]
        meta["log_events"] = b.save_logs(max(t0, last_end), t_end)
        last_end = t_end
        cond["rounds_meta"].append(meta)
        json.dump(cond, open(os.path.join(a.out, "conditions.json"), "w"), indent=1)
    cond["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cond["client"]["invoke_clients_made"] = b.invoke_clients
    json.dump(cond, open(os.path.join(a.out, "conditions.json"), "w"), indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
