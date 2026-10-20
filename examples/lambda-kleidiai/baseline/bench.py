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
import buildcache  # noqa: E402
import deploy  # noqa: E402
import phases  # noqa: E402
import summarize  # noqa: E402

_lock = threading.Lock()
observed_class = phases.observed_class


def at_least(minimum: int):
    """argparse type for an integer count >= minimum."""

    def parse(value: str) -> int:
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"must be an integer >= {minimum}, got {value!r}"
            ) from None
        if n < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}, got {value!r}")
        return n

    return parse


positive_int = at_least(1)


def nonnegative_int(value: str) -> int:
    """argparse type for a count that may be zero."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return n


class CleanupError(RuntimeError):
    """One or more published versions could not be deleted (they keep billing for SnapStart cache)."""


def _gone(e: BaseException) -> bool:
    return type(e).__name__ == "ResourceNotFoundException"


def delete_versions(b, published: dict[str, dict]) -> list[str]:
    """Delete every version in `published` ({arm: {"version": v}}), attempting each even if another fails. Returns
    the failures; an already-deleted version is not one."""
    failures = []
    for arm, pub in published.items():
        try:
            b.delete_version(arm, pub["version"])
        except Exception as e:  # the next version must still be deleted
            if not _gone(e):
                failures.append(f"{arm}:{pub['version']}: {type(e).__name__}: {e}")
                print(f"FAILED to delete {arm} version {pub['version']}: {e}", file=sys.stderr)
    return failures


class Bench:
    """Two kinds of Lambda client.

    `lam` makes the lifecycle calls (configuration, publish, readiness polls, deletes). They are idempotent, so it
    has a short read timeout and standard retries: a pooled connection that died while idle costs one retry, not a
    900 s hang. Every cold, restore and warm wave invokes through a NEW client (new_invoke_client), so no measured
    invoke reuses a connection pooled before the publish/SnapStart-readiness wait or by an earlier wave. A silently
    dead connection would otherwise time out with no Lambda START/REPORT and pollute client timing. This changes
    client round-trip timing only; REPORT lines and phase records come from Lambda."""

    def __init__(self, sess, st: dict, reqs: dict, out: str, concurrency: int):
        self.st, self.out, self.sess = st, out, sess
        self.concurrency = concurrency
        self.lam = sess.client(
            "lambda",
            config=Config(
                read_timeout=120,
                connect_timeout=10,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        self.logs = sess.client("logs")
        self.invoke_clients = 0
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
        self,
        arm: str,
        qualifier: str,
        rid: str,
        rnd: int,
        wave: str,
        seq: int,
        lam=None,
    ) -> dict:
        name = self.st["functions"][arm]["name"]
        client = lam or self.lam
        t_wall = time.time()
        t = time.perf_counter()
        try:
            r = client.invoke(
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
        elif problem := summarize.answer_problem(p, sorted(self.reqs[rid].get("questions", {}))):
            # Lambda and the handler returned without error, but there is no answer to check: a failure, so it is
            # counted as an error and kept out of the latency distributions instead of vanishing from correctness.
            rec["error"] = f"malformed:{problem}"
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
                ex.submit(
                    self.invoke,
                    arm,
                    qualifier,
                    rid or self.probe,
                    rnd,
                    wave,
                    seq0 + i,
                    lam,
                )
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
        # wait_updated can sys.exit, and the caller never learns v, so delete the version here. A failed delete is
        # reported, but the wait's failure is what propagates; run.sh's teardown is the second chance.
        try:
            c = deploy.wait_updated(self.lam, name, v)
        except BaseException:
            for f in delete_versions(self, {arm: {"version": v}}):
                print("publish cleanup:", f, file=sys.stderr)
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
    # Two rounds is the least that gives run-to-run variance. Zero rounds, concurrency or warm passes would produce a
    # run with no invocations, or with a whole start class missing, that could still look complete.
    ap.add_argument("--rounds", type=at_least(2), default=5, help="at least 2")
    ap.add_argument("--concurrency", type=positive_int, default=10)
    ap.add_argument(
        "--warm",
        type=positive_int,
        default=10,
        help="warm passes per arm per round (each `concurrency` wide); at least the number of fixed requests, so "
        "every request is answered",
    )
    ap.add_argument("--region", default="us-east-2")
    # No default: an implicit credential chain could benchmark (and bill) whatever account it resolves to.
    ap.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile (default $AWS_PROFILE); required, and must be the account deploy.py recorded",
    )
    ap.add_argument(
        "--memory",
        type=deploy.memory_mb,
        help="function memory in MB the deployment must have (default: what deploy.py recorded); "
        "the run refuses to start if either function differs",
    )
    ap.add_argument(
        "--arms",
        default="inst,rep",
        help="run order of the two arms; must be exactly inst and rep (the REPORT-only control)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--log-lag",
        type=nonnegative_int,
        default=20,
        help="seconds to wait for CloudWatch before fetching",
    )
    a = ap.parse_args()
    arms = a.arms.split(",")
    if p := summarize.arms_problem(arms):
        ap.error(p)
    reqs = json.load(open(a.requests))
    if p := summarize.request_set_problem(reqs):
        ap.error(f"{a.requests}: {p}")
    if p := summarize.warm_problem(a.warm, reqs):
        ap.error(p)
    if not a.profile:
        ap.error("--profile or AWS_PROFILE is required: no implicit credentials")

    # Cost is part of the result; a Region without a pinned price table would spend money on a run whose cost
    # summarize.py cannot honestly compute.
    if a.region not in summarize.PRICES_BY_REGION:
        sys.exit(
            f"no pinned prices for {a.region} (have {sorted(summarize.PRICES_BY_REGION)}); "
            "add a dated price table to summarize.PRICES_BY_REGION first"
        )
    st = deploy.load_state(a.build)
    if st.get("region") != a.region or "functions" not in st:
        sys.exit(f"no deployment for {a.region} in {a.build}; run deploy.py up")
    if p := summarize.new_run_provenance_problem(st.get("manifest") or {}):
        sys.exit(f"refusing to benchmark: {p}")
    # The harness (bench.py, summarize.py, requests.json, ...) is part of what reproduces the result.
    harness = buildcache.harness_provenance()
    if harness["dirty_inputs"]:
        sys.exit(
            f"refusing to benchmark: harness files differ from {harness['commit']}: {harness['dirty_inputs']}"
        )
    memory = a.memory or st.get("memory_mb", deploy.MEMORY)
    os.makedirs(a.out, exist_ok=True)
    # invocations.jsonl and logs/ are appended to: a reused directory would mix two runs' evidence.
    if any(os.path.exists(os.path.join(a.out, f)) for f in ("invocations.jsonl", "logs")):
        sys.exit(f"{a.out} already holds a run; use a new --out")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    deploy.check_account(sess, st)
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
        if c.get("MemorySize") != memory:
            sys.exit(
                f"{arm}: deployed MemorySize {c.get('MemorySize')} MB, run wants {memory} MB; "
                f"deploy.py up --memory {memory}"
            )
        # An arm is what its LKAI_PHASES says: `rep` with phases on would measure no overhead at all.
        flag = c["Environment"]["Variables"].get("LKAI_PHASES")
        if flag != summarize.ARM_PHASES[arm]:
            sys.exit(
                f"{arm}: deployed LKAI_PHASES={flag!r}, the {arm} arm needs {summarize.ARM_PHASES[arm]!r}"
            )
    cond = {
        "schema": "lkai-conditions/1",
        "region": a.region,
        "memory_mb": memory,
        "rounds": a.rounds,
        "concurrency": a.concurrency,
        "warm_per_arm_per_round": a.warm,
        "arms": arms,
        "requests_file": os.path.relpath(a.requests, HERE),
        "requests": reqs,
        "probe": reqs["probe"],
        "functions": fns,
        "build_manifest": st["manifest"],
        "harness": harness,
        "runtime_version_arn": st.get("runtime_version_arn"),
        "client": {
            "python": platform.python_version(),
            "boto3": boto3.__version__,
            "host": platform.machine(),
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
        failed = True
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
            failed = False
        finally:
            # Every version is attempted; a failed delete must not leave the next one billing. If the round itself
            # failed, that exception propagates and the cleanup failures are reported beside it.
            cleanup_failures = delete_versions(b, published)
            if published:
                meta["t_versions_deleted"] = time.time()
            if cleanup_failures:
                meta["cleanup_failures"] = cleanup_failures
                cond["rounds_meta"].append(meta)
                json.dump(cond, open(os.path.join(a.out, "conditions.json"), "w"), indent=1)
                if not failed:
                    raise CleanupError(
                        f"round {rnd}: {len(cleanup_failures)} version(s) not deleted: {cleanup_failures}"
                    )
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
    json.dump(cond, open(os.path.join(a.out, "conditions.json"), "w"), indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
