"""I1: reproduce the first-restore-wave stall seen in bench.py runs, with a tiny SnapStart function instead of the model.

  AWS_PROFILE=<p> python repro.py up    --build build/                # role + one function per arm
  AWS_PROFILE=<p> python repro.py run   --build build/ --out results/<run-id> [--idle 200,400,700,1100]
  AWS_PROFILE=<p> python repro.py teardown --build build/             # functions, versions, log groups, role

Every arm runs in its own thread, with its own function and its own clients, in parallel:
  1. publish v1 (SnapStart), then a WARM-UP wave of `concurrency` simultaneous invokes of v1 through client R. This
     leaves `concurrency` pooled keep-alive connections in R, the way bench.py's cold wave did.
  2. publish v2 and wait for its snapshot, then idle until `idle` seconds have passed since the warm-up wave ended.
  3. a MEASURED wave of `concurrency` simultaneous invokes, which depends on the arm:
       reuse_poll    through R; R also made every publish/readiness poll (every 3 s) while idle. This is bench.py
                     at aac59d4: one client for lifecycle calls and invokes.
       reuse_nopoll  through R; lifecycle calls went through a separate client, so R's pool sat untouched.
       fresh_new     through a NEW client, on v2 (published during the idle and never invoked: alternatives (a), (b)).
       fresh_old     through a NEW client, on v1 (invoked once at warm-up, then idle for `idle` seconds).
The measured wave's client has no retries and a short read timeout, so a request that never reaches Lambda shows up
as an error instead of a 15-minute hang.

The run records:
  - each invoke's result, round trip, executed version, RequestId and REPORT start class;
  - the measured client's pooled connections just before the wave (local port, whether urllib3 would notice them
    dropped, and their TCP state from /proc/net/tcp);
  - the TCP state of every socket the process owns, 20 s and 60 s into the measured wave. Unacknowledged bytes
    (tx_queue > 0) with retransmits on an ESTABLISHED socket mean the request went into a path that no longer
    delivers it;
  - urllib3's debug log ("Starting new HTTPS connection", "Resetting dropped connection"), with thread names;
  - CloudWatch Invocations per function version, and START lines per version from the function's log group.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

PREFIX = "lkai-i1-df45"
ARMS = ("reuse_poll", "reuse_nopoll", "fresh_new", "fresh_old")
HANDLER = """
import os, time
def handler(event, context):
    time.sleep(float(event.get("sleep", 0.3)))
    return {"ok": 1, "pid": os.getpid(), "version": context.function_version}
"""
TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}
_lock = threading.Lock()


def state_path(build: str) -> str:
    return os.path.join(build, "repro-state.json")


def load_state(build: str) -> dict:
    p = state_path(build)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(build: str, st: dict) -> None:
    os.makedirs(build, exist_ok=True)
    json.dump(st, open(state_path(build), "w"), indent=1)


def fn_name(idle: int, arm: str) -> str:
    return f"{PREFIX}-{idle}-{arm.replace('_', '')}"


def invoke_cfg(concurrency: int, read_timeout: int) -> Config:
    return Config(
        read_timeout=read_timeout,
        connect_timeout=10,
        retries={"max_attempts": 0},
        max_pool_connections=max(16, concurrency * 2),
    )


def lifecycle_cfg() -> Config:
    return Config(
        read_timeout=60, connect_timeout=10, retries={"max_attempts": 5, "mode": "standard"}
    )


# ------------------------------------------------------------------ TCP evidence


def proc_tcp() -> dict[int, dict]:
    """Every TCP socket on the host keyed by inode: local/remote port, state, tx/rx queue, retransmits, timer."""
    out = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(path).read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            local, remote, st, queues, timer, retr = f[1], f[2], f[3], f[4], f[5], f[6]
            tx, rx = (int(x, 16) for x in queues.split(":"))
            out[int(f[9])] = {
                "lport": int(local.rsplit(":", 1)[1], 16),
                "rport": int(remote.rsplit(":", 1)[1], 16),
                "state": TCP_STATES.get(st, st),
                "tx_queue": tx,
                "rx_queue": rx,
                "timer": timer,
                "retransmits": int(retr, 16),
            }
    return out


def own_sockets() -> list[dict]:
    """TCP sockets this process holds open (via /proc/self/fd), with their /proc/net/tcp row."""
    inodes = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            m = re.match(r"socket:\[(\d+)\]", os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            continue
        if m:
            inodes.add(int(m.group(1)))
    tcp = proc_tcp()
    return sorted((tcp[i] for i in inodes if i in tcp), key=lambda r: r["lport"])


def pool_snapshot(client) -> list[dict]:
    """The client's pooled (idle) connections: local port, whether urllib3's reuse check would see them dropped,
    and the kernel's TCP state for that port."""
    from urllib3.util.wait import wait_for_read

    mgr = client._endpoint.http_session._manager  # noqa: SLF001 (botocore exposes no pool accessor)
    tcp = {r["lport"]: r for r in proc_tcp().values()}
    rows = []
    for key in list(mgr.pools.keys()):
        pool = mgr.pools.get(key)
        if pool is None:
            continue
        for conn in list(pool.pool.queue):
            if conn is None:  # urllib3 pre-fills the pool queue with None placeholders
                continue
            sock = getattr(conn, "sock", None)
            if sock is None:
                rows.append({"connected": False})
                continue
            try:
                lport = sock.getsockname()[1]
            except OSError:
                rows.append({"connected": False, "closed_fd": True})
                continue
            rows.append(
                {
                    "connected": True,
                    "lport": lport,
                    "urllib3_sees_dropped": bool(wait_for_read(sock, timeout=0.0)),
                    "tcp": tcp.get(lport),
                }
            )
    return rows


# ------------------------------------------------------------------ AWS


def wait_ready(lam, name: str, qualifier: str | None = None) -> dict:
    q = {"Qualifier": qualifier} if qualifier else {}
    while True:
        c = lam.get_function_configuration(FunctionName=name, **q)
        if c.get("State") == "Active" and c.get("LastUpdateStatus") in (None, "Successful"):
            return c
        if c.get("State") == "Failed" or c.get("LastUpdateStatus") == "Failed":
            raise RuntimeError(
                f"{name}:{qualifier}: {c.get('StateReason') or c.get('LastUpdateStatusReason')}"
            )
        time.sleep(3)


def zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("app.py", HANDLER)
    return buf.getvalue()


def up(sess, build: str, idles: list[int]) -> None:
    st = load_state(build)
    iam, lam = sess.client("iam"), sess.client("lambda", config=lifecycle_cfg())
    role = f"{PREFIX}-fn"
    try:
        arn = iam.get_role(RoleName=role)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        trust = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        arn = iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust))["Role"][
            "Arn"
        ]
        st["role"] = role
        save_state(build, st)
        time.sleep(10)
    logs_arn = f"arn:aws:logs:{sess.region_name}:*:log-group:/aws/lambda/{PREFIX}-*"
    pol = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [logs_arn, logs_arn + ":*"],
            }
        ],
    }
    iam.put_role_policy(RoleName=role, PolicyName="logs", PolicyDocument=json.dumps(pol))
    st["role"] = role
    st.setdefault("functions", [])
    logs = sess.client("logs")
    code = zip_bytes()
    for idle in idles:
        for arm in ARMS:
            name = fn_name(idle, arm)
            lg = f"/aws/lambda/{name}"
            try:
                logs.create_log_group(logGroupName=lg)
            except logs.exceptions.ResourceAlreadyExistsException:
                pass
            logs.put_retention_policy(logGroupName=lg, retentionInDays=3)
            for attempt in range(10):
                try:
                    lam.create_function(
                        FunctionName=name,
                        Runtime="python3.13",
                        Architectures=["arm64"],
                        MemorySize=128,
                        Timeout=30,
                        Role=arn,
                        Handler="app.handler",
                        Code={"ZipFile": code},
                        SnapStart={"ApplyOn": "PublishedVersions"},
                        Tags={"project": "lkai", "card": "df45"},
                    )
                    break
                except lam.exceptions.ResourceConflictException:
                    break
                except lam.exceptions.InvalidParameterValueException as e:  # role not assumable yet
                    if attempt == 9:
                        raise
                    print("waiting for role:", e, flush=True)
                    time.sleep(10)
            if name not in st["functions"]:
                st["functions"].append(name)
            save_state(build, st)
    for name in st["functions"]:
        wait_ready(lam, name)
    print("UP", len(st["functions"]), "functions", flush=True)


def teardown(sess, build: str) -> None:
    st = load_state(build)
    lam, logs, iam = (
        sess.client("lambda", config=lifecycle_cfg()),
        sess.client("logs"),
        sess.client("iam"),
    )
    for name in st.get("functions", []):
        try:
            lam.delete_function(
                FunctionName=name
            )  # deletes every version (and its SnapStart cache) too
        except lam.exceptions.ResourceNotFoundException:
            pass
        try:
            logs.delete_log_group(logGroupName=f"/aws/lambda/{name}")
        except logs.exceptions.ResourceNotFoundException:
            pass
    if st.get("role"):
        try:
            iam.delete_role_policy(RoleName=st["role"], PolicyName="logs")
        except iam.exceptions.NoSuchEntityException:
            pass
        try:
            iam.delete_role(RoleName=st["role"])
        except iam.exceptions.NoSuchEntityException:
            pass
    left = [
        f["FunctionName"]
        for p in lam.get_paginator("list_functions").paginate()
        for f in p["Functions"]
        if f["FunctionName"].startswith(PREFIX)
    ]
    print("TEARDOWN done; functions left with prefix:", left, flush=True)
    st["torn_down"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(build, st)


# ------------------------------------------------------------------ run


class Arm:
    def __init__(self, sess, out, idle: int, arm: str, concurrency: int, read_timeout: int):
        self.sess, self.out, self.idle, self.arm = sess, out, idle, arm
        self.name = fn_name(idle, arm)
        self.concurrency, self.read_timeout = concurrency, read_timeout
        self.R = sess.client("lambda", config=invoke_cfg(concurrency, read_timeout))
        # reuse_poll makes its lifecycle calls through R, as bench.py at aac59d4 does.
        self.L = self.R if arm == "reuse_poll" else sess.client("lambda", config=lifecycle_cfg())
        self.clients_made = 1 if self.L is self.R else 2
        self.meta: dict = {"arm": arm, "idle_s": idle, "function": self.name}

    def write(self, rec: dict) -> None:
        with _lock, open(os.path.join(self.out, "invocations.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def publish(self, tag: str) -> dict:
        # Lambda returns the existing version when nothing changed, so each publish first changes the
        # configuration (as bench.py's COLD_TAG does).
        self.L.update_function_configuration(
            FunctionName=self.name, Environment={"Variables": {"TAG": tag}}
        )
        wait_ready(self.L, self.name)
        t = time.time()
        v = self.L.publish_version(FunctionName=self.name)["Version"]
        c = wait_ready(self.L, self.name, v)
        return {
            "version": v,
            "t_publish": t,
            "t_ready": time.time(),
            "snapstart": c.get("SnapStart"),
        }

    def invoke(self, client, qualifier: str, wave: str, seq: int) -> dict:
        t_wall, t = time.time(), time.perf_counter()
        rec = {
            "arm": self.arm,
            "idle_s": self.idle,
            "wave": wave,
            "seq": seq,
            "qualifier": qualifier,
        }
        try:
            r = client.invoke(
                FunctionName=self.name,
                Qualifier=qualifier,
                LogType="Tail",
                Payload=json.dumps({"sleep": 0.3}).encode(),
            )
            body = json.loads(r["Payload"].read())
            tail = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
            rec.update(
                status=r["StatusCode"],
                error=r.get("FunctionError"),
                executed_version=r.get("ExecutedVersion"),
                lambda_request_id=r["ResponseMetadata"]["RequestId"],
                pid=body.get("pid"),
                restore=("Restore Duration" in tail),
                init=("Init Duration" in tail),
                report=next((ln for ln in tail.splitlines() if ln.startswith("REPORT")), None),
            )
        except Exception as e:  # a transport failure is the result under test
            rec.update(status=None, error=f"client:{type(e).__name__}", detail=str(e)[:300])
        rec.update(t_start=t_wall, rt_s=round(time.perf_counter() - t, 3))
        self.write(rec)
        return rec

    def wave(self, client, qualifier: str, wave: str) -> list[dict]:
        threading.current_thread().name = f"{self.arm}-{self.idle}-{wave}"
        with ThreadPoolExecutor(
            self.concurrency, thread_name_prefix=f"{self.arm}-{self.idle}-{wave}"
        ) as ex:
            return list(
                ex.map(lambda i: self.invoke(client, qualifier, wave, i), range(self.concurrency))
            )

    def run(self) -> dict:
        threading.current_thread().name = f"{self.arm}-{self.idle}"
        m = self.meta
        run_tag = f"{time.time():.3f}"
        m["v1"] = self.publish(f"v1-{run_tag}")
        recs = self.wave(self.R, m["v1"]["version"], "warmup")
        t_end = max(r["t_start"] + r["rt_s"] for r in recs)
        m["t_warmup_end"] = t_end
        m["warmup_ok"] = sum(not r["error"] for r in recs)
        m["pool_after_warmup"] = pool_snapshot(self.R)
        m["v2"] = self.publish(f"v2-{run_tag}")
        if m["v2"]["version"] == m["v1"]["version"]:
            raise RuntimeError("v2 publish returned v1")
        while time.time() < t_end + self.idle:
            if self.arm == "reuse_poll":  # bench.py keeps polling the same client between publishes
                self.L.get_function_configuration(
                    FunctionName=self.name, Qualifier=m["v2"]["version"]
                )
                time.sleep(3)
            else:
                time.sleep(min(3, max(0.0, t_end + self.idle - time.time())))
        if self.arm.startswith("reuse"):
            client = self.R
        else:
            client = self.sess.client(
                "lambda", config=invoke_cfg(self.concurrency, self.read_timeout)
            )
            self.clients_made += 1
        target = m["v1"]["version"] if self.arm == "fresh_old" else m["v2"]["version"]
        m["measured"] = {
            "target_version": target,
            "t_start": time.time(),
            "idle_since_warmup_s": round(time.time() - t_end, 1),
            "target_age_since_ready_s": round(
                time.time() - m["v1" if self.arm == "fresh_old" else "v2"]["t_ready"], 1
            ),
            "pool_before": pool_snapshot(client),
        }
        recs = self.wave(client, target, "measured")
        m["measured"]["ok"] = sum(not r["error"] for r in recs)
        m["measured"]["errors"] = sorted({r["error"] for r in recs if r["error"]})
        m["measured"]["t_end"] = max(r["t_start"] + r["rt_s"] for r in recs)
        m["clients_made"] = self.clients_made
        print(
            f"{self.arm:12s} idle {self.idle:5d}s: measured ok {m['measured']['ok']}/{len(recs)} "
            f"errors {m['measured']['errors']}",
            flush=True,
        )
        return m


def sampler(out: str, stop: threading.Event, marks: list[float]) -> None:
    """Snapshot this process's TCP sockets at each mark (epoch seconds) until stopped."""
    path = os.path.join(out, "tcp_samples.jsonl")
    pending = sorted(marks)
    while not stop.is_set():
        now = time.time()
        due = [t for t in pending if t <= now]
        for t in due:
            pending.remove(t)
            with open(path, "a") as fh:
                fh.write(json.dumps({"t": now, "mark": t, "sockets": own_sockets()}) + "\n")
        stop.wait(1)


def metrics(sess, st_arms: list[dict], t0: float, t1: float) -> dict:
    cw, logs = sess.client("cloudwatch"), sess.client("logs")
    out = {}
    for m in st_arms:
        name = m["function"]
        res = {}
        for key in ("v1", "v2"):
            v = m[key]["version"]
            q = {
                "Id": "inv",
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Invocations",
                        "Dimensions": [
                            {"Name": "FunctionName", "Value": name},
                            {"Name": "Resource", "Value": f"{name}:{v}"},
                        ],
                    },
                    "Period": 60,
                    "Stat": "Sum",
                },
            }
            r = cw.get_metric_data(
                MetricDataQueries=[q], StartTime=int(t0) - 120, EndTime=int(t1) + 300
            )["MetricDataResults"][0]
            starts = 0
            for page in logs.get_paginator("filter_log_events").paginate(
                logGroupName=f"/aws/lambda/{name}",
                startTime=int(t0 * 1000) - 120000,
                endTime=int(t1 * 1000) + 300000,
                filterPattern=f'"START RequestId" "Version: {v}"',
            ):
                starts += sum(
                    1 for e in page["events"] if e["message"].rstrip().endswith(f"Version: {v}")
                )
            res[v] = {"invocations_metric": sum(r["Values"]), "start_lines": starts}
        out[name] = res
    return out


def run(
    sess, build: str, out: str, idles: list[int], concurrency: int, read_timeout: int, lag: int
) -> None:
    os.makedirs(out, exist_ok=True)
    log = logging.getLogger("urllib3")
    log.setLevel(logging.DEBUG)
    h = logging.FileHandler(os.path.join(out, "urllib3.log"))
    h.setFormatter(logging.Formatter("%(created).3f %(threadName)s %(name)s %(message)s"))
    log.addHandler(h)
    cond = {
        "schema": "lkai-i1-repro/1",
        "region": sess.region_name,
        "idles_s": idles,
        "arms": list(ARMS),
        "concurrency": concurrency,
        "measured_read_timeout_s": read_timeout,
        "function": {
            "runtime": "python3.13",
            "arch": "arm64",
            "memory_mb": 128,
            "snapstart": "PublishedVersions",
        },
        "client": {
            "python": sys.version.split()[0],
            "boto3": boto3.__version__,
            "host": socket.gethostname().split(".")[0],
            "tcp_keepalive_time": open("/proc/sys/net/ipv4/tcp_keepalive_time").read().strip(),
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(cond, open(os.path.join(out, "conditions.json"), "w"), indent=1)
    arms = [Arm(sess, out, idle, arm, concurrency, read_timeout) for idle in idles for arm in ARMS]
    t0 = time.time()
    stop = threading.Event()
    results: list[dict] = []
    # Measured waves begin roughly idle + publish times after start; sample sockets often enough to catch each one.
    marks = [t0 + s for s in range(60, max(idles) + 900, 20)]
    smp = threading.Thread(target=sampler, args=(out, stop, marks), daemon=True)
    smp.start()
    try:
        with ThreadPoolExecutor(len(arms)) as ex:
            futs = [ex.submit(a.run) for a in arms]
            for f, a in zip(futs, arms, strict=True):
                try:
                    results.append(f.result())
                except Exception as e:
                    a.meta["failed"] = f"{type(e).__name__}: {e}"
                    results.append(a.meta)
    finally:
        stop.set()
        cond["arms_meta"] = results
        cond["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(cond, open(os.path.join(out, "conditions.json"), "w"), indent=1)
    t1 = time.time()
    print(f"waiting {lag}s for CloudWatch", flush=True)
    time.sleep(lag)
    cond["lambda_side"] = metrics(sess, [m for m in results if "v2" in m], t0, t1)
    json.dump(cond, open(os.path.join(out, "conditions.json"), "w"), indent=1)
    print("REPRO_DONE", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "run", "teardown"])
    ap.add_argument(
        "--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    )
    ap.add_argument("--out")
    ap.add_argument("--idle", default="200,400,700,1100", help="comma-separated idle lengths (s)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--read-timeout", type=int, default=120)
    ap.add_argument(
        "--log-lag", type=int, default=240, help="seconds to wait for CloudWatch metrics"
    )
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    a = ap.parse_args()
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    idles = [int(x) for x in a.idle.split(",")]
    if a.cmd == "up":
        up(sess, a.build, idles)
    elif a.cmd == "teardown":
        teardown(sess, a.build)
    else:
        if not a.out:
            sys.exit("--out required")
        run(sess, a.build, a.out, idles, a.concurrency, a.read_timeout, a.log_lag)


if __name__ == "__main__":
    main()
