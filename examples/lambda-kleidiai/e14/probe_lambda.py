"""Deploy, drive and read E14's diagnostic probe function lkai-19b2-probe. Not a measured arm.

  python probe_lambda.py build --p1-zip <P1 function.zip> --probe-lib <dir of patch_concat_probe libs>
  python probe_lambda.py up                      # needs e14deploy.py's state (role, bucket, P1 environment)
  python probe_lambda.py run --cold 10 --warm-passes 4 --out build/probe-<id>
  python probe_lambda.py logs --out build/probe-<id>

build: P1's zip + e14_entry.py + e14_probe_entry.py, with the six lib/libggml-cpu-*.so replaced by the probe build
       (concat_zip.build's copy rules: every other entry and attribute unchanged).
up:    lkai-19b2-probe with P1's configuration (copied from e14deploy's state), LKAI_E14_PROBE=1, handler
       e14_probe_entry.handler, the e14/ prefix of P1's bucket, the lkai-19b2-fn role.
run:   a new COLD_TAG, `cold` simultaneous cold invokes of P1's probe request, then `warm-passes` passes of P1's
       five fixed requests `cold` wide on those environments. One new invoke client per wave.
logs:  every E14_PROBE / E14 / LKAI / REPORT line in the run's window to <out>/logs.jsonl.gz.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import concat_zip  # noqa: E402
import deploy as p1  # noqa: E402
import e14deploy  # noqa: E402

NAME = "lkai-19b2-probe"
BUILD = os.path.join(HERE, "build")


def build(p1_zip: str, probe_lib: str) -> dict:
    """probe.zip: concat_zip's rows.zip recipe with the probe libs, plus e14_probe_entry.py."""
    import zipfile

    tmp = tempfile.mkdtemp()
    try:
        m = concat_zip.build(p1_zip, probe_lib, os.path.join(HERE, "e14_entry.py"), tmp)
        src = os.path.join(tmp, "rows.zip")
        dst = os.path.join(BUILD, "probe.zip")
        os.makedirs(BUILD, exist_ok=True)
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
            for info in zin.infolist():
                zout.writestr(info, zin.read(info))
            e = zipfile.ZipInfo("e14_probe_entry.py", date_time=(2026, 1, 1, 0, 0, 0))
            e.compress_type = zipfile.ZIP_DEFLATED
            e.external_attr = 0o100644 << 16
            e.create_system = 3
            zout.writestr(e, open(os.path.join(HERE, "e14_probe_entry.py"), "rb").read())
    finally:
        shutil.rmtree(tmp)
    out = {
        "zip_sha256": concat_zip.sha256(dst),
        "p1_zip_sha256": m["p1_zip_sha256"],
        "replaced_libs": {k: v["rows_sha256"] for k, v in m["replaced_in_rows"].items()},
    }
    json.dump(out, open(os.path.join(BUILD, "probe.json"), "w"), indent=1)
    return out


def up(sess, st: dict) -> dict:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    pm = json.load(open(os.path.join(BUILD, "probe.json")))
    key = f"e14/code/{pm['zip_sha256']}/function.zip"
    s3.upload_file(os.path.join(BUILD, "probe.zip"), st["bucket"], key)
    group = f"/aws/lambda/{NAME}"
    try:
        logs.create_log_group(logGroupName=group, tags=e14deploy.TAGS)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    logs.put_retention_policy(logGroupName=group, retentionInDays=14)
    env = dict(
        st["p1_environment"], COLD_TAG="0", LKAI_PHASES="1", E14_ARM="probe", LKAI_E14_PROBE="1"
    )
    cfg = dict(
        FunctionName=NAME,
        Runtime=p1.RUNTIME,
        Architectures=["arm64"],
        Handler="e14_probe_entry.handler",
        Role=st["role"],
        MemorySize=p1.MEMORY,
        EphemeralStorage={"Size": 512},
        Timeout=900,
        Environment={"Variables": env},
        Description="lambda-kleidiai E14 diagnostic: P1 kernel with a thread-0 concat timer (not a measured arm)",
    )
    try:
        lam.get_function(FunctionName=NAME)
        lam.update_function_code(FunctionName=NAME, S3Bucket=st["bucket"], S3Key=key)
        p1.wait_updated(lam, NAME)
        lam.update_function_configuration(**{k: v for k, v in cfg.items() if k != "Architectures"})
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(
            **cfg, Code={"S3Bucket": st["bucket"], "S3Key": key}, Tags=e14deploy.TAGS
        )
    c = p1.wait_updated(lam, NAME)
    lam.put_runtime_management_config(
        FunctionName=NAME, UpdateRuntimeOn="Manual", RuntimeVersionArn=st["runtime_version_arn"]
    )
    if p1.b64_to_hex(c["CodeSha256"]) != pm["zip_sha256"]:
        sys.exit(f"{NAME}: deployed code {c['CodeSha256']} is not {pm['zip_sha256']}")
    st.setdefault("probe", {}).update(
        {"name": NAME, "code_sha256": pm["zip_sha256"], "zip_key": key}
    )
    return st


def _invoke_wave(sess, payloads: list[dict], timeout: int) -> list[dict]:
    lam = sess.client(
        "lambda",
        config=Config(
            read_timeout=timeout,
            connect_timeout=10,
            retries={"max_attempts": 0},
            max_pool_connections=max(16, 2 * len(payloads)),
        ),
    )

    def one(p):
        t = time.time()
        try:
            r = lam.invoke(FunctionName=NAME, Payload=json.dumps(p).encode(), LogType="Tail")
            body = json.loads(r["Payload"].read() or b"null")
            tail = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
            return {
                "t0": t,
                "rt_s": time.time() - t,
                "error": r.get("FunctionError"),
                "lkai": (body or {}).get("lkai"),
                "report": next((ln for ln in tail.splitlines() if ln.startswith("REPORT")), None),
            }
        except Exception as e:  # recorded, never retried
            return {"t0": t, "rt_s": time.time() - t, "error": f"client:{type(e).__name__}"}

    with ThreadPoolExecutor(len(payloads)) as ex:
        return list(ex.map(one, payloads))


def run(sess, st: dict, cold: int, warm_passes: int, out: str, timeout: int) -> None:
    os.makedirs(out, exist_ok=True)
    reqs = json.load(open(os.path.join(os.path.dirname(HERE), "baseline", "requests.json")))
    ctl = sess.client(
        "lambda", config=Config(retries={"mode": "standard", "max_attempts": 8}, read_timeout=60)
    )
    c = ctl.get_function_configuration(FunctionName=NAME)
    tag = f"e14-probe-{time.time():.3f}"
    ctl.update_function_configuration(
        FunctionName=NAME,
        Environment={"Variables": dict(c["Environment"]["Variables"], COLD_TAG=tag)},
    )
    p1.wait_updated(ctl, NAME)
    meta = {
        "cold_tag": tag,
        "t0": time.time(),
        "cold": cold,
        "warm_passes": warm_passes,
        "waves": [],
    }
    rows = []
    probe = next(
        r["request"] for r in reqs["requests"] if r["id"] == reqs["probe"]
    )  # "probe" names a request id
    w = _invoke_wave(sess, [probe] * cold, timeout)
    rows += [r | {"wave": "cold", "request_id": "probe"} for r in w]
    print("cold ok", sum(1 for r in w if not r["error"]), "/", cold, flush=True)
    for p in range(warm_passes):
        for r in reqs["requests"]:
            w = _invoke_wave(sess, [r["request"]] * cold, timeout)
            rows += [x | {"wave": "warm", "pass": p, "request_id": r["id"]} for x in w]
    meta["t1"] = time.time()
    with gzip.open(os.path.join(out, "invocations.jsonl.gz"), "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump(meta, open(os.path.join(out, "run.json"), "w"), indent=1)
    print("PROBE_RUN_DONE", out, "errors", sum(1 for r in rows if r["error"]))


def fetch_logs(sess, out: str, lag: int = 20) -> int:
    meta = json.load(open(os.path.join(out, "run.json")))
    time.sleep(lag)
    logs = sess.client("logs")
    n = 0
    with gzip.open(os.path.join(out, "logs.jsonl.gz"), "wt") as f:
        for page in logs.get_paginator("filter_log_events").paginate(
            logGroupName=f"/aws/lambda/{NAME}",
            startTime=int(meta["t0"] * 1000) - 5000,
            endTime=int(time.time() * 1000),
        ):
            for e in page["events"]:
                f.write(
                    json.dumps({k: e[k] for k in ("timestamp", "logStreamName", "message")}) + "\n"
                )
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "up", "run", "logs"])
    ap.add_argument("--p1-zip")
    ap.add_argument("--probe-lib")
    ap.add_argument("--cold", type=int, default=10)
    ap.add_argument("--warm-passes", type=int, default=4)
    ap.add_argument("--read-timeout", type=int, default=300)
    ap.add_argument("--out")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.p1_zip, a.probe_lib)))
        return
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = e14deploy.load_state()
    if a.cmd == "up":
        e14deploy.save_state(up(sess, st))
        print("function", NAME, st["probe"]["code_sha256"][:12])
    elif a.cmd == "run":
        run(sess, st, a.cold, a.warm_passes, a.out, a.read_timeout)
    else:
        print("log events", fetch_logs(sess, a.out))


if __name__ == "__main__":
    main()
