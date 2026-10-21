"""Deploy experiment E10: one function per context size, all from the same zip and P1's configuration.

  python e10deploy.py up        [--p1-function lkai-p1-inst]   # upload the zip, role, log groups, functions
  python e10deploy.py show
  python e10deploy.py teardown  [--all]                       # published versions; --all: functions, role, zip

Arms (function lkai-fbb1-c<n_ctx>): n_ctx 512, 1024, 2048, 4096, 8192 and 16384. 16384 is P1's value, so that arm
is the control: P1's zip plus e10_entry.py, with every argument P1 passes. Every arm copies P1's configuration from
the deployed P1 function:
- python3.12, arm64, 10,240 MB, 512 MB ephemeral storage, 900 s timeout, SnapStart on published versions
- P1's environment with LKAI_PHASES=1, and P1's pinned runtime version ARN
- P1's weights object (same bucket, key and bytes), read-only
The arms differ in E10_N_CTX (and the E10_ARM label) only. The zip goes to the e10/ prefix of P1's bucket, so P1's
objects are untouched. The role may GetObject only the weights key and write only to lkai-fbb1-* log groups. State
is kept in build/deployment.json (git-ignored), in the shape baseline/bench.py's Bench reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import deploy as p1  # noqa: E402  (baseline/deploy.py: wait_updated, b64_to_hex, RUNTIME, MEMORY)

PREFIX = "lkai-fbb1"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E10"}
SIZES = (512, 1024, 2048, 4096, 8192, 16384)
CONTROL_SIZE = 16384  # P1's n_ctx (snapstart/engine.py load_engine with N_SEQ=17)
ARMS = tuple(f"c{n}" for n in SIZES)
CONTROL = f"c{CONTROL_SIZE}"
BUILD = os.path.join(HERE, "build")


def n_ctx_of(arm: str) -> int:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")
    return int(arm[1:])


def state_path() -> str:
    return os.path.join(BUILD, "deployment.json")


def load_state() -> dict:
    p = state_path()
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(st: dict) -> None:
    os.makedirs(BUILD, exist_ok=True)
    tmp = state_path() + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, state_path())


def arm_environment(p1_env: dict, arm: str) -> dict:
    """P1's environment, plus E10_N_CTX and the arm label. Nothing of P1's is changed or dropped."""
    return dict(p1_env, COLD_TAG="0", LKAI_PHASES="1", E10_ARM=arm, E10_N_CTX=str(n_ctx_of(arm)))


def adopt_p1(sess, st: dict, p1_function: str) -> None:
    lam = sess.client("lambda")
    c = lam.get_function_configuration(FunctionName=p1_function)
    rm = lam.get_runtime_management_config(FunctionName=p1_function)
    env = dict(c["Environment"]["Variables"])
    st["bucket"], st["model_key"] = env["MODEL_BUCKET"], env["MODEL_KEY"]
    st["p1_environment"] = {k: v for k, v in env.items() if k not in ("COLD_TAG", "LKAI_PHASES")}
    st["runtime_version_arn"] = rm["RuntimeVersionArn"]
    st["p1_function"] = p1_function
    st["p1_function_code_sha256"] = p1.b64_to_hex(c["CodeSha256"])


def ensure_role(iam, st: dict) -> str:
    name = f"{PREFIX}-fn"
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
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )["Role"]["Arn"]
        time.sleep(10)
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{st['bucket']}/{st['model_key']}",
            },
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": f"arn:aws:logs:{st['region']}:*:log-group:/aws/lambda/{PREFIX}-*:*",
            },
        ],
    }
    iam.put_role_policy(RoleName=name, PolicyName="lkai-e10", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict, arms: tuple[str, ...]) -> None:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(BUILD, "manifest.json")))
    if m.get("dirty_inputs"):
        sys.exit(f"build/manifest.json lists dirty inputs {m['dirty_inputs']}; commit and rebuild")
    st["manifest"] = m
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    z = m["zips"]["e10"]["sha256"]
    key = f"e10/code/{z}/function.zip"
    s3.upload_file(os.path.join(BUILD, "function.zip"), st["bucket"], key)
    st["zip_keys"] = {"e10": key}
    for arm in arms:
        name = f"{PREFIX}-{arm}"
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        cfg = dict(
            FunctionName=name,
            Runtime=p1.RUNTIME,
            Architectures=["arm64"],
            Handler="e10_entry.handler",
            Role=role,
            MemorySize=p1.MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": arm_environment(st["p1_environment"], arm)},
            Description=f"lambda-kleidiai E10 {arm}: P1 zip, n_ctx={n_ctx_of(arm)}",
        )
        for attempt in range(6):
            try:
                try:
                    lam.get_function(FunctionName=name)
                    lam.update_function_code(FunctionName=name, S3Bucket=st["bucket"], S3Key=key)
                    p1.wait_updated(lam, name)
                    lam.update_function_configuration(
                        **{k: v for k, v in cfg.items() if k != "Architectures"}
                    )
                except lam.exceptions.ResourceNotFoundException:
                    lam.create_function(
                        **cfg, Code={"S3Bucket": st["bucket"], "S3Key": key}, Tags=TAGS
                    )
                break
            except lam.exceptions.InvalidParameterValueException as e:  # role not yet assumable
                if attempt == 5:
                    raise
                print("retrying:", e)
                time.sleep(10)
        c = p1.wait_updated(lam, name)
        lam.put_runtime_management_config(
            FunctionName=name, UpdateRuntimeOn="Manual", RuntimeVersionArn=st["runtime_version_arn"]
        )
        if p1.b64_to_hex(c["CodeSha256"]) != z:
            sys.exit(f"{name}: deployed code {c['CodeSha256']} is not {z}")
        st["functions"][arm] = {
            "name": name,
            "arn": c["FunctionArn"],
            "code_sha256": z,
            "code_size": c["CodeSize"],
        }
        save_state(st)
        print("function", name, z[:12], c["CodeSize"])


def teardown(sess, st: dict, everything: bool) -> list[str]:
    lam, failures = sess.client("lambda"), []
    for f in st.get("functions", {}).values():
        try:
            for page in lam.get_paginator("list_versions_by_function").paginate(
                FunctionName=f["name"]
            ):
                for v in page["Versions"]:
                    if v["Version"] != "$LATEST":
                        lam.delete_function(FunctionName=f["name"], Qualifier=v["Version"])
                        print("deleted", f["name"], v["Version"])
            if everything:
                lam.delete_function(FunctionName=f["name"])
                print("deleted", f["name"])
        except lam.exceptions.ResourceNotFoundException:
            pass
        except Exception as e:  # the next function must still be cleaned up
            failures.append(f"{f['name']}: {e}")
    if everything:
        iam, logs, s3 = sess.client("iam"), sess.client("logs"), sess.client("s3")
        try:
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-e10")
            iam.delete_role(RoleName=f"{PREFIX}-fn")
        except iam.exceptions.NoSuchEntityException:
            pass
        except Exception as e:
            failures.append(f"role: {e}")
        for arm in st.get("functions", {}):
            try:
                logs.delete_log_group(logGroupName=f"/aws/lambda/{PREFIX}-{arm}")
            except logs.exceptions.ResourceNotFoundException:
                pass
            except Exception as e:
                failures.append(f"log group {arm}: {e}")
        for key in st.get("zip_keys", {}).values():
            try:
                s3.delete_object(Bucket=st["bucket"], Key=key)
            except Exception as e:
                failures.append(f"{key}: {e}")
        st["functions"], st["zip_keys"] = {}, {}
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "show", "teardown"])
    ap.add_argument("--p1-function", default="lkai-p1-inst")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required: no step uses the default credential chain")
    arms = tuple(a.arms.split(","))
    for arm in arms:
        n_ctx_of(arm)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = load_state()
    st.setdefault("region", a.region)
    st.setdefault("prefix", PREFIX)
    if st["region"] != a.region:
        sys.exit(f"{state_path()} is for {st['region']}")
    failures: list[str] = []
    try:
        if a.cmd == "up":
            adopt_p1(sess, st, a.p1_function)
            up(sess, st, arms)
        elif a.cmd == "teardown":
            failures = teardown(sess, st, a.all)
    finally:
        save_state(st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))
    if failures:
        sys.exit("teardown failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
