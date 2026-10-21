"""Deploy experiment E6: two functions with P1's configuration. They differ only in the bytecode their zip ships.

  python e6deploy.py up        [--p1-function lkai-p1-inst]   # upload both zips, role, log groups, functions
  python e6deploy.py show
  python e6deploy.py teardown  [--all]                       # published versions; --all: functions, role, zips

lkai-e6-ctl runs build/ctl.zip (P1's zip plus e6_entry.py). lkai-e6-pyc runs build/pyc.zip (the same, plus complete
unchecked-hash bytecode). Both use handler e6_entry.handler, which imports P1's lkai_entry unchanged. Both copy P1's
configuration from the deployed P1 function:
- python3.12, arm64, 10,240 MB, 512 MB ephemeral storage, 900 s timeout, SnapStart on published versions
- P1's environment with LKAI_PHASES=1, and P1's pinned runtime version ARN
- P1's weights object (same bucket, key and bytes), read-only
The zips go to the e6/ prefix of P1's bucket, so P1's objects are untouched. The role may GetObject only the
weights key, and may write only to lkai-e6-* log groups. State is kept in build/deployment.json (git-ignored), in
the shape baseline/bench.py's Bench reads.
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

PREFIX = "lkai-e6"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E6"}
ARMS = ("ctl", "pyc")
BUILD = os.path.join(HERE, "build")


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


def adopt_p1(sess, st: dict, p1_function: str) -> None:
    """P1's deployed identity: bucket, weights key, environment and runtime version ARN."""
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
    iam.put_role_policy(RoleName=name, PolicyName="lkai-e6", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict) -> None:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(BUILD, "manifest.json")))
    st["manifest"] = m
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    st.setdefault("zip_keys", {})
    for arm in ARMS:
        z = m["zips"][arm]["sha256"]
        key = f"e6/code/{z}/function.zip"
        s3.upload_file(os.path.join(BUILD, f"{arm}.zip"), st["bucket"], key)
        st["zip_keys"][arm] = key
        name = f"{PREFIX}-{arm}"
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        env = dict(st["p1_environment"], COLD_TAG="0", LKAI_PHASES="1", E6_ARM=arm)
        cfg = dict(
            FunctionName=name,
            Runtime=p1.RUNTIME,
            Architectures=["arm64"],
            Handler="e6_entry.handler",
            Role=role,
            MemorySize=p1.MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": env},
            Description=f"lambda-kleidiai E6 {arm}: P1 zip, {'complete unchecked-hash bytecode' if arm == 'pyc' else 'P1 bytecode as shipped'}",
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
        print("function", name, z[:12], c["CodeSize"])


def teardown(sess, st: dict, everything: bool) -> list[str]:
    """Delete every published version (and with --all the functions, role, log groups and zips). Each step is
    attempted even when an earlier one fails; the failures are returned."""
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
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-e6")
            iam.delete_role(RoleName=f"{PREFIX}-fn")
        except iam.exceptions.NoSuchEntityException:
            pass
        except Exception as e:
            failures.append(f"role: {e}")
        for arm in ARMS:
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
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required: no step uses the default credential chain")
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
            up(sess, st)
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
