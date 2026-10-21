"""Deploy E7: two functions with P1's configuration and one zip. They differ only in LKAI_SKIP_TIED_OUTPUT.

  python deploy_e7.py up        --build e7/build [--p1-function lkai-p1-inst]
  python deploy_e7.py show
  python deploy_e7.py teardown  [--all]       # published versions; --all: functions, role, log groups, zip

  lkai-f768-ctl    LKAI_SKIP_TIED_OUTPUT unset   the tied output head is created and repacked (P1's load)
  lkai-f768-skip   LKAI_SKIP_TIED_OUTPUT=1       the tied output head is not created

Both run e7/build/function.zip (handler e7_entry.handler, which imports P1's lkai_entry unchanged) and copy P1's
deployed configuration: python3.12, arm64, 10,240 MB, 512 MB ephemeral storage, 900 s, SnapStart on published
versions, P1's environment with LKAI_PHASES=1, and P1's pinned runtime version ARN. They read P1's weights object
(same bucket, key and bytes), so the download path is P1's. The zip goes under e7/ in P1's bucket; P1's objects are
untouched. The role may GetObject only the weights key and write only to the lkai-f768-* log groups. State goes to
e7/build/deployment.json (git-ignored), in the shape baseline/bench.py's Bench reads.
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
import deploy as p1  # noqa: E402  (baseline/deploy.py: wait_updated, b64_to_hex, file_sha256, RUNTIME, MEMORY)

PREFIX = "lkai-f768"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E7"}
SWITCH = "LKAI_SKIP_TIED_OUTPUT"
ARMS = {"ctl": None, "skip": "1"}  # the value of SWITCH; None = unset
HANDLER = "e7_entry.handler"
BUILD = os.path.join(HERE, "build")


def state_path(build: str) -> str:
    return os.path.join(build, "deployment.json")


def load_state(build: str = BUILD) -> dict:
    p = state_path(build)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(build: str, st: dict) -> None:
    os.makedirs(build, exist_ok=True)
    tmp = state_path(build) + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, state_path(build))


def arm_environment(p1_env: dict, arm: str) -> dict:
    """P1's environment (without its COLD_TAG and LKAI_PHASES), LKAI_PHASES=1, and the arm's one variable."""
    env = {k: v for k, v in p1_env.items() if k not in ("COLD_TAG", "LKAI_PHASES", SWITCH)}
    env |= {"COLD_TAG": "0", "LKAI_PHASES": "1"}
    if ARMS[arm] is not None:
        env[SWITCH] = ARMS[arm]
    return env


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
    st["memory_mb"] = c["MemorySize"]


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
                "Resource": [
                    f"arn:aws:logs:{st['region']}:*:log-group:/aws/lambda/{PREFIX}-{a}:*"
                    for a in ARMS
                ],
            },
        ],
    }
    iam.put_role_policy(RoleName=name, PolicyName="lkai-e7", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict, build: str) -> None:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(build, "manifest.json")))
    if m.get("dirty_inputs"):
        sys.exit(f"refusing to deploy a dirty build: {m['dirty_inputs'][:5]}")
    z = m["function_zip"]["sha256"]
    if p1.file_sha256(os.path.join(build, "function.zip")) != z:
        sys.exit("function.zip does not match manifest.json; rerun build.sh")
    if m["model"]["gguf_sha256"] not in st["model_key"]:
        sys.exit(f"P1's weights key {st['model_key']} is not the manifest's GGUF")
    st["manifest"] = m
    key = f"e7/code/{z}/function.zip"
    s3.upload_file(os.path.join(build, "function.zip"), st["bucket"], key)
    st["zip_key"] = key
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    for arm in ARMS:
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
            Handler=HANDLER,
            Role=role,
            MemorySize=st["memory_mb"],
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": arm_environment(st["p1_environment"], arm)},
            Description=f"lambda-kleidiai E7 {arm}: zip {z[:12]}, {SWITCH}={ARMS[arm] or 'unset'}",
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
            "memory_mb": c["MemorySize"],
        }
        print("function", name, z[:12], c["CodeSize"])


def teardown(sess, st: dict, everything: bool) -> list[str]:
    """Delete every published version (and with --all the functions, role, log groups and zip). Each step is
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
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-e7")
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
        if st.get("zip_key"):
            try:
                s3.delete_object(Bucket=st["bucket"], Key=st["zip_key"])
            except Exception as e:
                failures.append(f"{st['zip_key']}: {e}")
        st["functions"], st["zip_key"] = {}, None
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "show", "teardown"])
    ap.add_argument("--build", default=BUILD)
    ap.add_argument("--p1-function", default="lkai-p1-inst")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not a.profile:
        sys.exit("--profile or AWS_PROFILE is required: no step uses the default credential chain")
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = load_state(a.build)
    st.setdefault("region", a.region)
    st.setdefault("prefix", PREFIX)
    if st["region"] != a.region:
        sys.exit(f"{state_path(a.build)} is for {st['region']}")
    p1.check_account(sess, st)
    failures: list[str] = []
    try:
        if a.cmd == "up":
            adopt_p1(sess, st, a.p1_function)
            up(sess, st, a.build)
        elif a.cmd == "teardown":
            failures = teardown(sess, st, a.all)
    finally:
        save_state(a.build, st)
    if a.cmd == "show":
        print(
            json.dumps({k: v for k, v in st.items() if k not in ("manifest", "account")}, indent=1)
        )
    if failures:
        sys.exit("teardown failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
