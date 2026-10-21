"""Deploy experiment E15: three functions with P1's configuration. They differ only in the CPU variant libraries
and one switch.

  python e15deploy.py up        [--p1-function lkai-p1-inst]   # upload both zips, role, log groups, functions
  python e15deploy.py show
  python e15deploy.py teardown  [--all]                       # published versions; --all: functions, role, zips

  arm   zip        LKAI_E15_A   what it isolates
  ctl   ctl.zip    unset        P1's zip + the metadata wrapper
  off   cand.zip   unset        candidate-A libraries, switch off: the dormant build (off - ctl)
  a     cand.zip   1            the same zip, switch on: candidate A itself (a - off; also a - ctl)

Candidate A (e15/patch_a.py) lets the SVE / i8mm CPU variants (armv8.2_3, armv8.6_1, armv8.6_2) use llamafile's F32
tinyBLAS sgemm for F32 x F32 MUL_MAT. Upstream f74e2fa6a compiles that call out of those variants; the deployed
non-SVE variants already make it. Quantised MUL_MAT (the Q8_0 weights on KleidiAI) is unchanged.

Everything else is copied from the deployed P1 function: python3.12, arm64, 10,240 MB, 512 MB ephemeral storage,
900 s timeout, SnapStart on published versions, P1's environment with LKAI_PHASES=1, P1's pinned runtime version
ARN and P1's weights object (read-only). The zips go to the e15/ prefix of P1's bucket. The role may GetObject only
the weights key and write only to lkai-89cf-* log groups. State is in build/deployment.json (git-ignored), in the
shape baseline/bench.py's Bench reads.
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

PREFIX = "lkai-89cf"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E15"}
ARMS: dict[str, dict[str, str]] = {
    "ctl": {},
    "off": {},
    "a": {"LKAI_E15_A": "1"},
}
ARM_ZIP = {"ctl": "ctl", "off": "cand", "a": "cand"}
# Variables that would make an arm differ in more than the one switch. P1's environment must not set any of them.
SWITCH_KEYS = {"LKAI_E15_A", "LKAI_E15_B", "LKAI_GDN_ROWS", "LKAI_NODEPROF", "E15_ARM"}
BUILD = os.path.join(HERE, "build")


def arm_environment(p1_env: dict, arm: str) -> dict:
    """P1's environment and this arm's switch. Nothing else differs (E15_ARM is an observation label)."""
    if clash := sorted(SWITCH_KEYS & set(p1_env)):
        sys.exit(
            f"P1's environment already sets {clash}: the arms would not differ in one variable"
        )
    return dict(p1_env, COLD_TAG="0", LKAI_PHASES="1", E15_ARM=arm, **ARMS[arm])


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
    iam.put_role_policy(RoleName=name, PolicyName="lkai-89cf", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict) -> None:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(BUILD, "manifest.json")))
    if m["p1_manifest"]["function_zip"]["sha256"] != st["p1_function_code_sha256"]:
        sys.exit(
            f"build/ zips are built from P1 zip {m['p1_manifest']['function_zip']['sha256'][:12]}, but "
            f"{st['p1_function']} runs {st['p1_function_code_sha256'][:12]}"
        )
    st["manifest"] = m
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    st["zip_keys"] = {}
    for zname, zm in m["zips"].items():
        key = f"e15/code/{zm['sha256']}/function.zip"
        s3.upload_file(os.path.join(BUILD, f"{zname}.zip"), st["bucket"], key)
        st["zip_keys"][zname] = key
    for arm in ARMS:
        z = m["zips"][ARM_ZIP[arm]]["sha256"]
        key = st["zip_keys"][ARM_ZIP[arm]]
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
            Handler="e15_entry.handler",
            Role=role,
            MemorySize=p1.MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": arm_environment(st["p1_environment"], arm)},
            Description=f"lambda-kleidiai E15 {arm}: {ARM_ZIP[arm]}.zip, {ARMS[arm] or 'LKAI_E15_A unset'}",
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
        print("function", name, ARM_ZIP[arm], z[:12], ARMS[arm] or "switch unset")


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
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-89cf")
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
        for key in (st.get("zip_keys") or {}).values():
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
