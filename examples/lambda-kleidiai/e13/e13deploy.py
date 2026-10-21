"""Deploy experiment E13: four functions with P1's configuration and one zip. They differ only in libgomp's
wait policy, set through the function environment.

  python e13deploy.py up        [--p1-function lkai-p1-inst]   # upload the zip, role, log groups, functions
  python e13deploy.py show
  python e13deploy.py teardown  [--all]                       # published versions; --all: functions, role, zip

The variable is how long an idle OpenMP worker spins before it sleeps on a futex (libgomp's gomp_spin_count_var,
gcc 11 libgomp/env.c and config/linux/wait.h). ggml's barrier is `#pragma omp barrier` (ggml-cpu.c, ggml_barrier),
so every barrier wait in the warm forward goes through it. Thread count is not varied: every arm runs P1's 6
threads (handler.py THREADS = os.cpu_count()).

  arm      environment                    spin count                 throttled spin count
  ctl      (none; P1 as deployed)         300,000 (libgomp default)  100
  passive  OMP_WAIT_POLICY=passive        0 (sleep at once)          0
  spin30k  GOMP_SPINCOUNT=30000           30,000                     100
  active   OMP_WAIT_POLICY=active         30,000,000,000             1,000

The throttled count applies only when libgomp manages more threads than CPUs; with 6 threads on 6 CPUs it does not.
Every arm also sets OMP_DISPLAY_ENV=VERBOSE, so libgomp prints the effective GOMP_SPINCOUNT to the log at start-up:
proof that the variable reached the library, on every arm alike.

Everything else is copied from the deployed P1 function: python3.12, arm64, 10,240 MB, 512 MB ephemeral storage,
900 s timeout, SnapStart on published versions, P1's environment with LKAI_PHASES=1, P1's pinned runtime version
ARN and P1's weights object (read-only). The zip goes to the e13/ prefix of P1's bucket. The role may GetObject only
the weights key and write only to lkai-e13-* log groups. State is in build/deployment.json (git-ignored), in the
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

PREFIX = "lkai-e13"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E13"}
COMMON_ENV = {"OMP_DISPLAY_ENV": "VERBOSE"}
ARMS: dict[str, dict[str, str]] = {
    "ctl": {},
    "passive": {"OMP_WAIT_POLICY": "passive"},
    "spin30k": {"GOMP_SPINCOUNT": "30000"},
    "active": {"OMP_WAIT_POLICY": "active"},
}
# libgomp 11 (env.c initialize_env): the spin count each arm's environment yields, for the log check.
EXPECTED_SPINCOUNT = {"ctl": 300000, "passive": 0, "spin30k": 30000, "active": 30000000000}
# Variables that would make an arm differ in more than the wait policy. P1's environment must not set any of them.
OMP_KEYS = {
    "OMP_WAIT_POLICY",
    "GOMP_SPINCOUNT",
    "OMP_NUM_THREADS",
    "THREADS",
    "OMP_DISPLAY_ENV",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "KMP_BLOCKTIME",
}
BUILD = os.path.join(HERE, "build")


def arm_environment(p1_env: dict, arm: str) -> dict:
    """P1's environment, the shared settings, and this arm's wait-policy variables. Nothing else differs."""
    if clash := sorted(OMP_KEYS & set(p1_env)):
        sys.exit(
            f"P1's environment already sets {clash}: the arms would not differ in one variable"
        )
    return dict(p1_env, COLD_TAG="0", LKAI_PHASES="1", E13_ARM=arm, **COMMON_ENV, **ARMS[arm])


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
    iam.put_role_policy(RoleName=name, PolicyName="lkai-e13", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict) -> None:
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(BUILD, "manifest.json")))
    if m["p1_manifest"]["function_zip"]["sha256"] != st["p1_function_code_sha256"]:
        sys.exit(
            f"build/e13.zip is built from P1 zip {m['p1_manifest']['function_zip']['sha256'][:12]}, but "
            f"{st['p1_function']} runs {st['p1_function_code_sha256'][:12]}"
        )
    st["manifest"] = m
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    z = m["zip"]["sha256"]
    key = f"e13/code/{z}/function.zip"
    s3.upload_file(os.path.join(BUILD, "e13.zip"), st["bucket"], key)
    st["zip_key"] = key
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
            Handler="e13_entry.handler",
            Role=role,
            MemorySize=p1.MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": arm_environment(st["p1_environment"], arm)},
            Description=f"lambda-kleidiai E13 {arm}: P1 zip + metadata wrapper, OpenMP wait {ARMS[arm] or 'default'}",
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
        print("function", name, z[:12], ARMS[arm] or "default")


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
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-e13")
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
