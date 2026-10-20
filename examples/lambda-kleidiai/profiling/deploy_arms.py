"""Deploy the P3 profiler arms beside P1's REPORT-only control and write a bench.py deployment state for them.

  python deploy_arms.py up       --p1-state <P1 build/deployment.json> --build <dir with prof.zip, telemetry-layer.zip,
                                 prof-manifest.json> [--arms rep,cprofile,pyspy,sampler,xray,telemetry] [--region]
  python deploy_arms.py teardown [--all] --build <dir>

Each arm is its own function, `lkai-p3-<arm>`, created with P1's exact configuration (runtime python3.12 pinned to P1's
runtime version, arm64, 10,240 MB, 512 MB ephemeral, 900 s, SnapStart on published versions, P1's environment and the
same GGUF object). Arms that need no in-function code run P1's own zip unchanged; the others run prof.zip (P1's zip
bytes plus added files). They differ from P1's REPORT-only function in:
  handler      lkai_prof.handler (wraps lkai_entry.handler) on none/cprofile/pyspy/sampler; lkai_entry.handler else
  LKAI_PROF    none | cprofile | pyspy | sampler on those four arms
  LKAI_PHASES  1 on the four lkai_prof arms (their phase records show where the profiler's time goes); 0 elsewhere
  tracing      Active on the xray arm only
  layers       the telemetry extension layer on the telemetry arm only
  role         lkai-p3-fn: GetObject on the one GGUF key, logs on /aws/lambda/lkai-p3-*, and X-Ray PutTraceSegments
               (used by the xray arm only)
`rep` (lkai-p3-rep) is P1's REPORT-only arm rebuilt from P1's zip: the overhead control. P3 creates its own rather than
using lkai-p1-rep, so no other card's run can change P3's control (and P3 cannot change P1's).

The state file it writes (build/deployment.json) has P1's layout, so baseline/bench.py runs every arm unchanged:
  python baseline/bench.py --build <dir> --arms rep,cprofile,... --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
import deploy  # noqa: E402

ARMS = {
    # arm: (handler, LKAI_PHASES, LKAI_PROF, tracing, telemetry layer, zip). `rep` is P1's REPORT-only control,
    # recreated here from P1's own zip (not P1's function, which P1's own runs change), so P3 shares nothing mutable.
    "rep": ("lkai_entry.handler", "0", None, False, False, "p1"),
    "none": ("lkai_prof.handler", "1", "none", False, False, "prof"),
    "cprofile": ("lkai_prof.handler", "1", "cprofile", False, False, "prof"),
    "pyspy": ("lkai_prof.handler", "1", "pyspy", False, False, "prof"),
    "sampler": ("lkai_prof.handler", "1", "sampler", False, False, "prof"),
    "xray": ("lkai_entry.handler", "0", None, True, False, "p1"),
    "telemetry": ("lkai_entry.handler", "0", None, False, True, "p1"),
}
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-P3"}
PREFIX = "lkai-p3"


def ensure_role(iam, p1: dict, region: str) -> str:
    name = f"{PREFIX}-fn"
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])["Role"]["Arn"]
        time.sleep(10)
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{p1['bucket']}/{p1['model_key']}"},
        {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": f"arn:aws:logs:{region}:*:log-group:/aws/lambda/{PREFIX}-*:*"},
        {"Effect": "Allow", "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"], "Resource": "*"},
    ]}
    iam.put_role_policy(RoleName=name, PolicyName="lkai-p3", PolicyDocument=json.dumps(policy))
    return arn


def up(sess, p1: dict, build: str, arms: list[str]) -> dict:
    lam, logs, iam, s3 = sess.client("lambda"), sess.client("logs"), sess.client("iam"), sess.client("s3")
    region = sess.region_name
    pm = json.load(open(os.path.join(build, "prof-manifest.json")))
    zip_key = f"code/{pm['prof_zip_sha256']}/prof.zip"
    for path, key in ((os.path.join(build, "prof.zip"), zip_key),
                      (os.path.join(build, "prof-manifest.json"), f"code/{pm['prof_zip_sha256']}/prof-manifest.json")):
        try:
            s3.head_object(Bucket=p1["bucket"], Key=key)
        except s3.exceptions.ClientError:
            s3.upload_file(path, p1["bucket"], key)
    role = ensure_role(iam, p1, region)
    layer_arn = None
    if any(ARMS[a][4] for a in arms):
        layer_arn = lam.publish_layer_version(
            LayerName=f"{PREFIX}-telemetry", Description=f"lkai-telemetry {pm['telemetry_layer_sha256'][:12]}",
            Content={"ZipFile": open(os.path.join(build, "telemetry-layer.zip"), "rb").read()},
            CompatibleArchitectures=["arm64"], CompatibleRuntimes=["python3.12"])["LayerVersionArn"]
    p1_rep = lam.get_function_configuration(FunctionName=p1["functions"]["rep"]["name"])
    env0 = {k: v for k, v in p1_rep["Environment"]["Variables"].items()}
    st = {k: p1[k] for k in ("region", "bucket", "model_key", "runtime_version_arn")}
    st.update(prefix=PREFIX, zip_key=zip_key, role=role, telemetry_layer=layer_arn, p1_functions=p1["functions"],
              prof_manifest=pm, manifest=p1["manifest"], functions={})
    zips = {"p1": (p1["zip_key"], p1["manifest"]["function_zip"]["sha256"]), "prof": (zip_key, pm["prof_zip_sha256"])}
    for arm in arms:
        handler, phases_flag, prof, tracing, tel, zkind = ARMS[arm]
        key, want_sha = zips[zkind]
        name = f"{PREFIX}-{arm}"
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        env = dict(env0, LKAI_PHASES=phases_flag, COLD_TAG="0")
        if prof:
            env["LKAI_PROF"] = prof
        cfg = dict(FunctionName=name, Runtime=p1_rep["Runtime"], Handler=handler, Role=role,
                   MemorySize=p1_rep["MemorySize"], EphemeralStorage=p1_rep["EphemeralStorage"],
                   Timeout=p1_rep["Timeout"], SnapStart={"ApplyOn": "PublishedVersions"},
                   Environment={"Variables": env}, TracingConfig={"Mode": "Active" if tracing else "PassThrough"},
                   Layers=[layer_arn] if tel else [],
                   Description=f"lambda-kleidiai P3 profiler arm {arm}; code {want_sha[:12]}")
        for attempt in range(6):
            try:
                try:
                    lam.get_function(FunctionName=name)
                    lam.update_function_code(FunctionName=name, S3Bucket=p1["bucket"], S3Key=key)
                    deploy.wait_updated(lam, name)
                    lam.update_function_configuration(**cfg)
                except lam.exceptions.ResourceNotFoundException:
                    lam.create_function(**cfg, Architectures=p1_rep["Architectures"],
                                        Code={"S3Bucket": p1["bucket"], "S3Key": key}, Tags=TAGS)
                break
            except lam.exceptions.InvalidParameterValueException as e:
                if attempt == 5:
                    raise
                print("retrying:", e)
                time.sleep(10)
        c = deploy.wait_updated(lam, name)
        lam.put_runtime_management_config(FunctionName=name, UpdateRuntimeOn="Manual",
                                          RuntimeVersionArn=p1["runtime_version_arn"])
        if deploy.b64_to_hex(c["CodeSha256"]) != want_sha:
            sys.exit(f"{name}: deployed code is not the expected {zkind} zip")
        st["functions"][arm] = {"name": name, "arn": c["FunctionArn"], "code_sha256_b64": c["CodeSha256"],
                                "code_sha256_hex": want_sha, "code_size": c["CodeSize"], "handler": handler,
                                "lkai_prof": prof, "tracing": tracing, "telemetry_layer": bool(tel)}
        print("function", name, handler, prof or "", "tracing" if tracing else "", "telemetry" if tel else "")
    return st


def teardown(sess, st: dict, everything: bool) -> None:
    lam = sess.client("lambda")
    for f in st.get("functions", {}).values():
        if not f["name"].startswith(PREFIX + "-"):
            continue  # never touch another card's function
        try:
            for p in lam.get_paginator("list_versions_by_function").paginate(FunctionName=f["name"]):
                for v in p["Versions"]:
                    if v["Version"] != "$LATEST":
                        lam.delete_function(FunctionName=f["name"], Qualifier=v["Version"])
                        print("deleted", f["name"], v["Version"])
            if everything:
                lam.delete_function(FunctionName=f["name"])
                print("deleted", f["name"])
        except lam.exceptions.ResourceNotFoundException:
            pass
    if everything:
        iam = sess.client("iam")
        try:
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-p3")
            iam.delete_role(RoleName=f"{PREFIX}-fn")
        except iam.exceptions.NoSuchEntityException:
            pass
        if st.get("telemetry_layer"):
            name, ver = st["telemetry_layer"].rsplit(":", 1)
            lam.delete_layer_version(LayerName=name, VersionNumber=int(ver))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "teardown"])
    ap.add_argument("--p1-state")
    ap.add_argument("--build", required=True)
    ap.add_argument("--arms", default="rep,none,cprofile,pyspy,sampler,xray,telemetry")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    path = os.path.join(a.build, "deployment.json")
    if a.cmd == "up":
        p1 = json.load(open(a.p1_state))
        if p1["region"] != a.region:
            sys.exit(f"P1 state is for {p1['region']}")
        st = up(sess, p1, a.build, a.arms.split(","))
        json.dump(st, open(path, "w"), indent=1)
        print("state", path)
    else:
        teardown(sess, json.load(open(path)), a.all)


if __name__ == "__main__":
    main()
