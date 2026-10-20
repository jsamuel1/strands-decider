"""Package and deploy experiment E3: one function per download strategy, plus a control running P1's exact zip.

  python e3deploy.py build     --p1-zip <P1 function.zip> --p1-manifest <its manifest.json> --wheel <awscrt aarch64 whl>
  python e3deploy.py up        [--arms ...]          # upload the E3 zip, role, log groups, functions, runtime pin
  python e3deploy.py show
  python e3deploy.py teardown  [--all]               # published versions (SnapStart cache billing); --all: everything

The E3 zip is P1's function.zip byte for byte (every entry copied with its metadata), plus e3dl.py, e3_entry.py and
the awscrt wheel's files. Nothing P1 ships is changed; the handler becomes e3_entry.handler, which installs the
E3_DL strategy and then imports P1's lkai_entry. The control function runs P1's zip itself (handler
lkai_entry.handler), so the A/A arm (E3 zip, P1's loader) separates "new zip and wrapper" from "new strategy".

Every function has P1's configuration (python3.12, arm64, 10,240 MB, 512 MB ephemeral, 900 s, SnapStart on published
versions, P1's environment with LKAI_PHASES=1, the same runtime version ARN as P1's functions) and reads P1's weights
object (same bucket, key and bytes) read-only. Only E3_DL differs. The E3 zip goes to the e3/ prefix of P1's bucket:
P1's own objects are not touched. The role may GetObject only the weights key and write only lkai-e3-* log groups.
State goes to build/deployment.json (git-ignored), in the shape baseline/bench.py's Bench reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
import deploy as p1  # noqa: E402  (baseline/deploy.py: wait_updated, b64_to_hex, file_sha256)

PREFIX = "lkai-e3"
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E3"}
# arm -> E3_DL. "ctl" runs P1's zip and handler, so it has no E3_DL.
ARMS = {
    "ctl": None,
    "ldr": "loader",  # P1's own loader through E3's zip and wrapper: the A/A check
    "aa": "threads:64:32",  # P1's setting, re-implemented with counters: checks the re-implementation
    "t16x64": "threads:16:64",
    "t8x128": "threads:8:128",
    "t32x64": "threads:32:64",
    "t64x64": "threads:64:64",
    "crt8": "crt:8:25",
    "crt16": "crt:16:25",
    "crtf16": "crtfile:16:25",
}
E3_FILES = ("e3dl.py", "e3_entry.py")


def state_path() -> str:
    return os.path.join(HERE, "build", "deployment.json")


def load_state() -> dict:
    p = state_path()
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(state_path()), exist_ok=True)
    tmp = state_path() + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, state_path())


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def build(p1_zip: str, p1_manifest: str, wheel: str, out: str) -> dict:
    """E3's zip: P1's entries unchanged, then E3's files and awscrt. Refuses a name collision."""
    m1 = json.load(open(p1_manifest))
    if p1.file_sha256(p1_zip) != m1["function_zip"]["sha256"]:
        sys.exit(f"{p1_zip} is not the zip {p1_manifest} describes")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with zipfile.ZipFile(p1_zip) as src, zipfile.ZipFile(wheel) as whl, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        names = set()
        for info in src.infolist():
            dst.writestr(info, src.read(info))
            names.add(info.filename)
        added = []
        for f in E3_FILES:
            if f in names:
                sys.exit(f"{f} already in the P1 zip")
            info = zipfile.ZipInfo(f, date_time=(2026, 10, 20, 0, 0, 0))
            info.external_attr = 0o644 << 16
            dst.writestr(info, open(os.path.join(HERE, f), "rb").read(), zipfile.ZIP_DEFLATED)
            added.append(f)
        for info in whl.infolist():
            if info.filename.split("/", 1)[0].endswith(".dist-info") and not info.filename.endswith(
                ("METADATA", "RECORD", "WHEEL")
            ):
                continue
            if info.filename in names:
                sys.exit(f"{info.filename} already in the P1 zip")
            dst.writestr(info, whl.read(info))
            added.append(info.filename)
    os.replace(tmp, out)
    m = {
        "schema": "lkai-e3-build-manifest/1",
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "p1_manifest": m1,
        "model": m1["model"],
        "llama_cpp": m1["llama_cpp"],
        "images": m1["images"],
        "toolchain": m1["toolchain"],
        "assets_sha256": m1["assets_sha256"],
        "function_zip": {
            "bytes": os.path.getsize(out),
            "sha256": p1.file_sha256(out),
            "handler": "e3_entry.handler",
            "base": "P1 function.zip " + m1["function_zip"]["sha256"],
            "added": added,
        },
        "e3_sources_sha256": {
            f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in E3_FILES
        },
        "awscrt_wheel": {"file": os.path.basename(wheel), "sha256": p1.file_sha256(wheel)},
    }
    json.dump(m, open(os.path.join(os.path.dirname(out), "manifest.json"), "w"), indent=1)
    print("built", out, m["function_zip"]["bytes"], m["function_zip"]["sha256"])
    return m


def ensure_role(iam, st: dict) -> str:
    name = f"{PREFIX}-fn"
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
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
    iam.put_role_policy(RoleName=name, PolicyName="lkai-e3", PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def up(sess, st: dict, arms: list[str]) -> None:
    """Upload the E3 zip, then create or update one function per arm with P1's configuration."""
    lam, logs, s3 = sess.client("lambda"), sess.client("logs"), sess.client("s3")
    m = json.load(open(os.path.join(HERE, "build", "manifest.json")))
    z = m["function_zip"]["sha256"]
    st["e3_zip_key"] = f"e3/code/{z}/function.zip"
    s3.upload_file(os.path.join(HERE, "build", "function.zip"), st["bucket"], st["e3_zip_key"])
    st["manifest"] = m
    role = ensure_role(sess.client("iam"), st)
    st.setdefault("functions", {})
    for arm in arms:
        spec = ARMS[arm]
        name = f"{PREFIX}-{arm}"
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        env = dict(st["p1_environment"], COLD_TAG="0", LKAI_PHASES="1")
        if spec:
            env["E3_DL"] = spec
        key = st["p1_zip_key"] if spec is None else st["e3_zip_key"]
        cfg = dict(
            FunctionName=name,
            Runtime=p1.RUNTIME,
            Architectures=["arm64"],
            Handler="lkai_entry.handler" if spec is None else "e3_entry.handler",
            Role=role,
            MemorySize=p1.MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": env},
            Description=f"lambda-kleidiai E3 {arm}: E3_DL={spec or 'P1 zip, P1 loader'}",
        )
        for attempt in range(6):
            try:
                try:
                    lam.get_function(FunctionName=name)
                    lam.update_function_code(FunctionName=name, S3Bucket=st["bucket"], S3Key=key)
                    p1.wait_updated(lam, name)
                    lam.update_function_configuration(**{k: v for k, v in cfg.items() if k != "Architectures"})
                except lam.exceptions.ResourceNotFoundException:
                    lam.create_function(**cfg, Code={"S3Bucket": st["bucket"], "S3Key": key}, Tags=TAGS)
                break
            except lam.exceptions.InvalidParameterValueException as e:
                if attempt == 5:
                    raise
                print("retrying:", e)
                time.sleep(10)
        c = p1.wait_updated(lam, name)
        lam.put_runtime_management_config(
            FunctionName=name, UpdateRuntimeOn="Manual", RuntimeVersionArn=st["runtime_version_arn"]
        )
        want = st["p1_zip_sha256"] if spec is None else z
        if p1.b64_to_hex(c["CodeSha256"]) != want:
            sys.exit(f"{name}: deployed code {c['CodeSha256']} is not {want}")
        st["functions"][arm] = {
            "name": name,
            "arn": c["FunctionArn"],
            "e3_dl": spec,
            "code_sha256": want,
            "code_size": c["CodeSize"],
        }
        print("function", name, spec, want[:12], c["CodeSize"])


def adopt_p1(sess, st: dict, p1_function: str) -> None:
    """Read P1's deployed identity (bucket, weights key, zip key, environment, runtime ARN) from its function."""
    lam = sess.client("lambda")
    c = lam.get_function_configuration(FunctionName=p1_function)
    rm = lam.get_runtime_management_config(FunctionName=p1_function)
    env = dict(c["Environment"]["Variables"])
    st["bucket"], st["model_key"] = env["MODEL_BUCKET"], env["MODEL_KEY"]
    st["p1_environment"] = {k: v for k, v in env.items() if k not in ("COLD_TAG", "LKAI_PHASES")}
    st["runtime_version_arn"] = rm["RuntimeVersionArn"]
    st["p1_zip_sha256"] = p1.b64_to_hex(c["CodeSha256"])
    st["p1_zip_key"] = f"code/{st['p1_zip_sha256']}/function.zip"
    st["p1_function"] = p1_function


def teardown(sess, st: dict, everything: bool) -> None:
    lam = sess.client("lambda")
    for f in st.get("functions", {}).values():
        try:
            for page in lam.get_paginator("list_versions_by_function").paginate(FunctionName=f["name"]):
                for v in page["Versions"]:
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
            iam.delete_role_policy(RoleName=f"{PREFIX}-fn", PolicyName="lkai-e3")
            iam.delete_role(RoleName=f"{PREFIX}-fn")
        except iam.exceptions.NoSuchEntityException:
            pass
        if st.get("e3_zip_key"):
            sess.client("s3").delete_object(Bucket=st["bucket"], Key=st["e3_zip_key"])
        st["functions"] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "up", "show", "teardown"])
    ap.add_argument("--p1-zip")
    ap.add_argument("--p1-manifest")
    ap.add_argument("--wheel")
    ap.add_argument("--p1-function", default="lkai-p1-inst")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.p1_zip, a.p1_manifest, a.wheel, os.path.join(HERE, "build", "function.zip"))
        return
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = load_state()
    st.setdefault("region", a.region)
    st.setdefault("prefix", PREFIX)
    if st["region"] != a.region:
        sys.exit(f"{state_path()} is for {st['region']}")
    try:
        if a.cmd == "up":
            adopt_p1(sess, st, a.p1_function)
            up(sess, st, a.arms.split(","))
        elif a.cmd == "teardown":
            teardown(sess, st, a.all)
    finally:
        save_state(st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))


if __name__ == "__main__":
    main()
