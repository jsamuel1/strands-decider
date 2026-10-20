"""Deploy E1 (weights in an S3 Express One Zone directory bucket) next to its own general-purpose control.

  python deploy_e1.py up       [--source s3://<p1-bucket>] [--build build/] [--az-id use2-az1]
  python deploy_e1.py show
  python deploy_e1.py teardown [--all]          # versions (SnapStart cache billing); --all: everything E1 made

Four functions, all from P1's function.zip (same bytes, checked against its manifest), all instrumented
(LKAI_PHASES=1), all on P1's pinned runtime version. They differ only in the two factors below:

  arm   weights bucket                                  network
  gp    E1's own S3 general-purpose bucket (Standard)   no VPC (as P1)
  xz    S3 Express One Zone directory bucket, <az-id>   no VPC (as P1)
  gpv   the general-purpose bucket                      VPC: one private subnet in <az-id>, S3 gateway endpoint
  xzv   the directory bucket                            VPC: same subnet, S3 Express gateway endpoint

The loader (examples/lambda/app/loader.py) is unchanged: botocore recognises the `--x-s3` bucket name, calls
CreateSession and sends HEAD and the ranged GETs to the zonal endpoint. The only E1-specific grant is
s3express:CreateSession (ReadOnly) on the directory bucket. Every E1 resource is new and tagged
Experiment=lambda-kleidiai-E1: nothing P1 or another experiment uses is modified. The network is a new subnet with
its own route table, a security group whose only egress is HTTPS to the S3 and S3 Express prefix lists, and gateway
endpoints attached only to that route table.

Weights and zip are copied server-side from --source (P1's bucket, read only), or uploaded from a local P1 build
(--build, after baseline/build.sh). Each staged GGUF is read back in full and its SHA-256 compared with the manifest
before any function is created.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
import deploy as p1  # noqa: E402

TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E1"}
RUNTIME = p1.RUNTIME
MEMORY = p1.MEMORY
# P1's pinned managed runtime version (baseline/results/20261020T080835Z): a pinned condition, not rediscovered.
P1_RUNTIME_VERSION_ARN = "arn:aws:lambda:us-east-2::runtime:0fec05b8b43b0ecfbabe4fd585b35e423f655861556712f8a10b0e8739c36c07"
GGUF_NAME = "s2b-h512-Q8_0.gguf"
# arm -> (bucket kind, VPC attached)
ARMS = {"gp": ("gp", False), "xz": ("xz", False), "gpv": ("gp", True), "xzv": ("xz", True)}
SUBNET_CIDR = "172.31.64.0/24"


def directory_bucket_name(base: str, az_id: str) -> str:
    """S3 Express One Zone names end in --<az-id>--x-s3; botocore keys its CreateSession routing on that suffix."""
    return f"{base}--{az_id}--x-s3"


def is_directory_bucket(name: str) -> bool:
    return name.endswith("--x-s3")


VPC_ACCESS_POLICY_ARN = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
# The single execution role earlier revisions shared across all four arms (it carried the VPC policy for every arm).
# Kept only so `up` and `teardown --all` can retire it.
LEGACY_SHARED_ROLE_SUFFIX = "fn"


def role_name(st: dict, arm: str) -> str:
    """One execution role per arm, so each function holds only the grants its own bucket and network need."""
    return f"{st['prefix']}-{arm}-fn"


def role_policy(region: str, account: str, st: dict, arm: str) -> dict:
    """P1's role scoped to one arm: read its own weights object, write its own log group.

    A general-purpose arm gets exactly P1's statements. A directory-bucket arm replaces s3:GetObject with the one
    E1-specific grant, s3express:CreateSession on the directory bucket, ReadOnly sessions only (object reads on a
    directory bucket are authorised by the session, not by s3:GetObject).
    """
    kind, _ = ARMS[arm]
    read = (
        {
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{st['bucket']}/{st['model_key']}",
        }
        if kind == "gp"
        else {
            "Effect": "Allow",
            "Action": "s3express:CreateSession",
            "Resource": f"arn:aws:s3express:{region}:{account}:bucket/{st['xz_bucket']}",
            "Condition": {"StringEquals": {"s3express:SessionMode": "ReadOnly"}},
        }
    )
    return {
        "Version": "2012-10-17",
        "Statement": [
            read,
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/lambda/{st['prefix']}-{arm}:*"
                ],
            },
        ],
    }


def function_config(st: dict, arm: str, role: str) -> dict:
    """P1's configuration (baseline/deploy.py functions()) with this arm's bucket and VPC."""
    kind, vpc = ARMS[arm]
    env = {
        "MODEL_BUCKET": st["bucket"] if kind == "gp" else st["xz_bucket"],
        "MODEL_KEY": st["model_key"],
        "DEC_KV": "f32",
        "DEC_UBATCH": "512",
        "N_SEQ": "17",
        "LKAI_PHASES": "1",
        "COLD_TAG": "0",
    }
    cfg = {
        "FunctionName": f"{st['prefix']}-{arm}",
        "Runtime": RUNTIME,
        "Architectures": ["arm64"],
        "Handler": "lkai_entry.handler",
        "Role": role,
        "MemorySize": MEMORY,
        "EphemeralStorage": {"Size": 512},
        "Timeout": 900,
        "SnapStart": {"ApplyOn": "PublishedVersions"},
        "Environment": {"Variables": env},
        "VpcConfig": (
            {"SubnetIds": [st["vpc"]["subnet_id"]], "SecurityGroupIds": [st["vpc"]["sg_id"]]}
            if vpc
            else {"SubnetIds": [], "SecurityGroupIds": []}
        ),
        "Description": f"lambda-kleidiai E1 ({arm}: {kind} bucket, {'VPC ' + st['az_id'] if vpc else 'no VPC'})",
    }
    return cfg


# ---------------------------------------------------------------------------- network


def _tag_spec(kind: str, name: str) -> list[dict]:
    tags = [{"Key": k, "Value": v} for k, v in TAGS.items()] + [{"Key": "Name", "Value": name}]
    return [{"ResourceType": kind, "Tags": tags}]


def ensure_network(ec2, st: dict) -> dict:
    """A private subnet in st['az_id'] of the default VPC, reachable only to S3 and S3 Express."""
    if st.get("vpc", {}).get("sg_id"):
        return st["vpc"]
    v = st.setdefault("vpc", {})
    region, prefix = st["region"], st["prefix"]
    if "vpc_id" not in v:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
        if not vpcs:
            sys.exit("no default VPC; pass an existing VPC by editing deployment.json")
        v["vpc_id"] = vpcs[0]["VpcId"]
    v.setdefault("subnet_cidr", st.get("subnet_cidr", SUBNET_CIDR))
    if "subnet_id" not in v:
        v["subnet_id"] = ec2.create_subnet(
            VpcId=v["vpc_id"],
            CidrBlock=v["subnet_cidr"],
            AvailabilityZoneId=st["az_id"],
            TagSpecifications=_tag_spec("subnet", f"{prefix}-private-{st['az_id']}"),
        )["Subnet"]["SubnetId"]
    if "route_table_id" not in v:
        v["route_table_id"] = ec2.create_route_table(
            VpcId=v["vpc_id"], TagSpecifications=_tag_spec("route-table", f"{prefix}-private")
        )["RouteTable"]["RouteTableId"]
        ec2.associate_route_table(RouteTableId=v["route_table_id"], SubnetId=v["subnet_id"])
    eps = v.setdefault("endpoints", {})
    for svc in ("s3", "s3express"):
        if svc not in eps:
            eps[svc] = ec2.create_vpc_endpoint(
                VpcId=v["vpc_id"],
                ServiceName=f"com.amazonaws.{region}.{svc}",
                VpcEndpointType="Gateway",
                RouteTableIds=[v["route_table_id"]],
                TagSpecifications=_tag_spec("vpc-endpoint", f"{prefix}-{svc}"),
            )["VpcEndpoint"]["VpcEndpointId"]
    pls = {
        p["PrefixListName"]: p["PrefixListId"]
        for p in ec2.describe_prefix_lists(
            Filters=[
                {
                    "Name": "prefix-list-name",
                    "Values": [f"com.amazonaws.{region}.s3", f"com.amazonaws.{region}.s3express"],
                }
            ]
        )["PrefixLists"]
    }
    sg = ec2.create_security_group(
        GroupName=f"{prefix}-fn",
        Description="lambda-kleidiai E1: HTTPS egress to S3 and S3 Express only",
        VpcId=v["vpc_id"],
        TagSpecifications=_tag_spec("security-group", f"{prefix}-fn"),
    )["GroupId"]
    ec2.revoke_security_group_egress(
        GroupId=sg,
        IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
    )
    ec2.authorize_security_group_egress(
        GroupId=sg,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 443,
                "ToPort": 443,
                "PrefixListIds": [{"PrefixListId": pid} for pid in sorted(pls.values())],
            }
        ],
    )
    v["sg_id"] = sg
    v["prefix_lists"] = pls
    return v


# ---------------------------------------------------------------------------- buckets and objects


def ensure_buckets(s3, st: dict) -> None:
    region, prefix = st["region"], st["prefix"]
    if not st.get("bucket"):
        st["bucket"] = p1_style_bucket(s3, region, f"{prefix}-{region}-{os.urandom(4).hex()}")
    if not st.get("xz_bucket"):
        name = directory_bucket_name(f"{prefix}-{os.urandom(3).hex()}", st["az_id"])
        s3.create_bucket(
            Bucket=name,
            CreateBucketConfiguration={
                "Location": {"Type": "AvailabilityZone", "Name": st["az_id"]},
                "Bucket": {"Type": "Directory", "DataRedundancy": "SingleAvailabilityZone"},
            },
        )
        st["xz_bucket"] = name  # directory buckets block public access and use SSE-S3 by default


def p1_style_bucket(s3, region: str, name: str) -> str:
    """baseline/deploy.py ensure_bucket(): public access blocked, SSE-S3, TLS only; E1's tags."""
    s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": region})
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_bucket_policy(
        Bucket=name,
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "TLSOnly",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [f"arn:aws:s3:::{name}", f"arn:aws:s3:::{name}/*"],
                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                    }
                ],
            }
        ),
    )
    s3.put_bucket_tagging(
        Bucket=name, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]}
    )
    return name


def _present(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def object_sha256(s3, bucket: str, key: str, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    for c in body.iter_chunks(chunk):
        h.update(c)
    return h.hexdigest()


def stage(s3, st: dict, source: str | None, build: str | None) -> None:
    """GGUF into both buckets, function.zip and manifest into the general-purpose one, all hash-checked."""
    if source:
        src = source.removeprefix("s3://").rstrip("/")
        mkey = f"code/{st['zip_sha256']}/manifest.json"
        manifest = json.loads(s3.get_object(Bucket=src, Key=mkey)["Body"].read())
    else:
        manifest = json.load(open(os.path.join(build, "manifest.json")))
    g, z = manifest["model"]["gguf_sha256"], manifest["function_zip"]["sha256"]
    if st.get("zip_sha256") and st["zip_sha256"] != z:
        sys.exit(f"manifest names zip {z}, expected {st['zip_sha256']}")
    st["manifest"] = manifest
    st["model_key"] = f"weights/{g}/{GGUF_NAME}"
    st["zip_key"] = f"code/{z}/function.zip"
    items = [
        (st["bucket"], st["model_key"], "model/" + GGUF_NAME),
        (st["xz_bucket"], st["model_key"], "model/" + GGUF_NAME),
        (st["bucket"], st["zip_key"], "function.zip"),
        (st["bucket"], f"code/{z}/manifest.json", "manifest.json"),
    ]
    for bucket, key, local in items:
        if _present(s3, bucket, key):
            print("present", "xz" if is_directory_bucket(bucket) else "gp", key)
            continue
        t = time.time()
        if source:
            s3.copy({"Bucket": src, "Key": key}, bucket, key)
        else:
            s3.upload_file(os.path.join(build, local), bucket, key)
        print(
            f"staged {'xz' if is_directory_bucket(bucket) else 'gp'} {key} in {time.time() - t:.0f} s"
        )
    checks = st.setdefault("staged_sha256", {})
    for name, bucket, key, want in (
        ("gguf_gp", st["bucket"], st["model_key"], g),
        ("gguf_xz", st["xz_bucket"], st["model_key"], g),
        ("function_zip", st["bucket"], st["zip_key"], z),
    ):
        if checks.get(name) != want:
            got = object_sha256(s3, bucket, key)
            if got != want:
                sys.exit(f"{name}: staged object sha256 {got} != manifest {want}")
            checks[name] = got
        print("sha256 ok", name, want[:12])


# ---------------------------------------------------------------------------- IAM, functions


def ensure_role(iam, st: dict, account: str, arm: str) -> str:
    """This arm's role: its inline policy, plus AWSLambdaVPCAccessExecutionRole only when the arm is VPC-attached.

    Attachment is converged both ways: a no-VPC arm's role has the VPC policy detached if it is present.
    """
    name = role_name(st, arm)
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
    iam.put_role_policy(
        RoleName=name,
        PolicyName="lkai-e1",
        PolicyDocument=json.dumps(role_policy(st["region"], account, st, arm)),
    )
    attached = {
        p["PolicyArn"]
        for page in iam.get_paginator("list_attached_role_policies").paginate(RoleName=name)
        for p in page["AttachedPolicies"]
    }
    if ARMS[arm][1]:
        # ENI management (the managed policy AWS documents for VPC-attached functions).
        if VPC_ACCESS_POLICY_ARN not in attached:
            iam.attach_role_policy(RoleName=name, PolicyArn=VPC_ACCESS_POLICY_ARN)
    elif VPC_ACCESS_POLICY_ARN in attached:
        iam.detach_role_policy(RoleName=name, PolicyArn=VPC_ACCESS_POLICY_ARN)
    st.pop("role", None)  # the legacy shared role's ARN
    st.setdefault("roles", {})[arm] = arn
    return arn


def delete_role(iam, name: str) -> bool:
    """Detach every managed policy, delete every inline one, then the role. False if it does not exist."""
    try:
        for page in iam.get_paginator("list_attached_role_policies").paginate(RoleName=name):
            for p in page["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=name, PolicyArn=p["PolicyArn"])
        for page in iam.get_paginator("list_role_policies").paginate(RoleName=name):
            for p in page["PolicyNames"]:
                iam.delete_role_policy(RoleName=name, PolicyName=p)
        iam.delete_role(RoleName=name)
    except iam.exceptions.NoSuchEntityException:
        return False
    print("deleted role", name)
    return True


def retire_legacy_shared_role(iam, st: dict) -> None:
    """Once every function has moved to its own role, the old shared role (VPC policy on every arm) goes."""
    delete_role(iam, f"{st['prefix']}-{LEGACY_SHARED_ROLE_SUFFIX}")


def functions(sess, st: dict, account: str) -> None:
    lam, logs, iam = sess.client("lambda"), sess.client("logs"), sess.client("iam")
    st.setdefault("functions", {})
    for arm in ARMS:
        role = ensure_role(iam, st, account, arm)
        cfg = function_config(st, arm, role)
        name = cfg["FunctionName"]
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        for attempt in range(6):
            try:
                try:
                    lam.get_function(FunctionName=name)
                    lam.update_function_code(
                        FunctionName=name, S3Bucket=st["bucket"], S3Key=st["zip_key"]
                    )
                    p1.wait_updated(lam, name)
                    lam.update_function_configuration(
                        **{k: v for k, v in cfg.items() if k != "Architectures"}
                    )
                except lam.exceptions.ResourceNotFoundException:
                    lam.create_function(
                        **cfg, Code={"S3Bucket": st["bucket"], "S3Key": st["zip_key"]}, Tags=TAGS
                    )
                break
            except lam.exceptions.InvalidParameterValueException as e:  # role not assumable yet
                if attempt == 5:
                    raise
                print("retrying:", e)
                time.sleep(10)
        c = p1.wait_updated(lam, name)
        if p1.b64_to_hex(c["CodeSha256"]) != st["manifest"]["function_zip"]["sha256"]:
            sys.exit(f"{name}: deployed code is not P1's function.zip")
        lam.put_runtime_management_config(
            FunctionName=name, UpdateRuntimeOn="Manual", RuntimeVersionArn=st["runtime_version_arn"]
        )
        st["functions"][arm] = {
            "name": name,
            "arn": c["FunctionArn"],
            "code_sha256_b64": c["CodeSha256"],
            "code_size": c["CodeSize"],
            "bucket_kind": ARMS[arm][0],
            "vpc": ARMS[arm][1],
        }
        print("function", name, c["CodeSha256"], "role", role_name(st, arm))
    retire_legacy_shared_role(iam, st)


def smoke(lam, st: dict) -> None:
    """One $LATEST invoke per arm: proves the bucket, IAM and network path end to end before the benchmark."""
    for arm, f in st["functions"].items():
        r = lam.invoke(
            FunctionName=f["name"],
            Payload=json.dumps(
                {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "Ready?"}}}
            ).encode(),
        )
        body = r["Payload"].read()
        if r.get("FunctionError"):
            sys.exit(f"smoke {arm} failed: {body[:600]!r}")
        print("smoke ok", arm, body[:160].decode(errors="replace"))


PROBE_SRC = """
import boto3, botocore, platform
def handler(event, context):
    return {"boto3": boto3.__version__, "botocore": botocore.__version__, "python": platform.python_version()}
"""


def sdk_probe(lam, st: dict) -> dict:
    """The runtime-included SDK version on P1's pinned runtime: a throwaway function, deleted after one invoke."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("probe.py", PROBE_SRC)
    name = f"{st['prefix']}-sdkprobe"
    try:
        lam.create_function(
            FunctionName=name,
            Runtime=RUNTIME,
            Architectures=["arm64"],
            Handler="probe.handler",
            Role=st["roles"]["gp"],  # P1's grants, no VPC policy
            MemorySize=256,
            Timeout=30,
            Code={"ZipFile": buf.getvalue()},
            Tags=TAGS,
        )
        p1.wait_updated(lam, name)
        lam.put_runtime_management_config(
            FunctionName=name, UpdateRuntimeOn="Manual", RuntimeVersionArn=st["runtime_version_arn"]
        )
        out = json.loads(lam.invoke(FunctionName=name)["Payload"].read())
    finally:
        try:
            lam.delete_function(FunctionName=name)
        except lam.exceptions.ResourceNotFoundException:
            pass
    st["runtime_sdk"] = out
    print("runtime SDK", out)
    return out


# ---------------------------------------------------------------------------- teardown


def teardown(sess, st: dict, everything: bool) -> None:
    # P1's teardown deletes every published version (SnapStart cache billing); its --all branch is P1-specific.
    p1.teardown(sess, {"functions": st.get("functions", {})}, False)
    if not everything:
        return
    lam = sess.client("lambda")
    for f in st.get("functions", {}).values():
        try:
            lam.delete_function(FunctionName=f["name"])
            print("deleted", f["name"])
        except lam.exceptions.ResourceNotFoundException:
            pass
    iam = sess.client("iam")
    for arm in ARMS:
        delete_role(iam, role_name(st, arm))
    retire_legacy_shared_role(iam, st)
    st.pop("roles", None)
    s3 = sess.client("s3")
    for b in (st.get("bucket"), st.get("xz_bucket")):
        if not b:
            continue
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b):
            for o in page.get("Contents", []):
                s3.delete_object(Bucket=b, Key=o["Key"])
        s3.delete_bucket(Bucket=b)
        print("deleted bucket", b)
    v = st.get("vpc") or {}
    ec2 = sess.client("ec2")
    if v.get("endpoints"):
        ec2.delete_vpc_endpoints(VpcEndpointIds=list(v["endpoints"].values()))
    # Lambda releases a deleted function's hyperplane ENIs asynchronously (minutes); retry until it has.
    for kind, fn in (
        ("sg_id", lambda: ec2.delete_security_group(GroupId=v["sg_id"])),
        ("subnet_id", lambda: ec2.delete_subnet(SubnetId=v["subnet_id"])),
    ):
        if not v.get(kind):
            continue
        for _ in range(60):
            try:
                fn()
                break
            except ClientError as e:
                if e.response["Error"]["Code"] != "DependencyViolation":
                    raise
                time.sleep(30)
        else:
            print(f"{kind} {v[kind]} still in use; rerun teardown --all later")
            return
    if v.get("route_table_id"):
        ec2.delete_route_table(RouteTableId=v["route_table_id"])
    print("deleted E1 network")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "show", "teardown"])
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument(
        "--source", help="s3://<bucket> holding P1's weights/ and code/ objects (read only)"
    )
    ap.add_argument(
        "--from-build", help="a baseline/build directory to upload from instead of --source"
    )
    ap.add_argument(
        "--zip-sha256",
        default="26eb2651e8915f8ad09eae899feff583a448de6b8d2bc4943bccabb66cae189b",
        help="P1's function.zip (baseline/results/20261020T080835Z/build-manifest.json)",
    )
    ap.add_argument("--prefix", default="lkai-e1")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--az-id", default="use2-az1")
    ap.add_argument("--runtime-version-arn", default=P1_RUNTIME_VERSION_ARN)
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.build, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = p1.load_state(a.build)
    st.setdefault("prefix", a.prefix)
    st.setdefault("region", a.region)
    st.setdefault("az_id", a.az_id)
    st.setdefault("zip_sha256", a.zip_sha256)
    st.setdefault("runtime_version_arn", a.runtime_version_arn)
    if st["region"] != a.region:
        sys.exit(f"{a.build}/deployment.json is for {st['region']}")
    try:
        if a.cmd == "up":
            if not a.source and not a.from_build:
                sys.exit("pass --source s3://<p1-bucket> or --from-build <baseline/build>")
            account = sess.client("sts").get_caller_identity()["Account"]
            ensure_network(sess.client("ec2"), st)
            p1.save_state(a.build, st)
            s3 = sess.client("s3")
            ensure_buckets(s3, st)
            p1.save_state(a.build, st)
            stage(s3, st, a.source, a.from_build)
            p1.save_state(a.build, st)
            functions(sess, st, account)
            p1.save_state(a.build, st)
            lam = sess.client(
                "lambda", config=Config(read_timeout=900, retries={"max_attempts": 0})
            )
            sdk_probe(lam, st)
            smoke(lam, st)
        if a.cmd == "teardown":
            teardown(sess, st, a.all)
    finally:
        p1.save_state(a.build, st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))


if __name__ == "__main__":
    main()
