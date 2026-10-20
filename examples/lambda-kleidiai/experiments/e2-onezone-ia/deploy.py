"""E2 deployment: the P1 baseline with the weight object's storage class as the only variable.

  python deploy.py up       --source-bucket <bucket holding P1's pinned artifacts> [--prefix lkai-e2]
  python deploy.py show
  python deploy.py teardown [--all]          # versions (SnapStart cache billing); --all: everything

One new general-purpose bucket holds two server-side copies of P1's GGUF (sha256 8dbb8872…), one per storage class:

  weights/<gguf sha>/standard/s2b-h512-Q8_0.gguf     STANDARD    -> function <prefix>-std
  weights/<gguf sha>/onezone_ia/s2b-h512-Q8_0.gguf   ONEZONE_IA  -> function <prefix>-ozia

Each copy asks S3 for a full-object SHA-256 checksum, and `up` refuses to continue unless that checksum equals the
build manifest's GGUF sha256 and the object reports the intended storage class, so both arms read byte-identical
weights. Both functions run P1's published-run zip (sha256 26eb2651…) with the same configuration as P1's
instrumented arm (LKAI_PHASES=1, python3.12, arm64, 10,240 MB, 512 MB ephemeral, SnapStart on published versions)
and the runtime version pinned to the ARN P1 ran on. They differ only in MODEL_KEY. The state file has the shape
baseline/bench.py reads, so the benchmark is P1's harness unchanged (`bench.py --build <this build> --arms std,ozia`).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "..", "baseline"))
sys.path.insert(0, BASE)
import deploy as base  # noqa: E402

ARMS = {"std": "STANDARD", "ozia": "ONEZONE_IA"}
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E2"}
# P1's published baseline run 20261020T080835Z (baseline/results/20261020T080835Z/build-manifest.json).
PINNED_ZIP_SHA256 = "26eb2651e8915f8ad09eae899feff583a448de6b8d2bc4943bccabb66cae189b"
PINNED_GGUF_SHA256 = "8dbb8872e6af312804b8d9d82b6d023bd025a0cfc81d55703b79837d0f042d8e"
PINNED_RUNTIME = (
    "arn:aws:lambda:us-east-2::runtime:"
    "0fec05b8b43b0ecfbabe4fd585b35e423f655861556712f8a10b0e8739c36c07"
)
GGUF = "s2b-h512-Q8_0.gguf"
POLICY = "lkai-e2"


def weight_key(gguf_sha: str, storage_class: str) -> str:
    return f"weights/{gguf_sha}/{storage_class.lower()}/{GGUF}"


def object_problems(head: dict, storage_class: str, size: int, sha256_hex: str) -> list[str]:
    """Why a HeadObject(ChecksumMode=ENABLED) response is not the intended copy; empty when it is."""
    out = []
    got_class = head.get("StorageClass", "STANDARD")  # S3 omits the header for STANDARD
    if got_class != storage_class:
        out.append(f"storage class {got_class}, want {storage_class}")
    if head.get("ContentLength") != size:
        out.append(f"size {head.get('ContentLength')}, want {size}")
    want = base64.b64encode(bytes.fromhex(sha256_hex)).decode()
    if head.get("ChecksumSHA256") != want:
        out.append(f"ChecksumSHA256 {head.get('ChecksumSHA256')}, want {want} (= {sha256_hex})")
    if head.get("ChecksumType", "FULL_OBJECT") != "FULL_OBJECT":
        out.append(f"checksum type {head.get('ChecksumType')}, want FULL_OBJECT")
    return out


def _head(s3, bucket: str, key: str) -> dict | None:
    try:
        return s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def upload(sess, st: dict, source_bucket: str) -> None:
    """Copy P1's pinned zip, manifest and GGUF (server-side) into this experiment's own bucket."""
    s3 = sess.client("s3", config=Config(read_timeout=900, retries={"max_attempts": 5}))
    if "manifest" not in st:
        z = PINNED_ZIP_SHA256
        st["manifest"] = json.loads(
            s3.get_object(Bucket=source_bucket, Key=f"code/{z}/manifest.json")["Body"].read()
        )
    m = st["manifest"]
    if m["function_zip"]["sha256"] != PINNED_ZIP_SHA256:
        sys.exit(f"manifest zip {m['function_zip']['sha256']} is not P1's {PINNED_ZIP_SHA256}")
    if m["model"]["gguf_sha256"] != PINNED_GGUF_SHA256:
        sys.exit(f"manifest GGUF {m['model']['gguf_sha256']} is not P1's {PINNED_GGUF_SHA256}")
    bucket = base.ensure_bucket(s3, sess.region_name, st, st["prefix"])
    s3.put_bucket_tagging(
        Bucket=bucket, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]}
    )
    z, g, size = m["function_zip"]["sha256"], m["model"]["gguf_sha256"], m["model"]["gguf_bytes"]
    for key in (f"code/{z}/function.zip", f"code/{z}/manifest.json"):
        if _head(s3, bucket, key) is None:
            s3.copy_object(Bucket=bucket, Key=key, CopySource={"Bucket": source_bucket, "Key": key})
            print("copied", key)
    st["zip_key"] = f"code/{z}/function.zip"
    st["weights"] = {}
    for arm, sc in ARMS.items():
        key = weight_key(g, sc)
        h = _head(s3, bucket, key)
        if h is None:
            t = time.time()
            s3.copy_object(
                Bucket=bucket,
                Key=key,
                CopySource={"Bucket": source_bucket, "Key": f"weights/{g}/{GGUF}"},
                StorageClass=sc,
                ChecksumAlgorithm="SHA256",
            )
            print(f"copied {key} as {sc} in {time.time() - t:.0f} s")
            h = _head(s3, bucket, key)
        problems = object_problems(h or {}, sc, size, g)
        if problems:
            sys.exit(f"{key}: " + "; ".join(problems))
        st["weights"][arm] = {
            "key": key,
            "storage_class": sc,
            "bytes": h["ContentLength"],
            "checksum_sha256_b64": h["ChecksumSHA256"],
            "checksum_type": h.get("ChecksumType", "FULL_OBJECT"),
            "sha256_matches_manifest": True,
            "last_modified": h["LastModified"].isoformat()
            if hasattr(h["LastModified"], "isoformat")
            else str(h["LastModified"]),
        }
        print("verified", key, sc, "sha256", g[:12])


def ensure_role(iam, st: dict) -> str:
    name = f"{st['prefix']}-fn"
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
        time.sleep(10)  # IAM propagation before Lambda can assume it
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": [
                    f"arn:aws:s3:::{st['bucket']}/{w['key']}" for w in st["weights"].values()
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [
                    f"arn:aws:logs:{st['region']}:*:log-group:/aws/lambda/{st['prefix']}-{a}:*"
                    for a in ARMS
                ],
            },
        ],
    }
    iam.put_role_policy(RoleName=name, PolicyName=POLICY, PolicyDocument=json.dumps(policy))
    st["role"] = arn
    return arn


def function_config(st: dict, arm: str, role: str) -> dict:
    """P1's instrumented-arm configuration; only the name, MODEL_KEY and description differ."""
    return dict(
        FunctionName=f"{st['prefix']}-{arm}",
        Runtime=base.RUNTIME,
        Architectures=["arm64"],
        Handler="lkai_entry.handler",
        Role=role,
        MemorySize=base.MEMORY,
        EphemeralStorage={"Size": 512},
        Timeout=900,
        SnapStart={"ApplyOn": "PublishedVersions"},
        Environment={
            "Variables": {
                "MODEL_BUCKET": st["bucket"],
                "MODEL_KEY": st["weights"][arm]["key"],
                "DEC_KV": "f32",
                "DEC_UBATCH": "512",
                "N_SEQ": "17",
                "LKAI_PHASES": "1",
                "COLD_TAG": "0",
            }
        },
        Description=f"lambda-kleidiai E2 ({arm}: {ARMS[arm]} weights); "
        f"zip {st['manifest']['function_zip']['sha256'][:12]}",
    )


def functions(sess, st: dict) -> None:
    lam, logs, iam = sess.client("lambda"), sess.client("logs"), sess.client("iam")
    role = ensure_role(iam, st)
    st.setdefault("functions", {})
    for arm in ARMS:
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
                    base.wait_updated(lam, name)
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
        base.wait_updated(lam, name)
        lam.put_runtime_management_config(
            FunctionName=name, UpdateRuntimeOn="Manual", RuntimeVersionArn=PINNED_RUNTIME
        )
        c = base.wait_updated(lam, name)
        if base.b64_to_hex(c["CodeSha256"]) != PINNED_ZIP_SHA256:
            sys.exit(f"{name}: deployed code {c['CodeSha256']} is not the pinned zip")
        st["functions"][arm] = {
            "name": name,
            "arn": c["FunctionArn"],
            "code_sha256_b64": c["CodeSha256"],
            "code_size": c["CodeSize"],
            "storage_class": ARMS[arm],
        }
        print("function", name, c["CodeSha256"], ARMS[arm])
    st["runtime_version_arn"] = PINNED_RUNTIME
    smoke(lam, st)


def smoke(lam, st: dict) -> None:
    """One on-demand invoke per arm: the weights load from that arm's object and the runtime is the pinned one."""
    body = {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "Ready?"}}}
    for f in st["functions"].values():
        r = lam.invoke(FunctionName=f["name"], LogType="Tail", Payload=json.dumps(body).encode())
        out = r["Payload"].read()
        if r.get("FunctionError"):
            log = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
            sys.exit(f"{f['name']} smoke invoke failed: {out[:600]!r}\n{log[-1500:]}")
        rv = lam.get_runtime_management_config(FunctionName=f["name"])
        if rv.get("RuntimeVersionArn") != PINNED_RUNTIME:
            sys.exit(f"{f['name']}: runtime {rv.get('RuntimeVersionArn')} is not pinned")
        print("smoke ok", f["name"], out[:200].decode(errors="replace"))


def teardown(sess, st: dict, everything: bool) -> None:
    base.teardown(sess, {"functions": st.get("functions", {})}, everything)
    if not everything:
        return
    iam = sess.client("iam")
    try:
        iam.delete_role_policy(RoleName=f"{st['prefix']}-fn", PolicyName=POLICY)
        iam.delete_role(RoleName=f"{st['prefix']}-fn")
    except iam.exceptions.NoSuchEntityException:
        pass
    if st.get("bucket"):
        b = sess.resource("s3").Bucket(st["bucket"])
        b.objects.all().delete()
        b.delete()
        print("deleted bucket", st["bucket"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "upload", "functions", "show", "teardown"])
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--source-bucket", default=os.environ.get("SOURCE_BUCKET"))
    ap.add_argument("--prefix", default="lkai-e2")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.build, exist_ok=True)
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = base.load_state(a.build)
    st.setdefault("prefix", a.prefix)
    st.setdefault("region", a.region)
    if st["region"] != a.region:
        sys.exit(f"{a.build}/deployment.json is for {st['region']}")
    try:
        if a.cmd in ("up", "upload"):
            if not a.source_bucket:
                sys.exit(
                    "--source-bucket (or SOURCE_BUCKET): the bucket P1's deploy.py upload wrote"
                )
            upload(sess, st, a.source_bucket)
        if a.cmd in ("up", "functions"):
            if "weights" not in st:
                sys.exit("run upload first")
            functions(sess, st)
        if a.cmd == "teardown":
            teardown(sess, st, a.all)
    finally:
        base.save_state(a.build, st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))


if __name__ == "__main__":
    main()
