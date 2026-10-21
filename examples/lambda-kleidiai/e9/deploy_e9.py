"""Deploy E9: one bucket, one role, two functions. The arms differ only in when KleidiAI packing happens.

  python deploy_e9.py up       --build <e9-build> [--prefix lkai-f1ee] [--region us-east-2]
  python deploy_e9.py teardown --build <e9-build> [--all]

<e9-build> holds (build.sh, plus links to P1's build):
  base-function.zip, manifest.json      P1's published zip (26eb2651...) and its build manifest
  function.zip                          E9's zip: P1's sources plus patch_e9.py in libllama / libggml-cpu-*
  model/s2b-h512-Q8_0.gguf              P1's GGUF (the base arm's weights)
  model/s2b-h512-Q8_0-e9kr8.gguf        pack_e9.py's output (the e9 arm's weights)
  model/pack-kr8.json                   the pack record

Functions:
  <prefix>-base   P1's zip, P1's GGUF: the load repacks every Q8_0 weight (the baseline)
  <prefix>-e9     E9's zip, E9's GGUF: the load adopts the packed bytes (or converts them on a dot-product host)
Packing is one variable that needs both pieces: P1's loader cannot read packed bytes, and E9's loader on P1's GGUF
takes the unchanged repack (README, "Offline verification": E9 zip on P1's GGUF answers like P1's zip).

Both arms LKAI_PHASES=1; everything else is P1's deploy.functions (python3.12, arm64, 10,240 MB, 512 MB ephemeral,
900 s, SnapStart on published versions, runtime pinned to one version ARN). The role may GetObject the two GGUF keys
only. deployment.json names each arm's zip in zip_sha256_by_arm, which baseline/bench.py checks per arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import deploy  # noqa: E402

ARMS = {
    "base": ("base-function.zip", "s2b-h512-Q8_0.gguf"),
    "e9": ("function.zip", "s2b-h512-Q8_0-e9kr8.gguf"),
}
deploy.TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E9"}


def upload(sess, build: str, st: dict) -> None:
    s3 = sess.client("s3")
    m = json.load(open(os.path.join(build, "manifest.json")))
    pack = json.load(open(os.path.join(build, "model", "pack-kr8.json")))
    bucket = deploy.ensure_bucket(s3, sess.region_name, st, st["prefix"])
    zsha = {a: deploy.file_sha256(os.path.join(build, z)) for a, (z, _) in ARMS.items()}
    gsha = {"base": m["model"]["gguf_sha256"], "e9": pack["packed"]["sha256"]}
    if zsha["base"] != m["function_zip"]["sha256"]:
        sys.exit(f"{build}/base-function.zip is not the zip manifest.json names")
    if zsha["base"] == zsha["e9"]:
        sys.exit("the two arms have the same zip")
    st["zip_sha256_by_arm"] = zsha
    st["gguf_sha256_by_arm"] = gsha
    for arm, (z, g) in ARMS.items():
        for kind, path, key in (
            ("model_key", os.path.join(build, "model", g), f"weights/{gsha[arm]}/{g}"),
            ("zip_key", os.path.join(build, z), f"code/{zsha[arm]}/function.zip"),
        ):
            try:
                s3.head_object(Bucket=bucket, Key=key)
                print("present", key)
            except ClientError:
                t = time.time()
                s3.upload_file(path, bucket, key)
                print(
                    f"uploaded {key} {os.path.getsize(path) / 2**20:.0f} MiB in {time.time() - t:.0f} s"
                )
            st[f"{kind}_{arm}"] = key
    st.update(
        manifest=m,
        e9_pack={
            k: pack[k]
            for k in (
                "layout",
                "metadata_key",
                "packed_count",
                "source",
                "packed",
                "image",
                "tools",
            )
        },
    )


def functions(sess, st: dict) -> None:
    """P1's deploy.functions once per arm, with that arm's zip and MODEL_KEY; the role allows both GGUF keys."""
    keys = {a: st[f"model_key_{a}"] for a in ARMS}
    real_ensure_role, real_pin = deploy.ensure_role, deploy.pin_runtime

    def ensure_role(iam_, st_, bucket):
        arn = real_ensure_role(iam_, st_, bucket)
        doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": [f"arn:aws:s3:::{bucket}/{k}" for k in keys.values()],
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                    "Resource": [
                        f"arn:aws:logs:{st_['region']}:*:log-group:/aws/lambda/{st_['prefix']}-{a}:*"
                        for a in ARMS
                    ],
                },
            ],
        }
        iam_.put_role_policy(
            RoleName=f"{st_['prefix']}-fn",
            PolicyName="lkai-baseline",
            PolicyDocument=json.dumps(doc),
        )
        return arn

    def pin_runtime(lam, st_, logs=None):
        # P1's pin_runtime smoke-invokes functions["inst"]; here the first arm deployed (base). The second call
        # re-pins both functions to the version the first one saw.
        if "runtime_version_arn" in st_:
            for f in st_["functions"].values():
                lam.put_runtime_management_config(
                    FunctionName=f["name"],
                    UpdateRuntimeOn="Manual",
                    RuntimeVersionArn=st_["runtime_version_arn"],
                )
            return
        view = dict(st_, functions={"inst": st_["functions"]["base"]})
        real_pin(lam, view, logs)
        st_["runtime_version_arn"] = view["runtime_version_arn"]

    deploy.ensure_role, deploy.pin_runtime = ensure_role, pin_runtime
    try:
        for arm in ARMS:  # base first: its smoke invoke pins the runtime version
            deploy.ARMS = {arm: "1"}
            st["model_key"], st["zip_key"] = keys[arm], st[f"zip_key_{arm}"]
            deploy.functions(sess, st)
    finally:
        deploy.ensure_role, deploy.pin_runtime = real_ensure_role, real_pin
        deploy.ARMS = {a: "1" for a in ARMS}
        st.pop("model_key", None)
        st.pop("zip_key", None)
    lam = sess.client("lambda")
    for arm, f in st["functions"].items():
        c = lam.get_function_configuration(FunctionName=f["name"])
        if c["Environment"]["Variables"]["MODEL_KEY"] != keys[arm]:
            sys.exit(f"{f['name']}: MODEL_KEY is not {keys[arm]}")
        if deploy.b64_to_hex(c["CodeSha256"]) != st["zip_sha256_by_arm"][arm]:
            sys.exit(f"{f['name']}: deployed code is not the {arm} arm's zip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "teardown", "show"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--prefix", default="lkai-f1ee")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = deploy.load_state(a.build)
    st.setdefault("prefix", a.prefix)
    st.setdefault("region", a.region)
    if st["prefix"] != a.prefix or st["region"] != a.region:
        sys.exit(f"{a.build}/deployment.json is for {st['prefix']} in {st['region']}")
    try:
        if a.cmd == "up":
            upload(sess, a.build, st)
            functions(sess, st)
        elif a.cmd == "teardown":
            deploy.teardown(sess, st, a.all)
    finally:
        deploy.save_state(a.build, st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if "manifest" not in k}, indent=1))


if __name__ == "__main__":
    main()
