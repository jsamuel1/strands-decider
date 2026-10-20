"""Deploy E5: one bucket, one role, two functions from the P1 baseline zip, differing only in the weight file.

  python deploy_e5.py up       --build <e5-build> [--prefix lkai-e5] [--region us-east-2]
  python deploy_e5.py teardown --build <e5-build> [--all]

<e5-build> holds function.zip and manifest.json (from baseline/build.sh), model/s2b-h512-Q8_0.gguf (the baseline
weights) and model/s2b-h512-Q8_0-novocab.gguf (convert_no_vocab.py's output). Functions:
  <prefix>-base   MODEL_KEY = the baseline GGUF
  <prefix>-e5     MODEL_KEY = the no-vocab GGUF
Both run LKAI_PHASES=1 (the instrumented arm), so every phase is measured on both; everything else is P1's
deploy.functions: python3.12, arm64, 10,240 MB, 512 MB ephemeral, 900 s, SnapStart on published versions,
runtime pinned. The role may GetObject the two GGUF keys only. State goes to <e5-build>/deployment.json, in the
shape baseline/bench.py reads.
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

FILES = {"base": "s2b-h512-Q8_0.gguf", "e5": "s2b-h512-Q8_0-novocab.gguf"}
deploy.ARMS = {"base": "1", "e5": "1"}  # arm -> LKAI_PHASES; both instrumented
deploy.TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E5"}


def upload(sess, build: str, st: dict) -> None:
    s3 = sess.client("s3")
    m = json.load(open(os.path.join(build, "manifest.json")))
    e5 = json.load(open(os.path.join(build, "e5-manifest.json")))
    bucket = deploy.ensure_bucket(s3, sess.region_name, st, st["prefix"])
    z = m["function_zip"]["sha256"]
    sha = {"base": m["model"]["gguf_sha256"], "e5": e5["converted"]["sha256"]}
    items = {
        f"model_key_{a}": (os.path.join(build, "model", f), f"weights/{sha[a]}/{f}")
        for a, f in FILES.items()
    }
    items["zip_key"] = (os.path.join(build, "function.zip"), f"code/{z}/function.zip")
    for key_name, (path, key) in items.items():
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print("present", key)
        except ClientError:
            t = time.time()
            s3.upload_file(path, bucket, key)
            print(
                f"uploaded {key} {os.path.getsize(path) / 2**20:.0f} MiB in {time.time() - t:.0f} s"
            )
        st[key_name] = key
    st.update(manifest=m, e5_manifest=e5)


def functions(sess, st: dict) -> None:
    """P1's deploy.functions once per arm, with that arm's MODEL_KEY. The role allows both keys."""
    keys = {a: st[f"model_key_{a}"] for a in FILES}
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
                        for a in FILES
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
        # P1's pin_runtime smoke-invokes functions["inst"]; here that is the first arm deployed (base). It pins
        # every function in st["functions"], so the second call re-pins both to the version the first one saw.
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
        for arm in FILES:  # base first: its smoke invoke pins the runtime version
            deploy.ARMS = {arm: "1"}
            st["model_key"] = keys[arm]
            deploy.functions(sess, st)
    finally:
        deploy.ensure_role, deploy.pin_runtime = real_ensure_role, real_pin
        deploy.ARMS = {a: "1" for a in FILES}
        st.pop("model_key", None)
    lam = sess.client("lambda")
    for arm, f in st["functions"].items():
        env = lam.get_function_configuration(FunctionName=f["name"])["Environment"]["Variables"]
        if env["MODEL_KEY"] != keys[arm]:
            sys.exit(f"{f['name']}: MODEL_KEY {env['MODEL_KEY']} is not {keys[arm]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "teardown", "show"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--prefix", default="lkai-e5")
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
