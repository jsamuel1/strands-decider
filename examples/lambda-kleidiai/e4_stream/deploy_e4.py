"""Deploy E4: one bucket, one role and three functions from one zip, differing only in their environment.

  python deploy_e4.py up       --build <e4-build> [--prefix lkai-e4] [--region us-east-2]
  python deploy_e4.py teardown --build <e4-build> [--all]
  python deploy_e4.py show     --build <e4-build>

<e4-build> holds function.zip and manifest.json (e4_stream/build.sh) and model/s2b-h512-Q8_0.gguf (P1's weights,
byte for byte: the format is not E4's variable). Functions, all LKAI_PHASES=1 and handler e4_entry.handler:
  <prefix>-base        E4_STREAM=0                       download, then load: P1's sequence
  <prefix>-stream      E4_STREAM=1 STREAM_ORDER=arrival  the load follows the download front (the experiment)
  <prefix>-streamfile  E4_STREAM=1 STREAM_ORDER=file     the same overlap in llama.cpp's own tensor order (control)
Everything else is P1's deploy.functions (python3.12, arm64, 10,240 MB, 512 MB ephemeral, 900 s, SnapStart on
published versions, the runtime version pinned). The role may GetObject the one GGUF key only. State goes to
<e4-build>/deployment.json, in the shape baseline/bench.py reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import deploy  # noqa: E402

# --- arms ---
ARMS_ENV = {
    "base": {"E4_STREAM": "0"},
    "stream": {"E4_STREAM": "1", "STREAM_ORDER": "arrival"},
    "streamfile": {"E4_STREAM": "1", "STREAM_ORDER": "file"},
}
# --- end arms ---
HANDLER = "e4_entry.handler"
deploy.TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-E4"}


def functions(sess, st: dict) -> None:
    """P1's deploy.functions once per arm (LKAI_PHASES=1), then that arm's E4 variables and E4's handler."""
    lam = sess.client("lambda")
    real_pin = deploy.pin_runtime

    def pin_runtime(lam_, st_, logs=None):
        # P1's pin_runtime smoke-invokes functions["inst"] and pins every function in st["functions"]: give it
        # base as "inst" (one smoke invoke), then pin every arm to the version base reported.
        if "runtime_version_arn" not in st_:
            view = dict(st_, functions={"inst": st_["functions"]["base"]})
            real_pin(lam_, view, logs)
            st_["runtime_version_arn"] = view["runtime_version_arn"]
        for f in st_["functions"].values():
            lam_.put_runtime_management_config(
                FunctionName=f["name"],
                UpdateRuntimeOn="Manual",
                RuntimeVersionArn=st_["runtime_version_arn"],
            )

    real_functions_cfg, real_role = deploy.functions, deploy.ensure_role

    def ensure_role(iam_, st_, bucket):
        # deploy.functions runs once per arm with deploy.ARMS = {arm}; the policy must allow every arm's log group.
        arms, deploy.ARMS = deploy.ARMS, {x: "1" for x in ARMS_ENV}
        try:
            return real_role(iam_, st_, bucket)
        finally:
            deploy.ARMS = arms

    # pinned below, once the handler and environment are E4's
    deploy.pin_runtime = lambda *a, **k: None
    deploy.ensure_role = ensure_role
    try:
        for arm, extra in ARMS_ENV.items():
            deploy.ARMS = {arm: "1"}
            real_functions_cfg(sess, st)
            name = st["functions"][arm]["name"]
            c = lam.get_function_configuration(FunctionName=name)
            env = dict(c["Environment"]["Variables"], **extra)
            lam.update_function_configuration(
                FunctionName=name,
                Handler=HANDLER,
                Environment={"Variables": env},
                Description=f"lambda-kleidiai E4 ({arm}); zip {st['manifest']['function_zip']['sha256'][:12]}",
            )
            deploy.wait_updated(lam, name)
    finally:
        deploy.ARMS = {a: "1" for a in ARMS_ENV}
        deploy.pin_runtime, deploy.ensure_role = real_pin, real_role
    pin_runtime(lam, st, sess.client("logs"))
    for arm, extra in ARMS_ENV.items():
        c = lam.get_function_configuration(FunctionName=st["functions"][arm]["name"])
        env = c["Environment"]["Variables"]
        bad = {k: env.get(k) for k in extra if env.get(k) != extra[k]}
        if c["Handler"] != HANDLER or bad or env.get("MODEL_KEY") != st["model_key"]:
            sys.exit(f"{c['FunctionName']}: handler {c['Handler']}, env mismatch {bad}")
        others = {a: e for a, e in ARMS_ENV.items() if a != arm}
        for k in {k for e in others.values() for k in e} - set(extra):
            if k in env:
                sys.exit(f"{c['FunctionName']}: unexpected {k}={env[k]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "teardown", "show"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--prefix", default="lkai-e4")
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
    deploy.ARMS = {arm: "1" for arm in ARMS_ENV}
    try:
        if a.cmd == "up":
            deploy.upload(sess, a.build, st)
            functions(sess, st)
        elif a.cmd == "teardown":
            deploy.teardown(sess, st, a.all)
    finally:
        deploy.save_state(a.build, st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))


if __name__ == "__main__":
    main()
