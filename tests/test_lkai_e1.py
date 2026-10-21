"""examples/lambda-kleidiai/e1-s3-express: the one-variable deployment and the E1 summary, without AWS.

Pins what makes E1 a single-variable experiment against P1: every arm gets P1's configuration and differs only in the
weights bucket and (for the VPC pair) the network; each arm's role holds P1's grants for its own bucket and log group,
with a ReadOnly CreateSession grant replacing s3:GetObject on directory-bucket arms and the VPC managed policy only on
VPC arms; and the
summary's contrasts, status rule and per-bucket-type pricing are computed from the rows they claim.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import types

import pytest

E1 = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e1-s3-express")
BASE = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "baseline")
sys.path.insert(0, os.path.abspath(BASE))
sys.path.insert(0, os.path.abspath(E1))


@pytest.fixture(autouse=True)
def _aws_sdk_stand_ins(monkeypatch):
    if importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore"):
        return
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    botocore = types.ModuleType("botocore")
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    for name, mod in {
        "boto3": boto3,
        "botocore": botocore,
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in ("deploy", "deploy_e1"):
        monkeypatch.delitem(sys.modules, m, raising=False)


def _state():
    return {
        "prefix": "lkai-e1",
        "region": "us-east-2",
        "az_id": "use2-az1",
        "bucket": "lkai-e1-us-east-2-aaaa",
        "xz_bucket": "lkai-e1-bbbb--use2-az1--x-s3",
        "model_key": "weights/g/s2b-h512-Q8_0.gguf",
        "vpc": {"subnet_id": "subnet-1", "sg_id": "sg-1"},
    }


def test_directory_bucket_name_carries_az_suffix_botocore_routes_on():
    import deploy_e1

    name = deploy_e1.directory_bucket_name("lkai-e1-abc", "use2-az1")
    assert name == "lkai-e1-abc--use2-az1--x-s3"
    assert deploy_e1.is_directory_bucket(name)
    assert not deploy_e1.is_directory_bucket("lkai-e1-us-east-2-aaaa")


def _actions(pol):
    return {
        a
        for s in pol["Statement"]
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    }


def test_gp_role_is_p1s_grants_on_its_own_object_and_log_group():
    import deploy_e1

    for arm in ("gp", "gpv"):
        pol = deploy_e1.role_policy("us-east-2", "123456789012", _state(), arm)
        assert _actions(pol) == {"s3:GetObject", "logs:CreateLogStream", "logs:PutLogEvents"}
        get = next(s for s in pol["Statement"] if s["Action"] == "s3:GetObject")
        assert get["Resource"] == "arn:aws:s3:::lkai-e1-us-east-2-aaaa/weights/g/s2b-h512-Q8_0.gguf"
        logs = next(s for s in pol["Statement"] if "logs:PutLogEvents" in s["Action"])
        assert logs["Resource"] == [
            f"arn:aws:logs:us-east-2:123456789012:log-group:/aws/lambda/lkai-e1-{arm}:*"
        ]


def test_xz_role_adds_only_readonly_create_session_on_the_directory_bucket():
    import deploy_e1

    for arm in ("xz", "xzv"):
        pol = deploy_e1.role_policy("us-east-2", "123456789012", _state(), arm)
        assert _actions(pol) == {
            "s3express:CreateSession",
            "logs:CreateLogStream",
            "logs:PutLogEvents",
        }
        cs = next(s for s in pol["Statement"] if s["Action"] == "s3express:CreateSession")
        assert (
            cs["Resource"]
            == "arn:aws:s3express:us-east-2:123456789012:bucket/lkai-e1-bbbb--use2-az1--x-s3"
        )
        assert cs["Condition"] == {"StringEquals": {"s3express:SessionMode": "ReadOnly"}}
        logs = next(s for s in pol["Statement"] if "logs:PutLogEvents" in s["Action"])
        assert logs["Resource"] == [
            f"arn:aws:logs:us-east-2:123456789012:log-group:/aws/lambda/lkai-e1-{arm}:*"
        ]


class _NoSuchEntity(Exception):
    pass


class _IAM:
    """Records calls; roles and their attached/inline policies live in dicts."""

    exceptions = types.SimpleNamespace(NoSuchEntityException=_NoSuchEntity)

    def __init__(self, roles=None):
        # name -> {"attached": set, "inline": set}
        self.roles = roles or {}
        self.calls = []

    def _role(self, name):
        if name not in self.roles:
            raise _NoSuchEntity(name)
        return self.roles[name]

    def get_role(self, RoleName):
        self._role(RoleName)
        return {"Role": {"Arn": f"arn:aws:iam::123456789012:role/{RoleName}"}}

    def create_role(self, RoleName, **kw):
        self.calls.append(("create_role", RoleName))
        self.roles[RoleName] = {"attached": set(), "inline": set()}
        return self.get_role(RoleName)

    def put_role_policy(self, RoleName, PolicyName, PolicyDocument):
        self.calls.append(("put_role_policy", RoleName, json.loads(PolicyDocument)))
        self._role(RoleName)["inline"].add(PolicyName)

    def attach_role_policy(self, RoleName, PolicyArn):
        self.calls.append(("attach_role_policy", RoleName, PolicyArn))
        self._role(RoleName)["attached"].add(PolicyArn)

    def detach_role_policy(self, RoleName, PolicyArn):
        self.calls.append(("detach_role_policy", RoleName, PolicyArn))
        self._role(RoleName)["attached"].discard(PolicyArn)

    def delete_role_policy(self, RoleName, PolicyName):
        self._role(RoleName)["inline"].discard(PolicyName)

    def delete_role(self, RoleName):
        r = self._role(RoleName)
        assert not r["attached"] and not r["inline"], "IAM refuses to delete a role with policies"
        self.calls.append(("delete_role", RoleName))
        del self.roles[RoleName]

    def get_paginator(self, op):
        iam = self

        class P:
            def paginate(self, RoleName):
                r = iam._role(RoleName)
                if op == "list_attached_role_policies":
                    return [{"AttachedPolicies": [{"PolicyArn": a} for a in sorted(r["attached"])]}]
                return [{"PolicyNames": sorted(r["inline"])}]

        return P()


def test_only_vpc_arms_roles_carry_the_vpc_access_policy(monkeypatch):
    import deploy_e1

    monkeypatch.setattr(deploy_e1.time, "sleep", lambda s: None)
    iam, st = _IAM(), _state()
    arns = {arm: deploy_e1.ensure_role(iam, st, "123456789012", arm) for arm in deploy_e1.ARMS}
    assert len(set(arns.values())) == 4 and st["roles"] == arns and "role" not in st
    for arm, (_, vpc) in deploy_e1.ARMS.items():
        attached = iam.roles[deploy_e1.role_name(st, arm)]["attached"]
        assert attached == ({deploy_e1.VPC_ACCESS_POLICY_ARN} if vpc else set()), arm
    # every function is configured with its own arm's role
    for arm in deploy_e1.ARMS:
        assert deploy_e1.function_config(st, arm, st["roles"][arm])["Role"] == arns[arm]


def test_ensure_role_detaches_a_stray_vpc_policy_from_a_no_vpc_arm(monkeypatch):
    import deploy_e1

    st = _state()
    vpc = deploy_e1.VPC_ACCESS_POLICY_ARN
    iam = _IAM({deploy_e1.role_name(st, "gp"): {"attached": {vpc}, "inline": set()}})
    deploy_e1.ensure_role(iam, st, "123456789012", "gp")
    assert iam.roles[deploy_e1.role_name(st, "gp")]["attached"] == set()
    # a VPC arm that already has it is not re-attached
    iam = _IAM({deploy_e1.role_name(st, "gpv"): {"attached": {vpc}, "inline": set()}})
    deploy_e1.ensure_role(iam, st, "123456789012", "gpv")
    assert not [c for c in iam.calls if c[0] == "attach_role_policy"]


def test_legacy_shared_role_is_retired_and_teardown_deletes_every_arm_role(monkeypatch):
    import deploy_e1

    st = _state()
    legacy = "lkai-e1-fn"
    roles = {legacy: {"attached": {deploy_e1.VPC_ACCESS_POLICY_ARN}, "inline": {"lkai-e1"}}}
    roles |= {
        deploy_e1.role_name(st, a): {"attached": set(), "inline": {"lkai-e1"}} for a in ("gp", "xz")
    }
    iam = _IAM(roles)
    deploy_e1.retire_legacy_shared_role(iam, st)
    assert legacy not in iam.roles
    for arm in deploy_e1.ARMS:  # gpv/xzv absent: tolerated
        deploy_e1.delete_role(iam, deploy_e1.role_name(st, arm))
    assert iam.roles == {}
    # the per-arm names never collide with the legacy shared name
    assert legacy not in {deploy_e1.role_name(st, a) for a in deploy_e1.ARMS}


def test_arms_differ_only_in_bucket_and_vpc():
    import deploy_e1

    st = _state()
    cfgs = {a: deploy_e1.function_config(st, a, "role-arn") for a in deploy_e1.ARMS}
    varying = {"FunctionName", "Description", "VpcConfig", "Environment"}
    base = {k: v for k, v in cfgs["gp"].items() if k not in varying}
    for a, c in cfgs.items():
        assert {k: v for k, v in c.items() if k not in varying} == base, a
        env = dict(c["Environment"]["Variables"])
        bucket = env.pop("MODEL_BUCKET")
        assert env == {
            k: v for k, v in cfgs["gp"]["Environment"]["Variables"].items() if k != "MODEL_BUCKET"
        }
        assert bucket == (st["xz_bucket"] if a.startswith("xz") else st["bucket"])
        assert bool(c["VpcConfig"]["SubnetIds"]) == a.endswith("v")
    # P1's pinned configuration (baseline/deploy.py)
    assert base["Runtime"] == "python3.12" and base["Architectures"] == ["arm64"]
    assert base["MemorySize"] == 10240 and base["EphemeralStorage"] == {"Size": 512}
    assert (
        base["SnapStart"] == {"ApplyOn": "PublishedVersions"}
        and base["Handler"] == "lkai_entry.handler"
    )
    assert cfgs["gp"]["Environment"]["Variables"]["LKAI_PHASES"] == "1"


def test_vpc_arms_share_one_subnet_in_the_bucket_az():
    import deploy_e1

    st = _state()
    a, b = (deploy_e1.function_config(st, x, "r")["VpcConfig"] for x in ("gpv", "xzv"))
    assert a == b == {"SubnetIds": ["subnet-1"], "SecurityGroupIds": ["sg-1"]}


class _EC2:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(**kw):
            self.calls.append((name, kw))
            return {
                "describe_vpcs": {"Vpcs": [{"VpcId": "vpc-1"}]},
                "create_subnet": {"Subnet": {"SubnetId": "subnet-9"}},
                "create_route_table": {"RouteTable": {"RouteTableId": "rtb-9"}},
                "create_vpc_endpoint": {
                    "VpcEndpoint": {"VpcEndpointId": f"vpce-{len(self.calls)}"}
                },
                "describe_prefix_lists": {
                    "PrefixLists": [
                        {"PrefixListName": "com.amazonaws.us-east-2.s3", "PrefixListId": "pl-s3"},
                        {
                            "PrefixListName": "com.amazonaws.us-east-2.s3express",
                            "PrefixListId": "pl-xz",
                        },
                    ]
                },
                "create_security_group": {"GroupId": "sg-9"},
            }.get(name, {})

        return f


def test_network_is_new_and_egress_is_s3_only():
    import deploy_e1

    ec2 = _EC2()
    st = {"prefix": "lkai-e1", "region": "us-east-2", "az_id": "use2-az1"}
    v = deploy_e1.ensure_network(ec2, st)
    names = [c[0] for c in ec2.calls]
    assert v["subnet_id"] == "subnet-9" and v["route_table_id"] == "rtb-9" and v["sg_id"] == "sg-9"
    sub = dict(ec2.calls[names.index("create_subnet")][1])
    assert sub["AvailabilityZoneId"] == "use2-az1"
    # the gateway endpoints attach only to E1's own route table
    eps = [kw for n, kw in ec2.calls if n == "create_vpc_endpoint"]
    assert {e["ServiceName"] for e in eps} == {
        "com.amazonaws.us-east-2.s3",
        "com.amazonaws.us-east-2.s3express",
    }
    assert all(e["RouteTableIds"] == ["rtb-9"] and e["VpcEndpointType"] == "Gateway" for e in eps)
    assert "revoke_security_group_egress" in names
    egress = next(kw for n, kw in ec2.calls if n == "authorize_security_group_egress")[
        "IpPermissions"
    ]
    assert egress == [
        {
            "IpProtocol": "tcp",
            "FromPort": 443,
            "ToPort": 443,
            "PrefixListIds": [{"PrefixListId": "pl-s3"}, {"PrefixListId": "pl-xz"}],
        }
    ]
    n = len(ec2.calls)
    deploy_e1.ensure_network(ec2, st)  # idempotent once recorded
    assert len(ec2.calls) == n


def test_stage_refuses_a_staged_object_whose_hash_differs(monkeypatch):
    import deploy_e1

    st = _state() | {"zip_sha256": "z"}
    manifest = {"model": {"gguf_sha256": "g"}, "function_zip": {"sha256": "z"}}

    class S3:
        def get_object(self, Bucket, Key):
            assert Key == "code/z/manifest.json"
            return {"Body": types.SimpleNamespace(read=lambda: json.dumps(manifest).encode())}

        def head_object(self, **kw):
            return {}

    monkeypatch.setattr(
        deploy_e1,
        "object_sha256",
        lambda s3, b, k: "g" if "gguf" in k and b.endswith("aaaa") else "bad",
    )
    with pytest.raises(SystemExit, match="gguf_xz"):
        deploy_e1.stage(S3(), st, "s3://p1-bucket", None)


# ---------------------------------------------------------------------------- summary

ARMS = ("gp", "xz", "gpv", "xzv")
DL = {"gp": 25.0, "xz": 15.0, "gpv": 26.0, "xzv": 14.0}


def _row(arm, rnd, wave, rid="r1", dl=None, lat=200.0, cpu="0xd40"):
    if wave == "cold":
        rep = {
            "Duration": 9000.0,
            "Billed Duration": 30000.0,
            "Max Memory Used": 5000,
            "Init Duration": 700.0,
        }
        lk = {
            "inv": {"handler_s": 9.0, "wait_load_s": 8.8},
            "init": {"weight_download": dl, "model_load_mmap": 7.0, "model_ready": dl + 9},
            "counters": {"weight_mib_s": 2076712416 / 2**20 / dl},
        }
    elif wave == "restore":
        rep = {
            "Duration": 700.0,
            "Billed Duration": 900.0,
            "Max Memory Used": 5000,
            "Restore Duration": 3200.0,
            "Billed Restore Duration": 300.0,
        }
        lk = {"inv": {"handler_s": 0.7, "wait_load_s": 0.0, "restore_to_invoke_s": 0.01}}
    else:
        rep = {"Duration": 300.0, "Billed Duration": 301.0, "Max Memory Used": 5000}
        lk = {"inv": {"handler_s": 0.3, "wait_load_s": 0.0}}
    from phases import observed_class

    obs = observed_class(rep)
    return {
        "arm": arm,
        "round": rnd,
        "wave": wave,
        "seq": 0,
        "request_id": rid,
        "rt_s": 1.0,
        "error": None,
        "report": rep,
        "expected_class": wave,
        "observed_class": obs,
        "class_mismatch": obs != wave,
        "cpu_part": cpu,
        "log_tail": f"decider.engine init cpu_part={cpu} ggml_cpu=v{cpu} threads=6\n",
        "latency_ms": lat,
        "answers": {"q": {"type": "noul", "noul": 0.5}},
        "lkai": lk,
    }


def _run(tmp_path, dl=DL):
    rows = []
    for rnd in range(3):
        for arm in ARMS:
            for i in range(2):
                rows.append(_row(arm, rnd, "cold", dl=dl[arm] + 0.1 * i + 0.05 * rnd))
                rows.append(_row(arm, rnd, "restore"))
                rows.append(_row(arm, rnd, "warm"))
    fn = {
        a: {
            "FunctionName": f"lkai-e1-{a}",
            "Runtime": "python3.12",
            "Architectures": ["arm64"],
            "MemorySize": 10240,
            "EphemeralStorage": {"Size": 512},
            "Timeout": 900,
            "SnapStart": {"ApplyOn": "PublishedVersions"},
            "Handler": "lkai_entry.handler",
            "environment": {"LKAI_PHASES": "1"},
        }
        for a in ARMS
    }
    cond = {
        "region": "us-east-2",
        "rounds": 3,
        "concurrency": 2,
        "warm_per_arm_per_round": 1,
        "arms": list(ARMS),
        "probe": "r1",
        "requests_file": "requests.json",
        "started": "t0",
        "finished": "t1",
        "requests": {"requests": [{"id": "r1"}]},
        "client": {},
        "runtime_version_arn": "arn:aws:lambda:us-east-2::runtime:abc",
        "functions": fn,
        "build_manifest": {
            "repo_commit": "abc",
            "repo_dirty": False,
            "model": {
                "id": "m",
                "revision": "216d1cc",
                "format": "GGUF Q8_0",
                "gguf_bytes": 2076712416,
                "gguf_sha256": "g",
            },
            "llama_cpp": {"commit": "f74e"},
            "function_zip": {"sha256": "z"},
            "images": {},
            "toolchain": [],
            "assets_sha256": {},
        },
        "rounds_meta": [
            {
                "t_versions_deleted": 100.0,
                "published": {a: {"t_publish": 0.0, "version": str(r + 2)} for a in ARMS},
            }
            for r in range(3)
        ],
    }
    (tmp_path / "conditions.json").write_text(json.dumps(cond))
    (tmp_path / "invocations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "logs").mkdir()
    for a in ARMS:
        with gzip.open(tmp_path / "logs" / f"lkai-e1-{a}.jsonl.gz", "wt") as f:
            for r in range(3):
                rec = {
                    "kind": "before_snapshot",
                    "wall_ns": r,
                    "init": {"weight_download": DL[a] - 0.5},
                }
                f.write(
                    json.dumps(
                        {
                            "timestamp": 0,
                            "logStreamName": f"2026/10/20/[{r + 2}]x",
                            "message": "LKAI " + json.dumps(rec),
                        }
                    )
                    + "\n"
                )
    return str(tmp_path)


def test_summary_contrasts_and_status(tmp_path):
    import summarize_e1

    res = summarize_e1.summarize_e1(_run(tmp_path), None, None, {"experiment_id": "E1-s3-express"})
    e1 = res["e1"]
    h = e1["comparison"]["bucket_type_no_vpc"]["cold.weight_download"]
    assert (
        e1["comparison"]["bucket_type_no_vpc"]["a"],
        e1["comparison"]["bucket_type_no_vpc"]["b"],
    ) == ("xz", "gp")
    assert h["n"] == [6, 6] and h["rounds_a_faster"] == 3
    assert h["delta_p50_s"] == pytest.approx(-10.0, abs=0.11)
    assert e1["comparison"]["vpc_confound_gp"]["cold.weight_download"][
        "delta_p50_s"
    ] == pytest.approx(1.0, abs=0.11)
    assert res["status"] == "improved" and res["primary_arm"] == "xz"
    # compute is not the variable: the same model_load_mmap on every arm gives a zero delta
    assert e1["comparison"]["bucket_type_no_vpc"]["cold.model_load_mmap"]["delta_p50_s"] == 0
    d = e1["weight_download_pcts"]["xz"]
    assert (
        d["cold"]["n"] == 6 and d["snapshot_init"]["n"] == 3 and d["snapshot_init"]["p50"] == 14.5
    )
    assert d["cold"]["p50"] <= d["cold"]["p95"] <= d["cold"]["p99"]


def test_summary_no_change_when_rounds_disagree(tmp_path):
    import summarize_e1

    res = summarize_e1.summarize_e1(_run(tmp_path, {**DL, "xz": 25.0}), None, None, {})
    assert res["status"] == "no-change"


def test_host_variability_one_cpu_part_is_comparable(tmp_path):
    import summarize_e1

    hv = summarize_e1.summarize_e1(_run(tmp_path), None, None, {})["e1"]["host_variability"]
    assert hv["headline_status_by_cpu_part"] == {"0xd40": "improved"}
    assert hv["verdict"].startswith("comparable")
    assert hv["cpu_part_mix"]["xz"]["cold"] == {"0xd40": 6}
    assert hv["observed"]["ggml_cpu_variant_by_cpu_part"] == {"0xd40": {"v0xd40 threads=6": 72}}
    assert "cgroup cpu and memory limits" in hv["unknown"]
    assert hv["pooled_mix_confounded"] == []
    # no CPU part is mapped to a generation anywhere in the section
    assert "raviton" not in json.dumps(hv)


def _mixed_rows():
    """xz lands on a slow part for every warm request, gp on a fast one; within a part both arms are equal."""
    rows = []
    for rnd in range(3):
        for arm in ("gp", "xz"):
            for i in range(3):
                rows.append(_row(arm, rnd, "cold", dl=DL[arm] + 0.1 * i, cpu="0xaaa"))
                rows.append(_row(arm, rnd, "cold", dl=DL[arm] + 0.1 * i, cpu="0xbbb"))
            # per round: xz 4 on the slow part + 3 on the fast one, gp the reverse, so pooled medians differ
            slow, fast = (4, 3) if arm == "xz" else (3, 4)
            rows += [_row(arm, rnd, "warm", lat=600.0, cpu="0xaaa") for _ in range(slow)]
            rows += [_row(arm, rnd, "warm", lat=300.0, cpu="0xbbb") for _ in range(fast)]
    return rows


def test_host_variability_flags_pooled_change_no_cpu_part_shows():
    import summarize_e1

    rows = _mixed_rows()
    cmp = {n: summarize_e1.contrast(rows, a, b) for n, (a, b) in summarize_e1.CONTRASTS.items()}
    hv = summarize_e1.host_variability(rows, "improved", cmp)
    # cold download: xz faster within both parts, so the headline is comparable
    assert hv["headline_status_by_cpu_part"] == {"0xaaa": "improved", "0xbbb": "improved"}
    # warm: pooled xz - gp is +300 ms every round, but zero within each part -> a CPU-mix artifact
    flagged = {(x["contrast"], x["measure"]): x for x in hv["pooled_mix_confounded"]}
    w = flagged[("bucket_type_no_vpc", "warm.steady_state_inference")]
    assert w["pooled_status"] == "regressed"
    assert set(w["status_by_cpu_part"].values()) == {"no-change"}
    assert all(d == 0 for d in w["delta_p50_s_by_cpu_part"].values())


def test_host_variability_confounded_when_a_part_disagrees():
    import summarize_e1

    rows = [
        _row(arm, rnd, "cold", dl=d + 0.1 * i, cpu=part)
        for rnd in range(3)
        for i in range(3)
        for part, dls in (("0xaaa", DL), ("0xbbb", {**DL, "xz": 30.0}))
        for arm, d in dls.items()
    ]
    cmp = {n: summarize_e1.contrast(rows, a, b) for n, (a, b) in summarize_e1.CONTRASTS.items()}
    hv = summarize_e1.host_variability(rows, "improved", cmp)
    assert hv["headline_status_by_cpu_part"] == {"0xaaa": "improved", "0xbbb": "regressed"}
    assert hv["verdict"].startswith("confounded") and "0xbbb" in hv["verdict"]


def test_host_variability_skips_parts_below_min_samples():
    import summarize_e1

    rows = [_row(a, r, "cold", dl=DL[a], cpu="0xaaa") for r in range(3) for a in ("gp", "xz")]
    rows += [_row("xz", 0, "cold", dl=40.0, cpu="0xccc")]
    hv = summarize_e1.host_variability(rows, "improved", {})
    assert "0xccc" not in hv["within_cpu_part"]["bucket_type_no_vpc"]
    assert hv["cpu_part_mix"]["xz"]["cold"] == {"0xaaa": 3, "0xccc": 1}


def test_host_variability_missing_cpu_parts_stay_unknown():
    import summarize_e1

    rows = [
        _row(arm, rnd, "cold", dl=DL[arm] + 0.1 * i, cpu=None)
        for rnd in range(3)
        for arm in ("gp", "xz")
        for i in range(3)
    ]
    cmp = {n: summarize_e1.contrast(rows, a, b) for n, (a, b) in summarize_e1.CONTRASTS.items()}
    hv = summarize_e1.host_variability(rows, "improved", cmp)
    assert hv["cpu_part_mix"]["gp"]["cold"] == {"unknown": 9}
    assert hv["headline_status_by_cpu_part"] == {}
    assert hv["within_cpu_part"]["bucket_type_no_vpc"] == {}
    assert hv["verdict"].startswith("unknown")


def test_host_variability_comparable_verdict_names_unassessed_unknown_rows():
    import summarize_e1

    known = [_row(a, r, "cold", dl=DL[a], cpu="0xd40") for r in range(3) for a in ("gp", "xz")]
    known += [
        _row(a, r, "cold", dl=DL[a] + 0.1, cpu="0xd40") for r in range(3) for a in ("gp", "xz")
    ]
    known += [
        _row(a, r, "cold", dl=DL[a] + 0.2, cpu="0xd40") for r in range(3) for a in ("gp", "xz")
    ]
    missing = [_row("gp", r, "cold", dl=DL["gp"], cpu=None) for r in range(2)]
    rows = known + missing
    cmp = {n: summarize_e1.contrast(rows, a, b) for n, (a, b) in summarize_e1.CONTRASTS.items()}
    hv = summarize_e1.host_variability(rows, "improved", cmp)
    assert hv["cpu_part_mix"]["gp"]["cold"]["unknown"] == 2
    assert "unknown" not in hv["within_cpu_part"]["bucket_type_no_vpc"]
    assert hv["verdict"].startswith("comparable")
    assert "unassessed (xz 0, gp 2)" in hv["verdict"]

    hv_all_known = summarize_e1.host_variability(known, "improved", cmp)
    assert "unassessed" not in hv_all_known["verdict"]


def test_s3_costs_priced_by_bucket_type(tmp_path):
    import summarize_e1

    res = summarize_e1.summarize_e1(_run(tmp_path), None, None, {})
    c = res["e1"]["s3_costs_by_arm"]
    # 6 colds + 3 snapshot inits per arm; 31 ranged GETs + 1 HEAD each
    assert c["gp"]["weight_downloads"] == c["xz"]["weight_downloads"] == 9
    assert c["gp"]["s3_request_usd"] == pytest.approx(9 * 32 * 0.0004 / 1000, rel=1e-3)
    assert c["xz"]["s3_request_usd"] == pytest.approx(9 * 32 * 0.00003 / 1000, rel=1e-3)
    assert c["gp"]["s3_retrieval_usd"] == 0.0
    assert c["xz"]["s3_retrieval_usd"] == pytest.approx(9 * 2.076712416 * 0.0006, rel=1e-3)
    assert c["xz"]["s3_storage_usd_month"] == pytest.approx(2.076712416 * 0.11, rel=1e-3)
    k = c["xz"]["per_1000_cold_starts_usd"]
    assert k["total"] == pytest.approx(k["lambda_duration"] + k["s3"], abs=1e-3)
    assert res["cost"]["total_usd"] >= res["cost"]["s3_retrieval_usd"]


def test_meta_and_result_validate_against_p2_schema_if_present(tmp_path):
    import summarize_e1

    meta = json.load(open(os.path.join(E1, "meta.json")))
    assert meta["changed_variable"]["name"] == "weights bucket type"
    p2 = os.path.join(
        os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "results", "lkai_results.py"
    )
    if not os.path.exists(p2):
        pytest.skip("P2's validator is on its own branch")
    spec = importlib.util.spec_from_file_location("lkai_results", p2)
    lr = importlib.util.module_from_spec(spec)
    sys.modules["lkai_results"] = lr  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(lr)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"outputs": {"r1": {"0xd40": {"q": {"type": "noul", "noul": 0.5}}}}}))
    res = summarize_e1.summarize_e1(
        _run(tmp_path), str(ref), "aac59d4", meta | {"conclusion": "test"}, "1234567"
    )
    assert res["correctness"]["compared"] == res["correctness"]["bit_identical"] == 72
    problems = lr.validate(res)
    assert problems == [], "\n".join(problems)


def test_bench_e1_makes_a_fresh_lambda_client_per_wave(monkeypatch):
    import bench_e1

    made = []
    monkeypatch.setattr(
        bench_e1, "lambda_client", lambda sess, c: made.append(c) or f"client-{len(made)}"
    )
    b = bench_e1.FreshClientBench.__new__(bench_e1.FreshClientBench)
    b._sess, b.concurrency, b.clients_made, b.probe = object(), 3, 0, "r1"
    seen = []
    b.invoke = lambda arm, q, rid, rnd, wave, seq: seen.append(b.lam) or {}
    b.wave("gp", "1", 0, "restore")
    b.wave("xz", "1", 0, "restore")
    assert made == [3, 3] and b.clients_made == 2
    assert seen == ["client-1"] * 3 + ["client-2"] * 3


def test_run_script_drives_bench_e1_with_all_four_arms():
    script = open(os.path.join(E1, "run_e1.sh")).read()
    assert '"$HERE/bench_e1.py"' in script and "--arms gp,xz,gpv,xzv" in script
    assert script.index("trap cleanup EXIT") < script.index('"$PY" "$HERE/bench_e1.py"')
