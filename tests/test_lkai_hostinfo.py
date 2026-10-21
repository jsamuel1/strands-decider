"""examples/lambda-kleidiai/baseline/hostinfo.py and its wiring into lkai_entry (H1's per-environment host record).

No Lambda: /proc/cpuinfo texts are fixtures, and lkai_entry runs against test_lkai_baseline's stand-ins. The tests pin
what the record must and must not say: identifiers as the kernel shows them, None plus a reason for anything absent,
no processor generation, a full record on the first invoke's response, and log lines small enough to keep the
4 KB log tail's START line.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "baseline")
sys.path.insert(0, os.path.abspath(BASE))

import buildcache  # noqa: E402
import hostinfo  # noqa: E402
import phases  # noqa: E402
from test_lkai_baseline import _ctx, _install_fakes  # noqa: E402

# A processor generation or core name. Bare v1/v2/n1 are too short to test for: they collide with "cgroup v2".
GENERATION = re.compile(r"graviton\s*\d?|neoverse|cortex-a\d+", re.I)

CPU_BLOCK = """processor\t: {i}
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics {extra}
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: {part}
CPU revision\t: 1
"""


def _cpuinfo(parts: list[str], extra: str = "sve i8mm bf16") -> str:
    return "\n".join(CPU_BLOCK.format(i=i, part=p, extra=extra) for i, p in enumerate(parts))


def test_parse_cpuinfo_single_identity():
    c = hostinfo.parse_cpuinfo(_cpuinfo(["0xd40"] * 6))
    assert c["processors"] == 6
    assert (c["implementer"], c["architecture"], c["variant"], c["part"], c["revision"]) == (
        "0x41",
        "8",
        "0x1",
        "0xd40",
        "1",
    )
    assert c["identities"] == [{**{k: c[k] for k, _ in hostinfo.CPU_KEYS}, "count": 6}]
    assert {"asimd", "sve", "i8mm", "bf16"} <= set(c["features"])


def test_parse_cpuinfo_mixed_identities_are_listed_not_collapsed():
    c = hostinfo.parse_cpuinfo(_cpuinfo(["0xd40", "0xd40", "0xd0c"]))
    assert "part" not in c  # no single answer: the tuple list is the record
    assert sorted((i["part"], i["count"]) for i in c["identities"]) == [("0xd0c", 1), ("0xd40", 2)]


def test_parse_cpuinfo_distinct_feature_sets_kept():
    text = (
        _cpuinfo(["0xd40"])
        + "\n"
        + _cpuinfo(["0xd40"], extra="sve2").replace("processor\t: 0", "processor\t: 1")
    )
    c = hostinfo.parse_cpuinfo(text)
    assert len(c["features_sets"]) == 2 and "features" not in c


def test_snapshot_unreadable_fields_are_none_with_a_reason(monkeypatch):
    monkeypatch.setattr(hostinfo, "_read", lambda path, limit=0: None)
    monkeypatch.setattr(hostinfo, "_hwcaps", lambda: None)
    r = hostinfo.snapshot(env={})
    assert r["cpu"] is None and r["midr_el1"] is None and r["hwcap"] is None and r["cgroup"] is None
    for k in (
        "cpu",
        "midr_el1",
        "hwcap",
        "cgroup",
        "mem_total_kb",
        "availability_zone",
        "physical_host",
    ):
        assert r["unknown"][k], k
    for k in hostinfo.LAMBDA_ENV:
        assert r["lambda"][k] is None and f"lambda.{k}" in r["unknown"]
    assert r["unknown"]["processor_generation"].startswith("never inferred")


def test_snapshot_reads_cpuinfo_cgroup_and_lambda_env(monkeypatch):
    files = {
        "/proc/cpuinfo": _cpuinfo(["0xd0c"] * 2),
        "/sys/fs/cgroup/cpu.max": "max 100000\n",
        "/sys/fs/cgroup/memory.max": "10737418240\n",
        "/proc/meminfo": "MemTotal:       10972280 kB\nMemFree: 1 kB\n",
        "/sys/devices/system/cpu/online": "0-1\n",
    }
    monkeypatch.setattr(hostinfo, "_read", lambda path, limit=0: files.get(path))
    monkeypatch.setattr(hostinfo, "_hwcaps", lambda: {"hwcap": 123, "hwcap2": 4})
    r = hostinfo.snapshot(
        env={"AWS_REGION": "us-east-2", "AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "10240"}
    )
    assert r["cpu"]["part"] == "0xd0c" and r["hwcap"] == {"hwcap": 123, "hwcap2": 4}
    assert r["cgroup"] == {
        "version": "v2",
        "cpu_max": "max 100000",
        "memory_max": "10737418240",
        "cpuset_effective": None,
    }
    assert r["mem_total_kb"] == 10972280 and r["cores"]["online"] == "0-1"
    assert r["lambda"]["AWS_REGION"] == "us-east-2"
    assert "cpu" not in r["unknown"] and "hwcap" not in r["unknown"]
    assert not GENERATION.search(json.dumps(r))


def test_fingerprint_depends_on_identity_not_timing(monkeypatch):
    monkeypatch.setattr(
        hostinfo,
        "_read",
        lambda path, limit=0: _cpuinfo(["0xd40"]) if path == "/proc/cpuinfo" else None,
    )
    monkeypatch.setattr(hostinfo, "_hwcaps", lambda: {"hwcap": 1, "hwcap2": 2})
    a, b = hostinfo.snapshot(env={}), hostinfo.snapshot(env={"AWS_REGION": "x"})
    assert a["fingerprint"] == b["fingerprint"] and re.fullmatch(r"[0-9a-f]{16}", a["fingerprint"])
    monkeypatch.setattr(
        hostinfo,
        "_read",
        lambda path, limit=0: _cpuinfo(["0xd0c"]) if path == "/proc/cpuinfo" else None,
    )
    assert hostinfo.snapshot(env={})["fingerprint"] != a["fingerprint"]


def test_snapshot_on_this_host_never_raises():
    r = hostinfo.snapshot()
    assert r["schema"] == hostinfo.SCHEMA and r["probe_s"] < 1.0
    json.dumps(r)


# ------------------------------------------------------------------ lkai_entry wiring


def _fake_host(monkeypatch, part):
    rec = {
        "schema": hostinfo.SCHEMA,
        "cpu": {"part": part},
        "fingerprint": f"fp-{part}",
        "unknown": {},
    }
    seq = iter([dict(rec), {**rec, "cpu": {"part": "0xd0c"}, "fingerprint": "fp-restored"}])
    monkeypatch.setattr(hostinfo, "snapshot", lambda env=None: next(seq))


def test_entry_first_response_carries_full_host_and_log_carries_fingerprint(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "1")
    _fake_host(monkeypatch, "0xd40")
    _install_fakes(monkeypatch, "on-demand")
    import lkai_entry

    ctx = _ctx(1)
    ctx.log_stream_name = "2026/10/21/[$LATEST]abc"
    out1 = lkai_entry.handler({"state": "s"}, ctx)
    out2 = lkai_entry.handler({"state": "s"}, _ctx(2))
    assert out1["lkai"]["host"]["cpu"]["part"] == "0xd40"
    assert out1["lkai"]["log_stream"] == "2026/10/21/[$LATEST]abc"
    assert "host" not in out2["lkai"]  # once per environment
    recs = phases.parse_lkai(capsys.readouterr().out)
    assert [r["kind"] for r in recs] == ["invoke", "invoke"]
    assert recs[0]["host_fingerprint"] == "fp-0xd40" and "host" not in recs[0]
    assert "host_fingerprint" not in recs[1]


def test_entry_restore_rereads_host(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "1")
    _fake_host(monkeypatch, "0xd40")
    reg = _install_fakes(monkeypatch, "snap-start")
    import lkai_entry

    reg["before"]()
    reg["after"]()
    out = lkai_entry.handler({"state": "s"}, _ctx(1))
    recs = phases.parse_lkai(capsys.readouterr().out)
    snap, restore = recs[0], recs[1]
    assert snap["kind"] == "before_snapshot" and snap["host"]["fingerprint"] == "fp-0xd40"
    assert restore["kind"] == "after_restore" and restore["host_fingerprint"] == "fp-restored"
    assert "host" not in restore
    # The restored copy's host, not the snapshot builder's.
    assert out["lkai"]["host"]["fingerprint"] == "fp-restored"


def test_entry_log_lines_stay_small_with_a_real_host_record(monkeypatch, capsys):
    """bench.py keeps 4 KB of log tail per invocation; the invoke line must leave room for START/REPORT."""
    monkeypatch.setenv("LKAI_PHASES", "1")
    _install_fakes(monkeypatch, "on-demand")
    import lkai_entry

    out = lkai_entry.handler({"state": "s"}, _ctx(1))
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith(phases.PREFIX)]
    assert len(lines) == 1 and len(lines[0]) < 1500
    assert len(json.dumps(out["lkai"]["host"])) > 0


def test_entry_report_only_arm_does_not_read_host(monkeypatch, capsys):
    monkeypatch.setenv("LKAI_PHASES", "0")
    called = []
    monkeypatch.setattr(hostinfo, "snapshot", lambda env=None: called.append(1) or {})
    _install_fakes(monkeypatch, "on-demand")
    import lkai_entry

    out = lkai_entry.handler({"state": "s"}, _ctx(1))
    assert "lkai" not in out and not called


def test_hostinfo_is_a_keyed_zip_input_and_is_packaged():
    assert "examples/lambda-kleidiai/baseline/hostinfo.py" in buildcache.INPUTS["zip"]
    cb = open(os.path.join(BASE, "container_build.sh")).read()
    assert re.search(r"cp [^\n]*/recipe/baseline/hostinfo\.py [^\n]*\$P/", cb)
    assert re.search(r"import [^\n]*\bhostinfo\b", cb)  # the package's import check covers it


@pytest.mark.parametrize(
    "text",
    [
        open(os.path.join(BASE, "hostinfo.py")).read(),
        open(os.path.join(BASE, "lkai_entry.py")).read(),
    ],
    ids=["hostinfo.py", "lkai_entry.py"],
)
def test_instrument_names_no_generation(text):
    assert not GENERATION.search(text)
