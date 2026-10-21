"""examples/lambda-kleidiai/e6: the bytecode zip builder, the host-metadata wrapper and the bench's arm order.

No AWS and no llama.cpp: pyc_zip.py runs on a synthetic P1-shaped zip, and e6_entry's parsers run on canned /proc
text. The tests pin E6's one-variable contract: ctl is P1's zip plus the wrapper, byte for byte. pyc differs only
in bytecode, which the 3.12 loader accepts without checking the source.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import struct
import sys
import types
import zipfile

import pytest

E6 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e6")
)
BASE = os.path.abspath(os.path.join(E6, "..", "baseline"))
sys.path.insert(0, E6)
sys.path.insert(0, BASE)

import pyc_zip  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    """A tiny zip shaped like P1's: .py stamped 2026-01-01, one stale timestamp .pyc, one .py without bytecode, a
    non-Python asset and a native library stand-in."""
    path = tmp_path / "p1.zip"
    src = {
        "pkg/__init__.py": b"X = 1\n",
        "pkg/mod.py": b"def f():\n    return 41 + 1\n",
        "top.py": b"Y = 2\n",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in src.items():
            z.writestr(zipfile.ZipInfo(name, date_time=STAMP), data)
        # a timestamp .pyc whose mtime field is the build time, not the zip's 2026-01-01 stamp
        stale = (
            importlib.util.MAGIC_NUMBER
            + struct.pack("<III", 0, 1792482575, len(src["top.py"]))
            + b"\0"
        )
        z.writestr(
            zipfile.ZipInfo(f"__pycache__/top.{sys.implementation.cache_tag}.pyc", date_time=STAMP),
            stale,
        )
        z.writestr(zipfile.ZipInfo("assets/tokenizer.json", date_time=STAMP), b"{}")
        z.writestr(zipfile.ZipInfo("lib/libfake.so", date_time=STAMP), b"\x7fELF")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "schema": "lkai-build-manifest/1",
                "repo_commit": "0" * 40,
                "repo_dirty": False,
                "model": {
                    "id": "m",
                    "revision": "r",
                    "format": "GGUF Q8_0",
                    "gguf_bytes": 1,
                    "gguf_sha256": "x",
                },
                "llama_cpp": {},
                "images": {},
                "toolchain": [],
                "assets_sha256": {},
                "function_zip": {
                    "bytes": path.stat().st_size,
                    "sha256": sha,
                    "handler": "lkai_entry.handler",
                },
            }
        )
    )
    return str(path), str(man)


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12), reason="pyc_zip compiles for Lambda's python3.12 only"
)
def test_ctl_is_p1_plus_wrapper_and_pyc_differs_only_in_bytecode(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    out = tmp_path / "build"
    m = pyc_zip.build(p1, man, str(out))
    with (
        zipfile.ZipFile(p1) as a,
        zipfile.ZipFile(out / "ctl.zip") as c,
        zipfile.ZipFile(out / "pyc.zip") as p,
    ):
        an = {i.filename: i for i in a.infolist()}
        cn = {i.filename: i for i in c.infolist()}
        assert set(cn) - set(an) == {pyc_zip.WRAPPER}
        for name, info in an.items():  # every P1 entry, bytes and metadata
            assert c.read(name) == a.read(name)
            assert (cn[name].date_time, cn[name].external_attr) == (
                info.date_time,
                info.external_attr,
            )
        pn = set(p.namelist())
        for name in an:
            if not name.endswith(".pyc"):
                assert p.read(name) == a.read(name)
        assert {n for n in pn - set(an) if not n.endswith((".pyc", "/"))} == {pyc_zip.WRAPPER}
        assert p.read(pyc_zip.WRAPPER) == c.read(pyc_zip.WRAPPER)
        # one .pyc per P1 .py, all unchecked-hash; the wrapper has none in either zip
        pycs = [n for n in pn if n.endswith(".pyc")]
        assert len(pycs) == 3
        for n in pycs:
            flags = int.from_bytes(p.read(n)[4:8], "little")
            assert flags == 1, n  # hash-based (bit 0), unchecked (bit 1 clear)
        assert not any("e6_entry" in n for n in pycs)
    assert m["bytecode"]["p1"] == {
        "pyc": 1,
        "valid": 0,
        "stale": 1,
        "no_source": 0,
        "py": 3,
        "py_without_pyc": 2,
    }
    assert m["bytecode"]["e6_pyc"]["stale"] == 0 and m["bytecode"]["e6_pyc"]["valid"] == 3
    assert m["bytecode"]["failed"] == []
    assert m["base"].endswith(json.loads(open(man).read())["function_zip"]["sha256"])
    assert m["function_zip"]["sha256"] == m["zips"]["pyc"]["sha256"]


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12), reason="pyc_zip compiles for Lambda's python3.12 only"
)
def test_pyc_zip_bytecode_imports_without_source_check(tmp_path):
    """The shipped bytecode is used by the loader even when the source's mtime and size no longer match."""
    p1, man = _p1_like_zip(tmp_path)
    out = tmp_path / "build"
    pyc_zip.build(p1, man, str(out))
    root = tmp_path / "task"
    with zipfile.ZipFile(out / "pyc.zip") as z:
        z.extractall(root)
    # Prove the bytecode, not the source, is what runs: change the source behind it.
    (root / "pkg" / "mod.py").write_text("def f():\n    return -1  # never compiled\n")
    sys.path.insert(0, str(root))
    try:
        for k in [k for k in sys.modules if k == "pkg" or k.startswith("pkg.")]:
            del sys.modules[k]
        mod = importlib.import_module("pkg.mod")
        assert mod.f() == 42
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if k == "pkg" or k.startswith("pkg.")]:
            del sys.modules[k]


def test_build_refuses_a_zip_the_manifest_does_not_describe(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    with open(p1, "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit, match="is not the zip"):
        pyc_zip.build(p1, man, str(tmp_path / "build"))


def test_stale_pyc_counts_a_matching_timestamp_pyc_as_valid(tmp_path):
    path = tmp_path / "z.zip"
    import calendar

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("a.py", date_time=STAMP), b"A = 1\n")
        mt = calendar.timegm((*STAMP, 0, 0, -1))
        head = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, mt, 6)
        z.writestr(
            zipfile.ZipInfo(f"__pycache__/a.{sys.implementation.cache_tag}.pyc", date_time=STAMP),
            head,
        )
    with zipfile.ZipFile(path) as z:
        assert pyc_zip.stale_pyc(z) == {
            "pyc": 1,
            "valid": 1,
            "stale": 0,
            "no_source": 0,
            "py": 1,
            "py_without_pyc": 0,
        }


# ---------------------------------------------------------------------------- e6_entry (no lkai_entry import)


def _entry_module(monkeypatch):
    """e6_entry with P1's lkai_entry replaced by a stand-in, so its import needs no model."""
    stub = types.ModuleType("lkai_entry")
    stub.handler = lambda event, context: {
        "answers": {"q": {"choice": "a"}},
        "latency_ms": 1.0,
        "lkai": {"inv": {}},
    }
    monkeypatch.setitem(sys.modules, "lkai_entry", stub)
    sys.modules.pop("e6_entry", None)
    return importlib.import_module("e6_entry")


CPUINFO = """processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
CPU revision\t: 1

processor\t: 1
CPU part\t: 0xd40
"""


def test_cpuinfo_reads_the_first_block_and_counts_processors(monkeypatch):
    e6 = _entry_module(monkeypatch)
    c = e6.cpuinfo(CPUINFO)
    assert (c["implementer"], c["variant"], c["part"], c["revision"]) == (
        "0x41",
        "0x1",
        "0xd40",
        "1",
    )
    assert "i8mm" in c["features"].split()
    assert c["processors"] == 2


def test_cpuinfo_labels_what_it_cannot_read_unknown(monkeypatch):
    e6 = _entry_module(monkeypatch)
    c = e6.cpuinfo("processor\t: 0\n")
    assert c["part"] == c["implementer"] == c["features"] == "unknown"
    assert e6.cpuinfo(None)["processors"] == "unknown"


def test_auxv_extracts_hwcaps_and_rejects_a_truncated_vector(monkeypatch):
    e6 = _entry_module(monkeypatch)
    raw = struct.pack("<QQQQQQ", 16, 0xEFFFFB, 26, 0x2F3FF, 0, 0)
    assert e6.auxv(raw) == {"hwcap": "0xeffffb", "hwcap2": "0x2f3ff"}
    assert e6.auxv(raw[:-1]) == {"hwcap": "unknown", "hwcap2": "unknown"}
    assert e6.auxv(None) == {"hwcap": "unknown", "hwcap2": "unknown"}


def test_bytecode_use_separates_usable_hash_timestamp_stale_and_missing(monkeypatch, tmp_path):
    e6 = _entry_module(monkeypatch)
    root = str(tmp_path) + "/"
    mods = {}

    def mod(name: str, head: bytes | None, src: bytes = b"V = 1\n"):
        py = tmp_path / f"{name}.py"
        py.write_bytes(src)
        cached = tmp_path / "__pycache__" / f"{name}.pyc"
        if head is not None:
            cached.parent.mkdir(exist_ok=True)
            cached.write_bytes(head)
        spec = types.SimpleNamespace(origin=str(py), cached=str(cached))
        mods[name] = types.SimpleNamespace(__spec__=spec)
        return py

    magic = importlib.util.MAGIC_NUMBER
    mod("h", magic + struct.pack("<I", 1) + b"\0" * 8)  # unchecked hash
    mod("c", magic + struct.pack("<I", 3) + b"\0" * 8)  # checked hash: loader re-hashes the source
    py = mod("t", None)
    st = os.stat(py)
    (tmp_path / "__pycache__" / "t.pyc").write_bytes(
        magic + struct.pack("<III", 0, int(st.st_mtime), st.st_size)
    )
    mod("s", magic + struct.pack("<III", 0, 1, 6))  # stale timestamp
    mod("m", None)  # no bytecode
    mods["outside"] = types.SimpleNamespace(
        __spec__=types.SimpleNamespace(origin="/usr/lib/x.py", cached=None)
    )
    got = e6.bytecode_use(root, mods)
    assert got == {
        "task_root_modules": 5,
        "pyc_present": 4,
        "usable_unchecked_hash": 1,
        "usable_timestamp": 1,
        "recompiled_from_source": 3,
    }


def test_handler_attaches_metadata_to_the_first_invocation_only(monkeypatch, capsys):
    e6 = _entry_module(monkeypatch)
    ctx = types.SimpleNamespace(log_stream_name="2026/10/21/[$LATEST]abc")
    first = e6.handler({}, ctx)
    assert first["lkai"]["e6"]["log_stream"] == ctx.log_stream_name
    assert "cpu" in first["lkai"]["e6"] and "bytecode" in first["lkai"]["e6"]
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("E6 ")]
    assert len(line) == 1 and json.loads(line[0][3:])["log_stream"] == ctx.log_stream_name
    second = e6.handler({}, ctx)
    assert "e6" not in second["lkai"]
    assert not [x for x in capsys.readouterr().out.splitlines() if x.startswith("E6 ")]


# ---------------------------------------------------------------------------- bench arm order


@pytest.fixture
def e6bench(monkeypatch):
    if not (importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore")):
        pytest.skip("e6bench imports boto3 at module level")
    return importlib.import_module("e6bench")


def test_order_alternates_which_arm_goes_first(e6bench):
    assert [e6bench.order_for(["ctl", "pyc"], r)[0] for r in range(4)] == [
        "ctl",
        "pyc",
        "ctl",
        "pyc",
    ]


# ---------------------------------------------------------------------------- summarize_e6


def _row(
    arm, cls, rnd, cpu="0xd40", imports=0.7, total_ms=36000.0, answers=None, error=None, rid="r1"
):
    return {
        "arm": arm,
        "expected_class": cls,
        "observed_class": cls,
        "class_mismatch": False,
        "round": rnd,
        "cpu_part": cpu,
        "error": error,
        "request_id": rid,
        "answers": answers if answers is not None else {"q": {"choice": "a"}},
        "latency_ms": 240.0,
        "report": {"Init Duration": 800.0, "Duration": total_ms},
        "lkai": {"init": {"imports": imports}},
    }


def _summ():
    sys.modules.pop("summarize_e6", None)
    return importlib.import_module("summarize_e6")


def test_delta_is_matched_by_cpu_part_and_counts_rounds():
    s = _summ()
    rows = []
    for rnd in range(3):
        for i in range(3):
            rows.append(_row("ctl", "cold", rnd, imports=0.70 + i * 0.01))
            rows.append(_row("pyc", "cold", rnd, imports=0.35 + i * 0.01))
    rows.append(_row("pyc", "cold", 0, cpu="0xd0c", imports=0.1))  # an unmatched part is left out
    d = s.delta(rows, "cold", {"imports": s.COLD_KEYS["imports"]})["imports"]
    assert set(d["by_cpu_part"]) == {"0xd40"}
    cell = d["by_cpu_part"]["0xd40"]
    assert cell["delta_p50"] == pytest.approx(-0.35)
    assert cell["ci95"][1] < 0 and cell["n"] == [9, 9]
    assert d["rounds_below_zero"] == 3


def test_arm_agreement_flags_a_differing_answer_and_a_missing_ctl_cell():
    s = _summ()
    rows = [
        _row("ctl", "warm", 0),
        _row("pyc", "warm", 0),
        _row("pyc", "warm", 0, answers={"q": {"choice": "b"}}),
    ]
    rows.append(_row("pyc", "warm", 0, cpu="0xd4f"))
    g = s.arm_agreement(rows)
    assert (g["compared"], g["bit_identical"], g["no_ctl_answer_for_cpu_part"]) == (2, 1, 1)


def _res(cold, restore, warm, *, errors=0, agree=(10, 10, 0, 0)):
    cell = lambda lo, hi: {"by_cpu_part": {"0xd40": {"ci95": [lo, hi]}}}  # noqa: E731
    return {
        "errors": {"count": errors},
        "correctness": {"compared": 5, "bit_identical": 5, "choice_flips": 0},
        "e6": {
            "delta": {
                "cold": {"imports": cell(-0.4, -0.3), "first_request_total": cell(*cold)},
                "restore": {"first_request_total": cell(*restore)},
                "warm": {"steady_state_inference": cell(*warm)},
            },
            "arm_agreement": dict(
                zip(
                    (
                        "compared",
                        "bit_identical",
                        "no_ctl_answer_for_cpu_part",
                        "ctl_internal_disagreements",
                    ),
                    agree,
                    strict=True,
                )
            ),
        },
    }


def test_host_variability_analysis_records_shared_and_missing_cells():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, imports=0.70),
        _row("pyc", "cold", 0, imports=0.35),
        _row("ctl", "cold", 1, imports=0.72),
        _row("pyc", "cold", 1, imports=0.34),
        _row("ctl", "cold", 0, cpu="0xd0c", imports=0.90),
        _row("pyc", "cold", 1, cpu="0xd0c", imports=0.44),
    ]
    h = s.host_variability_analysis(rows)
    imports = h["observed"]["primary_metrics"]["cold_imports"]
    assert [cell["delta"] for cell in imports["shared_cells"]] == [-0.35, -0.38]
    assert [cell["outcome"] for cell in imports["shared_cells"]] == ["unassessed", "unassessed"]
    assert imports["by_cpu_part"]["0xd40"] == {
        "shared_cell_count": 2,
        "assessed_cell_count": 0,
        "unassessed_cells": [
            {"round": 0, "reason": "fewer than two control observations"},
            {"round": 1, "reason": "fewer than two control observations"},
        ],
        "control_within_cell_spread_range": None,
        "delta_range": [-0.38, -0.35],
        "delta_signs": ["negative"],
        "effect_abs_floor": 0.35,
        "cell_outcomes": {"unassessed": 2},
        "outcome": "insufficient-control-cell-spread",
    }
    assert imports["missing_cells"] == [
        {"round": 0, "cpu_part": "0xd0c", "n": {"ctl": 1, "pyc": 0}},
        {"round": 1, "cpu_part": "0xd0c", "n": {"ctl": 0, "pyc": 1}},
    ]
    assert h["assessment"]["primary_effect"]["comparable"] is False
    assert h["assessment"]["primary_effect"]["outcome"] == "unassessed"
    assert "not within-round interleaving" in h["method"]


def test_host_variability_all_in_band_is_comparable_without_directional_claim():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, imports=0.70),
        _row("ctl", "cold", 0, imports=0.80),
        _row("pyc", "cold", 0, imports=0.71),
        _row("ctl", "cold", 1, imports=0.70),
        _row("ctl", "cold", 1, imports=0.80),
        _row("pyc", "cold", 1, imports=0.79),
    ]
    h = s.host_variability_analysis(rows)
    effect = h["assessment"]["primary_effect"]
    part = h["observed"]["primary_metrics"]["cold_imports"]["by_cpu_part"]["0xd40"]
    assert part["control_within_cell_spread_range"] == [0.1, 0.1]
    assert part["delta_signs"] == ["negative", "positive"]
    assert part["outcome"] == "within-control-variation"
    assert effect["comparable"] is True
    assert effect["outcome"] == "within-control-variation"
    assert "sign variation is a limitation" in effect["reason"]


def test_host_variability_host_conditioned_outcome_flip_requires_rerun():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.70),
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.72),
        _row("pyc", "cold", 0, cpu="0xd40", imports=0.30),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.70),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.72),
        _row("pyc", "cold", 1, cpu="0xd40", imports=0.32),
        _row("ctl", "cold", 0, cpu="0xd0c", imports=0.90),
        _row("ctl", "cold", 0, cpu="0xd0c", imports=0.92),
        _row("pyc", "cold", 0, cpu="0xd0c", imports=1.30),
        _row("ctl", "cold", 1, cpu="0xd0c", imports=0.90),
        _row("ctl", "cold", 1, cpu="0xd0c", imports=0.92),
        _row("pyc", "cold", 1, cpu="0xd0c", imports=1.32),
    ]
    h = s.host_variability_analysis(rows)
    effect = h["assessment"]["primary_effect"]
    assert effect["by_cpu_part"] == {
        "0xd0c": "positive-beyond-control-variation",
        "0xd40": "negative-beyond-control-variation",
    }
    assert effect["comparable"] is False
    assert effect["outcome"] == "host-conditioned"
    assert h["assessment"]["rerun"].startswith("needed")


def test_host_variability_single_round_part_is_not_comparable():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, imports=0.70),
        _row("pyc", "cold", 0, imports=0.35),
    ]
    h = s.host_variability_analysis(rows)
    effect = h["assessment"]["primary_effect"]
    part = h["observed"]["primary_metrics"]["cold_imports"]["by_cpu_part"]["0xd40"]
    assert part["outcome"] == "insufficient-control-cell-spread"
    assert part["control_within_cell_spread_range"] is None
    assert effect["comparable"] is False


def test_host_variability_analysis_does_not_treat_unknown_cpu_as_a_stratum():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, cpu=None, imports=0.70),
        _row("pyc", "cold", 0, cpu=None, imports=0.35),
    ]
    h = s.host_variability_analysis(rows)
    imports = h["observed"]["primary_metrics"]["cold_imports"]
    assert imports["shared_cells"] == []
    assert imports["missing_cells"] == []
    assert h["unknown"]["cpu_identity_rows"]["cold_imports"] == {"ctl": 1, "pyc": 1}
    assert h["assessment"]["primary_effect"]["unassessed_unknown_cpu_rows"] == {
        "ctl": 1,
        "pyc": 1,
    }
    assert h["assessment"]["primary_effect"]["comparable"] is False


def test_host_variability_mixed_known_and_unknown_names_unassessed_rows():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, imports=0.70),
        _row("ctl", "cold", 0, imports=0.71),
        _row("pyc", "cold", 0, imports=0.35),
        _row("ctl", "cold", 1, imports=0.72),
        _row("ctl", "cold", 1, imports=0.73),
        _row("pyc", "cold", 1, imports=0.34),
        _row("ctl", "cold", 0, cpu=None, imports=9.0),
        _row("pyc", "cold", 0, cpu="unknown", imports=1.0),
    ]
    effect = s.host_variability_analysis(rows)["assessment"]["primary_effect"]
    assert effect["comparable"] is True
    assert effect["unassessed_cpu_parts"] == []
    assert effect["unassessed_unknown_cpu_rows"] == {"ctl": 1, "pyc": 1}


def test_host_variability_sparse_same_sign_parts_stay_explicitly_unassessed():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.70),
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.72),
        _row("pyc", "cold", 0, cpu="0xd40", imports=0.35),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.71),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.73),
        _row("pyc", "cold", 1, cpu="0xd40", imports=0.34),
        _row("ctl", "cold", 0, cpu="0xd0c", imports=0.90),
        _row("pyc", "cold", 0, cpu="0xd0c", imports=0.40),
    ]
    effect = s.host_variability_analysis(rows)["assessment"]["primary_effect"]
    assert effect["comparable"] is True
    assert effect["unassessed_cpu_parts"] == ["0xd0c"]
    assert effect["unassessed_conflicts"] == []


def test_host_variability_sparse_opposite_sign_part_blocks_comparability():
    s = _summ()
    rows = [
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.70),
        _row("ctl", "cold", 0, cpu="0xd40", imports=0.72),
        _row("pyc", "cold", 0, cpu="0xd40", imports=0.35),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.71),
        _row("ctl", "cold", 1, cpu="0xd40", imports=0.73),
        _row("pyc", "cold", 1, cpu="0xd40", imports=0.34),
        _row("ctl", "cold", 0, cpu="0xd0c", imports=0.90),
        _row("pyc", "cold", 0, cpu="0xd0c", imports=1.10),
    ]
    effect = s.host_variability_analysis(rows)["assessment"]["primary_effect"]
    assert effect["comparable"] is False
    assert effect["unassessed_conflicts"] == ["0xd0c"]


def test_host_variability_persists_each_cell_outcome():
    s = _summ()
    rows = [
        # round 0: delta -0.36 beyond control spread 0.02
        _row("ctl", "cold", 0, imports=0.70),
        _row("ctl", "cold", 0, imports=0.72),
        _row("pyc", "cold", 0, imports=0.35),
        # round 1: delta +0.01 inside control spread 0.10
        _row("ctl", "cold", 1, imports=0.70),
        _row("ctl", "cold", 1, imports=0.80),
        _row("pyc", "cold", 1, imports=0.76),
        # round 2: delta +0.30 beyond control spread 0.02
        _row("ctl", "cold", 2, imports=0.70),
        _row("ctl", "cold", 2, imports=0.72),
        _row("pyc", "cold", 2, imports=1.01),
        # round 3: one control observation, no spread estimate
        _row("ctl", "cold", 3, imports=0.70),
        _row("pyc", "cold", 3, imports=0.35),
    ]
    h = s.host_variability_analysis(rows)
    imports = h["observed"]["primary_metrics"]["cold_imports"]
    assert [(c["round"], c["outcome"]) for c in imports["shared_cells"]] == [
        (0, "negative-beyond-control-variation"),
        (1, "within-control-variation"),
        (2, "positive-beyond-control-variation"),
        (3, "unassessed"),
    ]
    part = imports["by_cpu_part"]["0xd40"]
    # a consumer can see which cells drove the mixed part outcome
    assert part["cell_outcomes"] == {
        "negative-beyond-control-variation": 1,
        "positive-beyond-control-variation": 1,
        "unassessed": 1,
        "within-control-variation": 1,
    }
    assert part["outcome"] == "mixed-or-overlapping-control-variation"
    assert part["assessed_cell_count"] == 3
    assert part["unassessed_cells"] == [
        {"round": 3, "reason": "fewer than two control observations"}
    ]
    assert h["assessment"]["primary_effect"]["comparable"] is False
    # every primary metric's cells carry an outcome, not only imports
    for metric in h["observed"]["primary_metrics"].values():
        assert all("outcome" in c for c in metric["shared_cells"])


def test_cell_outcome_boundaries():
    s = _summ()
    assert s.cell_outcome(-0.1, None) == "unassessed"
    assert s.cell_outcome(-0.1, 0.1) == "within-control-variation"
    assert s.cell_outcome(0.1, 0.1) == "within-control-variation"
    assert s.cell_outcome(0.0, 0.0) == "within-control-variation"
    assert s.cell_outcome(-0.11, 0.1) == "negative-beyond-control-variation"
    assert s.cell_outcome(0.11, 0.1) == "positive-beyond-control-variation"


def test_part_outcome_aggregates_from_persisted_cell_outcomes():
    s = _summ()

    def cell(rnd, d, spread, outcome):
        sign = "negative" if d < 0 else "positive"
        return {
            "round": rnd,
            "delta": d,
            "sign": sign,
            "control_spread": spread,
            "outcome": outcome,
        }

    # the aggregate reads each cell's stored outcome, so the persisted cells and the part agree
    part = s._part_variability(
        [
            cell(0, -0.3, 0.02, "negative-beyond-control-variation"),
            cell(1, -0.3, 0.02, "within-control-variation"),
        ]
    )
    assert part["outcome"] == "mixed-or-overlapping-control-variation"
    assert part["cell_outcomes"] == {
        "negative-beyond-control-variation": 1,
        "within-control-variation": 1,
    }


# ---------------------------------------------------------------- snapshot-init host binding


def _snap_run(tmp_path, streams: dict[str, dict[str, list[str]]], published=None):
    """A run dir with saved logs: streams = {arm: {stream name: [messages]}}."""
    import gzip as _gz

    published = published if published is not None else {"ctl": ["1", "2"], "pyc": ["1", "2"]}
    rounds = sorted({v for vs in published.values() for v in vs})
    cond = {
        "arms": ["ctl", "pyc"],
        "functions": {a: {"FunctionName": f"fn-{a}"} for a in ("ctl", "pyc")},
        "rounds_meta": [
            {
                "round": 100 + i,
                "published": {a: {"version": v} for a, vs in published.items() if v in vs},
            }
            for i, v in enumerate(rounds)
        ],
    }
    os.makedirs(tmp_path / "logs", exist_ok=True)
    for arm, by_stream in streams.items():
        with _gz.open(tmp_path / "logs" / f"fn-{arm}.jsonl.gz", "wt") as fh:
            for name, msgs in by_stream.items():
                for m in msgs:
                    fh.write(
                        json.dumps({"logStreamName": name, "message": m, "timestamp": 0}) + "\n"
                    )
    return str(tmp_path), cond


_WALL = iter(range(10**6))


def _snap_msgs(cpu, imports):
    out = []
    if cpu is not None:
        for part in [cpu] if isinstance(cpu, str) else cpu:
            out.append(f"2026-10-21 05:13:45,093 decider.engine init cpu_part={part} threads=6\n")
    rec = {
        "kind": "before_snapshot",
        "wall_ns": next(_WALL),
        "init": {"imports": imports, "import_total": 30.0},
    }
    out.append("LKAI " + json.dumps(rec) + "\n")
    return out


def test_snapshot_binding_uses_only_the_records_own_stream(tmp_path):
    s = _summ()
    streams = {
        "ctl": {
            "2026/10/21/[1]a": _snap_msgs("0xd40", 0.66),
            "2026/10/21/[2]b": _snap_msgs("0xd40", 0.67),
            # a neighbouring invocation stream names another part; it must not bind to a snapshot record
            "2026/10/21/[1]c": ["decider.engine init cpu_part=0xd0c threads=6\n"],
        },
        "pyc": {
            "2026/10/21/[1]d": _snap_msgs("0xd40", 0.33),
            "2026/10/21/[2]e": _snap_msgs("0xd40", 0.32),
        },
    }
    h = s.snapshot_host_analysis(*_snap_run(tmp_path, streams))
    assert h["accounting"]["ctl"] == {
        "records": 2,
        "linked": 2,
        "missing": 0,
        "ambiguous": 0,
        "excluded_unpublished_records": 0,
    }
    assert h["arm_cpu_part_mix"] == {"ctl:0xd40": 2, "pyc:0xd40": 2}
    part = h["metrics"]["imports"]["by_cpu_part"]["0xd40"]
    assert part["per_version"]["ctl"] == {"1": 0.66, "2": 0.67}
    assert part["control_spread"] == pytest.approx(0.01)
    assert part["outcome"] == "negative-beyond-control-variation"
    assert h["assessment"]["comparable"] is True
    assert set(h["metrics"]["imports"]["by_cpu_part"]) == {"0xd40"}


def test_snapshot_missing_and_ambiguous_streams_are_counted_not_stratified(tmp_path):
    s = _summ()
    streams = {
        "ctl": {
            "2026/10/21/[1]a": _snap_msgs("0xd40", 0.66),
            "2026/10/21/[2]b": _snap_msgs("0xd40", 0.67),
            "2026/10/21/[1]m": _snap_msgs(None, 0.10),  # no CPU line in its stream
            "2026/10/21/[2]u": _snap_msgs("unknown", 0.10),  # unknown is never a part
            "2026/10/21/[2]x": _snap_msgs(["0xd40", "0xd0c"], 0.10),  # two distinct parts
        },
        "pyc": {
            "2026/10/21/[1]d": _snap_msgs("0xd40", 0.33),
            "2026/10/21/[2]e": _snap_msgs("0xd40", 0.32),
        },
    }
    h = s.snapshot_host_analysis(*_snap_run(tmp_path, streams))
    acc = h["accounting"]["ctl"]
    assert (acc["records"], acc["linked"], acc["missing"], acc["ambiguous"]) == (5, 2, 2, 1)
    amb = [r for r in h["records"]["ctl"] if r["binding"] == "ambiguous"]
    assert amb[0]["cpu_part"] is None and amb[0]["stream_cpu_parts"] == ["0xd0c", "0xd40"]
    # only linked records enter the comparison: the 0.10 values never touch the 0xd40 medians
    part = h["metrics"]["imports"]["by_cpu_part"]["0xd40"]
    assert part["n"] == {"ctl": 2, "pyc": 2} and part["range"]["ctl"] == [0.66, 0.67]
    assert set(h["metrics"]["imports"]["by_cpu_part"]) == {"0xd40"}
    assert "ctl:None" not in h["arm_cpu_part_mix"] and "ctl:unknown" not in h["arm_cpu_part_mix"]


def test_snapshot_records_from_unpublished_versions_are_excluded(tmp_path):
    s = _summ()
    streams = {
        "ctl": {
            "2026/10/21/[1]a": _snap_msgs("0xd40", 0.66),
            "2026/10/21/[2]b": _snap_msgs("0xd40", 0.67),
            "2026/10/21/[9]z": _snap_msgs("0xd40", 5.0),  # a stale version from another run
            "2026/10/21/[$LATEST]l": _snap_msgs("0xd40", 5.0),
        },
        "pyc": {
            "2026/10/21/[1]d": _snap_msgs("0xd40", 0.33),
            "2026/10/21/[2]e": _snap_msgs("0xd40", 0.32),
        },
    }
    h = s.snapshot_host_analysis(*_snap_run(tmp_path, streams))
    assert h["accounting"]["ctl"]["records"] == 2
    assert h["accounting"]["ctl"]["excluded_unpublished_records"] == 2
    assert {r["version"] for r in h["records"]["ctl"]} == {"1", "2"}
    assert h["metrics"]["imports"]["by_cpu_part"]["0xd40"]["range"]["ctl"] == [0.66, 0.67]


def test_snapshot_all_missing_is_unassessed_not_comparable(tmp_path):
    s = _summ()
    streams = {
        "ctl": {
            "2026/10/21/[1]a": _snap_msgs(None, 0.66),
            "2026/10/21/[2]b": _snap_msgs(None, 0.67),
        },
        "pyc": {
            "2026/10/21/[1]d": _snap_msgs(None, 0.33),
            "2026/10/21/[2]e": _snap_msgs(None, 0.32),
        },
    }
    h = s.snapshot_host_analysis(*_snap_run(tmp_path, streams))
    for arm in ("ctl", "pyc"):
        assert (h["accounting"][arm]["linked"], h["accounting"][arm]["missing"]) == (0, 2)
    assert h["arm_cpu_part_mix"] == {}
    assert h["metrics"]["imports"]["by_cpu_part"] == {}
    assert h["assessment"]["comparable"] is False
    assert h["assessment"]["outcome"] == "unassessed"


def test_snapshot_parts_that_disagree_are_host_conditioned(tmp_path):
    s = _summ()
    streams = {
        "ctl": {
            "2026/10/21/[1]a": _snap_msgs("0xd40", 0.66),
            "2026/10/21/[2]b": _snap_msgs("0xd40", 0.67),
            "2026/10/21/[1]c": _snap_msgs("0xd0c", 0.80),
            "2026/10/21/[2]f": _snap_msgs("0xd0c", 0.90),
        },
        "pyc": {
            "2026/10/21/[1]d": _snap_msgs("0xd40", 0.33),
            "2026/10/21/[2]e": _snap_msgs("0xd40", 0.32),
            "2026/10/21/[1]g": _snap_msgs("0xd0c", 0.84),
            "2026/10/21/[2]h": _snap_msgs("0xd0c", 0.86),
        },
    }
    h = s.snapshot_host_analysis(*_snap_run(tmp_path, streams))
    assert h["assessment"]["by_cpu_part"] == {
        "0xd0c": "within-control-variation",
        "0xd40": "negative-beyond-control-variation",
    }
    assert h["assessment"]["comparable"] is False
    assert h["assessment"]["outcome"] == "host-conditioned"


def test_status_follows_the_headlines_not_the_targeted_phase():
    s = _summ()
    null = (-0.1, 0.1)
    # imports improves (fixture) but no headline does: no-change, not improved
    assert s.status_of(_res(null, null, null)) == "no-change"
    assert s.status_of(_res((-0.5, -0.2), null, null)) == "improved"
    assert s.status_of(_res((-0.5, -0.2), null, (0.01, 0.02))) == "regressed"
    assert s.status_of(_res((-0.5, -0.2), null, null, errors=1)) == "inconclusive"
    assert s.status_of(_res((-0.5, -0.2), null, null, agree=(10, 9, 0, 0))) == "inconclusive"
    assert s.status_of(_res((-0.5, -0.2), null, null, agree=(10, 10, 1, 0))) == "inconclusive"
