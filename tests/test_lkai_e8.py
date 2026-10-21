"""examples/lambda-kleidiai/e8: the dormant parallel-repack patch, the zips, the wrapper, the bench gates and the summary.

No AWS and no arm64 build. The patch runs on a minimal stand-in holding the exact anchors P1's pinned kleidiai.cpp has
after llama/patch_llama.py (and on the pinned tree itself when a checkout is available). make_zips runs on a synthetic
P1 zip. e8_host's parsers run on canned /proc text, and summarize_e8 runs on synthetic rows and hash files. The tests
pin E8's one-variable contract: with the switch unset the serial loops are upstream's text, the off and par arms share
one zip and differ only in GGML_KLEIDIAI_REPACK_THREADS, off differs from P1's zip only in the six libggml-cpu
variants, and every comparison is made within one CPU part.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import itertools
import json
import os
import re
import struct
import sys
import types
import zipfile

import pytest

E8 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e8")
)
BASE = os.path.abspath(os.path.join(E8, "..", "baseline"))
sys.path.insert(0, E8)
sys.path.insert(0, BASE)

import e8_host  # noqa: E402
import make_zips  # noqa: E402
import patch_parallel_repack as patch  # noqa: E402
import summarize_e8 as s8  # noqa: E402


def _sdk_stand_ins(monkeypatch) -> None:
    """deploy_e8.py and bench_e8.py import boto3/botocore at module level; the test env need not have them."""
    for m in ("deploy_e8", "bench_e8", "deploy", "bench"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    if (
        "boto3" not in sys.modules
        and importlib.util.find_spec("boto3")
        and importlib.util.find_spec("botocore")
    ):
        return
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    for name, mod in {
        "boto3": boto3,
        "botocore": types.ModuleType("botocore"),
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def deploy_e8(monkeypatch):
    _sdk_stand_ins(monkeypatch)
    return importlib.import_module("deploy_e8")


@pytest.fixture
def bench_e8(monkeypatch):
    _sdk_stand_ins(monkeypatch)
    return importlib.import_module("bench_e8")


# --- patch -----------------------------------------------------------------------------------------------------------

REQUANT_LOOP = (
    patch.LOOP_HEAD
    + " = static_cast<const uint8_t *>(data) + row * row_stride;\n"
    + "                // ... upstream's per-row requantisation ...\n"
    + patch.LOOP_TAIL
)
STAND_IN = (
    "#include <vector>\n#include <array>\n"
    "static inline int64_t ggml_ne(const ggml_tensor * tensor, int dim) { return 0; }\n\n"
    + patch.NAMESPACE
    + "size_t x, size_t y) { return x; }\n"
    "    int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {\n"
    "        if (want_q8 && slot_total > 0) {\n" + REQUANT_LOOP + "        }\n\n"
    "        if (want_f32 && slot_total > 0) {\n            bias.resize(n, 0.0f);\n        }\n"
    "            } else if (rhs_type == GGML_TYPE_Q8_0) {\n" + patch.SERIAL_PACK + "            }\n"
    "    }\n"
)


def test_patch_keeps_upstreams_serial_loops_verbatim_as_the_switch_off_path():
    out = patch.apply(STAND_IN)
    # the serial requant loop and the serial pack call are still there, unedited, in the else branches
    assert out.count(REQUANT_LOOP) == 1
    assert out.count(patch.SERIAL_PACK) == 1
    assert "} else\n" + REQUANT_LOOP in out
    assert "} else {\n" + patch.SERIAL_PACK in out
    # the parallel copies run only when the switch asks for more than one thread, and only for Q8_0
    assert out.count("tensor->type == GGML_TYPE_Q8_0 ? kleidiai_repack_threads() : 0") == 2
    assert "if (repack_nth > 1)" in out and "if (pack_nth > 1 && nr > 0)" in out
    # the block loop is the serial loop with only its bounds changed
    block = REQUANT_LOOP.replace(patch.SERIAL_FOR, patch.BLOCK_FOR, 1)
    assert out.count(block) == 1


def test_patch_packs_nr_aligned_blocks_at_their_packed_offsets():
    out = patch.apply(STAND_IN)
    assert "kleidiai_parallel_rows(pack_nth, n, nr," in out
    assert "dst_ptr + rhs_info->packed_size_ex(r0, k, nr, kr, 0)" in out
    assert "qdata.data() + r0 * k" in out and "scales.data() + r0" in out


def test_patch_switch_parse_is_unset_or_at_most_one_means_serial():
    # the C helper: getenv unset -> 0; atoi; capped at 64; > 1 is the only parallel value
    assert 'getenv("GGML_KLEIDIAI_REPACK_THREADS")' in patch.HELPERS
    assert "int v = s ? atoi(s) : 0;" in patch.HELPERS
    assert "if (v > 64)" in patch.HELPERS


def test_patch_refuses_missing_anchors_and_a_second_application():
    with pytest.raises(AssertionError, match="requant loop anchor"):
        patch.apply(STAND_IN.replace(patch.LOOP_HEAD, "for (;;) {"))
    with pytest.raises(AssertionError, match="q8 pack anchor"):
        patch.apply(STAND_IN.replace(patch.SERIAL_PACK, ""))
    with pytest.raises(AssertionError, match="already carries"):
        patch.apply(patch.apply(STAND_IN))


def _row_blocks(nth: int, n: int, align: int) -> list[tuple[int, int]]:
    """Python mirror of kleidiai_parallel_rows' block arithmetic."""
    groups = (n + align - 1) // align
    per = (groups + nth - 1) // nth
    out = [(0, min(n, per * align))]
    for t in range(1, nth):
        r0, r1 = min(n, t * per * align), min(n, (t + 1) * per * align)
        if r0 < r1:
            out.append((r0, r1))
    return out


@pytest.mark.parametrize("nth", [2, 3, 6, 7, 64])
@pytest.mark.parametrize(
    "n,align", [(2048, 4), (6144, 4), (248320, 4), (1023, 4), (5, 4), (17, 1), (3, 8)]
)
def test_row_blocks_tile_every_row_once_on_nr_boundaries(nth, n, align):
    blocks = _row_blocks(nth, n, align)
    assert blocks[0][0] == 0 and blocks[-1][1] == n
    assert all(a[1] == b[0] for a, b in itertools.pairwise(blocks))
    assert all(r0 % align == 0 for r0, _ in blocks)
    assert len(blocks) <= nth


def test_patch_anchors_match_the_pinned_tree_when_available():
    """When P1's pinned llama.cpp (patched by llama/patch_llama.py) is on disk, the real file must take the patch."""
    cands = glob.glob(
        os.path.join(BASE, "build", "llama.cpp", "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp")
    )
    if not cands:
        pytest.skip("no P1 llama.cpp checkout in baseline/build")
    s = open(cands[0]).read()
    if "kleidiai_repack_threads" in s:
        pytest.skip("checkout already carries the E8 patch")
    out = patch.apply(s)
    assert out.count("kleidiai_parallel_rows(") == 3  # the definition and two call sites


# --- zips ------------------------------------------------------------------------------------------------------------


def _p1_zip(tmp_path) -> tuple[str, str, str]:
    p1 = tmp_path / "p1.zip"
    with zipfile.ZipFile(p1, "w") as z:
        for name, data in [
            ("lkai_entry.py", b"print(1)\n"),
            ("lib/libllama.so.0.7.0", b"LLAMA"),
            ("lib/libggml-cpu-armv8.0_1.so", b"P1-a"),
            ("lib/libggml-cpu-armv8.6_1.so", b"P1-b"),
        ]:
            info = zipfile.ZipInfo(name, date_time=make_zips.STAMP)
            info.external_attr = (
                0o100755 if name.endswith(".so") or ".so." in name else 0o100644
            ) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        link = zipfile.ZipInfo("lib/libggml-cpu.so", date_time=make_zips.STAMP)
        link.external_attr = 0o120777 << 16
        z.writestr(link, b"libggml-cpu-armv8.0_1.so")
    man = tmp_path / "p1-manifest.json"
    man.write_text(json.dumps({"function_zip": {"sha256": make_zips.sha256(str(p1))}}))
    libs = tmp_path / "libs"
    libs.mkdir()
    (libs / "libggml-cpu-armv8.0_1.so").write_bytes(b"E8-a")
    (libs / "libggml-cpu-armv8.6_1.so").write_bytes(b"E8-b")
    return str(p1), str(man), str(libs)


def test_off_zip_differs_from_base_only_in_the_variant_libraries(tmp_path):
    p1, man, libs = _p1_zip(tmp_path)
    rec = make_zips.build(p1, man, libs, str(tmp_path / "out"))
    with (
        zipfile.ZipFile(p1) as a,
        zipfile.ZipFile(tmp_path / "out/base.zip") as b,
        zipfile.ZipFile(tmp_path / "out/off.zip") as o,
    ):
        for info in a.infolist():
            bi, oi = b.getinfo(info.filename), o.getinfo(info.filename)
            # base keeps every P1 entry byte for byte with its metadata
            assert b.read(bi) == a.read(info)
            assert (bi.date_time, bi.external_attr, bi.compress_type) == (
                info.date_time,
                info.external_attr,
                info.compress_type,
            )
            assert (oi.date_time, oi.external_attr, oi.compress_type) == (
                info.date_time,
                info.external_attr,
                info.compress_type,
            )
            if info.filename in rec["replaced"]:
                assert o.read(oi) != a.read(info)
            else:
                assert o.read(oi) == a.read(info)
        assert set(b.namelist()) == set(o.namelist()) == set(a.namelist()) | set(make_zips.WRAPPER)
    # the symlinked loader name is not a variant, so it is never replaced
    assert sorted(rec["replaced"]) == [
        "lib/libggml-cpu-armv8.0_1.so",
        "lib/libggml-cpu-armv8.6_1.so",
    ]


def test_make_zips_refuses_a_wrong_p1_zip_and_an_incomplete_lib_set(tmp_path):
    p1, man, libs = _p1_zip(tmp_path)
    os.remove(os.path.join(libs, "libggml-cpu-armv8.6_1.so"))
    with pytest.raises(SystemExit, match="exactly P1's variants"):
        make_zips.build(p1, man, libs, str(tmp_path / "out"))
    json.dump({"function_zip": {"sha256": "0" * 64}}, open(man, "w"))
    with pytest.raises(SystemExit, match="is not the zip"):
        make_zips.build(p1, man, libs, str(tmp_path / "out"))


def test_make_zips_refuses_a_patched_variant_identical_to_p1s(tmp_path):
    p1, man, libs = _p1_zip(tmp_path)
    open(os.path.join(libs, "libggml-cpu-armv8.0_1.so"), "wb").write(b"P1-a")
    with pytest.raises(SystemExit, match="did not reach it"):
        make_zips.build(p1, man, libs, str(tmp_path / "out"))


# --- deploy and bench gates ------------------------------------------------------------------------------------------


def test_arms_share_the_off_zip_and_differ_only_in_the_switch(deploy_e8):
    p1_env = {
        "MODEL_BUCKET": "b",
        "MODEL_KEY": "k",
        "N_SEQ": "17",
        "COLD_TAG": "x",
        "LKAI_PHASES": "0",
    }
    envs = {a: deploy_e8.arm_environment(p1_env, a) for a in deploy_e8.ARMS}
    assert deploy_e8.ARMS["off"][0] == deploy_e8.ARMS["par"][0] == "off"
    assert deploy_e8.ARMS["base"][0] == "base"
    assert deploy_e8.SWITCH not in envs["base"] and deploy_e8.SWITCH not in envs["off"]
    assert envs["par"][deploy_e8.SWITCH] == "6"
    strip = lambda e: {k: v for k, v in e.items() if k not in ("E8_ARM", deploy_e8.SWITCH)}  # noqa: E731
    assert strip(envs["base"]) == strip(envs["off"]) == strip(envs["par"])
    assert all(e["LKAI_PHASES"] == "1" and e["COLD_TAG"] == "0" for e in envs.values())
    assert deploy_e8.PREFIX == "lkai-5a61"


def test_bench_refuses_an_arm_whose_switch_or_phases_are_wrong(bench_e8):
    ok = {"LKAI_PHASES": "1", "E8_ARM": "par", "GGML_KLEIDIAI_REPACK_THREADS": "6"}
    assert bench_e8.arm_problem("par", ok) is None
    assert "needs '6'" in bench_e8.arm_problem("par", ok | {"GGML_KLEIDIAI_REPACK_THREADS": "3"})
    assert "needs None" in bench_e8.arm_problem("off", ok | {"E8_ARM": "off"})
    assert "LKAI_PHASES" in bench_e8.arm_problem("par", ok | {"LKAI_PHASES": "0"})
    assert "not an E8 arm" in bench_e8.arm_problem("inst", ok)


def test_bench_read_timeout_is_stripped_and_defaults_to_p1s(bench_e8):
    assert bench_e8.read_timeout_from(["--rounds", "5"]) == (900, ["--rounds", "5"])
    assert bench_e8.read_timeout_from(["--read-timeout", "300", "--out", "x"]) == (
        300,
        ["--out", "x"],
    )
    assert bench_e8.read_timeout_from(["--read-timeout=120"]) == (120, [])


def test_bench_per_arm_code_check_accepts_only_deployed_e8_zips(bench_e8, monkeypatch):
    import base64

    import deploy

    monkeypatch.setattr(deploy, "b64_to_hex", deploy.b64_to_hex)
    h = lambda x: base64.b64encode(bytes.fromhex(x)).decode()  # noqa: E731
    st = {
        "manifest": {"zips": {"base": {"sha256": "aa" * 32}, "off": {"sha256": "bb" * 32}}},
        "functions": {"base": {"code_sha256": "aa" * 32}, "off": {"code_sha256": "bb" * 32}},
    }
    bench_e8.per_arm_code_check(st)
    assert deploy.b64_to_hex(h("aa" * 32)) == st["manifest"]["function_zip"]["sha256"]
    assert deploy.b64_to_hex(h("bb" * 32)) == st["manifest"]["function_zip"]["sha256"]
    assert deploy.b64_to_hex(h("cc" * 32)) == "cc" * 32  # foreign code never passes
    assert (
        bench_e8.REAL_B64_TO_HEX(h("aa" * 32)) == "aa" * 32
    )  # the per-arm check keeps the real decoder


def test_bench_sources_list_every_e8_file(bench_e8):
    listed = {os.path.basename(p) for p in bench_e8.E8_SOURCES}
    tracked = {f for f in os.listdir(E8) if f.endswith((".py", ".cpp", ".sh"))}
    assert tracked <= listed


# --- host metadata ---------------------------------------------------------------------------------------------------

CPUINFO = """processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd i8mm
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
CPU revision\t: 1

processor\t: 1
CPU part\t: 0xd0c
"""


def test_cpuinfo_reads_the_first_block_and_labels_missing_fields_unknown():
    c = e8_host.cpuinfo(CPUINFO)
    assert (c["implementer"], c["part"], c["variant"], c["revision"], c["processors"]) == (
        "0x41",
        "0xd40",
        "0x1",
        "1",
        2,
    )
    assert e8_host.cpuinfo(None)["part"] == "unknown"
    assert e8_host.cpuinfo("processor : 0\n")["features"] == "unknown"


def test_auxv_and_loaded_variant_parsers():
    raw = struct.pack("<QQQQQQ", 16, 0xFF, 26, 0x2, 0, 0)
    assert e8_host.auxv(raw) == {"hwcap": "0xff", "hwcap2": "0x2"}
    assert e8_host.auxv(raw[:-1]) == {"hwcap": "unknown", "hwcap2": "unknown"}
    maps = (
        "ffff-ffff r-xp 0 0:0 1 /var/task/lib/libggml-cpu-armv8.6_1.so\n"
        "ffff-ffff r--p 0 0:0 1 /var/task/lib/libggml-cpu-armv8.6_1.so\n"
        "ffff-ffff r-xp 0 0:0 1 /var/task/lib/libllama.so.0.7.0\n"
    )
    assert e8_host.loaded_ggml_cpu(maps) == ["libggml-cpu-armv8.6_1.so"]
    assert e8_host.loaded_ggml_cpu(None) == "unknown"
    assert e8_host.proc_status_kb("VmRSS:\t  10 kB\nVmHWM:\t 20 kB\n") == {"VmRSS": 10, "VmHWM": 20}


def test_host_metadata_records_the_switch(monkeypatch):
    monkeypatch.delenv("GGML_KLEIDIAI_REPACK_THREADS", raising=False)
    assert e8_host.host_metadata()["repack_threads_env"] == "unset"
    monkeypatch.setenv("GGML_KLEIDIAI_REPACK_THREADS", "6")
    assert e8_host.host_metadata()["repack_threads_env"] == "6"


def test_handler_attaches_the_record_to_the_first_invocation_only(monkeypatch, capsys):
    calls = []

    def p1_handler(event, context):
        calls.append(event)
        return {"answers": {}, "latency_ms": 1, "lkai": {"inv": {}}}

    monkeypatch.setitem(sys.modules, "lkai_entry", types.SimpleNamespace(handler=p1_handler))
    monkeypatch.delitem(sys.modules, "e8_entry", raising=False)
    e8_entry = importlib.import_module("e8_entry")
    ctx = types.SimpleNamespace(log_stream_name="2026/10/21/[$LATEST]abc")
    first = e8_entry.handler({"a": 1}, ctx)
    second = e8_entry.handler({"a": 2}, ctx)
    assert first["lkai"]["e8"]["log_stream"] == "2026/10/21/[$LATEST]abc"
    assert "e8" not in second["lkai"]
    assert capsys.readouterr().out.count("E8 {") == 1
    assert len(calls) == 2


# --- summary ---------------------------------------------------------------------------------------------------------


def _row(arm, cls, part, rnd, v, **kw):
    return {
        "arm": arm,
        "expected_class": cls,
        "observed_class": cls,
        "class_mismatch": False,
        "error": None,
        "cpu_part": part,
        "round": rnd,
        "request_id": kw.get("rid", "r1"),
        "answers": kw.get("answers", {"q": {"choice": "a"}}),
        "latency_ms": v * 1000,
        "report": {"Init Duration": v * 1000, "Duration": 100.0, "Max Memory Used": 5000},
        "lkai": {"init": {"model_load_mmap": v}},
    }


def test_delta_is_within_cpu_part_and_never_strata_on_a_missing_part():
    rows = []
    for rnd in range(2):
        for i in range(4):
            rows += [
                _row("off", "cold", "0xd40", rnd, 8.0 + 0.01 * i),
                _row("par", "cold", "0xd40", rnd, 3.0 + 0.01 * i),
            ]
            rows += [
                _row("off", "cold", "0xd0c", rnd, 12.0 + 0.01 * i),
                _row("par", "cold", "0xd0c", rnd, 5.0),
            ]
    rows += [_row("par", "cold", None, 0, 0.1), _row("off", "cold", "unknown", 0, 99.0)]
    d = s8.delta(rows, "cold", s8.COLD_KEYS, "off", "par")["model_load_mmap"]
    assert set(d["by_cpu_part"]) == {"0xd40", "0xd0c"}
    assert d["by_cpu_part"]["0xd40"]["delta_p50"] == pytest.approx(-5.0)
    assert d["by_cpu_part"]["0xd40"]["ci95"][1] < 0
    assert d["rows_without_cpu_part"] == {"par": 1, "off": 1}
    assert {(c["round"], c["cpu_part"]) for c in d["cells"]} == {
        (0, "0xd40"),
        (0, "0xd0c"),
        (1, "0xd40"),
        (1, "0xd0c"),
    }
    assert d["cell_delta_signs"] == [-1]


def test_cells_record_a_part_only_one_arm_drew_in_a_round():
    rows = [_row("off", "cold", "0xd40", 0, 8.0), _row("par", "cold", "0xd40", 0, 3.0)]
    rows += [_row("off", "cold", "0xd4f", 0, 7.0)]
    d = s8.delta(rows, "cold", s8.COLD_KEYS, "off", "par")["model_load_mmap"]
    assert d["cells_missing_an_arm"] == [{"round": 0, "cpu_part": "0xd4f", "n": [1, 0]}]


def test_arm_agreement_flags_a_differing_answer_and_a_missing_base_part():
    rows = [_row("base", "warm", "0xd40", 0, 1.0), _row("off", "warm", "0xd40", 0, 1.0)]
    rows += [_row("par", "warm", "0xd40", 0, 1.0, answers={"q": {"choice": "b"}})]
    rows += [_row("par", "warm", "0xd0c", 0, 1.0)]
    g = s8.arm_agreement(rows)
    assert g["off"] == {"compared": 1, "bit_identical": 1, "no_base_answer_for_cpu_part": 0}
    assert g["par"] == {"compared": 1, "bit_identical": 0, "no_base_answer_for_cpu_part": 1}


def _hash_file(path, hashes, load_s):
    lines = [f"T {n} q8_0 CPU_KLEIDIAI 1 100 {h}\n" for n, h in hashes.items()]
    lines.append(f"S tensors={len(hashes)} packed={len(hashes)} model_load_s={load_s} all=ff\n")
    open(path, "w").writelines(lines)


def test_offline_hash_compares_every_run_with_p1s_first(tmp_path):
    ref = {"blk.0.w": "aa", "blk.1.w": "bb"}
    for rep in range(2):
        for cfg, load in (("p1", 8.0), ("off", 8.1), ("t6", 2.2)):
            _hash_file(tmp_path / f"{rep}-{cfg}.txt", ref, load)
    h = s8.offline_hash(str(tmp_path))
    assert h["all_identical"] is True and h["tensors"] == 2
    assert h["by_config"]["t6"]["model_load_s_p50"] == pytest.approx(2.2)
    _hash_file(tmp_path / "1-t6.txt", ref | {"blk.1.w": "cc"}, 2.2)
    h = s8.offline_hash(str(tmp_path))
    assert h["all_identical"] is False
    assert h["mismatches"] == [{"repeat": 1, "config": "t6", "tensors": ["blk.1.w"], "n": 1}]
    assert s8.offline_hash(str(tmp_path / "none"))["available"] is False


def _res(effect_ci, dormant_ci=(-1.0, 1.0), hashes_ok=True):
    by = lambda ci: {"0xd40": {"ci95": list(ci)}}  # noqa: E731
    pair = lambda ci: {  # noqa: E731
        "cold": {"first_request_total": {"by_cpu_part": by(ci)}},
        "restore": {"first_request_total": {"by_cpu_part": by((-1.0, 1.0))}},
        "warm": {"steady_state_inference": {"by_cpu_part": by((-1.0, 1.0))}},
    }
    agree = {"compared": 5, "bit_identical": 5, "no_base_answer_for_cpu_part": 0}
    return {
        "errors": {"count": 0},
        "correctness": {"compared": 5, "bit_identical": 5, "choice_flips": 0},
        "e8": {
            "pairs": {
                "effect": pair(effect_ci),
                "dormant": pair(dormant_ci),
                "total": pair(effect_ci),
            },
            "arm_agreement": {"base_internal_disagreements": 0, "off": agree, "par": agree},
            "offline_hash": {"all_identical": hashes_ok},
        },
    }


def test_status_follows_the_effect_headlines_and_needs_a_quiet_dormant_pair_and_clean_hashes():
    assert s8.status_of(_res((-6.0, -4.0))) == "improved"
    assert s8.status_of(_res((-1.0, 1.0))) == "no-change"
    assert s8.status_of(_res((0.5, 1.0))) == "regressed"
    assert s8.status_of(_res((-6.0, -4.0), dormant_ci=(0.2, 0.4))) == "inconclusive"
    assert s8.status_of(_res((-6.0, -4.0), hashes_ok=False)) == "inconclusive"


def test_no_graviton_generation_is_named_in_e8_sources():
    """Host variability rule: parts are reported as read; no Graviton generation is inferred from them."""
    pat = re.compile(r"graviton\s*[2-5]|neoverse[- ]?(n1|v1|v2|n2)", re.I)
    for path in glob.glob(os.path.join(E8, "*")):
        if os.path.isfile(path) and path.endswith((".py", ".sh", ".cpp", ".md", ".json")):
            assert not pat.search(open(path).read()), path
