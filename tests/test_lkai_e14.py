"""examples/lambda-kleidiai/e14: the concat patches and the Lambda-side analysis of P3's sampler logs.

No llama.cpp, no AWS. The patch tests run the patchers on a copy of the pinned kernel's text; the partition test mirrors
the patched loop's index arithmetic and checks that every dst row is written exactly once for every thread count; the
analysis tests build small synthetic sampler logs.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
from pathlib import Path

import pytest

E14 = Path(__file__).resolve().parent.parent / "examples" / "lambda-kleidiai" / "e14"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"e14_{name}", E14 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch_concat = _load("patch_concat")
patch_probe = _load("patch_concat_probe")
analyse = _load("analyse_lambda")
concat_zip = _load("concat_zip")

DISPATCH = """void ggml_compute_forward_concat(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } break;
        default:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}
"""


def _ops(tmp_path: Path, f16_too: bool = True) -> Path:
    """A minimal ops.cpp with the pinned f32 kernel, an f16 twin carrying the same loop text, and the dispatcher."""
    f32 = (
        "// ggml_compute_forward_concat\nstatic void ggml_compute_forward_concat_f32(\n    const ggml_compute_params * "
        "params,\n    ggml_tensor * dst) {\n    int64_t o[4] = {0, 0, 0, 0};\n" + patch_concat.OLD
    )
    f16 = patch_concat.OLD.replace("float", "ggml_fp16_t") if f16_too else ""
    root = tmp_path / "llama"
    (root / "ggml/src/ggml-cpu").mkdir(parents=True)
    p = root / "ggml/src/ggml-cpu/ops.cpp"
    p.write_text(
        "static void ggml_compute_forward_concat_f16(\n" + f16 + "\n" + f32 + "\n" + DISPATCH
    )
    return root


def test_concat_patch_rewrites_only_the_f32_kernel(tmp_path):
    root = _ops(tmp_path)
    before = (root / "ggml/src/ggml-cpu/ops.cpp").read_text()
    patch_concat.patch(root)
    after = (root / "ggml/src/ggml-cpu/ops.cpp").read_text()
    assert after.count("strands-decider concat patch") == 1
    f16_before = before[: before.index("// ggml_compute_forward_concat")]
    assert after.startswith(f16_before)  # the f16 path is untouched
    assert "for (int i2 = ith; i2 < ne2; i2 += nth)" not in after[after.index("concat_f32(") :]
    with pytest.raises(AssertionError, match="already patched"):
        patch_concat.patch(root)


def test_concat_patch_keeps_the_element_copy(tmp_path):
    """The element loop, the src0/src1 selection and the store are the same statements: only the row walk changes."""
    old_body = re.findall(r"if \(i0 < ne00.*?\*y = \*x;", patch_concat.OLD, re.S)
    new_body = re.findall(r"if \(i0 < ne00.*?\*y = \*x;", patch_concat.NEW, re.S)
    norm = [re.sub(r"\s+", " ", b) for b in (*old_body, *new_body)]
    assert len(old_body) == len(new_body) == 1 and norm[0] == norm[1]


def test_concat_patch_refuses_a_moved_kernel(tmp_path):
    root = _ops(tmp_path)
    p = root / "ggml/src/ggml-cpu/ops.cpp"
    p.write_text(
        p.read_text().replace("// TODO: smarter multi-theading", "// upstream rewrote this")
    )
    with pytest.raises(AssertionError, match="anchor"):
        patch_concat.patch(root)


def _rows_written(ne1: int, ne2: int, ne3: int, nth: int) -> list[tuple[int, int, int]]:
    """The patched kernel's row walk, transcribed: every (i1, i2, i3) each thread writes."""
    nr = ne1 * ne2 * ne3
    dr = (nr + nth - 1) // nth
    out = []
    for ith in range(nth):
        ir0 = dr * ith
        ir1 = min(ir0 + dr, nr)
        for ir in range(ir0, ir1):
            i3 = ir // (ne1 * ne2)
            i2 = (ir - i3 * ne1 * ne2) // ne1
            i1 = ir - i3 * ne1 * ne2 - i2 * ne1
            out.append((i1, i2, i3))
    return out


@pytest.mark.parametrize(
    "shape", [(6144, 1, 1), (6144, 4, 1), (6144, 17, 1), (7, 3, 2), (1, 1, 1), (5, 1, 1)]
)
@pytest.mark.parametrize("nth", [1, 2, 5, 6, 8, 64])
def test_patched_walk_writes_every_row_once(shape, nth):
    ne1, ne2, ne3 = shape
    rows = _rows_written(ne1, ne2, ne3, nth)
    want = [(i1, i2, i3) for i3 in range(ne3) for i2 in range(ne2) for i1 in range(ne1)]
    assert sorted(rows) == sorted(want) and len(rows) == len(want)


def test_patched_walk_uses_every_thread_for_one_sequence():
    """conv_input of a one-sequence ubatch: [d_conv-1+n_tokens, 6144, 1]. Unpatched, thread 0 alone copies it."""
    nth = 6
    per_thread = [0] * nth
    nr, dr = 6144, (6144 + nth - 1) // nth
    for ith in range(nth):
        per_thread[ith] = max(0, min(dr * (ith + 1), nr) - dr * ith)
    assert per_thread == [1024] * 6
    unpatched = [
        len(range(ith, 1, nth)) * 6144 for ith in range(nth)
    ]  # for i2 = ith; i2 < ne2(=1); i2 += nth
    assert unpatched == [6144, 0, 0, 0, 0, 0]


def test_probe_patch_wraps_only_the_f32_dispatch(tmp_path):
    root = _ops(tmp_path)
    patch_probe.patch(root)
    s = (root / "ggml/src/ggml-cpu/ops.cpp").read_text()
    assert s.count("lkai_e14_add(dst") == 1 and "LKAI_E14_PROBE" in s
    # a row reused after lkai_e14_probe_reset starts from zero, not from the previous key's counts
    assert "lkai_e14_rows[r].n = lkai_e14_rows[r].ns = lkai_e14_rows[r].bytes = 0;" in s
    assert s.index("lkai E14 offline concat probe") < s.index(
        "static void ggml_compute_forward_concat_f32("
    )
    with pytest.raises(AssertionError, match="already patched"):
        patch_probe.patch(root)


def _log(stream: str, msg: str) -> dict:
    return {"timestamp": 0, "logStreamName": stream, "message": msg}


def _inv(stream: str, n: int, cls: str, part: str | None, rid: str) -> dict:
    inv = {"n": n, "class": cls, "engine_s": 0.25}
    if part is not None:
        inv["cpu_part"] = part
    return _log(
        stream,
        "LKAI "
        + json.dumps({"kind": "invoke", "request_id": rid, "inv": inv}, separators=(",", ":")),
    )


def _prof(stream: str, n: int, samples: int, concat: int) -> dict:
    top = [
        {"symbol": "libggml-cpu-armv8.6_1.so:ggml_compute_forward_concat", "samples": concat},
        {"symbol": "libggml-cpu-armv8.6_1.so:kai_kernel_matmul", "samples": samples - concat},
    ]
    body = {
        "kind": "invoke",
        "n": n,
        "cpu_s": 1.125,
        "sampler": {"phases": {"invoke": {"samples": samples, "top_self": top}}},
    }
    return _log(stream, "LKAI_PROF " + json.dumps(body, separators=(",", ":")))


def _write_run(tmp_path: Path, logs: list[dict], rows: list[dict]) -> Path:
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    with gzip.open(run / "logs" / "lkai-p3-sampler.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in logs)
    with gzip.open(run / "invocations.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return run


def _row(rid: str, req: str) -> dict:
    return {
        "arm": "sampler",
        "request_id": req,
        "lkai": {"request_id": rid},
        "report": {"Duration": 250.0},
    }


def test_analysis_splits_by_part_and_origin_and_counts_all_samples(tmp_path):
    logs = [
        _inv("s1", 1, "restore", "0xd40", "a1"),
        _prof("s1", 1, 100, 2),
        _inv("s1", 2, "warm", "0xd40", "a2"),
        _prof("s1", 2, 50, 5),
        _inv("s1", 3, "warm", "0xd40", "a3"),
        _prof("s1", 3, 50, 3),
        _inv("s2", 1, "cold", "0xd0c", "b1"),
        _prof("s2", 1, 40, 1),
        _inv("s2", 2, "warm", "0xd0c", "b2"),
        _prof("s2", 2, 20, 4),
    ]
    rows = [
        _row("a1", "r1"),
        _row("a2", "r1"),
        _row("a3", "r2"),
        _row("b1", "r1"),
        _row("b2", "r1"),
    ]
    d = analyse.analyse(_write_run(tmp_path, logs, rows))
    assert d["records"] == 5
    assert d["by_class_part"]["warm|0xd40"] == {
        "invocations": 2,
        "samples": 100,
        "concat_samples": 8,
        "concat_share": 0.08,
        "cpu_s_per_engine_s_p50": 4.5,
    }
    assert set(d["warm_by_part_origin"]) == {"0xd40|origin=restore", "0xd0c|origin=cold"}
    assert d["warm_by_part_n"]["0xd40|n=2"]["concat_samples"] == 5
    assert d["warm_by_part_request"]["0xd40|r2"]["concat_share"] == 0.06


def test_analysis_keeps_missing_cpu_part_as_unknown_and_never_names_a_generation(tmp_path):
    logs = [_inv("s1", 1, "warm", None, "a1"), _prof("s1", 1, 10, 1)]
    d = analyse.analyse(_write_run(tmp_path, logs, [_row("a1", "r1")]))
    assert list(d["by_class_part"]) == ["warm|unknown"]
    text = json.dumps(d) + (E14 / "analyse_lambda.py").read_text()
    assert not re.search(r"graviton\s*\d", text, re.I)


def test_analysis_skips_unlinked_profiles(tmp_path):
    logs = [_prof("s9", 4, 10, 1), _inv("s1", 1, "warm", "0xd40", "a1"), _prof("s1", 1, 10, 0)]
    d = analyse.analyse(_write_run(tmp_path, logs, [_row("a1", "r1")]))
    assert d["records"] == 1 and d["by_class_part"]["warm|0xd40"]["concat_samples"] == 0


def _p1_zip(tmp_path: Path) -> Path:
    import zipfile

    z = tmp_path / "p1.zip"
    with zipfile.ZipFile(z, "w") as f:
        for name, data in (
            ("lkai_entry.py", b"x = 1\n"),
            ("lib/libggml-cpu-armv8.6_1.so", b"P1-86"),
            ("lib/libggml-cpu-armv8.2_2.so", b"P1-82"),
            ("lib/libllama.so.0", b"llama"),
        ):
            i = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            i.external_attr = 0o100755 << 16
            f.writestr(i, data)
        s = zipfile.ZipInfo("lib/libllama.so", date_time=(2026, 1, 1, 0, 0, 0))
        s.external_attr = 0o120777 << 16  # a symlink entry, as P1's zip -y stores them
        f.writestr(s, b"libllama.so.0")
    return z


def test_zips_differ_only_in_the_ggml_cpu_variants(tmp_path):
    import zipfile

    libs = tmp_path / "rows"
    libs.mkdir()
    (libs / "libggml-cpu-armv8.6_1.so").write_bytes(b"ROWS-86")
    (libs / "libggml-cpu-armv8.2_2.so").write_bytes(b"ROWS-82")
    (tmp_path / "e14_entry.py").write_text("# entry\n")
    m = concat_zip.build(
        str(_p1_zip(tmp_path)), str(libs), str(tmp_path / "e14_entry.py"), str(tmp_path / "out")
    )
    p1 = zipfile.ZipFile(tmp_path / "p1.zip")
    ctl = zipfile.ZipFile(tmp_path / "out" / "ctl.zip")
    rows = zipfile.ZipFile(tmp_path / "out" / "rows.zip")
    for i in p1.infolist():  # ctl carries every P1 entry unchanged, attributes included
        j = ctl.getinfo(i.filename)
        assert ctl.read(j) == p1.read(i) and j.external_attr == i.external_attr
    assert set(ctl.namelist()) - set(p1.namelist()) == {"e14_entry.py"}
    changed = {n for n in ctl.namelist() if ctl.read(n) != rows.read(n)}
    assert changed == {"lib/libggml-cpu-armv8.6_1.so", "lib/libggml-cpu-armv8.2_2.so"}
    assert rows.read("lib/libggml-cpu-armv8.6_1.so") == b"ROWS-86"
    assert rows.getinfo("lib/libllama.so").external_attr == 0o120777 << 16
    assert set(m["replaced_in_rows"]) == changed


def test_zip_builder_refuses_a_different_variant_set(tmp_path):
    libs = tmp_path / "rows"
    libs.mkdir()
    (libs / "libggml-cpu-armv8.6_1.so").write_bytes(b"ROWS-86")
    (tmp_path / "e14_entry.py").write_text("# entry\n")
    with pytest.raises(SystemExit, match="variant set differs"):
        concat_zip.build(
            str(_p1_zip(tmp_path)), str(libs), str(tmp_path / "e14_entry.py"), str(tmp_path / "out")
        )


def test_entry_metadata_parsers(monkeypatch):
    import struct
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "lkai_entry", types.SimpleNamespace(handler=lambda e, c: {"lkai": {}})
    )
    spec = importlib.util.spec_from_file_location("e14_entry_t", E14 / "e14_entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\nprocessor\t: 1\nCPU part\t: 0xd40\n"
    c = mod.cpuinfo(info)
    assert c["part"] == "0xd40" and c["processors"] == 2 and c["variant"] == "unknown"
    raw = struct.pack("<QQQQQQ", 16, 0xFF, 26, 0x1, 0, 0)
    assert mod.auxv(raw) == {"hwcap": "0xff", "hwcap2": "0x1"}
    assert mod.auxv(b"\x00" * 7) == {"hwcap": "unknown", "hwcap2": "unknown"}
    out = mod.handler({}, types.SimpleNamespace(log_stream_name="s"))
    rec = out["lkai"]["e14"]
    assert set(rec) == {
        "at_init",
        "at_first_invoke",
        "ggml_cpu",
        "log_stream",
    }  # restore: re-read at first invoke
    assert "e14" not in mod.handler({}, None)["lkai"]  # only the first invocation carries it


summ = _load("summarize_e14")


def _wrow(arm, rnd, part, rid, ms):
    return {
        "arm": arm,
        "round": rnd,
        "cpu_part": part,
        "request_id": rid,
        "latency_ms": ms,
        "report": {"Duration": ms + 2},
        "leg": "cold",
        "error": None,
        "class_mismatch": False,
        "expected_class": "warm",
    }


def test_cells_compare_within_round_part_and_request_only():
    rows = (
        [_wrow("ctl", 0, "0xd40", "a", v) for v in (250, 252, 251)]
        + [_wrow("rows", 0, "0xd40", "a", v) for v in (240, 241, 239)]
        + [_wrow("ctl", 0, "0xd0c", "a", 450)]
        + [_wrow("rows", 0, "0xd0c", "a", 440)]
        + [_wrow("rows", 1, "0xd40", "a", 100)]
    )
    cells = summ.cell_table(
        rows, summ._engine, lambda r: (r["round"], summ._part(r), r["request_id"])
    )
    by = {tuple(c["cell"]): c for c in cells}
    assert by[(0, "0xd40", "a")]["outcome"] == "negative-beyond-control-variation"
    assert by[(0, "0xd40", "a")]["n"] == [3, 3]
    assert by[(0, "0xd0c", "a")]["outcome"] == "unassessed"  # one control sample: no spread
    assert by[(1, "0xd40", "a")]["outcome"] == "unmatched"
    s = summ.part_summary(cells)
    assert s["0xd40"]["outcomes"] == {"negative-beyond-control-variation": 1, "unmatched": 1}


def test_pooled_gain_hidden_by_part_mix_is_not_reported_as_within_part_gain():
    """rows lands on the faster part more often: pooled, it looks faster; within each part it is not."""
    rows = (
        [_wrow("ctl", 0, "0xd40", "a", v) for v in (250, 251, 249)]
        + [_wrow("ctl", 0, "0xd0c", "a", v) for v in (450, 452, 448, 451, 449)]
        + [_wrow("rows", 0, "0xd40", "a", v) for v in (250, 251, 249, 250, 250)]
        + [_wrow("rows", 0, "0xd0c", "a", v) for v in (450, 451, 449)]
    )
    pooled_ctl = statistics_median([r["latency_ms"] for r in rows if r["arm"] == "ctl"])
    pooled_rows = statistics_median([r["latency_ms"] for r in rows if r["arm"] == "rows"])
    assert pooled_rows < pooled_ctl  # the confound
    s = summ.part_summary(
        summ.cell_table(rows, summ._engine, lambda r: (r["round"], summ._part(r), r["request_id"]))
    )
    assert all(p["outcomes"] == {"within-control-variation": 1} for p in s.values())


def statistics_median(xs):
    import statistics

    return statistics.median(xs)


def test_missing_cpu_part_is_its_own_unknown_stratum():
    rows = [
        _wrow("ctl", 0, None, "a", 250),
        _wrow("ctl", 0, None, "a", 252),
        _wrow("rows", 0, None, "a", 240),
    ]
    s = summ.part_summary(
        summ.cell_table(rows, summ._engine, lambda r: (r["round"], summ._part(r), r["request_id"]))
    )
    assert list(s) == ["unknown"] and "never a comparison stratum" in s["unknown"]["note"]


def test_no_generation_names_in_e14_sources():
    for p in E14.glob("*.py"):
        assert not re.search(r"graviton\s*[2-5]", p.read_text(), re.I), p.name
