"""E15 (lambda-kleidiai): the offline node profiler's patch and its summariser."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

E15 = Path(__file__).resolve().parents[1] / "examples" / "lambda-kleidiai" / "e15"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"lkai_e15_{name}", E15 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


summ = _load("summarize_nodeprof")
patch = _load("patch_nodeprof")
cand = _load("patch_candidates")


def test_parse_key_reads_fused_count_and_flags():
    k = summ.parse_key("fused+1:RMS_NORM|attn_norm|f32|f32||inpL|")
    assert (k["op"], k["fused"], k["name"]) == ("RMS_NORM", 1, "attn_norm")
    k = summ.parse_key(
        "MUL_MAT|kq|f32|f32|CPU|cache_k_l3 (view) (permuted)|src1-noncontig,s0=256x768x2nc,s1=256x4x8"
    )
    assert k["src1_noncontig"] and not k["hadamard"]
    k = summ.parse_key("MUL_MAT|node_17|f32|f32|CPU|prism.hadamard.512|,hadamard")
    assert k["hadamard"]


@pytest.mark.parametrize(
    ("key", "category"),
    [
        (
            "MUL_MAT|ffn_up|q8_0|f32|CPU_KLEIDIAI|ffn_up.weight|",
            "KleidiAI matmul (Q8_0 weights, i8mm)",
        ),
        (
            "MUL_MAT|node_17|f32|f32|CPU|prism.hadamard.512|,hadamard",
            "Hadamard rotation (MUL_MAT -> ggml_compute_forward_fwht)",
        ),
        (
            "MUL_MAT|kq|f32|f32|CPU|cache_k_l3|src1-noncontig",
            "attention KQ (F32 MUL_MAT, ggml_vec_dot_f32)",
        ),
        ("MUL_MAT|kqv|f32|f32|CPU|cache_v_l3|", "attention KQV (F32 MUL_MAT, ggml_vec_dot_f32)"),
        ("CONCAT|conv_input|f32|f32||conv_states_reshaped|src1-noncontig", "CONCAT (E14)"),
        ("SOFT_MAX|kq_soft_max|f32|f32||kq|", "other: SOFT_MAX"),
    ],
)
def test_categorise_routes_each_key_once(key, category):
    assert list(summ.categorise([(key, 3, 7)])) == [category]


def _round(path: Path, rows: list[tuple[str, int, int]], graphs=2, graph_ns=100, overflow=0):
    lines = [f"#graphs\t{graphs}", f"#graph_ns\t{graph_ns}", f"#overflow_ns\t{overflow}"]
    lines += [f"{k}\t{n}\t{ns}" for k, n, ns in rows]
    path.write_text("\n".join(lines) + "\n")


def test_summarise_shares_are_of_graph_time_and_per_round(tmp_path):
    _round(
        tmp_path / "nodeprof1.tsv",
        [
            ("MUL_MAT|ffn_up|q8_0|f32|CPU_KLEIDIAI|w|", 4, 60),
            ("GATED_DELTA_NET|n|f32|f32||q|", 2, 30),
        ],
    )
    _round(
        tmp_path / "nodeprof2.tsv",
        [
            ("MUL_MAT|ffn_up|q8_0|f32|CPU_KLEIDIAI|w|", 4, 50),
            ("GATED_DELTA_NET|n|f32|f32||q|", 2, 40),
        ],
    )
    (tmp_path / "ctl1.json").write_text(
        json.dumps(
            {
                "phases_s": {"steady_state_total": 1.0},
                "steady_state_s": {"r": [0.1, 0.3, 0.2]},
                "answers": {},
            }
        )
    )
    s = summ.summarise(tmp_path)
    by = {r["category"]: r for r in s["table"]}
    assert by["KleidiAI matmul (Q8_0 weights, i8mm)"]["share_pct"] == [60.0, 50.0]
    assert by["GATED_DELTA_NET"]["share_pct"] == [30.0, 40.0]
    assert by["GATED_DELTA_NET"]["nodes_per_graph"] == 1
    assert s["runs"]["ctl1"]["p50_s"] == {"r": 0.2}
    assert "OFFLINE, NOT LAMBDA" in s["evidence"]
    assert "| GATED_DELTA_NET | 30.00% / 40.00% |" in summ.to_markdown(s)


def test_summarise_refuses_overflowed_key_table(tmp_path):
    _round(tmp_path / "nodeprof1.tsv", [], overflow=5)
    with pytest.raises(SystemExit, match="outside the key table"):
        summ.summarise(tmp_path)


def test_summarise_needs_a_table(tmp_path):
    with pytest.raises(SystemExit, match="no nodeprof"):
        summ.summarise(tmp_path)


def _fake_cpu_c() -> str:
    return (
        '#include "tiled/tiled.h"\n'
        "static thread_ret_t ggml_graph_compute_thread(void * data) {\n"
        + patch.LOOP_OLD
        + "        /* skip */\n"
        + patch.COMPUTE_OLD
        + patch.BARRIER_OLD
        + "\n"
        + patch.FINAL_OLD
        + "    return 0;\n}\n"
    )


def test_patch_applies_once_and_refuses_twice(tmp_path):
    c = tmp_path / "ggml/src/ggml-cpu/ggml-cpu.c"
    c.parent.mkdir(parents=True)
    c.write_text(_fake_cpu_c())
    patch.patch(tmp_path)
    s = c.read_text()
    assert s.index("lkai_nodeprof_dump") < s.index("static thread_ret_t ggml_graph_compute_thread")
    for new in (patch.LOOP_NEW, patch.COMPUTE_NEW, patch.BARRIER_NEW, patch.FINAL_NEW):
        assert s.count(new) == 1
    # reset must clear accumulated rows, not only the row count (warm-up leaked in without it)
    assert "memset(lkai_np_rows, 0, sizeof(lkai_np_rows))" in s
    with pytest.raises(AssertionError, match="already patched"):
        patch.patch(tmp_path)


def test_patch_refuses_a_moved_anchor(tmp_path):
    c = tmp_path / "ggml/src/ggml-cpu/ggml-cpu.c"
    c.parent.mkdir(parents=True)
    c.write_text(_fake_cpu_c().replace(patch.BARRIER_OLD, ""))
    with pytest.raises(AssertionError, match="anchor not unique"):
        patch.patch(tmp_path)


def _fake_tree(tmp_path: Path) -> Path:
    cpu = tmp_path / "ggml/src/ggml-cpu"
    cpu.mkdir(parents=True)
    (cpu / "ggml-cpu.c").write_text(
        "#include <x>\n"
        + cand.A_UNDEF_OLD
        + "void mm(void) {\n"
        + cand.A_GEMM1_OLD
        + "    }\n"
        + cand.A_GEMM2_OLD
        + "    }\n}\n"
    )
    (cpu / "ops.cpp").write_text(
        "// strands-decider prefill patch: scalar-gate, single-snapshot heads in one pass\n"
        + cand.B_FN_ANCHOR
        + "    const ggml_compute_params * params, ggml_tensor * dst) {\n"
        + cand.B_DISPATCH_OLD
        + "    while (1) {\n"
        + cand.B_CALL_OLD
        + "    }\n}\n"
    )
    return tmp_path


def test_candidate_a_gates_only_f32_sgemm_calls(tmp_path):
    root = _fake_tree(tmp_path)
    cand.patch_a(root)
    s = (root / "ggml/src/ggml-cpu/ggml-cpu.c").read_text()
    assert cand.A_GEMM1_NEW in s and cand.A_GEMM2_NEW in s
    assert "#undef GGML_USE_LLAMAFILE" not in s
    assert "src0->type == GGML_TYPE_F32" in s and 'getenv("LKAI_E15_A")' in s
    # non-SVE builds keep upstream behaviour: the gate compiles to 1
    assert "#define lkai_sgemm_ok(src0) 1" in s
    with pytest.raises(AssertionError, match="A already applied"):
        cand.patch_a(root)


def test_candidate_b_dispatches_rowblock_and_keeps_baseline_kernel(tmp_path):
    root = _fake_tree(tmp_path)
    cand.patch_b(root)
    s = (root / "ggml/src/ggml-cpu/ops.cpp").read_text()
    assert s.index("lkai_gdn_rowblock(const ggml_compute_params") < s.index(cand.B_FN_ANCHOR)
    assert "ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);" in s
    assert 'getenv("LKAI_E15_B")' in s
    with pytest.raises(AssertionError, match="B already applied"):
        cand.patch_b(root)


def test_candidate_b_needs_the_baseline_gdn_patch(tmp_path):
    root = _fake_tree(tmp_path)
    ops = root / "ggml/src/ggml-cpu/ops.cpp"
    ops.write_text(ops.read_text().replace("strands-decider prefill patch", "upstream"))
    with pytest.raises(AssertionError, match=r"needs patch_llama\.py gdn"):
        cand.patch_b(root)
