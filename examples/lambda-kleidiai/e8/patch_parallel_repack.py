"""E8 patch for llama.cpp f74e2fa6a (after llama/patch_llama.py): a dormant multi-threaded KleidiAI weight repack.

  python patch_parallel_repack.py /path/to/llama.cpp

At model load, every Q8_0 weight in the CPU_KLEIDIAI buffer goes through tensor_traits::repack (kleidiai.cpp), called
from ggml_backend_cpu_kleidiai_buffer_set_tensor, one tensor at a time, on the loading thread. repack has two loops:
1. requantisation: each Q8_0 row becomes int8 values plus one per-row scale (qdata, scales);
2. the RHS pack: kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon interleaves groups of `nr` rows into the kernel's layout.

GGML_KLEIDIAI_REPACK_THREADS=N (N > 1) splits both loops of one tensor into N contiguous row blocks on N threads.
Unset, empty, 0 or 1 runs upstream's code path unchanged (the baseline). The pack's blocks start on a multiple of the
kernel's `nr`, so every packed row group is written by exactly one thread, from the same rows and scales the serial
pack reads; the requantisation's rows are independent. The bytes written are therefore the same; the offline check
(e8/hash_check.sh) confirms that per tensor rather than assuming it. Q6_K (a shared scratch row) and every other weight
type keep the serial path. The memset that zero-fills qdata stays serial.

The patch adds two functions and wraps two loops; it changes nothing in the packing routine or its parameters.
The byte identity is checked by an independent tool (e8/kaihash.cpp) that reads the loaded buffers back, so the same
check runs on P1's unpatched libraries.
"""

import argparse
from pathlib import Path

HELPERS = r"""
// strands-decider E8 patch: optional multi-threaded load-time repack (GGML_KLEIDIAI_REPACK_THREADS). Unset, empty,
// 0 or 1: upstream's serial path runs unchanged.
static int kleidiai_repack_threads() {
    static const int n = [] {
        const char * s = getenv("GGML_KLEIDIAI_REPACK_THREADS");
        int v = s ? atoi(s) : 0;
        if (v > 64) {
            v = 64;
        }
        if (v > 1) {
            fprintf(stderr, "kleidiai repack: threads=%d\n", v);  // once per process, only when the switch is on
        }
        return v;
    }();
    return n;
}

// fn(row_begin, row_end) over at most nth contiguous blocks of [0, n_rows). Every block boundary but n_rows is a
// multiple of align. The calling thread runs the first block.
template <typename F>
static void kleidiai_parallel_rows(int nth, size_t n_rows, size_t align, F && fn) {
    const size_t groups = (n_rows + align - 1) / align;
    const size_t per    = (groups + (size_t) nth - 1) / (size_t) nth;
    std::vector<std::thread> pool;
    for (int t = 1; t < nth; ++t) {
        const size_t r0 = std::min(n_rows, (size_t) t * per * align);
        const size_t r1 = std::min(n_rows, (size_t) (t + 1) * per * align);
        if (r0 < r1) {
            pool.emplace_back([&fn, r0, r1] { fn(r0, r1); });
        }
    }
    fn(0, std::min(n_rows, per * align));
    for (auto & th : pool) {
        th.join();
    }
}

"""


SERIAL_PACK = (
    "                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,\n"
    "                                       qdata.data(), nullptr, scales.data(),\n"
    "                                       dst_ptr, 0, &params);\n"
)
NAMESPACE = "namespace ggml::cpu::kleidiai {\n\nstatic size_t round_down("

# (what, anchor, replacement). Each anchor must occur exactly once in P1's patched kleidiai.cpp.
EDITS = [
    ("include", "#include <vector>\n", "#include <vector>\n#include <thread>\n"),
    ("namespace", NAMESPACE, HELPERS + NAMESPACE),
    # 2. The Q8_0 RHS pack: one call per nr-aligned row block, each into its own packed row groups.
    #    packed_size_ex(r0, ...) is the byte offset of row r0's group whenever r0 % nr == 0.
    (
        "q8 pack",
        SERIAL_PACK,
        "                const int pack_nth = tensor->type == GGML_TYPE_Q8_0 ? kleidiai_repack_threads() : 0;\n"
        "                if (pack_nth > 1 && nr > 0) {\n"
        "                    kleidiai_parallel_rows(pack_nth, n, nr, [&](size_t r0, size_t r1) {\n"
        "                        rhs_info->pack_func_ex(1, r1 - r0, k, nr, kr, sr, 0, 0,\n"
        "                                               qdata.data() + r0 * k, nullptr, scales.data() + r0,\n"
        "                                               dst_ptr + rhs_info->packed_size_ex(r0, k, nr, kr, 0), 0,\n"
        "                                               &params);\n"
        "                    });\n"
        "                } else {\n" + SERIAL_PACK + "                }\n",
    ),
]


LOOP_HEAD = (
    "            for (size_t row = 0; row < n; ++row) {\n                const uint8_t * row_data"
)
LOOP_TAIL = (
    "                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);\n"
    "                        }\n                    }\n                }\n            }\n"
)
SERIAL_FOR = "for (size_t row = 0; row < n; ++row)"
BLOCK_FOR = "for (size_t row = row_begin; row < row_end; ++row)"


def requant_edit(s: str) -> str:
    """1. The per-row requantisation. Upstream's loop stays byte for byte as the `else` branch, so the switch-off
    path compiles from upstream's text. The parallel branch is a copy of that loop over [row_begin, row_end), taken
    only for Q8_0 with N > 1 (Q6_K's shared row_f32 keeps it serial)."""
    if s.count(LOOP_HEAD) != 1:
        raise AssertionError("kleidiai.cpp requant loop anchor not found")
    start = s.index(LOOP_HEAD)
    if s.count(LOOP_TAIL, start) != 1:
        raise AssertionError("kleidiai.cpp requant loop end anchor not found")
    end = s.index(LOOP_TAIL, start) + len(LOOP_TAIL)
    loop = s[start:end]
    block = loop.replace(SERIAL_FOR, BLOCK_FOR, 1)
    new = (
        "            const int repack_nth = tensor->type == GGML_TYPE_Q8_0 ? kleidiai_repack_threads() : 0;\n"
        "            if (repack_nth > 1) {\n"
        "                kleidiai_parallel_rows(repack_nth, n, 1, [&](size_t row_begin, size_t row_end) {\n"
        + block
        + "                });\n"
        "            } else\n" + loop
    )
    return s[:start] + new + s[end:]


def apply(s: str) -> str:
    """The patched text of kleidiai.cpp. Refuses a tree whose anchors are missing or already patched."""
    if "kleidiai_repack_threads" in s:
        raise AssertionError("kleidiai.cpp already carries the E8 patch")
    s = requant_edit(s)
    for what, old, new in EDITS:
        if s.count(old) != 1:
            raise AssertionError(f"kleidiai.cpp {what} anchor not found")
        s = s.replace(old, new)
    return s


def patch(root: Path) -> None:
    k = root / "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"
    k.write_text(apply(k.read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    patch(Path(ap.parse_args().root))
    print("patched (E8 parallel repack)", ap.parse_args().root)
