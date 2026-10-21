"""E15 offline node profiler for llama.cpp f74e2fa6a's CPU backend. OFFLINE ONLY: never part of a deployed zip.

  python patch_nodeprof.py /path/to/llama.cpp     (after llama/patch_llama.py)

With LKAI_NODEPROF=1 in the environment, thread 0 of ggml_graph_compute_thread reads CLOCK_MONOTONIC when each graph
node starts and again when the barrier after it releases. Every thread meets that barrier, so the interval is the
node's wall time across all threads (compute plus load imbalance), and consecutive intervals tile the graph. The
per-node times go into a per-graph buffer; thread 0 folds them into a keyed table only after the graph's final
barrier, so the bookkeeping is never inside a timed node. A key is

  [fused+N:]op | node name | src0 type | src1 type | src0 buffer type (MUL_MAT) | src0 name | flags

with layer numbers removed (`blk.12.` and `-12`). Two exported functions read the table:
  lkai_nodeprof_reset()          clear it
  lkai_nodeprof_dump(path)       write it as TSV: key, nodes, total ns; plus `#graphs` and `#graph_ns`

Unset (or LKAI_NODEPROF=0) the loop takes one extra branch per node and nothing else changes; the arithmetic is the
same either way.
"""

import argparse
from pathlib import Path

HEADER = r"""
// lkai E15 offline node profiler (examples/lambda-kleidiai/e15/patch_nodeprof.py). OFFLINE ONLY, never deployed.
#define LKAI_NP_ROWS 4096
#define LKAI_NP_NODES 65536
struct lkai_np_row { char key[256]; int64_t n; int64_t ns; };
static struct lkai_np_row lkai_np_rows[LKAI_NP_ROWS];
static int lkai_np_nrows;
static int64_t lkai_np_overflow;
static int64_t lkai_np_graphs, lkai_np_graph_ns;
static int lkai_np_on = -1;
static int64_t lkai_np_buf_ns[LKAI_NP_NODES];
static int lkai_np_buf_nf[LKAI_NP_NODES];

static inline int64_t lkai_np_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int lkai_np_enabled(void) {
    if (lkai_np_on < 0) {
        const char * e = getenv("LKAI_NODEPROF");
        lkai_np_on = (e && e[0] == '1') ? 1 : 0;
    }
    return lkai_np_on;
}

// Copy `s` without layer numbers: "blk.<digits>." and "-<digits>" at a word end are dropped.
static void lkai_np_norm(char * out, size_t n, const char * s) {
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 1 < n;) {
        if (strncmp(s + i, "blk.", 4) == 0) {
            size_t j = i + 4;
            while (s[j] >= '0' && s[j] <= '9') j++;
            if (j > i + 4 && s[j] == '.') { i = j + 1; continue; }
        }
        if (s[i] == '-' && s[i + 1] >= '0' && s[i + 1] <= '9') {
            size_t j = i + 1;
            while (s[j] >= '0' && s[j] <= '9') j++;
            if (s[j] == 0 || s[j] == ' ' || s[j] == ')' || s[j] == '.') { i = j; continue; }
        }
        out[o++] = s[i++];
    }
    out[o] = 0;
}

static void lkai_np_key(char * k, size_t n, const struct ggml_tensor * t, int nf) {
    char name[GGML_MAX_NAME + 8], s0name[GGML_MAX_NAME + 8], fused[24] = "";
    const struct ggml_tensor * s0 = t->src[0];
    const struct ggml_tensor * s1 = t->src[1];
    lkai_np_norm(name, sizeof(name), t->name);
    lkai_np_norm(s0name, sizeof(s0name), s0 ? s0->name : "-");
    const char * bt = "";
    int hadamard = 0;
    if (t->op == GGML_OP_MUL_MAT && s0) {
        if (s0->buffer) {
            bt = ggml_backend_buft_name(ggml_backend_buffer_get_type(s0->buffer));
        }
        hadamard = ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD;
    }
    if (nf > 0) {
        snprintf(fused, sizeof(fused), "fused+%d:", nf);
    }
    // Non-KleidiAI F32 matmuls (attention): shapes and contiguity, to tell which llamafile_sgemm guard sends them to
    // ggml_vec_dot_f32. k = ne00 must be a multiple of 4, src1 must be contiguous, m = ne01 a multiple of 4.
    char shape[96] = "";
    if (t->op == GGML_OP_MUL_MAT && s0 && s1 && s0->type == GGML_TYPE_F32 && !hadamard) {
        snprintf(shape, sizeof(shape), ",s0=%lldx%lldx%lld%s,s1=%lldx%lldx%lld", (long long) s0->ne[0],
                 (long long) s0->ne[1], (long long) s0->ne[2], ggml_is_contiguous(s0) ? "" : "nc",
                 (long long) s1->ne[0], (long long) s1->ne[1], (long long) s1->ne[2]);
    }
    snprintf(k, n, "%s%s|%s|%s|%s|%s|%s|%s%s%s", fused, ggml_op_desc(t), name, s0 ? ggml_type_name(s0->type) : "-",
             s1 ? ggml_type_name(s1->type) : "-", bt, s0name, (s1 && !ggml_is_contiguous(s1)) ? "src1-noncontig" : "",
             hadamard ? ",hadamard" : "", shape);
}

static void lkai_np_flush(const struct ggml_cgraph * g, int64_t graph_ns) {
    char key[256];
    lkai_np_graphs++;
    lkai_np_graph_ns += graph_ns;
    const int nn = g->n_nodes < LKAI_NP_NODES ? g->n_nodes : LKAI_NP_NODES;
    for (int i = 0; i < nn; i++) {
        if (lkai_np_buf_ns[i] < 0) {
            continue;
        }
        lkai_np_key(key, sizeof(key), g->nodes[i], lkai_np_buf_nf[i]);
        int r = 0;
        while (r < lkai_np_nrows && strcmp(lkai_np_rows[r].key, key) != 0) r++;
        if (r == lkai_np_nrows) {
            if (lkai_np_nrows == LKAI_NP_ROWS) { lkai_np_overflow += lkai_np_buf_ns[i]; continue; }
            snprintf(lkai_np_rows[r].key, sizeof(lkai_np_rows[r].key), "%s", key);
            lkai_np_nrows++;
        }
        lkai_np_rows[r].n++;
        lkai_np_rows[r].ns += lkai_np_buf_ns[i];
    }
}

__attribute__((visibility("default"))) void lkai_nodeprof_reset(void) {
    memset(lkai_np_rows, 0, sizeof(lkai_np_rows));  // rows are reused by key: n and ns must restart at 0
    lkai_np_nrows = 0;
    lkai_np_overflow = 0;
    lkai_np_graphs = 0;
    lkai_np_graph_ns = 0;
}

__attribute__((visibility("default"))) int lkai_nodeprof_dump(const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "#graphs\t%lld\n#graph_ns\t%lld\n#overflow_ns\t%lld\n", (long long) lkai_np_graphs,
            (long long) lkai_np_graph_ns, (long long) lkai_np_overflow);
    for (int r = 0; r < lkai_np_nrows; r++) {
        fprintf(f, "%s\t%lld\t%lld\n", lkai_np_rows[r].key, (long long) lkai_np_rows[r].n, (long long) lkai_np_rows[r].ns);
    }
    return fclose(f);
}
"""

LOOP_OLD = """    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
"""
LOOP_NEW = """    const int lkai_np = state->ith == 0 && lkai_np_enabled() && cgraph->n_nodes <= LKAI_NP_NODES;
    const int64_t lkai_g0 = lkai_np ? lkai_np_now() : 0;
    int lkai_last = -1;
    int64_t lkai_t0 = 0;
    if (lkai_np) {
        for (int i = 0; i < cgraph->n_nodes; i++) lkai_np_buf_ns[i] = -1;
    }
    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        const int lkai_first = node_n;
"""
COMPUTE_OLD = """        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
"""
COMPUTE_NEW = """        if (lkai_np) {
            lkai_t0 = lkai_np_now();
        }
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
"""
BARRIER_OLD = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }
"""
BARRIER_NEW = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
        if (lkai_np) {
            lkai_np_buf_nf[lkai_first] = node_n - lkai_first;
            if (node_n + 1 < cgraph->n_nodes) {
                lkai_np_buf_ns[lkai_first] = lkai_np_now() - lkai_t0;
            } else {
                lkai_last = lkai_first;  // timed after the final barrier below
            }
        }
    }
"""
FINAL_OLD = """    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif
"""
FINAL_NEW = """    ggml_barrier(state->threadpool);

    if (lkai_np) {
        const int64_t lkai_g1 = lkai_np_now();
        if (lkai_last >= 0) {
            lkai_np_buf_ns[lkai_last] = lkai_g1 - lkai_t0;
        }
        lkai_np_flush(cgraph, lkai_g1 - lkai_g0);
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif
"""


def patch(root: Path) -> None:
    c = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    s = c.read_text()
    assert "lkai E15" not in s, "already patched"
    for old, new in (
        (LOOP_OLD, LOOP_NEW),
        (COMPUTE_OLD, COMPUTE_NEW),
        (BARRIER_OLD, BARRIER_NEW),
        (FINAL_OLD, FINAL_NEW),
    ):
        assert s.count(old) == 1, f"anchor not unique: {old[:60]!r}"
        s = s.replace(old, new)
    # The profiler's definitions go just before their only user, after every include.
    anchor = "static thread_ret_t ggml_graph_compute_thread(void * data) {"
    assert s.count(anchor) == 1, "ggml_graph_compute_thread anchor"
    idx = s.index(anchor)
    s = s[:idx] + HEADER + "\n" + s[idx:]
    c.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    patch(ap.parse_args().root)


if __name__ == "__main__":
    main()
