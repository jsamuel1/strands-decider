"""E14 OFFLINE-ONLY probe for llama.cpp f74e2fa6a: what every f32 CONCAT copies and how long thread 0 spends on it.

  python patch_concat_probe.py /path/to/llama.cpp          (after llama/patch_llama.py; never part of a deployed zip)

With LKAI_E14_PROBE=1 in the environment, thread 0 reads CLOCK_MONOTONIC around its share of each f32 CONCAT and adds
the call to a table keyed by (dim, dst ne0..ne3, src0 ne0, src1 contiguous?, node name). In the unpatched kernel work is
split by dst's dim 2, so for a one-sequence ubatch (ne2 = 1) thread 0 does the whole copy and its time is the copy's
time; the other threads wait at the barrier after the node. lkai_e14_probe_dump(path) writes the table as TSV.
Unset, the kernel takes one extra branch per call and nothing else changes.
"""

import argparse
from pathlib import Path

HEADER = r"""
// lkai E14 offline concat probe (examples/lambda-kleidiai/e14/patch_concat_probe.py). OFFLINE ONLY, never deployed.
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#define LKAI_E14_ROWS 512
struct lkai_e14_row { char key[160]; int64_t n; int64_t ns; int64_t bytes; };
static struct lkai_e14_row lkai_e14_rows[LKAI_E14_ROWS];
static int lkai_e14_nrows;
static int lkai_e14_on = -1;
static inline int64_t lkai_e14_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int lkai_e14_enabled(void) {
    if (lkai_e14_on < 0) {
        const char * e = getenv("LKAI_E14_PROBE");
        lkai_e14_on = (e && e[0] == '1') ? 1 : 0;
    }
    return lkai_e14_on;
}
static void lkai_e14_add(const ggml_tensor * dst, int64_t ns) {
    char key[160];
    const ggml_tensor * s0 = dst->src[0];
    const ggml_tensor * s1 = dst->src[1];
    snprintf(key, sizeof(key), "dim=%d|dst=%lldx%lldx%lldx%lld|src0_ne0=%lld|src1_contig=%d|%.40s",
             ggml_get_op_params_i32(dst, 0), (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
             (long long) dst->ne[3], (long long) s0->ne[0], (int) ggml_is_contiguous(s1), dst->name);
    int r = 0;
    while (r < lkai_e14_nrows && strcmp(lkai_e14_rows[r].key, key) != 0) r++;
    if (r == lkai_e14_nrows) {
        if (lkai_e14_nrows == LKAI_E14_ROWS) return;
        snprintf(lkai_e14_rows[r].key, sizeof(lkai_e14_rows[r].key), "%s", key);
        lkai_e14_rows[r].n = lkai_e14_rows[r].ns = lkai_e14_rows[r].bytes = 0;  // a reset leaves old counts here
        lkai_e14_nrows++;
    }
    lkai_e14_rows[r].n++;
    lkai_e14_rows[r].ns += ns;
    lkai_e14_rows[r].bytes += (int64_t) ggml_nbytes(dst);
}
extern "C" __attribute__((visibility("default"))) void lkai_e14_probe_reset(void) { lkai_e14_nrows = 0; }
extern "C" __attribute__((visibility("default"))) int lkai_e14_probe_dump(const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) return -1;
    for (int r = 0; r < lkai_e14_nrows; r++) {
        fprintf(f, "%s\t%lld\t%lld\t%lld\n", lkai_e14_rows[r].key, (long long) lkai_e14_rows[r].n,
                (long long) lkai_e14_rows[r].ns, (long long) lkai_e14_rows[r].bytes);
    }
    return fclose(f);
}
"""

OLD = """        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } break;"""
NEW = """        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                const int lkai_p = params->ith == 0 && lkai_e14_enabled();
                const int64_t lkai_t0 = lkai_p ? lkai_e14_now() : 0;
                ggml_compute_forward_concat_f32(params, dst);
                if (lkai_p) {
                    lkai_e14_add(dst, lkai_e14_now() - lkai_t0);
                }
            } break;"""

ANCHOR = "// ggml_compute_forward_concat\n"


def patch(root: Path) -> None:
    ops = root / "ggml/src/ggml-cpu/ops.cpp"
    s = ops.read_text()
    assert "lkai E14 offline concat probe" not in s, "already patched"
    assert s.count(OLD) == 1, "concat dispatch anchor not found"
    assert s.count(ANCHOR) == 1, "concat section anchor not found"
    s = s.replace(OLD, NEW).replace(ANCHOR, HEADER + "\n" + ANCHOR)
    ops.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    patch(ap.parse_args().root)
    print("patched concat_probe", ap.parse_args().root)


if __name__ == "__main__":
    main()
