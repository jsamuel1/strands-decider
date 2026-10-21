"""E15 Phase 2 candidates for llama.cpp f74e2fa6a's CPU backend, each behind its own environment switch.

  python patch_candidates.py /path/to/llama.cpp      (after llama/patch_llama.py; with or without patch_nodeprof.py)

Offline probing builds both into one set of libggml-cpu-*.so so that arms can be interleaved on one host. The
deployed experiment ships exactly one of them (see README).

LKAI_E15_A=1  llamafile tinyBLAS F32 sgemm for F32 x F32 MUL_MAT in the SVE / i8mm variants.
              Upstream's ggml-cpu.c undefines GGML_USE_LLAMAFILE whenever __ARM_FEATURE_SVE or
              __ARM_FEATURE_MATMUL_INT8 is set, so armv8.6_1 (part 0xd40) runs attention's F32 KQV through the per-row
              ggml_vec_dot_f32 loop although sgemm.cpp is compiled in. With the switch on, those variants call
              llamafile_sgemm for src0 F32 only; every quantised type keeps its own path. Non-SVE variants are
              unchanged (they already use llamafile for every type). Floating-point summation order changes.

LKAI_E15_B=1  Row-blocked GATED_DELTA_NET for the baseline's single-pass case (scalar gate, K == 1, S_v <= 512).
              patch_llama.py's `gdn` kernel walks tokens outermost and the whole S_v x S_v state per token. Each
              state row j evolves independently (its next-token S^T k term is row j's own dot product), so this
              runs each (sequence, head, block of LKAI_GDN_ROWS rows) over all tokens with the block resident in L1,
              and gives threads H * n_seqs * S_v / rows work items instead of H * n_seqs. Every element sees the
              same operations in the same order as the baseline kernel, so outputs should be bit-identical.
"""

import argparse
from pathlib import Path

# ---- A: llamafile F32 sgemm in SVE / i8mm variants -------------------------------------------------------------
A_UNDEF_OLD = """#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif
"""
A_UNDEF_NEW = """#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
// lkai E15 candidate A (examples/lambda-kleidiai/e15/patch_candidates.py): upstream undefines GGML_USE_LLAMAFILE here.
// Keep it, but let these variants call llamafile_sgemm only for F32 src0, and only with LKAI_E15_A=1.
#ifdef GGML_USE_LLAMAFILE
#define LKAI_SGEMM_SVE_GATE 1
#endif
#endif
#ifdef LKAI_SGEMM_SVE_GATE
#include <stdlib.h>
static int lkai_e15_a = -1;
__attribute__((visibility("default"))) void lkai_e15_set_a(int on) { lkai_e15_a = on ? 1 : 0; }
static inline int lkai_sgemm_ok(const struct ggml_tensor * src0) {
    if (lkai_e15_a < 0) {
        const char * e = getenv("LKAI_E15_A");
        lkai_e15_a = (e && e[0] == '1') ? 1 : 0;
    }
    return lkai_e15_a && src0->type == GGML_TYPE_F32;
}
#else
#define lkai_sgemm_ok(src0) 1
#endif
"""
A_GEMM1_OLD = "    if (!params->use_ref && src1_cont) {\n"
A_GEMM1_NEW = "    if (!params->use_ref && src1_cont && lkai_sgemm_ok(src0)) {\n"
A_GEMM2_OLD = "    if (!params->use_ref && src1->type != vec_dot_type) {\n"
A_GEMM2_NEW = "    if (!params->use_ref && src1->type != vec_dot_type && lkai_sgemm_ok(src0)) {\n"

# ---- B: row-blocked gated delta net ----------------------------------------------------------------------------
B_FN_ANCHOR = "static void ggml_compute_forward_gated_delta_net_f32(\n"
B_FN = r"""// lkai E15 candidate B (examples/lambda-kleidiai/e15/patch_candidates.py): row-blocked single-pass GDN.
// Work item ir = ((iv3 * H) + iv1) * nblk + blk. Same per-element arithmetic as patch_llama.py's gdn kernel.
static int lkai_e15_b = -1;
static int64_t lkai_gdn_rows = 0;

// Runtime switch for paired in-process probing (e15/paired_drive.py); the environment sets the initial value.
extern "C" __attribute__((visibility("default"))) void lkai_e15_set_b(int on, int rows) {
    lkai_e15_b = on ? 1 : 0;
    lkai_gdn_rows = rows > 0 ? rows : 16;
}

static int64_t lkai_gdn_rowblock_rows(const ggml_tensor * dst) {
    if (lkai_e15_b < 0) {
        const char * e = getenv("LKAI_E15_B");
        lkai_e15_b = (e && e[0] == '1') ? 1 : 0;
        const char * r = getenv("LKAI_GDN_ROWS");
        lkai_gdn_rows = (r && atoi(r) > 0) ? atoi(r) : 16;
    }
    if (!lkai_e15_b) {
        return 0;
    }
    const int64_t S_v = dst->src[2]->ne[0];
    const bool kda = dst->src[3]->ne[0] == S_v;
    const int64_t K = ggml_get_op_params_i32(dst, 0);
    const int64_t n_tokens = dst->src[2]->ne[2];
    if (kda || K != 1 || S_v > 512 || S_v % lkai_gdn_rows != 0 || n_tokens > 4096) {
        return 0;
    }
    return lkai_gdn_rows;
}

static void lkai_gdn_rowblock(const ggml_compute_params * params, ggml_tensor * dst, int64_t R, int64_t ir0,
                              int64_t ir1) {
    GGML_UNUSED(params);
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t nblk     = S_v / R;

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbg, src_g, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float * attn_out_base  = (float *)dst->data;
    float * state_out_base = (float *)dst->data + attn_score_elems;
    const float * state_in_base = (const float *)src_state->data;
    const int64_t rq3 = nev3 / neq3;
    const int64_t rk3 = nev3 / nek3;
    const float scale = 1.0f / sqrtf((float) S_v);
    GGML_UNUSED(n_seqs);

    float egs[4096];
    float betas[4096];
    float dotk[512];
    int64_t cur_head = -1;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t blk  = ir % nblk;
        const int64_t head = ir / nblk;
        const int64_t iv1 = head % H;
        const int64_t iv3 = head / H;
        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        if (head != cur_head) {  // the per-token scalars, computed exactly as the baseline kernel does
            for (int64_t t = 0; t < n_tokens; t++) {
                betas[t] = *(const float *)((const char *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
                egs[t] = expf(*(const float *)((const char *)src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1));
            }
            cur_head = head;
        }

        const int64_t j0 = blk * R;
        float * s_out = state_out_base + (iv3 * H + iv1) * S_v * S_v;
        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out + j0 * S_v, s_in + j0 * S_v, R * S_v * sizeof(float));

        {
            const float * k0 = (const float *)((const char *)src_k->data + ik3 * nbk3 + ik1 * nbk1);
            for (int64_t j = j0; j < j0 + R; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k0, 0, 1);
                dotk[j] = sum;
            }
        }
        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;
        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            const float * k_d = (const float *)((const char *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            const float * v_d = (const float *)((const char *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);
            const float * kn  = t + 1 < n_tokens
                ? (const float *)((const char *)src_k->data + ik3 * nbk3 + (t + 1) * nbk2 + ik1 * nbk1) : nullptr;
            const float beta_val = betas[t];
            const float eg = egs[t];
            for (int64_t j = j0; j < j0 + R; ++j) {
                float * row = &s_out[j * S_v];
                const float dj = (v_d[j] - eg * dotk[j]) * beta_val;
                float acc = 0.0f, accn = 0.0f;
                int64_t i = 0;
#if defined(__ARM_NEON)
                const float32x4_t veg = vdupq_n_f32(eg), vdj = vdupq_n_f32(dj);
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, b0 = a0, b1 = a0;
                for (; i + 8 <= S_v; i += 8) {
                    const float32x4_t r0 = vfmaq_f32(vmulq_f32(veg, vld1q_f32(row + i)), vdj, vld1q_f32(k_d + i));
                    const float32x4_t r1 = vfmaq_f32(vmulq_f32(veg, vld1q_f32(row + i + 4)), vdj, vld1q_f32(k_d + i + 4));
                    vst1q_f32(row + i, r0);
                    vst1q_f32(row + i + 4, r1);
                    a0 = vfmaq_f32(a0, r0, vld1q_f32(q_d + i));
                    a1 = vfmaq_f32(a1, r1, vld1q_f32(q_d + i + 4));
                    if (kn) {
                        b0 = vfmaq_f32(b0, r0, vld1q_f32(kn + i));
                        b1 = vfmaq_f32(b1, r1, vld1q_f32(kn + i + 4));
                    }
                }
                acc = vaddvq_f32(vaddq_f32(a0, a1));
                accn = vaddvq_f32(vaddq_f32(b0, b1));
#endif
                for (; i < S_v; ++i) {
                    const float r = eg * row[i] + dj * k_d[i];
                    row[i] = r;
                    acc += r * q_d[i];
                    if (kn) {
                        accn += r * kn[i];
                    }
                }
                attn_data[t * S_v * H + j] = acc * scale;
                dotk[j] = accn;
            }
        }
    }
}

"""
B_DISPATCH_OLD = """    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];
"""
B_DISPATCH_NEW = """    ggml_tensor * V = dst->src[2];
    const int64_t lkai_rows = lkai_gdn_rowblock_rows(dst);  // 0 = baseline kernel
    int64_t nr = V->ne[1] * V->ne[3] * (lkai_rows ? V->ne[0] / lkai_rows : 1);
"""
B_CALL_OLD = "        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);\n"
B_CALL_NEW = """        if (lkai_rows) {
            lkai_gdn_rowblock(params, dst, lkai_rows, ir0, ir1);
        } else {
            ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        }
"""


def _sub(s: str, old: str, new: str, what: str) -> str:
    assert s.count(old) == 1, f"{what}: anchor not unique"
    return s.replace(old, new)


def patch_a(root: Path) -> None:
    c = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    s = c.read_text()
    assert "lkai E15 candidate A" not in s, "A already applied"
    s = _sub(s, A_UNDEF_OLD, A_UNDEF_NEW, "A undef")
    s = _sub(s, A_GEMM1_OLD, A_GEMM1_NEW, "A gemm1")
    s = _sub(s, A_GEMM2_OLD, A_GEMM2_NEW, "A gemm2")
    c.write_text(s)


def patch_b(root: Path) -> None:
    o = root / "ggml/src/ggml-cpu/ops.cpp"
    s = o.read_text()
    assert "lkai E15 candidate B" not in s, "B already applied"
    assert "strands-decider prefill patch: scalar-gate, single-snapshot heads" in s, (
        "needs patch_llama.py gdn"
    )
    s = _sub(s, B_FN_ANCHOR, B_FN + B_FN_ANCHOR, "B function")
    s = _sub(s, B_DISPATCH_OLD, B_DISPATCH_NEW, "B dispatch")
    s = _sub(s, B_CALL_OLD, B_CALL_NEW, "B call")
    o.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument(
        "--only", choices=["A", "B"], help="apply one candidate (the deployed experiment ships one)"
    )
    a = ap.parse_args()
    if a.only in (None, "A"):
        patch_a(a.root)
    if a.only in (None, "B"):
        patch_b(a.root)


if __name__ == "__main__":
    main()
