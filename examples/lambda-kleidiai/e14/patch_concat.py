"""E14's single change for llama.cpp f74e2fa6a: split the f32 CONCAT kernel's rows across every thread.

  python patch_concat.py /path/to/llama.cpp          (after llama/patch_llama.py)

ggml_compute_forward_concat_f32 hands out work by dst's dim 2 only (`for i2 = ith; i2 < ne2; i2 += nth`). In qwen35's
Gated DeltaNet layers the only CONCAT is build_conv_state's `conv_input`:

    concat(conv_states [d_conv-1, conv_channels, n_seqs], transpose(qkv_mixed) [n_tokens, conv_channels, n_seqs], 0)

so ne2 is the ubatch's sequence count. A one-sequence ubatch (the state prefix, and every single-question request)
runs the whole copy on thread 0 while the other threads wait at the next barrier. The patch walks dst's rows
(i1, i2, i3) flattened, one contiguous block per thread. The element loop is unchanged: each dst element is still
written once from the same source element, so the output is bit-identical. Only which thread copies which row
changes. The f16, i8 and generic paths are untouched.
"""

import argparse
from pathlib import Path

OLD = """    const float * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const float *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const float *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    float * y = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}
"""

NEW = """    const float * x;

    // strands-decider concat patch (examples/lambda-kleidiai/e14): one contiguous block of dst rows (i1, i2, i3) per
    // thread instead of dim 2 only, so a one-sequence ubatch no longer copies on one thread. Same element loop.
    const int64_t nr  = ne1*ne2*ne3;
    const int64_t dr  = (nr + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = ir0 + dr < nr ? ir0 + dr : nr;
    for (int64_t ir = ir0; ir < ir1; ir++) {
        const int64_t i3 = ir/(ne1*ne2);
        const int64_t i2 = (ir - i3*ne1*ne2)/ne1;
        const int64_t i1 = ir - i3*ne1*ne2 - i2*ne1;
        for (int64_t i0 = 0; i0 < ne0; i0++) {
            if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = (const float *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
            } else {
                x = (const float *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
            }

            float * y = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

            *y = *x;
        }
    }
}
"""

ANCHOR = "static void ggml_compute_forward_concat_f32("


def patch(root: Path) -> None:
    ops = root / "ggml/src/ggml-cpu/ops.cpp"
    s = ops.read_text()
    assert "strands-decider concat patch" not in s, "already patched"
    start = s.index(ANCHOR)
    end = s.index("void ggml_compute_forward_concat(", start)
    body = s[start:end]
    assert body.count(OLD) == 1, "concat_f32 loop anchor not found"
    s = s[:start] + body.replace(OLD, NEW) + s[end:]
    ops.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    patch(ap.parse_args().root)
    print("patched concat_rows", ap.parse_args().root)


if __name__ == "__main__":
    main()
