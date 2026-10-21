"""E9 loader patch: adopt KleidiAI-packed Q8_0 weights from the GGUF instead of repacking them at load.

  python patch_e9.py /path/to/llama.cpp      (after llama/patch_llama.py, on llama.cpp f74e2fa6a)

The pinned loader sends every Q8_0 matmul weight to the CPU_KLEIDIAI buffer type, whose set_tensor ("repack")
dequantizes the Q8_0 blocks, requantizes each row to int8 with one scale, and packs the result in the selected
kernel's layout (kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon). An E9 GGUF (e9/pack_e9.py) stores that packed result in
place of each per-layer Q8_0 tensor's bytes. The packed form is smaller than the Q8_0 bytes for every tensor in
this model, so the tensor infos, offsets and file size are unchanged; each such tensor ends with a 64-byte trailer:

  char magic[8] = "E9KAIPK1"; uint32 nr, kr, sr, reserved; uint64 n, k, payload_bytes; 16 zero bytes

The payload (the first payload_bytes) is exactly what repack would leave in tensor->data: the KleidiAI weight
header (magic KLAI, version 1, one slot at offset 64) and the packed slot.

kleidiai.cpp, in repack():
  - payload layout equal to the selected kernel's (nr, kr, sr): copy the payload, no repack (the adopt path);
  - another layout (a CPU that selects the dot-product kernel, kr=4, for a file packed for i8mm, kr=8): unpack the
    int8 rows and row scales from the payload and pack them with the selected kernel. The dequantize/requantize
    pass is skipped; the values are the ones a runtime repack would compute, so the packed bytes are the same;
  - anything else (no trailer, a header that does not match): the unchanged runtime repack.
  - E9_DUMP_DIR=<dir> (diagnostics only): write each Q8_0 tensor's payload as packed in memory to <dir>, so a test
    can hash it. E9_DIAG_NO_I8MM=1 (diagnostics only) hides i8mm from KleidiAI's kernel choice so one host can run
    the dot-product layout. Both are unset in every deployed function.
  - one stderr line per path and process: "E9KAI adopt|convert|repack nr=<nr> kr=<kr>".
llama-model-loader.cpp, in load_all_data(): a mapped Q8_0 tensor that carries an E9 trailer must sit in a
CPU_KLEIDIAI buffer, and a GGUF with the key strands_decider.e9.kleidiai_prepacked must be loaded with mmap.
Otherwise its packed bytes would be read as Q8_0 blocks, so the load fails instead. A CPU without KleidiAI's Q8_0
kernels cannot use the file: the original Q8_0 blocks are not recoverable from the per-row requantization.

token_embd.weight stays Q8_0: GET_ROWS reads it natively, and its tied output copy is still repacked at load (E7's
variable, not E9's).
"""

import argparse
from pathlib import Path

E9_TRAILER = r"""
// strands-decider E9 patch: KleidiAI-packed Q8_0 weights stored in the GGUF (e9/pack_e9.py).
struct e9_trailer {
    char     magic[8];
    uint32_t nr, kr, sr, reserved;
    uint64_t n, k, payload_bytes, zero[2];
};
static_assert(sizeof(e9_trailer) == 64, "E9 trailer is 64 bytes");

// One stderr line per load path and process (E9KAI <path> nr= kr=), so a run's logs show which path each host took.
static void e9_note(int path, size_t nr, size_t kr) {
    static std::atomic<int> seen{0};
    static const char * names[] = {"adopt", "convert", "repack"};
    if (!(seen.fetch_or(1 << path) & (1 << path))) {
        fprintf(stderr, "E9KAI %s nr=%zu kr=%zu\n", names[path], nr, kr);
    }
}

static const e9_trailer * e9_find_trailer(const void * data, size_t data_size, size_t n, size_t k) {
    if (data_size < sizeof(e9_trailer)) {
        return nullptr;
    }
    const auto * t = reinterpret_cast<const e9_trailer *>(static_cast<const uint8_t *>(data) + data_size - sizeof(e9_trailer));
    if (memcmp(t->magic, "E9KAIPK1", 8) != 0) {
        return nullptr;
    }
    // A trailer that does not describe this tensor would leave packed bytes to be read as Q8_0 blocks: refuse it.
    GGML_ASSERT(t->n == n && t->k == k && t->sr == 1 && t->nr > 0 && t->kr > 0 && ((k + 31) / 32 * 32) % t->kr == 0 &&
                t->payload_bytes <= data_size - sizeof(e9_trailer) && "E9 trailer does not match its tensor");
    return t;
}

static void e9_dump(const struct ggml_tensor * tensor, size_t bytes) {
    static const char * dir = getenv("E9_DUMP_DIR");
    if (!dir || !*dir) {
        return;
    }
    std::string path = std::string(dir) + "/" + tensor->name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f && "E9_DUMP_DIR not writable");
    GGML_ASSERT(fwrite(tensor->data, 1, bytes, f) == bytes);
    fclose(f);
}

class tensor_traits : public ggml::cpu::tensor_traits {
"""

ADOPT = r"""        std::vector<float> row_f32;

        // strands-decider E9 patch: adopt or convert a payload stored by e9/pack_e9.py.
        bool e9_have_q = false;
        const e9_trailer * t9 = tensor->type == GGML_TYPE_Q8_0 ? e9_find_trailer(data, data_size, n, k) : nullptr;
        if (t9) {
            GGML_ASSERT(slot_total > 0 && "E9-prepacked tensor but no KleidiAI Q8_0 kernel");
            const kleidiai_weight_header * h9 = kleidiai_weight_header_from_ptr(data);
            const size_t k_int = (k + 31) / 32 * 32;
            const size_t stride = t9->nr * (k_int + 3 * sizeof(float));
            const size_t have = (n + t9->nr - 1) / t9->nr * stride;  // kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon
            // A trailer whose payload is not one well-formed slot would be read as Q8_0 blocks below: refuse it.
            GGML_ASSERT(kleidiai_is_weight_header_valid(h9) && h9->slot_count == 1 &&
                        h9->offsets[0] == align_up(sizeof(kleidiai_weight_header), GGML_KLEIDIAI_PACK_ALIGN) &&
                        h9->sizes[0] == have && h9->offsets[0] + have == t9->payload_bytes &&
                        "E9 trailer present but its payload is malformed");
            {
                const kernel_info * kern = &kernel_chain[0]->gemm;
                if (slot_total == 1 && kernel_chain[0]->rhs_type == GGML_TYPE_Q8_0 &&
                    t9->nr == kern->get_nr() && t9->kr == kern->get_kr() && t9->sr == kern->get_sr()) {
                    memcpy(tensor->data, data, t9->payload_bytes);
                    e9_dump(tensor, t9->payload_bytes);
                    e9_note(0, t9->nr, t9->kr);
                    return 0;
                }
                {
                    // Another kernel's layout (or a fallback chain): recover the int8 rows and row scales, then pack
                    // them below exactly as the runtime repack would after its requantization.
                    const size_t nr9 = t9->nr, kr9 = t9->kr;
                    const uint8_t * src = static_cast<const uint8_t *>(data) + h9->offsets[0];
                    qdata.assign(n * k, 0);
                    scales.assign(n, 0.0f);
                    for (size_t g = 0; g < (n + nr9 - 1) / nr9; ++g) {
                        const int8_t * blocks = reinterpret_cast<const int8_t *>(src + g * stride);
                        for (size_t b = 0; b < nr9 * k_int / kr9; ++b) {
                            const size_t row = g * nr9 + b % nr9;
                            const size_t k0  = (b / nr9) * kr9;
                            if (row >= n) {
                                continue;
                            }
                            for (size_t i = 0; i < kr9 && k0 + i < k; ++i) {
                                qdata[row * k + k0 + i] = blocks[b * kr9 + i];
                            }
                        }
                        const float * sc = reinterpret_cast<const float *>(src + g * stride + nr9 * k_int + nr9 * sizeof(int32_t));
                        for (size_t i = 0; i < nr9 && g * nr9 + i < n; ++i) {
                            scales[g * nr9 + i] = sc[i];
                        }
                    }
                    e9_have_q = true;
                    e9_note(1, kernel_chain[0]->gemm.get_nr(), kernel_chain[0]->gemm.get_kr());
                }
            }
        }

        if (want_q8 && slot_total > 0 && !e9_have_q) {
"""

DUMP = r"""            memcpy(tensor->data, data, data_size);
        }

        if (tensor->type == GGML_TYPE_Q8_0 && header->slot_count == 1) {  // strands-decider E9 patch (diagnostics)
            e9_dump(tensor, header->offsets[0] + header->sizes[0]);
            if (!e9_have_q) {
                e9_note(2, kernel_chain[0]->gemm.get_nr(), kernel_chain[0]->gemm.get_kr());
            }
        }

        return 0;
    }
};
"""

LOADER = r"""            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            // strands-decider E9 patch: a tensor carrying an E9 trailer holds KleidiAI-packed bytes, not Q8_0 blocks.
            // Only CPU_KLEIDIAI's set_tensor can read it; anywhere else it would be used as Q8_0 silently.
            if (cur->type == GGML_TYPE_Q8_0 && n_size >= 64 && memcmp(data + n_size - 64, "E9KAIPK1", 8) == 0 &&
                (cur->buffer == nullptr ||
                 strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)), "CPU_KLEIDIAI") != 0)) {
                throw std::runtime_error(format("tensor '%s' is KleidiAI-prepacked (E9) but is not in a CPU_KLEIDIAI buffer",
                                                ggml_get_name(cur)));
            }
"""

NO_MMAP = r"""        const bool from_mapping = use_mmap || lazy.has(cur);

        // strands-decider E9 patch: the prepacked-tensor check above needs the mapping; refuse to read E9 files without it.
        if (!from_mapping && gguf_find_key(metadata, "strands_decider.e9.kleidiai_prepacked") >= 0) {
            throw std::runtime_error("an E9 KleidiAI-prepacked GGUF needs mmap loading");
        }
"""


NO_I8MM = r"""        // strands-decider E9 patch (diagnostics only): E9_DIAG_NO_I8MM=1 hides i8mm from KleidiAI's kernel choice, so
        // one host can exercise the dot-product layout (kr=4) and E9's convert path. Unset in every deployed function.
        if (getenv("E9_DIAG_NO_I8MM")) {
            ctx.features = static_cast<cpu_feature>(ctx.features & ~CPU_FEATURE_I8MM);
        }

        // Kernel selection
"""


def sub(path: Path, old: str, new: str, what: str) -> None:
    s = path.read_text()
    assert s.count(old) == 1, f"{what}: anchor not found exactly once in {path}"
    path.write_text(s.replace(old, new))


def patch(root: Path) -> None:
    k = root / "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"
    sub(
        k,
        "class tensor_traits : public ggml::cpu::tensor_traits {\n",
        E9_TRAILER.lstrip("\n"),
        "trailer",
    )
    sub(
        k,
        "        std::vector<float> row_f32;\n\n        if (want_q8 && slot_total > 0) {\n",
        ADOPT,
        "adopt",
    )
    sub(
        k,
        "            memcpy(tensor->data, data, data_size);\n        }\n\n        return 0;\n    }\n};\n",
        DUMP,
        "dump",
    )
    sub(k, "        // Kernel selection\n", NO_I8MM, "no-i8mm")
    lm = root / "src/llama-model-loader.cpp"
    sub(
        lm,
        "            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;\n",
        LOADER,
        "loader",
    )
    sub(lm, "        const bool from_mapping = use_mmap || lazy.has(cur);\n", NO_MMAP, "no-mmap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    patch(ap.parse_args().root)
    print("E9 patch applied")
