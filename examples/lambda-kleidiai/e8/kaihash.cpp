// E8 offline identity check (NOT Lambda): load a GGUF with the libraries of one zip, then hash every weight as the
// loaded model holds it, so the serial and the multi-threaded KleidiAI repack can be compared byte for byte.
//
//   kaihash <ggml-dir> <model.gguf>    (prints one line per tensor, then a summary line)
//
// It reads the loaded buffers back through llama_internal_get_tensor_map, the same map llama.cpp's own tests use,
// so it needs no change to the libraries it checks: the same binary hashes P1's unpatched libraries and E8's.
// The load uses decshim's model parameters (backends from <ggml-dir>, n_gpu_layers 0, mmap), and the time of
// llama_model_load_from_file is reported as model_load_s.
//
// For a CPU_KLEIDIAI tensor, repack writes a header (magic, version, slot_count, then offsets[] and sizes[] for each
// slot) and the packed slots, at 64-byte-aligned offsets. The bytes between them, and the header entries past
// slot_count, are never written, so the hash covers exactly what repack writes: magic, version, slot_count, the used
// offsets and sizes, and each slot. A tensor whose header magic is not "KLAI" (repack fell back to a raw copy) and
// every tensor in another buffer type is hashed over ggml_nbytes. The hash is FNV-1a 64.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

namespace {
constexpr uint32_t KLAI_MAGIC = 0x4b4c4149;  // GGML_KLEIDIAI_PACK_MAGIC
constexpr int MAX_SLOTS = 2;                 // GGML_KLEIDIAI_MAX_KERNEL_SLOTS

struct header {  // kleidiai_weight_header in kleidiai.cpp (f74e2fa6a)
    uint32_t magic;
    uint16_t version;
    uint16_t slot_count;
    uint64_t offsets[MAX_SLOTS];
    uint64_t sizes[MAX_SLOTS];
};

struct fnv {
    uint64_t h = 1469598103934665603ULL;
    size_t n = 0;
    void mix(const void * p, size_t len) {
        const auto * b = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < len; ++i) {
            h = (h ^ b[i]) * 1099511628211ULL;
        }
        n += len;
    }
};

void quiet(ggml_log_level, const char *, void *) {}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <ggml-dir> <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_log_set(quiet, nullptr);
    ggml_backend_load_all_from_path(argv[1]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    const auto t0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    fnv all;
    int n_tensors = 0, n_packed = 0;
    for (const auto & [name, t] : llama_internal_get_tensor_map(model)) {
        const char * buft = t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "none";
        fnv f;
        int slots = -1;
        if (t->data && !strcmp(buft, "CPU_KLEIDIAI")) {
            header hd;
            memcpy(&hd, t->data, sizeof(hd));
            if (hd.magic == KLAI_MAGIC && hd.slot_count >= 1 && hd.slot_count <= MAX_SLOTS) {
                slots = hd.slot_count;
                f.mix(&hd.magic, sizeof(hd.magic));
                f.mix(&hd.version, sizeof(hd.version));
                f.mix(&hd.slot_count, sizeof(hd.slot_count));
                f.mix(hd.offsets, sizeof(uint64_t) * slots);
                f.mix(hd.sizes, sizeof(uint64_t) * slots);
                for (int i = 0; i < slots; ++i) {
                    f.mix(static_cast<const uint8_t *>(t->data) + hd.offsets[i], hd.sizes[i]);
                }
                ++n_packed;
            }
        }
        if (slots < 0 && t->data) {
            f.mix(t->data, ggml_nbytes(t));
            slots = 0;
        }
        ++n_tensors;
        all.mix(name.data(), name.size());
        all.mix(&f.h, sizeof(f.h));
        printf("T %s %s %s %d %zu %016" PRIx64 "\n", name.c_str(), ggml_type_name(t->type), buft, slots, f.n, f.h);
    }
    printf("S tensors=%d packed=%d model_load_s=%.4f all=%016" PRIx64 "\n", n_tensors, n_packed, load_s, all.h);
    llama_model_free(model);
    return 0;
}
