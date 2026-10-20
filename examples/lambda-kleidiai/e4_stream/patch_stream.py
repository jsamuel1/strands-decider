"""E4 patch for llama.cpp f74e2fa6a (applied after llama/patch_llama.py): a dormant hook that lets a model load start
while its GGUF is still being written into the memfd it is mapped from.

  python patch_stream.py /path/to/llama.cpp

Adds to llama.h:

    typedef int      (*llama_load_wait_t) (void * ud, uint64_t offs, uint64_t size);
    typedef uint64_t (*llama_load_order_t)(void * ud, uint64_t offs, uint64_t size);
    void llama_set_load_stream(llama_load_wait_t wait, llama_load_order_t order, void * ud);

With no hook set (the default, and the `base` arm) llama_model_loader::load_all_data is unchanged: one null pointer
test per tensor. With a hook set:

- `wait(ud, offs, size)` is called before the loader READS a tensor's bytes [offs, offs + size) of the file: before
  KleidiAI's repack (ggml_backend_tensor_set from the mapping), before a check_tensors validation, and before every
  read on the no-mmap path. A tensor the mmap path only aliases (a CPU-buffer tensor such as token_embd, whose pages
  are read at inference) is not waited for; the caller must wait for the whole file before the first inference.
  A non-zero return aborts the load (llama_model_load_from_file returns NULL).
- `order(ud, offs, size)`, if set, gives each tensor a sort key; the tensors of each buffer context are loaded in
  ascending key order (stable), instead of creation order. The loader's work per tensor is the same in any order:
  tensor storage was allocated before load_all_data, and the aliased address of an mmap tensor is its file offset.

- At the end of the last load_all_data call the loader waits for the whole file, so llama_model_load_from_file
  returns only once every byte has arrived: the aliased tensors are then complete before any inference.

Adds to gguf.h (ggml-base, a separate library):

    typedef int (*gguf_read_wait_t)(void * ud, uint64_t offs, uint64_t size);
    void gguf_set_read_wait(gguf_read_wait_t wait, void * ud);

called before header reads of gguf_init_from_file (the KV metadata, including a 248k-token vocabulary read string by
string, and the tensor index, all parsed before any tensor). The parser reads sequentially from the start, so the
patch keeps a watermark: when a read ends past it, the hook is asked for [0, end) with end rounded up to 4 MiB plus
a 1 MiB margin for stdio read-ahead (the hook clips to the file size), and the watermark moves there. An 11 MB
header thus costs about four calls, not one per read.
"""

import argparse
from pathlib import Path

HDR_ANCHOR = "    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);\n"
HDR_ADD = """
    // strands-decider E4 stream patch: start a model load while the GGUF is still arriving (examples/lambda-kleidiai/
    // e4_stream/patch_stream.py). wait() is called before each tensor's bytes are read and returns 0 once they are
    // present (non-zero aborts the load); order(), if set, sorts each buffer context's tensors. NULL wait: no-op.
    typedef int      (*llama_load_wait_t) (void * ud, uint64_t offs, uint64_t size);
    typedef uint64_t (*llama_load_order_t)(void * ud, uint64_t offs, uint64_t size);
    LLAMA_API void llama_set_load_stream(llama_load_wait_t wait, llama_load_order_t order, void * ud);
"""

GLOBALS_ANCHOR = '#include "llama-model-loader.h"\n'
GLOBALS_ADD = """
// strands-decider E4 stream patch: the load-stream hook (llama.h, llama_set_load_stream).
static llama_load_wait_t  g_load_wait  = nullptr;
static llama_load_order_t g_load_order = nullptr;
static void *             g_load_ud    = nullptr;

void llama_set_load_stream(llama_load_wait_t wait, llama_load_order_t order, void * ud) {
    g_load_wait  = wait;
    g_load_order = order;
    g_load_ud    = ud;
}

static void llama_stream_wait(const char * name, size_t offs, size_t size) {
    if (g_load_wait && g_load_wait(g_load_ud, (uint64_t) offs, (uint64_t) size) != 0) {
        throw std::runtime_error(std::string("load stream: bytes of tensor ") + name + " never arrived");
    }
}
"""

ORDER_ANCHOR = """    // without mmap, tensors in non-host buffers are staged through a temporary buffer sized like the tensor
    // load them biggest-first so the largest staging buffer is allocated while the fewest weights are resident
    if (!use_mmap) {"""
ORDER_ADD = """    // strands-decider E4 stream patch: load in the order the hook's bytes are expected to arrive
    if (g_load_wait && g_load_order) {
        std::vector<std::pair<uint64_t, ggml_tensor *>> keyed;
        keyed.reserve(tensors.size());
        for (ggml_tensor * t : tensors) {
            const auto * w = get_weight(ggml_get_name(t));
            keyed.emplace_back(w ? g_load_order(g_load_ud, w->offs, ggml_nbytes(t)) : 0, t);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (size_t i = 0; i < keyed.size(); ++i) {
            tensors[i] = keyed[i].second;
        }
    }

"""

MMAP_ANCHOR = """            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {"""
MMAP_ADD = """            // strands-decider E4 stream patch: wait for the bytes this branch reads (validation or repack/copy);
            // a tensor that is only aliased into the mapping is read at inference, after the caller's final wait
            if (check_tensors || !(buf_mmap && cur->data == nullptr)) {
                llama_stream_wait(ggml_get_name(cur), weight->offs, n_size);
            }

"""

READ_ANCHOR = """        } else {
            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {"""
READ_NEW = """        } else {
            const auto & file = files.at(weight->idx);
            llama_stream_wait(ggml_get_name(cur), weight->offs, n_size);  // strands-decider E4 stream patch

            if (ggml_backend_buffer_is_host(cur->buffer)) {"""

FINAL_ANCHOR = """    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata"""
FINAL_NEW = """    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // strands-decider E4 stream patch: every byte (aliased tensors included) before the load returns
        for (const auto & file : files) {
            llama_stream_wait("(whole file)", 0, file->size());
        }
        // unmap offloaded tensors and metadata"""

GGUF_H_ANCHOR = "    GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);\n"
GGUF_H_ADD = """    // strands-decider E4 stream patch: called before each header read of gguf_init_from_file; returns 0 once
    // [offs, offs + size) of the file is present (non-zero fails the read). NULL: no-op.
    typedef int (*gguf_read_wait_t)(void * ud, uint64_t offs, uint64_t size);
    GGML_API void gguf_set_read_wait(gguf_read_wait_t wait, void * ud);
"""

GGUF_C_ANCHOR = """static size_t gguf_file_reader_callback(void * userdata, void * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);
"""
GGUF_C_NEW = """// strands-decider E4 stream patch: the header read-wait hook (gguf.h, gguf_set_read_wait). Bytes [0, g_gguf_ready)
// are known present, so only a read that ends past the watermark calls the hook.
static gguf_read_wait_t g_gguf_read_wait    = nullptr;
static void *           g_gguf_read_wait_ud = nullptr;
static uint64_t         g_gguf_ready        = 0;

void gguf_set_read_wait(gguf_read_wait_t wait, void * ud) {
    g_gguf_read_wait    = wait;
    g_gguf_read_wait_ud = ud;
    g_gguf_ready        = 0;
}

static size_t gguf_file_reader_callback(void * userdata, void * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);
    if (g_gguf_read_wait && offset + len > g_gguf_ready) {
        // stdio reads ahead into its buffer: ask for [0, end) with a 1 MiB margin, end rounded up to 4 MiB
        const uint64_t step = 4u << 20;
        const uint64_t end  = (offset + len + (1u << 20) + step - 1) / step * step;
        if (g_gguf_read_wait(g_gguf_read_wait_ud, 0, end) != 0) {
            return 0;
        }
        g_gguf_ready = end;
    }
"""

MARK = "strands-decider E4 stream patch"


def patch(root: Path) -> bool:
    """Apply the patch; returns False if it was already applied. Asserts every anchor is unique."""
    files = {
        "h": root / "include/llama.h",
        "s": root / "src/llama-model-loader.cpp",
        "gh": root / "ggml/include/gguf.h",
        "gc": root / "ggml/src/gguf.cpp",
    }
    t = {k: p.read_text() for k, p in files.items()}
    marked = [k for k, v in t.items() if MARK in v]
    if marked:
        assert len(marked) == len(t), f"half-applied E4 stream patch ({marked})"
        return False
    edits = (  # (file, anchor, replacement)
        ("h", HDR_ANCHOR, HDR_ANCHOR + HDR_ADD),
        ("s", GLOBALS_ANCHOR, GLOBALS_ANCHOR + GLOBALS_ADD),
        ("s", ORDER_ANCHOR, ORDER_ADD + ORDER_ANCHOR),
        ("s", MMAP_ANCHOR, MMAP_ADD + MMAP_ANCHOR),
        ("s", READ_ANCHOR, READ_NEW),
        ("s", FINAL_ANCHOR, FINAL_NEW),
        ("gh", GGUF_H_ANCHOR, GGUF_H_ANCHOR + GGUF_H_ADD),
        ("gc", GGUF_C_ANCHOR, GGUF_C_NEW),
    )
    for k, anchor, _ in edits:
        assert t[k].count(anchor) == 1, (
            f"{files[k].name}: anchor not found exactly once: {anchor[:60]!r}"
        )
    for k, anchor, new in edits:
        t[k] = t[k].replace(anchor, new)
    for k, p in files.items():
        p.write_text(t[k])
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    print("E4 stream patch", "applied" if patch(a.root) else "already applied")


if __name__ == "__main__":
    main()
