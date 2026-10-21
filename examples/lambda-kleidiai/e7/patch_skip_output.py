"""E7 patch for llama.cpp f74e2fa6a (applied after llama/patch_llama.py): a dormant switch that keeps qwen35 from
creating the tied output head, so KleidiAI never repacks it.

  python patch_skip_output.py /path/to/llama.cpp

The deployed GGUF has no `output.weight`. qwen35's load_arch_tensors then creates `output` as a TENSOR_DUPLICATED
view of `token_embd.weight`. The loader assigns that duplicate to the output layer's buffer list, so it gets its own
tensor in the CPU_KLEIDIAI buffer and is repacked at load (about 540 MB of Q8_0), separately from token_embd, which
stays in the CPU buffer for GET_ROWS. The decider never reads it: decshim always sets cparams.embeddings with no
pooling, and P1's prefill patch returns from the qwen35 graph before the LM head (`build_lora_mm(model.output, ...)`).
The only other reader is the MTP draft graph, which needs load_mtp (false by default, never set by decshim).

With LKAI_SKIP_TIED_OUTPUT=1 in the environment at load time, the duplicate is not created and `model.output` stays
NULL. Unset, or any other value, the code path is upstream's plus one getenv per model load. The graph keeps P1's
behaviour in embeddings mode. If anything ever builds the LM head with the head skipped (embeddings off, or an MTP
graph), qwen35 aborts with a message instead of dereferencing NULL, so a misuse cannot silently produce output.

Adds two exported queries, so the function can record that the switch took effect (read after the load, outside
P1's phase timers):

    LLAMA_API int32_t  llama_model_lkai_tied_output_skipped(const struct llama_model * model);  // 1 or 0
    LLAMA_API uint64_t llama_model_lkai_buft_bytes(const struct llama_model * model, const char * buft_name);

The first reads the flag the loader set. The second sums the model's weight buffers whose buffer type has that name
(for example "CPU_KLEIDIAI" or "CPU"), from llama_model::memory_breakdown(), so the size of the repacked copy that
the switch removes is measured rather than assumed.
"""

import argparse
from pathlib import Path

LOAD_OLD = """    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }
"""
LOAD_NEW = """    // if output is NULL, init from the input tok embed
    // strands-decider E7 patch: LKAI_SKIP_TIED_OUTPUT=1 leaves a tied head uncreated (no CPU_KLEIDIAI repack of it).
    // The prefill graph never reads it in embeddings mode; graph() aborts if anything else tries to.
    if (output == NULL) {
        const char * skip = getenv("LKAI_SKIP_TIED_OUTPUT");
        if (skip && strcmp(skip, "1") == 0) {
            lkai_tied_output_skipped = true;
        } else {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
        }
    }
"""

GRAPH_OLD = """    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
"""
GRAPH_NEW = """    // LM head
    // strands-decider E7 patch: a head skipped at load (LKAI_SKIP_TIED_OUTPUT=1) cannot serve logits.
    GGML_ASSERT(model.output && "LKAI_SKIP_TIED_OUTPUT=1 skipped the tied LM head: logits need it (embeddings mode only)");
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
"""

MTP_OLD = """    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;"""
MTP_NEW = """    // strands-decider E7 patch: the MTP fallback head is the tied output too; the GGML_ASSERT below names it NULL.
    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;"""

INCLUDES_OLD = '#include "models.h"\n#include "llama-memory-recurrent.h"\n'
INCLUDES_NEW = INCLUDES_OLD + "\n#include <cstdlib>\n#include <cstring>\n"

# The flag lives on llama_model (llama-model.h), so the C API can read it without knowing the arch class.
MODEL_H_OLD = "    struct ggml_tensor * output_s    = nullptr;\n"
MODEL_H_NEW = MODEL_H_OLD + (
    "    bool lkai_tied_output_skipped = false;  // strands-decider E7 patch (LKAI_SKIP_TIED_OUTPUT=1)\n"
)

API_H_OLD = "    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);\n"
API_H_NEW = API_H_OLD + (
    "\n    // strands-decider E7 patch: 1 if LKAI_SKIP_TIED_OUTPUT=1 kept the tied LM head from being created, else 0\n"
    "    LLAMA_API int32_t llama_model_lkai_tied_output_skipped(const struct llama_model * model);\n"
    "    // strands-decider E7 patch: bytes of the model's weight buffers whose buffer type is named buft_name\n"
    "    LLAMA_API uint64_t llama_model_lkai_buft_bytes(const struct llama_model * model, const char * buft_name);\n"
)

API_C_OLD = """uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}
"""
API_C_NEW = (
    API_C_OLD
    + """
int32_t llama_model_lkai_tied_output_skipped(const llama_model * model) {
    return model->lkai_tied_output_skipped ? 1 : 0;
}

uint64_t llama_model_lkai_buft_bytes(const llama_model * model, const char * buft_name) {
    uint64_t n = 0;
    for (const auto & [buft, size] : model->memory_breakdown()) {
        if (buft && buft_name && strcmp(ggml_backend_buft_name(buft), buft_name) == 0) {
            n += size;
        }
    }
    return n;
}
"""
)


def _replace(path: Path, old: str, new: str, what: str) -> None:
    s = path.read_text()
    assert s.count(old) == 1, f"{path.name}: {what} anchor not found exactly once"
    path.write_text(s.replace(old, new))


def patch(root: Path) -> None:
    q = root / "src/models/qwen35.cpp"
    _replace(q, INCLUDES_OLD, INCLUDES_NEW, "includes")
    _replace(q, LOAD_OLD, LOAD_NEW, "tied output")
    _replace(q, GRAPH_OLD, GRAPH_NEW, "LM head")
    _replace(q, MTP_OLD, MTP_NEW, "MTP head")
    _replace(root / "src/llama-model.h", MODEL_H_OLD, MODEL_H_NEW, "llama_model output_s")
    _replace(root / "include/llama.h", API_H_OLD, API_H_NEW, "llama_model_n_params decl")
    _replace(root / "src/llama-model.cpp", API_C_OLD, API_C_NEW, "llama_model_n_params def")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    a = ap.parse_args()
    patch(Path(a.root))
    print("patched", a.root, "e7 skip-tied-output")
