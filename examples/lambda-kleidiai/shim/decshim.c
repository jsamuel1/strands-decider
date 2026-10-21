// Prefill-only llama.cpp entry points for the Strands decider: one forward over a prompt, returning the final
// (post-norm) hidden states at the requested positions. No sampling, no generated tokens.
#include "ggml-backend.h"
#include "llama.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    struct llama_model * model;
    struct llama_context * ctx;
    struct llama_batch batch;
    int n_embd;
    int n_ctx;
    bool clear_data;
} dec_t;

static void quiet(enum ggml_log_level level, const char * text, void * ud) {
    (void) level; (void) text; (void) ud;
}

// Phase timers of the last dec_load_seq, CLOCK_MONOTONIC nanoseconds (baseline/phases.py reads them): backend load
// (dlopen of the libggml-cpu variants), llama_model_load_from_file (mmap of the GGUF and KleidiAI's weight repack),
// llama_init_from_model (context, KV cache and compute buffers), and the whole call. Four clock reads per load.
enum { DEC_PH_BACKEND, DEC_PH_MODEL, DEC_PH_CONTEXT, DEC_PH_TOTAL, DEC_PH_N };
static int64_t dec_phase_ns[DEC_PH_N];

static int64_t dec_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000 + t.tv_nsec;
}

// Copies up to n phase durations into out; returns how many the shim records.
int dec_load_phases_ns(int64_t * out, int n) {
    for (int i = 0; i < n && i < DEC_PH_N; i++) {
        out[i] = dec_phase_ns[i];
    }
    return DEC_PH_N;
}

void * dec_load_seq(const char * path, int n_threads, int n_ctx, int use_mmap, int verbose, int n_seq);

void * dec_load(const char * path, int n_threads, int n_ctx, int use_mmap, int verbose) {
    return dec_load_seq(path, n_threads, n_ctx, use_mmap, verbose, 1);
}

// n_seq > 1 enables dec_eval_fork: one shared state plus up to n_seq - 1 question sequences. The KV cache is unified,
// so the state's cells are shared by every question rather than copied; each Gated DeltaNet layer keeps one recurrent
// state per sequence. n_ctx must hold the state plus all question tokens of one fork.
void * dec_load_seq(const char * path, int n_threads, int n_ctx, int use_mmap, int verbose, int n_seq) {
    const int64_t t0 = dec_now_ns();
    memset(dec_phase_ns, 0, sizeof(dec_phase_ns));
    if (!verbose) {
        llama_log_set(quiet, NULL);
    }
    // Portable builds (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS) ship one CPU backend per ISA level and load the best
    // one for this host: Lambda's arm64 fleet mixes Graviton generations, and a Neoverse-V1 native build dies with
    // SIGILL on older hosts. DEC_GGML_DIR names the directory with the libggml-cpu-*.so variants.
    const char * dir = getenv("DEC_GGML_DIR");
    if (dir) {
        ggml_backend_load_all_from_path(dir);
    } else {
        ggml_backend_load_all();
    }
    llama_backend_init();
    const int64_t t1 = dec_now_ns();
    dec_phase_ns[DEC_PH_BACKEND] = t1 - t0;
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    struct llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        return NULL;
    }
    const int64_t t2 = dec_now_ns();
    dec_phase_ns[DEC_PH_MODEL] = t2 - t1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;   // a whole 4k-window prompt in one ubatch: one large GEMM per weight
    cp.n_ubatch = n_ctx < 4224 ? n_ctx : 4224;  // larger fork batches split; this bounds the compute buffers
    // rotabl copy: DEC_UBATCH=n overrides the ubatch size (capped at n_ctx); llama.cpp then splits each n_batch
    // decode into ceil(n_tokens / n) ubatches, carrying the KV cache and the recurrent states across them.
    const char * ub = getenv("DEC_UBATCH");
    if (ub && atoi(ub) > 0) {
        cp.n_ubatch = atoi(ub) < n_ctx ? atoi(ub) : n_ctx;
    }
    cp.n_seq_max = n_seq;
    cp.kv_unified = true;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    // CPU flash attention was 24% of a 512-token forward on Graviton3 (perf), and off is 15-35% faster end to end.
    const char * fa = getenv("DEC_FA");
    cp.flash_attn_type = (fa && fa[0] == '1') ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    // DEC_KV=f32: fp32 K/V. With an fp16 cache the 6 attention layers' QK and V products ran through
    // ggml_vec_dot_f16 (15% of a 300-token forward on Graviton3); fp32 takes the GEMM path.
    const char * kv = getenv("DEC_KV");
    if (kv && strcmp(kv, "f32") == 0) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
    }
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return NULL;
    }
    const int64_t t3 = dec_now_ns();
    dec_phase_ns[DEC_PH_CONTEXT] = t3 - t2;
    dec_phase_ns[DEC_PH_TOTAL] = t3 - t0;
    dec_t * d = calloc(1, sizeof(dec_t));
    d->model = model;
    d->ctx = ctx;
    d->n_embd = llama_model_n_embd(model);
    d->n_ctx = n_ctx;
    d->batch = llama_batch_init(n_ctx, 0, 1);
    // E12: DEC_CLEAR_DATA=0 resets only the cache metadata between requests (llama_memory_clear(mem, false)) and
    // leaves the K/V and recurrent buffers as the last request wrote them. Unset or any other value keeps the
    // baseline, which also zeroes every buffer byte. Metadata-only is enough at llama.cpp f74e2fa6a: the KQ mask
    // gives every empty attention cell -INFINITY, so a stale K/V row has weight 0, and the recurrent cache zeroes one
    // state row in the graph (build_rs: rs_z, scale by 0) and copies it into every cell a sequence starts in.
    const char * cd = getenv("DEC_CLEAR_DATA");
    d->clear_data = !(cd && strcmp(cd, "0") == 0);
    return d;
}

int dec_n_embd(void * h) { return ((dec_t *) h)->n_embd; }

// 1 if dec_eval / dec_eval_fork zero the cache buffers before each request (the baseline), 0 if they reset only the
// cache metadata (DEC_CLEAR_DATA=0).
int dec_clear_data(void * h) { return ((dec_t *) h)->clear_data ? 1 : 0; }

int dec_set_threads(void * h, int n_threads) {
    llama_set_n_threads(((dec_t *) h)->ctx, n_threads, n_threads);
    return 0;
}

// toks[n] -> out[n_out * n_embd]: the hidden state after the final norm at each out_pos (ascending or not).
int dec_eval(void * h, const int32_t * toks, int n, const int32_t * out_pos, int n_out, float * out) {
    dec_t * d = (dec_t *) h;
    if (n <= 0 || n > d->n_ctx) {
        return -2;
    }
    llama_memory_clear(llama_get_memory(d->ctx), d->clear_data);
    struct llama_batch b = d->batch;
    b.n_tokens = n;
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 0;
    }
    for (int j = 0; j < n_out; j++) {
        if (out_pos[j] < 0 || out_pos[j] >= n) {
            return -3;
        }
        b.logits[out_pos[j]] = 1;
    }
    int rc = llama_decode(d->ctx, b);
    if (rc != 0) {
        return rc;
    }
    for (int j = 0; j < n_out; j++) {
        const float * e = llama_get_embeddings_ith(d->ctx, out_pos[j]);
        if (!e) {
            return -4;
        }
        memcpy(out + (size_t) j * d->n_embd, e, sizeof(float) * (size_t) d->n_embd);
    }
    return 0;
}

// Shared-state fork, the CPU counterpart of SystemOneEngine's shared-prefix path: decode the state once (sequence 0,
// no outputs), copy it to sequences 1..nq, then decode every question's tokens in one batch. Question i has qlens[i]
// tokens (concatenated in qtoks) and wants n_out[i] outputs at question-relative positions (concatenated in out_pos).
// out receives the hidden states in that order. Returns 0, or a negative code / llama_decode's code.
int dec_eval_fork(void * h, const int32_t * state, int n_state, const int32_t * qtoks, const int32_t * qlens, int nq,
                  const int32_t * out_pos, const int32_t * n_out, float * out) {
    dec_t * d = (dec_t *) h;
    if (nq <= 0 || nq + 1 > (int) llama_n_seq_max(d->ctx)) {
        return -5;
    }
    int total = n_state;
    for (int i = 0; i < nq; i++) {
        total += qlens[i];
    }
    if (n_state <= 0 || total > d->n_ctx) {
        return -2;
    }
    llama_memory_t mem = llama_get_memory(d->ctx);
    llama_memory_clear(mem, d->clear_data);
    struct llama_batch b = d->batch;
    b.n_tokens = n_state;
    for (int i = 0; i < n_state; i++) {
        b.token[i] = state[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 0;
    }
    int rc = llama_decode(d->ctx, b);
    if (rc != 0) {
        return rc;
    }
    for (int i = 0; i < nq; i++) {
        llama_memory_seq_cp(mem, 0, i + 1, -1, -1);
    }
    int n = 0, q0 = 0, o0 = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < qlens[i]; j++) {
            b.token[n] = qtoks[q0 + j];
            b.pos[n] = n_state + j;
            b.n_seq_id[n] = 1;
            b.seq_id[n][0] = i + 1;
            b.logits[n] = 0;
            n++;
        }
        for (int k = 0; k < n_out[i]; k++) {
            const int p = out_pos[o0 + k];
            if (p < 0 || p >= qlens[i]) {
                return -3;
            }
            b.logits[n - qlens[i] + p] = 1;
        }
        q0 += qlens[i];
        o0 += n_out[i];
    }
    b.n_tokens = n;
    rc = llama_decode(d->ctx, b);
    if (rc != 0) {
        return rc;
    }
    int base = 0, row = 0;
    o0 = 0;
    for (int i = 0; i < nq; i++) {
        for (int k = 0; k < n_out[i]; k++) {
            const float * e = llama_get_embeddings_ith(d->ctx, base + out_pos[o0 + k]);
            if (!e) {
                return -4;
            }
            memcpy(out + (size_t) row * d->n_embd, e, sizeof(float) * (size_t) d->n_embd);
            row++;
        }
        base += qlens[i];
        o0 += n_out[i];
    }
    return 0;
}

void dec_free(void * h) {
    dec_t * d = (dec_t *) h;
    llama_batch_free(d->batch);
    llama_free(d->ctx);
    llama_model_free(d->model);
    free(d);
}
