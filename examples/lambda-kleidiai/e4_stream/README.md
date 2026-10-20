# E4: stream the weights into the loader during the download (same format)

**Variable.** Whether the llama.cpp model load overlaps the weight download. Everything else is P1's baseline
(`../baseline/`, pushed HEAD `aac59d45555ebded9357c23f6f8e4f939edf25d7`): the same GGUF bytes
(`s2b-h512-Q8_0.gguf`, sha256 `8dbb8872…`), the same download (HEAD, then 31 ranged GETs of 64 MiB on 32 threads, 8 MiB
body reads, into one memfd), the same mmap load path and KleidiAI kernels, the same function configuration, harness and
request set. Format changes are E5's; download part size and concurrency are E3's.

## Feasibility (answered first)

The current format and loader can stream. Evidence, from llama.cpp `f74e2fa6a` (P1's pin) and the deployed GGUF's first
16 MiB (fetched with one ranged GET, parsed with a GGUF v3 reader):

| Question | Answer |
|---|---|
| Header and index location | At the front. GGUF v3: KV metadata (50 keys, including a 248k-token vocabulary) then the tensor index (335 entries) occupy bytes [0, 11,000,018); tensor data starts at 11,000,032 (alignment 32), 0.53 % of the 2,076,712,416-byte file. Index order equals file order. |
| Sequential or random access | Both. `gguf_init_from_file` reads the header sequentially. `llama_model_loader::load_all_data` then visits each tensor once, per buffer context, in creation order; the CPU_KLEIDIAI context (every Q8_0 matmul weight) and the CPU context (token_embd, norms) are interleaved in the file, so within the file the accesses jump forward and back. |
| Real file / mmap needed? | Not for correctness. The loader has an mmap path (P1: `/proc/self/fd/<memfd>`, `MAP_SHARED`), a `FILE *` path (`llama_model_load_from_file_ptr`, only `fread`/`fseek`/`ftell`) and a buffer path (`llama_model_init_from_user`, which however disables extra buffer types, i.e. KleidiAI). With mmap, a KleidiAI tensor is read once (repacked into its own buffer) during the load; a CPU tensor is only aliased into the mapping and read at inference. |
| Full materialization before init? | No. Before `load_all_data` the loader reads only the header (no tensor data: checked every `load_data_range`/`mapping->addr()` caller; qwen35 has no lazy tensors and no prism-hadamard metadata). Each tensor's bytes are needed only when `load_all_data` reaches it; the aliased CPU tensors only at the first inference. |

So the load can start once the header has arrived, provided each tensor read waits for its bytes and the first
inference waits for the whole file. Compatible formats for E5, for reference: any format with the index first and data
after it (GGUF as is; safetensors, whose 8-byte length plus JSON header precede the data). A format with a trailing
index (zip-style central directory) would need the tail first.

## Mechanism

`patch_stream.py` (applied after P1's `llama/patch_llama.py`) adds two dormant hooks, NULL by default:

- `llama_set_load_stream(wait, order, ud)`: `load_all_data` calls `wait(offs, size)` before each tensor read (the
  KleidiAI repack, a validation, any `FILE *` read), and once for the whole file before it returns. If `order` is set,
  each buffer context's tensors are loaded by `order(offs, size)` ascending instead of creation order.
- `gguf_set_read_wait(wait, ud)`: the header parser calls it with a watermark (about 4 calls for the 11 MB header).

`stream_loader.py` runs PR #77's download (same parameters) in a thread, tracks the contiguous bytes received per range,
and returns the memfd path once the first MiB is in. Its `order` key is "bytes a range must have received before this
tensor is complete": every range moves at about the same rate, so this is the expected arrival order. The base arm
never sets the hooks, so it runs P1's code path with one null check per tensor.

Arms, one zip (`e4_entry.handler`), differing only in environment:

| Arm | Environment | Meaning |
|---|---|---|
| `base` | `E4_STREAM=0` | download, then load (P1's sequence) |
| `stream` | `E4_STREAM=1 STREAM_ORDER=arrival` | the experiment |
| `streamfile` | `E4_STREAM=1 STREAM_ORDER=file` | overlap in llama.cpp's own order: shows how much the ordering matters |

## Reproduce

```bash
# arm64 Linux host with Docker; P1's build (baseline/build.sh) is reused when present
AWS_PROFILE=<profile> examples/lambda-kleidiai/e4_stream/run.sh <run-id>
# no AWS: each arm in the pinned Lambda image against a 80 MiB/s local stand-in for S3
examples/lambda-kleidiai/e4_stream/localtest.sh
```

## Results

See `results/<run-id>/results.json` and the P2 dashboard; the summary is filled in below once the run completes.
