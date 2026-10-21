# lambda-kleidiai E9: KleidiAI-prepacked weights (pack at build time instead of at load)

One variable against the [P1 baseline](../baseline/README.md) (`aac59d4`): **when the KleidiAI packing of the Q8_0
matmul weights happens**. Baseline: at load, in `llama_model_load_from_file`. E9: at build time, with the bytes shipped
already in KleidiAI's layout. The download path, request set, configuration and everything else are P1's.

Status: **feasible, offline-verified; Lambda not yet measured** (see [Status](#status)).

## Feasibility

### The packed layout and its size

The pinned loader (llama.cpp `f74e2fa6a`, KleidiAI v1.31.0) sends every Q8_0 matmul weight to the `CPU_KLEIDIAI`
buffer type. Its `set_tensor` (`kleidiai.cpp`, `repack`) does two things per tensor:

1. per row: dequantize the Q8_0 blocks, take max |value|, `scale = max / 127`, requantize to int8 with
   `lround(value / scale)` clamped to ±127;
2. pack the int8 rows and row scales with `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` for the selected kernel.

The packed slot is `ceil(n/nr)` groups of `nr × (k_round32 + 12)` bytes: `nr × k` int8 values interleaved in
`kr`-wide blocks, then `nr` int32 row sums, `nr` float scales and `nr` zero biases. Both kernels this model uses
have `nr = 4`; they differ only in `kr`:

| Kernel selected by the loader | Required CPU features | `nr`, `kr` | CPU parts P3 recorded with it |
|---|---|---|---|
| `qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` | i8mm + dotprod | 4, 8 | `0xd40` (also this DevSpace) |
| `qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod` | dotprod | 4, 4 | `0xd0c` |

P3 recorded `0xd4f` loading `libggml-cpu-armv8.6_2` but not which KleidiAI kernel it selected: unknown here.

For this model the packed form is **smaller** than Q8_0: 1,952,244,864 bytes against 2,063,474,688 (0.946×), and
it fits inside each tensor's own Q8_0 bytes together with a 64-byte header and a 64-byte trailer, for every one of
the 194 per-layer Q8_0 tensors (`pack_e9.py` refuses a tensor that does not fit). So the E9
GGUF keeps every tensor name, shape, type, offset and byte size. It is the baseline file plus one 128-byte metadata
key:

| | Baseline | E9 |
|---|---|---|
| File | `s2b-h512-Q8_0.gguf` | `s2b-h512-Q8_0-e9kr8.gguf` |
| Bytes | 2,076,712,416 | 2,076,712,544 (+128) |
| sha256 | `8dbb8872e6af…` | `183137fb500e…` |
| Tensors prepacked | none | 194 (all Q8_0 except `token_embd.weight`) |

The download is therefore unchanged by construction: +128 bytes at ~79 MiB/s is ~2 µs. There is no download/size
trade-off to net against the load saving.

`token_embd.weight` stays Q8_0: `GET_ROWS` reads it natively. Its tied output-head copy is still repacked at load, in
both arms. Skipping that repack is E7's variable, not E9's.

### The loader change

`patch_e9.py` (applied after `llama/patch_llama.py`) changes two files.

- `kleidiai.cpp` `repack()`: a tensor ending in an `E9KAIPK1` trailer holds the KleidiAI weight header and packed
  slot. If the stored `(nr, kr, sr)` equals the kernel the loader selected, the payload is copied as is (**adopt**).
  If it does not (**convert**), the int8 rows and row scales are read back from the payload and packed for the
  selected kernel. The dequantize/requantize pass is skipped, but the inputs to the pack are the same values, so the
  packed bytes are the same. A tensor without a trailer takes the unchanged runtime repack. A trailer that does not
  describe its tensor, or a malformed payload, aborts the load.
- `llama-model-loader.cpp` `load_all_data()`: a mapped tensor carrying an E9 trailer must be in a `CPU_KLEIDIAI`
  buffer, and a GGUF with the `strands_decider.e9.kleidiai_prepacked` key must be loaded with mmap. Otherwise the
  packed bytes would be used as Q8_0 blocks, so the load fails instead.
- One stderr line per path and process: `E9KAI adopt|convert|repack nr=<nr> kr=<kr>`, so each environment's logs
  record which path it took.
- Diagnostics only, unset in every deployed function: `E9_DUMP_DIR` writes each packed payload for hashing, and
  `E9_DIAG_NO_I8MM=1` hides i8mm from KleidiAI's kernel choice, so one host can run the dot-product path.

### The other CPU variant

Ship one file, packed for the i8mm layout (`kr = 8`), and let a host that selects the dot-product kernel convert at
load. Shipping both layouts would double the file (and the download, ~25 s at the measured rate) to save a fraction
of a convert on the hosts that need it. The convert path still skips the dequantize/requantize step.

A CPU with no KleidiAI Q8_0 kernel (no dotprod) cannot use an E9 file, because the Q8_0 blocks cannot be recovered
from the per-row requantization; the load aborts. No such CPU part was observed in P1's or P3's Lambda runs, but that
is a sample, not a guarantee.

## Offline verification

`offline.sh` on the arm64 DevSpace, in P1's pinned Lambda image, with the E9 zip and both GGUFs. Results:
[`results/offline-20261021/offline-verdict.json`](results/offline-20261021/offline-verdict.json).

**Packed bytes.** With `E9_DUMP_DIR`, every Q8_0 tensor's packed payload, as the loader leaves it in memory, was
hashed. That is 187 tensors: the 186 prepacked tensors the model loads, plus the tied output head (runtime-repacked
in both arms). The 8 `blk.24.*` MTP tensors are packed in the file but not loaded by this loader.

| Comparison | Path in the E9 arm | Payload hashes |
|---|---|---|
| E9 GGUF vs P1 GGUF, i8mm kernel | adopt (`kr = 8`) | 187 / 187 identical, digest `6fe03c59e832…` |
| E9 GGUF vs P1 GGUF, `E9_DIAG_NO_I8MM=1` (dot-product kernel) | convert (`kr = 4`) | 187 / 187 identical, digest `20745aefac4e…` |

**Answers.** All 11 probes (both parity paths, the E9 zip on P1's GGUF, and every timing run, both arms) are
bit-identical to P1's `reference/outputs.json` for CPU part `0xd40`. The E9 zip on P1's GGUF loads through the
unchanged repack (`E9KAI repack`) and answers like P1's zip, so the patch is inert without a trailer.

`numpy` packing (`pack_e9.py`) is also pinned in `tests/test_lkai_e9.py` against a loop-for-loop Python translation
of the two C routines (requantize with float32 rounding and `lround`, and `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon`
including the clamped padding rows), for both `kr` values and the convert round-trip.

**Fixed-host load time.** Labelled fixed-host: one arm64 DevSpace, not Lambda latency. CPU part `0xd40` as read
from `/proc/cpuinfo`, 16 vCPUs, kernel 6.1.188, 6 threads. Three pairs of processes, alternating arm order per pair,
3 loads per process:

| `model_load_mmap` | n | median | min–max |
|---|---|---|---|
| baseline (P1 zip, P1 GGUF) | 9 | 8.056 s | 8.024–8.421 s |
| E9 (E9 zip, E9 GGUF) | 9 | 3.212 s | 3.146–3.439 s |
| difference | | **−4.84 s (−60%)** | ranges do not overlap |

The ~3.2 s left is mostly the tied output head's runtime repack (E5 measured ~1.7 s for it), the copy of ~1.95 GB of
payload into the KleidiAI buffer, and faulting in the memfd mapping.

Limitation: another card's benchmark shared the DevSpace during these runs (load average 12–13 on 16 vCPUs at the
end). Arms alternated, so both saw it, and the two arms' ranges are far apart, but the absolute numbers and the
first-inference timings in the probe files are inflated by it. Only `model_load_mmap` is compared.

## Hypothesis for Lambda

Cold `model_load_mmap` (P1 p50 8.39 s on `0xd40`) falls by about 4.8 s, and the cold first request by about the same.
There is no extra download time to subtract. The snapshot init inside `publish_version` falls by the same amount.
Restore and warm should not change: a restored environment already holds the packed model.

## Status

Offline feasibility and correctness are done. The Lambda measurement (P1's alternating cold/restore/warm harness,
with the fresh-client-per-wave fix from I1, per-CPU-part comparison, and the host metadata the run can observe) has
not been run: it needs operator approval for the account, region and budget. Until then this experiment is
published as offline-only.

## Reproduce

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/f1ee-prepacked && git checkout FETCH_HEAD
examples/lambda-kleidiai/baseline/build.sh                     # P1's build (or reuse one)
LKAI_BUILD=examples/lambda-kleidiai/baseline/build examples/lambda-kleidiai/e9/build.sh
examples/lambda-kleidiai/e9/offline.sh examples/lambda-kleidiai/baseline/build examples/lambda-kleidiai/e9/build out 3
python3 examples/lambda-kleidiai/e9/offline_verdict.py out examples/lambda-kleidiai/baseline/reference/outputs.json
pytest tests/test_lkai_e9.py
```

## Files

| | |
|---|---|
| `pack_e9.py` | the build-time packer (numpy; same arithmetic as the runtime repack) |
| `patch_e9.py` | the loader change: adopt, convert, refuse |
| `build.sh`, `container_pack.sh` | E9's zip (P1's `container_build.sh` on the patched tree) and GGUF, in P1's pinned images |
| `probe_e9.py`, `offline.sh`, `offline_verdict.py` | offline parity, answers and fixed-host timing |
| `results/offline-20261021/` | probe outputs, stderr (`E9KAI` lines), host info, pack record, llama.cpp diff |
