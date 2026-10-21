# lambda-kleidiai E9: KleidiAI-prepacked weights (pack at build time instead of at load)

One variable against the [P1 baseline](../baseline/README.md) (`aac59d4`): **when the KleidiAI packing of the Q8_0
matmul weights happens**. Baseline: at load, in `llama_model_load_from_file`. E9: at build time, with the bytes shipped
already in KleidiAI's layout. The download path, request set, configuration and everything else are P1's.

Status: **improved**. Feasible, offline-verified (packed bytes hash-identical to the runtime repack), and measured on
Lambda in run `20261021T101749Z`: cold first request −4.87 s on CPU part `0xd40` (−13.6%), every answer
bit-identical. Results: [`results/20261021T101749Z/results.json`](results/20261021T101749Z/results.json)
(`lkai-result/1`), published to the [P2 dashboard](https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results).

```bash
AWS_PROFILE=<profile> examples/lambda-kleidiai/e9/run.sh <run-id>
```

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

## Hypothesis (before the Lambda run)

Cold `model_load_mmap` (P1 p50 8.39 s on `0xd40`) falls by about 4.8 s, and the cold first request by about the same.
There is no extra download time to subtract. The snapshot init inside `publish_version` falls by the same amount.
Restore and warm should not change: a restored environment already holds the packed model.

## Lambda results: run `20261021T101749Z`

us-east-2, measured at branch commit `55692de` against baseline `aac59d4`. `deploy_e9.py` deployed `lkai-f1ee-base`
(P1's zip `26eb2651…` + P1's GGUF) and `lkai-f1ee-e9` (E9's zip `488890bd…` + the E9 GGUF), with P1's configuration
(python3.12, arm64, 10,240 MB, SnapStart on published versions, one pinned runtime version ARN) in their own bucket,
role and log groups. `baseline/bench.py` ran unchanged in its method: 5 rounds, arm order alternating between
rounds, per round 10 cold, 10 restore and 10 warm passes of 10 per arm, with I1's fresh invoke client per wave
(`5764b9e`; 120 clients) and a per-arm zip check. 1,200 invocations, **0 errors, 0 class mismatches**, 600/600
answers per arm **bit-identical** to P1's `reference/outputs.json` (0 choice flips, max |Δ| 0.0). Cost: $2.36
(SnapStart cache $1.63 for 10 versions at the 3-hour minimum). All published versions were deleted.

E9 minus baseline, p50, within each observed CPU part (parts are `/proc/cpuinfo` ids; no generation was observed or
inferred). n is base / e9.

| Class | Phase | `0xd40` base → e9 (n 39 / 45) | Δ `0xd40` | Δ `0xd4f` (n 4 / 3) | Δ `0xd0c` (n 7 / 2) |
|---|---|---|---|---|---|
| cold | first request (Init Duration + Duration) | 35.94 → 31.07 s | **−4.87 s (−13.6%)** | −4.46 s | −8.22 s |
| cold | weight download | 25.01 → 25.01 s | −0.000 s | +0.04 s | −0.02 s |
| cold | deserialize / mmap (`model_load_mmap`) | 8.35 → 3.44 s | **−4.91 s (−59%)** | −4.28 s | −8.09 s |
| cold | model init (`context_init`) | 0.322 → 0.319 s | −0.004 s | −0.011 s | +0.019 s |
| cold | first inference (warm-up) | 0.252 → 0.252 s | +0.000 s | −0.007 s | −0.007 s |
| cold | first request inference (probe) | 0.237 → 0.242 s | +0.005 s | −0.002 s | +0.005 s |
| cold | REPORT Init Duration | 0.794 → 0.801 s | +0.008 s | −0.005 s | −0.009 s |
| cold | max memory used | 5,158 → 5,158 MB | 0 MB | −4 MB | −4 MB |

| Class | Metric | Δ `0xd40` (n base / e9) | Δ `0xd0c` (n base / e9) |
|---|---|---|---|
| restore | first request (Restore Duration + Duration) | −0.058 s (42 / 44) | +0.153 s (8 / 6) |
| restore | REPORT Restore Duration | −0.053 s | +0.205 s |
| restore | max memory used | −18 MB | +18 MB |
| warm | steady-state inference | −0.004 s (420 / 440) | −0.027 s (80 / 60) |
| warm | max memory used | −21 MB | +11 MB |

- **Cold** improves on every part, in every round that drew the part. On `0xd40` every E9 per-round median of the
  cold first request (30.90–31.19 s) is below every baseline one (35.85–36.04 s); `model_load_mmap` per-round
  ranges are 0.08 s (base) and 0.09 s (e9) against a 4.91 s effect. Pooled −5.00 s; unmatched −4.90 s.
- **No download cost.** The 128 extra bytes do not move weight download (25.01 s p50 on both arms).
- **Snapshot init** inside `publish_version` (40 records, each record's part from its own log stream, 0 missing,
  0 ambiguous): `model_load_mmap` 8.11 → 3.24 s on `0xd40` and 19.14 → 11.24 s on `0xd0c`, lower in 5/5 rounds on
  both.
- **Load paths** (`results.json` → `load_paths`, from the `E9KAI` lines): every `0xd40` and `0xd4f` environment
  adopted the i8mm-layout payloads; every `0xd0c` environment selected the dot-product kernel and converted, which
  still removed 8.1 s of its 19.7 s load. The tied output head took the runtime repack everywhere, as designed.
- **Restore and warm** are unchanged: their deltas are inside the per-round ranges and change sign across parts.
  A restored environment already holds the packed model, so E9 has nothing to change there. The `0xd0c` restore
  +0.15 s rests on 8 / 6 samples.
- **Peak memory** is unchanged within ~20 MB, with no consistent sign.

### Host variability

`results.json` → `host_variability` (also `host-variability.json`), from the run's own records, with E5's method.

**Observed** per environment: `/proc/cpuinfo` "CPU part", the `libggml-cpu` variants mapped after init, the
KleidiAI load path (`E9KAI` line), REPORT Memory Size, the pinned runtime version ARN, the environment id and the
log stream. **Unknown** (not recorded): CPU implementer, variant and revision, MIDR, feature flags, HWCAP/HWCAP2,
online core count, cgroup limits, kernel and AZ. Thread count (6) is configured, not observed. No Graviton
generation was observed.

| CPU part | ggml CPU variants | KleidiAI path (e9 arm) | Cold envs base / e9 | Restore envs base / e9 |
|---|---|---|---|---|
| `0xd40` | `armv8.0_1`, `armv8.6_1` | adopt, `kr = 8` | 39 / 45 | 42 / 44 |
| `0xd4f` | `armv8.0_1`, `armv8.6_2` | adopt, `kr = 8` | 4 / 3 | 0 / 0 |
| `0xd0c` | `armv8.0_1`, `armv8.2_2` | convert, `kr = 4` | 7 / 2 | 8 / 6 |

200 environments; none reported two parts. **Could the mix change a conclusion?** No. The cold effect has the same
sign in every part and every round, it is ~17 times the larger per-round range on `0xd40`, and the unmatched
comparison (−4.90 s) agrees with the matched one (−5.00 s). Restore and warm have no consistent sign, which is the
"unchanged" conclusion. **Rerun: none needed.** H1's fixed-host EC2 baseline is not needed either: the offline
DevSpace runs already isolate the load's CPU work on one `0xd40` host (8.06 → 3.21 s), and Lambda decides the
deployment conclusion.

Limitations: `0xd4f` (cold only) and `0xd0c` have few samples, so only `0xd40` has a tight per-part interval.
Within a round the cold and restore waves run one arm after the other, so drift within a round is not controlled
for them; arm order alternated between rounds, and e9 was lower in rounds of both orders.

## Conclusion

Ship the weights prepacked. Doing KleidiAI's packing at build time removes ~4.9 s from cold `model_load_mmap` and
from the cold first request on `0xd40` (−13.6% end to end), more on `0xd0c` (−8.2 s via the convert path), and the
same from every snapshot init. It costs no download time (+128 bytes), no memory, and changes no output. Restore
and warm are unaffected.

Compatibility and cost of adoption: the E9 file needs E9's loader patch (P1's loader would misread it; the patch
refuses instead), it is tied to llama.cpp `f74e2fa6a` + KleidiAI v1.31.0's packed layout, and a CPU without
KleidiAI's Q8_0 kernels (no dotprod) cannot load it. None was observed. A loader upgrade must re-run
`offline.sh`'s hash check.

The weight download (25 s) is now ~80% of the cold first request. The ~3.4 s left in `model_load_mmap` is mostly
the tied output head's runtime repack (E7's variable) and copying the payloads into the KleidiAI buffer. E9 and
E7 should compound; R's ladder should measure them together.

## Status

Done: feasibility, offline correctness and fixed-host timing, the Lambda measurement, and the host-variability
assessment. The Lambda run was approved by the operator (playground1, us-east-2, $5 cap; spent $2.36).

## Reproduce

```bash
git fetch https://github.com/jsamuel1/strands-decider lkai/f1ee-prepacked && git checkout FETCH_HEAD
examples/lambda-kleidiai/baseline/build.sh                     # P1's build (or reuse one)
AWS_PROFILE=<profile> examples/lambda-kleidiai/e9/run.sh <run-id>   # build, deploy, bench, summarize, redact
examples/lambda-kleidiai/e9/offline.sh examples/lambda-kleidiai/baseline/build examples/lambda-kleidiai/e9/build out 3
python3 examples/lambda-kleidiai/e9/offline_verdict.py out examples/lambda-kleidiai/baseline/reference/outputs.json
python examples/lambda-kleidiai/e9/host_variability.py examples/lambda-kleidiai/e9/results/<run-id>
pytest tests/test_lkai_e9.py
```

## Files

| | |
|---|---|
| `pack_e9.py` | the build-time packer (numpy; same arithmetic as the runtime repack) |
| `patch_e9.py` | the loader change: adopt, convert, refuse |
| `build.sh`, `container_pack.sh` | E9's zip (P1's `container_build.sh` on the patched tree) and GGUF, in P1's pinned images |
| `probe_e9.py`, `offline.sh`, `offline_verdict.py` | offline parity, answers and fixed-host timing |
| `deploy_e9.py`, `run.sh` | two functions (P1's zip + GGUF, E9's zip + GGUF) and the whole Lambda experiment |
| `summarize_e9.py`, `host_variability.py` | `lkai-result/1`, `e9_vs_base`, `load_paths`, host assessment |
| `meta.json` | hypothesis, changed variable, conclusion |
| `results/offline-20261021/` | probe outputs, stderr (`E9KAI` lines), host info, pack record, llama.cpp diff |
| `results/20261021T101749Z/` | the Lambda run (redacted): invocations, logs, conditions, results |

`baseline/bench.py` on this branch carries two harness-only changes, recorded here as deliberate (P1 `aac59d4` →
this branch): I1's fresh invoke client per wave (`5764b9e`), and a per-arm zip check that reads
`deployment.json`'s `zip_sha256_by_arm` (default: every arm runs `build/function.zip`, as before).
