# lambda-kleidiai E5: mmap-friendly weight format (no-vocab GGUF vs baseline GGUF)

One variable against the [P1 baseline](../baseline/README.md): the weight file's format. Same code (P1's
`function.zip`, sha256 `26eb2651…`), same Q8_0 tensors byte for byte, same download path, same configuration, same
request set and harness. Results: [`results/20261020T114406Z/results.json`](results/20261020T114406Z/results.json)
(`lkai-result/1`, status `improved`), published to the
[P2 dashboard](https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results).

```bash
AWS_PROFILE=<profile> examples/lambda-kleidiai/e5/run.sh <run-id>
```

Needs an arm64 Linux host with Docker, like `baseline/run.sh`. `LKAI_BUILD` points at P1's `build/` (default
`baseline/build`); when it already holds the zip and GGUF of P1's published run (checked by sha256), the build is
skipped and E5's baseline arm uses exactly those. Knobs are `baseline/run.sh`'s: `REGION`, `ROUNDS`, `CONCURRENCY`,
`WARM`, `PREFIX` (default `lkai-e5`), `TEARDOWN`.

## Which formats the loader can use

The loader is llama.cpp `f74e2fa6a` with `llama/patch_llama.py`, called through `shim/decshim.c` with
`LLAMA_LOAD_MODE_MMAP` on the `memfd` the PR #77 loader fills. Reading its source:

- **The weights are mmapped, then copied.** Every Q8_0 matmul weight goes to the `CPU_KLEIDIAI` extra buffer type
  (`ggml-cpu/kleidiai/kleidiai.cpp`). Its `set_tensor` re-packs each tensor from the mapping: dequantize the Q8_0
  blocks, requantize per row to int8, and pack with the KleidiAI kernel's layout. `token_embd.weight` (540 MB) is
  mapped for `GET_ROWS` and also repacked a second time as the tied output head (`TENSOR_DUPLICATED`), although the
  prefill patch never runs the LM head.
- **No on-disk pre-packed layout exists in this loader.** The repacked GGUF types (`Q4_0_4_4`, `Q4_0_4_8`,
  `Q4_0_8_8`) were removed upstream ("use Q4_0 and runtime repack"), and KleidiAI's packed format is a runtime
  buffer with no GGUF type. A format that skipped the repack therefore needs a loader change.
- **What a format can change is the metadata.** `llama_model_load_from_file` parses the whole key-value section before
  it maps a tensor. In the baseline file that section is 11.0 MB, almost all of it llama.cpp's text vocabulary:
  `tokenizer.ggml.tokens` (248,320 strings, 4.85 MB), `tokenizer.ggml.merges` (247,587 strings, 5.09 MB) and
  `token_type` (0.99 MB). llama.cpp builds hash maps of the tokens and merges. The engine never uses them: it
  tokenizes in Python with `assets/tokenizer.json` and passes token ids to `decshim`.

Diagnostics on the arm64 DevSpace (CPU part `0xd40` as read by the probe, 6 threads, `public.ecr.aws/lambda/python:3.12`, the deployed zip;
`.agents` scratch, not committed) split the ~8 s of `model_load_mmap`:

| Variant (diagnostic only) | `model_load_mmap` | Outputs |
|---|---|---|
| baseline GGUF | 7.90–8.05 s (6 loads) | reference |
| `GGML_KLEIDIAI_SKIP=token_embd` (no repack of the tied head) | 6.25–6.27 s | bit-identical |
| `GGML_KLEIDIAI_SKIP=.` (no KleidiAI repack at all) | 1.80–1.84 s | differ; forwards 1.6× slower |
| **no-vocab GGUF (E5)** | **7.38–7.74 s** | **bit-identical** |

KleidiAI's repack is about 6 s of the load, but avoiding it is a loader or configuration change, not a format change,
so it is out of E5's scope and is listed under follow-ups. The tied-head repack is the same: the GGUF already
omits `output.weight`, and the tensor is created as a duplicate by the loader.

## The conversion

```bash
python examples/lambda-kleidiai/e5/convert_no_vocab.py s2b-h512-Q8_0.gguf s2b-h512-Q8_0-novocab.gguf
```

`run.sh` runs it in `container_convert.sh`: image
`python@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b`, llama.cpp
`f74e2fa6a31cc33ab4ee4ae516641cfe62df3924`'s gguf-py 0.19.0, numpy 2.5.4, pyyaml 6.0.3,
tqdm 4.67.1, and Python 3.12.15. The conversion record pins each value; `manifest_e5.py` rejects a converter
checkout that differs from the published baseline manifest. The script copies every key except the `tokenizer.*`
keys, writes `tokenizer.ggml.model = "none"` (llama.cpp's
`LLAMA_VOCAB_TYPE_NONE`) and `qwen35.vocab_size = 248320` (the loader's `n_vocab` for the `token_embd` shape), and
copies the tensor infos and tensor bytes unchanged at the same 32-byte alignment. `tensor_identity.py` proves the
tensors are unchanged. The record is [`results/20261020T114406Z/e5-manifest.json`](results/20261020T114406Z/e5-manifest.json).
Its `llama_cpp` commit, image digest and `gguf_py` version were filled in after run `20261020T114406Z` from the
baseline pins; that run's record left them empty or imprecise (`provenance_correction` says how and why). The tensor
bytes and digest were measured by the run itself.

| | Baseline | E5 |
|---|---|---|
| File | `s2b-h512-Q8_0.gguf` | `s2b-h512-Q8_0-novocab.gguf` |
| Bytes | 2,076,712,416 | 2,065,771,264 (−10,941,152, −0.53%) |
| sha256 | `8dbb8872e6af…` | `29aa5abe0fe3…` |
| Data section offset | 11,000,032 | 58,880 |
| Tensors | 335: 195 Q8_0, 140 F32 | the same 335, same order, types, shapes and bytes |
| Tensor digest (sha256 over each tensor's name and bytes) | `8141645499ea…` | `8141645499ea…` |

Same quantization and the same numerical content, so the only expected output difference is none.

## The benchmark

`deploy_e5.py` deploys `lkai-e5-base` and `lkai-e5-e5` with P1's `deploy.functions`: same zip, same configuration
(python3.12, arm64, 10,240 MB, SnapStart on published versions, runtime pinned to the same version ARN) and the same
environment except `MODEL_KEY`, which names each arm's GGUF in the one bucket. Both arms are instrumented
(`LKAI_PHASES=1`); P1 measured that overhead as within run-to-run spread. `baseline/bench.py --arms base,e5` runs
unchanged: 5 rounds, arm order alternating between rounds (not within-round pairing), per round 10 cold, 10 restore
and 10 warm passes of 10 per arm, with every expected start class checked against the REPORT line. `summarize_e5.py` adds per-arm correctness and `e5_vs_base`: the
median difference within each CPU part (`/proc/cpuinfo` "CPU part") both arms drew, pooled by sample count (P1's method for its
overhead comparison). `host_variability.py` adds the host-variability assessment below.

## Results: run `20261020T114406Z`

us-east-2, branch commit `16c452e`, baseline commit `aac59d4`. 1,200 invocations, 0 errors, 0 class mismatches.
Every answer of both arms (600 per arm) was bit-identical to P1's `reference/outputs.json` for its request and CPU
part: 0 choice flips, maximum absolute delta 0.0.

E5 minus baseline, p50, matched by CPU part (part `0xd40` carries most samples; pooled = weighted over the
parts both arms drew). CPU parts are the `/proc/cpuinfo` ids the handler read; this run recorded nothing that names a
Graviton generation (see [Host variability](#host-variability)).

| Class | Phase | Baseline p50 (`0xd40`) | E5 p50 (`0xd40`) | Δ `0xd40` | Δ pooled |
|---|---|---|---|---|---|
| cold | first request (Init Duration + Duration) | 36.03 s | 35.17 s | −0.86 s (−2.4%) | −0.87 s |
| cold | weight download | 25.01 s | 24.84 s | −0.17 s | −0.18 s |
| cold | deserialize / mmap (`model_load_mmap`) | 8.39 s | 7.72 s | −0.66 s (−7.9%) | −0.66 s |
| cold | model init (`context_init`) | 0.324 s | 0.326 s | +0.002 s | +0.002 s |
| cold | first inference (warm-up) | 0.250 s | 0.251 s | +0.001 s | +0.001 s |
| cold | first request inference (probe) | 0.240 s | 0.237 s | −0.003 s | −0.003 s |
| cold | `model_ready` (PR #77's load, end to end) | 34.98 s | 34.17 s | −0.81 s | −0.83 s |
| cold | max memory used | 5,158 MB | 5,060 MB | −98 MB | −94 MB |
| restore | first request (Restore Duration + Duration) | 3.91 s | 3.91 s | −0.004 s | −0.018 s |
| restore | REPORT Restore Duration | 3.235 s | 3.220 s | −0.015 s | −0.025 s |
| restore | first request inference | 0.671 s | 0.682 s | +0.011 s | +0.007 s |
| restore | max memory used | 5,150 MB | 5,070 MB | −80 MB | −80 MB |
| warm | steady-state inference | 0.628 s | 0.656 s | +0.028 s | +0.024 s |
| warm | max memory used | 5,167 MB | 5,091 MB | −76 MB | −77 MB |

- **Deserialize/mmap** fell on every CPU part: `0xd40` −7.9%, `0xd4f` (n = 7 baseline / 4 E5) 7.13 → 6.53 s,
  `0xd0c` (n = 2 / 3) 19.59 → 18.84 s. The snapshot inits inside `publish_version` show the same change
  (`model_load_mmap` p50 8.10 → 7.59 s, n = 10 each).
- **Run-to-run variance** (CV of the per-round medians of the cold first request): baseline 0.36%, E5 0.30%; the
  range of the per-round medians is 0.30 s and 0.29 s. The 0.87 s cold improvement is about three times that range,
  and every one of E5's five per-round medians (35.01–35.30 s) is below every one of the baseline's (35.83–36.13 s).
- **Restore and warm are unchanged** within their spread (restore CV 1.3% / 2.3%, warm 6.0% / 3.1%). This is expected:
  the snapshot already holds the loaded model, so a restore never parses the file. The warm `0xd40` +4.5% is not
  consistent across CPU parts (`0xd0c` −3.5%) and is inside the warm per-round range (0.09 s baseline).
- **Peak memory** fell 76–98 MB in every class, about ten times the 11 MB of metadata removed. A likely cause is the
  vocabulary's hash maps and strings that llama.cpp keeps for the model's lifetime; this run did not profile it.
- **Weight download** is unchanged by design (same loader, same 64 MiB ranges); the 0.17 s is the 0.5% fewer bytes
  at ~80 MiB/s, plus noise.
- **Cost** of the run: $2.37 (SnapStart cache $1.63 for 10 versions at the 3-hour minimum). 120 weight downloads,
  249 GB from S3 within the region. All published versions were deleted; the two functions and the bucket remain.

## Host variability

Assessed after the run under the initiative's HOST VARIABILITY rules, from this run's own records, with no new
invocations: `python host_variability.py results/20261020T114406Z` (also written by `summarize_e5.py` as
`results/<run-id>/host-variability.json` and `results.json` `host_variability`).

**What the run observed.** Per execution environment: `/proc/cpuinfo` "CPU part", the `libggml-cpu` variants
ggml selected at init, REPORT Memory Size, the pinned runtime version ARN, and the environment id and log stream.
Not recorded, so unknown: CPU implementer, variant and revision, MIDR, feature flags, HWCAP/HWCAP2, online core
count, cgroup limits, kernel, and AZ. Thread count (6) is configured, not observed. No Graviton generation was
observed; this README names parts by id only and infers nothing from timings.

**What it drew.** 200 execution environments, each on one CPU part (no environment reported two). Three parts
appeared, each with one ggml variant set on both arms:

| CPU part | ggml CPU variants | Environments base / E5 (cold, restore, warm) |
|---|---|---|
| `0xd40` | `armv8.0_1`, `armv8.6_1` | 41 / 43, 47 / 45, 47 / 45 |
| `0xd4f` | `armv8.0_1`, `armv8.6_2` | 7 / 4, 0 / 0, 0 / 0 |
| `0xd0c` | `armv8.0_1`, `armv8.2_2` | 2 / 3, 3 / 5, 3 / 5 |

The parts differ in speed (cold `model_load_mmap` p50 7.1 s on `0xd4f`, 8.4 s on `0xd40`, 19.6 s on `0xd0c`), so the
mix matters, and the arms' mixes are not equal.

**Could the mix change a conclusion?** No. Every comparison above is within one part, and arm order alternates
between rounds. Arms run sequentially within a round, so time-within-round drift remains a limitation. Per part:

| Class, metric | Δ p50 per part (E5 − base) | Rounds E5 below base | Unmatched Δ (all parts pooled) |
|---|---|---|---|
| cold first request | `0xd40` −0.86 s, `0xd4f` −0.70 s, `0xd0c` −1.51 s | 5/5, 2/2, 2/2 | −0.83 s |
| cold `model_load_mmap` | −0.66 s, −0.60 s, −0.75 s | 5/5, 2/2, 2/2 | −0.65 s |
| cold max memory | −98, −71, −67 MB | 5/5, 2/2, 2/2 | −98 MB |
| restore first request | `0xd40` −0.004 s, `0xd0c` −0.23 s | 3/5, 2/2 | +0.013 s |
| warm steady-state inference | `0xd40` +0.028 s, `0xd0c` −0.041 s | 2/5, 2/2 | +0.007 s |

- **Cold** improves on every part and in every round that drew the part. The base arm drew more of the fastest part
  (`0xd4f` 7 vs 4) and E5 more of the slowest (`0xd0c` 3 vs 2), which biases an unmatched comparison against E5; the
  matched −0.87 s is larger than the unmatched −0.83 s. The local diagnostics, on one fixed `0xd40` host, show the
  same direction and size (`model_load_mmap` 7.90–8.05 s → 7.38–7.74 s).
- **Restore and warm** have no consistent sign across parts or rounds, which is the "unchanged" conclusion.
  `0xd0c` has 3–5 environments per arm, too few to read its restore −0.23 s as an effect.
- Only `0xd40` has enough samples for a tight per-part interval; `0xd4f` (cold only) and `0xd0c` are small.

**Rerun: none.** No conclusion depends on the CPU mix, so the HOST VARIABILITY rules do not call for one, and no new
spend was approved. The unrecorded fields above stay unknown for this run; recording them is a harness change that
H1 owns, and E5 does not need the fixed-host EC2 baseline because the cold effect is already measured within each
part.

## Conclusion

Drop the text vocabulary from the deployed GGUF. It takes 0.66 s off cold deserialization and about 0.87 s off the
cold first request (−2.4%), lowers peak memory by ~80–95 MB, and changes no output. The cold improvement holds within
every CPU part the run drew. It does nothing for SnapStart
restores or warm requests. The larger load cost, KleidiAI's ~6 s runtime repack (including ~1.7 s for the tied output
head that is never used), cannot be removed by any file format this loader reads.

## Follow-ups (not measured here)

- Skip the tied output head's repack (`GGML_KLEIDIAI_SKIP=token_embd` or not creating `output` in prefill mode):
  −1.7 s of load locally with bit-identical answers. A configuration or loader change, not a format.
- Persist KleidiAI's packed buffers (a loader change) to remove most of the remaining ~6 s.

## Files

| | |
|---|---|
| `convert_no_vocab.py` | the conversion |
| `tensor_identity.py` | tensor-by-tensor proof the numerical content is unchanged |
| `container_convert.sh` | runs both in the pinned image and records tool versions |
| `manifest_e5.py` | `build/manifest.json` (P1's published one when hashes match) and `e5-manifest.json` |
| `deploy_e5.py` | two functions, one variable (`MODEL_KEY`) |
| `summarize_e5.py` | `lkai-result/1` with per-arm correctness, `e5_vs_base` and `host_variability` |
| `host_variability.py` | observed host metadata, per-CPU-part comparison and verdict, from a run's own records |
| `run.sh` | the whole experiment in one command |
| `results/<run-id>/` | redacted run artifacts and `results.json` |
