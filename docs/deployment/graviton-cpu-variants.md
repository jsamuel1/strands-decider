# Graviton CPU variants

The decision function's image (`deploy/lambda/decision/`) runs llama.cpp on whichever Graviton
generation Lambda, Fargate or an ECS container instance gives it. Two builds exist:

| Build | `-c cpu=` | What it is |
| --- | --- | --- |
| `--lambda` | `baseline` (default) | One static library for Graviton2's features (`armv8.2-a+fp16+dotprod`), run as-is on every generation. |
| `--lambda-variants` | `variants` | ggml's CPU backend built once per Arm tier as `libggml-cpu-*.so` modules, plus one per Graviton generation; the shim loads the best one the host supports. |

`scripts/build_native.sh` builds both; `patches/0002-ggml-cpu-graviton-variants.patch` adds the
Graviton modules to llama.cpp.

## The Graviton modules

Each is tuned for its core and loads only on it, matched by the MIDR part number in
`/sys/devices/system/cpu/cpu0/regs/identification/midr_el1` (or `/proc/cpuinfo`):

| Module | Part | Core | Flags | Features used |
| --- | --- | --- | --- | --- |
| `graviton2` | 0xd0c | Neoverse N1 | `-mcpu=neoverse-n1` | dotprod, fp16 |
| `graviton3` | 0xd40 | Neoverse V1 | `-mcpu=neoverse-v1+i8mm+nosve` | dotprod, fp16, i8mm |
| `graviton4` | 0xd4f | Neoverse V2 | `-mcpu=neoverse-v2+i8mm+nosve` | dotprod, fp16, i8mm |
| `graviton5` | 0xd84 | Neoverse V3 | `-march=armv8.6-a+i8mm+nosve -mtune=neoverse-v2` | dotprod, fp16, i8mm |

A matching part outranks every generic tier. Any other CPU, and a Graviton missing one of a
module's features, gets the best generic tier instead, so no host loads code it cannot run.

The part numbers and features were read from `c6g`, `c7g`, `c8g` and `c9g` instances in
us-east-1. None of them reports SME, Graviton5 included.

Graviton5 is tuned for Neoverse V2 because GCC 14, the newest in the Lambda base image's
repositories, has no `neoverse-v3`; GCC 15 does. To change it, set `GGML_CPU_GRAVITON5_TUNE`.

How fast the `--lambda` build already is on each generation, before any module (ECS on EC2,
E4B, 6 threads, median ms):

| Request | Graviton2 | Graviton3 | Graviton4 | Graviton5 |
| --- | --- | --- | --- | --- |
| ~40 tokens | 5,712 | 3,769 | 3,061 | 2,279 |
| ~1.1k tokens | 37,776 | 24,921 | 20,432 | 15,073 |
| ~4.1k tokens | 180,136 | 121,160 | 101,129 | 71,977 |
| one image | 18,115 | 12,043 | 9,811 | 7,212 |

Graviton5 is about 2.5× Graviton2 on the same build. Lambda gave Graviton2, 3 and 4 hosts in
us-east-1, and Fargate gave Graviton3.

The variants build needs GCC 14 (`gcc14`, `gcc14-c++` from the Amazon Linux 2023 repositories).
The base image's libstdc++ is GCC 14's, so what it builds runs on the unchanged base image.

## Why the Graviton modules are `+nosve`

At llama.cpp `c56be86`, ggml's SVE paths are slower than its NEON ones on every Graviton,
and much slower on long requests.

Measured on ECS, E4B (Q8_0), 6 threads, one module loaded at a time, median ms
(`deploy/lambda/bench/ecs_bench.py`; Fargate gave Graviton3 and agreed with EC2 to within 4%):

| Generation | Request | `--lambda` (static) | NEON + i8mm module | SVE module |
| --- | --- | --- | --- | --- |
| Graviton3 (`c7g`) | ~40 tokens | 3,769 | 3,269 (1.15×) | 4,419 (0.85×) |
| | ~1.1k tokens | 24,921 | 22,022 (1.13×) | 64,001 (0.39×) |
| | ~4.1k tokens | 121,160 | 111,543 (1.09×) | 582,924 (0.21×) |
| | one image | 12,043 | 10,505 (1.15×) | 27,410 (0.44×) |
| Graviton4 (`c8g`) | ~40 tokens | 3,061 | 2,655 (1.15×) | 3,502 (0.87×) |
| | ~1.1k tokens | 20,432 | 17,993 (1.14×) | 47,560 (0.43×) |
| | ~4.1k tokens | 101,129 | 89,338 (1.13×) | 424,028 (0.24×) |
| | one image | 9,811 | 8,832 (1.11×) | 23,019 (0.43×) |
| Graviton5 (`c9g`) | ~40 tokens | 2,279 | 1,961 (1.16×) | 2,686 (0.85×) |
| | ~1.1k tokens | 15,073 | 12,923 (1.17×) | 38,095 (0.40×) |
| | ~4.1k tokens | 71,977 | 63,913 (1.13×) | 346,899 (0.21×) |
| | one image | 7,212 | 6,371 (1.13×) | 18,367 (0.39×) |

What these results show:

- **SVE costs speed everywhere.** That includes Graviton3, whose SVE is 256 bits wide.
  Graviton4 and 5 implement SVE at 128 bits, NEON's width, so there SVE never adds width.
- **The cost grows with context length.** At this pin, several of ggml's SVE branches fall back
  to scalar loops (`ggml_vec_mad_f32_unroll` and `ggml_vec_mad1_f32` in `ggml-cpu/vec.h` are
  marked "TODO: Write SVE code").
- **i8mm is the gain, and it doesn't need SVE.** i8mm is the Q8_0 matrix multiply (`smmla`), so
  the modules keep it and leave SVE out.
- **The modules answer like the build that ships today.** On E4B's `golden_vectors.jsonl`
  (1,032 text questions; `deploy/lambda/bench/conformance_container.py`), `graviton4` and
  `graviton5` scored exactly what `--lambda` scored on the same host: 99.71% top-1, mean |dp|
  0.00376, max 0.0508. That is inside `tests/core_conformance.py`'s bound (top-1 ≥ 98%, mean
  |dp| ≤ 0.02).

- **SVE's accuracy is fine on the real checkpoint.** On Graviton3, `graviton3_sve` scored 99.61%
  top-1 and 0.00384 mean |dp|, and `armv8.2_3` (SVE without i8mm) 99.81% and 0.00361, against
  `--lambda`'s 99.71% and 0.00374. SVE is left out for speed alone.

`build_native.sh` fails the build if a Graviton3, 4 or 5 module contains an SVE instruction.

### The image build's tiny-model check

`shim_check.py` holds the shim to transformers on a tiny random Gemma 4, with each golden
question within 0.01. On that check, every module faster than Graviton2's instruction set fails
on one question, `team`, a near-tie:

| Module | Features | urgent | team | severity |
| --- | --- | --- | --- | --- |
| `armv8.2_2` | NEON, dotprod + fp16 (`--lambda`'s ISA) | 0.0048 | 0.0068 | 0.0006 |
| `armv8.2_3` | + SVE, no i8mm | 0.0079 | 0.0116 | 0.0040 |
| `graviton4` | + i8mm, no SVE | 0.0079 | 0.0116 | 0.0040 |
| x86 `sandybridge` | AVX, no FMA | — | 0.0106 | — |

The failures don't reflect a real accuracy loss. On the real checkpoint, `graviton4` matches
`--lambda` exactly (above). The tolerance is tighter than the normal spread between correct
kernels on that question, and `--lambda` itself uses 0.0068 of it.

The check also misses a real loss. KleidiAI's answers on the tiny model were identical to the
plain modules', while on E4B it gets 11 more questions wrong ([KleidiAI](#kleidiai--c-kleidiaitrue)).
My guess, not verified, is that the tiny model's matrices are too small for KleidiAI's kernels
to be used. So the tiny model rejects sound kernels and can pass lossy ones.

So a variants build's modules are held to 0.015 on this check (`check_variants.sh` passes
`shim_check.py --golden-dp 0.015`), and the `--lambda` build stays at 0.01. Because the tiny model
is a weak signal for these kernels, the variants build also has a release check on the real
checkpoint ([Release check](#release-check)).

### Preconditions for turning SVE on

`GRAVITON_SVE=1` (`-c graviton_sve=true`) adds `graviton3_sve`, `graviton4_sve`, `graviton5_sve`
and `graviton5_sme`. They take precedence over the NEON modules, for measuring only.

Make SVE the default only when all of the following hold at the llama.cpp pin it would ship with.
Re-check them after each bump of `LLAMA_CPP_REV`, because ggml's SVE kernels change upstream.

1. **SVE is faster than the NEON + i8mm modules where it would run.** On each generation it
   would load on (`c7g`, `c8g`, `c9g`), on Lambda and on ECS, at every request size in the bench
   mix, by enough to matter. Use `graviton_bench.py` and `ecs_bench.py` with the same requests
   and the same thread count. At `c56be86` it is 0.21–0.87× of them (above).
2. **The shipped checkpoints agree at least as well.** On E2B and E4B, against their
   `golden_vectors.jsonl`, the SVE modules must score at least the NEON modules' top-1 and no
   larger a mean |dp| on the same host (`conformance_container.py --module`).
3. **Every SVE module passes `scripts/check_variants.sh` on every generation that loads it.**
   - The image build's own check covers only the modules its build host loads.
   - CodeBuild's arm64 hosts are Graviton2, so they load none of the SVE modules.

SME (`graviton5_sme`) also needs a Graviton that reports SME.

## KleidiAI (`-c kleidiai=true`)

KleidiAI is Arm's set of hand-tuned matrix-multiply kernels, which ggml can call in place of its
own. With the Graviton modules it is the fastest build on every generation, Graviton2 included.
It is also measurably less accurate. It is off by default.

Speed, measured on ECS (E4B, 6 threads; the ranges cover the ~40-token, ~1.1k-token, ~4.1k-token
and one-image requests):

| Generation | Graviton module | Graviton module + KleidiAI |
| --- | --- | --- |
| Graviton2 | 0.99–1.01× | 1.18–1.34× |
| Graviton3 | 1.09–1.15× | 1.38–1.70× (Fargate 1.41–1.73×) |
| Graviton4 | 1.11–1.15× | 1.28–1.49× |
| Graviton5 | 1.13–1.17× | 1.32–1.53× |

Accuracy, on E4B's golden vectors (1,032 text questions). KleidiAI scored the same on every
generation:

| Build | Top-1 | Wrong | Mean \|dp\| | Max \|dp\| |
| --- | --- | --- | --- | --- |
| `--lambda`, and the Graviton4/5 modules | 99.71% | 3 | 0.00376 | 0.0508 |
| `graviton2` module | 99.61% | 4 | 0.00354 | 0.0423 |
| any Graviton module + KleidiAI | 98.64% | 14 | 0.01079 | 0.1304 |

What these results show:

- **KleidiAI conforms, narrowly.** It is inside `tests/core_conformance.py`'s bound (top-1 ≥
  98%, mean |dp| ≤ 0.02), but 0.6 points above the top-1 floor.
- **It costs real accuracy.** It gets 11 more top answers wrong out of 1,032, its mean error is
  nearly 3× the static build's, and one question moves by 0.13.
- **The cause is KleidiAI's own kernels.** The numbers are identical on every generation, so
  they don't come from the Graviton module underneath.

Why, from `ggml-cpu/kleidiai/kleidiai.cpp` at `c56be86`:

- **Q8_0 and Q6_K weights are quantized a second time.** KleidiAI dequantizes each weight row,
  then quantizes it again to int8 with one scale for the whole row (`max|w| / 127`; its
  `qsi8cx` format). That throws away Q8_0's scale per 32 values.
- **Activations are quantized per row too (`qai8dx`),** where ggml uses 32-value blocks.
- **Q4_0 is KleidiAI's native format.** Its Q4_0 repack (`qsi4c32pscalef16`) keeps the 32-value
  blocks and their f16 scales, and its activations are quantized in 32-value blocks (`qsi8d32`),
  as ggml's are. So KleidiAI on Q4_0 should answer like ggml on Q4_0.
- **Q4_K is re-packed into 32-value blocks with f16 scales.** Its per-block minimums may not
  carry over exactly, so measure it rather than assume.

This is a different problem from SVE's:

| | SVE modules | KleidiAI |
| --- | --- | --- |
| E4B golden vectors | same as `--lambda` | 11 more wrong, ~3× the mean error |
| Tiny-model check | fails (0.0116), like other sound kernels | passes |
| Cause | normal rounding spread between correct kernels | Q8_0 weights re-quantized per row |
| Change from `--lambda` on the bench requests | at most 0.005–0.007 | up to 0.046 |

On Lambda (us-east-1, E4B, warm requests, median), the speedups match ECS's:

| Host Lambda gave | `cpu=variants` | `cpu=variants`, `kleidiai=true` |
| --- | --- | --- |
| Graviton2 (most hosts) | 0.99–1.01× | 1.19–1.36× |
| Graviton3 | 1.08–1.12× | 1.39–1.67× |
| Graviton4 | 1.12–1.16× | 1.34–1.56× |

Each Lambda host loaded its own Graviton module (`cpu_backend` in the function's log). Lambda
gave no Graviton5 hosts in these runs.

KleidiAI also costs cold-start time. It re-quantizes every Q8_0 weight matrix when the model
loads. The median model load of a new environment was 107 s with KleidiAI, against 79 s for
`--lambda` and 73 s for `cpu=variants`.

Turn it on for a deployment only where that accuracy cost is acceptable for its questions. On
Lambda, KleidiAI with the earlier SVE modules was slower on Graviton3 and 4. Use it only with the
default `+nosve` modules.

## Decoder quantization (`-c decoder_quant=`)

The build quantizes the decoder to Q8_0 unless `-c decoder_quant=` picks another type. It
accepts `Q8_0`, `Q6_K`, `Q4_K_M` and `Q4_0`. The vision tower stays Q8_0.

What llama.cpp offers at `c56be86`, and which formats have fast Arm kernels:

| Types | ggml's Arm kernels | KleidiAI |
| --- | --- | --- |
| F32, BF16, F16 | plain | none on Graviton (its f16/bf16 kernels need SME) |
| Q8_0 | repacked, i8mm/dotprod GEMM | re-quantized per row: lossy |
| Q6_K | repacked | re-quantized per row: lossy |
| Q5_K_M, Q5_K_S | repacked | none |
| Q4_K_M, Q4_K_S | repacked | re-packed into 32-value blocks |
| Q4_0 | repacked | native, no re-quantization |
| IQ4_NL | repacked | none |
| Q5_0, Q5_1, Q4_1, IQ4_XS, Q3_K_*, IQ3_*, Q2_K*, IQ2_*, IQ1_*, TQ1_0, TQ2_0, Q1_0, Q2_0, MXFP4 | some repacked (Q2_K, Q1_0, MXFP4) | none |

Limits on what can ship:

- **Lambda's image limit is 10 GB.** E4B's files are 8.2 GB at Q8_0, so F16 and BF16
  (about 16 GB) can't ship on Lambda. They can on ECS, but the build refuses them today.
- **Lambda's memory is 10,240 MB.** The build refuses a model whose resident weights exceed
  6.0 GB.
- **Formats below 4 bits** are unlikely to hold `tests/core_conformance.py`'s bound. They're
  unmeasured.

Measured on E4B's golden vectors (1,032 text questions), each with and without KleidiAI:

| Decoder | Source | Graviton module | Graviton module + KleidiAI | Conforms |
| --- | --- | --- | --- | --- |
| Q8_0 (default) | BF16 | 99.71%, 0.00376 | 98.64%, 0.01079 | yes, both |
| Q4_0 | BF16 | 97.09%, 0.03573 | 97.09%, 0.03575 | no |
| Q4_K_M | BF16 | 96.03%, 0.03147 | 96.03%, 0.03154 | no |
| Q6_K | BF16 | 99.22%, 0.00815 | 98.35%, 0.01229 | yes, both |
| Q4_K_M | Q8_0, re-quantized | 96.12%, 0.03617 | 96.12%, 0.03597 | no |
| Q4_0 | Q8_0, re-quantized | 97.38%, 0.03363 | 97.38%, 0.03363 | no |
| Q6_K | Q8_0, re-quantized | 98.93%, 0.01096 | 99.22%, 0.01316 | yes, both |

Each cell is top-1, then mean |dp|. The repo's bar (`tests/core_conformance.py`) is top-1 ≥ 98%
and mean |dp| ≤ 0.02.

What these results show:

- **KleidiAI adds no loss on the 4-bit formats.** That fits the code above: they keep
  32-value block scales.
- **The 4-bit formats lose too much on their own.** Their loss is about 3× what KleidiAI adds to
  Q8_0.
- **Q6_K conforms, with or without KleidiAI.** From BF16, plain Q6_K is between Q8_0 and
  Q8_0 + KleidiAI: 5 more wrong of 1,032 than Q8_0, and about 2× its mean error, on a 6.2 GB
  decoder against Q8_0's 8.0 GB.
- **KleidiAI on Q6_K is the least accurate combination that conforms.** Its per-row
  re-quantization adds to Q6_K's own loss.
- **KleidiAI barely changes Q6_K.** Re-quantizing 6-bit weights to per-row int8 loses little
  relative to the 6-bit format itself.

The re-quantized rows came from the shipped Q8_0, so they overstate the format's loss a
little, though Q4_K_M from BF16 scored no better than from Q8_0.

Q6_K's speed, measured on ECS (E4B, 6 threads), against the Q8_0 `--lambda` build:

| Generation | Q8_0 module | Q8_0 + KleidiAI | Q6_K module | Q6_K + KleidiAI |
| --- | --- | --- | --- | --- |
| Graviton2 | 0.99–1.01× | 1.18–1.34× | 0.72–0.81× | 1.18–1.34× |
| Graviton3 | 1.09–1.15× | 1.38–1.70× | 0.79–0.91× | 1.37–1.70× |
| Graviton4 | 1.11–1.15× | 1.28–1.49× | 0.79–0.88× | 1.28–1.49× |
| Graviton5 | 1.13–1.17× | 1.32–1.53× | 0.80–0.90× | 1.32–1.54× |

Fargate gave Graviton3 and agreed with EC2 to within 3%.

What these results show:

- **Plain Q6_K is slower than the shipped build.** ggml's Q6_K kernels are slower than its Q8_0
  i8mm path, so the smaller file doesn't buy speed.
- **Q6_K + KleidiAI runs exactly as fast as Q8_0 + KleidiAI.** KleidiAI re-quantizes both to
  the same per-row int8, so they do the same work, but Q6_K is slightly less accurate.
- **Q6_K saves only image size,** 6.2 GB against 8.0 GB for E4B's decoder.

**Q8_0 stays the default.** It is the most accurate format and, with the Graviton modules, the
fastest at that accuracy. `-c decoder_quant=Q6_K` is for a deployment that needs the smaller
image more than the speed. Pair it with `kleidiai=true`, because without KleidiAI it's slower
than Q8_0.

## Release check

Run this whenever `LLAMA_CPP_REV` or `patches/` change, and before changing a default here. The
image build can't run it, because CodeBuild's arm64 hosts are Graviton2 and load none of the
Graviton3–5 modules.

1. **Build a measurement image.** Build `deploy/lambda/decision/Dockerfile` with
   `NATIVE=--lambda-variants`, then the `--lambda` build. Push both to an ECR repository the hosts
   can pull from.
2. **Run every module on every generation.** On one instance of each generation (`c6g`, `c7g`,
   `c8g`, `c9g`), run `deploy/lambda/bench/conformance_container.py` inside each image, against
   the checkpoint's `golden_vectors.jsonl`:
   - in the `--lambda` image, with no `--module`;
   - in the variants image, once per Graviton module, with `--module graviton<N>`;
   - in a `KLEIDIAI=1` image too, for the opt-in KleidiAI results.

   Write all the results to one S3 prefix with `--s3`. Accuracy doesn't depend on the thread
   count, so a large instance with `--threads 24` or more is fine.
3. **Check the results.** `deploy/lambda/bench/conformance_report.py s3://<bucket>/<prefix>`
   fails unless every module conforms (`tests/core_conformance.py`). Each module must also be no
   worse than the `--lambda` build on its host: at most 2 more wrong top answers per 1,032
   questions, and a mean |dp| at most 10% higher. KleidiAI results are reported but don't fail
   the check.

Results at `c56be86` (E4B, 1,032 questions):

| Generation | `--lambda` | Graviton module | Graviton module + KleidiAI |
| --- | --- | --- | --- |
| Graviton2 | 99.71%, 0.00376 | 99.61%, 0.00354 | 98.64%, 0.01079 |
| Graviton3 | 99.71%, 0.00374 | 99.71%, 0.00374 | — |
| Graviton4 | 99.71%, 0.00376 | 99.71%, 0.00376 | 98.64%, 0.01079 |
| Graviton5 | 99.71%, 0.00376 | 99.71%, 0.00376 | 98.64%, 0.01079 |

Each cell is top-1, then mean |dp|.

## Checking a build

Run `scripts/check_variants.sh SRC LIBDIR` on each generation a change can reach. It runs
`shim_check.py` once per module the host loads, each module on its own, and reports the others
as skipped.

The image build's own check covers only its build host's module, so this script is the check
for the Graviton3, 4 and 5 modules.

## Measuring

`deploy/lambda/bench/` has three drivers:

- **`graviton_bench.py run`:** drives deployed Lambda functions, retiring environments between
  rounds so Lambda hands out new hosts.
- **`ecs_bench.py`:** runs the image as ECS tasks, pinned to each generation's instance type on
  EC2, and on Fargate.
- **`bench_container.py`:** runs inside the image on either platform.

`graviton_bench.py report` tables all of them by platform, generation and build. Every row
records the host's `cpu_part`, its features and the module it loaded (`cpu_backend`), which the
function also logs on every request.
