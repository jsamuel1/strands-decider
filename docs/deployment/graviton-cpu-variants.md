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
