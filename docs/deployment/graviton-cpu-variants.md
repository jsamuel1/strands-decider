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

ggml's SVE kernels move the function's answers further from the float32 reference than its NEON
ones do.

Measured on Graviton4 (`m8g`) at llama.cpp `c56be86`, with `shim_check.py`'s tiny Gemma 4 and
one module loaded at a time (`scripts/check_variants.sh`). Each cell is the largest probability
difference from transformers on that golden question; the tolerance is 0.01:

| Module | Adds | urgent | team | severity | Result |
| --- | --- | --- | --- | --- | --- |
| `armv8.2_2` | NEON, dotprod+fp16 (what `--lambda` builds) | 0.0048 | 0.0068 | 0.0006 | passes |
| `armv8.2_3` | + SVE | 0.0079 | 0.0116 | 0.0040 | fails |
| `armv8.6_1` | + i8mm | 0.0079 | 0.0116 | 0.0040 | fails |
| `armv8.6_2` | + SVE2 | 0.0079 | 0.0116 | 0.0040 | fails |
| `graviton4` with SVE | + neoverse-v2 tuning | 0.0079 | 0.0116 | 0.0040 | fails |

What these results show:

- **SVE alone causes the loss.** i8mm, SVE2 and the core's tuning add none.
- **i8mm doesn't need SVE.** i8mm is the Q8_0 matrix multiply (`smmla`), so the modules keep it.
- **SVE adds no width on Graviton4 and 5.** Both implement SVE at 128 bits, NEON's width. Only
  Graviton3's SVE is 256 bits.
- **KleidiAI didn't change it.** With `KLEIDIAI=1`, the answers on this model were identical.

This test has a thin margin, and SVE isn't the only kernel path that crosses it:

- Baseline already uses 0.0068 of the 0.01 tolerance on `team`, a near-tie on a random tiny
  model.
- On x86, ggml's `sandybridge` module (AVX without FMA or F16C) also misses `team`, at 0.0106,
  while every other x86 module passes.

So the failure says the SVE path is less precise than the NEON one, not that it is wrong. On the
real checkpoints, precondition 2 below is what decides it.

`build_native.sh` fails the build if a Graviton3, 4 or 5 module contains an SVE instruction.

### Preconditions for turning SVE on

`GRAVITON_SVE=1` (`-c graviton_sve=true`) adds `graviton3_sve`, `graviton4_sve`, `graviton5_sve`
and `graviton5_sme`. They take precedence over the NEON modules, for measuring.

Make SVE the default only when all of the following hold at the llama.cpp pin it would ship with.
Re-check them after each bump of `LLAMA_CPP_REV`, because ggml's SVE kernels change upstream.

1. **Every SVE module passes on every generation that loads it.** `scripts/check_variants.sh`
   must pass on `c7g` (256-bit SVE), `c8g` and `c9g` (128-bit).
   - The image build's own `shim_check.py` tests only the module its build host loads.
   - CodeBuild's arm64 hosts are Graviton2, so they load none of the SVE modules.
2. **The shipped checkpoints agree at least as well.** On E2B and E4B, against their
   `golden_vectors.jsonl`, the SVE modules must match the reference at least as well as the NEON
   ones on the same host:
   - as many top answers as the reference's;
   - no larger a mean probability difference. The [Lambda README](https://github.com/strands-labs/strands-decider/tree/main/deploy/lambda)'s
     v2 measurements are E4B 398 of 400 and 0.0047.
3. **SVE is faster where it would run.** `deploy/lambda/bench/` must show SVE beating the NEON
   modules on that generation, on Lambda and on ECS, by enough to matter. Use
   `graviton_bench.py` and `ecs_bench.py` with the same requests and the same thread count.

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
