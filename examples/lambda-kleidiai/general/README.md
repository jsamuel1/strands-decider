# G: general (non-Lambda) performance ladder

Which Lambda-measured winners (R, `lkai/ce4d-recommendations`) should become general defaults of the decider
llama.cpp/decshim engine (EC2, container, laptop)? This ladder measures each one off Lambda, on fixed hosts.

**Every number here is FIXED HOST, NOT LAMBDA LATENCY.**

## Verdicts

| change | verdict | why |
|---|---|---|
| E8 six-thread KleidiAI repack (`GGML_KLEIDIAI_REPACK_THREADS`) | **general default** | Load −5.65 s on arm64 local-file (−5.67 s memfd). Nothing else changes. Answers are bit-identical. Inert on x86_64. Set the value to the engine's thread count. |
| E9 pre-packed GGUF + loader | **general opt-in** (arm64 KleidiAI builds only) | Load −4.83 s, but E8 is 0.83 s faster in the same rounds. x86_64 refuses the file, as designed. **Hazard:** an unpatched engine loads it without error and returns NaN / wrong answers on both arches. |
| E7 skip tied output head | **general opt-in** (decshim-only engines) | Load −2.03 s, peak RSS −488 MB on arm64. x86_64 RSS −575 MB. Answers bit-identical. Aborts if logits are requested, so it is not a llama.cpp-wide default. |
| E10 `n_ctx` 4096 (model window) | **general default** | Peak RSS −306 MB arm64, −320 MB x86_64. Small load/forward gains; nothing worse. Answers bit-identical, including the full 4,096-token forward. Derive it from the head config's `max_length`. |
| E5 no-vocab GGUF | **general opt-in** (engines that tokenize outside llama.cpp) | Load −0.50 s, RSS −83 MB arm64; RSS −55 MB x86_64. The file has no usable vocabulary for llama.cpp tokenization. Ship it beside the full GGUF. |
| E6 complete bytecode | **general default** (packaged builds) | Imports −0.288 s arm64, −0.209 s x86_64 (read-only package). Nothing else changes. On a writable FS CPython writes the same `.pyc` after first start. |

No rung degraded any measured non-Lambda path. `summarize.py --verdicts` refuses a "general default" for any rung
with a **WORSE** cell or changed answers (operator rule, 2026-10-22). E8 vs E9, same rounds: E9 loads **0.835 s**
slower (local) and **0.832 s** slower (memfd) than E8 (thresholds 0.066 / 0.028 s), with no other difference.
Both remove the same repack cost, and E8 needs no new file format. Prefer E8.

## Method

- **Baseline** P1 `feat/lambda-kleidiai-baseline` @ `70eab7be69f86ddeb8c64158d94df344b5180098`. This branch is cut from it.
- **One variable per rung, each against P1 alone (not cumulative).** Rung commits were verified with
  `git ls-remote https://github.com/jsamuel1/strands-decider 'refs/heads/*'`:
  E8 `lkai/5a61-parallel-repack` bef9c89, E9 `lkai/f1ee-prepacked` 5dce29d, E7 `lkai/f768-skip-output-repack`
  0af73df, E10 `lkai/fbb1-kv-size` 60f4586, E5 `lkai/3259-mmap-format` 06b1831, E6 `lkai/fada-hotspot` e989e39.
- **Artifacts:** each rung's published zip / GGUF, checked against its published sha256 (`inputs.json`, `stage.sh`).
  Rebuilt shared libraries are not bit-reproducible, so reusing the measured bytes is what keeps "one variable" true.
- **Arms** (`make_arms.py`): `p1`; `e8` (E8 package, switch=6) and dormant `e8off` (same package, unset); `e9`
  (E9 package + E9 GGUF) and dormant `e9p1` (E9 package + P1 GGUF); `e7` (switch=1) and dormant `e7off`; `e10`
  (P1 package, `--n-ctx 4096`, the vendored E10 shim logic); `e5` (P1 package + E5 GGUF); `e6` (P1 package with
  complete bytecode).
- **Paths:** local-file mmap GGUF (no memfd, no S3) is the primary general path. memfd is kept for continuity with
  Lambda. The page cache is warm, since the hosts had no sudo to drop caches.
- **Per load:** a new container (P1's pinned Lambda image, `--network none`, package mounted read-only at
  `/var/task`), `--cpuset-cpus 10-15`, 6 threads. P1 env: `DEC_KV=f32 DEC_UBATCH=512 N_SEQ=17 OPENBLAS_NUM_THREADS=1`.
  Each load does: model load, first forward, first request, then P1's five requests × 3 warm passes and a
  4,096-token forward × 2. Peak RSS is `VmHWM`.
- **Interleaving:** every round runs every cell (arm × path) in a cyclic rotation of one order. Over the rounds this
  is a Latin square, so no arm owns a position. Comparisons are same run, same round, same path and same CPU identity.
- **Pre-registered rule** (`summarize.py`), fixed before any result was read:
  1. Compute the per-round paired delta, `arm − p1`.
  2. An effect is real only if all paired deltas share a sign and |median| exceeds p1's own per-round range.
     For peak RSS, which repeats to a page, the bar is also at least 1% of p1.
  3. Otherwise the cell is "no change". Fewer than 3 rounds is "unassessed".
- **Bit-identity:** every answer is compared as canonical JSON text with the same run's P1. On arm64 it is also
  compared with P1's published 0xd40 reference (`baseline/reference/outputs.json`). There is no x86_64 reference.

## Hosts (recorded per load in `host_before`)

| | arm64 (primary) | x86_64 |
|---|---|---|
| host | DevSpace ds-lkai-p1-rs5ghbzv (H1's fixed host) | developer cloud desktop |
| CPU identity | part 0xd40 (as printed; no generation inferred) | GenuineIntel family 6 model 173 stepping 1, "Xeon 6975P-C" |
| CPUs / used | 16 / 10-15 (6 threads) | 32 vCPU / 10-15 (SMT siblings 26-31 not reserved) |
| kernel | 6.1.188-233.393.amzn2023.aarch64 | 6.1.188-233.393.amzn2023.x86_64 |
| cgroup cpu.max | max 100000 (no quota) | max 100000 |
| governor | not exposed (cpufreq absent) | not exposed |
| ggml CPU variant | armv8.2_3 + armv8.6_1 (KleidiAI) | sapphirerapids (KleidiAI cannot be built for x86) |
| other load (1-min loadavg) | 3.3–6.9 | **15–42: co-tenanted**, so x86 timing bars are wide |
| docker | 25.0.8 | 25.0.14 |
| run window (UTC) | 2026-10-25 09:05:52 → 10:14:23 (80 loads, 0 failed) | 09:19:01 → 10:05:53 (40 loads; 5 failed = E9 refusals by design); E6 10:17:19 → 10:29:41 (10 loads) |

The x86_64 build (`x86_build.sh`) is P1's `container_build.sh` with two changes, recorded in
`container_build_x86.diff`: `-DGGML_CPU_KLEIDIAI=OFF`, and no quantize step. It uses the same pinned image, by its
amd64 digest `sha256:586cea70…f970`, which is checked against the manifest list. Each rung's llama.cpp patches are
applied unchanged. On x86 the E8 and E9 `kleidiai.cpp` changes are not compiled in.

## Results (median p1 → arm (paired median Δ; ±rule threshold) outcome; n = rounds)

"load" = `dec_load_total` (model load + context init); "load+first request" = P1's `model_ready`.

| rung | CPU identity | path | n | load s | load+first request s | first forward s | warm forward s | 4,096-token forward s | peak RSS MB | imports s | answers vs P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e8 | 0xd40 | arm64 local | 5 | 7.934 → 2.279 (-5.649; ±0.056) **better** | 9.378 → 3.792 (-5.585; ±0.086) **better** | 0.229 → 0.228 (-0.001; ±0.003) no change | 0.565 → 0.564 (-0.003; ±0.006) no change | 15.790 → 15.822 (+0.015; ±0.257) no change | 4879 → 4880 (+0; ±49) no change | 0.459 → 0.457 (+0.001; ±0.014) no change | 50/50 req, 10/10 fwd |
| e8 | 0xd40 | arm64 memfd | 5 | 8.009 → 2.342 (-5.666; ±0.089) **better** | 10.649 → 5.016 (-5.637; ±0.162) **better** | 0.230 → 0.229 (-0.001; ±0.005) no change | 0.565 → 0.565 (-0.001; ±0.005) no change | 15.915 → 15.953 (+0.036; ±0.116) no change | 4863 → 4880 (+17; ±49) no change | 0.456 → 0.458 (+0.001; ±0.010) no change | 50/50 req, 10/10 fwd |
| e8 | GenuineIntel 6/173/1 | x86_64 local | 5 | 1.119 → 1.074 (-0.062; ±0.729) no change | 3.256 → 3.339 (+0.068; ±0.771) no change | 0.608 → 0.541 (-0.043; ±0.304) no change | 1.092 → 1.085 (-0.059; ±0.147) no change | 21.825 → 21.545 (-0.134; ±8.906) no change | 5183 → 5182 (-0; ±52) no change | 0.362 → 0.422 (+0.046; ±0.109) no change | 25/25 req, 5/5 fwd |
| e9 | 0xd40 | arm64 local | 5 | 7.934 → 3.105 (-4.827; ±0.056) **better** | 9.378 → 4.546 (-4.832; ±0.086) **better** | 0.229 → 0.229 (-0.000; ±0.003) no change | 0.565 → 0.564 (-0.002; ±0.006) no change | 15.790 → 15.796 (-0.002; ±0.257) no change | 4879 → 4854 (-26; ±49) no change | 0.459 → 0.458 (-0.002; ±0.014) no change | 50/50 req, 10/10 fwd |
| e9 | 0xd40 | arm64 memfd | 5 | 8.009 → 3.170 (-4.834; ±0.089) **better** | 10.649 → 5.779 (-4.851; ±0.162) **better** | 0.230 → 0.230 (-0.001; ±0.005) no change | 0.565 → 0.564 (-0.000; ±0.005) no change | 15.915 → 15.894 (-0.034; ±0.116) no change | 4863 → 4854 (-9; ±49) no change | 0.456 → 0.457 (+0.002; ±0.010) no change | 50/50 req, 10/10 fwd |
| e9 | GenuineIntel 6/173/1 | x86_64 local | 5 | refused at load, 5/5 (see below) | | | | | | | — |
| e7 | 0xd40 | arm64 local | 5 | 7.934 → 5.901 (-2.031; ±0.056) **better** | 9.378 → 7.351 (-2.012; ±0.086) **better** | 0.229 → 0.230 (+0.003; ±0.003) no change | 0.565 → 0.565 (-0.001; ±0.006) no change | 15.790 → 15.788 (-0.019; ±0.257) no change | 4879 → 4392 (-488; ±49) **better** | 0.459 → 0.459 (+0.001; ±0.014) no change | 50/50 req, 10/10 fwd |
| e7 | 0xd40 | arm64 memfd | 5 | 8.009 → 5.980 (-2.028; ±0.089) **better** | 10.649 → 8.587 (-2.065; ±0.162) **better** | 0.230 → 0.229 (-0.001; ±0.005) no change | 0.565 → 0.565 (+0.000; ±0.005) no change | 15.915 → 15.883 (-0.022; ±0.116) no change | 4863 → 4392 (-471; ±49) **better** | 0.456 → 0.457 (+0.001; ±0.010) no change | 50/50 req, 10/10 fwd |
| e7 | GenuineIntel 6/173/1 | x86_64 local | 5 | 1.119 → 0.960 (-0.085; ±0.729) no change | 3.256 → 2.990 (-0.420; ±0.771) no change | 0.608 → 0.465 (-0.118; ±0.304) no change | 1.092 → 1.043 (-0.055; ±0.147) no change | 21.825 → 20.792 (-1.252; ±8.906) no change | 5183 → 4606 (-575; ±52) **better** | 0.362 → 0.422 (+0.019; ±0.109) no change | 25/25 req, 5/5 fwd |
| e10 | 0xd40 | arm64 local | 5 | 7.934 → 7.877 (-0.062; ±0.056) **better** | 9.378 → 9.286 (-0.108; ±0.086) **better** | 0.229 → 0.224 (-0.004; ±0.003) **better** | 0.565 → 0.559 (-0.006; ±0.006) no change | 15.790 → 15.726 (-0.064; ±0.257) no change | 4879 → 4573 (-306; ±49) **better** | 0.459 → 0.455 (-0.004; ±0.014) no change | 50/50 req, 10/10 fwd |
| e10 | 0xd40 | arm64 memfd | 5 | 8.009 → 7.955 (-0.043; ±0.089) no change | 10.649 → 10.568 (-0.036; ±0.162) no change | 0.230 → 0.225 (-0.004; ±0.005) no change | 0.565 → 0.558 (-0.007; ±0.005) **better** | 15.915 → 15.719 (-0.162; ±0.116) **better** | 4863 → 4573 (-290; ±49) **better** | 0.456 → 0.457 (+0.000; ±0.010) no change | 50/50 req, 10/10 fwd |
| e10 | GenuineIntel 6/173/1 | x86_64 local | 5 | 1.119 → 0.909 (-0.209; ±0.729) no change | 3.256 → 2.780 (-0.277; ±0.771) no change | 0.608 → 0.437 (-0.049; ±0.304) no change | 1.092 → 1.017 (-0.074; ±0.147) no change | 21.825 → 21.002 (-0.387; ±8.906) no change | 5183 → 4863 (-320; ±52) **better** | 0.362 → 0.358 (-0.008; ±0.109) no change | 25/25 req, 5/5 fwd |
| e5 | 0xd40 | arm64 local | 5 | 7.934 → 7.436 (-0.504; ±0.056) **better** | 9.378 → 8.873 (-0.498; ±0.086) **better** | 0.229 → 0.230 (+0.002; ±0.003) no change | 0.565 → 0.565 (+0.000; ±0.006) no change | 15.790 → 15.798 (-0.020; ±0.257) no change | 4879 → 4796 (-83; ±49) **better** | 0.459 → 0.454 (-0.005; ±0.014) no change | 50/50 req, 10/10 fwd |
| e5 | 0xd40 | arm64 memfd | 5 | 8.009 → 7.513 (-0.487; ±0.089) **better** | 10.649 → 10.124 (-0.487; ±0.162) **better** | 0.230 → 0.230 (-0.001; ±0.005) no change | 0.565 → 0.563 (-0.003; ±0.005) no change | 15.915 → 15.892 (-0.013; ±0.116) no change | 4863 → 4796 (-67; ±49) **better** | 0.456 → 0.458 (+0.001; ±0.010) no change | 50/50 req, 10/10 fwd |
| e5 | GenuineIntel 6/173/1 | x86_64 local | 5 | 1.119 → 0.619 (-0.347; ±0.729) no change | 3.256 → 2.758 (-0.499; ±0.771) no change | 0.608 → 0.456 (-0.048; ±0.304) no change | 1.092 → 1.066 (-0.001; ±0.147) no change | 21.825 → 21.597 (-1.049; ±8.906) no change | 5183 → 5128 (-55; ±52) **better** | 0.362 → 0.387 (+0.011; ±0.109) no change | 25/25 req, 5/5 fwd |
| e6 | 0xd40 | arm64 local | 5 | 7.934 → 7.940 (+0.019; ±0.056) no change | 9.378 → 9.101 (-0.276; ±0.086) **better** | 0.229 → 0.230 (+0.001; ±0.003) no change | 0.565 → 0.564 (-0.001; ±0.006) no change | 15.790 → 15.794 (-0.017; ±0.257) no change | 4879 → 4878 (-1; ±49) no change | 0.459 → 0.170 (-0.288; ±0.014) **better** | 25/25 req, 5/5 fwd |
| e6 | GenuineIntel 6/173/1 | x86_64 local | 5 | 0.957 → 1.051 (+0.091; ±0.184) no change | 2.895 → 2.995 (+0.065; ±0.206) no change | 0.466 → 0.456 (+0.006; ±0.175) no change | 0.975 → 1.022 (+0.052; ±5.881) no change | 19.473 → 20.406 (+0.842; ±23.698) no change | 5182 → 5181 (-0; ±52) no change | 0.344 → 0.139 (-0.209; ±0.100) **better** | 25/25 req, 5/5 fwd |

On arm64, every arm's answers are also bit-identical to P1's published 0xd40 reference: E8, E9, E7, E10 and E5
match 50/50; E6 matches 25/25 (local only). On arm64 the repeat passes within a load were identical for every arm.

**Dormant checks** (patched code, variable off; paired against p1, local path):

| arm | 0xd40 load Δ (±) | 0xd40 answers | x86_64 |
|---|---|---|---|
| e8off | +0.058 s (±0.056) "worse": 0.7%, at the rule's edge | 25/25 identical | n/a (same libs as e8) |
| e9p1 | −0.087 s (±0.056) "better" | 25/25 identical | no change, 25/25 |
| e7off | +0.007 s, no change | 25/25 identical | no change, 25/25 |

The e8off and e9p1 residues are under 0.1 s and sit just at p1's own 0.056 s range. Their signs are opposite, and
neither is the shipped configuration (E8 ships with the switch on). Both are recorded as observations, not effects
of a default.

## Findings beyond the timings

1. **E9's file is unsafe without its loader.** On both arches, P1's unpatched engine loads
   `s2b-h512-Q8_0-e9kr8.gguf` without error. Every probabilities field comes back NaN and choices are wrong: all 5
   requests DIFFER from P1's 0xd40 reference, and 4/5 differ from x86 P1. Files: `hazard/arm-p1libs-e9gguf.json`,
   `hazard/x86-p1libs-e9gguf.json`. With E9's loader on x86, the file is refused:
   `tensor 'blk.0.attn_qkv.weight' is KleidiAI-prepacked (E9) but is not in a CPU_KLEIDIAI buffer`. So E9 GGUFs
   must never sit where a generic loader can pick them up.
2. **x86_64 P1 itself returns NaN for one request.** In every arm, P1 included, `incident-review-4q` returns NaN
   probabilities (12 fields: `nan_fields` in `ladder.json`) on the auto-selected `libggml-cpu-sapphirerapids.so`.
   Removing that variant library gives finite answers (retries 0.551, freeze 0.351, alarm 0.019, capacity 0.078),
   close to arm64's (0.554 / 0.355 / 0.019 / 0.072). This predates every rung and is identical across arms, so the
   comparison is unaffected. It is still a P1 x86_64 defect (AVX-512 variant path), outside this ladder; recorded
   for follow-up. NaN also makes probe.py's per-load `answers_repeat_identical` read False on x86 (NaN ≠ NaN).
   summarize.py compares canonical JSON text instead.
3. **E8's switch value:** the shipped value 6 equals `THREADS` here. The general default should follow the
   engine's thread count. Other counts were not measured in this ladder.

## Deviations and limits

- The x86_64 host is a shared, co-tenanted desktop (loadavg 15–42), and CPUs 10-15 share SMT cores with 26-31,
  which were not reserved. x86 timing bars are therefore wide (load ±0.73 s; 4,096 forward ±8.9 s), so x86 timing
  can only rule out large regressions. Memory and answers are exact.
- E6's published package was built from a P1 zip `d148d706…` (51,941,882 B). P1's published zip is `26eb2651…`
  (51,941,898 B). The arm64 e6 package was rebuilt from `26eb2651…` with `e6/pyc_zip.py` @ e989e39 in P1's image:
  1035 compiled, 0 failed, CPython 3.12.15, magic `cb0d0d0a`. x86 e6 used `x86_e6.sh`, the same compile rule
  (UNCHECKED_HASH, `dfile=/var/task/…`), applied to the unzipped x86 package, because `pyc_zip.py` only accepts
  P1's arm64 zip.
- Page cache was warm for every load, because there was no sudo to drop caches. Cold-disk load is not measured.
- No AWS calls were made. Lambda numbers (R) are not comparable with these.

## Reproduce

```bash
# arm64 fixed host (the DevSpace); artifacts verified against inputs.json
general/stage.sh <artifact dir> <stage dir>                 # verifies hashes, builds pkg-*, e6 pyc
python3 general/make_arms.py <stage dir> <artifact dir> <stage dir>/arms.json
python3 general/run_ladder.py <stage dir>/arms.json <run dir> --rounds 5 --cpuset 10-15 --threads 6
# x86_64
general/x86_build.sh <p1.zip> <x86 dir>                     # pkg-p1/e8/e9/e7, diffs
general/x86_e6.sh <x86 dir>/pkg-p1 <x86 dir>/pkg-e6
python3 general/x86_arms.py <x86 dir> <dir with p1.gguf e9.gguf e5.gguf> <x86 dir>/arms.json
python3 general/run_ladder.py <x86 dir>/arms.json <x86 run> --rounds 5 --platform linux/amd64 --cpuset 10-15 --threads 6 --only p1,e8,e9,e9p1,e7,e7off,e10,e5
python3 general/run_ladder.py <x86 dir>/arms.json <x86 e6 run> --rounds 5 --platform linux/amd64 --cpuset 10-15 --threads 6 --only p1,e6
# analysis, operator-rule guard, P2 result files
python3 general/summarize.py <run dir> <x86 run> <x86 e6 run> --out <out> --verdicts general/verdicts.json --commit <this branch's SHA> --run-id 20261025T090459Z-general
python3 general/tables.py <out>/ladder.json
python3 results/publish_result.py <out>/p2/<rung>/results.json --validate-only   # P2 (feat/lambda-kleidiai-results)
```

`runs/` holds this ladder's raw probe files and `out/` holds the analysis; `out/p2/<rung>/results.json` are the P2
records (experiment ids `G-general-E*`, run id `20261025T090459Z-general`).

## P2 pointers

Published with `results/publish_result.py` (feat/lambda-kleidiai-results) to the P2 store and dashboard
<https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results>, all `status: improved`:

| rung | P2 run | dashboard version |
|---|---|---|
| E8 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E8-parallel-repack/20261025T090459Z-general/result.json` | 59 |
| E9 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E9-prepacked/20261025T090459Z-general/result.json` | 60 |
| E7 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E7-skip-tied-output/20261025T090459Z-general/result.json` | 61 |
| E10 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E10-kv-size/20261025T090459Z-general/result.json` | 62 |
| E5 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E5-no-vocab-gguf/20261025T090459Z-general/result.json` | 63 |
| E6 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E6-bytecode/20261025T090459Z-general/result.json` | 64 |
