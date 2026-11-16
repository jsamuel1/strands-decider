# G: general (non-Lambda) performance ladder

Which Lambda-measured winners (R, `lkai/ce4d-recommendations`) should become general defaults of the decider
llama.cpp/decshim engine (EC2, container, laptop)? This ladder measures each one off Lambda, on fixed hosts.

**Every number here is FIXED HOST, NOT LAMBDA LATENCY.**

## Verdicts

| change | verdict | why |
|---|---|---|
| E8 six-thread KleidiAI repack (`GGML_KLEIDIAI_REPACK_THREADS`) | **general default** | Load −5.67 s on arm64 local-file (−5.68 s memfd). Nothing else changes. Answers are bit-identical. Inert on x86_64. Set the value to the engine's thread count. |
| E9 pre-packed GGUF + loader | **general opt-in** (arm64 KleidiAI builds only) | Load −4.80 s, but E8 is 0.88 s faster in the same rounds. x86_64 refuses the file, as designed (6/6; cause recorded by the probe in a refusal-only run, 6/6). **Hazard:** an unpatched engine loads it without error and returns NaN / wrong answers on both arches. |
| E7 skip tied output head | **general opt-in** (decshim-only engines) | Load −2.06 s, peak RSS −504 MB on arm64. x86_64 RSS −576 MB. Answers bit-identical. Aborts if logits are requested, so it is not a llama.cpp-wide default. |
| E10 `n_ctx` 4096 (model window) | **general default** | Peak RSS −290 MB arm64, −281 MB x86_64. Every timing no change; nothing worse. Answers bit-identical, including the full 4,096-token forward. Derive it from the head config's `max_length`. |
| E5 no-vocab GGUF | **general opt-in** (engines that tokenize outside llama.cpp) | Load −0.53 s, RSS −83 MB arm64; RSS −54 MB x86_64. The file has no usable vocabulary for llama.cpp tokenization. Ship it beside the full GGUF. |
| E6 complete bytecode | **general default** (packaged builds) | Imports −0.296 s arm64 (read-only package). x86_64 no change inside the co-tenanted host's wide range; nothing worse. On a writable FS CPython writes the same `.pyc` after first start. |

No rung degraded any measured non-Lambda path. `summarize.py --verdicts` enforces the operator rule (2026-10-22)
fail-closed: it refuses a "general default" unless every required host/path cell (`verdicts.json`
`required_cells`) is present with at least `min_rounds` (6) paired rounds; every headline metric is present,
finite, assessed and not **WORSE**; the arm ran before its control exactly as often as after; no load of the arm,
its dormant check or p1 failed unexpectedly; every arm row has a P1 row from the same run, round and path, and
every request AND every 4,096-token answer is identical to THAT round's P1; the long forward filled the window, and
repeats within each load were identical; on 0xd40 (`reference_identities`) every answer was compared with P1's
published reference and none differs; and no answer field (requests and the 4,096-token forward) is NaN or
±Infinity beyond the fields P1 itself returns (declared in `control_nan_fields`). Declared refusals
(`expected_refusals`) must appear exactly once per round and path, with the declared exit code and the declared
machine-readable reason (or exact diagnostic line); every failed load in every run must be one of them; and each
published P2 status must follow from the primary cell. E8 vs E9, same rounds: E9 loads **0.884 s** slower (local) and **0.809 s** slower
(memfd) than E8 (thresholds 0.116 / 0.163 s), with no other difference. Both remove the same repack cost, and E8
needs no new file format. Prefer E8.

## Method

- **Branch** <https://github.com/jsamuel1/strands-decider/tree/lkai/ac1b-general-ladder>, cut from P1. The
  measurements were taken with the scripts at `c718f7e24d7bf45bab2eb0b1e836d60f902fc6ef` (the DevSpace copy of
  `general/*.py *.sh` was byte-checked against it before the run). The analysis and P2 records were produced at
  `5a0f7b93d82dc0c1f10d56be34910a48be119734` (`commits.ladder` in every P2 record), which changes only the P2
  conditions note in `summarize.py`; no measurement script differs between the two. Later commits on the branch add
  only the raw rows, generated outputs, verdict text and this README; none changes a script.
- **Baseline** P1 `feat/lambda-kleidiai-baseline` @ `70eab7be69f86ddeb8c64158d94df344b5180098`.
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
- **Interleaving** (`run_ladder.py schedule()`): every round runs every cell (arm × path) once. Rounds come in
  reversal pairs: round 2k is a seeded random permutation, round 2k+1 the same permutation reversed (seed 20261114,
  `schedule.json` in each run, written before the first load). Over the 6 rounds actually run, every candidate runs
  before its control exactly 3 times and after it 3 times, and every cell's mean position is the middle of the
  round, so a linear drift within a round adds the same to every cell. summarize.py records the before/after count
  for every comparison (`order` in `ladder.json`, 3/3 in every cell) and the guard refuses an unbalanced one.
  Comparisons are same run, same round, same path and same CPU identity.
- **Warm cache from round 0:** before the first load, `run_ladder.py` reads every package tree, GGUF, request and
  reference file once and starts the image once (`prime.json`). A first attempt without this step on the freshly
  restarted DevSpace gave round 0 a cold page cache (E6 imports +0.03 s in round 0, −0.27 to −0.32 s in rounds 1–5);
  that attempt was discarded unread for any verdict and both hosts were rerun.
- **Pre-registered rule** (`summarize.py`), fixed before any result was read:
  1. Compute the per-round paired delta, `arm − p1`.
  2. An effect is real only if all paired deltas share a sign and |median| exceeds p1's own per-round range.
     For peak RSS, which repeats to a page, the bar is also at least 1% of p1.
  3. Otherwise the cell is "no change". Fewer than 3 rounds is "unassessed" (and refuses a general default).
- **Bit-identity:** every answer (and every repeat within a load) is compared as canonical JSON text, so NaN equals
  NaN, with the same run's P1 in the same round and path. On arm64 every answer is also compared with P1's
  published 0xd40 reference (`baseline/reference/outputs.json`); a missing comparison refuses a general default.
  There is no x86_64 reference.
- **Refusals:** P1's shim silences llama.cpp's log, so a failed load only says `dec_load_seq failed`. On that
  failure `probe.py` loads once more with the shim's verbose flag and a recording log callback, matches llama's
  ERROR lines against its `REFUSALS` table (exact diagnostic substrings), prints one `LKAI_REFUSAL` record and
  exits 3. `run_ladder.py` stores the record as `refusal` in `runlog.jsonl`. Any other failure exits with another
  code and has no reason.

## Hosts (recorded per load in `host_before`)

| | arm64 (primary) | x86_64 |
|---|---|---|
| host | DevSpace ds-lkai-p1-rs5ghbzv (H1's fixed host) | developer cloud desktop |
| CPU identity | part 0xd40 (as printed; no generation inferred) | GenuineIntel family 6 model 173 stepping 1, "Xeon 6975P-C" |
| CPUs / used | 16 / 10-15 (6 threads) | 32 vCPU / 10-15 (SMT siblings 26-31 not reserved) |
| kernel | 6.1.189-236.400.amzn2023.aarch64 | 6.1.189-236.400.amzn2023.x86_64 |
| cgroup cpu.max | max 100000 (no quota) | max 100000 |
| governor | not exposed (cpufreq absent) | not exposed |
| ggml CPU variant | armv8.2_3 + armv8.6_1 (KleidiAI) | sapphirerapids (KleidiAI cannot be built for x86) |
| other load (1-min loadavg) | 1.4–6.8 | **19–85: co-tenanted**, so x86 timing bars are wide |
| docker | 25.0.8 | 25.0.14 |
| run (UTC) | `20261113T054759Z`, 05:48 → 07:13, 96 loads, 0 failed | `20261113T054808Z-x86`, 05:48 → 07:12, 54 loads; 6 failed = E9 refusals by design |

The x86_64 build (`x86_build.sh`) is P1's `container_build.sh` with two changes, recorded in
`container_build_x86.diff`: `-DGGML_CPU_KLEIDIAI=OFF`, and no quantize step. It uses the same pinned image, by its
amd64 digest `sha256:586cea70…f970`, which is checked against the manifest list. Each rung's llama.cpp patches are
applied unchanged. On x86 the E8 and E9 `kleidiai.cpp` changes are not compiled in. The x86 packages were rebuilt
for this run; the shared libraries are not bit-reproducible (`out/sha-*.txt` are this build's), while the build
diff and every rung's llama.cpp diff (`out/*.diff`) are byte-identical to the previous build's.

## Results (median p1 → arm (paired median Δ; ±rule threshold) outcome; n = rounds)

"load" = `dec_load_total` (model load + context init); "load+first request" = P1's `model_ready`.

| rung | CPU identity | path | n | load s | load+first request s | first forward s | warm forward s | 4,096-token forward s | peak RSS MB | imports s | answers vs P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e10 | 0xd40 | arm64 local | 6 | 8.085 → 8.066 (-0.021; ±0.447) no change | 9.616 → 9.606 (-0.021; ±0.578) no change | 0.235 → 0.232 (-0.002; ±0.018) no change | 0.587 → 0.576 (-0.011; ±0.045) no change | 16.188 → 16.142 (-0.076; ±0.274) no change | 4879 → 4589 (-290; ±49) **better** | 0.484 → 0.486 (+0.002; ±0.059) no change | 60/60 req, 12/12 fwd |
| e10 | 0xd40 | arm64 memfd | 6 | 8.174 → 8.124 (-0.041; ±0.213) no change | 10.949 → 11.000 (+0.050; ±0.274) no change | 0.233 → 0.229 (-0.003; ±0.014) no change | 0.578 → 0.573 (-0.006; ±0.024) no change | 16.150 → 16.147 (+0.005; ±0.374) no change | 4879 → 4589 (-290; ±49) **better** | 0.489 → 0.493 (+0.008; ±0.025) no change | 60/60 req, 12/12 fwd |
| e10 | GenuineIntel 6/173/1 | x86_64 local | 6 | 1.190 → 1.049 (-0.154; ±0.436) no change | 4.071 → 3.677 (-0.250; ±2.949) no change | 0.554 → 0.452 (-0.102; ±0.641) no change | 1.095 → 1.069 (-0.022; ±0.128) no change | 21.381 → 22.486 (+1.104; ±2.638) no change | 5181 → 4900 (-281; ±52) **better** | 0.368 → 0.501 (+0.128; ±1.409) no change | 30/30 req, 6/6 fwd |
| e5 | 0xd40 | arm64 local | 6 | 8.085 → 7.575 (-0.530; ±0.447) **better** | 9.616 → 9.106 (-0.536; ±0.578) no change | 0.235 → 0.235 (+0.001; ±0.018) no change | 0.587 → 0.587 (-0.002; ±0.045) no change | 16.188 → 16.235 (+0.047; ±0.274) no change | 4879 → 4796 (-83; ±49) **better** | 0.484 → 0.484 (-0.008; ±0.059) no change | 60/60 req, 12/12 fwd |
| e5 | 0xd40 | arm64 memfd | 6 | 8.174 → 7.606 (-0.551; ±0.213) **better** | 10.949 → 10.563 (-0.385; ±0.274) no change | 0.233 → 0.237 (+0.003; ±0.014) no change | 0.578 → 0.580 (+0.001; ±0.024) no change | 16.150 → 16.110 (-0.029; ±0.374) no change | 4879 → 4796 (-83; ±49) **better** | 0.489 → 0.480 (-0.003; ±0.025) no change | 60/60 req, 12/12 fwd |
| e5 | GenuineIntel 6/173/1 | x86_64 local | 6 | 1.190 → 0.927 (-0.205; ±0.436) no change | 4.071 → 5.108 (+1.272; ±2.949) no change | 0.554 → 0.580 (+0.024; ±0.641) no change | 1.095 → 1.172 (+0.098; ±0.128) no change | 21.381 → 24.382 (+1.459; ±2.638) no change | 5181 → 5127 (-54; ±52) **better** | 0.368 → 0.858 (+0.248; ±1.409) no change | 30/30 req, 6/6 fwd |
| e6 | 0xd40 | arm64 local | 6 | 8.085 → 8.139 (+0.028; ±0.447) no change | 9.616 → 9.392 (-0.249; ±0.578) no change | 0.235 → 0.235 (-0.000; ±0.018) no change | 0.587 → 0.587 (-0.005; ±0.045) no change | 16.188 → 16.210 (+0.022; ±0.274) no change | 4879 → 4879 (-1; ±49) no change | 0.484 → 0.188 (-0.296; ±0.059) **better** | 30/30 req, 6/6 fwd |
| e6 | GenuineIntel 6/173/1 | x86_64 local | 6 | 1.190 → 1.181 (+0.099; ±0.436) no change | 4.071 → 4.267 (+0.388; ±2.949) no change | 0.554 → 0.512 (-0.007; ±0.641) no change | 1.095 → 1.144 (+0.059; ±0.128) no change | 21.381 → 24.646 (+2.498; ±2.638) no change | 5181 → 5180 (-2; ±52) no change | 0.368 → 0.598 (+0.037; ±1.409) no change | 30/30 req, 6/6 fwd |
| e7 | 0xd40 | arm64 local | 6 | 8.085 → 6.034 (-2.064; ±0.447) **better** | 9.616 → 7.575 (-2.116; ±0.578) **better** | 0.235 → 0.234 (-0.001; ±0.018) no change | 0.587 → 0.575 (-0.009; ±0.045) no change | 16.188 → 16.176 (-0.008; ±0.274) no change | 4879 → 4375 (-504; ±49) **better** | 0.484 → 0.484 (-0.011; ±0.059) no change | 60/60 req, 12/12 fwd |
| e7 | 0xd40 | arm64 memfd | 6 | 8.174 → 6.142 (-2.041; ±0.213) **better** | 10.949 → 8.928 (-2.044; ±0.274) **better** | 0.233 → 0.237 (+0.002; ±0.014) no change | 0.578 → 0.576 (+0.004; ±0.024) no change | 16.150 → 16.155 (-0.011; ±0.374) no change | 4879 → 4375 (-504; ±49) **better** | 0.489 → 0.488 (+0.006; ±0.025) no change | 60/60 req, 12/12 fwd |
| e7 | GenuineIntel 6/173/1 | x86_64 local | 6 | 1.190 → 1.197 (+0.007; ±0.436) no change | 4.071 → 4.005 (-0.066; ±2.949) no change | 0.554 → 0.513 (-0.044; ±0.641) no change | 1.095 → 1.122 (+0.036; ±0.128) no change | 21.381 → 22.441 (+0.964; ±2.638) no change | 5181 → 4605 (-576; ±52) **better** | 0.368 → 0.686 (+0.240; ±1.409) no change | 30/30 req, 6/6 fwd |
| e8 | 0xd40 | arm64 local | 6 | 8.085 → 2.432 (-5.668; ±0.447) **better** | 9.616 → 4.052 (-5.621; ±0.578) **better** | 0.235 → 0.235 (-0.002; ±0.018) no change | 0.587 → 0.578 (-0.013; ±0.045) no change | 16.188 → 16.179 (-0.014; ±0.274) no change | 4879 → 4880 (+0; ±49) no change | 0.484 → 0.493 (+0.007; ±0.059) no change | 60/60 req, 12/12 fwd |
| e8 | 0xd40 | arm64 memfd | 6 | 8.174 → 2.537 (-5.677; ±0.213) **better** | 10.949 → 5.382 (-5.623; ±0.274) **better** | 0.233 → 0.234 (+0.001; ±0.014) no change | 0.578 → 0.581 (+0.007; ±0.024) no change | 16.150 → 16.181 (+0.034; ±0.374) no change | 4879 → 4880 (+0; ±49) no change | 0.489 → 0.484 (+0.002; ±0.025) no change | 60/60 req, 12/12 fwd |
| e8 | GenuineIntel 6/173/1 | x86_64 local | 6 | 1.190 → 1.299 (+0.080; ±0.436) no change | 4.071 → 4.027 (-0.084; ±2.949) no change | 0.554 → 0.494 (-0.048; ±0.641) no change | 1.095 → 1.104 (+0.060; ±0.128) no change | 21.381 → 22.927 (+1.342; ±2.638) no change | 5181 → 5181 (-1; ±52) no change | 0.368 → 0.769 (+0.107; ±1.409) no change | 30/30 req, 6/6 fwd |
| e9 | 0xd40 | arm64 local | 6 | 8.085 → 3.317 (-4.804; ±0.447) **better** | 9.616 → 4.888 (-4.788; ±0.578) **better** | 0.235 → 0.237 (+0.001; ±0.018) no change | 0.587 → 0.581 (-0.004; ±0.045) no change | 16.188 → 16.225 (+0.037; ±0.274) no change | 4879 → 4854 (-26; ±49) no change | 0.484 → 0.491 (+0.001; ±0.059) no change | 60/60 req, 12/12 fwd |
| e9 | 0xd40 | arm64 memfd | 6 | 8.174 → 3.337 (-4.838; ±0.213) **better** | 10.949 → 6.334 (-4.608; ±0.274) **better** | 0.233 → 0.237 (+0.002; ±0.014) no change | 0.578 → 0.582 (+0.002; ±0.024) no change | 16.150 → 16.230 (+0.098; ±0.374) no change | 4879 → 4854 (-26; ±49) no change | 0.489 → 0.495 (+0.011; ±0.025) no change | 60/60 req, 12/12 fwd |

On arm64, every arm's answers are also bit-identical to P1's published 0xd40 reference: E8, E9, E7, E10 and E5
match 60/60; E6 matches 30/30 (local only). On both arches the repeat passes within every load were identical.

**Dormant checks** (patched code, variable off; paired against p1, local path):

| arm | 0xd40 load Δ (±) | 0xd40 answers | x86_64 |
|---|---|---|---|
| e8off | +0.062 s (±0.447), no change | 30/30 req, 6/6 fwd identical | n/a (same libs as e8) |
| e9p1 | −0.070 s (±0.447), no change | 30/30, 6/6 identical | no change, 30/30, 6/6 |
| e7off | +0.056 s (±0.447), no change | 30/30, 6/6 identical | no change, 30/30, 6/6 |

## Findings beyond the timings

1. **E9's file is unsafe without its loader.** On both arches, P1's unpatched engine loads
   `s2b-h512-Q8_0-e9kr8.gguf` without error. Every probabilities field comes back NaN and choices are wrong: all 5
   requests DIFFER from P1's 0xd40 reference, and 4/5 differ from x86 P1. Files: `hazard/arm-p1libs-e9gguf.json`,
   `hazard/x86-p1libs-e9gguf.json`. With E9's loader on x86, the file is refused. The measured x86 run's six E9
   failures only show the shim's generic `dec_load_seq failed`, because P1's shim silences llama's log. The cause
   is therefore proven by a refusal-only x86 run, `runs/20261116T050246Z-x86-e9refusal` (6 rounds, e9 local only;
   E9 GGUF sha256 as `inputs.json`; x86 E9 package rebuilt with `x86_build.sh`, every library byte-identical to
   the measured one, `sha-e9.txt` and `provenance.json`). In every round the probe's `refusal` record carries
   llama.cpp's error `tensor 'blk.0.attn_qkv.weight' is KleidiAI-prepacked (E9) but is not in a CPU_KLEIDIAI
   buffer`, reason `e9-prepacked-not-in-kleidiai-buffer`, exit 3. `verdicts.json` declares both runs: the measured
   one by its exact generic diagnostic and exit 1, the refusal run by that reason and exit 3. So E9 GGUFs
   must never sit where a generic loader can pick them up.
2. **x86_64 P1 itself returns NaN for one request.** In every arm, P1 included, `incident-review-4q` returns NaN
   probabilities (12 fields: `nan_fields` in `ladder.json`) on the auto-selected `libggml-cpu-sapphirerapids.so`.
   Removing that variant library gives finite answers (retries 0.551, freeze 0.351, alarm 0.019, capacity 0.078),
   close to arm64's (0.554 / 0.355 / 0.019 / 0.072). This predates every rung and is identical across arms, so the
   comparison is unaffected. It is still a P1 x86_64 defect (AVX-512 variant path), outside this ladder; recorded
   for follow-up. These 12 fields are declared in `verdicts.json` `control_nan_fields`; the guard requires every
   arm to return exactly P1's set and refuses any other non-finite answer.
3. **E8's switch value:** the shipped value 6 equals `THREADS` here. The general default should follow the
   engine's thread count. Other counts were not measured in this ladder.

## Deviations and limits

- The x86_64 host is a shared, co-tenanted desktop (loadavg 19–85), and CPUs 10-15 share SMT cores with 26-31,
  which were not reserved. x86 timing bars are therefore wide (load ±0.44 s; imports ±1.41 s; 4,096 forward
  ±2.6 s), so x86 timing can only rule out large regressions. Memory and answers are exact.
- The arm64 load range (p1 local ±0.447 s) is wider than in the October run because one p1 round read 8.51 s;
  smaller effects (E10 load, E5 load+first request) are inside it and read "no change". The large effects
  reproduce: the E8, E9 and E7 paired load deltas agree with the October run to within 0.04 s.
- E6's published package was built from a P1 zip `d148d706…` (51,941,882 B). P1's published zip is `26eb2651…`
  (51,941,898 B). The arm64 e6 package was rebuilt from `26eb2651…` with `e6/pyc_zip.py` @ e989e39 in P1's image:
  1035 compiled, 0 failed, CPython 3.12.15, magic `cb0d0d0a`; `stage.sh` now stops if `pyc_zip.py`'s manifest
  lists any failure. x86 e6 used `x86_e6.sh` → `pyc_tree.py`, the same compile rule (UNCHECKED_HASH,
  `dfile=/var/task/…`), applied to the unzipped x86 package, because `pyc_zip.py` only accepts P1's arm64 zip. It
  exits nonzero on any compile failure; the measured package's manifest is
  `runs/20261113T054808Z-x86/pkg-e6.manifest.json` (1035 compiled, 0 failed, CPython 3.12.15, tree sha256).
- Page cache was warm for every load, because there was no sudo to drop caches. Cold-disk load is not measured.
- No AWS calls were made. Lambda numbers (R) are not comparable with these.

## Reproduce

```bash
# arm64 fixed host (the DevSpace); artifacts verified against inputs.json
general/stage.sh <artifact dir> <stage dir>                 # verifies hashes, builds pkg-*, e6 pyc (fails on any compile failure)
python3 general/make_arms.py <stage dir> <artifact dir> <stage dir>/arms.json
python3 general/run_ladder.py <stage dir>/arms.json <arm64 run> --rounds 6 --seed 20261114 --cpuset 10-15 --threads 6
# x86_64
general/x86_build.sh <p1.zip> <x86 dir>                     # pkg-p1/e8/e9/e7, diffs
general/x86_e6.sh <x86 dir>/pkg-p1 <x86 dir>/pkg-e6         # writes <x86 dir>/pkg-e6.manifest.json, fails on any compile failure
python3 general/x86_arms.py <x86 dir> <dir with p1.gguf e9.gguf e5.gguf> <x86 dir>/arms.json
python3 general/run_ladder.py <x86 dir>/arms.json <x86 run> --rounds 6 --seed 20261114 --platform linux/amd64 --cpuset 10-15 --threads 6
# x86_64 E9 refusal cause: an arms.json with only the e9 arm (x86_arms.py writes only the packages present)
python3 general/run_ladder.py <x86 e9 arms.json> <x86 refusal run> --rounds 6 --seed 20261114 --platform linux/amd64 --cpuset 10-15 --threads 6
# analysis, fail-closed operator-rule guard, P2 result files
python3 general/summarize.py <arm64 run> <x86 run> <x86 refusal run> --out <out> --verdicts general/verdicts.json --commit <analysis SHA> --run-id 20261113T054759Z-general
python3 general/tables.py <out>/ladder.json
python3 results/publish_result.py <out>/p2/<rung>/results.json --validate-only   # P2 (feat/lambda-kleidiai-results)
```

`runs/` holds this ladder's raw probe files and `out/` holds the analysis; `out/p2/<rung>/results.json` are the P2
records (experiment ids `G-general-E*`, run id `20261113T054759Z-general`). The earlier 5-round run
(`20261025T090459Z-general`, rotation schedule) is superseded by this one and its rows were removed from the branch.

## P2 pointers

Published with `results/publish_result.py` (feat/lambda-kleidiai-results) to the P2 store and dashboard
<https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results>, all `status: improved`. Each stored
`result.json` was read back and equals `out/p2/<rung>/results.json` apart from the publisher's own `published`
block; the stored `ladder.json` / `probes.jsonl` sha256 match the committed files.

| rung | P2 run (current, 6 balanced rounds) | dashboard version |
|---|---|---|
| E8 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E8-parallel-repack/20261113T054759Z-general/result.json` | 65 |
| E9 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E9-prepacked/20261113T054759Z-general/result.json` | 66 |
| E7 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E7-skip-tied-output/20261113T054759Z-general/result.json` | 67 |
| E10 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E10-kv-size/20261113T054759Z-general/result.json` | 68 |
| E5 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E5-no-vocab-gguf/20261113T054759Z-general/result.json` | 69 |
| E6 | `s3://strands-decider-lkai-results-apse2/runs/G-general-E6-bytecode/20261113T054759Z-general/result.json` | 70 |

Superseded (5-round rotation schedule, kept in the append-only store, not used for any verdict above):
E8 `G-general-E8-parallel-repack/20261025T090459Z-general` (v59), E9 `G-general-E9-prepacked/20261025T090459Z-general` (v60), E7 `G-general-E7-skip-tied-output/20261025T090459Z-general` (v61), E10 `G-general-E10-kv-size/20261025T090459Z-general` (v62), E5 `G-general-E5-no-vocab-gguf/20261025T090459Z-general` (v63), E6 `G-general-E6-bytecode/20261025T090459Z-general` (v64).
