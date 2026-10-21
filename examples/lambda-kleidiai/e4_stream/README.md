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

Arms, one zip (`e4_entry.handler`), differing only in environment (`deploy_e4.ARMS_SETS`: `E4_ARMS=r1` or `r2`):

| Arm | Environment | Meaning |
|---|---|---|
| `base` | `E4_STREAM=0` | download, then load (P1's sequence) |
| `stream` | `E4_STREAM=1 STREAM_ORDER=arrival E4_MMAP_THRESHOLD=4194304` | the experiment (run 2) |
| `basethr` | `E4_STREAM=0 E4_MMAP_THRESHOLD=4194304` | the mmap-threshold pin alone (run 2) |
| `streamdyn` | `E4_STREAM=1 STREAM_ORDER=arrival` | the overlap without the pin (run 2; run 1's `stream`) |
| `streamfile` | `E4_STREAM=1 STREAM_ORDER=file` | overlap in llama.cpp's own order (run 1): shows how much the ordering matters |

## Reproduce

```bash
# arm64 Linux host with Docker; P1's build (baseline/build.sh) is reused when present
AWS_PROFILE=<profile> E4_ARMS=r2 examples/lambda-kleidiai/e4_stream/run.sh <run-id>
# no AWS: each arm in the pinned Lambda image against a 80 MiB/s local stand-in for S3
examples/lambda-kleidiai/e4_stream/localtest.sh
```

## Results

Both runs are on the P2 dashboard (`E4-stream-load`); the redacted raw data is in `results/<run-id>/`. Lambda
python3.12 arm64, 10,240 MB, SnapStart on published versions, us-east-2; P1's harness (`bench_e4.py` = `bench.py` with
a fresh Lambda client per wave), 5 rounds x 10 per class and arm. Arm waves run sequentially, with order reversed on
odd rounds. Deltas are medians within the same CPU part (cpuinfo `CPU part`), pooled by sample count.

**Run 2, `20261020T181753Z` (status improved).** Zip `ddd2f998…`, commit `4c5fcf0`. 0 errors, 0 class mismatches,
2,400 of 2,400 answers bit-identical to P1's reference.

| p50 (pooled delta vs base) | base | stream | basethr (pin only) | streamdyn (no pin) |
|---|---|---|---|---|
| cold first-request (Init + Duration) | 35.92 s | 31.40 s (-5.64 s) | 36.05 s (+0.19 s) | 31.35 s (-5.50 s) |
| cold download+load | 33.71 s | 29.16 s (-5.64 s) | 33.81 s | 29.15 s |
| cold first inference | | (+0.001 s) | (-0.002 s) | (+0.001 s) |
| cold Max Memory Used | 5,158 MB | 5,122 MB (-34) | 5,122 MB (-35) | 5,631 MB (+476) |
| restore first-request | 3.93 s | 3.92 s (-0.02 s) | 3.89 s (-0.07 s) | 4.27 s (+0.34 s) |
| restore Max Memory Used | 5,152 MB | 5,110 MB (-41) | 5,112 MB (-40) | 5,647 MB (+495) |
| warm inference | 0.706 s | 0.728 s (-0.02 s) | 0.722 s (-0.01 s) | 0.681 s (-0.04 s) |

**Run 1, `20261020T131645Z` (status inconclusive).** Arms base / stream (no pin) / streamfile. stream: cold -4.97 s,
restore +0.41 s, memory +470 MB. streamfile (overlap in llama.cpp's creation order): cold -1.10 s, so the arrival
order is what makes the overlap pay.

Findings:

- **Cold start.** The overlap saves 4.0-9.3 s of the first request depending on the CPU part (0xd40: 35.92 ->
  31.38 s, -4.54 s, -13 %; 0xd4f: -3.97 s, -12 %, 4 samples; 0xd0c, whose repack is slowest: 48.49 -> 39.23 s,
  -9.26 s, -19 %), 5.64 s pooled. All of it is in download+load; the download itself is unchanged (25.0 s), and
  the loader spends about 18.5 s of it blocked on bytes. Lambda picks the CPU part per environment, so the arms' raw
  p90s differ partly by how many 0xd0c environments each drew.
- **Memory.** Without a fix the overlap leaves about +480 MB resident (run 1, and streamdyn in run 2). Cause, found
  locally with `malloc_stats` on the same zip and real S3: the KleidiAI repack frees large blocks while the 32 download
  threads still read 8 MiB bodies, which raises glibc's dynamic mmap threshold, so each thread's arena keeps a ~16.7 MiB
  free top that `malloc_trim` cannot return. Pinning `M_MMAP_THRESHOLD` at 4 MiB (`E4_MMAP_THRESHOLD`) stops it; the pin
  alone also trims base by about 35-58 MB.
- **Restore.** SnapStart already hides the download and the load: both run in `publish_version`'s snapshot init,
  never in a restore. A restore only sees what the snapshot holds, which is why run 1's extra memory showed up as
  +0.41 s of restore. With the pin, restore is unchanged (-0.02 s, inside run-to-run spread). E4 helps on-demand cold
  starts and shortens snapshot creation; it does not speed up restores.
- **Correctness.** Every arm of both runs gives bit-identical answers to P1 (same request, same CPU part).

### Host variability (initiative rule, assessed on the recorded data; no rerun)

`results.json` carries `host_variability` (built by `summarize_e4.host_variability`): what the run observed, what it
did not record, and each arm's per-round delta against base within each CPU part.

| | Run 1 | Run 2 |
|---|---|---|
| Recorded per request | cpuinfo `CPU part`: 0xd40, 0xd0c, 0xd4f | same |
| Recorded per cold environment | ggml CPU variants (one set per part: 0xd40 `armv8.6_1`, 0xd0c `armv8.2_2`, 0xd4f `armv8.6_2`), `threads=6` (`os.cpu_count()`; `THREADS` unset) | same |
| Runtime | `python:3.12.mainlinev2.v43` on all 180 INIT_START streams | same, all 241 |
| Not recorded (unknown) | cpuinfo implementer/variant/revision, Features, AT_HWCAP/HWCAP2, cgroup limits, kernel, AZ, physical host | same |

No Graviton generation is claimed: the runs recorded the part number only. The parts are drawn unevenly across arms
(run 2 cold 0xd40: base 27, stream 36, basethr 42, streamdyn 29 of 50), so the unsplit per-arm `phases`
distributions mix parts and must not be compared across arms. Every conclusion above uses `<arm>_vs_base.by_cpu_part`
instead, and it holds round by round:

| Within-part delta vs base, range over rounds | 0xd40 | 0xd0c | 0xd4f |
|---|---|---|---|
| run 2 stream, cold first-request | -4.31 to -4.75 s (5) | -7.50 to -9.52 s (5) | -3.94 s (1) |
| run 2 stream, restore first-request | -0.05 to +0.06 s (5) | -0.01 to -0.10 s (2) | no restores |
| run 2 stream, restore Max Memory | -38 to -44 MB (5) | -36 to -38 MB (2) | no restores |
| run 2 streamdyn, restore first-request | +0.32 to +0.42 s (5) | +0.05 s (1) | no restores |
| run 1 stream, restore first-request | +0.33 to +0.61 s (5) | +0.43 to +0.57 s (2) | no restores |
| run 1 streamfile, cold first-request | -0.99 to -1.08 s (5) | -1.46 to -3.28 s (2) | -0.66 to -1.18 s (3) |

Base's own round-to-round spread of cold first-request is 0.2 s on 0xd40 and 0.6 s on 0xd0c, against a 4-9 s effect,
so host placement cannot flip the cold conclusion, and the restore "no change" with the pin is mixed-sign inside
one part. A fixed-host EC2 baseline (H1) is not needed for E4. Limits: no restore landed on 0xd4f; 0xd4f cold for
`stream` rests on 4 samples in 1 round.

Cost: run 1 $3.54, run 2 $4.73 (harness estimate, including the SnapStart 3-hour cache minimum).
