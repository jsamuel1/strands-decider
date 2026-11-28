# Running with low memory

This page is for running a checkpoint in less memory than it normally takes: E4B in under 10 GB, the
12B in a smaller ECS task, or the 26B-A4B (a mixture of experts) in 16 or 10 GB. It covers what
holds the memory, each option for reducing it, and what each costs in speed and accuracy.

Most options here are measurement switches on the `feat/repack-file` branch, not defaults. The
[switches](#the-switches) are listed at the end. For how the builds and CPU modules themselves
compare, see [graviton-cpu-variants.md](graviton-cpu-variants.md#memory).

## Summary

| Option | Unreclaimable memory | Speed with enough memory | Accuracy | When to use it |
| --- | --- | --- | --- | --- |
| [Repack into a file](#repack-into-a-file) (`patches/0004`) | −85% (E4B) to −95% (12B, 26B-A4B) | 1.00× | identical | A limit below the repacked copy, on a host with a fast local disk |
| [Share one repack file](#share-one-repack-file-across-processes) (`patches/0005`) | as above, and one copy for every process on a host | 1.00×; later loads 2.7× faster | identical | Several workers or replicas on one host |
| [Leave the experts unrepacked](#leave-the-experts-unrepacked) (`patches/0006`) | 26B-A4B: 27.0 GB to 3.9 GB | 0.78–0.94× | passes the release check | A mixture of experts, when no repack file can be written |
| [Quantize only the experts](#quantize-only-the-experts) | 26B-A4B: 27.0 GB to 16.2 GB; file 26.9 GB to 15.4 GB | 0.91–1.06× | fails the release check (98.6% top-1) | Opt-in, where 4 more wrong answers in 1,032 is acceptable |
| [Skip the prefetch](#skip-the-prefetch) (the handler) | none | unchanged | identical | Always: the handler does it when the model files exceed memory |
| No repacking (`STRANDS_DECIDER_NO_REPACK=1`) | −85% (E4B) | 0.63–0.78× | conforms | Superseded by the repack file |

With enough memory, the repack file, the shared repack file and the skipped prefetch run at full
speed with identical answers. Leaving the experts unrepacked costs 6–22%, and no repacking 22–37%.
4-bit experts keep the speed but lose some accuracy.

Below what a model needs, nothing runs at full speed. Every request then re-reads weights from
disk, at the disk's speed.

## Where the memory goes

The figure that decides whether a process is killed is the memory the kernel can't reclaim:
anonymous memory, `RssAnon` in `/proc/self/status`. A checkpoint holds four kinds of memory:

- **ggml's repacked copy of the weights: almost all of the unreclaimable memory.** GGUF stores
  Q8_0 in its portable block layout. ggml's i8mm GEMM wants rows interleaved (`q8_0_4x8` on
  Graviton3–5, `q8_0_4x4` on Graviton2), and it can't rearrange the read-only mapped file, so it
  writes an anonymous copy at load. That's 5.5 GB for E4B, 12.7 GB for the 12B and 27.0 GB for the
  26B-A4B.
- **The mapped GGUF: reclaimable.** The kernel drops its pages under pressure and reads them in
  again when they're used. Once a weight is repacked, its GGUF pages aren't read again, so they go
  first. That's why the 12B fits a 16 GB limit and the 26B-A4B a 32 GB one, with the repack in
  memory, despite peaks of 27.8 GB and 54.8 GB with no limit.
- **The state's buffers: unreclaimable, per request.** The KV cache and compute buffers grow with
  the request (E4B: about 1.1 GB at ~4.1k tokens, 2.9 GB at ~9.6k), and are freed after it.
- **The handler's prefetch: reclaimable.** It reads the model files into the page cache before
  llama.cpp loads them, which helps Lambda, whose image files load lazily. When the files are bigger
  than memory, it's wasted: they evict themselves as they're read.

## The options

All measured on Graviton5 (`c9g.8xlarge`, cores 0–5, 6 threads, `cpu=variants`) with the model on
local gp3 (16,000 IOPS, 1,000 MB/s), under a container memory limit (`--memory`, no swap). Speed
is median latency against the same model's in-memory repack with no limit. Accuracy is on each
checkpoint's 1,032 golden-vector questions.

### Repack into a file

`patches/0004` adds `GGML_CPU_REPACK_FILE_DIR=<dir>`. With it, ggml repacks exactly as before, but
into an unlinked file in `<dir>` mapped shared, instead of anonymous memory. The kernel can then
write the repacked weights back to the file and drop them under memory pressure, as it does the
GGUF's own mapped pages, and read them in again when they're used. Unset, or if the file can't be
made, nothing changes.

| Model, repack, limit | ~40 tokens | ~1.1k | ~4.1k | image | `RssAnon` after load | peak RSS | load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E4B, memory, none | 1.00× | 1.00× | 1.00× | 1.00× | 5,555 MB | 11,581 MB | 16.2 s |
| E4B, file, none | 1.00× | 1.00× | 1.00× | 1.00× | 821 MB | 11,580 MB | 15.7 s |
| E4B, memory, 10 GB | 1.00× | 1.00× | 1.00× | 1.00× | 5,555 MB | 10,199 MB | 17.3 s |
| E4B, file, 10 GB | 1.00× | 1.00× | 1.00× | 1.00× | 821 MB | 10,186 MB | 20.3 s |
| E4B, memory, 6 GB | killed (out of memory) | | | | | | |
| E4B, file, 6 GB | 0.99× | 0.99× | 0.83× | 0.86× | 821 MB | 6,116 MB | 37.0 s |
| 12B, memory, none | 1.00× | 1.00× | 1.00× | 1.00× | 12,702 MB | 27,834 MB | 26.3 s |
| 12B, file, none | 1.00× | 1.00× | 1.00× | 1.00× | 637 MB | 27,834 MB | 26.4 s |
| 12B, memory, 16 GB | 1.00× | 1.00× | 1.00× | 1.01× | 12,702 MB | 16,332 MB | 27.8 s |
| 12B, file, 16 GB | 1.03× | 1.01× | 1.00× | 1.03× | 637 MB | 16,322 MB | 53.0 s |
| 12B, memory, 12 GB | killed (out of memory) | | | | | | |
| 12B, file, 12 GB | 0.15× | 0.53× | 0.85× | 0.26× | 637 MB | 12,258 MB | 144.8 s |

The 26B-A4B's results are in [The 26B-A4B](#the-26b-a4b). Every run's answers were identical to the
in-memory repack's, to the last digit: it's the same repack in a different kind of memory.

What these results show:

- **The repacked weights become reclaimable.** Unreclaimable memory falls by 85% for E4B
  (5.5 GB to 0.8 GB) and 95% for the 12B (12.7 GB to 0.6 GB), as with no repacking, but at full
  speed.
- **With room to spare, it costs nothing.** Speed, answers and load time match the in-memory
  repack. Under a limit the load is slower (53 s against 28 s for the 12B at 16 GB), because the
  kernel writes pages back while ggml is still repacking.
- **It lowers the smallest limit a model runs in.** With the repack in memory, the limit must hold
  the repacked copy plus the state's buffers: E4B is killed at 6 GB and the 12B at 12 GB. With the
  file, both run, with the kernel reading weights back from disk. E4B at 6 GB loses 17% on a
  ~4.1k-token request. The 12B at 12 GB runs at 0.15–0.85×, worst on short requests, which spend
  their time reading weights in.
- **On Lambda, the file would go in `/tmp`.** E4B's repack is 4.7 GB, inside the 10 GB of ephemeral
  storage, and it would be written on every new environment. It isn't measured on Lambda, whose
  `/tmp` is slower than gp3. It doesn't combine with SnapStart, which allows 512 MB of ephemeral
  storage.

### Share one repack file across processes

`patches/0005` names the repack file instead of unlinking it, so every process that loads the same
model with the same CPU module shares one copy, and it's kept for the next process.
`STRANDS_DECIDER_REPACK_CACHE=<dir>` turns it on; the shim names the files after the GGUF's and the
CPU module's size and modification time, so a new model or build writes new ones.

- **The first process writes it:** under an exclusive lock, to `<name>.partial`, flushed and renamed
  into place once the model has loaded. A finished file is never a partial one.
- **Later processes map it:** read-only, and skip repacking.
- **A process that finds another one writing** repacks into an unlinked file of its own.

Three 26B-A4B (Q8_0) containers on one 64 GB host, 6 cores each, sharing `/work/cache`:

| Container | Load | `RssAnon` after load |
| --- | --- | --- |
| First (writes the 26.8 GB file) | 71.6 s | 1,467 MB |
| Second and third (started together, once the first had loaded) | 19.9 s each | 1,367 MB |
| All three restarted together, the file in place | 19.7–21.8 s | 1,367 MB |
| *For comparison: one container, the repack in memory* | *54.5 s* | *27,020 MB* |

- **Three containers fit where one in-memory repack barely does.** With all three running
  requests, the host's unreclaimable memory peaked at 10.6 GB: each process's own 1.4 GB, plus the
  state's buffers. The GGUF and the repack file, 49.7 GB of page cache, are shared. Three in-memory
  repacks would need 81 GB, more than the host has.
- **Every load after the first is 2.7× faster,** 20 s against 54.5 s, because it maps the
  finished file instead of repacking.
- **The first load costs about 10 s more** than an unlinked file (71.6 s against 62.2 s): the flush
  before the rename.
- **The first container ran at 0.96–0.99× speed,** alone on the host while it ran its requests. The
  other five ran three at a time, sharing the host's memory bandwidth, so their speeds aren't
  comparable.

### Leave the experts unrepacked

A mixture of experts' expert weights are 90% of the 26B-A4B (24.3 GB of 26.8 GB). `patches/0006`
adds `GGML_CPU_REPACK_NO_EXPERTS=1` (the shim's `STRANDS_DECIDER_NO_EXPERT_REPACK=1`): the repack
declines the expert weights (`MUL_MAT_ID`'s), which stay mapped from the GGUF, and repacks the rest.

llama.cpp's `--override-tensor exps=CPU` can't do this: an override to the CPU buffer still
considers the repack, so it repacks everything. Measured: the same 27 GB, the same answers.

| 26B-A4B Q8_0, limit | ~40 tokens | ~1.1k | ~4.1k | image | `RssAnon` after load | peak RSS | load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| repack in memory, none | 1.00× | 1.00× | 1.00× | 1.00× | 27,020 MB | 54,815 MB | 54.5 s |
| experts unrepacked, none | 0.81× | 0.78× | 0.85× | 0.94× | 3,877 MB | 31,710 MB | 46.9 s |
| experts unrepacked, 16 GB | 0.06× | 0.19× | 0.48× | 0.28× | 3,654 MB | 16,345 MB | 162.3 s |
| experts unrepacked, 10 GB | 0.03× | 0.12× | 0.42× | 0.15× | 3,654 MB | 10,213 MB | 194.9 s |
| *repack in a file, 16 GB* | *0.10×* | *0.32×* | *0.68×* | *0.40×* | *1,238 MB* | *16,347 MB* | *215.0 s* |
| *repack in a file, 10 GB* | *0.04×* | *0.18×* | *0.56×* | *0.21×* | *1,238 MB* | *10,215 MB* | *229.1 s* |

On Graviton4's golden vectors: 99.32% top-1, mean |dp| 0.00792, against the full repack's 99.03%
and 0.00801. It passes the release check.

- **Unreclaimable memory falls 86%, from 27.0 GB to 3.9 GB,** with no file to write: the experts
  are the GGUF's own pages.
- **With enough memory it costs 6–22% speed,** since the expert GEMMs run on the GGUF's layout.
- **Under a limit, the repack file is faster** at every size of request. The file holds the
  experts in the fast layout; here, every re-read expert runs the slower kernel too.
- **Use it when no file can be written:** a read-only or slow file system, or a disk too small for
  the repack.

### Quantize only the experts

`EXPERT_QUANT` (the build's `--expert-quant`) gives the expert weights their own type, passed to
`llama-quantize` as `--tensor-type _exps\.weight=<type>`; everything else stays Q8_0. Q4_K can't be
used: its 256-wide blocks don't divide the 26B-A4B's 704-wide down experts, and `llama-quantize`
would fall back to a type ggml can't repack. Q4_0 and IQ4_NL have 32-wide blocks, and both are
repacked for i8mm.

Both bring the 26B-A4B's file from 26.9 GB to 15.4 GB.

| 26B-A4B, limit | ~40 tokens | ~1.1k | ~4.1k | image | `RssAnon` after load | peak RSS | load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q8_0, repack in memory, none | 1.00× | 1.00× | 1.00× | 1.00× | 27,020 MB | 54,815 MB | 54.5 s |
| Q4_0 experts, repack in memory, none | 1.06× | 0.97× | 0.98× | 1.00× | 16,161 MB | 31,004 MB | 37.5 s |
| Q4_0 experts, repack in a file, 16 GB | 0.93× | 0.91× | 0.95× | 0.93× | 1,487 MB | 16,327 MB | 71.5 s |
| Q4_0 experts, repack in a file, 10 GB | 0.12× | 0.42× | 0.71× | 0.47× | 1,238 MB | 10,215 MB | 128.0 s |
| Q4_0 experts, experts unrepacked, 10 GB | 0.07× | 0.21× | 0.56× | 0.30× | 3,654 MB | 10,215 MB | 114.7 s |
| IQ4_NL experts, repack in memory, none | 1.00× | 0.91× | 0.94× | 0.99× | 16,117 MB | 30,960 MB | 37.6 s |
| IQ4_NL experts, repack in a file, 16 GB | 0.91× | 0.86× | 0.91× | 0.92× | 1,443 MB | 16,327 MB | 70.5 s |
| IQ4_NL experts, repack in a file, 10 GB | 0.12× | 0.34× | 0.70× | 0.47× | 1,239 MB | 10,215 MB | 127.5 s |
| IQ4_NL experts, experts unrepacked, 10 GB | 0.07× | 0.22× | 0.55× | 0.29× | 3,654 MB | 10,214 MB | 116.4 s |

On Graviton4's golden vectors:

| Experts | Top-1 | Mean \|dp\| | Max \|dp\| | Release check against Q8_0 |
| --- | --- | --- | --- | --- |
| Q8_0 | 99.03% | 0.00801 | 0.159 | (the reference) |
| Q4_0 | 98.55% | 0.01383 | 0.270 | fails: 5 more wrong, mean error 1.73× |
| IQ4_NL | 98.64% | 0.01355 | 0.157 | fails: 4 more wrong, mean error 1.69× |

- **4-bit experts make the 26B-A4B fit 16 GB at nearly full speed:** 0.86–0.95× with the repack in
  a file, against 0.10–0.68× for Q8_0. At 10 GB they're 1.3–3× faster than Q8_0, since there's
  less to re-read.
- **Under a limit, the repack file still beats leaving the experts unrepacked,** as with Q8_0: at
  10 GB, 0.12–0.71× against 0.07–0.56×.
- **With enough memory they run at about full speed,** Q4_0 0.97–1.06× and IQ4_NL 0.91–1.00×,
  and load in 37.5 s, against 54.5 s.
- **Both conform to the standard but fail the release check,** like KleidiAI on E4B (98.64%).
  IQ4_NL is slightly better on every measure, and its worst case matches Q8_0's. They're an
  opt-in trade of accuracy for memory.

### Skip the prefetch

The handler skips its prefetch when the model files are bigger than the memory it may use: the
Lambda function's size, else the container's cgroup limit, else the host's memory. With room, the
prefetch runs as before.

| Run | Load with the prefetch | Load without it |
| --- | --- | --- |
| 26B-A4B Q8_0, repack in a file, 16 GB (EC2) | 225.3 s | 215.0 s |
| 26B-A4B Q8_0, repack in a file, 10 GB (EC2) | 275.0 s | 229.1 s |
| 26B-A4B on Lambda from S3 Files: the handler's model read | 370.8 s (Q8_0) | 13.4 s (IQ4_NL experts) |

From local gp3 it saves 5–17% of the load: at 10 GB the prefetched pages were evicted before
llama.cpp used them, so they were read twice. On Lambda from S3 Files it removes six minutes of
reading. Speed and answers don't change.

### What doesn't help

- **A Q8_0 KV cache:** no saving, and as low as 0.41× on long states. The state's buffers are
  mostly compute buffers, and CPU flash attention dequantizes Q8_0 K and V on every access
  ([measurements](graviton-cpu-variants.md#options-that-change-memory)).
- **Splitting the state into smaller batches:** it would break the prefix_bidir mask, under which
  every state token sees every other.
- **One expert per Lambda function, or one layer group per function:** every one of the 30 layers
  routes each token to 8 of 128 experts, so one forward pass would need 30 rounds of shipping a
  state's activations out and back, hundreds of MB per layer at ~4.1k tokens. Lambda functions
  can't connect to each other, a synchronous invocation carries at most 6 MB, and a request can't
  be sent to a particular warm environment. Splitting a model by expert or layer needs ECS tasks.

## Mixtures of experts

### Why a request touches nearly every expert

The 26B-A4B has 30 layers of 128 experts, and each token picks 8 of them in each layer: about 4B of
its 26B parameters are active per token, not per request. The state is decoded as one batch, and
each token makes its own pick, so with even routing a request touches 1 − (120/128)<sup>n</sup> of
a layer's experts:

| Tokens in the request | Experts touched |
| --- | --- |
| 40 | about 92% |
| 100 | about 99.8% |
| 1,000 | effectively all |

A mixture of experts saves compute in that batch: each token is multiplied through 8 experts, not
128. It saves memory and memory bandwidth only when tokens go through one at a time, as in text
generation, and the decider never generates. Below the weights' size, every request re-reads nearly
all of them.

### Which experts are used

`STRANDS_DECIDER_EXPERT_STATS=<path>` counts each layer's routing (llama.cpp's `ffn_moe_topk`
tensors). Over the states of the 26B-A4B's 1,032 golden-vector questions (234,115 tokens per layer):

| | Typical layer (median) | Range over the 30 layers |
| --- | --- | --- |
| Experts carrying 50% of the routing | about 12 | 5–26 |
| Experts carrying 90% | 46 | 17–79 |
| Experts carrying 99% | 91 | 42–110 |
| Effective number of experts | 48 | 19–88 |
| Busiest expert's share | 8% | 3.9–12.3% |

- **Routing is skewed, most of all in the last layer.** Layer 29 never picked 54 of its experts,
  and 17 carry 90% of its routing. Layers 0–3 are the most even.
- **1,295 of the 3,840 experts (34%) each get under 0.1% of their layer's routing.** Pruning them
  would remove about 8 GB of expert weights, if accuracy holds.
- **The skew doesn't shrink a request's working set.** The cold experts are each picked rarely, but
  a request of thousands of tokens still picks most of them at least once.

### Which experts are repacked

All of them, at load, whether or not they're used. Each layer's experts are one 3-D tensor per
projection (`ffn_gate_up_exps.weight`, 2816 × 1408 × 128, and `ffn_down_exps.weight`), and ggml
repacks whole tensors as llama.cpp loads them, before any routing. So the 27 GB repacked copy holds
all 3,840 experts, including layer 29's 54 that are never picked.

| Option | Experts repacked | What memory holds afterwards |
| --- | --- | --- |
| Repack in memory (the default) | all, at every load | all of them, unreclaimable |
| Repack into a file | all, written at every load | the ones in use; the rest can be dropped |
| Share one repack file | all, written once per host | the ones in use, shared by every process |
| Leave the experts unrepacked | none | the ones in use, in the GGUF's layout |

With a repack file, an unused expert costs disk, not memory: it's written once, and the kernel can
drop its pages. Repacking only the experts a request uses would need a ggml change, since the
`MUL_MAT_ID` kernels assume one layout for a whole tensor (see [Not yet tried](#not-yet-tried)).

## The 26B-A4B

All the 26B-A4B's runs, on Graviton5 with the model on local gp3:

| Model, option, limit | ~40 tokens | ~1.1k | ~4.1k | image | `RssAnon` after load | peak RSS | load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q8_0, repack in memory, none | 1.00× | 1.00× | 1.00× | 1.00× | 27,020 MB | 54,815 MB | 54.5 s |
| Q8_0, repack in a file, none | 1.00× | 1.00× | 1.00× | 1.00× | 1,459 MB | 54,817 MB | 62.2 s |
| Q8_0, repack in memory, 32 GB | 1.01× | 1.00× | 1.00× | 1.00× | 27,004 MB | 32,667 MB | 60.4 s |
| Q8_0, repack in a file, 32 GB | 0.99× | 0.94× | 1.00× | 0.98× | 1,451 MB | 32,654 MB | 99.4 s |
| Q8_0, repack in a file, 16 GB | 0.10× | 0.32× | 0.68× | 0.40× | 1,238 MB | 16,347 MB | 215.0 s |
| Q8_0, experts unrepacked, 16 GB | 0.06× | 0.19× | 0.48× | 0.28× | 3,654 MB | 16,345 MB | 162.3 s |
| Q4_0 experts, repack in a file, 16 GB | 0.93× | 0.91× | 0.95× | 0.93× | 1,487 MB | 16,327 MB | 71.5 s |
| IQ4_NL experts, repack in a file, 16 GB | 0.91× | 0.86× | 0.91× | 0.92× | 1,443 MB | 16,327 MB | 70.5 s |
| Q8_0, repack in a file, 10 GB | 0.04× | 0.18× | 0.56× | 0.21× | 1,238 MB | 10,215 MB | 229.1 s |
| Q8_0, experts unrepacked, 10 GB | 0.03× | 0.12× | 0.42× | 0.15× | 3,654 MB | 10,213 MB | 194.9 s |
| Q8_0, no repacking, 10 GB | 0.02× | 0.11× | 0.36× | 0.14× | 1,367 MB | 10,197 MB | 226.1 s |
| Q4_0 experts, repack in a file, 10 GB | 0.12× | 0.42× | 0.71× | 0.47× | 1,238 MB | 10,215 MB | 128.0 s |
| IQ4_NL experts, repack in a file, 10 GB | 0.12× | 0.34× | 0.70× | 0.47× | 1,239 MB | 10,215 MB | 127.5 s |
| IQ4_NL experts, experts unrepacked, 10 GB | 0.07× | 0.22× | 0.55× | 0.29× | 3,654 MB | 10,214 MB | 116.4 s |

The 16 GB and 10 GB rows with a repack file skipped the prefetch; the earlier runs with it, at
225.3 s and 275.0 s, ran at the same speed.

- **For full speed, give it 32 GB,** with either repack. The 48 GB estimated from its 54.8 GB peak
  isn't needed: the kernel drops the GGUF pages of weights already repacked.
- **For 16 GB at nearly full speed, quantize the experts to 4-bit** and repack into a file:
  0.86–0.95×, with the accuracy cost above.
- **At 10 GB it runs, but slowly:** 0.36–0.71× on a ~4.1k-token request, and as low as 0.02× on a
  short one, which is mostly re-reading weights.

## Lambda

Lambda gives a function at most 10,240 MB of memory, a 10 GB image and 10 GB of `/tmp`. E4B fits,
with a thin margin (see [graviton-cpu-variants.md](graviton-cpu-variants.md#memory)). The 26B-A4B's
27 GB GGUF fits neither the image nor `/tmp`.

**S3 Files can map it from S3, but it doesn't run.** The test function mounted an S3 Files file
system: files over 128 KiB aren't copied to the file system's own storage, so reads stream from
the bucket at the price of S3 GETs. It ran with 10,240 MB of memory, `DECIDER_MODEL_DIR` on the
mount, and no repack file, which wouldn't fit `/tmp`.

- **The function read at about 75 MB/s.** The Q8_0 prefetch, 32 readers in 4 MiB blocks, took
  370.8 s for 27 GB. From EC2 the same mount gave 77 MB/s with one reader and 800 MB/s with 16, so
  the function seems held to about one reader's rate.
- **Neither model got through its first invocation.**
  - Q8_0, with no repacking, was still loading and warming up at the 900 s timeout.
  - IQ4_NL experts, unrepacked with the rest repacked, skipped the prefetch (model read in
    13.4 s), but timed out the same way.
  - With 10 GB of memory, each forward pass re-reads most of the weights at 75 MB/s: about six
    minutes for Q8_0's, three for IQ4_NL's. The handler runs two warm-ups before the first
    request, and a timeout resets the environment, so the load starts over.
- **S3 Files on Lambda suits a model whose working set fits the function,** read once per
  environment. A model that's re-read on every request needs a local disk.
- **EFS isn't measured.** It serves reads from its own storage at about a millisecond, so one reader
  would likely be faster than S3 Files' streams from S3. If the 75 MB/s is the function's network
  limit, EFS would hit it too. With Elastic throughput it would also charge for every GB read,
  about $0.50–0.80 per 26B-A4B request.

For Lambda, E4B remains the model.

## Not yet tried

- **Pruning the rarely used experts,** measured against the release check: about 8 GB of the 26B-A4B,
  if accuracy holds (see [Which experts are used](#which-experts-are-used)).
- **Repacking only the busiest experts,** leaving the rest in the GGUF's layout, by splitting each
  layer's expert tensors into hot and cold parts. A typical layer's busiest 46 experts carry 90% of
  its routing, so this could keep most of the repack's speed for about 36% of its memory, when the
  model fits. It helps where a full repack file doesn't fit on disk or is too slow to write. Below
  the model's size it doesn't help: each request still touches many cold experts. A ggml change.
- **Local NVMe:** Graviton instances with instance storage (the `d` variants, such as c7gd or
  m8gd) would re-read weights several times faster than gp3, which is what limits every run below
  a model's size.
- **Memory-optimized instances** (r8g): more memory per vCPU avoids the problem, at a higher price
  per vCPU-hour.

## The switches

All are for measuring, on `feat/repack-file`.

| Switch | What it does |
| --- | --- |
| `GGML_CPU_REPACK_FILE_DIR=<dir>` | Repack into an unlinked file in `<dir>` (`patches/0004`) |
| `STRANDS_DECIDER_REPACK_CACHE=<dir>` | Repack into named files in `<dir>`, shared by every process and kept (`patches/0005`); sets the variable above and `GGML_CPU_REPACK_FILE_KEY` |
| `STRANDS_DECIDER_NO_EXPERT_REPACK=1` | Leave the expert weights unrepacked, mapped from the GGUF (`patches/0006`'s `GGML_CPU_REPACK_NO_EXPERTS`) |
| `STRANDS_DECIDER_NO_REPACK=1` | No repacking at all |
| `STRANDS_DECIDER_EXPERT_STATS=<path>` | Count each layer's expert routing, written to `<path>` as JSON at exit |
| `EXPERT_QUANT=Q4_0` or `IQ4_NL` (build argument) | Quantize the expert weights apart from the rest |
| `DECIDER_MODEL_DIR=<dir>` | Load the model from `<dir>` instead of the image's `/opt/model` |
