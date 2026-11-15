"""Lambda entry point (handler "app.handler") of strands-decider-2b-v2 and -v2q: PR #77's handler.py, unchanged, plus
two things around it.

1. {"mode": "native_eval", "in": "s3://.../shard.jsonl.gz", "out": "s3://.../out/shard.jsonl.gz"} runs eval_np.run_shard
   on the loaded engine (the deployed image's eval mode, torch-free); the result is the shard's stats.
2. A direct invocation's response also carries "timing" {"engine_s", "init_s", "threads", "cpu"}, the deployed
   handler's field, so client/drive.py and bench/lambda_shim.py run unchanged. engine_s is PR #77's latency_ms / 1000
   (engine.evaluate), and cpu is the CPU part of the host serving this request (a SnapStart snapshot restores on any).
Everything else, including the function-URL routes, the SnapStart load at import and the memfd weights, is PR #77's.

v2b: OPENBLAS_NUM_THREADS defaults to 1, set before numpy loads. numpy's OpenBLAS otherwise starts one worker per
vCPU for the pointer head's float64 products, and its idle workers busy-wait after every call, on the cores llama.cpp's
6 OpenMP threads need for the next forward: on Graviton3 at 6 threads that added about 33 ms to every forward of the
eval mode (0.490 vs 0.456 s p50 at <= 512 tokens, the deployed engine 0.456 s). One BLAS thread costs the head about
1 ms and gives the same log-probs. An OPENBLAS_NUM_THREADS in the function's environment still wins.

E16: engine (numpy and the rest of the model stack) is not imported here. PR #77's handler imports it inside the load,
beside the weight download, and the eval mode imports it once the load has finished.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import handler as _pr77  # noqa: E402


def handler(event, context):
    if isinstance(event, dict) and event.get("mode") == "native_eval":
        eng, _ = _pr77._ENGINE.result()
        import engine as _engine  # imported by the load, which has finished
        from eval_np import run_shard
        return run_shard(eng, _engine.SHIM, event["in"], event["out"], _pr77._cpu_part())
    out = _pr77.handler(event, context)
    if isinstance(out, dict) and "latency_ms" in out and "statusCode" not in out:
        _, init_s = _pr77._ENGINE.result()
        out["timing"] = {"engine_s": round(out["latency_ms"] / 1000, 4), "init_s": init_s, "threads": _pr77.THREADS,
                         "cpu": _pr77._cpu_part()}
    return out
