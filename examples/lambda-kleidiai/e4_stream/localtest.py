"""Box test of E4's zip as Lambda runs it, with no AWS: the weight download reads a local GGUF through a stand-in S3
client that serves byte ranges at a fixed bandwidth, so a streamed load really overlaps a slow download.

Run inside the pinned Lambda python3.12 image with the unzipped function at /task and the GGUF at /model:

  E4_STREAM=0|1 [STREAM_ORDER=arrival|file] [E4_FAKE_MIB_S=80] python3.12 localtest.py /model/s2b-h512-Q8_0.gguf out.json

The process imports e4_entry exactly as the Lambda runtime would (AWS_LAMBDA_INITIALIZATION_TYPE=snap-start, so the
load runs at import), then answers every request in baseline/requests.json. out.json holds the answers, the phase
record and the stream counters. Running it for each arm and comparing the answers checks that a streamed load gives
bit-identical outputs; the phase records show the overlap. The bandwidth is shared by all ranges, as a Lambda
environment's network is.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

# Import the package's modules (PYTHONPATH=/task), never the source copies beside this script.
sys.path[:] = [
    p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))
]

GGUF, OUT = sys.argv[1], sys.argv[2]
MIB_S = float(os.environ.get("E4_FAKE_MIB_S", "80"))
REQS = os.environ.get("E4_REQUESTS", "/recipe/baseline/requests.json")


class _Bucket:
    """A token bucket shared by every range: MIB_S MiB/s in total."""

    def __init__(self, rate: float):
        self.rate, self.t, self.lock = rate, time.monotonic(), threading.Lock()

    def take(self, n: int) -> None:
        with self.lock:
            self.t = max(self.t, time.monotonic()) + n / self.rate
            delay = self.t - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class _Body:
    def __init__(self, a: int, b: int, bucket: _Bucket):
        self.a, self.b, self.bucket = a, b, bucket

    def iter_chunks(self, n: int):
        with open(GGUF, "rb") as f:
            f.seek(self.a)
            left = self.b - self.a + 1
            while left:
                # botocore's StreamingBody.iter_chunks(n) returns n-byte reads (PR #77 asks for 8 MiB); the same
                # allocation sizes matter for the process's memory peak
                k = min(n, left)
                self.bucket.take(k)
                yield f.read(k)
                left -= k


class LocalS3:
    def __init__(self, rate: float):
        self.bucket = _Bucket(rate * 2**20)

    def head_object(self, Bucket, Key):
        return {"ContentLength": os.path.getsize(GGUF)}

    def get_object(self, Bucket, Key, Range):
        a, b = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        return {"Body": _Body(a, b, self.bucket)}


os.environ.setdefault("MODEL_BUCKET", "local")
os.environ.setdefault("MODEL_KEY", os.path.basename(GGUF))
os.environ.setdefault("AWS_LAMBDA_INITIALIZATION_TYPE", "snap-start")
os.environ.setdefault("LKAI_PHASES", "1")
for k, v in {"DEC_KV": "f32", "DEC_UBATCH": "512", "N_SEQ": "17"}.items():
    os.environ.setdefault(k, v)

import loader  # noqa: E402

if (
    os.environ.get("E4_REAL_S3") != "1"
):  # E4_REAL_S3=1: PR #77's own boto3 client, real S3 (MODEL_BUCKET/KEY)
    _s3 = LocalS3(MIB_S)
    loader._client = lambda: _s3  # noqa: SLF001  (the stand-in S3 client)

_peak: dict = {}
_stop = threading.Event()


def _proc_mem() -> dict:
    """Current memory of this process (kB), from /proc/self/status; memfd pages count as RssShmem once mapped."""
    out = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"):
            out[k] = int(v.split()[0])
    return out


def _sample() -> None:
    """Peak of each /proc/self/status memory field, sampled every 50 ms."""
    while not _stop.is_set():
        m = _proc_mem()
        for k, v in m.items():
            _peak[k] = max(_peak.get(k, 0), v)
        _stop.wait(0.05)


threading.Thread(target=_sample, daemon=True).start()

t0 = time.monotonic()
import e4_entry  # noqa: E402  (loads the model)
import lkai_entry  # noqa: E402

import_s = time.monotonic() - t0
reqs = json.load(open(REQS))
answers, lat = {}, {}
for r in reqs["requests"]:
    out = e4_entry.handler(r["request"], None)
    answers[r["id"]] = out["answers"]
    lat[r["id"]] = out["latency_ms"]
_stop.set()

rec = {
    "arm": {
        "E4_STREAM": os.environ.get("E4_STREAM", "0"),
        "STREAM_ORDER": os.environ.get("STREAM_ORDER"),
        "MALLOC_MMAP_THRESHOLD_": os.environ.get("MALLOC_MMAP_THRESHOLD_"),
    },
    "fake_mib_s": MIB_S,
    "import_and_load_s": round(import_s, 3),
    "mem_kb": _proc_mem(),
    "mem_peak_kb": dict(_peak),
    "init": lkai_entry.REC.init,
    "counters": lkai_entry.REC.counters,
    "latency_ms": lat,
    "answers": answers,
}
json.dump(rec, open(OUT, "w"), indent=1, default=str)
print(
    json.dumps(
        {k: rec[k] for k in ("arm", "import_and_load_s", "mem_kb", "mem_peak_kb")}
        | {"init": rec["init"]}
    ),
    flush=True,
)
print("LOCALTEST_DONE", flush=True)
