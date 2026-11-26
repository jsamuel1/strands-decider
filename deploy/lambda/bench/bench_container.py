"""The decision function's image, benchmarked in place: on EC2 (docker run), on ECS (a task), or
anywhere the image runs. It imports the Lambda handler, so the model, the shim and its CPU
backend are exactly the function's, and runs a request set through it with a fixed number of
threads (Lambda at 10 GB has 6 vCPUs). It prints one JSON line per request, and with --s3 uploads
them; the first line is the host and the load.

  docker run --rm --cpuset-cpus 0-5 -m 10g -e BENCH_THREADS=6 \\
      -v $PWD:/bench --entrypoint python <image> /bench/bench_container.py \\
      --requests /bench/requests.json --label variants --rounds 3

requests.json is graviton_bench.py's mix: `python graviton_bench.py requests golden_vectors.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--requests", required=True, help="a JSON list of {name, rid, request}, or s3://"
    )
    p.add_argument("--label", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--s3", help="s3://bucket/key for the results")
    args = p.parse_args()

    threads = int(os.environ.get("BENCH_THREADS", "6"))
    # The handler sizes llama.cpp's threads by os.cpu_count(), which on a host bigger than the
    # function counts every core, not the container's.
    os.cpu_count = lambda: threads  # type: ignore[assignment]
    sys.path.insert(0, os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
    sys.path.append("/var/runtime")  # boto3, as the Lambda base image ships it

    requests = json.loads(_read(args.requests))
    started = time.perf_counter()
    import handler  # loads the model

    model = handler._model.result()
    load_s = time.perf_counter() - started
    rows = [
        {
            "label": args.label,
            "name": "host",
            "cpu_part": handler._CPU_PART,
            "cpu_features": handler._CPU_FEATURES,
            "cpu_backend": model.llama.cpu_backend,
            "threads": threads,
            "load_s": round(load_s, 1),
            "instance": _instance_type(),
            "platform": platform.platform(),
        }
    ]
    print(json.dumps(rows[0]), flush=True)
    handler.handler(requests[0]["request"], None)  # warm: the first request pages the model in
    for rnd in range(args.rounds):
        for req in requests:
            t = time.perf_counter()
            payload = handler.handler(req["request"], None)
            row = {
                "label": args.label,
                "round": rnd,
                "name": req["name"],
                "rid": req["rid"],
                "wall_ms": round((time.perf_counter() - t) * 1000, 1),
                "latency_ms": payload["latency_ms"],
                "input_tokens": payload["usage"]["input_tokens"],
                "timings_ms": model.llama.last_timings,
                "answers": _numbers(payload["answers"]),
                "cpu_part": handler._CPU_PART,
                "cpu_backend": model.llama.cpu_backend,
                "instance": rows[0]["instance"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.s3:
        import boto3

        bucket, key = args.s3[5:].split("/", 1)
        body = "".join(json.dumps(r) + "\n" for r in rows).encode()
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)


def _read(path: str) -> str:
    if path.startswith("s3://"):
        import boto3

        bucket, key = path[5:].split("/", 1)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    with open(path) as f:
        return f.read()


def _numbers(answers: dict) -> dict[str, float]:
    out = {}
    for q, a in answers.items():
        if a.get("type") == "noul":
            out[q] = a["noul"]
        for o, v in (a.get("probabilities") or {}).items():
            out[f"{q}.{o}"] = v
    return out


def _instance_type() -> str:
    """The EC2 instance type, the Fargate task's CPU architecture, or "?"."""
    import urllib.request

    meta = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    try:
        if meta:
            with urllib.request.urlopen(meta + "/task", timeout=2) as r:
                task = json.load(r)
            return f"{task.get('LaunchType', 'ECS')}:{task.get('Limits', {})}"
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            token = r.read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode()
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
