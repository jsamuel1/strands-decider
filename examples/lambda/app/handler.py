"""The System One API as an AWS Lambda function, served by llama.cpp (Q8_0) without torch.

Function URL requests get `strands-decider serve`'s API: POST /v1/systemone and GET /health,
with its request and response bodies and status codes. A direct invocation (the Invoke API,
Step Functions, an EventBridge target) passes a System One request as the event and gets the
response as the result; a request that is not valid fails the invocation with a ValueError.

The model loads once per execution environment. With SnapStart it loads at import, when a
version is published, and new environments restore from the snapshot. Without SnapStart it
loads in a background thread: Lambda gives an on-demand init 10 s, and re-runs an init that
overruns inside the first invocation, which would throw away 10 s of the load on every cold
start. Every invocation, /health included, waits for the load, so the one that started an
environment stays open until the model is ready; Lambda never freezes it partway through.

The engine module (numpy, tokenizers, pydantic and strands_decider) is imported by the load, in a
thread that runs beside the weight download, not at module import: nothing before the download
needs it, so the download starts without waiting for it and the import overlaps the download.
Request handling imports the schema after the load, when the engine has already loaded it.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import loader

if TYPE_CHECKING:
    from engine import HiddenStateEngine

    from strands_decider.schema import SystemOneRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
log = logging.getLogger("decider.app")

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = int(os.environ.get("THREADS", "0")) or (os.cpu_count() or 1)
# About 5 minutes on a Graviton3 host and 8 on a Graviton2, inside the 15-minute timeout. 0: no limit.
MAX_REQUEST_TOKENS = int(os.environ.get("MAX_REQUEST_TOKENS", "24576"))
# 1 refuses a prompt over the window with a 422, as `serve --strict-window` does; by
# default the state is trimmed to fit, as `serve` does.
STRICT_WINDOW = os.environ.get("STRICT_WINDOW", "0") == "1"
ROUTES = {"/health": "GET", "/v1/systemone": "POST"}


def _import_engine() -> Any:
    import engine

    return engine


def _s3_client() -> Any:
    return loader._client()  # noqa: SLF001  loader's own lazily built, shared S3 client


def _load() -> tuple[HiddenStateEngine, float]:
    t0 = time.perf_counter()
    # MODEL_PATH: a local GGUF, for local_test.sh; on Lambda the model comes from S3.
    if os.environ.get("MODEL_PATH"):
        gguf, engine_mod = os.environ["MODEL_PATH"], _import_engine()
    else:
        # The engine import is CPU work and the download is network-bound: run them side by side. The S3 client
        # (boto3's import and setup, also CPU work) is built first, so the import does not hold back the requests.
        _s3_client()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-import") as pool:
            imported = pool.submit(_import_engine)
            gguf = loader.download_to_memfd(os.environ["MODEL_BUCKET"], os.environ["MODEL_KEY"])
            engine_mod = imported.result()
    engine, _ = engine_mod.load_engine(
        os.path.join(HERE, "lib", "libdecider_hidden.so"),
        gguf,
        os.path.join(HERE, "assets"),
        THREADS,
        max_request_tokens=MAX_REQUEST_TOKENS or None,
        strict_window=STRICT_WINDOW,
    )
    engine.cfg.model_name = os.environ.get("MODEL_NAME") or f"{engine.cfg.model_name}-q8_0"
    # Pay the first-forward costs before the snapshot is taken.
    engine.evaluate(
        {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "A warm-up?"}}}
    )
    init_seconds = round(time.perf_counter() - t0, 2)
    log.info("model ready in %.1fs, threads=%d", init_seconds, THREADS)
    return engine, init_seconds


def _exit_if_failed(future: Future) -> None:
    # A failed background load would fail every invocation; exit so Lambda replaces the
    # environment, as it would after a failed init.
    if future.exception() is not None:
        log.error("model load failed", exc_info=future.exception())
        os._exit(1)


if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "on-demand":
    _ENGINE: Future = ThreadPoolExecutor(max_workers=1).submit(_load)
    _ENGINE.add_done_callback(_exit_if_failed)
else:  # snap-start, provisioned concurrency, or not on Lambda
    _ENGINE = Future()
    _ENGINE.set_result(_load())


def _cpu_part() -> str:
    """CPU part of the host serving this request.

    0xd0c is Graviton2, 0xd40 Graviton3 and 0xd4f Graviton4. Read per request, because a
    SnapStart snapshot can restore on any of them.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("CPU part"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def _evaluate(engine: HiddenStateEngine, request: SystemOneRequest) -> dict[str, Any]:
    started = time.perf_counter()
    payload = engine.evaluate(request).model_dump()
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "latency_ms=%s cpu_part=%s input_tokens=%s",
        payload["latency_ms"],
        _cpu_part(),
        payload["usage"]["input_tokens"],
    )
    return payload


def _response(status: int, body: Any) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "x-decider-cpu-part": _cpu_part()},
        "body": json.dumps(body, default=str),
    }


def _http(event: dict[str, Any]) -> dict[str, Any]:
    path = event.get("rawPath", "/")
    method = event["requestContext"]["http"]["method"]
    if path not in ROUTES:
        return _response(404, {"detail": "Not Found"})
    if method != ROUTES[path]:
        return _response(405, {"detail": "Method Not Allowed"})
    engine, init_seconds = _ENGINE.result()  # waits out a background load
    if path == "/health":
        return _response(
            200,
            {
                "status": "ok",
                "model": engine.cfg.model_name,
                "backend": "llama.cpp",
                "quant": "Q8_0",
                "threads": THREADS,
                "init_seconds": init_seconds,
                "max_length": engine.cfg.max_length,
                "strict_window": engine.cfg.strict_window,
                "max_request_tokens": MAX_REQUEST_TOKENS,
            },
        )
    from pydantic import ValidationError

    from strands_decider.schema import SystemOneRequest

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        request = SystemOneRequest.model_validate_json(raw)
    except ValidationError as e:  # FastAPI's shape, as `serve` answers
        errors = e.errors(include_url=False, include_context=False)
        return _response(422, {"detail": [{**err, "loc": ["body", *err["loc"]]} for err in errors]})
    try:
        return _response(200, _evaluate(engine, request))
    except ValueError as e:  # over the window or the request budget
        return _response(422, {"detail": str(e)})


def handler(event: Any, context: Any) -> dict[str, Any]:
    if isinstance(event, dict) and "http" in event.get("requestContext", {}):
        return _http(event)
    engine, _ = _ENGINE.result()
    from strands_decider.schema import SystemOneRequest  # loaded by the engine import

    return _evaluate(engine, SystemOneRequest.model_validate(event))
