"""lambda-kleidiai results store: validate, publish and render lkai-result/1 runs. Standard library only.

One file per experiment run (results/schema.json), stored at

    <store>/runs/<experiment_id>/<run_id>/result.json
    <store>/runs/<experiment_id>/<run_id>/artifacts/<path>      raw logs, invocations, profiles

Nothing under runs/ is shared: a publisher writes only its own run's prefix, and refuses to replace a run that
already exists with different bytes. The dashboard (<store>/site/index.html and one page per run) is derived:
`render_site` rebuilds all of it from every result.json in the store, so publishers never edit a shared file and a
lost race is repaired by the next regeneration.

The store is a local directory or s3://bucket/prefix (S3Store imports boto3 lazily; nothing else needs it).
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import re
import statistics
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "schema.json")
SCHEMA_ID = "lkai-result/1"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# An AWS ARN or bare id that names an account: committed and published results must not carry one.
ACCOUNT_RE = re.compile(r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:\d{12}:|\b\d{12}\b")
STATUSES = ("baseline", "improved", "no-change", "regressed", "inconclusive", "failed")
CLASSES = ("cold", "restore", "warm")
REQUIRED_PHASES_BY_CLASS = {
    "cold": {
        "weight_download",
        "model_load_mmap",
        "model_ready",
        "first_request_inference",
    },
    "restore": {"report_restore_duration", "first_request_inference"},
    "warm": {"steady_state_inference"},
}
# The overview's headline phase per start class, and the brief's phases in display order.
HEADLINE = {
    "cold": "first_request_total",
    "restore": "first_request_total",
    "warm": "steady_state_inference",
}
PHASE_ORDER = (
    "first_request_total",
    "report_init_duration",
    "report_restore_duration",
    "report_billed_restore_duration",
    "restore_to_invoke",
    "runtime_to_import",
    "imports",
    "weight_download",
    "weight_download_mib_s",
    "ggml_backend_load",
    "model_load_mmap",
    "context_init",
    "assets_load",
    "warmup_inference",
    "model_ready",
    "import_total",
    "first_request_inference",
    "steady_state_inference",
    "handler_total",
    "wait_for_load",
    "report_duration",
    "report_billed_duration",
    "report_max_memory_mb",
    "client_round_trip",
)


# ---------------------------------------------------------------------------- distributions


def pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile, as baseline/summarize.py computes it."""
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def histogram(xs: list[float], bins: int = 12) -> dict:
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return {"edges": [lo, hi], "counts": [len(xs)]}
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        counts[min(bins - 1, int((x - lo) / w))] += 1
    return {"edges": [round(lo + i * w, 6) for i in range(bins + 1)], "counts": counts}


def dist(xs: Iterable[float | None], unit: str = "s") -> dict | None:
    """The lkai-result/1 distribution block. Identical to baseline/summarize.py's, so experiments that do not
    use summarize.py still publish comparable numbers."""
    v = [x for x in xs if x is not None]
    if not v:
        return None
    r = lambda x: round(x, 6)  # noqa: E731
    return {
        "unit": unit,
        "n": len(v),
        "p50": r(pct(v, 50)),
        "p90": r(pct(v, 90)),
        "p99": r(pct(v, 99)),
        "min": r(min(v)),
        "max": r(max(v)),
        "mean": r(statistics.fmean(v)),
        "stdev": r(statistics.stdev(v)) if len(v) > 1 else 0.0,
        "histogram": histogram(v),
    }


def is_dist(v: Any) -> bool:
    return isinstance(v, dict) and "p50" in v and "n" in v


# ---------------------------------------------------------------------------- validation


class ValidationError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _schema_errors(v: Any, s: dict, root: dict, path: str) -> list[str]:
    """The JSON Schema subset schema.json uses. Returns problems as 'path: message'."""
    errs: list[str] = []
    if "$ref" in s:
        name = s["$ref"].removeprefix("#/$defs/")
        errs += _schema_errors(v, root["$defs"][name], root, path)
    if "type" in s:
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_TYPES[t](v) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(v).__name__}"]
    if "const" in s and v != s["const"]:
        errs.append(f"{path}: must be {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not one of {s['enum']}")
    if isinstance(v, str):
        if len(v) < s.get("minLength", 0):
            errs.append(f"{path}: empty")
        if "pattern" in s and not re.search(s["pattern"], v):
            errs.append(f"{path}: {v!r} does not match {s['pattern']}")
    if _TYPES["number"](v) and "minimum" in s and v < s["minimum"]:
        errs.append(f"{path}: {v} < {s['minimum']}")
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0):
            errs.append(f"{path}: needs at least {s['minItems']} item(s)")
        if "items" in s:
            for i, x in enumerate(v):
                errs += _schema_errors(x, s["items"], root, f"{path}[{i}]")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing {k!r}")
        if len(v) < s.get("minProperties", 0):
            errs.append(f"{path}: needs at least {s['minProperties']} entr(y/ies)")
        props = s.get("properties", {})
        for k, x in v.items():
            if k in props:
                errs += _schema_errors(x, props[k], root, f"{path}.{k}")
            elif isinstance(s.get("additionalProperties"), dict):
                errs += _schema_errors(x, s["additionalProperties"], root, f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected {k!r}")
    if "anyOf" in s:
        branches = [_schema_errors(v, b, root, path) for b in s["anyOf"]]
        if all(branches):
            errs.append(f"{path}: matches no allowed shape ({min(branches, key=len)[0]})")
    for sub in s.get("allOf", []):
        errs += _schema_errors(v, sub, root, path)
    if "if" in s and not _schema_errors(v, s["if"], root, path) and "then" in s:
        errs += _schema_errors(v, s["then"], root, path)
    return errs


def _walk_dists(phases: dict) -> Iterator[tuple[str, dict]]:
    for arm, classes in (phases or {}).items():
        for cls, ph in (classes or {}).items():
            for name, v in (ph or {}).items():
                if is_dist(v):
                    yield f"phases.{arm}.{cls}.{name}", v
                elif isinstance(v, dict):
                    for sub, d in v.items():
                        if is_dist(d):
                            yield f"phases.{arm}.{cls}.{name}.{sub}", d


def _strings(v: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(v, str):
        yield path, v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _strings(x, f"{path}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from _strings(x, f"{path}[{i}]")


def _numbers(v: Any, path: str = "$") -> Iterator[tuple[str, int | float]]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        yield path, v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _numbers(x, f"{path}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from _numbers(x, f"{path}[{i}]")


def safe_relpath(p: str) -> bool:
    n = os.path.normpath(p)
    return bool(p) and not os.path.isabs(p) and n != ".." and not n.startswith("../")


def validate(res: Any, schema: dict | None = None) -> list[str]:
    """Every problem with one run file: schema shape, then the semantic checks the schema cannot express."""
    schema = schema or load_schema()
    errs = _schema_errors(res, schema, schema, "$")
    if errs or not isinstance(res, dict):
        return errs
    for path, n in _numbers(res):
        if isinstance(n, float) and not math.isfinite(n):
            errs.append(f"{path}: number must be finite")
    for path, d in _walk_dists(res.get("phases", {})):
        if not all(math.isfinite(d[k]) for k in ("p50", "p90", "p99", "min", "max")):
            errs.append(f"{path}: distribution values must be finite")
            continue
        if not d["min"] <= d["p50"] <= d["p90"] <= d["p99"] <= d["max"]:
            errs.append(f"{path}: percentiles out of order (min<=p50<=p90<=p99<=max)")
        h = d.get("histogram")
        if h:
            if len(h["edges"]) != len(h["counts"]) + 1:
                errs.append(f"{path}.histogram: needs len(edges) == len(counts) + 1")
            if sum(h["counts"]) != d["n"]:
                errs.append(f"{path}.histogram: counts sum to {sum(h['counts'])}, n is {d['n']}")
            if any(b < a for a, b in zip(h["edges"], h["edges"][1:], strict=False)):
                errs.append(f"{path}.histogram: edges must not decrease")
    e = res["errors"]
    if e["count"] < len(e["rows"]):
        errs.append("$.errors: count is less than the number of rows")
    if res["status"] == "baseline" and res["changed_variable"] is not None:
        errs.append("$.changed_variable: a baseline changes nothing (null)")
    if res["status"] != "failed":
        phases = res.get("phases") or {}
        for arm_name, arm in phases.items():
            for cls in CLASSES:
                start_class = arm.get(cls)
                if not isinstance(start_class, dict):
                    errs.append(f"$.phases.{arm_name}: missing {cls!r} start class")
                    continue
                for name in sorted(REQUIRED_PHASES_BY_CLASS[cls] - start_class.keys()):
                    errs.append(f"$.phases.{arm_name}.{cls}: missing required phase {name!r}")
            cold = arm.get("cold") or {}
            max_memory = cold.get("report_max_memory_mb")
            if not (is_dist(max_memory) and max_memory.get("unit") == "MB"):
                errs.append(
                    f"$.phases.{arm_name}.cold: report_max_memory_mb MB distribution is required"
                )
        conditions = res.get("conditions") or {}
        if not conditions.get("request_ids"):
            errs.append("$.conditions.request_ids: needs at least one fixed request id")
        if not conditions.get("started") or not conditions.get("finished"):
            errs.append("$.conditions: started and finished timestamps are required")
    for name, p in (res.get("artifacts") or {}).items():
        if not safe_relpath(p):
            errs.append(f"$.artifacts.{name}: {p!r} must be relative to the run directory")
    for i, a in enumerate(res.get("profiling_artifacts") or []):
        if "path" not in a and "url" not in a:
            errs.append(f"$.profiling_artifacts[{i}]: needs path or url")
        if "path" in a and not safe_relpath(a["path"]):
            errs.append(f"$.profiling_artifacts[{i}].path: must be relative")
    for path, s in _strings(res):
        if path.startswith("$.published"):
            continue  # the store's own location; written by publish, not by the experiment
        if ACCOUNT_RE.search(s):
            errs.append(f"{path}: looks like an AWS account id; keep account ids out of results")
    return errs


def check(res: Any) -> None:
    problems = validate(res)
    if problems:
        raise ValidationError(problems)


# ---------------------------------------------------------------------------- stores


class LocalStore:
    """A directory with the same layout as the S3 store. Used by tests and for a local preview."""

    def __init__(self, root: str):
        self.root = root

    def uri(self, key: str) -> str:
        return "file://" + os.path.abspath(os.path.join(self.root, key))

    def link(self, key: str) -> str:
        return self.uri(key)

    def get(self, key: str) -> bytes | None:
        try:
            with open(os.path.join(self.root, key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        if_absent: bool = False,
    ) -> None:
        """Atomic replace, or with `if_absent` atomic create that raises KeyExists if the key is taken."""
        p = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp-{os.getpid()}-{threading.get_ident()}"
        with open(tmp, "wb") as f:
            f.write(data)
        if not if_absent:
            os.replace(tmp, p)
            return
        try:
            os.link(tmp, p)  # fails if p exists: create-only, like S3 If-None-Match: *
        except FileExistsError:
            raise KeyExists(key) from None
        finally:
            os.unlink(tmp)

    def list(self, prefix: str) -> list[str]:
        base = os.path.join(self.root, prefix)
        out = []
        for d, _, files in os.walk(base):
            for f in files:
                if ".tmp-" in f:
                    continue  # an in-flight write, not an object
                out.append(os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/"))
        return sorted(out)


class S3Store:
    """s3://bucket/prefix. Private bucket: links are S3 console object URLs (need account access)."""

    def __init__(self, url: str, region: str | None = None, profile: str | None = None):
        m = re.match(r"^s3://([^/]+)/?(.*)$", url)
        if not m:
            raise ValueError(f"not an s3:// URL: {url}")
        self.bucket, self.prefix = m.group(1), m.group(2).strip("/")
        import boto3  # only the S3 store needs it

        sess = boto3.Session(profile_name=profile, region_name=region)
        self.region = region or sess.region_name
        self.s3 = sess.client("s3", region_name=self.region)

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._k(key)}"

    def link(self, key: str) -> str:
        return (
            f"https://{self.region}.console.aws.amazon.com/s3/object/{self.bucket}"
            f"?region={self.region}&prefix={self._k(key)}"
        )

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        if_absent: bool = False,
    ) -> None:
        """PutObject with a SHA-256 checksum S3 verifies before storing, so the object is exactly `data`.
        With `if_absent`, a conditional create (If-None-Match: *) that raises KeyExists if the key is taken."""
        kw = {
            "Bucket": self.bucket,
            "Key": self._k(key),
            "Body": data,
            "ContentType": content_type,
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(data).digest()).decode(),
        }
        if if_absent:
            kw["IfNoneMatch"] = "*"
        try:
            self.s3.put_object(**kw)
        except (
            Exception
        ) as e:  # botocore ClientError; matched on its response so tests need no botocore
            code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code")
            if if_absent and code in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise KeyExists(key) from None
            raise

    def list(self, prefix: str) -> list[str]:
        out = []
        pages = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self._k(prefix)
        )
        for page in pages:
            for o in page.get("Contents", []):
                out.append(o["Key"][len(self.prefix) + 1 :] if self.prefix else o["Key"])
        return sorted(out)


def open_store(url: str, region: str | None = None, profile: str | None = None):
    return S3Store(url, region, profile) if url.startswith("s3://") else LocalStore(url)


def run_prefix(experiment_id: str, run_id: str) -> str:
    for v in (experiment_id, run_id):
        if not ID_RE.match(v):
            raise ValueError(f"unsafe id {v!r}")
    return f"runs/{experiment_id}/{run_id}"


def load_runs(store) -> list[dict]:
    runs = []
    for key in store.list("runs/"):
        if key.count("/") == 3 and key.endswith("/result.json"):
            data = store.get(key)
            if data is not None:
                runs.append(json.loads(data))
    return runs


# ---------------------------------------------------------------------------- publish


class PublishConflict(RuntimeError):
    pass


class KeyExists(PublishConflict):
    """A create-only put found the key already taken (another publisher won the race)."""


class RegenerateDidNotSettle(RuntimeError):
    """The run set kept changing while rendering; the page may be missing a run, so it is not published."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _ctype(path: str) -> str:
    for ext, t in (
        (".json", "application/json"),
        (".jsonl", "application/x-ndjson"),
        (".html", "text/html; charset=utf-8"),
        (".txt", "text/plain; charset=utf-8"),
        (".md", "text/markdown; charset=utf-8"),
        (".svg", "image/svg+xml"),
        (".gz", "application/gzip"),
    ):
        if path.endswith(ext):
            return t
    return "application/octet-stream"


def _artifact_files(run_dir: str, rel: str) -> list[str]:
    root = os.path.realpath(run_dir)
    p = os.path.join(run_dir, rel)
    resolved = os.path.realpath(p)
    if os.path.commonpath((root, resolved)) != root:
        raise ValueError(f"artifact {rel!r} resolves outside the run directory {run_dir}")

    def checked(path: str) -> str:
        if os.path.commonpath((root, os.path.realpath(path))) != root:
            raise ValueError(f"artifact {path!r} resolves outside the run directory {run_dir}")
        return os.path.relpath(path, run_dir).replace(os.sep, "/")

    if os.path.isdir(p):
        return sorted(checked(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    if os.path.isfile(p):
        return [checked(p)]
    return []


@dataclass
class PublishReport:
    run_key: str
    uploaded: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    result_uri: str = ""


def publish_run(
    res: dict,
    run_dir: str | None,
    store,
    extra: Iterable[str] = (),
    replace: bool = False,
) -> PublishReport:
    """Upload one run's artifacts and result.json under runs/<experiment_id>/<run_id>/ and nowhere else.

    Idempotent: an object already holding the same bytes is skipped, so re-running publish is safe. Validation
    requires every listed raw artifact to exist before any object is uploaded. A run whose result.json already
    exists with different content is refused unless `replace` (run ids are meant to be unique; overwriting
    someone else's measurements by accident is the failure this prevents)."""
    check(res)
    prefix = run_prefix(res["experiment_id"], res["run_id"])
    rep = PublishReport(run_key=f"{prefix}/result.json")
    res = json.loads(json.dumps(res))  # never mutate the caller's copy
    rels: list[str] = []
    listed = list((res.get("artifacts") or {}).values()) + [
        a["path"] for a in res.get("profiling_artifacts") or [] if "path" in a
    ]
    if listed and not run_dir:
        raise ValueError("run_dir is required to publish listed artifacts")
    if run_dir:
        for rel in listed:
            files = _artifact_files(run_dir, rel)
            (rels.extend(files) if files else rep.missing.append(rel))
        for x in extra:
            rel = os.path.relpath(x, run_dir)
            if not safe_relpath(rel):
                raise ValueError(f"{x} is outside the run directory {run_dir}")
            rels.extend(_artifact_files(run_dir, rel) or [rel])
    if rep.missing:
        raise FileNotFoundError("required artifact(s) missing: " + ", ".join(sorted(rep.missing)))
    artifact_data = {}
    links = {}
    for rel in sorted(set(rels)):
        key = f"{prefix}/artifacts/{rel}"
        with open(os.path.join(run_dir or ".", rel), "rb") as f:
            data = f.read()
        artifact_data[key] = data
        links[rel] = {"uri": store.uri(key), "link": store.link(key), "sha256": _sha(data)}
    res["published"] = {
        "result": store.uri(rep.run_key),
        "result_link": store.link(rep.run_key),
        "artifacts": links,
        "missing_artifacts": [],
    }
    body = (json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    old_result = store.get(rep.run_key)
    if old_result is not None and _sha(old_result) != _sha(body) and not replace:
        raise PublishConflict(
            f"{store.uri(rep.run_key)} already exists with different content; "
            "use a new run_id, or --replace if you mean to correct this run"
        )

    for key, data in artifact_data.items():
        old = store.get(key)
        if old is not None and _sha(old) == _sha(data):
            rep.unchanged.append(key)
        elif replace:
            store.put(key, data, _ctype(key))
            rep.uploaded.append(key)
        else:
            # Create-only, like result.json below: a publisher that loses a race on this run id must not overwrite
            # the winner's artifacts, whose sha256 the winner's result.json already records.
            try:
                store.put(key, data, _ctype(key), if_absent=True)
            except KeyExists:
                if _sha(store.get(key) or b"") != _sha(data):
                    raise PublishConflict(
                        f"{store.uri(key)} already holds different bytes; "
                        "use a new run_id, or --replace if you mean to correct this run"
                    ) from None
                rep.unchanged.append(key)
            else:
                rep.uploaded.append(key)
    if old_result is not None and _sha(old_result) == _sha(body):
        rep.unchanged.append(rep.run_key)
    elif old_result is None and not replace:
        # Create-only: two publishers racing on one run id cannot both win; the loser is refused, not overwritten.
        try:
            store.put(rep.run_key, body, "application/json", if_absent=True)
        except KeyExists:
            if _sha(store.get(rep.run_key) or b"") != _sha(body):
                raise PublishConflict(
                    f"{store.uri(rep.run_key)} was created by another publisher with different content; "
                    "use a new run_id"
                ) from None
            rep.unchanged.append(rep.run_key)
        else:
            rep.uploaded.append(rep.run_key)
    else:
        store.put(rep.run_key, body, "application/json")
        rep.uploaded.append(rep.run_key)
    rep.result_uri = store.uri(rep.run_key)
    return rep


def _fingerprint(runs: list[dict]) -> list[tuple[str, str]]:
    """Run keys plus content hashes: a --replace correction changes the input even when the key set does not."""
    return sorted((run_key(r), _sha(json.dumps(r, sort_keys=True).encode())) for r in runs)


def regenerate(store, out_dir: str | None = None, include_fixtures: bool | None = None) -> dict:
    """Rebuild every page from every result.json in the store, upload site/, and write the combined single-file
    dashboard to out_dir (for Artifactory).

    site/ is written only from a render whose input (run keys and content) was unchanged between loading it and
    rendering it, so a render that never settles writes nothing. After writing, the run set is read once more: if a
    run landed meanwhile, its publisher may already have written newer pages that ours replaced, so we render and
    write again. RegenerateDidNotSettle is raised, and out_dir is not written, when 5 attempts never end on a
    confirmed-current write; the caller then exits 4 and `--regenerate-only` repairs site/ once publishers quiesce."""
    runs = load_runs(store)
    for _ in range(5):
        seen = _fingerprint(runs)
        site = render_site(runs, include_fixtures=include_fixtures)
        latest = load_runs(store)
        if _fingerprint(latest) != seen:
            runs = latest  # someone published meanwhile: render again so their run is on the page
            continue
        # Publish only a render whose input was observed stable: a refusal leaves site/ untouched.
        for name, page in site.items():
            store.put(f"site/{name}", page.encode(), "text/html; charset=utf-8")
        latest = load_runs(store)
        if _fingerprint(latest) == seen:
            break
        # A run landed while we wrote. Its publisher may already have written newer pages that ours just replaced,
        # so render again rather than leave stale pages as the last write to site/.
        runs = latest
    else:
        raise RegenerateDidNotSettle(
            "run set changed on each of 5 renders; rerun with --regenerate-only once publishers quiesce"
        )
    keys = {run_key(r) for r in runs}
    if out_dir:
        for name, page in site.items():
            p = os.path.join(out_dir, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(page)
    return {"runs": sorted(keys), "pages": sorted(site)}


# ---------------------------------------------------------------------------- rendering

CSS = """
:root{--fg:#1d2330;--mut:#5b6474;--bd:#d9dee7;--bg:#fff;--alt:#f5f7fa;--ok:#1a7f37;--bad:#b42318;
--warn:#9a6700;--na:#57606a;--acc:#0b5cad}
*{box-sizing:border-box}body{font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);
background:var(--bg);margin:0;padding:24px;max-width:1400px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--bd);
padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
.mut{color:var(--mut)}code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
table{border-collapse:collapse;width:100%;margin:6px 0 12px}th,td{border:1px solid var(--bd);padding:4px 6px;
text-align:left;vertical-align:top}th{background:var(--alt);font-weight:600}td.n{text-align:right;
font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.s-baseline{background:var(--acc)}.s-improved{background:var(--ok)}.s-regressed{background:var(--bad)}
.s-no-change{background:var(--na)}.s-inconclusive{background:var(--warn)}.s-failed{background:var(--bad)}
.fixture{background:#fff4ce;border:1px solid #e6c200;padding:8px 12px;border-radius:6px;margin:10px 0}
.fx{background:#e6c200;color:#3b2f00}.failed-row td{background:#fff1f0}.null-row td{background:#f6f8fa}
.better{color:var(--ok);font-weight:600}.worse{color:var(--bad);font-weight:600}
.kv{display:grid;grid-template-columns:220px 1fr;gap:2px 12px}.kv div:nth-child(odd){color:var(--mut)}
pre{background:var(--alt);padding:8px;border-radius:4px;overflow:auto;white-space:pre-wrap}
section.run{border-top:3px solid var(--bd);margin-top:32px;padding-top:4px}
nav a{margin-right:10px}svg text{font:10px system-ui,sans-serif;fill:#5b6474}
"""


def esc(v: Any) -> str:
    """HTML-escaped text, with AWS account ids masked: shown text never names an account (link targets may)."""
    return html.escape(re.sub(r"\b\d{12}\b", "<account>", "" if v is None else str(v)), quote=True)


def esc_attr(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def safe_url(u: str | None) -> str | None:
    if u and re.match(r"^(https?://|s3://|file://|#|[A-Za-z0-9._/-]+\.html(#.*)?$)", u):
        return u
    return None


def a(href: str | None, text: Any) -> str:
    u = safe_url(href)
    return f'<a href="{esc_attr(u)}">{esc(text)}</a>' if u else esc(text)


def run_key(r: dict) -> str:
    return f"{r['experiment_id']}/{r['run_id']}"


def anchor(r: dict) -> str:
    key = run_key(r)
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", key)
    return f"run-{slug}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def primary_arm(r: dict) -> str | None:
    ph = r.get("phases") or {}
    if r.get("primary_arm") in ph:
        return r["primary_arm"]
    if "inst" in ph:
        return "inst"
    return next(iter(ph), None)


def phase(r: dict, cls: str, name: str) -> dict | None:
    arm = primary_arm(r)
    v = ((r.get("phases") or {}).get(arm) or {}).get(cls, {}) if arm else {}
    v = (v or {}).get(name)
    return v if is_dist(v) else None


NA = "&ndash;"  # rendered as HTML; an en dash in source trips RUF001


def fmt(v: float | None, unit: str = "s") -> str:
    if v is None:
        return NA
    if unit == "s":
        return f"{v * 1000:.0f} ms" if abs(v) < 1 else f"{v:.3f} s"
    return f"{v:,.4g} {unit}"


def delta(v: float | None, base: float | None, lower_is_better: bool = True) -> str:
    if v is None or base in (None, 0):
        return ""
    d = 100 * (v - base) / base
    good = d < 0 if lower_is_better else d > 0
    cls = "better" if good and abs(d) >= 0.05 else ("worse" if abs(d) >= 0.05 else "")
    return f' <span class="{cls}">({d:+.1f}%)</span>'


def pick_baseline(runs: list[dict]) -> dict | None:
    """The newest real baseline run; a fixture baseline only when no real one exists."""
    bases = [r for r in runs if r["status"] == "baseline"]
    real = [r for r in bases if not r.get("fixture")]
    pool = real or bases
    return (
        max(pool, key=lambda r: (r.get("conditions") or {}).get("started") or r["run_id"])
        if pool
        else None
    )


def svg_hist(d: dict, base: dict | None = None, w: int = 260, h: int = 70) -> str:
    """Histogram bars, with p50/p90/p99 ticks; the baseline's p50 as a dashed line on the same axis."""
    hist = d.get("histogram") or {"edges": [d["min"], d["max"]], "counts": [d["n"]]}
    edges, counts = hist["edges"], hist["counts"]
    lo, hi = edges[0], edges[-1]
    if base:
        lo, hi = min(lo, base["min"]), max(hi, base["max"])
    span = (hi - lo) or 1.0
    x = lambda v: 4 + (w - 8) * (v - lo) / span  # noqa: E731
    top = max(counts) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 14}" role="img" '
        f'aria-label="histogram n={d["n"]}">'
    ]
    for i, c in enumerate(counts):
        x0, x1 = x(edges[i]), x(edges[i + 1]) if i + 1 < len(edges) else x(edges[-1])
        bh = (h - 6) * c / top
        parts.append(
            f'<rect x="{x0:.1f}" y="{h - bh:.1f}" width="{max(1.0, x1 - x0 - 1):.1f}" '
            f'height="{bh:.1f}" fill="#7aa7d9"><title>{c}</title></rect>'
        )
    for k, col in (("p50", "#0b5cad"), ("p90", "#9a6700"), ("p99", "#b42318")):
        parts.append(
            f'<line x1="{x(d[k]):.1f}" x2="{x(d[k]):.1f}" y1="0" y2="{h}" stroke="{col}" '
            f'stroke-width="1.5"><title>{k} {d[k]}</title></line>'
        )
    if base:
        parts.append(
            f'<line x1="{x(base["p50"]):.1f}" x2="{x(base["p50"]):.1f}" y1="0" y2="{h}" '
            f'stroke="#1d2330" stroke-dasharray="3,2"><title>baseline p50 {base["p50"]}</title></line>'
        )
    u = d.get("unit", "")
    parts.append(f'<text x="4" y="{h + 11}">{esc(f"{lo:.4g}")}{esc(u)}</text>')
    parts.append(
        f'<text x="{w - 4}" y="{h + 11}" text-anchor="end">{esc(f"{hi:.4g}")}{esc(u)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _phase_names(classes: dict) -> list[str]:
    names = [n for n in PHASE_ORDER if n in classes]
    return names + sorted(n for n in classes if n not in names and n not in ("n",))


def _phase_rows(ph: dict, bph: dict, prefix: str = "", is_base: bool = False) -> list[str]:
    rows = []
    for name in _phase_names(ph):
        v = ph[name]
        if isinstance(v, dict) and not is_dist(v):
            rows += _phase_rows(v, (bph or {}).get(name) or {}, f"{prefix}{name}.", is_base)
            continue
        label = esc(prefix + name)
        if v is None:
            rows.append(
                f'<tr class="null-row"><td>{label}</td><td colspan="9">not measured (null)</td></tr>'
            )
            continue
        if not is_dist(v):
            continue
        b = (bph or {}).get(name) if not is_base else None
        b = b if is_dist(b) else None
        u = v.get("unit", "s")
        lb = u != "MiB/s"
        cells = "".join(
            f'<td class="n">{fmt(v[k], u)}{delta(v[k], b[k] if b else None, lb) if k in ("p50", "p90", "p99") else ""}</td>'
            for k in ("p50", "p90", "p99", "min", "max")
        )
        rows.append(
            f'<tr><td>{label}</td><td class="n">{v["n"]}</td>{cells}'
            f'<td class="n">{fmt(v.get("stdev"), u)}</td><td class="n">{fmt(b["p50"], u) if b else NA}</td>'
            f"<td>{svg_hist(v, b)}</td></tr>"
        )
    return rows


def _kv(pairs: list[tuple[str, Any]]) -> str:
    return (
        '<div class="kv">'
        + "".join(f"<div>{esc(k)}</div><div>{v}</div>" for k, v in pairs)
        + "</div>"
    )


def _changed(r: dict) -> str:
    cv = r.get("changed_variable")
    if cv is None:
        return "— (baseline)" if r["status"] == "baseline" else "<em>not stated</em>"
    if isinstance(cv, dict):
        extra = ", ".join(f"{k}: {v}" for k, v in cv.items() if k != "name")
        return esc(cv.get("name")) + (f' <span class="mut">({esc(extra)})</span>' if extra else "")
    return esc(cv)


def _badge(r: dict) -> str:
    out = f'<span class="badge s-{esc(r["status"])}">{esc(r["status"])}</span>'
    if r.get("fixture"):
        out += ' <span class="badge fx">FIXTURE</span>'
    return out


def _json_block(v: Any) -> str:
    return f"<pre>{esc(json.dumps(v, indent=1, sort_keys=True))}</pre>"


def render_run(r: dict, base: dict | None, href=None) -> str:
    """The detail section for one run. Every brief field is shown, and a missing one says so."""
    is_base = base is not None and run_key(r) == run_key(base)
    b = None if is_base else base
    dep, model, cond = r.get("deployment") or {}, r.get("model") or {}, r.get("conditions") or {}
    pub = r.get("published") or {}
    out = [f'<section class="run" id="{anchor(r)}">']
    out.append(f"<h2>{esc(r['experiment_id'])} / {esc(r['run_id'])} {_badge(r)}</h2>")
    if r.get("fixture"):
        out.append('<div class="fixture">Synthetic fixture data: not a measurement.</div>')
    if r["status"] == "failed":
        out.append(
            '<div class="fixture" style="background:#fff1f0;border-color:#b42318">'
            "<b>Failed run.</b> Published so the attempt and its cause are on record.</div>"
        )
    out.append(
        _kv(
            [
                ("Hypothesis", esc(r.get("hypothesis"))),
                ("Changed variable", _changed(r)),
                ("Conclusion", esc(r.get("conclusion")) or "<em>none recorded</em>"),
                (
                    "Compared against",
                    a((href or (lambda x: f"#{anchor(x)}"))(b), run_key(b))
                    if b
                    else ("this is the baseline" if is_base else "<em>no baseline published</em>"),
                ),
                (
                    "Experiment commit",
                    f"<code>{esc((r.get('commits') or {}).get('experiment'))}</code>",
                ),
                (
                    "Baseline commit",
                    f"<code>{esc((r.get('commits') or {}).get('baseline'))}</code>"
                    + (
                        " <b>(dirty tree at build)</b>"
                        if (r.get("commits") or {}).get("repo_dirty_at_build")
                        else ""
                    ),
                ),
                ("Region / AZ", f"{esc(dep.get('region'))} / {esc(dep.get('az'))}"),
                (
                    "Runtime",
                    f"{esc(dep.get('runtime'))} <span class='mut'>{esc(dep.get('runtime_version_arn'))}</span>",
                ),
                (
                    "Architecture / memory",
                    f"{esc(dep.get('architecture'))} / {esc(dep.get('memory_mb'))} MB",
                ),
                ("SnapStart", esc(json.dumps(dep.get("snapstart"))) if "snapstart" in dep else NA),
                (
                    "Functions",
                    esc(", ".join(f"{k}={v}" for k, v in (dep.get("functions") or {}).items())),
                ),
                (
                    "CPU parts observed",
                    esc(json.dumps(dep.get("cpu_parts_observed")))
                    if dep.get("cpu_parts_observed")
                    else NA,
                ),
                (
                    "Model",
                    f"{esc(model.get('id'))} @ {esc(model.get('revision'))}, {esc(model.get('format'))}"
                    + (
                        f", {model['gguf_bytes'] / 2**30:.2f} GiB"
                        if isinstance(model.get("gguf_bytes"), int)
                        else ""
                    ),
                ),
                (
                    "Conditions",
                    esc(
                        ", ".join(
                            f"{k}={cond[k]}"
                            for k in ("rounds", "concurrency", "warm_per_arm_per_round", "arms")
                            if k in cond
                        )
                    ),
                ),
                ("Requests", esc(", ".join(cond.get("request_ids") or [])) or NA),
                ("Weight fetch", esc(cond.get("weight_fetch"))),
                ("Started / finished", f"{esc(cond.get('started'))} / {esc(cond.get('finished'))}"),
                ("Result file", a(pub.get("result_link"), pub.get("result") or "not published")),
            ]
        )
    )
    cl = r.get("classification") or {}
    out.append("<h3>Start-class classification</h3>")
    out.append(
        _kv(
            [
                ("Method", esc(cl.get("method"))),
                ("Expected → observed", esc(json.dumps(cl.get("expected_vs_observed")))),
                ("Mismatches (excluded)", esc(cl.get("mismatches"))),
            ]
        )
        if cl
        else "<p class='mut'>none recorded</p>"
    )
    phases = r.get("phases") or {}
    bph_all = (b or {}).get("phases") or {}
    barm = primary_arm(b) if b else None
    if not phases:
        out.append("<h3>Latency</h3><p><b>No latency measured.</b></p>")
    for arm, classes in phases.items():
        for cls in CLASSES + tuple(c for c in classes if c not in CLASSES):
            ph = classes.get(cls)
            if not ph:
                continue
            bph = (bph_all.get(arm) or bph_all.get(barm) or {}).get(cls) or {}
            out.append(
                f"<h3>Arm <code>{esc(arm)}</code> · {esc(cls)} <span class='mut'>(n={esc(ph.get('n'))})</span></h3>"
            )
            out.append(
                "<table><tr><th>phase</th><th>n</th><th>p50</th><th>p90</th><th>p99</th><th>min</th><th>max</th>"
                "<th>stdev</th><th>baseline p50</th><th>distribution (bars; p50 blue, p90 amber, p99 red; "
                "baseline p50 dashed)</th></tr>"
                + "".join(_phase_rows(ph, bph, is_base=is_base))
                + "</table>"
            )
    if r.get("variance"):
        out.append("<h3>Run-to-run variance (per-round medians)</h3>" + _json_block(r["variance"]))
    if r.get("instrumentation_overhead"):
        out.append("<h3>Instrumentation overhead</h3>" + _json_block(r["instrumentation_overhead"]))
    errs = r.get("errors") or {}
    out.append(f"<h3>Errors: {esc(errs.get('count'))}</h3>")
    if errs.get("rows"):
        out.append(
            _json_block(errs["rows"][:50])
            + (
                f"<p class='mut'>{len(errs['rows']) - 50} more in result.json</p>"
                if len(errs["rows"]) > 50
                else ""
            )
        )
    out.append("<h3>Correctness vs baseline</h3>")
    c = r.get("correctness")
    out.append(_json_block(c) if c else "<p><b>Not compared</b> (correctness is null).</p>")
    out.append("<h3>Cost and resources</h3>")
    out.append(_json_block(r["cost"]) if r.get("cost") else "<p class='mut'>none recorded</p>")
    out.append("<h3>Raw artifacts and profiles</h3><ul>")
    links = pub.get("artifacts") or {}
    for name, rel in (r.get("artifacts") or {}).items():
        hits = [k for k in links if k == rel or k.startswith(rel.rstrip("/") + "/")]
        if not hits:
            out.append(
                f"<li>{esc(name)}: <code>{esc(rel)}</code> <span class='worse'>not uploaded</span></li>"
            )
        for k in hits:
            out.append(
                f"<li>{esc(name)}: {a(links[k]['link'], k)} <span class='mut mono'>{esc(links[k]['uri'])}</span></li>"
            )
    for p in r.get("profiling_artifacts") or []:
        if "url" in p:
            out.append(
                f"<li>profile {esc(p['name'])} ({esc(p.get('tool'))}): {a(p['url'], p['url'])}</li>"
            )
        else:
            lk = links.get(os.path.normpath(p["path"]))
            out.append(
                f"<li>profile {esc(p['name'])} ({esc(p.get('tool'))}): "
                + (
                    a(lk["link"], p["path"])
                    if lk
                    else f"<code>{esc(p['path'])}</code> <span class='worse'>not uploaded</span>"
                )
                + "</li>"
            )
    if not links and not r.get("profiling_artifacts"):
        out.append("<li><b>none published</b></li>")
    out.append("</ul>")
    if pub.get("missing_artifacts"):
        out.append(
            f"<p class='worse'>Listed but missing at publish time: {esc(', '.join(pub['missing_artifacts']))}</p>"
        )
    out.append("<h3>Reproduce</h3><pre>" + esc("\n".join(r.get("reproduce") or [])) + "</pre>")
    out.append("</section>")
    return "\n".join(out)


def _overview(runs: list[dict], base: dict | None, href) -> str:
    head = (
        "<table><tr><th>experiment / run</th><th>status</th><th>changed variable</th>"
        "<th>cold first request p50</th><th>restore first request p50</th><th>warm inference p50</th>"
        "<th>weight download p50</th><th>errors</th><th>correctness</th><th>cost USD</th>"
        "<th>conclusion</th></tr>"
    )
    rows = []
    for r in runs:
        is_base = base is not None and run_key(r) == run_key(base)
        b = None if is_base else base
        cells = []
        for cls, name in (
            ("cold", HEADLINE["cold"]),
            ("restore", HEADLINE["restore"]),
            ("warm", HEADLINE["warm"]),
            ("cold", "weight_download"),
        ):
            d = phase(r, cls, name)
            bd = phase(b, cls, name) if b else None
            if d:
                n = d["n"]
                dd = delta(d["p50"], bd["p50"] if bd else None)
                cells.append(
                    f'<td class="n">{fmt(d["p50"])}{dd}<br><span class="mut">n={n}</span></td>'
                )
            else:
                cells.append('<td class="n mut">not measured</td>')
        c = r.get("correctness")
        corr = (
            f"{c['bit_identical']}/{c['compared']} identical, {c['choice_flips']} flips, max Δ {c['max_abs_delta']:.3g}"
            if c
            else "not compared"
        )
        cost = (r.get("cost") or {}).get("total_usd")
        cls = (
            "failed-row"
            if r["status"] == "failed"
            else ("null-row" if r["status"] in ("no-change", "inconclusive") else "")
        )
        rows.append(
            f'<tr class="{cls}"><td>{a(href(r), run_key(r))}</td><td>{_badge(r)}</td><td>{_changed(r)}</td>'
            + "".join(cells)
            + f'<td class="n">{esc((r.get("errors") or {}).get("count"))}</td><td>{esc(corr)}</td>'
            f'<td class="n">{esc(cost) if cost is not None else NA}</td><td>{esc(r.get("conclusion")) or "<em>none</em>"}</td></tr>'
        )
    return head + "".join(rows) + "</table>"


def _order(runs: list[dict], base: dict | None) -> list[dict]:
    return sorted(
        runs,
        key=lambda r: (
            0 if base and run_key(r) == run_key(base) else 1,
            STATUSES.index(r["status"]) if r["status"] in STATUSES else 99,
            r["experiment_id"],
            r["run_id"],
        ),
    )


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<base target="_blank">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>\n"
    )


def render_site(runs: list[dict], include_fixtures: bool | None = None) -> dict[str, str]:
    """{file name: HTML}. `dashboard.html` is the single self-contained page (overview + every run's detail,
    in-page links); `index.html` is the overview linking `<experiment>/<run>.html` detail pages. No external
    resources: inline CSS and SVG only, so it renders inside Artifactory's sandboxed iframe or from S3.

    Fixture runs are shown only when no real run exists (include_fixtures=None), always labelled."""
    real = [r for r in runs if not r.get("fixture")]
    if include_fixtures is None:
        include_fixtures = not real
    shown = runs if include_fixtures else real
    base = pick_baseline(shown)
    ordered = _order(shown, base)
    title = "lambda-kleidiai experiments: results dashboard"
    counts = {s: sum(r["status"] == s for r in ordered) for s in STATUSES}
    banner = ""
    if any(r.get("fixture") for r in ordered):
        banner = (
            '<div class="fixture"><b>FIXTURE DATA.</b> Rows marked FIXTURE are synthetic examples of the format, '
            "not measurements. They disappear from this page once a real run is published.</div>"
        )
    if base is None:
        banner += '<div class="fixture"><b>No baseline published yet.</b> Deltas need a status=baseline run (P1).</div>'
    hidden = len(runs) - len(shown)
    head = (
        f"<h1>{esc(title)}</h1><p class='mut'>{len(ordered)} run(s): "
        + ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        + (f"; {hidden} fixture run(s) hidden" if hidden else "")
        + ". Generated from the results store; each run name links to its detail, which links its result.json and raw artifacts. "
        "Percentiles are nearest-rank; with n&lt;100, p99 is the maximum. Deltas compare p50 against the baseline's "
        "same phase, same start class.</p>"
        + banner
        + ("<p>Baseline: " + esc(run_key(base)) + "</p>" if base else "")
    )
    single = head + "<h2>Overview</h2>" + _overview(ordered, base, lambda r: f"#{anchor(r)}")
    single += (
        "<nav>"
        + "".join(f'<a href="#{anchor(r)}">{esc(run_key(r))}</a>' for r in ordered)
        + "</nav>"
    )
    single += "".join(render_run(r, base) for r in ordered)
    site = {"dashboard.html": _page(title, single)}
    site["index.html"] = _page(
        title,
        head
        + "<h2>Overview</h2>"
        + _overview(ordered, base, lambda r: f"{r['experiment_id']}/{r['run_id']}.html"),
    )
    for r in ordered:
        site[f"{r['experiment_id']}/{r['run_id']}.html"] = _page(
            f"{run_key(r)} · lambda-kleidiai",
            '<p><a href="../index.html">← all experiments</a></p>'
            + render_run(r, base, href=lambda x: f"../{x['experiment_id']}/{x['run_id']}.html"),
        )
    return site
