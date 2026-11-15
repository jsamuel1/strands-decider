"""examples/lambda-kleidiai/e16: lazy model-stack imports. Zip builder, harness hook, placement and bench order.

No AWS and no llama.cpp. The tests pin E16's one-variable contract: the e6 and lazy zips differ only in handler.py,
app.py and their bytecode. PR #77's handler with E16's change imports engine off the init's main thread, beside the
download. The P1 harness patches engine wherever it is first imported, so it measures either placement unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
import types
import zipfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E16 = os.path.join(ROOT, "examples", "lambda-kleidiai", "e16")
BASE = os.path.join(ROOT, "examples", "lambda-kleidiai", "baseline")
APP = os.path.join(ROOT, "examples", "lambda", "app")
SNAP = os.path.join(ROOT, "examples", "lambda-kleidiai", "snapstart")
for p in (E16, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import phases  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)
MODS = (
    "lkai_entry",
    "engine",
    "loader",
    "handler",
    "app",
    "snapshot_restore_py",
    "hostinfo",
    "e16_entry",
)


@pytest.fixture(autouse=True)
def _clean_modules(monkeypatch):
    for m in MODS:
        monkeypatch.delitem(sys.modules, m, raising=False)
    yield
    for m in MODS:
        sys.modules.pop(m, None)


# ---------------------------------------------------------------------------- lazy_zip


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    """A tiny zip shaped like P1's: its own handler.py/app.py/lkai_entry.py (P1's bytes), a package, an asset."""
    path = tmp_path / "p1.zip"
    src = {
        # P1's handler: module-level engine and pydantic imports, as PR #77 ships it
        "handler.py": b"import loader\nfrom engine import HiddenStateEngine, load_engine\nfrom pydantic import ValidationError\n",
        "app.py": b"import os\nimport engine as _engine\nimport handler as _pr77\n",
        "lkai_entry.py": b"X = 0\n",
        "pkg/__init__.py": b"Y = 1\n",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in src.items():
            z.writestr(zipfile.ZipInfo(name, date_time=STAMP), data)
        z.writestr(zipfile.ZipInfo("assets/tokenizer.json", date_time=STAMP), b"{}")
        # a stale timestamp .pyc, as P1 ships 149 of them
        z.writestr(
            zipfile.ZipInfo(
                f"pkg/__pycache__/__init__.{sys.implementation.cache_tag}.pyc", date_time=STAMP
            ),
            importlib.util.MAGIC_NUMBER + b"\0\0\0\0" + b"\1\0\0\0" + b"\6\0\0\0",
        )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "model": {"id": "m"},
                "llama_cpp": {},
                "images": {},
                "toolchain": [],
                "assets_sha256": {},
                "function_zip": {
                    "bytes": path.stat().st_size,
                    "sha256": sha,
                    "handler": "lkai_entry.handler",
                },
            }
        )
    )
    return str(path), str(man)


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12), reason="lazy_zip compiles for Lambda's python3.12 only"
)
def test_zips_differ_only_in_import_placement(tmp_path):
    import lazy_zip

    p1, man = _p1_like_zip(tmp_path)
    m = lazy_zip.build(p1, man, str(tmp_path / "build"))
    with (
        zipfile.ZipFile(p1) as a,
        zipfile.ZipFile(tmp_path / "build" / "e6.zip") as e,
        zipfile.ZipFile(tmp_path / "build" / "lazy.zip") as z,
    ):
        assert set(e.namelist()) == set(z.namelist())
        for zz in (e, z):
            assert len(zz.namelist()) == len(set(zz.namelist()))  # no entry written twice
        differ = sorted(n for n in e.namelist() if e.read(n) != z.read(n))
        tag = sys.implementation.cache_tag
        assert differ == [
            f"__pycache__/app.{tag}.pyc",
            f"__pycache__/handler.{tag}.pyc",
            "app.py",
            "handler.py",
        ]
        # e6 keeps P1's placement files byte for byte; lazy takes this commit's
        for name in ("handler.py", "app.py", "pkg/__init__.py", "assets/tokenizer.json"):
            assert e.read(name) == a.read(name)
        assert z.read("handler.py") == open(os.path.join(APP, "handler.py"), "rb").read()
        assert z.read("app.py") == open(os.path.join(SNAP, "app.py"), "rb").read()
        # the harness is the same in both, and replaces P1's lkai_entry
        for name in ("lkai_entry.py", "hostinfo.py", "e16_entry.py"):
            assert (
                e.read(name)
                == z.read(name)
                == open(os.path.join(ROOT, lazy_zip.HARNESS[name]), "rb").read()
            )
        # E6's change in both arms: complete unchecked-hash bytecode
        for zz in (e, z):
            pycs = [n for n in zz.namelist() if n.endswith(".pyc")]
            assert len(pycs) == sum(n.endswith(".py") for n in zz.namelist())
            assert all(int.from_bytes(zz.read(n)[4:8], "little") == 1 for n in pycs)
    assert m["zips"]["e6"]["bytecode"]["stale"] == 0
    assert m["function_zip"]["sha256"] == m["zips"]["lazy"]["sha256"]


# ---------------------------------------------------------------------------- the deployment's placement


def _stub_loader(events: list, release: threading.Event):
    loader = types.ModuleType("loader")
    loader._client = lambda: events.append(("client", threading.current_thread().name))

    def download_to_memfd(bucket, key):
        events.append(("download_start", threading.current_thread().name))
        assert release.wait(5)
        events.append(("download_end", "engine" in sys.modules))
        return "/proc/self/fd/0"

    loader.download_to_memfd = download_to_memfd
    return loader


def _engine_module_file(tmp_path, events: list) -> None:
    """An engine.py on disk: its import is recorded with the thread that ran it, then releases the download."""
    (tmp_path / "engine.py").write_text(
        "import threading, sys\n"
        "sys._e16_events.append(('engine_import', threading.current_thread().name))\n"
        "sys._e16_release.set()\n"
        "class HeadRowsEngine:\n"
        "    def __init__(self):\n"
        "        self.cfg = type('C', (), {'model_name': 'm'})()\n"
        "    def evaluate(self, req):\n"
        "        return {'answers': {}}\n"
        "class Shim:\n"
        "    def __init__(self, *a, **k):\n"
        "        self.lib = type('L', (), {})()\n"
        "def load_engine(*a, **k):\n"
        "    Shim()\n"
        "    return HeadRowsEngine(), None\n"
    )


def test_handler_imports_engine_beside_the_download(tmp_path, monkeypatch):
    events: list = []
    release = threading.Event()
    monkeypatch.setattr(sys, "_e16_events", events, raising=False)
    monkeypatch.setattr(sys, "_e16_release", release, raising=False)
    _engine_module_file(tmp_path, events)
    monkeypatch.setitem(sys.modules, "loader", _stub_loader(events, release))
    monkeypatch.syspath_prepend(APP)
    monkeypatch.syspath_prepend(str(tmp_path))  # first: APP has PR #77's own engine.py
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "on-demand")
    monkeypatch.setenv("MODEL_BUCKET", "b")
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    import handler

    assert "engine" not in sys.modules or dict(events).get("engine_import") != "MainThread"
    handler._ENGINE.result(timeout=10)
    kinds = [k for k, _ in events]
    assert kinds.index("client") < kinds.index("download_start") < kinds.index("download_end")
    assert dict(events)["engine_import"] != "MainThread"  # off the init's main thread
    assert (
        dict(events)["download_end"] is True
    )  # the download waited for nothing; the import overlapped it
    assert "pydantic" not in handler.__dict__ and "SystemOneRequest" not in handler.__dict__


def test_app_does_not_import_engine_at_module_level(monkeypatch):
    stub = types.ModuleType("handler")
    monkeypatch.setitem(sys.modules, "handler", stub)
    monkeypatch.syspath_prepend(SNAP)
    import app

    assert "engine" not in sys.modules
    assert "_engine" not in app.__dict__


# ---------------------------------------------------------------------------- the harness hook


def _harness_tree(tmp_path, eager_engine: bool):
    """app.py/handler.py stand-ins on disk; engine.py imported eagerly by app (P1, e6 arm) or by the load (lazy)."""
    (tmp_path / "app.py").write_text(
        ("import engine\n" if eager_engine else "") + "import handler as _pr77\n"
        "def handler(event, context):\n"
        "    eng, _ = _pr77._ENGINE.result()\n"
        "    return {'latency_ms': 1.0, 'answers': eng.evaluate(event)['answers'], 'usage': {}}\n"
    )
    (tmp_path / "handler.py").write_text(
        "from concurrent.futures import ThreadPoolExecutor\n"
        "import loader\n"
        "def _load():\n"
        "    loader._client()\n"
        "    loader.download_to_memfd('b', 'k')\n"
        "    import engine\n"
        "    eng, _ = engine.load_engine()\n"
        "    eng.evaluate({})\n"
        "    return eng, 1.0\n"
        "_ENGINE = ThreadPoolExecutor(1).submit(_load)\n"
        "def _cpu_part():\n"
        "    return '0xd40'\n"
    )


@pytest.mark.parametrize("eager", [True, False], ids=["e6", "lazy"])
def test_harness_patches_engine_wherever_it_is_imported(tmp_path, monkeypatch, capsys, eager):
    events: list = []
    release = threading.Event()
    release.set()
    monkeypatch.setattr(sys, "_e16_events", events, raising=False)
    monkeypatch.setattr(sys, "_e16_release", release, raising=False)
    _engine_module_file(tmp_path, events)
    _harness_tree(tmp_path, eager)
    monkeypatch.setitem(sys.modules, "loader", _stub_loader(events, release))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(BASE)
    monkeypatch.setenv("LKAI_PHASES", "1")
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "on-demand")
    meta_path = list(sys.meta_path)
    import lkai_entry

    out = lkai_entry.handler(
        {}, types.SimpleNamespace(aws_request_id="r", function_version="$LATEST")
    )
    rec = phases.parse_lkai(capsys.readouterr().out)[-1]
    init, counters = rec["init"], rec["counters"]
    for ph in (
        "imports",
        "engine_import",
        "load_start",
        "weight_download",
        "shim_init",
        "engine_load",
        "warmup_inference",
        "import_total",
    ):
        assert ph in init, ph
    thread = counters["engine_import_thread"]
    assert (thread == "MainThread") is eager
    assert out["lkai"]["inv"]["class"] == "cold"
    # the hook removed itself once it had found engine
    assert [f for f in sys.meta_path if type(f).__name__ == "_EngineHook"] == []
    assert len(sys.meta_path) == len(meta_path)


def test_harness_patches_a_preloaded_engine(tmp_path, monkeypatch, capsys):
    """An engine already in sys.modules (P1's tests' stand-ins) is patched at once, with no hook installed."""
    events: list = []
    release = threading.Event()
    release.set()
    monkeypatch.setattr(sys, "_e16_events", events, raising=False)
    monkeypatch.setattr(sys, "_e16_release", release, raising=False)
    _engine_module_file(tmp_path, events)
    _harness_tree(tmp_path, True)
    monkeypatch.setitem(sys.modules, "loader", _stub_loader(events, release))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(BASE)
    monkeypatch.setenv("LKAI_PHASES", "1")
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "on-demand")
    importlib.import_module("engine")
    import lkai_entry

    lkai_entry.handler({}, types.SimpleNamespace(aws_request_id="r", function_version="$LATEST"))
    rec = phases.parse_lkai(capsys.readouterr().out)[-1]
    assert "shim_init" in rec["init"] and "engine_import" not in rec["init"]


# ---------------------------------------------------------------------------- e16_entry


def test_e16_entry_records_stack_and_host(monkeypatch):
    stub = types.ModuleType("lkai_entry")
    stub.REC = types.SimpleNamespace(restored=False)
    stub.handler = lambda e, c: {"answers": {}, "lkai": {"inv": {"n": 1}}}
    monkeypatch.setitem(sys.modules, "lkai_entry", stub)
    monkeypatch.syspath_prepend(E16)
    monkeypatch.setenv("E16_ARM", "lazy")
    import e16_entry

    out = e16_entry.handler({}, types.SimpleNamespace(log_stream_name="s"))
    rec = out["lkai"]["e16"]
    assert rec["arm"] == "lazy" and rec["log_stream"] == "s" and rec["host_read"] == "init"
    assert set(rec["main_thread_stack_at_init_end"]) == set(e16_entry.STACK)
    assert rec["host"]["schema"] == "lkai-host/1"
    out2 = e16_entry.handler({}, types.SimpleNamespace(log_stream_name="s"))
    assert "e16" not in out2["lkai"]  # first invocation only


def test_stack_loaded_counts_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, "lkai_entry", types.ModuleType("lkai_entry"))
    monkeypatch.syspath_prepend(E16)
    import e16_entry

    fake = {"numpy": 1, "numpy.linalg": 1, "pydantic": 1, "json": 1, "engine": 1}
    got = e16_entry.stack_loaded(fake)
    assert (
        got["numpy"] == 2 and got["pydantic"] == 1 and got["engine"] == 1 and got["tokenizers"] == 0
    )
    assert "json" not in got


# ---------------------------------------------------------------------------- bench order


def test_submission_interleaves_arms_and_alternates_first(monkeypatch):
    if importlib.util.find_spec("boto3") is None:  # the bench imports the SDK at module level
        exc = types.ModuleType("botocore.exceptions")
        exc.ClientError = type("ClientError", (Exception,), {})
        cfg = types.ModuleType("botocore.config")
        cfg.Config = lambda **k: None
        boto3 = types.ModuleType("boto3")
        boto3.Session, boto3.__version__ = (lambda **k: None), "stand-in"
        for name, mod in {
            "boto3": boto3,
            "botocore": types.ModuleType("botocore"),
            "botocore.exceptions": exc,
            "botocore.config": cfg,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
    for m in ("deploy", "bench", "e6bench", "e6deploy", "e16bench", "e16deploy"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.syspath_prepend(os.path.join(ROOT, "examples", "lambda-kleidiai", "e6"))
    import e16bench

    assert e16bench.submission(["e6", "lazy"], 2, 0) == [
        ("e6", 0),
        ("lazy", 0),
        ("e6", 1),
        ("lazy", 1),
    ]
    assert e16bench.submission(["e6", "lazy"], 2, 1) == [
        ("lazy", 0),
        ("e6", 0),
        ("lazy", 1),
        ("e6", 1),
    ]
