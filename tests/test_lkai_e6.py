"""examples/lambda-kleidiai/e6: the bytecode zip builder, the host-metadata wrapper and the bench's arm order.

No AWS and no llama.cpp: pyc_zip.py runs on a synthetic P1-shaped zip, and e6_entry's parsers run on canned /proc
text. The tests pin E6's one-variable contract: ctl is P1's zip plus the wrapper, byte for byte. pyc differs only
in bytecode, which the 3.12 loader accepts without checking the source.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import struct
import sys
import types
import zipfile

import pytest

E6 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e6")
)
BASE = os.path.abspath(os.path.join(E6, "..", "baseline"))
sys.path.insert(0, E6)
sys.path.insert(0, BASE)

import pyc_zip  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)


def _p1_like_zip(tmp_path) -> tuple[str, str]:
    """A tiny zip shaped like P1's: .py stamped 2026-01-01, one stale timestamp .pyc, one .py without bytecode, a
    non-Python asset and a native library stand-in."""
    path = tmp_path / "p1.zip"
    src = {"pkg/__init__.py": b"X = 1\n", "pkg/mod.py": b"def f():\n    return 41 + 1\n", "top.py": b"Y = 2\n"}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in src.items():
            z.writestr(zipfile.ZipInfo(name, date_time=STAMP), data)
        # a timestamp .pyc whose mtime field is the build time, not the zip's 2026-01-01 stamp
        stale = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, 1792482575, len(src["top.py"])) + b"\0"
        z.writestr(zipfile.ZipInfo(f"__pycache__/top.{sys.implementation.cache_tag}.pyc", date_time=STAMP), stale)
        z.writestr(zipfile.ZipInfo("assets/tokenizer.json", date_time=STAMP), b"{}")
        z.writestr(zipfile.ZipInfo("lib/libfake.so", date_time=STAMP), b"\x7fELF")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "schema": "lkai-build-manifest/1",
                "repo_commit": "0" * 40,
                "repo_dirty": False,
                "model": {"id": "m", "revision": "r", "format": "GGUF Q8_0", "gguf_bytes": 1, "gguf_sha256": "x"},
                "llama_cpp": {},
                "images": {},
                "toolchain": [],
                "assets_sha256": {},
                "function_zip": {"bytes": path.stat().st_size, "sha256": sha, "handler": "lkai_entry.handler"},
            }
        )
    )
    return str(path), str(man)


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="pyc_zip compiles for Lambda's python3.12 only")
def test_ctl_is_p1_plus_wrapper_and_pyc_differs_only_in_bytecode(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    out = tmp_path / "build"
    m = pyc_zip.build(p1, man, str(out))
    with zipfile.ZipFile(p1) as a, zipfile.ZipFile(out / "ctl.zip") as c, zipfile.ZipFile(out / "pyc.zip") as p:
        an = {i.filename: i for i in a.infolist()}
        cn = {i.filename: i for i in c.infolist()}
        assert set(cn) - set(an) == {pyc_zip.WRAPPER}
        for name, info in an.items():  # every P1 entry, bytes and metadata
            assert c.read(name) == a.read(name)
            assert (cn[name].date_time, cn[name].external_attr) == (info.date_time, info.external_attr)
        pn = set(p.namelist())
        for name in an:
            if not name.endswith(".pyc"):
                assert p.read(name) == a.read(name)
        assert {n for n in pn - set(an) if not n.endswith((".pyc", "/"))} == {pyc_zip.WRAPPER}
        assert p.read(pyc_zip.WRAPPER) == c.read(pyc_zip.WRAPPER)
        # one .pyc per P1 .py, all unchecked-hash; the wrapper has none in either zip
        pycs = [n for n in pn if n.endswith(".pyc")]
        assert len(pycs) == 3
        for n in pycs:
            flags = int.from_bytes(p.read(n)[4:8], "little")
            assert flags == 1, n  # hash-based (bit 0), unchecked (bit 1 clear)
        assert not any("e6_entry" in n for n in pycs)
    assert m["bytecode"]["p1"] == {"pyc": 1, "valid": 0, "stale": 1, "no_source": 0, "py": 3, "py_without_pyc": 2}
    assert m["bytecode"]["e6_pyc"]["stale"] == 0 and m["bytecode"]["e6_pyc"]["valid"] == 3
    assert m["bytecode"]["failed"] == []
    assert m["base"].endswith(json.loads(open(man).read())["function_zip"]["sha256"])
    assert m["function_zip"]["sha256"] == m["zips"]["pyc"]["sha256"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="pyc_zip compiles for Lambda's python3.12 only")
def test_pyc_zip_bytecode_imports_without_source_check(tmp_path):
    """The shipped bytecode is used by the loader even when the source's mtime and size no longer match."""
    p1, man = _p1_like_zip(tmp_path)
    out = tmp_path / "build"
    pyc_zip.build(p1, man, str(out))
    root = tmp_path / "task"
    with zipfile.ZipFile(out / "pyc.zip") as z:
        z.extractall(root)
    # Prove the bytecode, not the source, is what runs: change the source behind it.
    (root / "pkg" / "mod.py").write_text("def f():\n    return -1  # never compiled\n")
    sys.path.insert(0, str(root))
    try:
        for k in [k for k in sys.modules if k == "pkg" or k.startswith("pkg.")]:
            del sys.modules[k]
        mod = importlib.import_module("pkg.mod")
        assert mod.f() == 42
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if k == "pkg" or k.startswith("pkg.")]:
            del sys.modules[k]


def test_build_refuses_a_zip_the_manifest_does_not_describe(tmp_path):
    p1, man = _p1_like_zip(tmp_path)
    with open(p1, "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit, match="is not the zip"):
        pyc_zip.build(p1, man, str(tmp_path / "build"))


def test_stale_pyc_counts_a_matching_timestamp_pyc_as_valid(tmp_path):
    path = tmp_path / "z.zip"
    import calendar

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("a.py", date_time=STAMP), b"A = 1\n")
        mt = calendar.timegm((*STAMP, 0, 0, -1))
        head = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, mt, 6)
        z.writestr(zipfile.ZipInfo(f"__pycache__/a.{sys.implementation.cache_tag}.pyc", date_time=STAMP), head)
    with zipfile.ZipFile(path) as z:
        assert pyc_zip.stale_pyc(z) == {"pyc": 1, "valid": 1, "stale": 0, "no_source": 0, "py": 1, "py_without_pyc": 0}


# ---------------------------------------------------------------------------- e6_entry (no lkai_entry import)


def _entry_module(monkeypatch):
    """e6_entry with P1's lkai_entry replaced by a stand-in, so its import needs no model."""
    stub = types.ModuleType("lkai_entry")
    stub.handler = lambda event, context: {"answers": {"q": {"choice": "a"}}, "latency_ms": 1.0, "lkai": {"inv": {}}}
    monkeypatch.setitem(sys.modules, "lkai_entry", stub)
    sys.modules.pop("e6_entry", None)
    return importlib.import_module("e6_entry")


CPUINFO = """processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
CPU revision\t: 1

processor\t: 1
CPU part\t: 0xd40
"""


def test_cpuinfo_reads_the_first_block_and_counts_processors(monkeypatch):
    e6 = _entry_module(monkeypatch)
    c = e6.cpuinfo(CPUINFO)
    assert (c["implementer"], c["variant"], c["part"], c["revision"]) == ("0x41", "0x1", "0xd40", "1")
    assert "i8mm" in c["features"].split()
    assert c["processors"] == 2


def test_cpuinfo_labels_what_it_cannot_read_unknown(monkeypatch):
    e6 = _entry_module(monkeypatch)
    c = e6.cpuinfo("processor\t: 0\n")
    assert c["part"] == c["implementer"] == c["features"] == "unknown"
    assert e6.cpuinfo(None)["processors"] == "unknown"


def test_auxv_extracts_hwcaps_and_rejects_a_truncated_vector(monkeypatch):
    e6 = _entry_module(monkeypatch)
    raw = struct.pack("<QQQQQQ", 16, 0xEFFFFB, 26, 0x2F3FF, 0, 0)
    assert e6.auxv(raw) == {"hwcap": "0xeffffb", "hwcap2": "0x2f3ff"}
    assert e6.auxv(raw[:-1]) == {"hwcap": "unknown", "hwcap2": "unknown"}
    assert e6.auxv(None) == {"hwcap": "unknown", "hwcap2": "unknown"}


def test_bytecode_use_separates_usable_hash_timestamp_stale_and_missing(monkeypatch, tmp_path):
    e6 = _entry_module(monkeypatch)
    root = str(tmp_path) + "/"
    mods = {}

    def mod(name: str, head: bytes | None, src: bytes = b"V = 1\n"):
        py = tmp_path / f"{name}.py"
        py.write_bytes(src)
        cached = tmp_path / "__pycache__" / f"{name}.pyc"
        if head is not None:
            cached.parent.mkdir(exist_ok=True)
            cached.write_bytes(head)
        spec = types.SimpleNamespace(origin=str(py), cached=str(cached))
        mods[name] = types.SimpleNamespace(__spec__=spec)
        return py

    magic = importlib.util.MAGIC_NUMBER
    mod("h", magic + struct.pack("<I", 1) + b"\0" * 8)  # unchecked hash
    mod("c", magic + struct.pack("<I", 3) + b"\0" * 8)  # checked hash: loader re-hashes the source
    py = mod("t", None)
    st = os.stat(py)
    (tmp_path / "__pycache__" / "t.pyc").write_bytes(magic + struct.pack("<III", 0, int(st.st_mtime), st.st_size))
    mod("s", magic + struct.pack("<III", 0, 1, 6))  # stale timestamp
    mod("m", None)  # no bytecode
    mods["outside"] = types.SimpleNamespace(__spec__=types.SimpleNamespace(origin="/usr/lib/x.py", cached=None))
    got = e6.bytecode_use(root, mods)
    assert got == {
        "task_root_modules": 5,
        "pyc_present": 4,
        "usable_unchecked_hash": 1,
        "usable_timestamp": 1,
        "recompiled_from_source": 3,
    }


def test_handler_attaches_metadata_to_the_first_invocation_only(monkeypatch, capsys):
    e6 = _entry_module(monkeypatch)
    ctx = types.SimpleNamespace(log_stream_name="2026/10/21/[$LATEST]abc")
    first = e6.handler({}, ctx)
    assert first["lkai"]["e6"]["log_stream"] == ctx.log_stream_name
    assert "cpu" in first["lkai"]["e6"] and "bytecode" in first["lkai"]["e6"]
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("E6 ")]
    assert len(line) == 1 and json.loads(line[0][3:])["log_stream"] == ctx.log_stream_name
    second = e6.handler({}, ctx)
    assert "e6" not in second["lkai"]
    assert not [x for x in capsys.readouterr().out.splitlines() if x.startswith("E6 ")]


# ---------------------------------------------------------------------------- bench arm order


@pytest.fixture
def e6bench(monkeypatch):
    if not (importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore")):
        pytest.skip("e6bench imports boto3 at module level")
    return importlib.import_module("e6bench")


def test_order_alternates_which_arm_goes_first(e6bench):
    assert [e6bench.order_for(["ctl", "pyc"], r)[0] for r in range(4)] == ["ctl", "pyc", "ctl", "pyc"]
