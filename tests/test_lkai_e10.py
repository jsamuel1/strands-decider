"""examples/lambda-kleidiai/e10: the n_ctx wrapper, the zip builder, the arm configuration and the bench's order.

No AWS and no llama.cpp. e10_entry is imported in a subprocess with a stand-in `engine` and `lkai_entry`, so the
import hook is exercised exactly as Lambda runs it. The tests pin E10's one-variable contract: every arm deploys P1's
zip plus the wrapper, byte for byte; arms differ only in E10_N_CTX; the wrapper changes only Shim's n_ctx argument;
and an over-context request is refused with a ValueError before any decode, never truncated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import zipfile

import pytest

E10 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e10")
)
BASE = os.path.abspath(os.path.join(E10, "..", "baseline"))
sys.path.insert(0, E10)
sys.path.insert(0, BASE)

import importlib  # noqa: E402
import importlib.util  # noqa: E402
import types  # noqa: E402

import make_zip  # noqa: E402


def _sdk_modules() -> dict:
    """Stand-ins for boto3/botocore when the test env lacks them (as tests/test_lkai_baseline.py does)."""
    if importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore"):
        return {}
    boto3 = types.ModuleType("boto3")
    boto3.Session = lambda **kw: None
    boto3.__version__ = "stand-in"
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = type("ClientError", (Exception,), {})
    cfg = types.ModuleType("botocore.config")
    cfg.Config = lambda **kw: None
    return {
        "boto3": boto3,
        "botocore": types.ModuleType("botocore"),
        "botocore.exceptions": exc,
        "botocore.config": cfg,
    }


@pytest.fixture
def aws_mods(monkeypatch):
    for name, mod in _sdk_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in ("deploy", "bench", "e10deploy", "e10bench"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    return importlib.import_module("e10deploy"), importlib.import_module("e10bench")


STAMP = (2026, 1, 1, 0, 0, 0)

FAKE_ENGINE = """
CALLS = []
class Shim:
    def __init__(self, lib_path, model_path, threads, n_ctx, n_seq):
        CALLS.append((lib_path, model_path, threads, n_ctx, n_seq))
        self.n_ctx, self.n_seq = n_ctx, n_seq
    def forward_shared(self, state_ids, q_ids, rows=None):
        return ["fwd", len(state_ids), [len(q) for q in q_ids], rows]
def load_engine(lib, gguf, assets, threads):
    return Shim(lib, gguf, threads, n_ctx=16384, n_seq=17)
"""

FAKE_LKAI = """
import engine
class Timed(engine.Shim):
    pass
engine.Shim = Timed
def handler(event, context):
    s = engine.Shim("lib", "m", 6, 16384, 17)
    return {"lkai": {"inv": {}}, "n_ctx": s.n_ctx, "calls": [list(c) for c in engine.CALLS],
            "fwd": s.forward_shared(*event["fwd"]) if "fwd" in event else None}
"""


def _run(tmp_path, env_n_ctx: str | None, event: dict) -> subprocess.CompletedProcess:
    (tmp_path / "engine.py").write_text(FAKE_ENGINE)
    (tmp_path / "lkai_entry.py").write_text(FAKE_LKAI)
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path[:0] = [{str(tmp_path)!r}, {E10!r}]
        import e10_entry
        class Ctx: log_stream_name = "2026/10/21/[$LATEST]abc"
        try:
            out = e10_entry.handler({event!r}, Ctx())
        except ValueError as e:
            out = {{"value_error": type(e).__name__, "msg": str(e)}}
        print(json.dumps(out))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "E10_N_CTX"}
    if env_n_ctx is not None:
        env["E10_N_CTX"] = env_n_ctx
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path
    )


def test_wrapper_replaces_only_n_ctx(tmp_path):
    p = _run(tmp_path, "2048", {})
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["n_ctx"] == 2048
    assert out["calls"] == [["lib", "m", 6, 2048, 17]]  # threads, path and n_seq untouched
    rec = out["lkai"]["e10"]
    assert rec["n_ctx"] == {"env": 2048, "requested": [16384], "applied": [2048]}
    assert rec["log_stream"] == "2026/10/21/[$LATEST]abc"
    assert "E10 {" in p.stdout


def _module_defs() -> dict:
    """e10_entry's definitions without its import-time side effects (reading E10_N_CTX, importing lkai_entry)."""
    src = os.path.join(E10, "e10_entry.py")
    mod: dict = {"__name__": "e10_probe"}
    exec(compile(open(src).read().split("META = host_metadata()")[0], src, "exec"), mod)
    return mod


def test_wrapper_wraps_keyword_n_ctx():
    ns: dict = {}
    exec(FAKE_ENGINE, ns)
    _module_defs()["wrap_shim"](ns["Shim"], 1024)
    s = ns["Shim"]("l", "m", 4, n_ctx=16384, n_seq=17)
    assert (s.n_ctx, s.n_seq) == (1024, 17)


def test_control_arm_runs_p1_arguments(tmp_path):
    p = _run(tmp_path, "16384", {})
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["calls"] == [["lib", "m", 6, 16384, 17]]


@pytest.mark.parametrize("value", [None, "", "0", "-1", "abc"])
def test_missing_or_invalid_n_ctx_fails_init(tmp_path, value):
    p = _run(tmp_path, value, {})
    assert p.returncode != 0
    assert "E10_N_CTX" in p.stderr


def test_over_context_request_is_refused_before_decode(tmp_path):
    # state 400 + longest question 200 > 512: refused with ValueError, forward_shared never reached
    p = _run(tmp_path, "512", {"fwd": [[1] * 400, [[2] * 100, [3] * 200]]})
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["value_error"] == "ContextTooSmall"
    assert "600 tokens" in out["msg"] and "512" in out["msg"] and "at least 88" in out["msg"]


def test_fitting_request_passes_through_unchanged(tmp_path):
    # state 300 + longest 200 = 500 <= 512, though the sum (600) is not: fork groups split, nothing is refused
    p = _run(tmp_path, "512", {"fwd": [[1] * 300, [[2] * 100, [3] * 200]]})
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["fwd"] == ["fwd", 300, [100, 200], None]


def test_check_fits_boundary():
    mod = _module_defs()
    mod["check_fits"](412, [100], 512)  # exactly full
    with pytest.raises(ValueError):
        mod["check_fits"](413, [100], 512)


def test_proc_memory_and_cpuinfo_parsers():
    mod = _module_defs()
    st = "Name:\tpython\nVmHWM:\t 5242880 kB\nVmRSS:\t 5000000 kB\n"
    assert mod["proc_memory"](st) == {"vm_hwm_kib": 5242880, "vm_rss_kib": 5000000}
    assert mod["proc_memory"](None) == {"vm_hwm_kib": "unknown", "vm_rss_kib": "unknown"}
    c = mod["cpuinfo"](
        "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\nprocessor\t: 1\n"
    )
    assert c["part"] == "0xd40" and c["processors"] == 2 and c["revision"] == "unknown"


def _p1_zip(tmp_path) -> tuple[str, str]:
    path = tmp_path / "p1.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in {
            "lkai_entry.py": b"x=1\n",
            "lib/libfake.so": b"\x7fELF",
            "assets/t.json": b"{}",
        }.items():
            z.writestr(zipfile.ZipInfo(name, date_time=STAMP), data)
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {"repo_commit": "0" * 40, "model": {"gguf_bytes": 1}, "images": {}, "toolchain": []}
        )
    )
    return str(path), str(man)


def test_zip_is_p1_plus_wrapper(tmp_path):
    p1z, man = _p1_zip(tmp_path)
    out = tmp_path / "build"
    assert make_zip.main(["--p1-zip", p1z, "--p1-manifest", man, "--out", str(out)]) == 0
    with zipfile.ZipFile(p1z) as a, zipfile.ZipFile(out / "function.zip") as b:
        ia = {i.filename: i for i in a.infolist()}
        ib = {i.filename: i for i in b.infolist()}
        assert set(ib) - set(ia) == {"e10_entry.py"}
        for n, i in ia.items():
            assert b.read(n) == a.read(n)
            assert (ib[n].date_time, ib[n].external_attr, ib[n].compress_type) == (
                i.date_time,
                i.external_attr,
                i.compress_type,
            )
        assert b.read("e10_entry.py") == open(os.path.join(E10, "e10_entry.py"), "rb").read()
    m = json.load(open(out / "manifest.json"))
    assert m["base"]["p1_zip_sha256"] == hashlib.sha256(open(p1z, "rb").read()).hexdigest()
    assert (
        m["zips"]["e10"]["sha256"]
        == hashlib.sha256((out / "function.zip").read_bytes()).hexdigest()
    )


def test_zip_refuses_existing_wrapper(tmp_path):
    p1z, man = _p1_zip(tmp_path)
    with zipfile.ZipFile(p1z, "a") as z:
        z.writestr("e10_entry.py", b"")
    with pytest.raises(SystemExit):
        make_zip.main(["--p1-zip", p1z, "--p1-manifest", man, "--out", str(tmp_path / "b")])


def test_arms_and_control(aws_mods):
    e10deploy, _ = aws_mods
    assert e10deploy.SIZES == (512, 1024, 2048, 4096, 8192, 16384)
    assert e10deploy.CONTROL == "c16384" and e10deploy.n_ctx_of("c4096") == 4096
    with pytest.raises(ValueError):
        e10deploy.n_ctx_of("c3000")


def test_arm_environment_differs_only_in_n_ctx(aws_mods):
    e10deploy, _ = aws_mods
    p1 = {
        "N_SEQ": "17",
        "DEC_KV": "f32",
        "DEC_UBATCH": "512",
        "MODEL_KEY": "k",
        "MODEL_BUCKET": "b",
    }
    envs = {arm: e10deploy.arm_environment(p1, arm) for arm in e10deploy.ARMS}
    for arm, env in envs.items():
        assert {k: env[k] for k in p1} == p1
        assert env["E10_N_CTX"] == arm[1:] and env["E10_ARM"] == arm and env["LKAI_PHASES"] == "1"
        assert set(env) == set(p1) | {"COLD_TAG", "LKAI_PHASES", "E10_ARM", "E10_N_CTX"}


def test_order_is_a_latin_square(aws_mods):
    e10deploy, e10bench = aws_mods
    arms = list(e10deploy.ARMS)
    orders = [e10bench.order_for(arms, r) for r in range(len(arms))]
    for o in orders:
        assert sorted(o) == sorted(arms)
    for pos in range(len(arms)):
        assert sorted(o[pos] for o in orders) == sorted(arms)  # every arm in every position once


# ---------------------------------------------------------------- summarize_e10


def _row(
    arm,
    rnd,
    cls,
    cpu,
    ctx_init=0.3,
    lat=0.24,
    mem=5000,
    answers=None,
    rid="ticket-route-1q",
    err=None,
):
    return {
        "arm": arm,
        "round": rnd,
        "wave": cls,
        "expected_class": cls,
        "observed_class": cls,
        "class_mismatch": False,
        "error": err,
        "cpu_part": cpu,
        "request_id": rid,
        "latency_ms": lat * 1000,
        "answers": answers or {"q": {"type": "noul", "noul": 0.5}},
        "report": {"Init Duration": 800.0, "Duration": 35000.0, "Max Memory Used": mem},
        "lkai": {
            "init": {"context_init": ctx_init},
            "e10": {"n_ctx": {"env": 1, "applied": [1]}, "vm_hwm_kib": 1024},
        },
    }


@pytest.fixture
def s10(aws_mods):
    sys.modules.pop("summarize", None)
    sys.modules.pop("summarize_e10", None)
    return importlib.import_module("summarize_e10")


def test_host_cells_compare_within_round_and_part_only(s10):
    rows = []
    for rnd in range(2):
        rows += [_row("c16384", rnd, "cold", "0xd40", ctx_init=v) for v in (0.30, 0.32, 0.31)]
        rows += [_row("c4096", rnd, "cold", "0xd40", ctx_init=v) for v in (0.20, 0.21)]
        rows += [
            _row("c4096", rnd, "cold", "0xd0c", ctx_init=0.5)
        ]  # no control on 0xd0c: missing, not compared
    rows += [_row("c4096", 0, "cold", None, ctx_init=9.9)]  # unknown CPU: counted, never a stratum
    hv = s10.host_variability_analysis(rows, ["c4096", "c16384"])
    m = hv["observed"]["primary_metrics"]["cold_context_init"]["arms"]["c4096"]
    assert [c["cpu_part"] for c in m["shared_cells"]] == ["0xd40", "0xd40"]
    assert all(c["outcome"] == "negative-beyond-control-variation" for c in m["shared_cells"])
    assert {(c["round"], c["cpu_part"]) for c in m["missing_cells"]} == {(0, "0xd0c"), (1, "0xd0c")}
    assert m["by_cpu_part"]["0xd40"]["outcome"] == "negative-beyond-control-variation"
    assert hv["unknown"]["cpu_identity_rows"]["cold_context_init"]["c4096"] == 1
    assert "Graviton" not in json.dumps(hv["observed"])


def test_within_control_spread_is_null_even_with_signs(s10):
    rows = []
    for rnd, d in enumerate((-0.005, 0.005)):
        rows += [_row("c16384", rnd, "warm", "0xd40", lat=v) for v in (0.23, 0.25)]
        rows += [_row("c4096", rnd, "warm", "0xd40", lat=0.24 + d)]
    hv = s10.host_variability_analysis(rows, ["c4096", "c16384"])
    part = hv["observed"]["primary_metrics"]["warm_steady_state_inference"]["arms"]["c4096"][
        "by_cpu_part"
    ]["0xd40"]
    assert part["outcome"] == "within-control-variation" and part["delta_signs"] == [
        "negative",
        "positive",
    ]


def test_arm_agreement_counts_differences_and_uncovered_parts(s10):
    a, b = {"q": {"type": "noul", "noul": 0.5}}, {"q": {"type": "noul", "noul": 0.6}}
    rows = [
        _row("c16384", 0, "warm", "0xd40", answers=a),
        _row("c512", 0, "warm", "0xd40", answers=a),
        _row("c512", 0, "warm", "0xd40", answers=b),
        _row("c512", 0, "warm", "0xd4f", answers=a),
    ]
    g = s10.arm_agreement(rows, ["c512", "c16384"])
    assert g["c512"] == {"compared": 2, "bit_identical": 1, "no_control_answer_for_cpu_part": 1}
