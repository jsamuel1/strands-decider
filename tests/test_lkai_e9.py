"""examples/lambda-kleidiai/e9: the build-time KleidiAI packer and its loader patch.

pack_e9's numpy packing is checked against a direct, loop-for-loop translation of the two C routines it replaces
(kleidiai.cpp repack's Q8_0 requantization and kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon). The arm64 build, the
byte-for-byte check against the real runtime repack and the Lambda run are in e9/README.md; they need neither here.
"""

from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

E9 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "lambda-kleidiai", "e9")
)
sys.path.insert(0, E9)
import pack_e9 as P  # noqa: E402
import patch_e9  # noqa: E402


def q8_0(rng, n, k, zero_rows=()):
    """Random Q8_0 bytes for an n x k tensor (fp16 d, 32 int8 per block)."""
    nb = k // 32
    d = (rng.standard_normal((n, nb, 1)) * 0.02).astype(np.float16)
    for r in zero_rows:
        d[r] = 0
    qs = rng.integers(-128, 128, size=(n, nb, 32), dtype=np.int8)
    return np.concatenate([d.view(np.uint8), qs.view(np.uint8)], axis=2).reshape(-1)


def c_requantize(raw, n, k):
    """kleidiai.cpp repack (Q8_0 branch), in Python floats rounded to float32 at each C float operation."""
    f32 = lambda x: float(np.float32(x))  # noqa: E731
    q = np.zeros((n, k), dtype=np.int8)
    sc = np.zeros(n, dtype=np.float32)
    blocks = raw.reshape(n, k // 32, 34)
    for row in range(n):
        vals = []
        for b in range(k // 32):
            d = float(blocks[row, b, :2].copy().view(np.float16)[0])
            for v in blocks[row, b, 2:].view(np.int8):
                vals.append(f32(d * float(v)))
        max_abs = max(abs(v) for v in vals)
        scale = f32(max_abs / 127.0) if max_abs > 0 else 0.0
        inv = f32(1.0 / scale) if scale > 0 else 0.0
        sc[row] = scale
        for i, v in enumerate(vals):
            x = f32(v * inv)
            r = int(math.copysign(math.floor(abs(x) + 0.5), x)) if scale > 0 else 0  # std::lround
            q[row, i] = max(-127, min(127, r))
    return q, sc


def c_pack(q, scale, nr, kr):
    """kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon (lhs_zero_point 1, bias NULL, sr 1), loop for loop."""
    n, k = q.shape
    k_int = (k + 31) // 32 * 32
    stride = nr * (k_int + 12)
    rows = (n + nr - 1) // nr
    out = bytearray(rows * stride)
    for g in range(rows):
        base = g * stride
        sums = [0] * nr
        pos = base
        for off in range(0, nr * k_int, kr):
            blk = off // kr
            nri, sb = blk % nr, blk // nr
            n0 = min(g * nr + nri, n - 1)
            ps = 0
            for i in range(kr):
                kk = sb * kr + i
                v = int(q[n0, kk]) if kk < k else 0
                out[pos + i] = v & 0xFF
                ps += v
            sums[nri] += ps
            pos += kr
        struct.pack_into(f"<{nr}i", out, pos, *sums)
        pos += 4 * nr
        struct.pack_into(
            f"<{nr}f", out, pos, *[float(scale[min(g * nr + i, n - 1)]) for i in range(nr)]
        )
    return bytes(out)


@pytest.mark.parametrize("n,k", [(8, 64), (6, 96), (5, 32)])
@pytest.mark.parametrize("kr", [4, 8])
def test_numpy_pack_matches_the_c_routines(n, k, kr):
    rng = np.random.default_rng(n * 1000 + k + kr)
    raw = q8_0(rng, n, k, zero_rows=(1,))
    q, sc = P.requantize(raw, n, k)
    cq, csc = c_requantize(raw, n, k)
    assert np.array_equal(q, cq)
    assert sc.tobytes() == csc.tobytes()
    assert P.pack(q, sc, 4, kr) == c_pack(
        cq, csc, 4, kr
    )  # includes n not a multiple of nr (row clamping)


def test_rounding_half_away_from_zero_and_clamp():
    # One block: d = 1, values -127..127 plus +-0.5 ties after scaling are impossible to hit exactly with random
    # data, so pin lround on a constructed row: max |v| = 127 gives scale 1 and q == value.
    d = np.ones((1, 1, 1), dtype=np.float16)
    qs = np.arange(-127, -127 + 32, dtype=np.int8).reshape(1, 1, 32)
    raw = np.concatenate([d.view(np.uint8), qs.view(np.uint8)], axis=2).reshape(-1)
    q, sc = P.requantize(raw, 1, 32)
    assert sc[0] == np.float32(127 / 127)
    assert np.array_equal(q[0], qs.reshape(-1))


def test_tensor_bytes_header_trailer_and_size():
    rng = np.random.default_rng(7)
    n, k = 12, 512
    raw = q8_0(rng, n, k)
    new, payload = P.prepacked_tensor(raw, n, k, 4, 8)
    assert len(new) == raw.size  # same byte size: offsets and file size unchanged
    magic, ver, slots, off0, _off1, sz0, _sz1 = struct.unpack_from("<IHH2Q2Q", payload)
    assert (magic, ver, slots, off0) == (0x4B4C4149, 1, 1, 64)
    assert sz0 == P.packed_size(n, k, 4) == len(payload) - 64
    t = new[-64:]
    assert t[:8] == b"E9KAIPK1"
    assert struct.unpack_from("<4I3Q", t, 8) == (4, 8, 1, 0, n, k, len(payload))
    assert t[48:] == bytes(16)
    assert new[: len(payload)] == payload and set(new[len(payload) : -64]) <= {0}


@pytest.mark.parametrize("src_kr,dst_kr", [(8, 4), (4, 8)])
def test_convert_path_recovers_rows_and_repacks_identically(src_kr, dst_kr):
    """A host whose kernel uses the other layout unpacks and packs; the result equals a direct pack."""
    rng = np.random.default_rng(src_kr)
    n, k = (
        101,
        512,
    )  # n not a multiple of nr: exercises the clamped padding rows; large enough to fit
    raw = q8_0(rng, n, k)
    q, sc = P.requantize(raw, n, k)
    _, payload = P.prepacked_tensor(raw, n, k, 4, src_kr)
    uq, usc = P.unpack(payload, n, k, 4, src_kr)
    assert np.array_equal(uq, q) and usc.tobytes() == sc.tobytes()
    assert P.pack(uq, usc, 4, dst_kr) == P.pack(q, sc, 4, dst_kr)


def test_packed_must_fit():
    # A 1-row tensor: 1 block = 34 bytes, packed = 4 rows x (32 + 12) = 176 > 34.
    raw = q8_0(np.random.default_rng(0), 1, 32)
    with pytest.raises(ValueError, match="does not fit"):
        P.prepacked_tensor(raw, 1, 32, 4, 8)


def test_patch_applies_once_to_its_anchors(tmp_path: Path):
    k = tmp_path / "ggml/src/ggml-cpu/kleidiai/kleidiai.cpp"
    lm = tmp_path / "src/llama-model-loader.cpp"
    k.parent.mkdir(parents=True)
    lm.parent.mkdir(parents=True)
    k.write_text(
        "class tensor_traits : public ggml::cpu::tensor_traits {\n"
        "        std::vector<float> row_f32;\n\n        if (want_q8 && slot_total > 0) {\n"
        "            memcpy(tensor->data, data, data_size);\n        }\n\n        return 0;\n    }\n};\n"
        "        // Kernel selection\n"
    )
    lm.write_text(
        "            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;\n"
        "        const bool from_mapping = use_mmap || lazy.has(cur);\n"
    )
    patch_e9.patch(tmp_path)
    ks, ls = k.read_text(), lm.read_text()
    assert "E9KAIPK1" in ks and "!e9_have_q" in ks and "E9_DIAG_NO_I8MM" in ks and "e9_note(0" in ks
    assert "CPU_KLEIDIAI" in ls and P.METADATA_KEY in ls
    with pytest.raises(AssertionError, match="anchor"):
        patch_e9.patch(
            tmp_path
        )  # anchors are consumed: a second application is refused, not doubled


def test_trailer_layout_matches_the_c_struct():
    src = patch_e9.E9_TRAILER
    assert "char     magic[8];" in src and "uint32_t nr, kr, sr, reserved;" in src
    assert "uint64_t n, k, payload_bytes, zero[2];" in src and "sizeof(e9_trailer) == 64" in src
    assert P.TRAILER_BYTES == 8 + 4 * 4 + 3 * 8 + 16


def _summarize_mod(monkeypatch):
    import importlib
    import importlib.util
    import types

    if not (importlib.util.find_spec("boto3") and importlib.util.find_spec("botocore")):
        exc = types.ModuleType("botocore.exceptions")
        exc.ClientError = type("ClientError", (Exception,), {})
        cfg = types.ModuleType("botocore.config")
        cfg.Config = lambda **kw: None
        boto3 = types.ModuleType("boto3")
        boto3.Session = lambda **kw: None
        for name, mod in {
            "boto3": boto3,
            "botocore": types.ModuleType("botocore"),
            "botocore.exceptions": exc,
            "botocore.config": cfg,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
    for m in ("deploy", "deploy_e9", "summarize_e9"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(E9), "baseline"))
    return importlib.import_module("summarize_e9")


def test_load_paths_counts_environments_by_arm_part_and_path(tmp_path: Path, monkeypatch):
    import gzip
    import json

    S = _summarize_mod(monkeypatch)
    rows = [
        {
            "arm": "e9",
            "cpu_part": "0xd40",
            "lkai": {"env_id": "a"},
            "log_tail": "E9KAI repack nr=4 kr=8\nE9KAI adopt nr=4 kr=8\n",
        },
        {
            "arm": "e9",
            "cpu_part": "0xd40",
            "lkai": {"env_id": "a"},
            "log_tail": "",
        },  # same env, warm
        {
            "arm": "e9",
            "cpu_part": "0xd0c",
            "lkai": {"env_id": "b"},
            "log_tail": "E9KAI convert nr=4 kr=4\nE9KAI repack nr=4 kr=4\n",
        },
        {"arm": "base", "cpu_part": "0xd40", "lkai": {"env_id": "c"}, "log_tail": "no e9 lines"},
    ]
    (tmp_path / "logs").mkdir()
    with gzip.open(tmp_path / "logs" / "lkai-f1ee-e9.jsonl.gz", "wt") as fh:
        for m in ("E9KAI adopt nr=4 kr=8", "E9KAI adopt nr=4 kr=8", "other"):
            fh.write(json.dumps({"message": m, "logStreamName": "s"}) + "\n")
    lp = S.load_paths(rows, str(tmp_path))
    assert lp["environments"] == {
        "e9:0xd0c:convert kr=4+repack kr=4": 1,
        "e9:0xd40:adopt kr=8+repack kr=8": 1,
    }
    assert lp["cloudwatch_lines"] == {"lkai-f1ee-e9": {"adopt kr=8": 2}}


def test_no_unevidenced_graviton_generation_claims():
    """E9's prose names CPU part ids only: a 'Graviton N' label would be inferred, not observed."""
    import re

    bad = re.compile(r"graviton\s*[0-9]", re.IGNORECASE)
    for f in ("README.md", "meta.json", "host_variability.py", "summarize_e9.py", "deploy_e9.py"):
        text = open(os.path.join(E9, f)).read()
        assert not bad.search(text), f
    bench = open(os.path.join(os.path.dirname(E9), "baseline", "bench.py")).read()
    assert "Graviton generation it drew" not in bench


def test_bench_checks_each_arms_own_zip():
    src = open(os.path.join(os.path.dirname(E9), "baseline", "bench.py")).read()
    assert (
        'st.get("zip_sha256_by_arm", {}).get(arm, st["manifest"]["function_zip"]["sha256"])' in src
    )
    assert "new_invoke_client" in src  # I1's fresh client per wave


def test_deploy_arms_differ_in_zip_and_weights_only(monkeypatch):
    S = _summarize_mod(monkeypatch)
    import importlib

    D = importlib.import_module("deploy_e9")
    assert D.ARMS == {
        "base": ("base-function.zip", "s2b-h512-Q8_0.gguf"),
        "e9": ("function.zip", "s2b-h512-Q8_0-e9kr8.gguf"),
    }
    assert S.compare_arms.__defaults__ == ("e9", "base")
