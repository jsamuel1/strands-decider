"""E9 build-time packer: write the baseline GGUF with each KleidiAI-bound Q8_0 weight already in KleidiAI's layout.

  python pack_e9.py <in.gguf> <out.gguf> [--kr 8] [--record pack.json]

The pinned loader (llama.cpp f74e2fa6a) repacks every Q8_0 matmul weight when it loads (kleidiai.cpp, repack):

  1. per row: dequantize the Q8_0 blocks (value = fp16 d * int8 q, in float32), take max |value|,
     scale = max / 127, q' = clamp(lround(value * (1 / scale)), -127, 127);
  2. pack q' and the row scales with kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon for the selected kernel (nr, kr, sr).

This script does both at build time, in numpy, with the same float32 arithmetic, and stores the result where the
tensor's Q8_0 bytes were: the KleidiAI weight header (magic KLAI, version 1, one slot at offset 64), the packed slot,
zero padding, and a 64-byte trailer that patch_e9.py's loader recognises:

  char magic[8] = "E9KAIPK1"; uint32 nr, kr, sr, 0; uint64 n, k, payload_bytes; 16 zero bytes

Tensor names, shapes, types, order and byte sizes are unchanged, so the file is the baseline's size plus one
metadata key, strands_decider.e9.kleidiai_prepacked (a string naming the layout). token_embd.weight is left as
Q8_0: the graph reads it with GET_ROWS, which needs native blocks, and its tied output copy is repacked at load.

--kr 8 is the i8mm kernel's layout (qai8dxp4x8_qsi8cxp4x8, nr=4); --kr 4 is the dot-product kernel's
(qai8dxp4x4_qsi8cxp4x4, nr=4). A host that selects the other kernel converts the layout at load (patch_e9.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys

import numpy as np

QK8_0 = 32
Q8_0_BLOCK = 2 + QK8_0  # fp16 d + 32 int8
KLAI_MAGIC = 0x4B4C4149
KLAI_VERSION = 1
HEADER_BYTES = 64  # sizeof(kleidiai_weight_header) = 40, aligned up to GGML_KLEIDIAI_PACK_ALIGN (64)
TRAILER_MAGIC = b"E9KAIPK1"
TRAILER_BYTES = 64
METADATA_KEY = "strands_decider.e9.kleidiai_prepacked"
SKIP = ("token_embd.weight",)


def requantize(raw: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """kleidiai.cpp repack's Q8_0 -> per-row int8 step. raw: the tensor's Q8_0 bytes, n rows of k values."""
    if k % QK8_0:
        raise ValueError(f"k={k} is not a multiple of {QK8_0}")
    blocks = np.ascontiguousarray(raw, dtype=np.uint8).reshape(n, k // QK8_0, Q8_0_BLOCK)
    d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)  # (n, nb, 1), exact
    qs = blocks[:, :, 2:].view(np.int8).astype(np.float32)  # (n, nb, 32)
    value = (d * qs).reshape(n, k)  # float32 products, as `d * static_cast<float>(blk.qs[l])`
    max_abs = np.abs(value).max(axis=1)
    scale = (max_abs / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(scale > 0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    x = (value * inv[:, None]).astype(np.float32)  # float32 product, then std::lround (half away from zero)
    ax = np.abs(x).astype(np.float64)
    r = np.sign(x).astype(np.float64) * np.floor(ax + 0.5)
    q = np.clip(r, -127, 127).astype(np.int8)
    q[scale == 0] = 0
    return q, scale


def packed_size(n: int, k: int, nr: int) -> int:
    """kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon."""
    k_int = -(-k // 32) * 32
    return -(-n // nr) * nr * (k_int + 12)


def pack(q: np.ndarray, scale: np.ndarray, nr: int, kr: int) -> bytes:
    """kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon with lhs_zero_point = 1, no bias, sr = 1."""
    n, k = q.shape
    k_int = -(-k // 32) * 32
    if k_int % kr:
        raise ValueError(f"k_internal {k_int} is not a multiple of kr={kr}")
    groups = -(-n // nr)
    rows = np.minimum(np.arange(groups * nr), n - 1)  # the kernel clamps out-of-range rows to n - 1
    qp = np.zeros((groups * nr, k_int), dtype=np.int8)
    qp[:, :k] = q[rows]
    # [group][k block][row in group][kr]
    blk = qp.reshape(groups, nr, k_int // kr, kr).transpose(0, 2, 1, 3).reshape(groups, nr * k_int)
    sums = qp.astype(np.int32).sum(axis=1).reshape(groups, nr).astype("<i4")
    sc = scale[rows].astype("<f4").reshape(groups, nr)
    bias = np.zeros((groups, nr), dtype="<f4")
    out = np.concatenate(
        [blk.view(np.uint8), sums.view(np.uint8), sc.view(np.uint8), bias.view(np.uint8)], axis=1
    )
    b = out.tobytes()
    assert len(b) == packed_size(n, k, nr)
    return b


def header(slot_bytes: int) -> bytes:
    h = struct.pack("<IHH2Q2Q", KLAI_MAGIC, KLAI_VERSION, 1, HEADER_BYTES, 0, slot_bytes, 0)
    return h + bytes(HEADER_BYTES - len(h))


def trailer(nr: int, kr: int, n: int, k: int, payload_bytes: int) -> bytes:
    t = TRAILER_MAGIC + struct.pack("<4I3Q", nr, kr, 1, 0, n, k, payload_bytes)
    return t + bytes(TRAILER_BYTES - len(t))


def prepacked_tensor(raw: np.ndarray, n: int, k: int, nr: int, kr: int) -> tuple[bytes, bytes]:
    """The E9 bytes for one Q8_0 tensor, and its payload (what the loader leaves in tensor->data)."""
    q, scale = requantize(raw, n, k)
    slot = pack(q, scale, nr, kr)
    payload = header(len(slot)) + slot
    size = raw.size
    if len(payload) + TRAILER_BYTES > size:
        raise ValueError(f"packed {len(payload)} + trailer does not fit in {size} Q8_0 bytes")
    return payload + bytes(size - len(payload) - TRAILER_BYTES) + trailer(nr, kr, n, k, len(payload)), payload


def unpack(payload: bytes, n: int, k: int, nr: int, kr: int) -> tuple[np.ndarray, np.ndarray]:
    """patch_e9.py's convert path, in numpy: the int8 rows and row scales back from a payload."""
    k_int = -(-k // 32) * 32
    groups = -(-n // nr)
    stride = nr * (k_int + 12)
    slot = np.frombuffer(payload, dtype=np.uint8, offset=HEADER_BYTES, count=groups * stride).reshape(groups, stride)
    blk = slot[:, : nr * k_int].view(np.int8).reshape(groups, k_int // kr, nr, kr).transpose(0, 2, 1, 3)
    q = blk.reshape(groups * nr, k_int)[:n, :k].copy()
    sc = slot[:, nr * k_int + 4 * nr : nr * k_int + 8 * nr].copy().view("<f4").reshape(-1)[:n]
    return q, sc.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--kr", type=int, choices=(4, 8), default=8)
    ap.add_argument("--nr", type=int, default=4)
    ap.add_argument("--record")
    a = ap.parse_args()
    import gguf

    r = gguf.GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(a.dst, arch=arch, endianess=r.endianess)
    align = r.fields.get("general.alignment")
    if align is not None:
        w.data_alignment = align.contents()
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        sub = f.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), t, sub_type=sub)
    layout = f"qsi8cxp nr={a.nr} kr={a.kr} sr=1 (kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon)"
    w.add_key_value(METADATA_KEY, layout, gguf.GGUFValueType.STRING)
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    rec = {"layout": layout, "metadata_key": METADATA_KEY, "tensors": {}}
    whole = hashlib.sha256()
    for t in r.tensors:
        data = t.data
        if t.tensor_type.name == "Q8_0" and t.name not in SKIP:
            k, n = int(t.shape[0]), int(np.prod(t.shape[1:]))
            new, payload = prepacked_tensor(np.asarray(data).reshape(-1), n, k, a.nr, a.kr)
            data = np.frombuffer(new, dtype=np.uint8).reshape(data.shape)
            rec["tensors"][t.name] = {
                "n": n,
                "k": k,
                "payload_bytes": len(payload),
                "slot_sha256": hashlib.sha256(payload[HEADER_BYTES:]).hexdigest(),
            }
        whole.update(t.name.encode() + b"\0" + np.asarray(data).tobytes())
        w.write_tensor_data(data, tensor_endianess=r.endianess)
    w.close()
    rec["packed_count"] = len(rec["tensors"])
    rec["tensor_digest"] = whole.hexdigest()
    if a.record:
        json.dump(rec, open(a.record, "w"), indent=1, sort_keys=True)
    print(f"wrote {a.dst}: {rec['packed_count']} tensors prepacked ({layout})", file=sys.stderr)


if __name__ == "__main__":
    main()
