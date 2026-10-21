"""Aggregate E10's offline evidence into offline/sizes.json (strict JSON): llama.cpp's buffer sizes per n_ctx, the
context each request set needs, and what each n_ctx refuses.

  python offline_sizes.py --probes offline/probe-*.json --jevbench offline/jevbench-w4096-tokens.json \
      --p1-requests ../baseline/requests.json --out offline/sizes.json

Buffer sizes come from llama.cpp's own load log (offline_probe.py, x86): they depend on n_ctx, the model's shape and
the context parameters (f32 K/V, N_SEQ=17, kv_unified, ubatch 512), not on the CPU, so they hold for Lambda too.
Timings and RSS in the probe files are x86-only and are NOT Lambda numbers.
"""

from __future__ import annotations

import argparse
import json
import math

MiB = 1024 * 1024
# Qwen3.5-2B (config.json): 24 layers, every 4th full attention -> 6 attention + 18 Gated DeltaNet layers;
# 2 KV heads x 256 head dim; GDN conv (d_conv-1)*(d_inner + 2*n_group*d_state), state d_state*d_inner.
ATTN_LAYERS, KV_HEADS, HEAD_DIM, F32 = 6, 2, 256, 4
KV_BYTES_PER_CELL = ATTN_LAYERS * KV_HEADS * HEAD_DIM * 2 * F32  # K and V
RECR_LAYERS, N_SEQ = 18, 17
RS_BYTES = RECR_LAYERS * N_SEQ * ((4 - 1) * (2048 + 2 * 16 * 128) + 128 * 2048) * F32


def _clean(v):
    """NaN/inf -> None, so the file is strict JSON."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def needs(tokens: list[int], sizes: list[int]) -> dict:
    return {
        "n": len(tokens),
        "max": max(tokens),
        "p50": sorted(tokens)[len(tokens) // 2],
        "refused_at": {str(s): sum(t > s for t in tokens) for s in sizes},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", required=True)
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--p1-need", required=True, help="offline/p1-context-need.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    probes = sorted((json.load(open(p)) for p in a.probes), key=lambda d: d["n_ctx"])
    sizes = [d["n_ctx"] for d in probes]
    out_sizes = {}
    for d in probes:
        b, n = d["buffers"], d["n_ctx"]
        kv_calc = KV_BYTES_PER_CELL * n
        assert abs(b["kv_mib"][0] - kv_calc / MiB) < 0.01, (n, b["kv_mib"], kv_calc)
        assert abs(b["rs_mib"][0] - RS_BYTES / MiB) < 0.01, (b["rs_mib"], RS_BYTES)
        out_sizes[str(n)] = {
            "attention_kv_bytes": kv_calc,
            "attention_kv_mib": b["kv_mib"][0],
            "recurrent_state_bytes": RS_BYTES,
            "recurrent_state_mib": b["rs_mib"][0],
            "compute_buffer_mib": b["compute_mib"][0],
            "output_buffer_mib": b["output_mib"][0],
            "cleared_per_request_bytes": kv_calc + RS_BYTES,
            "kv_log": b["kv_line"][0],
            "rs_log": b["rs_line"][0],
        }
    jev = json.load(open(a.jevbench))
    p1 = json.load(open(a.p1_need))
    res = {
        "schema": "lkai-e10-sizes/1",
        "source": "llama.cpp f74e2fa6a load log (offline_probe.py), checked against the closed-form sizes: "
        f"attention K+V = {KV_BYTES_PER_CELL} B per cell (6 layers x 2 heads x 256 x f32 x 2); recurrent "
        f"state = {RS_BYTES} B (18 GDN layers x 17 seqs, independent of n_ctx)",
        "sizes": out_sizes,
        "capacity": {
            "rule": "a request needs n_ctx >= state + its longest question (Shim.forward_shared cuts fork groups "
            "to fit). The 4,096-token window (head_config max_length) already bounds state + longest question, "
            "so n_ctx >= 4096 refuses nothing the window admits; below 4096 a request over n_ctx gets a 422 "
            "(e10_entry.ContextTooSmall) before any decode",
            "p1_requests": p1
            | {"refused_at": {str(s): [r for r, v in p1["need"].items() if v > s] for s in sizes}},
            "jevbench_public_w4096": needs([t["input_tokens"] for t in jev["tasks"]], sizes)
            | {"source": jev["source"], "note": jev["note"]},
            "window_max_length": 4096,
        },
        "x86_probe_note": "context_init, memory-clear and RSS figures in probe-*.json are from an x86 host (Intel "
        "Xeon 6975P-C, 16 threads) and are not Lambda numbers",
    }
    json.dump(_clean(res), open(a.out, "w"), indent=1, allow_nan=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
