"""Summarise E15's per-node wall-time tables (OFFLINE, NOT LAMBDA) into the per-op table Phase 1 publishes.

  python summarize_nodeprof.py <out dir from nodeprof.sh> [--json table.json] [--md table.md]

Input: nodeprof<r>.tsv (one per round) written by patch_nodeprof.py's lkai_nodeprof_dump, plus ctl/off/on<r>.json
from offline_drive.py. Each TSV row is `key<TAB>nodes<TAB>ns`, key = op|name|src0 type|src1 type|src0 buffer type|
src0 name|flags. Rows are grouped into the categories below; a share is that category's node wall time over the sum
of graph wall times (`#graph_ns`), so the shares of one round add up to 100% of graph time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# (category, predicate on the parsed key). The first match wins, so order matters.
CATEGORIES = [
    (
        "KleidiAI matmul (Q8_0 weights, i8mm)",
        lambda k: k["op"] == "MUL_MAT" and k["buft"] == "CPU_KLEIDIAI",
    ),
    (
        "Hadamard rotation (MUL_MAT -> ggml_compute_forward_fwht)",
        lambda k: k["op"] == "MUL_MAT" and k["hadamard"],
    ),
    (
        "attention KQ (F32 MUL_MAT, ggml_vec_dot_f32)",
        lambda k: k["op"] == "MUL_MAT" and k["name"] == "kq",
    ),
    (
        "attention KQV (F32 MUL_MAT, ggml_vec_dot_f32)",
        lambda k: k["op"] == "MUL_MAT" and k["name"] == "kqv",
    ),
    ("other MUL_MAT", lambda k: k["op"] == "MUL_MAT"),
    ("GATED_DELTA_NET", lambda k: k["op"] == "GATED_DELTA_NET"),
    ("CONCAT (E14)", lambda k: k["op"] == "CONCAT"),
    ("SSM_CONV", lambda k: k["op"] == "SSM_CONV"),
    ("RMS_NORM (+ fused MUL)", lambda k: k["op"] == "RMS_NORM"),
    ("SWIGLU", lambda k: k["op"] == "SWIGLU"),
    ("SILU", lambda k: k["op"] == "SILU"),
]


def parse_key(key: str) -> dict:
    op, name, t0, t1, buft, s0, flags = key.split("|")
    fused = 0
    if op.startswith("fused+"):
        f, op = op.split(":", 1)
        fused = int(f[len("fused+") :])
    return {
        "op": op,
        "fused": fused,
        "name": name,
        "src0_type": t0,
        "src1_type": t1,
        "buft": buft,
        "src0": s0,
        "hadamard": "hadamard" in flags,
        "src1_noncontig": "src1-noncontig" in flags,
    }


def read_tsv(path: Path) -> tuple[dict, list[tuple[str, int, int]]]:
    meta, rows = {}, []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if parts[0].startswith("#"):
            meta[parts[0][1:]] = int(parts[1])
        else:
            rows.append((parts[0], int(parts[1]), int(parts[2])))
    return meta, rows


def categorise(rows: list[tuple[str, int, int]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, n, ns in rows:
        k = parse_key(key)
        cat = next((c for c, pred in CATEGORIES if pred(k)), f"other: {k['op']}")
        e = out.setdefault(cat, {"ns": 0, "nodes": 0})
        e["ns"] += ns
        e["nodes"] += n
    return out


def summarise(out_dir: Path) -> dict:
    rounds = sorted(out_dir.glob("nodeprof*.tsv"))
    if not rounds:
        raise SystemExit(f"no nodeprof*.tsv in {out_dir}")
    per_round = []
    for p in rounds:
        meta, rows = read_tsv(p)
        if meta.get("overflow_ns", 0):
            raise SystemExit(f"{p}: {meta['overflow_ns']} ns fell outside the key table")
        cats = categorise(rows)
        node_ns = sum(c["ns"] for c in cats.values())
        per_round.append(
            {
                "file": p.name,
                "graphs": meta["graphs"],
                "graph_ns": meta["graph_ns"],
                "node_ns": node_ns,
                "categories": cats,
            }
        )
    names = sorted(
        {c for r in per_round for c in r["categories"]},
        key=lambda c: -statistics.mean(r["categories"].get(c, {"ns": 0})["ns"] for r in per_round),
    )
    table = []
    for c in names:
        shares = [100 * r["categories"].get(c, {"ns": 0})["ns"] / r["graph_ns"] for r in per_round]
        per_graph_ms = [
            r["categories"].get(c, {"ns": 0})["ns"] / r["graphs"] / 1e6 for r in per_round
        ]
        table.append(
            {
                "category": c,
                "share_pct": [round(s, 3) for s in shares],
                "share_pct_mean": round(statistics.mean(shares), 3),
                "ms_per_graph": [round(x, 3) for x in per_graph_ms],
                "nodes_per_graph": per_round[0]["categories"].get(c, {"nodes": 0})["nodes"]
                / per_round[0]["graphs"],
            }
        )
    runs = {}
    for f in sorted(out_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if "steady_state_s" in d:
            runs[f.stem] = {
                "steady_state_total_s": d["phases_s"].get("steady_state_total"),
                "p50_s": {
                    k: round(statistics.median(v), 4) for k, v in d["steady_state_s"].items()
                },
                "answers": d.get("answers"),
            }
    return {
        "schema": "lkai-e15-nodeprof/1",
        "evidence": "OFFLINE, NOT LAMBDA: per-node wall time on an arm64 host in P1's Lambda image, 6 CPUs",
        "host": (out_dir / "host.txt").read_text() if (out_dir / "host.txt").exists() else None,
        "rounds": [{k: v for k, v in r.items() if k != "categories"} for r in per_round],
        "table": table,
        "runs": runs,
    }


def to_markdown(s: dict) -> str:
    lines = [
        "| Category | Share of warm graph wall time (per round) | ms per forward (per round) | Nodes per forward |",
        "|---|---|---|---|",
    ]
    for row in s["table"]:
        sh = " / ".join(f"{x:.2f}%" for x in row["share_pct"])
        ms = " / ".join(f"{x:.2f}" for x in row["ms_per_graph"])
        lines.append(f"| {row['category']} | {sh} | {ms} | {row['nodes_per_graph']:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    s = summarise(a.out_dir)
    if a.json:
        a.json.write_text(json.dumps(s, indent=1) + "\n")
    md = to_markdown(s)
    if a.md:
        a.md.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
