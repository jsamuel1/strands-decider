#!/usr/bin/env python3
"""Render README tables (rung x path, with p1's per-round range) from summarize.py's ladder.json."""

from __future__ import annotations

import json
import sys

COLS = (
    ("load_s", "load s"),
    ("model_ready_s", "load+first request s"),
    ("first_inference_s", "first forward s"),
    ("warm_s", "warm forward s"),
    ("forward4096_s", "4,096-token forward s"),
    ("peak_rss_mb", "peak RSS MB"),
    ("imports_s", "imports s"),
)


def cell(m: dict) -> str:
    f = "{:+.0f}" if m["unit"] == "MB" else "{:+.3f}"
    g = "{:.0f}" if m["unit"] == "MB" else "{:.3f}"
    mark = {"better": "**better**", "worse": "**WORSE**"}.get(m["outcome"], m["outcome"])
    return (
        f"{g.format(m['control_median'])} → {g.format(m['arm_median'])} "
        f"({f.format(m['delta_median'])}; ±{g.format(m['threshold'])}) {mark}"
    )


def main() -> None:
    s = json.load(open(sys.argv[1]))
    print(
        "| rung | CPU identity | path | n | "
        + " | ".join(c for _, c in COLS)
        + " | answers vs P1 |"
    )
    print("|---|---|---|---|" + "---|" * (len(COLS) + 1))
    for rung, cells in s["rungs"].items():
        for key, v in cells.items():
            ident, mode = key.split("|")
            if mode == "correctness":
                continue
            c = cells[f"{ident}|correctness"]
            ans = (
                f"{c['vs_run_p1_requests']['identical']}/{c['vs_run_p1_requests']['compared']} req, "
                f"{c['vs_run_p1_forward4096']['identical']}/{c['vs_run_p1_forward4096']['compared']} fwd"
            )
            arch = "x86_64" if ident.startswith("Genuine") else "arm64"
            print(
                f"| {rung} | {ident} | {arch} {mode} | {v['n']} | "
                + " | ".join(cell(v["metrics"][k]) for k, _ in COLS)
                + f" | {ans} |"
            )


if __name__ == "__main__":
    main()
