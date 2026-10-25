#!/usr/bin/env python3
"""Run the general ladder's arms on a fixed host, interleaved, one fresh container per load. FIXED HOST, NOT LAMBDA.

  python3 run_ladder.py <arms.json> <out_dir> [--rounds 5] [--cpuset 10-15] [--threads 6] [--image IMG]
                        [--platform linux/arm64] [--only arm1,arm2] [--warm-passes 3] [--long-passes 2]

arms.json: {"image": ..., "requests": ..., "reference": ..., "arms": [{"name", "pkg", "gguf", "env": {},
"args": [], "modes": ["local", "memfd"]}]}. Paths are host paths; each load mounts the arm's unzipped package
read-only at /var/task (as Lambda does), the GGUF read-only, and this directory.

Interleaving: every (arm, mode) cell runs once per round. Round r runs the cells in the base order rotated by r
(a cyclic Latin square over rounds when rounds >= cells), so every cell takes a different position in each round
and drift within a round is spread over all arms instead of landing on one. Standard library only (Python 3.9+,
the DevSpace's system python).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def cells(arms: list[dict], only: set[str] | None = None) -> list[tuple[dict, str]]:
    out = []
    for arm in arms:
        if only and arm["name"] not in only:
            continue
        for mode in arm.get("modes", ["local"]):
            out.append((arm, mode))
    return out


def order(n: int, r: int) -> list[int]:
    """Round r's cell order: the base order rotated by r positions."""
    return [(i + r) % n for i in range(n)]


def docker_cmd(cfg: dict, arm: dict, mode: str, out_name: str, a) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        a.platform,
        "--cpuset-cpus",
        a.cpuset,
        "--network",
        "none",
        "--entrypoint",
        "python3.12",
        "-w",
        "/tmp",
        "-e",
        f"THREADS={a.threads}",
        "-e",
        "LD_LIBRARY_PATH=/var/task/lib",
        "-e",
        "PYTHONPATH=/var/task",
    ]
    for k, v in sorted((arm.get("env") or {}).items()):
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "-v",
        f"{arm['pkg']}:/var/task:ro",
        "-v",
        f"{os.path.realpath(arm['gguf'])}:/m/model.gguf:ro",
        "-v",
        f"{HERE}:/g:ro",
        "-v",
        f"{cfg['requests']}:/m/requests.json:ro",
        "-v",
        f"{cfg['reference']}:/m/reference.json:ro",
        "-v",
        f"{a.out}:/out",
        cfg.get("image") or a.image,
        "/g/probe.py",
        "/var/task",
        "/m/model.gguf",
        "/m/requests.json",
        "/m/reference.json",
        f"/out/{out_name}",
        "--mode",
        mode,
        "--arm",
        arm["name"],
        "--warm-passes",
        str(a.warm_passes),
        "--long-passes",
        str(a.long_passes),
    ]
    return cmd + [str(x) for x in arm.get("args", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cpuset", default="10-15")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--image", default="")
    ap.add_argument("--platform", default="linux/arm64")
    ap.add_argument("--only", default="")
    ap.add_argument("--warm-passes", type=int, default=3)
    ap.add_argument("--long-passes", type=int, default=2)
    ap.add_argument("--start-round", type=int, default=0)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.arms))
    cs = cells(cfg["arms"], set(a.only.split(",")) if a.only else None)
    log = open(os.path.join(a.out, "runlog.jsonl"), "a")
    for r in range(a.start_round, a.rounds):
        for pos, i in enumerate(order(len(cs), r)):
            arm, mode = cs[i]
            name = f"r{r}-{arm['name']}-{mode}.json"
            if os.path.exists(os.path.join(a.out, name)):
                continue  # resumable: a finished load is never rerun
            t0 = time.time()
            p = subprocess.run(docker_cmd(cfg, arm, mode, name, a), capture_output=True, text=True)
            rec = {
                "round": r,
                "pos": pos,
                "arm": arm["name"],
                "mode": mode,
                "rc": p.returncode,
                "wall_s": round(time.time() - t0, 3),
                "started": t0,
                "stdout": p.stdout[-2000:],
                "stderr": p.stderr[-4000:],
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(
                json.dumps({k: rec[k] for k in ("round", "pos", "arm", "mode", "rc", "wall_s")}),
                flush=True,
            )
            if p.returncode != 0:
                print(p.stderr[-2000:], file=sys.stderr, flush=True)
    print("LADDER_DONE", a.out)


if __name__ == "__main__":
    main()
