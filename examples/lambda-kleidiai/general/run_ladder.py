#!/usr/bin/env python3
"""Run the general ladder's arms on a fixed host, interleaved, one fresh container per load. FIXED HOST, NOT LAMBDA.

  python3 run_ladder.py <arms.json> <out_dir> [--rounds 6] [--seed 20261113] [--cpuset 10-15] [--threads 6]
                        [--image IMG] [--platform linux/arm64] [--only arm1,arm2] [--warm-passes 3] [--long-passes 2]

arms.json: {"image": ..., "requests": ..., "reference": ..., "arms": [{"name", "pkg", "gguf", "env": {},
"args": [], "modes": ["local", "memfd"]}]}. Paths are host paths; each load mounts the arm's unzipped package
read-only at /var/task (as Lambda does), the GGUF read-only, and this directory.

Interleaving: every (arm, mode) cell runs once per round, in a reversal-paired random order. Rounds come in pairs:
round 2k runs a seeded random permutation of the cells, round 2k+1 runs the same permutation reversed. Within the
rounds actually run (an even count, enforced), every pair of cells is therefore in each order exactly rounds/2 times
(so each candidate runs before its control as often as after it), and every cell's mean position is the middle of
the round (so a linear drift within a round adds the same to every cell). Different pairs use different
permutations, so no fixed neighbour sits next to a cell in every round. The schedule is written to schedule.json
before the first load. Standard library only (Python 3.9+, the DevSpace's system python).
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def schedule(n: int, rounds: int, seed: int) -> list[list[int]]:
    """Every round's cell order: seeded random permutations, each followed by its reversal (rounds must be even)."""
    if rounds < 2 or rounds % 2:
        raise ValueError(f"rounds must be an even number >= 2 (reversal pairs), got {rounds}")
    rng = random.Random(seed)
    out = []
    for _ in range(rounds // 2):
        p = list(range(n))
        rng.shuffle(p)
        out += [p, p[::-1]]
    return out


def balance(sched: list[list[int]], i: int, j: int) -> tuple[int, int]:
    """(rounds where cell i runs before cell j, rounds where it runs after)."""
    before = sum(1 for o in sched if o.index(i) < o.index(j))
    return before, len(sched) - before


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


REFUSAL_TAG = "LKAI_REFUSAL "  # probe.py's refusal line


def parse_refusal(stdout: str) -> dict | None:
    """probe.py's one `LKAI_REFUSAL <json>` line, or None. Two lines, or one that is not a JSON object, is malformed
    and recorded as such (the analysis then refuses it), never silently dropped."""
    lines = [ln[len(REFUSAL_TAG) :] for ln in stdout.splitlines() if ln.startswith(REFUSAL_TAG)]
    if not lines:
        return None
    try:
        r = json.loads(lines[0])
    except ValueError:
        r = None
    if len(lines) != 1 or not isinstance(r, dict):
        return {"reason": "malformed-refusal-record", "lines": len(lines)}
    return r


def prime(cfg: dict, cs: list[tuple[dict, str]], a) -> dict:
    """Read every file a load will touch (each arm's package tree and GGUF, the requests and reference) once, and
    start the image once, before round 0. The ladder measures a warm page cache in EVERY round; without this, round 0
    pays each artifact's first disk read (a freshly started host), and that one round can flip a paired delta's sign."""
    t0 = time.time()
    paths = set()
    for arm, _ in cs:
        paths |= {os.path.realpath(arm["pkg"]), os.path.realpath(arm["gguf"])}
    paths |= {os.path.realpath(cfg["requests"]), os.path.realpath(cfg["reference"])}
    files = n = 0
    for p in sorted(paths):
        walk = [(p, [], [""])] if os.path.isfile(p) else os.walk(p)
        for d, _, fs in walk:
            for f in fs:
                q = os.path.realpath(os.path.join(d, f)) if f else p
                if not os.path.isfile(q):
                    continue
                with open(q, "rb") as fh:
                    while b := fh.read(16 << 20):
                        n += len(b)
                files += 1
    image = cfg.get("image") or a.image
    rc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            a.platform,
            "--network",
            "none",
            "--entrypoint",
            "python3.12",
            image,
            "-c",
            "pass",
        ],
        capture_output=True,
    ).returncode
    return {"files": files, "bytes": n, "image_start_rc": rc, "seconds": round(time.time() - t0, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261113)
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
    sched = schedule(len(cs), a.rounds, a.seed)
    names = [f"{arm['name']}-{mode}" for arm, mode in cs]
    sp = os.path.join(a.out, "schedule.json")
    plan = {
        "rounds": a.rounds,
        "seed": a.seed,
        "cells": names,
        "order": [[names[i] for i in o] for o in sched],
    }
    if os.path.exists(sp) and json.load(open(sp)) != plan:
        raise SystemExit(
            f"{sp} holds another schedule: resume with the same arms, --only, --rounds and --seed"
        )
    json.dump(plan, open(sp, "w"), indent=1)
    if not os.path.exists(os.path.join(a.out, "prime.json")):
        pr = prime(cfg, cs, a)
        json.dump(pr, open(os.path.join(a.out, "prime.json"), "w"), indent=1)
        print(json.dumps({"prime": pr}), flush=True)
    log = open(os.path.join(a.out, "runlog.jsonl"), "a")
    for r in range(a.start_round, a.rounds):
        for pos, i in enumerate(sched[r]):
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
            refusal = parse_refusal(p.stdout)
            if refusal is not None:
                rec["refusal"] = refusal
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
