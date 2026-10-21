"""E7 benchmark: P1's harness (baseline/bench.py) over E7's two arms, ctl and skip.

  python bench_e7.py --build e7/build --arms ctl,skip --rounds 5 --concurrency 10 --warm 10 --read-timeout 300 \
      --out e7/build/runs/<id>

Every invocation, its order and timing, and everything recorded are bench.py's at P1's pushed HEAD, which already
opens a fresh Lambda client for every wave (I1's fix). Per round: a new COLD_TAG on both arms, a cold wave per arm,
a published version per arm, a restore wave per arm, then `warm` passes that alternate the arm pass by pass. The arm
that goes first alternates every round. This file changes only what E7 needs:
- the arms are ctl and skip, both LKAI_PHASES=1 (bench.py otherwise insists on inst and rep);
- before starting, each deployed arm must carry exactly its LKAI_SKIP_TIED_OUTPUT value (deploy_e7.ARMS);
- harness provenance also covers E7's own files;
- --read-timeout bounds one invoke (bench.py's 900 s), and is recorded in conditions.json.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import bench  # noqa: E402
import buildcache  # noqa: E402
import deploy_e7  # noqa: E402
import summarize  # noqa: E402
from botocore.config import Config  # noqa: E402

E7_SOURCES = [
    f"examples/lambda-kleidiai/e7/{f}"
    for f in (
        "patch_skip_output.py",
        "e7_entry.py",
        "e7_host.py",
        "build.sh",
        "deploy_e7.py",
        "bench_e7.py",
        "summarize_e7.py",
    )
]


def read_timeout_from(argv: list[str]) -> tuple[int, list[str]]:
    """Strip --read-timeout N from argv (bench.py does not know it); default bench.py's 900 s."""
    out, t, i = [], 900, 0
    while i < len(argv):
        if argv[i] == "--read-timeout":
            t = bench.positive_int(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--read-timeout="):
            t = bench.positive_int(argv[i].split("=", 1)[1])
        else:
            out.append(argv[i])
        i += 1
    return t, out


def arm_problem(arm: str, env: dict) -> str | None:
    """Why a deployed arm's environment is not that arm's one variable, or None."""
    want = deploy_e7.ARMS.get(arm, "?")
    have = env.get(deploy_e7.SWITCH)
    if have != want:
        return f"{arm}: deployed {deploy_e7.SWITCH}={have!r}, the arm needs {want!r}"
    if env.get("LKAI_PHASES") != "1":
        return f"{arm}: deployed LKAI_PHASES={env.get('LKAI_PHASES')!r}, E7 arms need '1'"
    return None


def install(read_timeout: int) -> None:
    summarize.ARM_PHASES = {arm: "1" for arm in deploy_e7.ARMS}
    buildcache.UNKEYED_SOURCES = [*buildcache.UNKEYED_SOURCES, *E7_SOURCES]

    def new_invoke_client(self):
        """bench.py's fresh per-wave client (no retries, a pool as wide as the wave), with --read-timeout."""
        self.invoke_clients += 1
        return self.sess.client(
            "lambda",
            config=Config(
                read_timeout=read_timeout,
                connect_timeout=10,
                retries={"max_attempts": 0},
                max_pool_connections=max(16, self.concurrency * 2),
            ),
        )

    bench.Bench.new_invoke_client = new_invoke_client
    real_init = bench.Bench.__init__

    def init(self, sess, st, *a, **k):
        real_init(self, sess, st, *a, **k)
        for arm, f in st["functions"].items():
            c = self.lam.get_function_configuration(FunctionName=f["name"])
            if p := arm_problem(arm, c["Environment"]["Variables"]):
                sys.exit(f"refusing to benchmark: {p}")

    bench.Bench.__init__ = init


def stamp_conditions(out: str, read_timeout: int) -> None:
    """Record E7's harness additions in the run's conditions.json (bench.py writes the rest)."""
    path = os.path.join(out, "conditions.json")
    if not os.path.exists(path):
        return
    c = json.load(open(path))
    c["experiment"] = "E7"
    c["read_timeout_s"] = read_timeout
    c["harness_e7"] = "examples/lambda-kleidiai/e7/bench_e7.py over baseline/bench.py"
    json.dump(c, open(path, "w"), indent=1)


def out_dir(argv: list[str]) -> str | None:
    for i, x in enumerate(argv):
        if x == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--out="):
            return x.split("=", 1)[1]
    return None


if __name__ == "__main__":
    rt, rest = read_timeout_from(sys.argv[1:])
    install(rt)
    sys.argv = [sys.argv[0], *rest]
    try:
        bench.main()
    finally:
        if out := out_dir(rest):
            stamp_conditions(out, rt)
