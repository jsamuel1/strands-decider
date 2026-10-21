"""E8 benchmark: P1's harness (baseline/bench.py) over E8's three arms, base, off and par.

  python bench_e8.py --build e8/build --arms base,off,par --rounds 5 --concurrency 10 --warm 10 --read-timeout 300 \
      --out e8/build/runs/<id>

Every invocation, its order and timing, and everything recorded are bench.py's at P1's pushed HEAD (70eab7b), which
opens a fresh Lambda client for every wave (I1's fix). Per round: a new COLD_TAG on every arm, a cold wave per arm, a
published version per arm, a restore wave per arm, then `warm` passes that alternate arm order pass by pass. The arm
order reverses every round, so each arm runs both early and late in a round. Arms run as sequential waves within a
round (alternation, not simultaneous interleaving): time-within-round drift is a stated limitation.

This file changes only what E8 needs:
- the arms are base, off and par, all LKAI_PHASES=1 (bench.py otherwise insists on exactly inst and rep);
- each arm's deployed code is checked against its own zip (base.zip or off.zip), not one function.zip;
- before starting, each deployed arm must carry exactly its GGML_KLEIDIAI_REPACK_THREADS value (deploy_e8.ARMS);
- harness provenance also covers E8's own files;
- --read-timeout bounds one invoke (bench.py's 900 s) and is recorded in conditions.json.
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
import deploy  # noqa: E402
import deploy_e8  # noqa: E402
import summarize  # noqa: E402
from botocore.config import Config  # noqa: E402

# The real CodeSha256 decoder, kept before per_arm_code_check replaces deploy.b64_to_hex for bench.py's one-zip check.
REAL_B64_TO_HEX = deploy.b64_to_hex

E8_SOURCES = [
    f"examples/lambda-kleidiai/e8/{f}"
    for f in (
        "patch_parallel_repack.py",
        "kaihash.cpp",
        "make_zips.py",
        "e8_entry.py",
        "e8_host.py",
        "build.sh",
        "hash_check.sh",
        "probe_e8.py",
        "probe_e8.sh",
        "deploy_e8.py",
        "bench_e8.py",
        "summarize_e8.py",
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
    """Why a deployed arm's environment is not that arm's configuration, or None."""
    if arm not in deploy_e8.ARMS:
        return f"{arm}: not an E8 arm ({sorted(deploy_e8.ARMS)})"
    want = deploy_e8.ARMS[arm][1]
    have = env.get(deploy_e8.SWITCH)
    if have != want:
        return f"{arm}: deployed {deploy_e8.SWITCH}={have!r}, the arm needs {want!r}"
    if env.get("LKAI_PHASES") != "1":
        return f"{arm}: deployed LKAI_PHASES={env.get('LKAI_PHASES')!r}, E8 arms need '1'"
    if env.get("E8_ARM") != arm:
        return f"{arm}: deployed E8_ARM={env.get('E8_ARM')!r}"
    return None


def install(read_timeout: int) -> None:
    summarize.ARM_PHASES = {arm: "1" for arm in deploy_e8.ARMS}
    buildcache.UNKEYED_SOURCES = [*buildcache.UNKEYED_SOURCES, *E8_SOURCES]

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
            if REAL_B64_TO_HEX(c["CodeSha256"]) != f["code_sha256"]:
                sys.exit(
                    f"refusing to benchmark: {arm} runs {c['CodeSha256']}, not its {f['zip']}.zip"
                )

    bench.Bench.__init__ = init


def per_arm_code_check(st: dict) -> None:
    """bench.py compares every arm's deployed code with the one manifest.function_zip; E8 has one zip per arm. Its
    check is satisfied for code that is one of E8's deployed zips, and Bench.__init__ (above) then checks each arm
    against its own zip with the real decoder, so an arm running the other arm's zip is still refused."""
    known = {f["code_sha256"] for f in st.get("functions", {}).values()}

    def b64_to_hex(v: str) -> str:
        h = REAL_B64_TO_HEX(v)
        return PER_ARM if h in known else h

    st["manifest"]["function_zip"] = {"sha256": PER_ARM, "zips": st["manifest"]["zips"]}
    deploy.b64_to_hex = b64_to_hex


PER_ARM = "e8-per-arm (see zips)"


def stamp_conditions(out: str, read_timeout: int) -> None:
    """Record E8's harness additions in the run's conditions.json (bench.py writes the rest)."""
    path = os.path.join(out, "conditions.json")
    if not os.path.exists(path):
        return
    c = json.load(open(path))
    c["experiment"] = "E8"
    c["read_timeout_s"] = read_timeout
    c["harness_e8"] = "examples/lambda-kleidiai/e8/bench_e8.py over baseline/bench.py"
    c["e8_arms"] = {
        a: {"zip": z, deploy_e8.SWITCH: v or "unset"} for a, (z, v) in deploy_e8.ARMS.items()
    }
    json.dump(c, open(path, "w"), indent=1)


def out_dir(argv: list[str]) -> str | None:
    for i, x in enumerate(argv):
        if x == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--out="):
            return x.split("=", 1)[1]
    return None


def build_dir(argv: list[str]) -> str:
    for i, x in enumerate(argv):
        if x == "--build" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--build="):
            return x.split("=", 1)[1]
    return deploy_e8.BUILD


if __name__ == "__main__":
    rt, rest = read_timeout_from(sys.argv[1:])
    install(rt)
    real_load = deploy.load_state

    def load_state(build: str) -> dict:
        st = real_load(build)
        if st.get("manifest"):
            per_arm_code_check(st)
        return st

    deploy.load_state = load_state
    sys.argv = [sys.argv[0], *rest]
    try:
        bench.main()
    finally:
        if out := out_dir(rest):
            stamp_conditions(out, rt)
