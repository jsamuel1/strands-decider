#!/usr/bin/env python3
"""Write the general ladder's arms.json for a staged host (stage.sh output). Standard library only.

  python3 make_arms.py <stage dir> <artifact dir> <out arms.json> [--image IMG]

Every arm is P1 except its one variable (inputs.json -> rung_sources):

  p1        P1's zip + P1's GGUF                                   the control, in every round
  e8        E8's off.zip + P1's GGUF, GGML_KLEIDIAI_REPACK_THREADS=6
  e8off     E8's off.zip + P1's GGUF, switch unset                 E8's dormant-patch check
  e9        E9's zip + E9's prepacked GGUF
  e9p1      E9's zip + P1's GGUF                                   E9's loader is inert without a trailer
  e7        E7's zip + P1's GGUF, LKAI_SKIP_TIED_OUTPUT=1
  e7off     E7's zip + P1's GGUF, switch unset                     E7's dormant-patch check
  e10       P1's zip + P1's GGUF, n_ctx 4096 (E10's wrap_shim)
  e5        P1's zip + E5's no-vocab GGUF
  e6        E6's pyc.zip (P1's zip + complete unchecked-hash bytecode) + P1's GGUF

On x86_64 only the arms whose artifacts exist for x86 are written (the P1 zip is arm64-only: see README).
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
IMG = "public.ecr.aws/lambda/python@sha256:37fe9ca4183c36ec57e4c5eabb8893cd9fbc9c0f3053b903d9a57eb2bd6bafcc"


def arms(stage: str, art: str) -> list[dict]:
    pkg = lambda n: os.path.join(stage, f"pkg-{n}")  # noqa: E731
    g = lambda n: os.path.join(art, n)  # noqa: E731
    both = ["local", "memfd"]
    return [
        {"name": "p1", "pkg": pkg("p1"), "gguf": g("p1.gguf"), "modes": both},
        {
            "name": "e8",
            "pkg": pkg("e8off"),
            "gguf": g("p1.gguf"),
            "modes": both,
            "env": {"GGML_KLEIDIAI_REPACK_THREADS": "6"},
        },
        {"name": "e8off", "pkg": pkg("e8off"), "gguf": g("p1.gguf"), "modes": ["local"]},
        {"name": "e9", "pkg": pkg("e9"), "gguf": g("e9.gguf"), "modes": both},
        {"name": "e9p1", "pkg": pkg("e9"), "gguf": g("p1.gguf"), "modes": ["local"]},
        {
            "name": "e7",
            "pkg": pkg("e7"),
            "gguf": g("p1.gguf"),
            "modes": both,
            "env": {"LKAI_SKIP_TIED_OUTPUT": "1"},
        },
        {"name": "e7off", "pkg": pkg("e7"), "gguf": g("p1.gguf"), "modes": ["local"]},
        {
            "name": "e10",
            "pkg": pkg("p1"),
            "gguf": g("p1.gguf"),
            "modes": both,
            "args": ["--n-ctx", "4096"],
        },
        {"name": "e5", "pkg": pkg("p1"), "gguf": g("e5.gguf"), "modes": both},
        {"name": "e6", "pkg": pkg("e6pyc"), "gguf": g("p1.gguf"), "modes": ["local"]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("art")
    ap.add_argument("out")
    ap.add_argument("--image", default=IMG)
    a = ap.parse_args()
    stage, art = os.path.abspath(a.stage), os.path.abspath(a.art)
    present = [x for x in arms(stage, art) if os.path.isdir(x["pkg"]) and os.path.exists(x["gguf"])]
    cfg = {
        "image": a.image,
        "requests": os.path.join(BASE, "requests.json"),
        "reference": os.path.join(BASE, "reference", "outputs.json"),
        "arms": present,
    }
    json.dump(cfg, open(a.out, "w"), indent=1)
    print("arms", [x["name"] for x in present])


if __name__ == "__main__":
    main()
