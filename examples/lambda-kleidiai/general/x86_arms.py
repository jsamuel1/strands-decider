#!/usr/bin/env python3
"""Write the x86_64 arms.json from x86_build.sh's packages and the arm64 artifacts copied to this host.

  python3 x86_arms.py <x86 build dir> <artifact dir with p1.gguf, e9.gguf, e5.gguf> <out arms.json>

KleidiAI cannot be compiled for x86 (llama.cpp f74e2fa6a CMake refuses it), so on x86 E8's and E9's kleidiai.cpp
changes are not in the binary. These arms measure what a general default would do to an x86 deployment:
  p1     x86 P1 package + P1 GGUF                 the control
  e8     x86 E8 package + P1 GGUF, switch = 6     the switch has nothing to act on
  e9     x86 E9 package + E9 GGUF                 must refuse (no CPU_KLEIDIAI buffer), never misread the bytes
  e9p1   x86 E9 package + P1 GGUF                 the loader patch alone, inert without a trailer
  e7     x86 E7 package + P1 GGUF, skip = 1       the tied-head skip is generic llama.cpp, so it can still act
  e7off  x86 E7 package + P1 GGUF, unset
  e10    x86 P1 package + n_ctx 4096
  e5     x86 P1 package + E5 GGUF
  e6     x86 P1 package with complete unchecked-hash bytecode (x86_e6.sh)
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
IMG_AMD64 = "public.ecr.aws/lambda/python@sha256:586cea700429caaab11d50f7b96258a77bd4899d253e9559acfb647fe383f970"


def arms(b: str, art: str) -> list[dict]:
    pkg = lambda n: os.path.join(b, f"pkg-{n}")  # noqa: E731
    g = lambda n: os.path.join(art, n)  # noqa: E731
    return [
        {"name": "p1", "pkg": pkg("p1"), "gguf": g("p1.gguf")},
        {
            "name": "e8",
            "pkg": pkg("e8"),
            "gguf": g("p1.gguf"),
            "env": {"GGML_KLEIDIAI_REPACK_THREADS": "6"},
        },
        {"name": "e9", "pkg": pkg("e9"), "gguf": g("e9.gguf")},
        {"name": "e9p1", "pkg": pkg("e9"), "gguf": g("p1.gguf")},
        {
            "name": "e7",
            "pkg": pkg("e7"),
            "gguf": g("p1.gguf"),
            "env": {"LKAI_SKIP_TIED_OUTPUT": "1"},
        },
        {"name": "e7off", "pkg": pkg("e7"), "gguf": g("p1.gguf")},
        {"name": "e10", "pkg": pkg("p1"), "gguf": g("p1.gguf"), "args": ["--n-ctx", "4096"]},
        {"name": "e5", "pkg": pkg("p1"), "gguf": g("e5.gguf")},
        {"name": "e6", "pkg": pkg("e6"), "gguf": g("p1.gguf")},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("art")
    ap.add_argument("out")
    a = ap.parse_args()
    present = [
        x
        for x in arms(os.path.abspath(a.build), os.path.abspath(a.art))
        if os.path.isdir(x["pkg"]) and os.path.exists(x["gguf"])
    ]
    json.dump(
        {
            "image": IMG_AMD64,
            "requests": os.path.join(BASE, "requests.json"),
            "reference": os.path.join(BASE, "reference", "outputs.json"),
            "arms": present,
        },
        open(a.out, "w"),
        indent=1,
    )
    print("arms", [x["name"] for x in present])


if __name__ == "__main__":
    main()
