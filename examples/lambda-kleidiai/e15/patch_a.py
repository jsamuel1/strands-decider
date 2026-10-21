"""Patch E15's deployed candidate into llama.cpp f74e2fa6a: candidate A only, behind LKAI_E15_A.

  python patch_a.py /path/to/llama.cpp        (after llama/patch_llama.py)

The deployed experiment ships one variable. This applies patch_candidates.py's candidate A (llamafile tinyBLAS F32
sgemm for F32 x F32 MUL_MAT in the SVE / i8mm CPU variants, enabled by LKAI_E15_A=1) and nothing else: no node
profiler and no candidate B.
"""

import argparse
from pathlib import Path

import patch_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    patch_candidates.patch_a(ap.parse_args().root)


if __name__ == "__main__":
    main()
