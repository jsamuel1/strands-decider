"""Fold `perf script -F comm,tid,ip,sym,dso` output into flame-graph stacks ("a;b;c count" per line), root first.

    perf script -i perf.data -F comm,tid,ip,sym,dso | python stackcollapse.py > out.folded

The folded file renders with any flame-graph tool (flamegraph.pl, speedscope, inferno). Frames are "sym (dso)",
with the dso's basename; an unresolved symbol is "[unknown] (dso)".
"""

from __future__ import annotations

import collections
import os
import sys


def collapse(lines: list[str]) -> collections.Counter[str]:
    stacks: collections.Counter[str] = collections.Counter()
    comm, frames = None, []
    for raw in [*lines, ""]:
        line = raw.rstrip("\n")
        if not line.strip():
            if comm is not None and frames:
                stacks[";".join([comm, *reversed(frames)])] += 1
            comm, frames = None, []
            continue
        if not line[0].isspace():
            comm = line.split()[0]
            continue
        parts = line.strip().split(None, 1)
        rest = parts[1] if len(parts) > 1 else "[unknown]"
        sym, _, dso = rest.rpartition(" (")
        dso = os.path.basename(dso.rstrip(")")) if dso else "?"
        sym = (sym or rest).split("+0x")[0]
        frames.append(f"{sym} ({dso})".replace(";", ":"))
    return stacks


def main() -> None:
    for stack, n in sorted(collapse(sys.stdin.readlines()).items()):
        print(f"{stack} {n}")


if __name__ == "__main__":
    main()
