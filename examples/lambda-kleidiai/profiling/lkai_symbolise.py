"""Symbolise lkaisamp.c's samples in-process: /proc/self/maps plus the ELF symbol tables of the mapped files.

Standard library only (the Lambda zip has no binutils). `summary(lib)` reads the sampler's buffer through samp_dump,
resolves every sample's program counter (and, on aarch64, its link register: the caller of a leaf) to `library:symbol`,
and returns bounded tables per phase tag: top symbols by self samples, top callers, and samples per library.

A program counter outside every symbol (stripped code, JIT, the vDSO) is reported as `library:+0xOFFSET` binned to
4 KiB so an offline objdump of the same zip can name it.
"""

from __future__ import annotations

import bisect
import os
import struct
import tempfile
from collections import Counter

_REC = struct.Struct("<QQII")
TAG_NAMES: dict[int, str] = {0: "untagged", 1: "init", 7: "invoke"}


class _Elf:
    """Function symbols (STT_FUNC, size > 0) of one ELF64 little-endian file, from .symtab and .dynsym."""

    def __init__(self, path: str):
        self.addrs: list[int] = []
        self.ends: list[int] = []
        self.names: list[str] = []
        self.load_bias_vaddr = 0  # vaddr of the first PT_LOAD with offset 0
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            return
        e_phoff, e_shoff = struct.unpack_from("<QQ", data, 0x20)
        e_phentsize, e_phnum, e_shentsize, e_shnum = struct.unpack_from("<HHHH", data, 0x36)
        for i in range(e_phnum):
            p_type, _flags, p_offset, p_vaddr = struct.unpack_from("<IIQQ", data, e_phoff + i * e_phentsize)
            if p_type == 1 and p_offset == 0:  # PT_LOAD
                self.load_bias_vaddr = p_vaddr
                break
        secs = [struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize) for i in range(e_shnum)]
        syms: dict[int, tuple[int, str]] = {}
        for _name, sh_type, _fl, _addr, off, size, link, _info, _al, entsize in secs:
            if sh_type not in (2, 11) or not entsize:  # SHT_SYMTAB, SHT_DYNSYM
                continue
            stroff = secs[link][4]
            for j in range(size // entsize):
                st_name, st_info, _o, shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, off + j * entsize)
                if (st_info & 0xF) != 2 or not st_value or not st_size or not shndx:
                    continue
                end = data.index(b"\0", stroff + st_name)
                name = data[stroff + st_name:end].decode(errors="replace")
                if st_value not in syms or len(name) < len(syms[st_value][1]):
                    syms[st_value] = (st_size, name)
        for a in sorted(syms):
            self.addrs.append(a)
            self.ends.append(a + syms[a][0])
            self.names.append(syms[a][1])

    def lookup(self, vaddr: int) -> str | None:
        i = bisect.bisect_right(self.addrs, vaddr) - 1
        if i >= 0 and vaddr < self.ends[i]:
            return self.names[i]
        return None


class Symbolizer:
    def __init__(self, maps_text: str):
        self.maps: list[tuple[int, int, int, str]] = []
        for line in maps_text.splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) < 6 or "x" not in parts[1]:
                continue
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            self.maps.append((lo, hi, int(parts[2], 16), parts[5].strip()))
        self.maps.sort()
        self._starts = [m[0] for m in self.maps]
        self._elf: dict[str, _Elf] = {}
        self._cache: dict[int, str] = {}

    def __call__(self, addr: int) -> str:
        if addr in self._cache:
            return self._cache[addr]
        i = bisect.bisect_right(self._starts, addr) - 1
        if i < 0 or addr >= self.maps[i][1]:
            s = "?"
        else:
            lo, _hi, off, path = self.maps[i]
            lib = os.path.basename(path) or path
            if path.startswith("/") and not path.startswith("/memfd:"):
                elf = self._elf.get(path) or self._elf.setdefault(path, _Elf(path))
                # file offset of addr, then the ELF vaddr it was linked at
                vaddr = addr - lo + off + elf.load_bias_vaddr
                name = elf.lookup(vaddr)
                s = f"{lib}:{name}" if name else f"{lib}:+{(addr - lo + off) & ~0xFFF:#x}"
            else:
                s = lib
        self._cache[addr] = s
        return s


def read_samples(path: str) -> tuple[dict, list[tuple[int, int, int, int]]]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"LKSAMP01":
        raise ValueError("not an lkaisamp dump")
    period, n, dropped, rec = struct.unpack_from("<QQQQ", data, 8)
    if rec != _REC.size:
        raise ValueError(f"record size {rec} != {_REC.size}")
    recs = [_REC.unpack_from(data, 40 + i * rec) for i in range(n)]
    return {"period_us": period, "n": n, "dropped": dropped}, recs


def tables(recs, sym: Symbolizer, tag_names: dict[int, str] | None = None, top: int = 25) -> dict:
    names = {**TAG_NAMES, **(tag_names or {})}
    by_tag: dict[int, list] = {}
    for pc, lr, tid, tag in recs:
        by_tag.setdefault(tag, []).append((pc, lr, tid))
    out = {}
    for tag, rs in sorted(by_tag.items()):
        self_c, lib_c, edge_c, thr = Counter(), Counter(), Counter(), Counter()
        for pc, lr, tid in rs:
            s = sym(pc)
            self_c[s] += 1
            lib_c[s.split(":", 1)[0]] += 1
            thr[tid] += 1
            if lr:
                edge_c[f"{sym(lr)} -> {s}"] += 1
        out[names.get(tag, f"tag{tag}")] = {
            "samples": len(rs),
            "threads": len(thr),
            "top_self": [{"symbol": k[:200], "samples": v, "pct": round(100 * v / len(rs), 2)}
                         for k, v in self_c.most_common(top)],
            "by_library": [{"library": k, "samples": v, "pct": round(100 * v / len(rs), 2)}
                           for k, v in lib_c.most_common(12)],
            "top_caller_edges": [{"edge": k[:300], "samples": v} for k, v in edge_c.most_common(12)],
        }
    return out


_SYM: dict = {}


def summary(lib, tag_names: dict[int, str] | None = None, top: int = 25) -> dict:
    """Dump the live sampler (a ctypes CDLL of lkaisamp.so) and symbolise it against this process's maps. The
    Symbolizer (parsed ELF tables) is reused while the maps are unchanged, so later invocations pay only the lookups."""
    fd, path = tempfile.mkstemp(prefix="lkaisamp-", suffix=".bin")
    try:
        if lib.samp_dump(fd) != 0:
            return {"error": "samp_dump failed"}
        os.close(fd)
        fd = -1
        hdr, recs = read_samples(path)
    finally:
        if fd >= 0:
            os.close(fd)
        os.unlink(path)
    with open("/proc/self/maps") as f:
        maps = f.read()
    if _SYM.get("maps") != maps:
        _SYM.update(maps=maps, sym=Symbolizer(maps))
    sym = _SYM["sym"]
    return hdr | {"phases": tables(recs, sym, tag_names, top)}


def summary_file(samples: str, maps: str, tag_names: dict[int, str] | None = None, top: int = 25) -> dict:
    """Offline: a dump plus the maps file saved beside it (offline_drive.py --samples)."""
    hdr, recs = read_samples(samples)
    with open(maps) as f:
        sym = Symbolizer(f.read())
    return hdr | {"phases": tables(recs, sym, tag_names, top)}
