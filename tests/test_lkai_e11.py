"""examples/lambda-kleidiai/e11: the dormant memfd-release patch, the arm environments, the paired wave order, the
memory parsers and the E11 summary. No llama.cpp build, Lambda or AWS needed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E11 = ROOT / "examples/lambda-kleidiai/e11"
BASE = ROOT / "examples/lambda-kleidiai/baseline"
sys.path.insert(0, str(E11))
sys.path.insert(0, str(BASE))

patch = importlib.import_module("patch_memfdrel")
memstat = importlib.import_module("memstat")
hostmeta = importlib.import_module("hostmeta")
e11deploy = importlib.import_module("e11deploy")
e11bench = importlib.import_module("e11bench")


# --- the loader patch ---------------------------------------------------------------------------------------------


def _fake_llama(root: Path) -> None:
    """The anchor regions of llama.cpp f74e2fa6a's loader, verbatim, inside filler."""
    (root / "src").mkdir(parents=True)
    (root / "src/llama-model-loader.h").write_text("struct x {\n" + patch.H_OLD + "};\n")
    (root / "src/llama-model-loader.cpp").write_text(
        "#include <regex>\n"
        + patch.INC_OLD
        + "\nvoid f() {\n            if (buf_mmap) {\n"
        + patch.ALIAS_OLD
        + "            }\n    if (size_done >= size_data) {\n        if (use_mmap) {\n"
        "            for (uint32_t idx = 0; idx < mappings.size(); idx++) {\n"
        + patch.REL_OLD
        + "    }\n}\n"
    )


def test_patch_applies_every_edit_once_and_refuses_reapplication(tmp_path):
    _fake_llama(tmp_path)
    patch.apply(tmp_path)
    h = (tmp_path / "src/llama-model-loader.h").read_text()
    c = (tmp_path / "src/llama-model-loader.cpp").read_text()
    assert h.count("mmaps_alias;") == 1
    assert c.count("lkai_memfd_release_stats[6] = {") == 1
    assert (
        c.count("mmaps_alias[weight->idx].emplace_back(weight->offs, weight->offs + n_size);") == 1
    )
    assert c.count('getenv("LKAI_RELEASE_MEMFD")') == 1
    # the release follows llama's own final unmap, and munmap precedes the hole punch inside it
    assert c.index("mapping->unmap_fragment(mmap_used.second") < c.index(
        'getenv("LKAI_RELEASE_MEMFD")'
    )
    rel = c[c.index('getenv("LKAI_RELEASE_MEMFD")') :]
    assert rel.index("unmap_fragment(r.first, r.second)") < rel.index("FALLOC_FL_PUNCH_HOLE")
    with pytest.raises(SystemExit, match="already carries"):
        patch.apply(tmp_path)


def test_patch_is_dormant_unless_flag_is_1():
    # The release block is entered only for use_mmap with LKAI_RELEASE_MEMFD starting with '1'.
    assert "if (use_mmap && lkai_rel && lkai_rel[0] == '1') {" in patch.REL_NEW


def test_patch_refuses_a_moved_anchor(tmp_path):
    _fake_llama(tmp_path)
    c = tmp_path / "src/llama-model-loader.cpp"
    c.write_text(
        c.read_text().replace("mmap_used.second = std::max", "mmap_used.second = std::min")
    )
    with pytest.raises(SystemExit, match="anchor found 0 times"):
        patch.apply(tmp_path)
    # nothing is half-applied
    assert "mmaps_alias" not in (tmp_path / "src/llama-model-loader.h").read_text()


def test_p1_patch_file_is_untouched_by_e11():
    # E11's change lives in e11/; P1's llama stage key (llama/patch_llama.py bytes) must not move.
    src = (ROOT / "examples/lambda-kleidiai/llama/patch_llama.py").read_text()
    assert "memfdrel" not in src and "LKAI_RELEASE_MEMFD" not in src


# --- arms ---------------------------------------------------------------------------------------------------------


P1_ENV = {
    "MODEL_BUCKET": "b",
    "MODEL_KEY": "k",
    "DEC_KV": "f32",
    "DEC_UBATCH": "512",
    "N_SEQ": "17",
    "LKAI_PHASES": "0",
    "COLD_TAG": "x",
}


def test_arm_environments_differ_only_in_the_release_flag():
    off, rel = (e11deploy.arm_environment(P1_ENV, a) for a in ("off", "rel"))
    assert "LKAI_RELEASE_MEMFD" not in off
    assert rel["LKAI_RELEASE_MEMFD"] == "1"
    assert off["LKAI_PHASES"] == rel["LKAI_PHASES"] == "1"
    strip = lambda e: {k: v for k, v in e.items() if k not in ("LKAI_RELEASE_MEMFD", "E11_ARM")}  # noqa: E731
    assert strip(off) == strip(rel)
    assert {k: off[k] for k in ("MODEL_BUCKET", "MODEL_KEY", "DEC_KV", "DEC_UBATCH", "N_SEQ")} == {
        k: P1_ENV[k] for k in ("MODEL_BUCKET", "MODEL_KEY", "DEC_KV", "DEC_UBATCH", "N_SEQ")
    }


def test_arm_environment_drops_an_inherited_flag_and_rejects_unknown_arms():
    env = e11deploy.arm_environment(dict(P1_ENV, LKAI_RELEASE_MEMFD="1"), "off")
    assert "LKAI_RELEASE_MEMFD" not in env
    with pytest.raises(ValueError):
        e11deploy.arm_environment(P1_ENV, "both")


def test_paired_waves_interleave_arms_and_flip_each_round():
    assert e11bench.order_for(["off", "rel"], 0) == ["off", "rel"]
    assert e11bench.order_for(["off", "rel"], 1) == ["rel", "off"]
    jobs = e11bench.interleave(["off", "rel"], 3)
    assert jobs == [("off", 0), ("rel", 0), ("off", 1), ("rel", 1), ("off", 2), ("rel", 2)]


# --- memory accounting --------------------------------------------------------------------------------------------


STATUS = (
    "Name:\tpython3.12\nVmHWM:\t 5012856 kB\nVmRSS:\t 4895916 kB\nRssAnon:\t 2910872 kB\nRssFile:\t   30884 kB\n"
    "RssShmem:\t 1954160 kB\nThreads:\t7\n"
)


def test_status_fields_parse_and_missing_ones_are_unknown():
    got = memstat.parse_kb_fields(STATUS, memstat.STATUS_KEYS)
    assert got["VmRSS"] == 4895916 and got["RssShmem"] == 1954160 and got["RssAnon"] == 2910872
    assert got["VmSwap"] == "unknown"


SMAPS = """\
ffff00000000-ffff74800000 r--s 00a7d000 00:01 1234                       /memfd:s2b-h512-Q8_0.gguf (deleted)
Size:            1904640 kB
Rss:             1904640 kB
Shared_Clean:          0 kB
Private_Dirty:   1904640 kB
ffff80000000-ffff80001000 r-xp 00000000 103:01 99                        /var/task/lib/libllama.so
Size:                  4 kB
Rss:                   4 kB
ffff90000000-ffff93000000 r--s 7f000000 00:01 1234                       /memfd:s2b-h512-Q8_0.gguf (deleted)
Size:              49152 kB
Rss:               49520 kB
Private_Dirty:     49520 kB
"""


def test_smaps_sums_only_the_matching_mappings():
    got = memstat.parse_smaps(SMAPS, "/memfd:s2b-h512-Q8_0.gguf")
    assert got["summary"]["entries"] == 2
    assert got["summary"]["Size_kb"] == 1904640 + 49152
    assert got["summary"]["Rss_kb"] == 1904640 + 49520
    assert memstat.parse_smaps(SMAPS, "/memfd:other")["summary"]["entries"] == 0


def test_cgroup_memory_stat_keeps_only_known_counters():
    got = memstat.parse_memory_stat("anon 100\nfile 7\nshmem 50\nfile_mapped 3\nkernel 9\n")
    assert got == {"anon": 100, "file": 7, "shmem": 50, "file_mapped": 3}
    assert memstat.parse_memory_stat(None)["shmem"] == "unknown"


def test_fd_resident_bytes_tracks_a_hole_punch(tmp_path):
    os = pytest.importorskip("os")
    if not hasattr(os, "memfd_create"):
        pytest.skip("needs memfd_create")
    import ctypes
    import fcntl  # noqa: F401  (Linux)

    fd = os.memfd_create("t")
    page = 4096
    os.write(fd, b"x" * (8 * page))
    assert memstat.fd_resident_bytes(fd, 8 * page) == 8 * page
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
    assert libc.fallocate(fd, 0x02 | 0x01, 2 * page, 4 * page) == 0  # PUNCH_HOLE | KEEP_SIZE
    assert memstat.fd_resident_bytes(fd, 8 * page) == 4 * page
    assert os.fstat(fd).st_size == 8 * page
    os.close(fd)


# --- host metadata ------------------------------------------------------------------------------------------------


def test_cpuinfo_reads_identity_without_naming_a_generation():
    text = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\nFeatures\t: fp asimd\n\nprocessor\t: 1\n"
    got = hostmeta.cpuinfo(text)
    assert got["part"] == "0xd40" and got["implementer"] == "0x41" and got["processors"] == 2
    assert got["variant"] == "unknown"
    assert not any("graviton" in str(v).lower() for v in got.values())
    assert hostmeta.cpuinfo(None)["processors"] == "unknown"


def test_no_e11_source_names_a_processor_generation():
    for p in E11.glob("*.py"):
        assert "graviton" not in p.read_text().lower().replace("graviton ec2", ""), p


def test_auxv_reads_hwcaps():
    import struct

    raw = struct.pack("<QQQQQQ", 16, 0xEFFFFFFF, 26, 0x2F3, 0, 0)
    assert hostmeta.auxv(raw) == {"hwcap": "0xefffffff", "hwcap2": "0x2f3"}
    assert hostmeta.auxv(b"\x00" * 5) == {"hwcap": "unknown", "hwcap2": "unknown"}


def test_bench_rejects_arms_other_than_off_and_rel(monkeypatch, capsys):
    pytest.importorskip("boto3")
    monkeypatch.setattr(
        sys, "argv", ["e11bench.py", "--arms", "off,off", "--out", "x", "--profile", "p"]
    )
    with pytest.raises(SystemExit):
        e11bench.main()
    assert "--arms must be exactly" in capsys.readouterr().err


def test_requests_file_is_p1s():
    reqs = json.loads((BASE / "requests.json").read_text())
    assert reqs["probe"] in {r["id"] for r in reqs["requests"]}
