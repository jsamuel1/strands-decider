"""E11's one loader change, applied on top of P1's patched llama.cpp (llama/patch_llama.py, left unchanged):

  python e11/patch_memfdrel.py /path/to/llama.cpp     # after llama/patch_llama.py

LKAI_RELEASE_MEMFD=1 (dormant unless set, so one build serves both arms): once the last tensor is loaded, release
every page of the weight file that no tensor still aliases. llama.cpp already munmaps the mapping outside
[first, last) of the tensors left in the mmap CPU buffer, but for a memfd that frees nothing, because shmem pages
belong to the file, not to the mapping. Every tensor copied into another buffer (each KleidiAI-repacked Q8_0 matmul
weight) therefore leaves its bytes resident for the life of the process, and in the SnapStart snapshot.

The patch records the exact byte range of every tensor the loader aliases into the mapping (the
ggml_backend_tensor_alloc(buf_mmap, ...) branch of load_all_data). At the final cleanup it computes the page-aligned
complement of those ranges per file, munmaps each released range first (so a stray read faults rather than reading
the zero pages a hole leaves), then punches it out of the file with fallocate(PUNCH_HOLE | KEEP_SIZE) through a
read-write reopen of /proc/self/fd/<fd>. Pages shared by an aliased tensor and a released neighbour are kept.
Results go to the exported lkai_memfd_release_stats: released bytes, kept bytes, microseconds, errno of the first
failure (0 = none), ranges released, files.
"""

import sys
from pathlib import Path

H_OLD = "    std::vector<std::pair<size_t, size_t>> mmaps_used;\n"
H_NEW = H_OLD + (
    "    // strands-decider memfdrel patch: per file, the [first, last) ranges of tensors aliased into the mapping\n"
    "    std::vector<std::vector<std::pair<size_t, size_t>>> mmaps_alias;\n"
)

INC_OLD = "#include <unordered_set>\n"
INC_NEW = INC_OLD + (
    "\n#ifdef __linux__\n#include <cerrno>\n#include <cstdio>\n#include <cstdlib>\n#include <fcntl.h>\n#include <linux/falloc.h>\n"
    "#include <sys/mman.h>\n#include <unistd.h>\n#endif\n\n"
    "// strands-decider memfdrel patch: {released bytes, kept bytes, microseconds, errno of the first failure,\n"
    "// ranges released, files, ranges released by the madvise(MADV_REMOVE) fallback}. All zero until a load runs\n"
    "// with LKAI_RELEASE_MEMFD=1.\n"
    'extern "C" __attribute__((visibility("default"))) int64_t lkai_memfd_release_stats[7];\n'
    "int64_t lkai_memfd_release_stats[7] = {0, 0, 0, 0, 0, 0, 0};\n"
)

ALIAS_OLD = (
    "                auto & mmap_used = mmaps_used[weight->idx];\n"
    "                mmap_used.first  = std::min(mmap_used.first,  weight->offs);\n"
    "                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);\n"
)
ALIAS_NEW = ALIAS_OLD + (
    "                if (mmaps_alias.size() <= weight->idx) {\n"
    "                    mmaps_alias.resize(weight->idx + 1);\n"
    "                }\n"
    "                mmaps_alias[weight->idx].emplace_back(weight->offs, weight->offs + n_size);\n"
)

REL_OLD = (
    "                mapping->unmap_fragment(0, mmap_used.first);\n"
    "                if (mmap_used.second != 0) {\n"
    "                    mapping->unmap_fragment(mmap_used.second, mapping->size());\n"
    "                }\n"
    "            }\n"
    "        }\n"
)
REL_NEW = (
    REL_OLD
    + r"""#ifdef __linux__
        // strands-decider memfdrel patch: release the file pages that no tensor aliases (e11/patch_memfdrel.py)
        const char * lkai_rel = getenv("LKAI_RELEASE_MEMFD");
        if (use_mmap && lkai_rel && lkai_rel[0] == '1') {
            const int64_t t0 = ggml_time_us();
            const size_t page = (size_t) sysconf(_SC_PAGESIZE);
            int64_t released = 0, kept = 0, err = 0, n_ranges = 0, via_madvise = 0;
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                auto & mapping = mappings.at(idx);
                const size_t size = mapping->size();
                std::vector<std::pair<size_t, size_t>> keep;
                if (idx < mmaps_alias.size()) {
                    keep = mmaps_alias[idx];
                }
                for (auto & r : keep) {  // whole pages around every aliased byte
                    r.first  = r.first & ~(page - 1);
                    r.second = std::min((r.second + page - 1) & ~(page - 1), size);
                }
                std::sort(keep.begin(), keep.end());
                std::vector<std::pair<size_t, size_t>> drop;
                size_t cur = 0;
                for (const auto & r : keep) {
                    if (r.first > cur) {
                        drop.emplace_back(cur, r.first);
                    }
                    cur = std::max(cur, r.second);
                }
                if (cur < size) {
                    drop.emplace_back(cur, (size + page - 1) & ~(page - 1));
                }
                // the descriptor llama reads through is read-only; reopen the same file for writing
                char path[64];
                snprintf(path, sizeof(path), "/proc/self/fd/%d", files.at(idx)->file_id());
                const int wfd = open(path, O_RDWR | O_CLOEXEC);
                if (wfd < 0) {
                    err = err ? err : errno;
                    LLAMA_LOG_WARN("%s: LKAI_RELEASE_MEMFD: open %s: %s\n", __func__, path, strerror(errno));
                    kept += (int64_t) size;
                    continue;
                }
                int64_t file_released = 0;
                for (const auto & r : drop) {
                    mapping->unmap_fragment(r.first, r.second);  // a stray read through llama's view now faults
                    const size_t len = std::min(r.second, size) - r.first;
                    // Punch the hole. Lambda's seccomp filter refuses fallocate (EPERM; e11/syscall_probe.py), so
                    // fall back to madvise(MADV_REMOVE) on a temporary writable shared view, which frees the same
                    // shmem backing pages. The view is unmapped at once; nothing reads through it.
                    bool done = fallocate(wfd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t) r.first,
                                          (off_t) len) == 0;
                    if (!done && (errno == EPERM || errno == ENOSYS || errno == EOPNOTSUPP)) {
                        void * v = mmap(NULL, r.second - r.first, PROT_READ | PROT_WRITE, MAP_SHARED, wfd,
                                        (off_t) r.first);
                        if (v != MAP_FAILED) {
                            done = madvise(v, r.second - r.first, MADV_REMOVE) == 0;
                            if (done) {
                                via_madvise++;
                            }
                            munmap(v, r.second - r.first);
                        }
                    }
                    if (!done) {
                        err = err ? err : errno;
                        LLAMA_LOG_WARN("%s: LKAI_RELEASE_MEMFD: release: %s\n", __func__, strerror(errno));
                        continue;
                    }
                    file_released += (int64_t) len;
                    n_ranges++;
                }
                close(wfd);
                released += file_released;
                kept += (int64_t) size - file_released;
            }
            lkai_memfd_release_stats[0] = released;
            lkai_memfd_release_stats[1] = kept;
            lkai_memfd_release_stats[2] = ggml_time_us() - t0;
            lkai_memfd_release_stats[3] = err;
            lkai_memfd_release_stats[4] = n_ranges;
            lkai_memfd_release_stats[5] = (int64_t) mappings.size();
            lkai_memfd_release_stats[6] = via_madvise;
        }
#endif
"""
)

EDITS = (
    ("src/llama-model-loader.h", H_OLD, H_NEW),
    ("src/llama-model-loader.cpp", INC_OLD, INC_NEW),
    ("src/llama-model-loader.cpp", ALIAS_OLD, ALIAS_NEW),
    ("src/llama-model-loader.cpp", REL_OLD, REL_NEW),
)


def apply(root: Path) -> None:
    texts: dict[str, str] = {}
    for rel, old, new in EDITS:
        s = texts.get(rel) or (root / rel).read_text()
        if "memfdrel patch" in s and rel not in texts:
            raise SystemExit(f"{rel}: already carries the memfdrel patch")
        if s.count(old) != 1:
            raise SystemExit(f"{rel}: anchor found {s.count(old)} times, want 1:\n{old}")
        texts[rel] = s.replace(old, new)
    for rel, s in texts.items():
        (root / rel).write_text(s)


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
    print("patched", sys.argv[1], "memfdrel")
