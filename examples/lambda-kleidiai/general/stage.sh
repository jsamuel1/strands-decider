#!/bin/bash
# Stage every general-ladder arm on a fixed host: verify each published artifact's sha256 (inputs.json), unzip each
# package once (read-only mounts at /var/task), and build E6's bytecode package from P1's zip with E6's own
# pyc_zip.py at lkai/fada-hotspot e989e39, inside P1's pinned Lambda python3.12 image.
#
#   general/stage.sh <artifact dir> <stage dir>
#
# <artifact dir> holds p1.zip, p1-manifest.json (P1's published build manifest), p1.gguf, and on arm64 also e8-off.zip,
# e9.zip, e9.gguf, e7.zip, e5.gguf. The repository must have the rung commits (git fetch the fork's lkai/* branches).
# Writes <stage dir>/pkg-<name>/ (unzipped, world-readable) and <stage dir>/staged.json (hashes as verified).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
A=$(cd "$1" && pwd)
mkdir -p "$2" && S=$(cd "$2" && pwd)
IMG=$(sed -n 's/^LAMBDA_IMAGE=//p' "$REPO/examples/lambda-kleidiai/baseline/build.sh")
ARCH=$(uname -m)
want() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['artifacts'][sys.argv[2]]['sha256'])" "$HERE/inputs.json" "$1"; }
check() {  # check <file> <inputs key>
  local got
  got=$(sha256sum "$1" | cut -c1-64)
  [ "$got" = "$(want "$2")" ] || { echo "FATAL: $1 is $got, not the published $2 $(want "$2")"; exit 1; }
  echo "  ok $2 ${got:0:16}"
}
unz() {  # unz <zip> <name>
  rm -rf "$S/pkg-$2" && mkdir -p "$S/pkg-$2"
  unzip -q "$1" -d "$S/pkg-$2"   # unzip, not zipfile: P1's zip stores lib/*.so links as symlinks
  chmod -R a+rX "$S/pkg-$2"
}
echo ">> verify artifacts ($ARCH)"
check "$A/p1.zip" p1_zip
check "$A/p1.gguf" p1_gguf
[ "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['function_zip']['sha256'])" "$A/p1-manifest.json")" \
  = "$(want p1_zip)" ] || { echo "FATAL: p1-manifest.json does not describe p1.zip"; exit 1; }
if [ "$ARCH" = aarch64 ]; then
  check "$A/e8-off.zip" e8_off_zip
  check "$A/e9.zip" e9_zip
  check "$A/e9.gguf" e9_gguf
  check "$A/e7.zip" e7_zip
  check "$A/e5.gguf" e5_gguf
  unz "$A/p1.zip" p1
  unz "$A/e8-off.zip" e8off
  unz "$A/e9.zip" e9
  unz "$A/e7.zip" e7
  echo ">> E6 bytecode package from p1.zip (pyc_zip.py at e989e39, CPython 3.12 in P1's image)"
  rm -rf "$S/e6src" "$S/e6build" && mkdir -p "$S/e6src" "$S/e6build"
  git -C "$REPO" show e989e39:examples/lambda-kleidiai/e6/pyc_zip.py > "$S/e6src/pyc_zip.py"
  git -C "$REPO" show e989e39:examples/lambda-kleidiai/e6/e6_entry.py > "$S/e6src/e6_entry.py"
  # pyc_zip.py stamps its manifest with `git rev-parse HEAD` of its own directory; the image has no git and /e6 is no
  # checkout, so a stub answers with the commit the two files were read from (and a clean status).
  mkdir -p "$S/e6src/bin"
  printf '#!/bin/sh\ncase "$*" in *rev-parse*) echo %s;; esac\n' "$(git -C "$REPO" rev-parse e989e39)" > "$S/e6src/bin/git"
  chmod +x "$S/e6src/bin/git"
  docker run --rm --platform linux/arm64 --network none --entrypoint python3.12 -e PATH=/e6/bin:/var/lang/bin:/usr/local/bin:/usr/bin:/bin \
    -v "$S/e6src:/e6:ro" -v "$(readlink -f "$A/p1.zip"):/a/p1.zip:ro" \
    -v "$(readlink -f "$A/p1-manifest.json"):/a/p1-manifest.json:ro" -v "$S/e6build:/out" "$IMG" \
    /e6/pyc_zip.py --p1-zip /a/p1.zip --p1-manifest /a/p1-manifest.json --out /out > "$S/e6build/stdout.json"
  unz "$S/e6build/pyc.zip" e6pyc
fi
python3 - "$S" "$A" "$ARCH" <<'EOF'
import hashlib, json, os, sys
s, a, arch = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()
files = {f: sha(os.path.join(a, f)) for f in sorted(os.listdir(a)) if f.endswith((".zip", ".gguf", ".json"))}
e6 = os.path.join(s, "e6build", "pyc.zip")
if os.path.exists(e6):
    files["e6build/pyc.zip"] = sha(e6)
    files["e6build/manifest.json"] = sha(os.path.join(s, "e6build", "manifest.json"))
json.dump({"arch": arch, "artifacts_sha256": files,
           "pkgs": sorted(d[4:] for d in os.listdir(s) if d.startswith("pkg-"))},
          open(os.path.join(s, "staged.json"), "w"), indent=1)
print("staged", arch, len(files), "files")
EOF
echo STAGE_DONE
