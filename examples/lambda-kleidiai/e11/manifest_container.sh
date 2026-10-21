#!/bin/bash
# Runs INSIDE the pinned Lambda python3.12 image as root, with the repo checkout at /repo and P1's build manifest
# directory at /p1b. Installs git (manifest_e11.py's dirty check needs it), then writes e11/build/manifest.json as
# the uid/gid in $RUN_AS so the file belongs to the host user.
set -euo pipefail
dnf install -y git util-linux >/dev/null 2>&1
cd /repo
exec setpriv --reuid="${RUN_AS%:*}" --regid="${RUN_AS#*:}" --clear-groups env HOME=/tmp \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' \
  python3.12 examples/lambda-kleidiai/e11/manifest_e11.py --p1-build /p1b --build examples/lambda-kleidiai/e11/build
