#!/bin/bash
# Runs INSIDE P1's pinned Lambda python3.12 image (arm64), started by graviton_profile.sh. OFFLINE, NOT LAMBDA.
#   /pkg  the deployed function.zip, unzipped (ro)    /w/model.gguf  the deployed GGUF (ro)
#   /p    examples/lambda-kleidiai/profiling (ro)     /out           results (rw)
set -euo pipefail
dnf install -y perf gcc util-linux >/dev/null 2>&1 || dnf install -y perf gcc util-linux
gcc -O2 -g -shared -fPIC -o /tmp/lkaisamp.so /p/lkaisamp.c
export PYTHONPATH=/pkg LD_LIBRARY_PATH=/pkg/lib OPENBLAS_NUM_THREADS=1
# Lambda's 10,240 MB function has 6 vCPUs: pin the engine to CPUs 0-5 (docker --cpus needs a cgroup v2 domain the
# DevSpace's docker cannot enter).
DRIVE="taskset -c 0-5 python3.12 /p/offline_drive.py --pkg /pkg --gguf /w/model.gguf --threads 6 --requests /w/requests.json"
L="$HOST_LABEL (offline, Lambda image, 6 CPUs)"
{ uname -a; grep -m1 'CPU part' /proc/cpuinfo; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; perf --version;
  python3.12 --version; cat /proc/sys/kernel/perf_event_paranoid; } > /out/host.txt 2>&1

echo ">> plain"
$DRIVE --passes 10 --host-label "$L" --out /out/plain.json >/dev/null 2>/out/plain.stderr
echo ">> plain (no memfd: mmap the file directly, page cache warm)"
$DRIVE --passes 3 --no-memfd --host-label "$L no-memfd" --out /out/plain-nomemfd.json >/dev/null 2>/out/plain-nomemfd.stderr
echo ">> perf stat"
perf stat -e task-clock,cycles,instructions,cache-misses,page-faults,major-faults,minor-faults,context-switches \
  -x, -o /out/perf-stat.csv -- $DRIVE --passes 10 --host-label "$L perf-stat" --out /out/perf-stat.json \
  >/dev/null 2>/out/perf-stat.stderr || echo "perf stat rc=$?" >> /out/perf-stat.stderr
echo ">> perf record (999 Hz, frame-pointer call graph)"
perf record -F 999 -g -o /out/perf.data -- $DRIVE --passes 10 --host-label "$L perf-record" \
  --out /out/perf-record.json >/dev/null 2>/out/perf-record.stderr || echo "perf record rc=$?" >> /out/perf-record.stderr
perf report -i /out/perf.data --no-children --sort dso,sym --stdio --percent-limit 0.3 2>/dev/null \
  | head -150 > /out/perf-report-self.txt || true
perf report -i /out/perf.data --children --sort sym --stdio --percent-limit 1 -g none 2>/dev/null \
  | head -120 > /out/perf-report-children.txt || true
# Per-phase split: the PHASE lines carry CLOCK_MONOTONIC times; perf script's time is the same clock (perf_clock).
perf script -i /out/perf.data -F time,tid,ip,sym,dso --ns 2>/dev/null | gzip > /out/perf-script.txt.gz || true
echo ">> lkaisamp (1 kHz CPU time)"
$DRIVE --passes 10 --sampler /tmp/lkaisamp.so --samples /out/sampler.samp --host-label "$L sampler" \
  --out /out/sampler.json >/dev/null 2>/out/sampler.stderr
echo ">> cProfile"
$DRIVE --passes 10 --cprofile /out/cprofile.prof --host-label "$L cprofile" --out /out/cprofile.json \
  >/dev/null 2>/out/cprofile.stderr
echo ">> py-spy --native"
/w/py-spy record --rate 100 --native --threads --format raw -o /out/pyspy-native.txt -- $DRIVE --passes 3 \
  --host-label "$L py-spy" --out /out/pyspy.json >/dev/null 2>/out/pyspy.stderr || echo "py-spy rc=$?" >> /out/pyspy.stderr
chmod -R a+rX /out
echo GRAVITON_CONTAINER_DONE
