#!/bin/bash
# Runs INSIDE P1's pinned Lambda python3.12 image (linux/arm64), called by fixed_host.sh with:
#   /pkg   the deployed function.zip, unzipped (ro)     /t   examples/lambda-kleidiai/tokenizer (ro)
#   /req   baseline/requests.json (ro)                    /out results (rw)
# Every result says FIXED HOST, NOT LAMBDA. PHASE selects: golden | bench | profile.
set -euo pipefail
export PYTHONPATH=/pkg PYTHONDONTWRITEBYTECODE=1
TOK=/pkg/assets/tokenizer.json
PY="taskset -c ${T0_CPUS:-0-5} python3.12"
cd /t
# taskset (util-linux) is not in the Lambda image; perf only for the profile phase.
command -v taskset >/dev/null || dnf install -y util-linux >/dev/null 2>&1 || dnf install -y util-linux
case "$PHASE" in
golden)
  python3.12 tokcorpus.py --tokenizer $TOK --requests /req/requests.json --out /out/corpus.jsonl
  python3.12 tokparity.py make --corpus /out/corpus.jsonl --tokenizer $TOK --out /out/golden.jsonl.gz
  python3.12 tokparity.py check --corpus /out/corpus.jsonl --tokenizer $TOK --golden /out/golden.jsonl.gz \
    > /out/parity-check.json
  ;;
bench)
  $PY tokbench.py --tokenizer $TOK --corpus /t/data/corpus.jsonl --out "/out/bench-$RUN.json" --label "$HOST_LABEL $RUN" \
    --rounds "${ROUNDS:-5}" --init-reps "${INIT_REPS:-10}" --iters "${ITERS:-200}" --warmup 20 > "/out/bench-$RUN.txt"
  ;;
profile)
  dnf install -y perf >/dev/null 2>&1 || dnf install -y perf
  perf --version > /out/perf-version.txt
  # The profiled loop: steady-state fit over the representative cases, plus init, many times, so samples are
  # plenty. Its timings are never reported as latency (profiling artifacts only).
  LOOP="$PY toksoak.py --tokenizer $TOK --corpus /t/data/corpus.jsonl"
  echo ">> plain (overhead control)"
  $LOOP --seconds 20 --out /out/soak-plain.json
  echo ">> perf stat: steady state"
  perf stat -x, -e task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,stalled-cycles-backend,stalled-cycles-frontend,page-faults,minor-faults,major-faults,context-switches,cpu-migrations \
    -o /out/perf-stat-steady.csv -- $LOOP --seconds 20 --out /out/soak-perfstat.json
  echo ">> perf stat: init (from_file x30)"
  perf stat -x, -e task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,stalled-cycles-backend,stalled-cycles-frontend,page-faults,minor-faults,major-faults \
    -o /out/perf-stat-init.csv -- $LOOP --init-only 30 --out /out/soak-init.json
  echo ">> perf record: steady state (999 Hz, dwarf call graph)"
  perf record -F 999 --call-graph dwarf,16384 -o /out/perf-steady.data -- $LOOP --seconds 15 --out /out/soak-rec.json \
    2> /out/perf-record-steady.stderr
  perf report -i /out/perf-steady.data --no-children --sort dso,symbol --stdio --percent-limit 0.3 \
    > /out/perf-report-steady.txt 2>/dev/null
  perf report -i /out/perf-steady.data --no-children --sort dso --stdio > /out/perf-report-steady-dso.txt 2>/dev/null
  perf script -i /out/perf-steady.data -F comm,tid,ip,sym,dso 2>/dev/null | python3.12 /t/stackcollapse.py \
    > /out/perf-steady.folded
  echo ">> perf record: init"
  perf record -F 999 --call-graph dwarf,16384 -o /out/perf-init.data -- $LOOP --init-only 20 --out /out/soak-rec-init.json \
    2> /out/perf-record-init.stderr
  perf report -i /out/perf-init.data --no-children --sort dso,symbol --stdio --percent-limit 0.3 \
    > /out/perf-report-init.txt 2>/dev/null
  perf script -i /out/perf-init.data -F comm,tid,ip,sym,dso 2>/dev/null | python3.12 /t/stackcollapse.py \
    > /out/perf-init.folded
  rm -f /out/perf-steady.data /out/perf-init.data
  ;;
*) echo "PHASE must be golden, bench or profile"; exit 2 ;;
esac
echo "TOK_CONTAINER_DONE $PHASE"
