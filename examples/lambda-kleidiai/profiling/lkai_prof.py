"""Profiler arms for the lambda-kleidiai baseline: handler "lkai_prof.handler", wrapping baseline/lkai_entry.py.

One function per arm, all from one zip, all identical to P1's functions except LKAI_PROF and the handler. LKAI_PROF
picks the profiler; its cost is measured against the REPORT-only arm (P1's lkai-p1-rep configuration, LKAI_PHASES=0):

  none      lkai_entry.handler with LKAI_PHASES as configured. With LKAI_PHASES=0 this is the overhead control.
  cprofile  cProfile around the whole init (import of lkai_entry: weight download, load, warm-up) and around each
            invocation's handler. Stats are written to /tmp and summarised to stdout (top 40 by cumulative time).
  pyspy     py-spy (shipped in bin/) records this process from a child at 100 Hz, Python frames only
            (--nonblocking; --native needs ptrace, which Lambda denies), from module import until the load is done,
            then on every invocation for its duration. Lambda lets a child read its parent with process_vm_readv
            (lkai_probe.py shows `py-spy dump` of the parent works), so non-blocking sampling works.
  sampler   lkaisamp.so (setitimer SIGPROF, 1 kHz CPU time): native PC + LR samples tagged by phase, symbolised in
            the function from /proc/self/maps and the ELF symbol tables of the mapped libraries.
  xray      no in-function change: the function's TracingConfig is Active, so Lambda emits its own segments
            (Initialization, Invocation, Overhead, and Restore). Nothing in the zip calls the X-Ray SDK.
  telemetry the Lambda Telemetry API extension in extensions/ (an external extension, a separate process) subscribes to
            platform events and logs them as `LKAI_TEL {json}` lines. Here LKAI_PROF only labels the arm; the
            extension is what Lambda starts.

Each profiler arm keeps LKAI_PHASES=1 so the phase records show where the profiler's time went; the overhead it adds
is still read from Lambda's own REPORT numbers, against the REPORT-only arm, as P1 measured its instrumentation.
Profiles are summarised as `LKAI_PROF {json}` stdout lines (CloudWatch keeps them), never as response fields beyond a
small `lkai_prof` stub, so the answers stay byte-identical to the baseline's.
"""

from __future__ import annotations

import os
import time

T0_NS = time.monotonic_ns()
MODE = os.environ.get("LKAI_PROF", "none")
HERE = os.path.dirname(os.path.abspath(__file__))

import json  # noqa: E402
import sys  # noqa: E402

_state: dict = {"n": 0, "init_emitted": False}


def _emit(kind: str, **fields) -> None:
    rec = {"schema": "lkai-prof/1", "kind": kind, "mode": MODE, "wall_ns": time.time_ns(), **fields}
    sys.stdout.write("LKAI_PROF " + json.dumps(rec, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ cProfile
if MODE == "cprofile":
    import cProfile
    import pstats

    _PROF = cProfile.Profile()

    def _cp_summary(prof: cProfile.Profile, top: int = 40) -> list[dict]:
        """The top functions by cumulative time and the top by own time (tottime), deduplicated."""
        st = pstats.Stats(prof)
        rows = []
        for (file, line, fn), (_cc, nc, tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
            rows.append({"fn": f"{os.path.basename(file)}:{line}:{fn}", "ncalls": nc, "tottime": round(tt, 6),
                         "cumtime": round(ct, 6)})
        keep = {id(r): r for r in sorted(rows, key=lambda r: -r["cumtime"])[:top]}
        keep.update({id(r): r for r in sorted(rows, key=lambda r: -r["tottime"])[:top]})
        return sorted(keep.values(), key=lambda r: -r["cumtime"])

    _PROF.enable()
    # Python 3.12's cProfile runs on sys.monitoring, which is process-wide: this one profile also sees PR #77's
    # background load thread and the 32 download threads. Only one may be active at a time (a second enable raises
    # "Another profiling tool is already active"), so the init profile is disabled before any invocation's starts.

# ------------------------------------------------------------------------------------------------ native sampler
_SAMP = None
if MODE == "sampler":
    import ctypes

    _SAMP = ctypes.CDLL(os.path.join(HERE, "lib", "liblkaisamp.so"))
    _SAMP.samp_start(int(os.environ.get("LKAI_SAMP_US", "1000")), 1 << 21)
    _SAMP.samp_phase(1)  # init

# ------------------------------------------------------------------------------------------------ py-spy
_SPY: dict = {}


def _spy_start(tag: str, duration_s: int) -> None:
    import subprocess

    out = f"/tmp/pyspy-{tag}.txt"
    t = time.monotonic()
    # Popen only: py-spy prints no "ready" line when stderr is not a terminal, so there is nothing to wait for. Its
    # attach time (process_vm_readv of the interpreter state) is lost from the profile, not added to the invocation.
    proc = subprocess.Popen(
        [os.path.join(HERE, "bin", "py-spy"), "record", "--pid", str(os.getpid()), "--rate", "100",
         "--threads", "--nonblocking", "--format", "raw", "--duration", str(duration_s), "-o", out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _SPY[tag] = (proc, out, t, b"", time.monotonic() - t)


def _spy_stop(tag: str, top: int = 40) -> dict:
    import signal
    import subprocess

    proc, out, t, head, attach_s = _SPY.pop(tag)
    proc.send_signal(signal.SIGINT)  # py-spy writes what it has on SIGINT
    try:
        _, err = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    res: dict = {"rc": proc.returncode, "stderr": (head + err).decode(errors="replace")[-600:],
                 "popen_s": round(attach_s, 4), "wall_s": round(time.monotonic() - t, 4)}
    try:
        with open(out, encoding="utf-8", errors="replace") as f:  # SIGINT can cut py-spy mid-character
            res.update(_collapse(f.read(), top))
    except OSError as e:
        res["error"] = str(e)
    return res


def _collapse(raw: str, top: int) -> dict:
    """py-spy raw (collapsed stacks 'a;b;c N'): total samples, top leaf frames and top inclusive frames."""
    leaf: dict[str, int] = {}
    incl: dict[str, int] = {}
    total = 0
    for line in raw.splitlines():
        stack, _, n = line.rpartition(" ")
        if not stack or not n.isdigit():
            continue
        c = int(n)
        total += c
        frames = stack.split(";")
        leaf[frames[-1]] = leaf.get(frames[-1], 0) + c
        for f in set(frames):
            incl[f] = incl.get(f, 0) + c
    srt = lambda d: [{"frame": k[:200], "samples": v} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]]  # noqa: E731
    return {"samples": total, "top_leaf": srt(leaf), "top_inclusive": srt(incl)}


if MODE == "pyspy":
    # Not --native: native unwinding suspends the target with ptrace, which Lambda refuses (EPERM, lkai_probe.py).
    _spy_start("init", 300)

# ------------------------------------------------------------------------------------------------ the baseline
_t_imp = time.monotonic_ns()
import lkai_entry  # noqa: E402  (runs the whole baseline init: download, load, warm-up; or at publish for SnapStart)

_init_s = (time.monotonic_ns() - _t_imp) / 1e9


def _init_profile() -> None:
    """Emit the init profile once, at the first invocation (on-demand: the load finishes in a thread by then). A
    profiler fault is logged and swallowed: it must never fail the request it is profiling."""
    if _state["init_emitted"]:
        return
    _state["init_emitted"] = True
    _safe(_init_profile_emit, "init")


def _safe(fn, where: str = "invoke") -> None:
    """Run a profiler hook; log its exception as an LKAI_PROF profiler_error record instead of raising it. A profiler
    fault never fails, or replaces the outcome of, the invocation it profiles (in run 20261020T124451Z one cold pyspy
    request failed on a py-spy output cut mid-character before this guard existed)."""
    try:
        fn()
    except Exception as e:
        try:
            _emit("profiler_error", where=where, error=f"{type(e).__name__}: {e}"[:300])
        except Exception:
            pass


def _init_profile_emit() -> None:
    if MODE == "cprofile":
        _PROF.disable()
        _emit("init", import_s=_init_s, top={"all_threads": _cp_summary(_PROF)})
    elif MODE == "pyspy" and "init" in _SPY:
        _emit("init", import_s=_init_s, pyspy=_spy_stop("init"))
    elif MODE == "sampler":
        import lkai_symbolise

        _SAMP.samp_phase(0)
        _emit("init", import_s=_init_s, cpu_s=round(time.process_time(), 4), sampler=lkai_symbolise.summary(_SAMP))
        _SAMP.samp_reset()


def _before_snapshot():
    # SnapStart: the init profile is taken before the snapshot, and the (ITIMER_PROF) sampler is disarmed so a
    # restored copy does not inherit a running timer pointed at a stale buffer state. Measured cost to know about
    # (run 20261020T124451Z): the sampler arm's first invocation after a restore emits its summary ~2.0 s after the
    # baseline lkai_entry.handler returns, still inside this module's handler and so before Lambda gets the response.
    # REPORT Duration grows by ~2.1 s while latency_ms does not. Not yet attributed; the candidate
    # is symbolisation re-reading the libraries from /var/task straight after restore (lkai_symbolise now keeps parsed
    # ELF tables across a maps change, which would remove it if so; that is not yet re-measured in Lambda).
    _init_profile()
    if _SAMP is not None:
        _SAMP.samp_stop()


def _after_restore():
    if _SAMP is not None:
        _SAMP.samp_reset()
        _SAMP.samp_start(int(os.environ.get("LKAI_SAMP_US", "1000")), 1 << 21)


try:
    from snapshot_restore_py import register_after_restore, register_before_snapshot

    # Registered after lkai_entry's own hooks, so they run after its before_snapshot record (LIFO order is not
    # documented for Python; the profile summary does not depend on it).
    register_before_snapshot(_before_snapshot)
    register_after_restore(_after_restore)
except ImportError:
    pass


def handler(event, context):
    _state["n"] += 1
    n = _state["n"]
    if isinstance(event, dict) and event.get("mode") == "lkai_probe":
        import lkai_probe

        return lkai_probe.handler(event, context)
    if n == 1:
        # On demand the load runs in PR #77's background thread. lkai_entry.handler waits for it anyway; waiting here
        # first lets the init profile cover the whole load and keeps it out of the first invocation's profile.
        import handler as _pr77

        _pr77._ENGINE.result()  # noqa: SLF001 (PR #77's load future; lkai_entry reads it the same way)
    _init_profile()
    if MODE == "cprofile":
        prof = cProfile.Profile()
        prof.enable()
        try:
            out = lkai_entry.handler(event, context)
        finally:
            prof.disable()
            _safe(lambda: _emit("invoke", n=n, top=_cp_summary(prof, 25)))
    elif MODE == "pyspy":
        _safe(lambda: _spy_start("inv", 60))
        try:
            out = lkai_entry.handler(event, context)
        finally:
            _safe(lambda: _emit("invoke", n=n, pyspy=_spy_stop("inv", 25)) if "inv" in _SPY else None)
    elif MODE == "sampler":
        import lkai_symbolise

        _SAMP.samp_phase(7)
        c0 = time.process_time()
        try:
            out = lkai_entry.handler(event, context)
        finally:
            _SAMP.samp_phase(0)
            # cpu_s (process CPU time, all threads) beside the sample count gives the effective sampling rate:
            # ITIMER_PROF signals coalesce, one pending per process, so 6 busy threads yield far fewer than 1 kHz each.
            cpu_s = round(time.process_time() - c0, 4)
            _safe(lambda: _emit("invoke", n=n, cpu_s=cpu_s, sampler=lkai_symbolise.summary(_SAMP)))
            _SAMP.samp_reset()
    else:
        out = lkai_entry.handler(event, context)
    if isinstance(out, dict) and "statusCode" not in out:
        out["lkai_prof"] = {"mode": MODE, "n": n}
    return out
