"""baseline/bench.py for E4, with one harness change: a fresh Lambda client for every wave.

  python bench_e4.py <bench.py's arguments>

bench.py keeps one client (pooled HTTPS connections, no retries) for the whole run. With three arms, each round
publishes three versions (about 155 s each) between the cold and restore waves, so a pooled connection sits idle
for about 470 s; the network path drops idle connections well before that, and the next invoke on one fails at
once with ConnectionClosedError (E4's first run: 9 of 30 restore invokes in round 0; E5's two-arm run, idle about
310 s, had none). A new client per wave opens new connections, so every wave starts from the same state. The
invocations, their order and timing, and everything recorded are bench.py's.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import bench  # noqa: E402

_init, _wave = bench.Bench.__init__, bench.Bench.wave


def _init_keep(self, sess, *a, **k):
    _init(self, sess, *a, **k)
    self._sess, self._cfg = sess, self.lam.meta.config


def _fresh_wave(self, *a, **k):
    self.lam = self._sess.client("lambda", config=self._cfg)
    return _wave(self, *a, **k)


bench.Bench.__init__, bench.Bench.wave = _init_keep, _fresh_wave

if __name__ == "__main__":
    bench.main()
