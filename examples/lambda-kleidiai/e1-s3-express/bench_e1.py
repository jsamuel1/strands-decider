"""E1's driver: P1's baseline/bench.py with one client-side change, a new Lambda client for every wave.

  python bench_e1.py <the same arguments as baseline/bench.py>

Why: bench.py keeps one pooled HTTPS client for the whole run. A wave that follows a long idle gap (the ~155 s
publish_version and snapshot wait before each restore wave) can reuse a pooled connection the network path has
silently dropped. The request is then written into a dead socket and waits for the 900 s read timeout. Lambda never
sees it: the first E1 attempt (20261020T1130Z-e1) lost 9 of 10 gp restore invokes this way, with no START/REPORT line
and an Invocations metric of 1 for that wave. A fresh client per wave opens new connections, so nothing in the
function, its configuration or the measured REPORT/LKAI numbers changes. Only the client round trip (rt_s, which
no headline uses) gains one TLS handshake, identically on every arm.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
import bench  # noqa: E402
from botocore.config import Config  # noqa: E402


def lambda_client(sess, concurrency: int):
    """bench.Bench's client configuration (read timeout 900 s, no retries, a pool per wave)."""
    return sess.client(
        "lambda",
        config=Config(
            read_timeout=900,
            connect_timeout=10,
            retries={"max_attempts": 0},
            max_pool_connections=max(16, concurrency * 2),
        ),
    )


class FreshClientBench(bench.Bench):
    def __init__(self, sess, st, reqs, out, concurrency):
        super().__init__(sess, st, reqs, out, concurrency)
        self._sess = sess
        self.clients_made = 0

    def wave(self, *args, **kwargs):
        self.lam = lambda_client(self._sess, self.concurrency)
        self.clients_made += 1
        return super().wave(*args, **kwargs)


def main() -> None:
    bench.Bench = FreshClientBench
    bench.main()


if __name__ == "__main__":
    main()
