"""Fetch the X-Ray traces of the xray arm for a bench run window and summarise Lambda's own segments.

  python fetch_xray.py <run-dir> [--function lkai-p3-xray] [--region us-east-2]

Writes <run-dir>/xray.jsonl (one trace per line: its id and the segment documents, as GetTraces returns them) and
<run-dir>/xray-summary.json: per subsegment name (Initialization, Restore, Invocation, Overhead, ...) a distribution of
durations, and the trace count. X-Ray samples the first request each second and 5% of the rest by default, so `traces`
is a sample of the invocations, not all of them. The trace ids are read from Lambda's `XRAY TraceId:` log lines.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline"))
import summarize as s1


def _walk(doc: dict, out: dict, prefix: str = "") -> None:
    name = doc.get("name", "?")
    if "start_time" in doc and "end_time" in doc:
        out[prefix + name].append(doc["end_time"] - doc["start_time"])
    for sub in doc.get("subsegments", []) or []:
        _walk(sub, out, prefix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--function", default="lkai-p3-xray")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    a = ap.parse_args()
    cond = json.load(open(os.path.join(a.run, "conditions.json")))
    win = [m["log_window"] for m in cond.get("rounds_meta", []) if "log_window" in m]
    t0, t1 = (min(w[0] for w in win), max(w[1] for w in win)) if win else (None, None)
    xr = boto3.Session(profile_name=a.profile, region_name=a.region).client("xray")
    # Trace ids from Lambda's own `XRAY TraceId: ... Sampled: true` log lines in the run's saved logs. Here
    # GetTraceSummaries returned nothing for traces BatchGetTraces resolves, so the logs are the index.
    ids = []
    lg = os.path.join(a.run, "logs", f"{a.function}.jsonl.gz")
    with gzip.open(lg, "rt") as f:
        for line in f:
            m = re.search(r"XRAY TraceId: (\S+)\s+SegmentId: \S+\s+Sampled: true", json.loads(line)["message"])
            if m:
                ids.append(m.group(1))
    ids = sorted(set(ids))
    durs: dict[str, list[float]] = defaultdict(list)
    origins: dict[str, int] = defaultdict(int)
    with open(os.path.join(a.run, "xray.jsonl"), "w") as fh:
        for i in range(0, len(ids), 5):
            for t in xr.batch_get_traces(TraceIds=ids[i:i + 5]).get("Traces", []):
                segs = [json.loads(s["Document"]) for s in t.get("Segments", [])]
                fh.write(json.dumps({"id": t["Id"], "segments": segs}) + "\n")
                for d in segs:
                    origins[d.get("origin", "?")] += 1
                    _walk(d, durs, f"{d.get('origin', '?')}/")
    res = {"schema": "lkai-xray/1", "function": a.function, "window": [t0, t1], "traces": len(ids),
           "segments_by_origin": dict(origins),
           "durations_s": {k: s1.dist(v) for k, v in sorted(durs.items())},
           "note": "X-Ray samples requests (default rule: 1/s reservoir + 5%), so `traces` is a sample"}
    json.dump(res, open(os.path.join(a.run, "xray-summary.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("traces", "segments_by_origin")}), sorted(res["durations_s"]))


if __name__ == "__main__":
    main()
