#!/usr/bin/env python3
"""Publish one lambda-kleidiai experiment run to the results store and rebuild the dashboard.

    python examples/lambda-kleidiai/results/publish_result.py <run>/results.json [extra artifacts...]

Steps, all idempotent:
  1. validate the run file against results/schema.json (and the semantic checks in lkai_results.validate);
  2. upload the artifacts it lists (artifacts{}, profiling_artifacts[].path) plus any extra paths, and then its
     result.json, under runs/<experiment_id>/<run_id>/ in the store. Nothing outside that prefix is written;
  3. regenerate a stable dashboard snapshot from every result.json in the store and write it to --out;
  4. upload --out/dashboard.html to Artifactory through its installed MCP server. The compare-and-swap SHA is read
     BEFORE step 2, so a page can only replace the version whose run set it already contains. The new version is
     then re-read by number and its bytes are checked against the generated page.

Ordering argument: runs are append-only, and a page is rendered from a run set loaded after its CAS SHA was read.
The version that SHA names was rendered from a run set loaded before it was uploaded, so before we read it. A CAS
win therefore never removes a run from the dashboard; a stale publisher loses the CAS (exit 4) instead.

The store is --store or $LKAI_RESULTS_STORE: s3://bucket/prefix, or a local directory for a dry run. Generated
pages are local build output; only the single-file dashboard is published to the durable Artifactory URL.
Exit codes: 0 published, 2 invalid run file, 3 run conflict/artifact refusal, 4 dashboard upload refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkai_results as lr

# The durable dashboard: Harmony Artifactory artifact (stable /view/ URL across versions), in cluster
# https://artifactory.amazon.dev/clusters/907e0057-b2bb-46c9-abd4-6e934dbf63d4 (lambda-kleidiai experiments).
DASHBOARD_ARTIFACT_ID = os.environ.get(
    "LKAI_DASHBOARD_ARTIFACT", "aece9ccf-c1d5-4217-b31e-e5b3c78ee759"
)
DASHBOARD_URL = os.environ.get(
    "LKAI_DASHBOARD_URL", "https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results"
)
ARTIFACTORY_MCP = os.environ.get("LKAI_ARTIFACTORY_MCP", "artifactory-mcp")
READY_POLLS, READY_POLL_S = (
    30,
    2.0,
)  # waitForReady gives up after 10 s; give the indexer a minute more


class DashboardPublishError(RuntimeError):
    pass


def _rpc_read(proc: subprocess.Popen, request_id: int, timeout_s: float = 60) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], max(0, deadline - time.monotonic()))
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") != request_id:
            continue
        if "error" in message:
            raise DashboardPublishError(f"Artifactory MCP RPC failed: {message['error']}")
        return message.get("result") or {}
    raise DashboardPublishError(f"Artifactory MCP request {request_id} timed out or closed")


def _rpc_send(proc: subprocess.Popen, request_id: int, method: str, params: dict) -> dict:
    proc.stdin.write(
        json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            separators=(",", ":"),
        )
        + "\n"
    )
    proc.stdin.flush()
    return _rpc_read(proc, request_id)


def _tool_payload(result: dict, tool: str) -> dict:
    if result.get("isError"):
        text = " ".join(
            item.get("text", "") for item in result.get("content", []) if isinstance(item, dict)
        )
        raise DashboardPublishError(f"Artifactory {tool} failed: {text or result}")
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            try:
                payload = json.loads(item.get("text", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    raise DashboardPublishError(f"Artifactory {tool} returned no JSON payload")


_HEAD_OPEN = re.compile(rb"<head\b[^>]*>", re.I)
# What artifactory-mcp may add on upload: its viewer bridges, spliced in right after <head> (comments and <script>
# blocks, including the JSON data-source map). Everything we generated must survive byte for byte.
_PLATFORM_BLOCK = re.compile(rb"<script\b[^>]*>.*?</script\s*>|<!--.*?-->", re.I | re.S)


def platform_splice_only(generated: bytes, hosted: bytes) -> bool:
    """True when `hosted` is `generated` byte for byte, apart from platform blocks inserted after <head>."""
    if hosted == generated:
        return True
    m = _HEAD_OPEN.search(generated)
    if not m:
        return False
    at, tail = m.end(), generated[m.end() :]
    if len(hosted) <= len(generated) or not (
        hosted.startswith(generated[:at]) and hosted.endswith(tail)
    ):
        return False
    inserted = hosted[at : len(hosted) - len(tail)]
    return _PLATFORM_BLOCK.sub(b"", inserted).strip() == b""


def _start_mcp(download_dir: str | None = None) -> subprocess.Popen:
    binary = shutil.which(ARTIFACTORY_MCP)
    if not binary:
        raise DashboardPublishError(
            f"{ARTIFACTORY_MCP!r} not found; install/configure Artifactory MCP or use "
            "--skip-dashboard-upload for a local-only preview"
        )
    env = dict(os.environ)
    if download_dir:
        env["ARTIFACTORY_DOWNLOAD_DIR"] = download_dir  # fresh readback, not a shared cache
    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        _rpc_send(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lkai-publish-result", "version": "1"},
            },
        )
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            + "\n"
        )
        proc.stdin.flush()
        return proc
    except BaseException:
        _stop_mcp(proc)
        raise


def _stop_mcp(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _call(proc: subprocess.Popen, request_id: int, tool: str, arguments: dict) -> dict:
    return _tool_payload(
        _rpc_send(proc, request_id, "tools/call", {"name": tool, "arguments": arguments}), tool
    )


def _get_artifact(
    proc: subprocess.Popen, artifact_id: str, request_id: int, version: int | None = None
) -> dict:
    args: dict = {"artifactId": artifact_id}
    if version is not None:
        args["version"] = version
    return _call(proc, request_id, "get_artifact", args)


def get_dashboard_sha(artifact_id: str = DASHBOARD_ARTIFACT_ID) -> str:
    """Read the dashboard generation before any run or derived-page writes begin."""
    proc = _start_mcp()
    try:
        artifact = _get_artifact(proc, artifact_id, 2).get("artifact") or {}
        sha = artifact.get("sha256")
        if not sha:
            raise DashboardPublishError("Artifactory get_artifact returned no current sha256")
        return sha
    finally:
        _stop_mcp(proc)


def publish_dashboard(
    file_path: str,
    artifact_id: str = DASHBOARD_ARTIFACT_ID,
    *,
    expected_current_sha256: str,
) -> dict:
    """CAS-upload one dashboard against the SHA read before the run set was loaded, then prove the version we
    created holds the generated bytes.

    `expected_current_sha256` is required: there is no fallback to a SHA read here, because a SHA read after
    rendering would let a stale page replace a newer one. Verification is pinned to OUR version number (a later
    publisher may legitimately be current by then): its version row must be READY with the SHA the upload
    reported, and its downloaded bytes must hash to that SHA and equal the generated page apart from
    artifactory-mcp's own <head> bridge splice (see platform_splice_only)."""
    if not expected_current_sha256:
        raise DashboardPublishError(
            "expected_current_sha256 is required (read it before rendering)"
        )
    with open(file_path, "rb") as f:
        generated = f.read()
    generated_sha = hashlib.sha256(generated).hexdigest()
    with tempfile.TemporaryDirectory(prefix="lkai-dashboard-readback-") as download_dir:
        proc = _start_mcp(download_dir)
        try:
            current = (_get_artifact(proc, artifact_id, 2).get("artifact") or {}).get("sha256")
            if not current:
                raise DashboardPublishError("Artifactory get_artifact returned no current sha256")
            if current != expected_current_sha256:
                raise DashboardPublishError(
                    "Artifactory dashboard changed while this publication was running; "
                    "rerun with --regenerate-only to include the newer runs"
                )
            uploaded = _call(
                proc,
                3,
                "upload_artifact_version",
                {
                    "artifactId": artifact_id,
                    "filePath": os.path.abspath(file_path),
                    "expectedCurrentSha256": expected_current_sha256,
                    "waitForReady": True,
                },
            )
            version, upload_sha = uploaded.get("versionNumber"), uploaded.get("sha256")
            if not isinstance(version, int) or not upload_sha:
                raise DashboardPublishError(
                    f"Artifactory upload returned no versionNumber/sha256: {uploaded}"
                )
            if uploaded.get("versionStatus") not in ("READY", "PENDING"):
                raise DashboardPublishError(
                    f"Artifactory version {version} is {uploaded.get('versionStatus')!r}, expected 'READY'"
                )
            ours: list = []
            for attempt in range(READY_POLLS):
                rows = _call(
                    proc, 4 + attempt, "list_artifact_versions", {"artifactId": artifact_id}
                )
                ours = [v for v in rows.get("versions") or [] if v.get("versionNumber") == version]
                if len(ours) == 1 and ours[0].get("status") != "PENDING":
                    break
                time.sleep(READY_POLL_S)
            if len(ours) != 1 or ours[0].get("status") != "READY":
                status = ours[0].get("status") if len(ours) == 1 else "missing"
                raise DashboardPublishError(
                    f"Artifactory version {version} is {status!r}, expected 'READY'"
                )
            if ours[0].get("sha256") != upload_sha:
                raise DashboardPublishError(
                    f"Artifactory version {version} sha256 {ours[0].get('sha256')} != uploaded {upload_sha}"
                )
            pinned = _get_artifact(proc, artifact_id, 4 + READY_POLLS, version=version)
            if (pinned.get("artifact") or {}).get("sha256") != upload_sha or not pinned.get(
                "localPath"
            ):
                raise DashboardPublishError(f"Artifactory readback of version {version} failed")
            with open(pinned["localPath"], "rb") as f:
                hosted = f.read()
        finally:
            _stop_mcp(proc)
    hosted_sha = hashlib.sha256(hosted).hexdigest()
    if hosted_sha != upload_sha:
        raise DashboardPublishError(
            f"downloaded version {version} hashes to {hosted_sha}, not {upload_sha}"
        )
    if not platform_splice_only(generated, hosted):
        raise DashboardPublishError(
            f"Artifactory version {version} content does not match the generated dashboard"
        )
    return {
        "artifactId": artifact_id,
        "artifactUrl": uploaded.get("artifactUrl") or pinned.get("artifactUrl"),
        "versionNumber": version,
        "sha256": hosted_sha,
        "generatedSha256": generated_sha,
        "status": "READY",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "result", nargs="?", help="the run's results.json (its directory is the run directory)"
    )
    ap.add_argument(
        "extra", nargs="*", help="more files or directories inside the run directory to upload"
    )
    ap.add_argument(
        "--store",
        default=os.environ.get("LKAI_RESULTS_STORE"),
        help="s3://bucket/prefix or a directory",
    )
    ap.add_argument("--region", default=os.environ.get("LKAI_RESULTS_REGION"))
    ap.add_argument("--profile", default=os.environ.get("LKAI_RESULTS_PROFILE"))
    ap.add_argument(
        "--out", default="build/lkai-dashboard", help="where the generated pages are written"
    )
    ap.add_argument(
        "--replace", action="store_true", help="overwrite this run's existing result.json"
    )
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument(
        "--regenerate-only", action="store_true", help="skip upload; rebuild the dashboard"
    )
    ap.add_argument(
        "--include-fixtures", action="store_true", help="show fixture runs beside real ones"
    )
    ap.add_argument(
        "--skip-dashboard-upload",
        action="store_true",
        help="build dashboard.html locally but do not publish its Artifactory version (local preview/tests)",
    )
    a = ap.parse_args(argv)

    if not a.regenerate_only:
        if not a.result:
            ap.error("result is required unless --regenerate-only")
        with open(a.result, encoding="utf-8") as f:
            res = json.load(f)
        problems = lr.validate(res)
        if problems:
            print(f"INVALID {a.result}", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 2
        print(
            f"valid {lr.SCHEMA_ID}: {res['experiment_id']}/{res['run_id']} status={res['status']}"
        )
        if a.validate_only:
            return 0
    if not a.store:
        ap.error("--store or LKAI_RESULTS_STORE is required")
    expected_dashboard_sha = None
    if not a.skip_dashboard_upload:
        try:
            expected_dashboard_sha = get_dashboard_sha()
        except DashboardPublishError as e:
            print(f"DASHBOARD REFUSED {e}", file=sys.stderr)
            return 4
    store = lr.open_store(a.store, a.region, a.profile)
    if not a.regenerate_only:
        try:
            rep = lr.publish_run(
                res, os.path.dirname(os.path.abspath(a.result)), store, a.extra, a.replace
            )
        except (lr.PublishConflict, FileNotFoundError, ValueError) as e:
            print(f"REFUSED {e}", file=sys.stderr)
            return 3
        print(f"result  {rep.result_uri}")
        print(f"uploaded {len(rep.uploaded)}, unchanged {len(rep.unchanged)}")
        for m in rep.missing:
            print(f"  missing artifact (listed, not found): {m}", file=sys.stderr)
    try:
        site = lr.regenerate(store, a.out, include_fixtures=True if a.include_fixtures else None)
    except lr.RegenerateDidNotSettle as e:
        print(f"DASHBOARD REFUSED {e}", file=sys.stderr)
        return 4
    print(f"dashboard rebuilt from {len(site['runs'])} run(s): {os.path.abspath(a.out)}")
    if not a.skip_dashboard_upload:
        dash = os.path.abspath(os.path.join(a.out, "dashboard.html"))
        try:
            published = publish_dashboard(dash, expected_current_sha256=expected_dashboard_sha)
        except DashboardPublishError as e:
            print(f"DASHBOARD REFUSED {e}", file=sys.stderr)
            return 4
        print("ARTIFACTORY " + json.dumps(published, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
