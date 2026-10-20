#!/usr/bin/env python3
"""Publish one lambda-kleidiai experiment run to the results store and rebuild the dashboard.

    python examples/lambda-kleidiai/results/publish_result.py <run>/results.json [extra artifacts...]

Steps, all idempotent:
  1. validate the run file against results/schema.json (and the semantic checks in lkai_results.validate);
  2. upload the artifacts it lists (artifacts{}, profiling_artifacts[].path) plus any extra paths, and then its
     result.json, under runs/<experiment_id>/<run_id>/ in the store. Nothing outside that prefix is written;
  3. regenerate site/ in the store from every result.json there, and write the single-file dashboard to --out;
  4. upload --out/dashboard.html to Artifactory through its installed MCP server, using current-SHA compare-and-swap,
     wait for READY, then re-fetch and verify the resulting version.

The store is --store or $LKAI_RESULTS_STORE: s3://bucket/prefix, or a local directory for a dry run.
Exit codes: 0 published, 2 invalid run file, 3 run conflict/artifact refusal, 4 dashboard upload refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import subprocess
import sys
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


def _start_mcp() -> subprocess.Popen:
    binary = shutil.which(ARTIFACTORY_MCP)
    if not binary:
        raise DashboardPublishError(
            f"{ARTIFACTORY_MCP!r} not found; install/configure Artifactory MCP or use "
            "--skip-dashboard-upload for a local-only preview"
        )
    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
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


def _get_artifact(proc: subprocess.Popen, artifact_id: str, request_id: int) -> dict:
    return _tool_payload(
        _rpc_send(
            proc,
            request_id,
            "tools/call",
            {"name": "get_artifact", "arguments": {"artifactId": artifact_id}},
        ),
        "get_artifact",
    )


def get_dashboard_sha(artifact_id: str = DASHBOARD_ARTIFACT_ID) -> str:
    """Read the dashboard generation before any run or derived-page writes begin."""
    proc = _start_mcp()
    try:
        artifact = (_get_artifact(proc, artifact_id, 2).get("artifact") or {})
        sha = artifact.get("sha256")
        if not sha:
            raise DashboardPublishError("Artifactory get_artifact returned no current sha256")
        return sha
    finally:
        _stop_mcp(proc)


def publish_dashboard(
    file_path: str,
    artifact_id: str = DASHBOARD_ARTIFACT_ID,
    expected_current_sha256: str | None = None,
) -> dict:
    """CAS-upload one dashboard and verify Artifactory stored the generated bytes."""
    with open(file_path, "rb") as f:
        generated_sha = hashlib.sha256(f.read()).hexdigest()
    proc = _start_mcp()
    try:
        before = _get_artifact(proc, artifact_id, 2)
        current = before.get("artifact") or {}
        current_sha = current.get("sha256")
        if not current_sha:
            raise DashboardPublishError("Artifactory get_artifact returned no current sha256")
        if expected_current_sha256 is not None and current_sha != expected_current_sha256:
            raise DashboardPublishError(
                "Artifactory dashboard changed while this publication was running; "
                "regenerate and retry"
            )
        uploaded = _tool_payload(
            _rpc_send(
                proc,
                3,
                "tools/call",
                {
                    "name": "upload_artifact_version",
                    "arguments": {
                        "artifactId": artifact_id,
                        "filePath": os.path.abspath(file_path),
                        "expectedCurrentSha256": expected_current_sha256 or current_sha,
                        "waitForReady": True,
                    },
                },
            ),
            "upload_artifact_version",
        )
        after = _get_artifact(proc, artifact_id, 4)
        artifact = after.get("artifact") or {}
        if artifact.get("status") != "READY":
            raise DashboardPublishError(
                f"Artifactory version is {artifact.get('status')!r}, expected 'READY'"
            )
        uploaded_version = uploaded.get("versionNumber") or (uploaded.get("artifact") or {}).get(
            "currentVersionNumber"
        )
        if uploaded_version and artifact.get("currentVersionNumber") != uploaded_version:
            raise DashboardPublishError(
                "Artifactory current version moved after upload; regenerate and retry"
            )
        if artifact.get("sha256") != generated_sha:
            raise DashboardPublishError(
                "Artifactory READY version sha256 does not match generated dashboard"
            )
        return {
            "artifactId": artifact_id,
            "artifactUrl": after.get("artifactUrl") or artifact.get("artifactUrl"),
            "versionNumber": artifact.get("currentVersionNumber"),
            "sha256": artifact.get("sha256"),
            "status": artifact.get("status"),
        }
    finally:
        _stop_mcp(proc)


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
        help="build site/dashboard.html but do not publish its Artifactory version (local preview/tests)",
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
    site = lr.regenerate(store, a.out, include_fixtures=True if a.include_fixtures else None)
    print(f"dashboard rebuilt from {len(site['runs'])} run(s): {store.uri('site/index.html')}")
    if not a.skip_dashboard_upload:
        dash = os.path.abspath(os.path.join(a.out, "dashboard.html"))
        try:
            published = publish_dashboard(
                dash, expected_current_sha256=expected_dashboard_sha
            )
        except DashboardPublishError as e:
            print(f"DASHBOARD REFUSED {e}", file=sys.stderr)
            return 4
        print("ARTIFACTORY " + json.dumps(published, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
