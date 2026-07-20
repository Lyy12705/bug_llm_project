from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://bugzilla.mozilla.org/rest"
DEFAULT_FIELDS = [
    "id",
    "summary",
    "product",
    "component",
    "priority",
    "severity",
    "status",
    "resolution",
    "creation_time",
    "last_change_time",
    "assigned_to",
    "creator",
    "op_sys",
    "platform",
    "version",
    "dupe_of",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a public BMO assignee-triage dataset.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Bugzilla REST base URL.")
    parser.add_argument("--products", nargs="+", default=["Core", "Firefox"], help="Bugzilla products to fetch.")
    parser.add_argument("--statuses", nargs="+", default=["RESOLVED", "VERIFIED", "CLOSED"], help="Bug statuses.")
    parser.add_argument("--resolutions", nargs="+", default=["FIXED"], help="Bug resolutions.")
    parser.add_argument("--start-date", default="2020-01-01", help="Fetch bugs created on or after this date.")
    parser.add_argument("--end-date", default=None, help="Optional client-side inclusive creation date limit.")
    parser.add_argument("--limit", type=int, default=500, help="Bugzilla page size.")
    parser.add_argument("--max-bugs", type=int, default=1000, help="Maximum bugs to write; 0 means no cap.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between requests.")
    parser.add_argument("--include-comments", action="store_true", help="Fetch the first public comment as description.")
    parser.add_argument(
        "--seal-output",
        action="store_true",
        help="Refuse to overwrite an existing output or manifest when creating a sealed holdout.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for local environments with broken CA bundles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_ROOT / "data" / "raw" / "bmo_bugs_raw.jsonl",
        help="Raw JSONL output path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAPER_ROOT / "data" / "raw" / "bmo_fetch_manifest.json",
        help="Fetch manifest JSON path.",
    )
    args = parser.parse_args()
    setattr(get_json, "insecure", args.insecure)

    if args.seal_output:
        existing = [path for path in (args.output, args.manifest) if path.exists()]
        if existing:
            raise SystemExit(
                "Sealed holdout target already exists: " + ", ".join(str(path) for path in existing)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    offset = 0
    seen_ids: set[int] = set()
    started_at = datetime.now(UTC).isoformat()
    end_date = parse_date(args.end_date) if args.end_date else None
    complete = True
    termination_reason = "source_exhausted"
    reached_limit = False
    reached_end_date = False
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".partial",
        delete=False,
    )
    temporary_output = Path(temporary_handle.name)
    try:
        with temporary_handle as handle:
            while True:
                bugs = search_bugs(args, offset)
                if not bugs:
                    break
                for bug in bugs:
                    bug_id = int(bug["id"])
                    if bug_id in seen_ids:
                        continue
                    seen_ids.add(bug_id)
                    if end_date and parse_date(str(bug.get("creation_time", ""))) > end_date:
                        reached_end_date = True
                        termination_reason = "end_date_reached"
                        break
                    if args.include_comments:
                        bug["description"] = fetch_first_comment(args.base_url, bug_id, args.sleep)
                    handle.write(json.dumps(bug, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    if args.max_bugs and written >= args.max_bugs:
                        complete = False
                        reached_limit = True
                        termination_reason = "max_bugs_reached"
                        break
                if reached_limit or reached_end_date:
                    break
                offset += args.limit
                time.sleep(args.sleep)
        temporary_output.replace(args.output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    write_manifest(
        args,
        started_at,
        written,
        offset,
        complete=complete,
        termination_reason=termination_reason,
    )


def search_bugs(args: argparse.Namespace, offset: int) -> list[dict[str, Any]]:
    params: list[tuple[str, Any]] = [
        ("include_fields", ",".join(DEFAULT_FIELDS)),
        ("creation_time", args.start_date),
        ("limit", args.limit),
        ("offset", offset),
        ("order", "creation_time"),
    ]
    params.extend(("product", product) for product in args.products)
    params.extend(("status", status) for status in args.statuses)
    params.extend(("resolution", resolution) for resolution in args.resolutions)
    payload = get_json(f"{args.base_url.rstrip('/')}/bug?{urllib.parse.urlencode(params)}")
    bugs = payload.get("bugs", [])
    if not isinstance(bugs, list):
        return []
    return [bug for bug in bugs if isinstance(bug, dict)]


def fetch_first_comment(base_url: str, bug_id: int, sleep: float) -> str:
    time.sleep(sleep)
    try:
        payload = get_json(f"{base_url.rstrip('/')}/bug/{bug_id}/comment")
    except RuntimeError:
        return ""
    comments_by_bug = payload.get("bugs", {})
    bug_comments = comments_by_bug.get(str(bug_id), {}) if isinstance(comments_by_bug, dict) else {}
    comments = bug_comments.get("comments", []) if isinstance(bug_comments, dict) else []
    public_comments = [comment for comment in comments if isinstance(comment, dict) and not comment.get("is_private")]
    if not public_comments:
        return ""
    return str(public_comments[0].get("text") or "").strip()


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "bug-llm-project-assignee-triage/1.0"})
    context = tls_context(insecure=bool(getattr(get_json, "insecure", False)))
    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Request failed: {url}: {exc}") from exc


def tls_context(*, insecure: bool) -> ssl.SSLContext | None:
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def parse_date(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    if len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    return datetime.fromisoformat(normalized)


def write_manifest(
    args: argparse.Namespace,
    started_at: str,
    written: int,
    offset: int,
    *,
    complete: bool,
    termination_reason: str,
) -> None:
    output_sha256, first_creation_time, last_creation_time = inspect_output(args.output)
    manifest = {
        "source": args.base_url,
        "products": args.products,
        "statuses": args.statuses,
        "resolutions": args.resolutions,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "limit": args.limit,
        "max_bugs": args.max_bugs,
        "include_comments": args.include_comments,
        "output": str(args.output),
        "output_sha256": output_sha256,
        "first_creation_time": first_creation_time,
        "last_creation_time": last_creation_time,
        "bugs_written": written,
        "last_offset": offset,
        "complete": complete,
        "termination_reason": termination_reason,
        "sealed_output": bool(args.seal_output),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def inspect_output(path: Path) -> tuple[str, str, str]:
    digest = hashlib.sha256()
    first_creation_time = ""
    last_creation_time = ""
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Fetched output is not valid JSONL: {path}") from exc
            creation_time = str(row.get("creation_time") or "").strip()
            if creation_time and not first_creation_time:
                first_creation_time = creation_time
            if creation_time:
                last_creation_time = creation_time
    return digest.hexdigest(), first_creation_time, last_creation_time


if __name__ == "__main__":
    main()
