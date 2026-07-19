from __future__ import annotations

import argparse
import json
import re
import statistics
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assignee_phase1_common import DEFAULT_RAW_PATH, DEFAULT_PHASE1_ROOT, clean, tokens, write_json


DEFAULT_PHASE2_ROOT = DEFAULT_PHASE1_ROOT.parent / "phase2_temporal_description"
HTML_RE = re.compile(r"<[^>]+>")
CODE_LOG_RE = re.compile(
    r"(\btraceback\b|\bassertion failure\b|\berror\b|\bexception\b|\bcrash\b|/builds/|```|0x[0-9a-f]+|\bat\s+\w+)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a versioned BMO raw dataset enriched with first public comments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PHASE2_ROOT / "data" / "bmo_paper_2024_3k_description_enriched_raw.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_PHASE2_ROOT / "reports" / "bmo_paper_2024_3k_description_enriched" / "description_enrichment_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_PHASE2_ROOT / "reports" / "bmo_paper_2024_3k_description_enriched" / "description_missing_root_cause.md",
    )
    parser.add_argument("--base-url", default="https://bugzilla.mozilla.org/rest")
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = load_existing(args.output) if args.resume and args.output.exists() and not args.force else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fetch_started_at = datetime.now(UTC).isoformat()

    enriched_rows = enrich_rows(
        rows=rows,
        existing=existing,
        base_url=args.base_url,
        sleep=args.sleep,
        timeout=args.timeout,
        insecure=args.insecure,
        workers=max(1, args.workers),
    )
    ordered_rows = ordered_enriched_rows(rows, existing, enriched_rows)
    write_enriched_jsonl(args.output, ordered_rows)

    audit = build_audit(
        rows=ordered_rows,
        input_path=args.input,
        output_path=args.output,
        fetch_started_at=fetch_started_at,
        fetch_finished_at=datetime.now(UTC).isoformat(),
    )
    write_json(args.audit_output, audit)
    write_report(args.report_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path) if row.get("id")}


def enrich_rows(
    *,
    rows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    base_url: str,
    sleep: float,
    timeout: int,
    insecure: bool,
    workers: int,
) -> dict[str, dict[str, Any]]:
    todo = [(index, row) for index, row in enumerate(rows, start=1) if str(row.get("id") or "").strip() not in existing]
    enriched = dict(existing)
    if workers <= 1:
        for index, row in todo:
            bug_id, enriched_row = enrich_one(
                row,
                base_url=base_url,
                sleep=sleep,
                timeout=timeout,
                insecure=insecure,
            )
            if bug_id:
                enriched[bug_id] = enriched_row
            if index % 100 == 0:
                print(json.dumps({"processed": index, "written_total": len(enriched)}, ensure_ascii=False), flush=True)
        return enriched

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                enrich_one,
                row,
                base_url=base_url,
                sleep=sleep,
                timeout=timeout,
                insecure=insecure,
            ): index
            for index, row in todo
        }
        completed = 0
        for future in as_completed(futures):
            bug_id, enriched_row = future.result()
            if bug_id:
                enriched[bug_id] = enriched_row
            completed += 1
            if completed % 100 == 0:
                print(
                    json.dumps(
                        {"completed_fetches": completed, "todo": len(todo), "written_total": len(enriched)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return enriched


def enrich_one(
    row: dict[str, Any],
    *,
    base_url: str,
    sleep: float,
    timeout: int,
    insecure: bool,
) -> tuple[str, dict[str, Any]]:
    bug_id = str(row.get("id") or "").strip()
    if not bug_id:
        return "", dict(row)
    description, metadata = fetch_first_public_comment(
        base_url=base_url,
        bug_id=bug_id,
        sleep=sleep,
        timeout=timeout,
        insecure=insecure,
    )
    enriched = dict(row)
    enriched["description"] = description
    enriched["description_source"] = "first_comment"
    enriched["description_fetch_timestamp"] = datetime.now(UTC).isoformat()
    enriched["description_missing"] = not bool(description.strip())
    enriched["description_length"] = len(description)
    enriched["description_public_comment_count"] = metadata.get("public_comment_count", 0)
    enriched["description_fetch_error"] = metadata.get("error", "")
    return bug_id, enriched


def ordered_enriched_rows(
    original_rows: list[dict[str, Any]], existing: dict[str, dict[str, Any]], enriched: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in original_rows:
        bug_id = str(row.get("id") or "").strip()
        output.append(enriched.get(bug_id) or existing.get(bug_id) or row)
    return output


def write_enriched_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fetch_first_public_comment(
    *,
    base_url: str,
    bug_id: str,
    sleep: float,
    timeout: int,
    insecure: bool,
) -> tuple[str, dict[str, Any]]:
    time.sleep(sleep)
    context = ssl._create_unverified_context() if insecure else None
    url = f"{base_url.rstrip('/')}/bug/{bug_id}/comment"
    request = urllib.request.Request(url, headers={"User-Agent": "bug-llm-project-assignee-triage/phase2"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return "", {"error": str(exc), "public_comment_count": 0}
    bug_comments = payload.get("bugs", {}).get(str(bug_id), {})
    comments = bug_comments.get("comments", []) if isinstance(bug_comments, dict) else []
    public_comments = [comment for comment in comments if isinstance(comment, dict) and not comment.get("is_private")]
    if not public_comments:
        return "", {"public_comment_count": 0}
    return clean(public_comments[0].get("text")), {"public_comment_count": len(public_comments)}


def build_audit(
    *, rows: list[dict[str, Any]], input_path: Path, output_path: Path, fetch_started_at: str, fetch_finished_at: str
) -> dict[str, Any]:
    lengths = [int(row.get("description_length") or len(clean(row.get("description")))) for row in rows]
    nonempty = [length for length in lengths if length > 0]
    token_lengths = [len(tokens(clean(row.get("description")))) for row in rows]
    return {
        "dataset_version": "bmo_paper_2024_3k_description_enriched",
        "input_raw_path": str(input_path),
        "output_raw_path": str(output_path),
        "description_source": "first_comment",
        "fetch_started_at": fetch_started_at,
        "fetch_finished_at": fetch_finished_at,
        "total_rows": len(rows),
        "description_nonempty_count": len(nonempty),
        "description_nonempty_ratio": round(len(nonempty) / len(rows), 6) if rows else 0.0,
        "description_missing_count": len(rows) - len(nonempty),
        "description_missing_ratio": round((len(rows) - len(nonempty)) / len(rows), 6) if rows else 0.0,
        "description_length_mean": round(statistics.mean(lengths), 3) if lengths else 0.0,
        "description_length_median": round(statistics.median(lengths), 3) if lengths else 0.0,
        "description_token_length_mean": round(statistics.mean(token_lengths), 3) if token_lengths else 0.0,
        "description_token_length_median": round(statistics.median(token_lengths), 3) if token_lengths else 0.0,
        "html_content_count": sum(1 for row in rows if HTML_RE.search(clean(row.get("description")))),
        "html_content_ratio": round(sum(1 for row in rows if HTML_RE.search(clean(row.get("description")))) / len(rows), 6) if rows else 0.0,
        "code_or_log_presence_count": sum(1 for row in rows if CODE_LOG_RE.search(clean(row.get("description")))),
        "code_or_log_presence_ratio": round(sum(1 for row in rows if CODE_LOG_RE.search(clean(row.get("description")))) / len(rows), 6) if rows else 0.0,
        "fetch_error_count": sum(1 for row in rows if clean(row.get("description_fetch_error"))),
    }


def write_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Description Missing Root-Cause Verification",
        "",
        "## Root Cause",
        "",
        "- The original BMO 3k raw file has no nonempty `description` values.",
        "- The fetch script stores first comments only when comment fetching is enabled.",
        "- The prepare script maps `description` from `row.get(\"description\")`, so missing raw descriptions propagate to processed rows.",
        "- This enriched version preserves original ticket fields and adds `description_source = first_comment`.",
        "",
        "## Enriched Dataset Audit",
        "",
        f"- Total rows: `{audit['total_rows']}`",
        f"- Nonempty descriptions: `{audit['description_nonempty_count']}` ({audit['description_nonempty_ratio']})",
        f"- Mean / median description length: `{audit['description_length_mean']}` / `{audit['description_length_median']}`",
        f"- Mean / median token length: `{audit['description_token_length_mean']}` / `{audit['description_token_length_median']}`",
        f"- HTML content ratio: `{audit['html_content_ratio']}`",
        f"- Code/log presence ratio: `{audit['code_or_log_presence_ratio']}`",
        f"- Fetch errors: `{audit['fetch_error_count']}`",
        "",
        "First public comment is a reasonable operational proxy for bug description, but it can include boilerplate, logs, quotes, and code snippets.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
