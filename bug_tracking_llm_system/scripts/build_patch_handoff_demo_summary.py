#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a report-ready fault localization and patch handoff demo summary.")
    parser.add_argument("--localization", required=True, help="Localization JSON or JSONL output.")
    parser.add_argument("--gold", default=None, help="Optional gold JSON/JSONL file.")
    parser.add_argument("--pipeline", default=None, help="Optional integrated pipeline JSON output.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    localization_rows = read_records(Path(args.localization))
    gold_rows = read_records(Path(args.gold)) if args.gold else []
    pipeline = read_records(Path(args.pipeline))[0] if args.pipeline else {}
    markdown = render_markdown(localization_rows, gold_rows, pipeline)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(output), "localization_rows": len(localization_rows)}, ensure_ascii=False, indent=2))


def render_markdown(localization_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]], pipeline: dict[str, Any]) -> str:
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    lines = [
        "# Fault Localization Demo And Patch Handoff Summary",
        "",
        "## Three-Case Localization Demo",
        "",
        "| Ticket | Gold file | Top-1 | Hit@3 | Confidence | Manual review | Patch generator |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in localization_rows:
        ticket_id = _ticket_id(row)
        gold_files = _file_values(gold_by_id.get(ticket_id, {}))
        candidates = _candidate_files(row)
        hit = any(_file_matches(candidate, gold) for candidate in candidates[:3] for gold in gold_files)
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(ticket_id),
                    escape_md(", ".join(gold_files) or "n/a"),
                    f"`{escape_md(candidates[0] if candidates else '')}`",
                    "yes" if hit else "no",
                    escape_md(str(row.get("confidence_level") or "")),
                    "yes" if row.get("should_manual_review") else "no",
                    "allowed" if row.get("recommend_patch_generation") else "manual review first",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top-k Candidate Details", ""])
    for row in localization_rows:
        lines.extend(
            [
                f"### {_ticket_id(row)}",
                "",
                "| Rank | File | Score | Reason |",
                "|---:|---|---:|---|",
            ]
        )
        for candidate in list(row.get("localized_candidates") or [])[:3]:
            if not isinstance(candidate, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(candidate.get("rank", "")),
                        f"`{escape_md(candidate.get('file_path', ''))}`",
                        str(candidate.get("score", "")),
                        escape_md(str(candidate.get("reason", ""))[:220]),
                    ]
                )
                + " |"
            )
        lines.append("")

    if pipeline:
        patch = pipeline.get("patch") if isinstance(pipeline.get("patch"), dict) else {}
        context = patch.get("localization_context") if isinstance(patch.get("localization_context"), list) else []
        lines.extend(
            [
                "## Integrated Pipeline Patch Handoff",
                "",
                f"- Pipeline status: `{pipeline.get('status', '')}`",
                f"- Patch status: `{patch.get('patch_status', '')}`",
                f"- Confidence level: `{patch.get('confidence_level', '')}`",
                f"- Requires manual patch: `{patch.get('requires_manual_patch', '')}`",
                f"- Requires manual review: `{patch.get('requires_manual_review', '')}`",
                f"- Recommend patch generation: `{patch.get('recommend_patch_generation', '')}`",
                "",
                "| Rank | File | Confidence | Manual review | Repository context |",
                "|---:|---|---|---|---|",
            ]
        )
        for item in context[:5]:
            if not isinstance(item, dict):
                continue
            repo_context = item.get("repository_context") if isinstance(item.get("repository_context"), dict) else {}
            context_text = repo_context.get("context_strategy") or ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("rank", "")),
                        f"`{escape_md(item.get('file_path', ''))}`",
                        escape_md(str(item.get("confidence_level", ""))),
                        "yes" if item.get("should_manual_review") else "no",
                        escape_md(str(context_text)),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    return [value] if isinstance(value, dict) else []


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or "")


def _file_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "files", "file_path", "file"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return [_normalize_path(value) for value in values if _normalize_path(value)]


def _candidate_files(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("localized_files")
    files: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            file_path = _normalize_path(str(candidate.get("file_path") or candidate.get("file") or ""))
            if file_path and file_path not in files:
                files.append(file_path)
    return files


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
