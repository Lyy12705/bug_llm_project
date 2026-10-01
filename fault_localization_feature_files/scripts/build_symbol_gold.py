#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.symbol_gold import (
    PATCH_SYMBOL_PROVENANCE_SCHEMA_VERSION,
    SYMBOL_GOLD_RECORD_SCHEMA_VERSION,
    SymbolGoldRecordV1,
    map_patch_hunks_to_symbol_gold,
    parse_unified_diff,
)
from utils.symbol_localization import (
    SYMBOL_PARSE_RESULT_SCHEMA_VERSION,
    SymbolParseResult,
    extract_symbols_from_source,
)


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".scala": "scala",
}
AUDIT_FIELDS = (
    "gold_id",
    "ticket_id",
    "repo",
    "base_commit",
    "file_path",
    "mapping_status",
    "mapping_method",
    "mapping_confidence",
    "exclusion_reason",
    "symbol_id",
    "qualified_name",
    "symbol_kind",
    "start_line",
    "start_column",
    "end_line",
    "end_column",
    "change_type",
    "hunk_index",
    "old_file_path",
    "new_file_path",
    "old_start_line",
    "old_line_count",
    "new_start_line",
    "new_line_count",
    "matched_old_lines",
    "context_old_lines",
    "insertion_anchor_line",
    "patch_sha256",
    "review_status",
    "file_correct",
    "qualified_name_correct",
    "symbol_kind_correct",
    "review_note",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build file-qualified symbol gold from developer patches and base commits."
    )
    parser.add_argument("--tickets", required=True, help="Input tickets JSONL containing patch fields.")
    parser.add_argument("--repo-cache-dir", required=True, help="Cached Git repositories directory.")
    parser.add_argument("--output", required=True, help="Output SymbolGoldRecordV1 JSONL path.")
    parser.add_argument("--audit-csv", required=True, help="Output flattened provenance review CSV path.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tickets_path = Path(args.tickets).resolve()
    repo_cache_dir = Path(args.repo_cache_dir).resolve()
    if not tickets_path.is_file():
        raise FileNotFoundError(f"Tickets JSONL does not exist: {tickets_path}")
    if not repo_cache_dir.is_dir():
        raise FileNotFoundError(f"Repository cache directory does not exist: {repo_cache_dir}")

    tickets = read_jsonl(tickets_path)
    records, summary = build_symbol_gold_records(tickets, repo_cache_dir=repo_cache_dir)
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit_csv).resolve()
    atomic_write_jsonl(output_path, [record.to_dict() for record in records])
    atomic_write_audit_csv(audit_path, records)
    summary.update(
        {
            "output": str(output_path),
            "audit_csv": str(audit_path),
            "symbol_gold_schema_version": SYMBOL_GOLD_RECORD_SCHEMA_VERSION,
            "patch_provenance_schema_version": PATCH_SYMBOL_PROVENANCE_SCHEMA_VERSION,
            "symbol_parse_result_schema_version": SYMBOL_PARSE_RESULT_SCHEMA_VERSION,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_symbol_gold_records(
    tickets: Sequence[dict[str, Any]],
    *,
    repo_cache_dir: Path,
) -> tuple[list[SymbolGoldRecordV1], dict[str, Any]]:
    records: list[SymbolGoldRecordV1] = []
    repositories: set[str] = set()
    for row_number, ticket in enumerate(tickets, start=1):
        ticket_id = _ticket_id(ticket)
        repo = str(ticket.get("repo") or "").strip()
        requested_commit = str(ticket.get("base_commit") or "").strip()
        patch = str(ticket.get("patch") or ticket.get("developer_patch") or "")
        if not ticket_id or not repo or not requested_commit or not patch:
            raise ValueError(
                f"Ticket row {row_number} requires ticket_id, repo, base_commit, and patch."
            )
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", requested_commit):
            raise ValueError(f"Ticket {ticket_id!r} has an invalid Git base_commit.")

        hunks = parse_unified_diff(patch)
        repository_path = resolve_repository(ticket, repo_cache_dir=repo_cache_dir)
        resolved_commit = (
            resolve_commit(repository_path, requested_commit)
            if repository_path is not None
            else None
        )
        base_commit = resolved_commit or requested_commit.lower()
        parse_results: list[SymbolParseResult] = []
        if repository_path is not None and resolved_commit is not None:
            old_paths = sorted(
                {
                    hunk.old_file_path
                    for hunk in hunks
                    if hunk.change_type != "added" and hunk.old_file_path
                }
            )
            parse_results = [
                extract_base_commit_symbols(
                    repository_path,
                    repo=repo,
                    base_commit=base_commit,
                    file_path=file_path,
                )
                for file_path in old_paths
            ]
        records.extend(
            map_patch_hunks_to_symbol_gold(
                ticket_id=ticket_id,
                repo=repo,
                base_commit=base_commit,
                hunks=hunks,
                parse_results=parse_results,
            )
        )
        repositories.add(repo)

    mapping_methods = Counter(record.mapping_method for record in records)
    exclusion_reasons = Counter(
        record.exclusion_reason for record in records if record.exclusion_reason
    )
    summary = {
        "ticket_count": len(tickets),
        "repository_count": len(repositories),
        "gold_record_count": len(records),
        "mapped_count": sum(record.mapping_status == "mapped" for record in records),
        "excluded_count": sum(record.mapping_status == "excluded" for record in records),
        "mapping_methods": dict(sorted(mapping_methods.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
    }
    return records, summary


def resolve_repository(ticket: dict[str, Any], *, repo_cache_dir: Path) -> Path | None:
    for key in ("local_repo_path", "repo_path", "repository_path"):
        if ticket.get(key):
            path = Path(str(ticket[key])).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Configured repository path does not exist: {path}")
            return path
    repo = str(ticket.get("repo") or "").strip()
    candidate = (repo_cache_dir / safe_name(repo)).resolve()
    return candidate if candidate.is_dir() else None


def resolve_commit(repository_path: Path, commit: str) -> str | None:
    completed = _run_git_bytes(
        repository_path,
        ["rev-parse", "--verify", f"{commit}^{{commit}}"],
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.decode("ascii", errors="ignore").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def extract_base_commit_symbols(
    repository_path: Path,
    *,
    repo: str,
    base_commit: str,
    file_path: str,
) -> SymbolParseResult:
    language = LANGUAGE_BY_SUFFIX.get(Path(file_path).suffix.lower(), "unknown")
    tree_entry = _run_git_bytes(
        repository_path,
        ["ls-tree", "-z", base_commit, "--", file_path],
    )
    if tree_entry.returncode != 0 or not tree_entry.stdout:
        return _file_diagnostic(
            repo=repo,
            base_commit=base_commit,
            file_path=file_path,
            language=language,
            status="missing_file",
            error_type="FileNotFoundError",
            message="File is absent from the requested base commit.",
        )
    metadata = tree_entry.stdout.split(b"\t", 1)[0].decode("ascii", errors="ignore")
    mode = metadata.split(" ", 1)[0]
    if mode == "120000":
        return _file_diagnostic(
            repo=repo,
            base_commit=base_commit,
            file_path=file_path,
            language=language,
            status="symlink_escape",
            error_type="RepositoryBoundaryError",
            message="Base-commit source path is a symbolic link and was not followed.",
        )

    content = _run_git_bytes(
        repository_path,
        ["show", f"{base_commit}:{file_path}"],
    )
    if content.returncode != 0:
        return _file_diagnostic(
            repo=repo,
            base_commit=base_commit,
            file_path=file_path,
            language=language,
            status="missing_file",
            error_type="FileNotFoundError",
            message="File could not be read from the requested base commit.",
        )
    source = content.stdout.decode("utf-8", errors="replace")
    return extract_symbols_from_source(
        source,
        repo=repo,
        base_commit=base_commit,
        file_path=file_path,
        language=language,
        ast_input_source="base_commit_developer_patch",
    )


def _file_diagnostic(
    *,
    repo: str,
    base_commit: str,
    file_path: str,
    language: str,
    status: str,
    error_type: str,
    message: str,
) -> SymbolParseResult:
    return SymbolParseResult(
        repo=repo,
        base_commit=base_commit,
        file_path=file_path,
        language=language,
        status=status,
        error_type=error_type,
        message=message,
    )


def _run_git_bytes(
    repository_path: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_path}",
            "-C",
            str(repository_path),
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def _ticket_id(row: dict[str, Any]) -> str:
    return str(
        row.get("ticket_id")
        or row.get("instance_id")
        or row.get("id")
        or row.get("bug_id")
        or ""
    ).strip()


def safe_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in normalized
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def atomic_write_audit_csv(path: Path, records: Sequence[SymbolGoldRecordV1]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        provenance = record.provenance
        writer.writerow(
            {
                "gold_id": record.gold_id,
                "ticket_id": record.ticket_id,
                "repo": record.repo,
                "base_commit": record.base_commit,
                "file_path": record.file_path,
                "mapping_status": record.mapping_status,
                "mapping_method": record.mapping_method,
                "mapping_confidence": record.mapping_confidence,
                "exclusion_reason": record.exclusion_reason,
                "symbol_id": record.symbol_id,
                "qualified_name": record.qualified_name,
                "symbol_kind": record.symbol_kind,
                "start_line": record.start_line,
                "start_column": record.start_column,
                "end_line": record.end_line,
                "end_column": record.end_column,
                "change_type": provenance.change_type,
                "hunk_index": provenance.hunk_index,
                "old_file_path": provenance.old_file_path,
                "new_file_path": provenance.new_file_path,
                "old_start_line": provenance.old_start_line,
                "old_line_count": provenance.old_line_count,
                "new_start_line": provenance.new_start_line,
                "new_line_count": provenance.new_line_count,
                "matched_old_lines": " ".join(map(str, provenance.matched_old_lines)),
                "context_old_lines": " ".join(map(str, provenance.context_old_lines)),
                "insertion_anchor_line": provenance.insertion_anchor_line,
                "patch_sha256": provenance.patch_sha256,
                "review_status": "",
                "file_correct": "",
                "qualified_name_correct": "",
                "symbol_kind_correct": "",
                "review_note": "",
            }
        )
    atomic_write_text(path, buffer.getvalue())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
