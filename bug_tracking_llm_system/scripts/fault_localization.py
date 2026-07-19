#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.fault_localization import (
    build_code_index,
    format_user_facing_localization_result,
    format_user_facing_validation_error,
    load_code_index,
    localize_ticket,
    validate_localization_request,
)
from utils.llm_client import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline fault localization for bug tickets.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ticket", help="Single ticket JSON path.")
    input_group.add_argument("--tickets-jsonl", help="Ticket JSONL path.")
    parser.add_argument("--repo-path", help="Repository or source folder. Required when --code-index is not supplied.")
    parser.add_argument("--code-index", help="Prebuilt code index JSON from build_code_index.py.")
    parser.add_argument(
        "--index-cache-dir",
        default=None,
        help="Optional directory for automatic code-index caching when --repo-path is used.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild and overwrite the cached code index.",
    )
    parser.add_argument("--output", default=None, help="Output JSON/JSONL path. Prints to stdout when omitted.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of localization candidates to return.")
    parser.add_argument(
        "--output-format",
        choices=("raw", "user-facing"),
        default="raw",
        help="raw keeps the evaluation JSON contract; user-facing emits a compact beta/demo result contract.",
    )
    parser.add_argument(
        "--include-code-preview",
        action="store_true",
        help="Include short code previews in user-facing output.",
    )
    parser.add_argument(
        "--min-ticket-chars",
        type=int,
        default=20,
        help="Minimum ticket text length before localization is considered valid.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "tfidf", "sbert", "sentence-transformers", "tfidf-sbert-rerank"),
        default="tfidf",
        help="Retrieval backend. auto uses cached sentence-transformers when available, otherwise TF-IDF.",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sbert-cache-dir", default=None, help="Optional persistent SBERT embedding cache directory.")
    parser.add_argument(
        "--allow-sbert-download",
        action="store_true",
        help="Allow sentence-transformers to download the model if it is not cached locally.",
    )
    parser.add_argument("--llm-rerank", action="store_true", help="Use Ollama/Code Llama to rerank retrieved chunks.")
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument(
        "--disable-repository-proximity",
        action="store_true",
        help="Disable lightweight import/symbol repository proximity scoring for controlled comparisons.",
    )
    parser.add_argument(
        "--progress",
        choices=("none", "text", "json"),
        default="none",
        help="Emit progress events to stderr for long-running localization jobs.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional JSONL file for progress events, useful for async job polling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reporter = ProgressReporter(mode=args.progress, file_path=args.progress_file)
    if not args.code_index and not args.repo_path:
        raise SystemExit("--repo-path is required when --code-index is not supplied.")

    tickets = _read_tickets(args.ticket or args.tickets_jsonl)
    reporter.emit(
        "localization_started",
        tickets=len(tickets),
        output_format=args.output_format,
        top_k=args.top_k,
        embedding_backend=args.embedding_backend,
    )
    repo_errors = _repository_input_errors(args.repo_path, code_index_supplied=bool(args.code_index))
    if repo_errors:
        reporter.emit("input_rejected", errors=repo_errors)
        if args.output_format == "user-facing":
            results = [
                format_user_facing_validation_error(
                    validate_localization_request(
                        ticket,
                        repo_path=args.repo_path,
                        code_index_supplied=bool(args.code_index),
                        min_ticket_chars=args.min_ticket_chars,
                    )
                )
                for ticket in tickets
            ]
            _write_results(results, args.output, force_jsonl=bool(args.tickets_jsonl))
            reporter.emit("localization_completed", status="invalid_input", results=len(results), output=args.output or "stdout")
            return
        raise SystemExit("; ".join(repo_errors))

    code_index, index_runtime = _load_or_build_code_index_for_cli(args, reporter)
    llm_client = OllamaClient(url=args.ollama_url, model=args.ollama_model) if args.llm_rerank else None
    results: list[dict[str, Any]] = []
    for position, ticket in enumerate(tickets, start=1):
        ticket_started = time.perf_counter()
        ticket_id = str(ticket.get("ticket_id") or ticket.get("id") or ticket.get("bug_id") or f"ticket-{position}")
        reporter.emit("ticket_started", ticket_id=ticket_id, position=position, total=len(tickets))
        validation = validate_localization_request(
            ticket,
            repo_path=code_index.repository_path,
            code_index_supplied=True,
            min_ticket_chars=args.min_ticket_chars,
        )
        if validation["errors"]:
            if args.output_format == "user-facing":
                formatted_error = format_user_facing_validation_error(validation)
                formatted_error["runtime"] = _ticket_runtime(index_runtime, ticket_started)
                results.append(formatted_error)
                reporter.emit("ticket_completed", ticket_id=ticket_id, status="invalid_input")
                continue
            reporter.emit("ticket_failed", ticket_id=ticket_id, errors=validation["errors"])
            raise ValueError("; ".join(validation["errors"]))

        result = localize_ticket(
            ticket,
            code_index=code_index,
            top_k=args.top_k,
            embedding_backend=args.embedding_backend,
            sbert_model=args.sbert_model,
            sbert_local_files_only=not args.allow_sbert_download,
            sbert_cache_dir=Path(args.sbert_cache_dir) if args.sbert_cache_dir else None,
            llm_client=llm_client,
            llm_rerank=args.llm_rerank,
            repository_proximity=not args.disable_repository_proximity,
        )
        runtime = _ticket_runtime(index_runtime, ticket_started)
        result["runtime"] = runtime
        if args.output_format == "user-facing":
            result = format_user_facing_localization_result(
                result,
                validation=validation,
                include_code_preview=args.include_code_preview,
                top_k=args.top_k,
            )
            result["runtime"] = runtime
        results.append(result)
        reporter.emit(
            "ticket_completed",
            ticket_id=ticket_id,
            status=result.get("status", "ok"),
            confidence_level=result.get("confidence_level")
            or (result.get("summary") or {}).get("confidence_level", ""),
            runtime_seconds=runtime["ticket_runtime_seconds"],
        )
    _write_results(results, args.output, force_jsonl=bool(args.tickets_jsonl))
    reporter.emit("localization_completed", status="ok", results=len(results), output=args.output or "stdout")


class ProgressReporter:
    def __init__(self, *, mode: str = "none", file_path: str | Path | None = None) -> None:
        self.mode = mode
        self.file_path = Path(file_path) if file_path else None

    def emit(self, event: str, **payload: Any) -> None:
        row = {
            "event": event,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        if self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.mode == "json":
            print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)
        elif self.mode == "text":
            detail = " ".join(f"{key}={value}" for key, value in payload.items() if value not in ("", None))
            print(f"[fault-localization] {event} {detail}".strip(), file=sys.stderr, flush=True)


def _load_or_build_code_index_for_cli(args: argparse.Namespace, reporter: ProgressReporter) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    if args.code_index:
        reporter.emit("index_loading", source="code_index", path=args.code_index)
        index = load_code_index(args.code_index)
        runtime = {
            "index_source": "code_index",
            "index_cache_hit": True,
            "index_cache_path": str(args.code_index),
            "index_runtime_seconds": round(time.perf_counter() - started, 4),
            "index_chunks": len(index.chunks),
        }
        reporter.emit("index_ready", **runtime)
        return index, runtime

    repo_path = Path(args.repo_path).expanduser().resolve()
    cache_path = _index_cache_path(repo_path, Path(args.index_cache_dir).expanduser()) if args.index_cache_dir else None
    if cache_path is not None and cache_path.exists() and not args.force_reindex:
        reporter.emit("index_cache_hit", path=str(cache_path), repo_path=str(repo_path))
        index = load_code_index(cache_path)
        runtime = {
            "index_source": "cache",
            "index_cache_hit": True,
            "index_cache_path": str(cache_path),
            "index_runtime_seconds": round(time.perf_counter() - started, 4),
            "index_chunks": len(index.chunks),
        }
        reporter.emit("index_ready", **runtime)
        return index, runtime

    reporter.emit(
        "index_build_started",
        repo_path=str(repo_path),
        cache_path=str(cache_path) if cache_path else "",
        force_reindex=bool(args.force_reindex),
    )
    index = build_code_index(repo_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(cache_path)
    runtime = {
        "index_source": "built",
        "index_cache_hit": False,
        "index_cache_path": str(cache_path) if cache_path else "",
        "index_runtime_seconds": round(time.perf_counter() - started, 4),
        "index_chunks": len(index.chunks),
    }
    reporter.emit("index_ready", **runtime)
    return index, runtime


def _ticket_runtime(index_runtime: dict[str, Any], ticket_started: float) -> dict[str, Any]:
    return {
        **index_runtime,
        "ticket_runtime_seconds": round(time.perf_counter() - ticket_started, 4),
    }


SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
}


def _index_cache_path(repo_path: Path, cache_dir: Path) -> Path:
    token = _repo_cache_token(repo_path)
    raw = json.dumps(
        {
            "repo_path": str(repo_path),
            "repo_token": token,
            "index_version": 1,
            "chunk_lines": 80,
            "overlap_lines": 20,
            "include_tests": False,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{_safe_cache_name(repo_path.name)}-{digest}.json"


def _repo_cache_token(repo_path: Path) -> str:
    git_token = _git_revision_token(repo_path)
    if git_token:
        return f"{git_token}:{_source_mtime_token(repo_path)}"
    return _source_mtime_token(repo_path)


def _git_revision_token(repo_path: Path) -> str:
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return ""
    try:
        if git_dir.is_file():
            text = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
            if text.startswith("gitdir:"):
                git_dir = (repo_path / text.split(":", maxsplit=1)[1].strip()).resolve()
        head_path = git_dir / "HEAD"
        head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
        if head.startswith("ref:"):
            ref_path = git_dir / head.split(":", maxsplit=1)[1].strip()
            if ref_path.exists():
                return f"git:{ref_path.read_text(encoding='utf-8', errors='ignore').strip()}"
        return f"git:{head}"
    except OSError:
        return ""


def _source_mtime_token(repo_path: Path) -> str:
    count = 0
    max_mtime_ns = 0
    total_size = 0
    ignored_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}
    try:
        for path in repo_path.rglob("*"):
            if any(part in ignored_dirs for part in path.parts):
                continue
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            count += 1
            try:
                stat = path.stat()
                max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
                total_size += stat.st_size
            except OSError:
                continue
    except OSError:
        return "source:unavailable"
    return f"source:count={count}:mtime_ns={max_mtime_ns}:size={total_size}"


def _safe_cache_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe or "repo"


def _read_tickets(path: str | Path) -> list[dict[str, Any]]:
    ticket_path = Path(path)
    if ticket_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with ticket_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at line {line_number}: {ticket_path}")
                rows.append(_normalize_ticket(value))
        return rows
    with ticket_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [_normalize_ticket(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [_normalize_ticket(row) for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [_normalize_ticket(value)]
    raise ValueError(f"Unsupported ticket file format: {ticket_path}")


def _normalize_ticket(row: dict[str, Any]) -> dict[str, Any]:
    ticket = dict(row)
    if "bug_report" in ticket and not any(ticket.get(key) for key in ("title", "summary", "description", "body")):
        ticket["description"] = ticket["bug_report"]
    if "json_ground_truth" in ticket and isinstance(ticket["json_ground_truth"], dict):
        for key, value in ticket["json_ground_truth"].items():
            ticket.setdefault(key, value)
    return ticket


def _repository_input_errors(repo_path: str | None, *, code_index_supplied: bool) -> list[str]:
    if code_index_supplied:
        return []
    if not repo_path:
        return ["Repository path is required when --code-index is not supplied."]
    root = Path(repo_path).expanduser()
    if not root.exists():
        return [f"Repository path does not exist: {root}"]
    if not root.is_dir():
        return [f"Repository path is not a directory: {root}"]
    return []


def _write_results(results: list[dict[str, Any]], output: str | None, *, force_jsonl: bool) -> None:
    if output is None:
        if len(results) == 1 and not force_jsonl:
            print(json.dumps(results[0], ensure_ascii=False, indent=2))
        else:
            for row in results:
                print(json.dumps(row, ensure_ascii=False))
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if force_jsonl or output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif len(results) == 1:
        output_path.write_text(json.dumps(results[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
