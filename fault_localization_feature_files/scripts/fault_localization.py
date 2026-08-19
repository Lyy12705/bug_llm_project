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
    IMPORT_GRAPH_MODES,
    CodeIndex,
    build_code_index,
    load_code_index,
    localize_ticket,
    repository_fingerprint,
)
from utils.llm_client import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline fault localization for bug tickets.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ticket", help="Single ticket JSON path.")
    input_group.add_argument("--tickets-jsonl", help="Ticket JSONL path.")
    parser.add_argument("--repo-path", help="Repository or source folder. Required when --code-index is not supplied.")
    parser.add_argument("--code-index", help="Prebuilt code index JSON from build_code_index.py.")
    parser.add_argument("--index-cache-dir", default=None, help="Optional persistent code-index cache directory.")
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild a cached index even when it is current.")
    parser.add_argument("--output", default=None, help="Output JSON/JSONL path. Prints to stdout when omitted.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of localization candidates to return.")
    parser.add_argument(
        "--candidate-file-k",
        type=int,
        default=20,
        help="Number of pre-LLM Stage-1 candidate files to preserve for evaluation.",
    )
    parser.add_argument(
        "--generic-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--domain-path-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--repository-proximity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--import-graph-mode",
        choices=IMPORT_GRAPH_MODES,
        default="off",
        help="E2 import relation mode: off, outgoing, or bidirectional.",
    )
    parser.add_argument("--min-ticket-chars", type=int, default=20, help="Sparse-ticket warning threshold.")
    parser.add_argument(
        "--no-file-aggregation",
        action="store_true",
        help="Return chunk-level candidates instead of one best candidate per file.",
    )
    parser.add_argument(
        "--advanced-file-aggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable supporting-chunk, symbol-coverage, and package-proximity bonuses.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "tfidf", "sbert", "sentence-transformers", "tfidf-sbert-rerank"),
        default="tfidf",
        help="Retrieval backend. auto uses cached sentence-transformers when available, otherwise TF-IDF.",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--semantic-candidate-k",
        type=int,
        default=50,
        help="TF-IDF candidate files represented by their best chunk and sent to SBERT.",
    )
    parser.add_argument(
        "--allow-sbert-download",
        action="store_true",
        help="Allow sentence-transformers to download the model if it is not cached locally.",
    )
    parser.add_argument("--llm-rerank", action="store_true", help="Use Ollama/Code Llama to rerank retrieved chunks.")
    parser.add_argument("--llm-candidate-k", type=int, default=10, help="Retrieval candidates sent to LLM rerank.")
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument(
        "--progress",
        choices=("none", "text", "json"),
        default="none",
        help="Emit batch progress to stderr.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Persist batch output every N new results; use 0 to disable.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed ticket IDs from an existing output file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.code_index and not args.repo_path:
        raise SystemExit("--repo-path is required when --code-index is not supplied.")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every cannot be negative.")
    if args.resume and not args.output:
        raise SystemExit("--resume requires --output.")
    if args.ollama_timeout <= 0:
        raise SystemExit("--ollama-timeout must be positive.")
    if args.llm_candidate_k <= 0:
        raise SystemExit("--llm-candidate-k must be positive.")
    if args.semantic_candidate_k <= 0:
        raise SystemExit("--semantic-candidate-k must be positive.")
    if args.candidate_file_k <= 0:
        raise SystemExit("--candidate-file-k must be positive.")
    if args.embedding_backend == "tfidf-sbert-rerank" and args.semantic_candidate_k < args.candidate_file_k:
        raise SystemExit("--semantic-candidate-k must be at least --candidate-file-k in hybrid mode.")

    reporter = ProgressReporter(args.progress)
    started = time.perf_counter()
    code_index, index_source = _load_or_build_code_index(args, reporter)
    llm_client = (
        OllamaClient(url=args.ollama_url, model=args.ollama_model, timeout=args.ollama_timeout)
        if args.llm_rerank
        else None
    )
    tickets = _read_tickets(args.ticket or args.tickets_jsonl)
    if not tickets:
        raise SystemExit("No ticket records were found in the input file.")

    results: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and args.output and Path(args.output).exists():
        input_ids = {_ticket_id(ticket) for ticket in tickets if _ticket_id(ticket)}
        results = [row for row in _read_result_records(args.output) if _ticket_id(row) in input_ids]
        completed_ids = {_ticket_id(row) for row in results if _ticket_id(row)}
        reporter.emit("resume_loaded", completed=len(completed_ids), output=args.output)

    new_results = 0
    reporter.emit("localization_started", tickets=len(tickets), index_source=index_source)
    for position, ticket in enumerate(tickets, start=1):
        ticket_id = _ticket_id(ticket)
        if ticket_id and ticket_id in completed_ids:
            reporter.emit("ticket_skipped", ticket_id=ticket_id, position=position, total=len(tickets))
            continue
        ticket_started = time.perf_counter()
        result = localize_ticket(
            ticket,
            code_index=code_index,
            top_k=args.top_k,
            embedding_backend=args.embedding_backend,
            sbert_model=args.sbert_model,
            sbert_local_files_only=not args.allow_sbert_download,
            semantic_candidate_k=args.semantic_candidate_k,
            candidate_file_k=args.candidate_file_k,
            generic_routing=args.generic_routing,
            domain_path_routing=args.domain_path_routing,
            repository_proximity=args.repository_proximity,
            import_graph_mode=args.import_graph_mode,
            llm_client=llm_client,
            llm_rerank=args.llm_rerank,
            llm_candidate_k=args.llm_candidate_k,
            file_aggregation=not args.no_file_aggregation,
            advanced_file_aggregation=args.advanced_file_aggregation,
            min_ticket_chars=args.min_ticket_chars,
        )
        result["runtime"] = {
            "localization_seconds": round(time.perf_counter() - ticket_started, 4),
            "index_source": index_source,
        }
        results.append(result)
        new_results += 1
        reporter.emit(
            "ticket_completed",
            ticket_id=ticket_id or f"row-{position}",
            position=position,
            total=len(tickets),
            confidence_level=result.get("confidence_level", ""),
        )
        if args.output and args.checkpoint_every and new_results % args.checkpoint_every == 0:
            _write_results(results, args.output, force_jsonl=bool(args.tickets_jsonl))
            reporter.emit("checkpoint_written", results=len(results), output=args.output)

    _write_results(results, args.output, force_jsonl=bool(args.tickets_jsonl))
    reporter.emit(
        "localization_completed",
        results=len(results),
        new_results=new_results,
        elapsed_seconds=round(time.perf_counter() - started, 4),
    )


def _load_or_build_code_index(args: argparse.Namespace, reporter: "ProgressReporter") -> tuple[CodeIndex, str]:
    if args.code_index:
        index = load_code_index(args.code_index)
        if args.repo_path:
            requested_root = Path(args.repo_path).resolve()
            indexed_root = Path(index.repository_path).resolve() if index.repository_path else None
            if indexed_root is not None and indexed_root != requested_root:
                raise SystemExit(
                    f"Code index repository mismatch: index={indexed_root}, requested={requested_root}."
                )
            indexed_fingerprint = str(index.settings.get("repository_fingerprint") or "")
            if indexed_fingerprint and indexed_fingerprint != repository_fingerprint(requested_root):
                raise SystemExit("Code index is stale for the requested repository; rebuild it or use --index-cache-dir.")
        reporter.emit("index_ready", source="explicit", chunks=len(index.chunks))
        return index, "explicit"

    root = Path(args.repo_path).resolve()
    current_fingerprint = repository_fingerprint(root)
    cache_path: Path | None = None
    previous_index = None
    if args.index_cache_dir:
        cache_root = Path(args.index_cache_dir).expanduser().resolve()
        cache_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        cache_path = cache_root / f"{root.name}-{cache_key}.json"
        if cache_path.exists():
            previous_index = load_code_index(cache_path)
            cached_fingerprint = str(previous_index.settings.get("repository_fingerprint") or "")
            if not args.force_reindex and cached_fingerprint == current_fingerprint:
                reporter.emit("index_ready", source="cache", chunks=len(previous_index.chunks), path=str(cache_path))
                return previous_index, "cache"

    reporter.emit("index_build_started", repository_path=str(root), incremental=previous_index is not None)
    index = build_code_index(root, previous_index=previous_index)
    if cache_path is not None:
        index.save(cache_path)
    stats = index.settings.get("index_stats", {})
    reporter.emit("index_ready", source="built", chunks=len(index.chunks), **stats)
    return index, "incremental" if previous_index is not None else "built"


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


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _read_result_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at line {line_number}: {input_path}")
            rows.append(value)
        return rows
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
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
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
    elif len(results) == 1:
        text = json.dumps(results[0], ensure_ascii=False, indent=2) + "\n"
    else:
        text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(output_path)


class ProgressReporter:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def emit(self, event: str, **details: Any) -> None:
        if self.mode == "none":
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        if self.mode == "json":
            message = json.dumps(payload, ensure_ascii=False)
        else:
            detail_text = " ".join(f"{key}={value}" for key, value in details.items())
            message = f"[{event}] {detail_text}".rstrip()
        print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
