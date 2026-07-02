#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.fault_localization import build_code_index, load_code_index, localize_ticket
from utils.llm_client import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline fault localization for bug tickets.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ticket", help="Single ticket JSON path.")
    input_group.add_argument("--tickets-jsonl", help="Ticket JSONL path.")
    parser.add_argument("--repo-path", help="Repository or source folder. Required when --code-index is not supplied.")
    parser.add_argument("--code-index", help="Prebuilt code index JSON from build_code_index.py.")
    parser.add_argument("--output", default=None, help="Output JSON/JSONL path. Prints to stdout when omitted.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of localization candidates to return.")
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "tfidf", "sbert", "sentence-transformers"),
        default="tfidf",
        help="Retrieval backend. auto uses cached sentence-transformers when available, otherwise TF-IDF.",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--allow-sbert-download",
        action="store_true",
        help="Allow sentence-transformers to download the model if it is not cached locally.",
    )
    parser.add_argument("--llm-rerank", action="store_true", help="Use Ollama/Code Llama to rerank retrieved chunks.")
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.code_index and not args.repo_path:
        raise SystemExit("--repo-path is required when --code-index is not supplied.")

    code_index = load_code_index(args.code_index) if args.code_index else build_code_index(args.repo_path)
    llm_client = OllamaClient(url=args.ollama_url, model=args.ollama_model) if args.llm_rerank else None
    tickets = _read_tickets(args.ticket or args.tickets_jsonl)
    results = [
        localize_ticket(
            ticket,
            code_index=code_index,
            top_k=args.top_k,
            embedding_backend=args.embedding_backend,
            sbert_model=args.sbert_model,
            sbert_local_files_only=not args.allow_sbert_download,
            llm_client=llm_client,
            llm_rerank=args.llm_rerank,
        )
        for ticket in tickets
    ]
    _write_results(results, args.output, force_jsonl=bool(args.tickets_jsonl))


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
