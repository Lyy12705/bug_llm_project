#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_fault_localization import evaluate_records
from utils.fault_localization import CodeIndex, build_code_index, load_code_index, localize_ticket
from utils.llm_client import OllamaClient


@dataclass(slots=True)
class BatchRunConfig:
    tickets_path: Path
    gold_path: Path | None
    repo_cache_dir: Path
    index_cache_dir: Path
    predictions_output: Path
    metrics_output: Path | None
    demo_cases_output: Path | None
    failures_output: Path | None
    limit: int | None = None
    offset: int = 0
    ticket_ids: set[str] | None = None
    clone_missing: bool = False
    fetch_missing_commits: bool = False
    checkout: bool = True
    force_reindex: bool = False
    include_tests: bool = False
    chunk_lines: int = 80
    overlap_lines: int = 20
    max_file_bytes: int = 500_000
    top_k: int = 5
    embedding_backend: str = "tfidf"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_local_files_only: bool = True
    sbert_cache_dir: Path | None = None
    llm_rerank: bool = False
    llm_candidate_k: int = 10
    llm_cache_dir: Path | None = None
    ollama_model: str = "codellama:7b-instruct"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_timeout: int = 180
    demo_case_limit: int = 5
    resume: bool = False
    progress_every: int = 10
    checkpoint_every: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fault localization over prepared SWE-bench Lite tickets and evaluate predictions."
    )
    parser.add_argument("--dataset-dir", default="data/fault_localization/swebench_lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tickets", default=None, help="Override tickets JSONL path.")
    parser.add_argument("--gold", default=None, help="Override gold JSONL path.")
    parser.add_argument("--repo-cache-dir", default=None, help="Directory containing checked-out SWE-bench repositories.")
    parser.add_argument("--index-cache-dir", default=None, help="Directory for per-repo/per-commit code indexes.")
    parser.add_argument("--output-dir", default="reports/fault_localization/swebench_lite")
    parser.add_argument("--pred-output", default=None, help="Prediction JSONL output path.")
    parser.add_argument("--metrics-output", default=None, help="Metrics JSON output path.")
    parser.add_argument("--demo-cases-output", default=None, help="Small JSON file for report/demo examples.")
    parser.add_argument("--failures-output", default=None, help="JSONL output for skipped or failed rows.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ticket-id", action="append", default=None, help="Run only specific ticket id. Can be repeated.")
    parser.add_argument("--ticket-id-file", default=None, help="JSON array or newline-delimited file of ticket ids to run.")
    parser.add_argument("--clone-missing", action="store_true", help="Clone missing repositories into --repo-cache-dir.")
    parser.add_argument("--fetch-missing-commits", action="store_true", help="Fetch from origin when a base commit is missing.")
    parser.add_argument("--no-checkout", action="store_true", help="Use the current repo working tree instead of checking out base_commit.")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--chunk-lines", type=int, default=80)
    parser.add_argument("--overlap-lines", type=int, default=20)
    parser.add_argument("--max-file-bytes", type=int, default=500_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "tfidf", "sbert", "sentence-transformers", "tfidf-sbert-rerank"),
        default="tfidf",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow-sbert-download", action="store_true")
    parser.add_argument(
        "--sbert-cache-dir",
        default=None,
        help="Directory for persistent SBERT embedding cache. Defaults to dataset_dir/embedding_cache.",
    )
    parser.add_argument("--llm-rerank", action="store_true")
    parser.add_argument(
        "--llm-candidate-k",
        type=int,
        default=10,
        help="Number of retrieval candidates sent to the optional LLM reranker.",
    )
    parser.add_argument(
        "--llm-cache-dir",
        default=None,
        help="Directory for optional LLM rerank response cache. Defaults to output_dir/llm_rerank_cache.",
    )
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--demo-case-limit", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing successful predictions and continue unfinished tickets.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress to stderr every N processed tickets. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write partial predictions/failures every N processed tickets. Use 0 to write only at the end.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    config = BatchRunConfig(
        tickets_path=Path(args.tickets) if args.tickets else dataset_dir / f"{args.split}_tickets.jsonl",
        gold_path=Path(args.gold) if args.gold else dataset_dir / f"{args.split}_gold.jsonl",
        repo_cache_dir=Path(args.repo_cache_dir) if args.repo_cache_dir else dataset_dir / "repos",
        index_cache_dir=Path(args.index_cache_dir) if args.index_cache_dir else dataset_dir / "indexes",
        predictions_output=Path(args.pred_output) if args.pred_output else output_dir / f"{args.split}_predictions.jsonl",
        metrics_output=Path(args.metrics_output) if args.metrics_output else output_dir / f"{args.split}_metrics.json",
        demo_cases_output=Path(args.demo_cases_output) if args.demo_cases_output else output_dir / f"{args.split}_demo_cases.json",
        failures_output=Path(args.failures_output) if args.failures_output else output_dir / f"{args.split}_failures.jsonl",
        limit=args.limit,
        offset=args.offset,
        ticket_ids=_ticket_id_filter(args.ticket_id, args.ticket_id_file),
        clone_missing=args.clone_missing,
        fetch_missing_commits=args.fetch_missing_commits,
        checkout=not args.no_checkout,
        force_reindex=args.force_reindex,
        include_tests=args.include_tests,
        chunk_lines=args.chunk_lines,
        overlap_lines=args.overlap_lines,
        max_file_bytes=args.max_file_bytes,
        top_k=args.top_k,
        embedding_backend=args.embedding_backend,
        sbert_model=args.sbert_model,
        sbert_local_files_only=not args.allow_sbert_download,
        sbert_cache_dir=Path(args.sbert_cache_dir) if args.sbert_cache_dir else dataset_dir / "embedding_cache",
        llm_rerank=args.llm_rerank,
        llm_candidate_k=args.llm_candidate_k,
        llm_cache_dir=Path(args.llm_cache_dir) if args.llm_cache_dir else output_dir / "llm_rerank_cache",
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
        demo_case_limit=args.demo_case_limit,
        resume=args.resume,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
    )
    summary = run_batch(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_batch(config: BatchRunConfig) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    config.repo_cache_dir.mkdir(parents=True, exist_ok=True)
    config.index_cache_dir.mkdir(parents=True, exist_ok=True)
    tickets = _select_tickets(_read_jsonl(config.tickets_path), config)
    gold_rows = _read_jsonl(config.gold_path) if config.gold_path and config.gold_path.exists() else []
    llm_client = (
        OllamaClient(url=config.ollama_url, model=config.ollama_model, timeout=config.ollama_timeout)
        if config.llm_rerank
        else None
    )
    predictions: list[dict[str, Any]] = _read_jsonl(config.predictions_output) if config.resume and config.predictions_output.exists() else []
    completed_ticket_ids = {_ticket_id(row) for row in predictions if _ticket_id(row)}
    failures: list[dict[str, Any]] = []
    skipped_existing = 0
    processed_this_run = 0

    for row_number, ticket in enumerate(tickets, start=1):
        ticket_id = _ticket_id(ticket) or f"row-{row_number}"
        if config.resume and ticket_id in completed_ticket_ids:
            skipped_existing += 1
            if _should_report_progress(config, row_number):
                _report_progress(
                    row_number=row_number,
                    total=len(tickets),
                    ticket_id=ticket_id,
                    status="skipped_existing",
                    predictions=len(predictions),
                    failures=len(failures),
                )
            continue
        try:
            repo_path = resolve_repository(ticket, config)
            if config.checkout:
                checkout_repository(
                    repo_path,
                    str(ticket.get("base_commit") or ""),
                    fetch_missing=config.fetch_missing_commits,
                )
            index = load_or_build_index(ticket, repo_path, config)
            result = localize_ticket(
                ticket,
                code_index=index,
                top_k=config.top_k,
                embedding_backend=config.embedding_backend,
                sbert_model=config.sbert_model,
                sbert_local_files_only=config.sbert_local_files_only,
                sbert_cache_dir=config.sbert_cache_dir,
                llm_client=llm_client,
                llm_rerank=config.llm_rerank,
                llm_candidate_k=config.llm_candidate_k,
                llm_cache_dir=config.llm_cache_dir,
            )
            result["repo"] = str(ticket.get("repo") or "")
            result["base_commit"] = str(ticket.get("base_commit") or "")
            predictions.append(result)
            completed_ticket_ids.add(ticket_id)
            status = "ok"
        except Exception as exc:
            failures.append(
                {
                    "ticket_id": ticket_id,
                    "repo": str(ticket.get("repo") or ""),
                    "base_commit": str(ticket.get("base_commit") or ""),
                    "error": str(exc),
                }
            )
            status = "failed"
        processed_this_run += 1
        if _should_report_progress(config, row_number):
            _report_progress(
                row_number=row_number,
                total=len(tickets),
                ticket_id=ticket_id,
                status=status,
                predictions=len(predictions),
                failures=len(failures),
            )
        if config.checkpoint_every > 0 and processed_this_run % config.checkpoint_every == 0:
            _write_partial_outputs(config, predictions, failures)

    _write_partial_outputs(config, predictions, failures)

    metrics: dict[str, Any] | None = None
    if gold_rows and config.metrics_output is not None:
        metrics = evaluate_records(gold_rows, predictions)
        config.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if gold_rows and config.demo_cases_output is not None:
        demo_cases = build_demo_cases(predictions, gold_rows, limit=config.demo_case_limit)
        config.demo_cases_output.parent.mkdir(parents=True, exist_ok=True)
        config.demo_cases_output.write_text(json.dumps(demo_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickets_input": str(config.tickets_path),
        "gold_input": str(config.gold_path) if config.gold_path else "",
        "tickets_selected": len(tickets),
        "predictions": len(predictions),
        "failures": len(failures),
        "skipped_existing_predictions": skipped_existing,
        "processed_this_run": processed_this_run,
        "predictions_output": str(config.predictions_output),
        "metrics_output": str(config.metrics_output) if config.metrics_output else "",
        "demo_cases_output": str(config.demo_cases_output) if config.demo_cases_output else "",
        "failures_output": str(config.failures_output) if config.failures_output else "",
        "method": {
            "top_k": config.top_k,
            "embedding_backend": config.embedding_backend,
            "sbert_cache_dir": str(config.sbert_cache_dir) if config.sbert_cache_dir else "",
            "llm_rerank": config.llm_rerank,
            "llm_candidate_k": config.llm_candidate_k if config.llm_rerank else 0,
            "llm_cache_dir": str(config.llm_cache_dir) if config.llm_rerank and config.llm_cache_dir else "",
            "ollama_model": config.ollama_model if config.llm_rerank else "",
            "ollama_timeout": config.ollama_timeout if config.llm_rerank else 0,
            "include_tests": config.include_tests,
            "resume": config.resume,
            "progress_every": config.progress_every,
            "checkpoint_every": config.checkpoint_every,
        },
    }
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def _write_partial_outputs(config: BatchRunConfig, predictions: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    _write_jsonl(config.predictions_output, predictions)
    if config.failures_output is not None:
        _write_jsonl(config.failures_output, failures)


def _should_report_progress(config: BatchRunConfig, row_number: int) -> bool:
    return config.progress_every > 0 and (row_number == 1 or row_number % config.progress_every == 0)


def _report_progress(
    *,
    row_number: int,
    total: int,
    ticket_id: str,
    status: str,
    predictions: int,
    failures: int,
) -> None:
    payload = {
        "event": "fault_localization_progress",
        "row": row_number,
        "total": total,
        "ticket_id": ticket_id,
        "status": status,
        "predictions": predictions,
        "failures": failures,
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def resolve_repository(ticket: dict[str, Any], config: BatchRunConfig) -> Path:
    for key in ("local_repo_path", "repo_path", "repository_path"):
        raw_path = ticket.get(key)
        if raw_path:
            path = Path(str(raw_path)).expanduser().resolve()
            if path.exists():
                return path
    repo = str(ticket.get("repo") or "").strip()
    if not repo:
        raise ValueError("Ticket does not include repo, local_repo_path, repo_path, or repository_path.")
    repo_path = (config.repo_cache_dir / _safe_repo_name(repo)).expanduser().resolve()
    if repo_path.exists():
        return repo_path.resolve()
    if not config.clone_missing:
        raise FileNotFoundError(
            f"Repository cache not found for {repo}: {repo_path}. "
            "Run with --clone-missing or add local_repo_path to the ticket JSONL."
        )
    repository_url = str(ticket.get("repository_url") or f"https://github.com/{repo}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["git", "clone", repository_url, str(repo_path)], cwd=repo_path.parent)
    return repo_path.resolve()


def checkout_repository(repo_path: Path, base_commit: str, *, fetch_missing: bool = False) -> None:
    if not base_commit:
        return
    if not (repo_path / ".git").exists():
        raise ValueError(f"Cannot checkout base_commit because this is not a git repository: {repo_path}")
    verify = _run_command(["git", "rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path, check=False)
    if verify.returncode != 0 and fetch_missing:
        _run_command(["git", "fetch", "--all", "--tags"], cwd=repo_path)
        verify = _run_command(["git", "rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path, check=False)
    if verify.returncode != 0:
        raise ValueError(f"base_commit is not available in {repo_path}: {base_commit}")
    _run_command(["git", "checkout", "--detach", base_commit], cwd=repo_path)


def load_or_build_index(ticket: dict[str, Any], repo_path: Path, config: BatchRunConfig) -> CodeIndex:
    index_path = _index_path(ticket, repo_path, config.index_cache_dir)
    if index_path.exists() and not config.force_reindex:
        return load_code_index(index_path)
    index = build_code_index(
        repo_path,
        chunk_lines=config.chunk_lines,
        overlap_lines=config.overlap_lines,
        include_tests=config.include_tests,
        max_file_bytes=config.max_file_bytes,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    return index


def build_demo_cases(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        ticket_id = _ticket_id(prediction)
        gold = gold_by_id.get(ticket_id, {})
        gold_files = _file_values(gold)
        candidates = prediction.get("localized_candidates") if isinstance(prediction.get("localized_candidates"), list) else []
        top_candidates: list[dict[str, Any]] = []
        hit_rank: int | None = None
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            file_path = str(candidate.get("file_path") or "")
            is_gold_match = any(_file_matches(file_path, gold_file) for gold_file in gold_files)
            if is_gold_match and hit_rank is None:
                hit_rank = rank
            top_candidates.append(
                {
                    "rank": rank,
                    "file_path": file_path,
                    "function_name": candidate.get("function_name") or candidate.get("class_name") or "",
                    "start_line": candidate.get("start_line"),
                    "end_line": candidate.get("end_line"),
                    "score": candidate.get("score"),
                    "is_gold_match": is_gold_match,
                    "reason": candidate.get("reason", ""),
                }
            )
        cases.append(
            {
                "ticket_id": ticket_id,
                "repo": prediction.get("repo", ""),
                "gold_files": gold_files,
                "hit_rank": hit_rank,
                "top_k_hit": hit_rank is not None,
                "bug_report_preview": str(prediction.get("bug_report") or "")[:800],
                "top_candidates": top_candidates,
            }
        )
        if len(cases) >= limit:
            break
    return cases


def _select_tickets(rows: list[dict[str, Any]], config: BatchRunConfig) -> list[dict[str, Any]]:
    selected = rows
    if config.ticket_ids:
        selected = [row for row in selected if _ticket_id(row) in config.ticket_ids]
    if config.offset:
        selected = selected[config.offset :]
    if config.limit is not None:
        selected = selected[: config.limit]
    return selected


def _ticket_id_filter(ticket_ids: list[str] | None, ticket_id_file: str | None) -> set[str] | None:
    values = set(ticket_ids or [])
    if ticket_id_file:
        path = Path(ticket_id_file)
        text = path.read_text(encoding="utf-8").strip()
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                values.update(str(item) for item in payload if str(item).strip())
            else:
                values.update(line.strip() for line in text.splitlines() if line.strip())
    return values or None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_command(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed in {cwd}: {' '.join(command)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    return completed


def _index_path(ticket: dict[str, Any], repo_path: Path, index_cache_dir: Path) -> Path:
    repo = str(ticket.get("repo") or repo_path.name)
    commit = str(ticket.get("base_commit") or "working-tree")
    return index_cache_dir.expanduser().resolve() / f"{_safe_repo_name(repo)}__{commit[:12]}.json"


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _file_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths", "file_path", "file"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    normalized: list[str] = []
    for value in values:
        path = value.replace("\\", "/").lstrip("./")
        if path and path not in normalized:
            normalized.append(path)
    return normalized


def _file_matches(predicted: str, gold: str) -> bool:
    left = predicted.replace("\\", "/").lstrip("./")
    right = gold.replace("\\", "/").lstrip("./")
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _safe_repo_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
