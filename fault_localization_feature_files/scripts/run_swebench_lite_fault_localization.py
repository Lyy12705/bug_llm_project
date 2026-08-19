#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.evaluate_fault_localization import (
    discover_base_commit_existing_gold_files,
    evaluate_records,
)
from utils.fault_localization import (
    FILE_AGGREGATION_MODES,
    FILE_AGGREGATION_PARAMETERS,
    IMPORT_GRAPH_MODES,
    IMPORT_GRAPH_PARAMETERS,
    CodeIndex,
    ImportGraph,
    build_code_index,
    build_import_graph,
    load_code_index,
    localize_ticket,
)
from utils.llm_client import OllamaClient


RUNNER_INDEX_VERSION = "fault-localization-index-v2"
RUNNER_METHOD_VERSION = "fault-localization-method-v13"
FROZEN_STAGE1_METHOD_VERSION = "fault-localization-method-v11"
FROZEN_STAGE1_PROTOCOL = "stage1-v11"
FROZEN_STAGE1_METHOD_SPEC: dict[str, Any] = {
    "method_version": FROZEN_STAGE1_METHOD_VERSION,
    "embedding_backend": "tfidf",
    "sbert_model": "",
    "semantic_candidate_k": 0,
    "candidate_file_k": 20,
    "generic_routing": False,
    "domain_path_routing": True,
    "repository_proximity": False,
    "advanced_file_aggregation": False,
    "llm_rerank": False,
    "llm_candidate_k": 0,
    "ollama_model": "",
    "ollama_timeout": 0,
    "file_aggregation": True,
    "top_k": 5,
    "include_tests": False,
    "chunk_lines": 80,
    "overlap_lines": 20,
    "max_file_bytes": 500_000,
    "min_ticket_chars": 20,
}


@dataclass(slots=True)
class BatchRunConfig:
    tickets_path: Path
    gold_path: Path | None
    repo_cache_dir: Path
    snapshot_cache_dir: Path
    index_cache_dir: Path
    predictions_output: Path
    metrics_output: Path | None
    failures_output: Path | None
    manifest_output: Path | None
    import_graph_cache_dir: Path | None = None
    fallback_index_cache_dirs: tuple[Path, ...] = ()
    limit: int | None = None
    offset: int = 0
    ticket_ids: set[str] | None = None
    clone_missing: bool = False
    fetch_missing_commits: bool = False
    force_reindex: bool = False
    ephemeral_snapshots: bool = False
    include_tests: bool = False
    chunk_lines: int = 80
    overlap_lines: int = 20
    max_file_bytes: int = 500_000
    top_k: int = 5
    embedding_backend: str = "tfidf"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_local_files_only: bool = True
    semantic_candidate_k: int = 50
    candidate_file_k: int = 20
    generic_routing: bool = False
    domain_path_routing: bool = True
    repository_proximity: bool = False
    import_graph_mode: str = "off"
    advanced_file_aggregation: bool = False
    file_aggregation_mode: str | None = None
    llm_rerank: bool = False
    llm_candidate_k: int = 10
    ollama_model: str = "codellama:7b-instruct"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_timeout: int = 180
    min_ticket_chars: int = 20
    frozen_stage1_v11: bool = False
    resume: bool = False
    progress: str = "text"
    checkpoint_every: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable fault localization over prepared SWE-bench Lite tickets."
    )
    parser.add_argument("--dataset-dir", default="data/fault_localization/swebench_lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tickets", default=None, help="Override the prepared tickets JSONL path.")
    parser.add_argument("--gold", default=None, help="Override the prepared gold JSONL path.")
    parser.add_argument("--repo-cache-dir", default=None, help="Cached source repositories.")
    parser.add_argument("--snapshot-cache-dir", default=None, help="Isolated repo/base_commit snapshots.")
    parser.add_argument("--index-cache-dir", default=None, help="Per-repo/per-commit code indexes.")
    parser.add_argument(
        "--import-graph-cache-dir",
        default=None,
        help="Reusable E2 import graphs; defaults to <index-cache-dir>/import_graphs.",
    )
    parser.add_argument(
        "--fallback-index-cache-dir",
        action="append",
        default=None,
        help="Optional read-only index cache searched after the primary cache. Repeatable.",
    )
    parser.add_argument("--output-dir", default="reports/fault_localization/swebench_lite_tfidf_baseline")
    parser.add_argument("--pred-output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--failures-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ticket-id", action="append", default=None)
    parser.add_argument("--clone-missing", action="store_true")
    parser.add_argument("--fetch-missing-commits", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument(
        "--ephemeral-snapshots",
        action="store_true",
        help="Delete each checked-out base_commit snapshot after its code index is saved.",
    )
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
    parser.add_argument("--semantic-candidate-k", type=int, default=50)
    parser.add_argument("--candidate-file-k", type=int, default=20)
    parser.add_argument(
        "--generic-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable experimental identifier and generic path routing.",
    )
    parser.add_argument(
        "--domain-path-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validated high-precision domain/path routing.",
    )
    parser.add_argument(
        "--repository-proximity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias for --import-graph-mode outgoing.",
    )
    parser.add_argument(
        "--import-graph-mode",
        choices=IMPORT_GRAPH_MODES,
        default="off",
        help="E2 ablation: off, outgoing imports, or bidirectional imports.",
    )
    parser.add_argument(
        "--advanced-file-aggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable experimental supporting-chunk, symbol, and package bonuses.",
    )
    parser.add_argument(
        "--file-aggregation-mode",
        choices=FILE_AGGREGATION_MODES,
        default=None,
        help=(
            "Select one E1 ablation explicitly: basic, supporting-chunks, "
            "supporting-symbols, or supporting-symbols-package."
        ),
    )
    parser.add_argument("--allow-sbert-download", action="store_true")
    parser.add_argument("--llm-rerank", action="store_true")
    parser.add_argument("--llm-candidate-k", type=int, default=10)
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--min-ticket-chars", type=int, default=20)
    parser.add_argument(
        "--frozen-stage1-v11",
        action="store_true",
        help="Reject execution unless every model option matches the frozen Stage-1 v11 protocol.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", choices=("none", "text", "json"), default="text")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    config = BatchRunConfig(
        tickets_path=Path(args.tickets) if args.tickets else dataset_dir / f"{args.split}_tickets.jsonl",
        gold_path=Path(args.gold) if args.gold else dataset_dir / f"{args.split}_gold.jsonl",
        repo_cache_dir=Path(args.repo_cache_dir) if args.repo_cache_dir else dataset_dir / "repos",
        snapshot_cache_dir=(
            Path(args.snapshot_cache_dir) if args.snapshot_cache_dir else dataset_dir / "snapshots"
        ),
        index_cache_dir=Path(args.index_cache_dir) if args.index_cache_dir else dataset_dir / "indexes",
        import_graph_cache_dir=(
            Path(args.import_graph_cache_dir) if args.import_graph_cache_dir else None
        ),
        fallback_index_cache_dirs=tuple(Path(path) for path in (args.fallback_index_cache_dir or [])),
        predictions_output=(
            Path(args.pred_output) if args.pred_output else output_dir / f"{args.split}_predictions.jsonl"
        ),
        metrics_output=(
            Path(args.metrics_output) if args.metrics_output else output_dir / f"{args.split}_metrics.json"
        ),
        failures_output=(
            Path(args.failures_output) if args.failures_output else output_dir / f"{args.split}_failures.jsonl"
        ),
        manifest_output=(
            Path(args.manifest_output) if args.manifest_output else output_dir / f"{args.split}_run_manifest.json"
        ),
        limit=args.limit,
        offset=args.offset,
        ticket_ids=set(args.ticket_id) if args.ticket_id else None,
        clone_missing=args.clone_missing,
        fetch_missing_commits=args.fetch_missing_commits,
        force_reindex=args.force_reindex,
        ephemeral_snapshots=args.ephemeral_snapshots,
        include_tests=args.include_tests,
        chunk_lines=args.chunk_lines,
        overlap_lines=args.overlap_lines,
        max_file_bytes=args.max_file_bytes,
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
        advanced_file_aggregation=args.advanced_file_aggregation,
        file_aggregation_mode=args.file_aggregation_mode,
        llm_rerank=args.llm_rerank,
        llm_candidate_k=args.llm_candidate_k,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
        min_ticket_chars=args.min_ticket_chars,
        frozen_stage1_v11=args.frozen_stage1_v11,
        resume=args.resume,
        progress=args.progress,
        checkpoint_every=args.checkpoint_every,
    )
    summary = run_batch(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_batch(config: BatchRunConfig) -> dict[str, Any]:
    _validate_config(config)
    started_at = datetime.now(timezone.utc)
    reporter = ProgressReporter(config.progress)
    method = _method_spec(config)
    method_id = _method_id(method)
    tickets = _select_tickets(_read_jsonl(config.tickets_path), config)
    _validate_ticket_ids(tickets)
    selected_ids = {_ticket_id(ticket) for ticket in tickets}
    gold_rows = (
        [row for row in _read_jsonl(config.gold_path) if _ticket_id(row) in selected_ids]
        if config.gold_path is not None and config.gold_path.exists()
        else []
    )

    predictions_by_id: dict[str, dict[str, Any]] = {}
    failures_by_id: dict[str, dict[str, Any]] = {}
    if config.resume and config.predictions_output.exists():
        for prediction in _read_jsonl(config.predictions_output):
            ticket_id = _ticket_id(prediction)
            if ticket_id not in selected_ids:
                continue
            previous_method = str((prediction.get("benchmark_run") or {}).get("method_id") or "")
            if previous_method and previous_method != method_id:
                raise ValueError(
                    f"Cannot resume prediction {ticket_id}: method_id {previous_method} != {method_id}."
                )
            predictions_by_id[ticket_id] = prediction
    if config.resume and config.failures_output is not None and config.failures_output.exists():
        failures_by_id = {
            _ticket_id(row): row
            for row in _read_jsonl(config.failures_output)
            if _ticket_id(row) in selected_ids
        }

    llm_client = (
        OllamaClient(
            url=config.ollama_url,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
        )
        if config.llm_rerank
        else None
    )
    processed_this_run = 0
    cache_hits = 0
    import_graph_cache_hits = 0
    import_graph_builds = 0
    previous_repo = ""
    previous_index: CodeIndex | None = None
    reporter.emit("run_started", tickets=len(tickets), method_id=method_id)

    for position, ticket in enumerate(tickets, start=1):
        ticket_id = _ticket_id(ticket)
        if ticket_id in predictions_by_id:
            reporter.emit("ticket_skipped", ticket_id=ticket_id, position=position, total=len(tickets))
            continue
        ticket_started = time.perf_counter()
        try:
            repo = str(ticket.get("repo") or "")
            if repo != previous_repo:
                previous_index = None
            index, index_source = load_or_build_index(
                ticket,
                config,
                reporter,
                previous_index=previous_index,
            )
            previous_repo = repo
            previous_index = index
            if index_source.endswith("cache"):
                cache_hits += 1
            import_graph_source = "off"
            if _config_import_graph_mode(config) != "off":
                index.import_graph, import_graph_source = load_or_build_import_graph(
                    ticket,
                    index,
                    config,
                    reporter,
                )
                import_graph_cache_hits += int(import_graph_source in {"embedded", "cache"})
                import_graph_builds += int(import_graph_source == "built")
            result = localize_ticket(
                ticket,
                code_index=index,
                top_k=config.top_k,
                embedding_backend=config.embedding_backend,
                sbert_model=config.sbert_model,
                sbert_local_files_only=config.sbert_local_files_only,
                semantic_candidate_k=config.semantic_candidate_k,
                candidate_file_k=config.candidate_file_k,
                generic_routing=config.generic_routing,
                domain_path_routing=config.domain_path_routing,
                repository_proximity=config.repository_proximity,
                import_graph_mode=config.import_graph_mode,
                advanced_file_aggregation=config.advanced_file_aggregation,
                file_aggregation_mode=config.file_aggregation_mode,
                llm_client=llm_client,
                llm_rerank=config.llm_rerank,
                llm_candidate_k=config.llm_candidate_k,
                file_aggregation=True,
                min_ticket_chars=config.min_ticket_chars,
            )
            result["repo"] = str(ticket.get("repo") or "")
            result["base_commit"] = str(ticket.get("base_commit") or "")
            result["benchmark_run"] = {
                "method_id": method_id,
                "method": method,
                "frozen_protocol": FROZEN_STAGE1_PROTOCOL if config.frozen_stage1_v11 else "",
                "index_source": index_source,
                "import_graph_source": import_graph_source,
                "runtime_seconds": round(time.perf_counter() - ticket_started, 4),
            }
            predictions_by_id[ticket_id] = result
            failures_by_id.pop(ticket_id, None)
            status = "ok"
        except Exception as exc:
            failures_by_id[ticket_id] = {
                "ticket_id": ticket_id,
                "repo": str(ticket.get("repo") or ""),
                "base_commit": str(ticket.get("base_commit") or ""),
                "method_id": method_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            status = "failed"
        processed_this_run += 1
        reporter.emit(
            "ticket_completed",
            ticket_id=ticket_id,
            position=position,
            total=len(tickets),
            status=status,
        )
        if config.checkpoint_every and processed_this_run % config.checkpoint_every == 0:
            _write_checkpoints(config, tickets, predictions_by_id, failures_by_id)
            reporter.emit("checkpoint_written", processed=processed_this_run)

    predictions = [predictions_by_id[_ticket_id(ticket)] for ticket in tickets if _ticket_id(ticket) in predictions_by_id]
    failures = [failures_by_id[_ticket_id(ticket)] for ticket in tickets if _ticket_id(ticket) in failures_by_id]
    _write_checkpoints(config, tickets, predictions_by_id, failures_by_id)

    reachable_gold = (
        discover_base_commit_existing_gold_files(
            gold_rows,
            config.repo_cache_dir,
            ticket_rows=tickets,
        )
        if gold_rows
        else None
    )
    metrics = (
        evaluate_records(
            gold_rows,
            predictions,
            reachable_gold_files_by_ticket=reachable_gold,
        )
        if gold_rows
        else None
    )
    llm_used = sum(
        bool((prediction.get("method") or {}).get("llm_rerank"))
        for prediction in predictions
    )
    run_diagnostics = {
        "llm_rerank_requested": config.llm_rerank,
        "llm_rerank_used_predictions": llm_used,
        "llm_rerank_fallback_predictions": len(predictions) - llm_used if config.llm_rerank else 0,
    }
    if metrics is not None and config.metrics_output is not None:
        metrics["run_diagnostics"] = run_diagnostics
        _atomic_write_json(config.metrics_output, metrics)

    summary: dict[str, Any] = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": method_id,
        "method": method,
        "frozen_protocol": FROZEN_STAGE1_PROTOCOL if config.frozen_stage1_v11 else "",
        "tickets_selected": len(tickets),
        "predictions": len(predictions),
        "failures": len(failures),
        "processed_this_run": processed_this_run,
        "skipped_existing": len(tickets) - processed_this_run,
        "index_cache_hits": cache_hits,
        "import_graph_cache_hits": import_graph_cache_hits,
        "import_graph_builds": import_graph_builds,
        "run_diagnostics": run_diagnostics,
        "predictions_output": str(config.predictions_output),
        "metrics_output": str(config.metrics_output) if config.metrics_output else "",
        "failures_output": str(config.failures_output) if config.failures_output else "",
    }
    if metrics is not None:
        summary["metrics"] = metrics
    if config.manifest_output is not None:
        _atomic_write_json(config.manifest_output, summary)
    reporter.emit("run_completed", predictions=len(predictions), failures=len(failures), method_id=method_id)
    return summary


def load_or_build_index(
    ticket: dict[str, Any],
    config: BatchRunConfig,
    reporter: "ProgressReporter",
    *,
    previous_index: CodeIndex | None = None,
) -> tuple[CodeIndex, str]:
    index_path = _index_path(ticket, config.index_cache_dir)
    repo = str(ticket.get("repo") or "")
    base_commit = str(ticket.get("base_commit") or "")
    cache_dirs = (config.index_cache_dir, *config.fallback_index_cache_dirs)
    if not config.force_reindex:
        for cache_position, cache_dir in enumerate(cache_dirs):
            cached_path = _index_path(ticket, cache_dir)
            if not cached_path.exists():
                continue
            cached_index = load_code_index(cached_path)
            if _index_matches_ticket(cached_index, ticket, config):
                source = "cache" if cache_position == 0 else "fallback_cache"
                reporter.emit(
                    "index_ready",
                    ticket_id=_ticket_id(ticket),
                    source=source,
                    path=str(cached_path),
                )
                return cached_index, source

        # Earlier benchmark runs used the first 12 commit characters and index
        # schema v1. Their chunks can be reused read-only when the immutable
        # filename and chunk settings match this ticket.
        for cache_position, cache_dir in enumerate(cache_dirs):
            legacy_index_path = (
                cache_dir.resolve()
                / f"{_safe_name(repo)}__{_safe_name(base_commit[:12])}.json"
            )
            if not legacy_index_path.exists():
                continue
            legacy_index = load_code_index(legacy_index_path)
            if _index_chunk_settings_match(legacy_index, config):
                source = "legacy_cache" if cache_position == 0 else "fallback_legacy_cache"
                reporter.emit(
                    "index_ready",
                    ticket_id=_ticket_id(ticket),
                    source=source,
                    path=str(legacy_index_path),
                )
                return legacy_index, source

    source_repo = resolve_repository(ticket, config)
    if config.ephemeral_snapshots:
        with tempfile.TemporaryDirectory(prefix="fault-localization-snapshot-") as temporary_dir:
            snapshot = materialize_snapshot(
                source_repo,
                ticket,
                config,
                snapshot_path=Path(temporary_dir) / "snapshot",
            )
            index = _build_and_save_index(
                snapshot,
                ticket,
                config,
                index_path,
                previous_index=previous_index,
                snapshot_label="ephemeral",
            )
    else:
        snapshot = materialize_snapshot(source_repo, ticket, config)
        index = _build_and_save_index(
            snapshot,
            ticket,
            config,
            index_path,
            previous_index=previous_index,
            snapshot_label=str(snapshot),
        )
    reporter.emit("index_ready", ticket_id=_ticket_id(ticket), source="built", path=str(index_path))
    return index, "built"


def load_or_build_import_graph(
    ticket: dict[str, Any],
    index: CodeIndex,
    config: BatchRunConfig,
    reporter: "ProgressReporter",
) -> tuple[ImportGraph, str]:
    """Load an E2 sidecar graph or build it once from a legacy Code Index."""

    if index.import_graph is not None:
        reporter.emit(
            "import_graph_ready",
            ticket_id=_ticket_id(ticket),
            source="embedded",
            edges=index.import_graph.stats.get("edges", 0),
        )
        return index.import_graph, "embedded"

    graph_path = _import_graph_path(ticket, config)
    repository_fingerprint = str(index.settings.get("repository_fingerprint") or "")
    if graph_path.exists():
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("schema") == "fault-localization-import-graph-v1"
                and str(payload.get("repository_fingerprint") or "") == repository_fingerprint
                and isinstance(payload.get("graph"), dict)
            ):
                graph = ImportGraph.from_dict(payload["graph"])
                reporter.emit(
                    "import_graph_ready",
                    ticket_id=_ticket_id(ticket),
                    source="cache",
                    path=str(graph_path),
                    edges=graph.stats.get("edges", 0),
                )
                return graph, "cache"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    graph = build_import_graph(index.chunks)
    payload = {
        "schema": "fault-localization-import-graph-v1",
        "repository_fingerprint": repository_fingerprint,
        "repo": str(ticket.get("repo") or ""),
        "base_commit": str(ticket.get("base_commit") or ""),
        "graph": graph.to_dict(),
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = graph_path.with_name(f"{graph_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(graph_path)
    reporter.emit(
        "import_graph_ready",
        ticket_id=_ticket_id(ticket),
        source="built",
        path=str(graph_path),
        edges=graph.stats.get("edges", 0),
    )
    return graph, "built"


def _build_and_save_index(
    snapshot: Path,
    ticket: dict[str, Any],
    config: BatchRunConfig,
    index_path: Path,
    *,
    previous_index: CodeIndex | None,
    snapshot_label: str,
) -> CodeIndex:
    reusable_index = previous_index if previous_index and _index_chunk_settings_match(previous_index, config) else None
    index = build_code_index(
        snapshot,
        chunk_lines=config.chunk_lines,
        overlap_lines=config.overlap_lines,
        include_tests=config.include_tests,
        max_file_bytes=config.max_file_bytes,
        previous_index=reusable_index,
    )
    index.settings.update(
        {
            "runner_index_version": RUNNER_INDEX_VERSION,
            "repo": str(ticket.get("repo") or ""),
            "base_commit": str(ticket.get("base_commit") or ""),
            "snapshot_path": snapshot_label,
        }
    )
    index.save(index_path)
    return index


def _index_matches_ticket(index: CodeIndex, ticket: dict[str, Any], config: BatchRunConfig) -> bool:
    return (
        index.version == 2
        and index.settings.get("runner_index_version") == RUNNER_INDEX_VERSION
        and index.settings.get("repo") == str(ticket.get("repo") or "")
        and index.settings.get("base_commit") == str(ticket.get("base_commit") or "")
        and _index_chunk_settings_match(index, config)
    )


def _index_chunk_settings_match(index: CodeIndex, config: BatchRunConfig) -> bool:
    return (
        index.version in {1, 2}
        and index.settings.get("chunk_lines") == config.chunk_lines
        and index.settings.get("overlap_lines") == config.overlap_lines
        and index.settings.get("include_tests") == config.include_tests
        and index.settings.get("max_file_bytes") == config.max_file_bytes
    )


def resolve_repository(ticket: dict[str, Any], config: BatchRunConfig) -> Path:
    for key in ("local_repo_path", "repo_path", "repository_path"):
        value = ticket.get(key)
        if value:
            path = Path(str(value)).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Local repository path does not exist: {path}")
            return path

    repo = str(ticket.get("repo") or "").strip()
    if not repo:
        raise ValueError("Ticket does not include repo or a local repository path.")
    repo_path = (config.repo_cache_dir / _safe_name(repo)).resolve()
    if repo_path.exists():
        return repo_path
    if not config.clone_missing:
        raise FileNotFoundError(
            f"Repository cache is missing for {repo}: {repo_path}. Use --clone-missing to clone it."
        )
    repository_url = str(ticket.get("repository_url") or f"https://github.com/{repo}")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--no-checkout", repository_url, str(repo_path)], cwd=repo_path.parent)
    return repo_path


def materialize_snapshot(
    source_repo: Path,
    ticket: dict[str, Any],
    config: BatchRunConfig,
    *,
    snapshot_path: Path | None = None,
) -> Path:
    if not (source_repo / ".git").exists():
        raise ValueError(f"Repository is not a non-bare git checkout: {source_repo}")
    base_commit = str(ticket.get("base_commit") or "").strip()
    if not base_commit:
        raise ValueError("SWE-bench ticket does not include base_commit.")
    resolved_commit = _resolve_commit(source_repo, base_commit, fetch_missing=config.fetch_missing_commits)
    repo = str(ticket.get("repo") or source_repo.name)
    snapshot_path = (
        snapshot_path.resolve()
        if snapshot_path is not None
        else (config.snapshot_cache_dir / f"{_safe_name(repo)}__{resolved_commit[:16]}").resolve()
    )
    if snapshot_path.exists():
        existing_head = _git_output(["rev-parse", "HEAD"], cwd=snapshot_path, check=False)
        if existing_head == resolved_commit:
            return snapshot_path
        raise ValueError(f"Snapshot path exists at the wrong commit: {snapshot_path}")

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--shared", "--no-checkout", str(source_repo), str(snapshot_path)], cwd=snapshot_path.parent)
    _run_git(["checkout", "--detach", resolved_commit], cwd=snapshot_path)
    return snapshot_path


def _resolve_commit(repo_path: Path, base_commit: str, *, fetch_missing: bool) -> str:
    resolved = _git_output(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path, check=False)
    if not resolved and fetch_missing:
        _run_git(["fetch", "--all", "--tags"], cwd=repo_path)
        resolved = _git_output(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path, check=False)
    if not resolved:
        raise ValueError(f"base_commit is unavailable in {repo_path}: {base_commit}")
    return resolved


def _write_checkpoints(
    config: BatchRunConfig,
    tickets: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    failures_by_id: dict[str, dict[str, Any]],
) -> None:
    predictions = [predictions_by_id[_ticket_id(ticket)] for ticket in tickets if _ticket_id(ticket) in predictions_by_id]
    failures = [failures_by_id[_ticket_id(ticket)] for ticket in tickets if _ticket_id(ticket) in failures_by_id]
    _atomic_write_jsonl(config.predictions_output, predictions)
    if config.failures_output is not None:
        _atomic_write_jsonl(config.failures_output, failures)


def _method_spec(config: BatchRunConfig) -> dict[str, Any]:
    method = {
        "method_version": (
            FROZEN_STAGE1_METHOD_VERSION if config.frozen_stage1_v11 else RUNNER_METHOD_VERSION
        ),
        "embedding_backend": config.embedding_backend,
        "sbert_model": config.sbert_model if config.embedding_backend != "tfidf" else "",
        "semantic_candidate_k": (
            config.semantic_candidate_k if config.embedding_backend == "tfidf-sbert-rerank" else 0
        ),
        "candidate_file_k": config.candidate_file_k,
        "generic_routing": config.generic_routing,
        "domain_path_routing": config.domain_path_routing,
        "repository_proximity": _config_import_graph_mode(config) != "off",
        "advanced_file_aggregation": config.advanced_file_aggregation,
        "llm_rerank": config.llm_rerank,
        "llm_candidate_k": config.llm_candidate_k if config.llm_rerank else 0,
        "ollama_model": config.ollama_model if config.llm_rerank else "",
        "ollama_timeout": config.ollama_timeout if config.llm_rerank else 0,
        "file_aggregation": True,
        "top_k": config.top_k,
        "include_tests": config.include_tests,
        "chunk_lines": config.chunk_lines,
        "overlap_lines": config.overlap_lines,
        "max_file_bytes": config.max_file_bytes,
        "min_ticket_chars": config.min_ticket_chars,
    }
    if not config.frozen_stage1_v11:
        method["import_graph_mode"] = _config_import_graph_mode(config)
        method["import_graph_parameters"] = IMPORT_GRAPH_PARAMETERS
        mode = config.file_aggregation_mode or (
            "supporting-symbols-package" if config.advanced_file_aggregation else "basic"
        )
        method["advanced_file_aggregation"] = mode != "basic"
        method["file_aggregation_mode"] = mode
        method["file_aggregation_parameters"] = FILE_AGGREGATION_PARAMETERS
    return method


def _config_import_graph_mode(config: BatchRunConfig) -> str:
    if config.import_graph_mode == "off" and config.repository_proximity:
        return "outgoing"
    return config.import_graph_mode


def _method_id(method: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(method, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    llm_suffix = "-llm" if method["llm_rerank"] else ""
    return f"{method['embedding_backend']}{llm_suffix}-{digest}"


def _select_tickets(rows: list[dict[str, Any]], config: BatchRunConfig) -> list[dict[str, Any]]:
    selected = [row for row in rows if not config.ticket_ids or _ticket_id(row) in config.ticket_ids]
    if config.offset:
        selected = selected[config.offset :]
    if config.limit is not None:
        selected = selected[: config.limit]
    return selected


def _validate_config(config: BatchRunConfig) -> None:
    if config.offset < 0:
        raise ValueError("offset cannot be negative.")
    if config.limit is not None and config.limit < 0:
        raise ValueError("limit cannot be negative.")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative.")
    if (
        config.top_k <= 0
        or config.candidate_file_k <= 0
        or config.chunk_lines <= 0
        or config.max_file_bytes <= 0
    ):
        raise ValueError("top_k, candidate_file_k, chunk_lines, and max_file_bytes must be positive.")
    if (
        config.semantic_candidate_k <= 0
        or config.llm_candidate_k <= 0
        or config.ollama_timeout <= 0
        or config.min_ticket_chars <= 0
    ):
        raise ValueError(
            "semantic_candidate_k, llm_candidate_k, ollama_timeout, and min_ticket_chars must be positive."
        )
    if config.overlap_lines < 0 or config.overlap_lines >= config.chunk_lines:
        raise ValueError("overlap_lines must be non-negative and smaller than chunk_lines.")
    if (
        config.embedding_backend == "tfidf-sbert-rerank"
        and config.semantic_candidate_k < config.candidate_file_k
    ):
        raise ValueError("semantic_candidate_k must be at least candidate_file_k in hybrid mode.")
    if config.file_aggregation_mode not in (None, *FILE_AGGREGATION_MODES):
        raise ValueError(
            "file_aggregation_mode must be one of: " + ", ".join(FILE_AGGREGATION_MODES)
        )
    if config.import_graph_mode not in IMPORT_GRAPH_MODES:
        raise ValueError("import_graph_mode must be one of: " + ", ".join(IMPORT_GRAPH_MODES))
    if config.frozen_stage1_v11 and config.file_aggregation_mode is not None:
        raise ValueError("Frozen stage1-v11 does not allow file_aggregation_mode.")
    if config.frozen_stage1_v11 and _config_import_graph_mode(config) != "off":
        raise ValueError("Frozen stage1-v11 does not allow Import Graph.")
    if not config.tickets_path.exists():
        raise FileNotFoundError(f"Tickets file does not exist: {config.tickets_path}")
    if config.frozen_stage1_v11:
        actual = _method_spec(config)
        drift = {
            key: {"expected": expected, "actual": actual.get(key)}
            for key, expected in FROZEN_STAGE1_METHOD_SPEC.items()
            if actual.get(key) != expected
        }
        if drift:
            details = ", ".join(
                f"{key}={values['actual']!r} (expected {values['expected']!r})"
                for key, values in drift.items()
            )
            raise ValueError(
                f"Configuration does not match frozen {FROZEN_STAGE1_PROTOCOL}: {details}."
            )


def _validate_ticket_ids(tickets: list[dict[str, Any]]) -> None:
    ids = [_ticket_id(ticket) for ticket in tickets]
    if any(not ticket_id for ticket_id in ids):
        raise ValueError("Every benchmark ticket must include ticket_id or instance_id-normalized ticket_id.")
    duplicates = sorted(ticket_id for ticket_id in set(ids) if ids.count(ticket_id) > 1)
    if duplicates:
        raise ValueError(f"Duplicate ticket IDs are not allowed: {', '.join(duplicates[:5])}")


def _index_path(ticket: dict[str, Any], index_cache_dir: Path) -> Path:
    repo = str(ticket.get("repo") or "local")
    commit = str(ticket.get("base_commit") or "missing-commit")
    return index_cache_dir.resolve() / f"{_safe_name(repo)}__{_safe_name(commit[:16])}.json"


def _import_graph_path(ticket: dict[str, Any], config: BatchRunConfig) -> Path:
    cache_dir = config.import_graph_cache_dir or (config.index_cache_dir / "import_graphs")
    repo = str(ticket.get("repo") or "local")
    commit = str(ticket.get("base_commit") or "missing-commit")
    settings = hashlib.sha256(
        json.dumps(
            {
                "include_tests": config.include_tests,
                "max_file_bytes": config.max_file_bytes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:8]
    return cache_dir.resolve() / (
        f"{_safe_name(repo)}__{_safe_name(commit[:16])}__{settings}.json"
    )


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or row.get("id") or row.get("bug_id") or "")


def _safe_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in normalized)


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


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


def _git_output(args: list[str], *, cwd: Path, check: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip() if completed.returncode == 0 else ""


class ProgressReporter:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def emit(self, event: str, **details: Any) -> None:
        if self.mode == "none":
            return
        payload = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(), **details}
        if self.mode == "json":
            message = json.dumps(payload, ensure_ascii=False)
        else:
            message = f"[{event}] " + " ".join(f"{key}={value}" for key, value in details.items())
        print(message.rstrip(), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
