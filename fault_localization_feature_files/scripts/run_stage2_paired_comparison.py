#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.evaluate_fault_localization import (
    _candidate_files,
    _candidate_symbols,
    _first_relevant_file_rank,
    _first_relevant_symbol_rank,
    _gold_files,
    _gold_symbols,
    _formal_prediction_symbols,
    _normalize_path,
    evaluate_records,
    read_records,
)
from utils.fault_localization import load_code_index, localize_ticket
from utils.llm_client import OllamaClient
from utils.merged_ground_truth import mapped_symbols
from utils.symbol_evaluation import SymbolEvaluationItemV1, evaluate_ranked_symbols


METRIC_NAMES = (
    "candidate_hit_at_20",
    "candidate_recall_at_20",
    "file_top_1_accuracy",
    "file_top_3_accuracy",
    "file_top_5_accuracy",
    "file_mrr",
    "symbol_top_1_accuracy",
    "symbol_top_3_accuracy",
    "symbol_top_5_accuracy",
    "symbol_mrr",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable paired Stage-2/Stage-3 Ollama comparison."
    )
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--allow-legacy-gold", action="store_true")
    parser.add_argument(
        "--selection-predictions",
        required=True,
        help="Predictions whose ticket IDs define the paired subset.",
    )
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="codellama:7b-instruct")
    parser.add_argument("--timeout", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline_predictions.jsonl"
    llm_path = output_dir / "llm_predictions.jsonl"
    summary_path = output_dir / "paired_summary.json"

    selected_ids = [
        _ticket_id(row) for row in read_records(args.selection_predictions) if _ticket_id(row)
    ]
    selected_ids = list(dict.fromkeys(selected_ids))
    tickets_by_id = {_ticket_id(row): row for row in read_records(args.tickets)}
    gold_by_id = {_ticket_id(row): row for row in read_records(args.gold)}
    missing = [ticket_id for ticket_id in selected_ids if ticket_id not in tickets_by_id]
    if missing:
        raise ValueError(f"Selected ticket IDs missing from tickets: {missing}")

    llm_by_id = {
        _ticket_id(row): row for row in read_records(llm_path) if _ticket_id(row)
    } if llm_path.exists() else {}
    client = OllamaClient(model=args.model, timeout=args.timeout)
    started = time.perf_counter()

    for position, ticket_id in enumerate(selected_ids, start=1):
        if ticket_id in llm_by_id:
            print(f"[{position}/{len(selected_ids)}] {ticket_id}: resumed", flush=True)
            continue
        ticket = tickets_by_id[ticket_id]
        index_path = _index_path(ticket, Path(args.index_dir))
        ticket_started = time.perf_counter()
        result = localize_ticket(
            ticket,
            code_index=load_code_index(index_path),
            top_k=5,
            embedding_backend="tfidf",
            candidate_file_k=20,
            domain_path_routing=True,
            llm_client=client,
            llm_rerank=True,
            llm_candidate_k=20,
            symbol_localization=True,
            symbol_llm_rerank=True,
            symbol_candidate_k=30,
            symbol_top_k=5,
        )
        result["paired_run"] = {
            "model": args.model,
            "json_schema": True,
            "batch_size": 5,
            "retrieval_score_hidden_from_llm": True,
            "runtime_seconds": round(time.perf_counter() - ticket_started, 3),
            "index_path": str(index_path),
        }
        llm_by_id[ticket_id] = result
        llm_rows = [llm_by_id[value] for value in selected_ids if value in llm_by_id]
        baseline_rows = [_baseline_from_integrated(row) for row in llm_rows]
        _atomic_write_jsonl(llm_path, llm_rows)
        _atomic_write_jsonl(baseline_path, baseline_rows)
        print(
            f"[{position}/{len(selected_ids)}] {ticket_id}: "
            f"file_llm={result['method']['llm_rerank']} "
            f"symbol_llm={result['stage3_diagnostics']['llm_rerank_used']} "
            f"warnings={len(result['warnings'])} "
            f"seconds={result['paired_run']['runtime_seconds']}",
            flush=True,
        )

    llm_rows = [llm_by_id[value] for value in selected_ids if value in llm_by_id]
    baseline_rows = [_baseline_from_integrated(row) for row in llm_rows]
    gold_rows = [gold_by_id[value] for value in selected_ids if value in gold_by_id]
    baseline_metrics = evaluate_records(
        gold_rows,
        baseline_rows,
        include_repository_breakdown=False,
        allow_legacy_gold=args.allow_legacy_gold,
    )
    llm_metrics = evaluate_records(
        gold_rows,
        llm_rows,
        include_repository_breakdown=False,
        allow_legacy_gold=args.allow_legacy_gold,
    )
    summary = {
        "tickets": len(selected_ids),
        "completed": len(llm_rows),
        "model": args.model,
        "json_schema": True,
        "batch_size": 5,
        "retrieval_score_hidden_from_llm": True,
        # Sum the persisted per-ticket runtimes so a resume-only invocation does
        # not overwrite the real model runtime with a near-zero value.
        "runtime_seconds": round(
            sum(
                float(row.get("paired_run", {}).get("runtime_seconds") or 0.0)
                for row in llm_rows
            ),
            3,
        ),
        "baseline": _selected_metrics(baseline_metrics),
        "llm": _selected_metrics(llm_metrics),
        "delta_llm_minus_baseline": {
            name: _metric_delta(name, baseline_metrics, llm_metrics)
            for name in METRIC_NAMES
        },
        "file_rank_outcomes": _paired_rank_outcomes(
            gold_rows,
            baseline_rows,
            llm_rows,
            level="file",
        ),
        "symbol_rank_outcomes": _paired_rank_outcomes(
            gold_rows,
            baseline_rows,
            llm_rows,
            level="symbol",
        ),
        "file_llm_used": sum(bool(row.get("method", {}).get("llm_rerank")) for row in llm_rows),
        "symbol_llm_used": sum(
            bool(row.get("stage3_diagnostics", {}).get("llm_rerank_used"))
            for row in llm_rows
        ),
        "predictions_with_warnings": sum(bool(row.get("warnings")) for row in llm_rows),
        "baseline_predictions": str(baseline_path),
        "llm_predictions": str(llm_path),
    }
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _baseline_from_integrated(row: dict[str, Any]) -> dict[str, Any]:
    baseline_files = [
        {"rank": rank, "file_path": str(candidate.get("file_path") or "")}
        for rank, candidate in enumerate((row.get("stage1_candidate_files") or [])[:5], start=1)
        if isinstance(candidate, dict) and candidate.get("file_path")
    ]
    return {
        "ticket_id": _ticket_id(row),
        "stage1_candidate_files": row.get("stage1_candidate_files") or [],
        "localized_files": baseline_files,
        "stage2_localized_files": baseline_files,
        "stage3_ranked_symbols": row.get("stage3_retrieval_symbols") or [],
    }


def _paired_rank_outcomes(
    gold_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    *,
    level: str,
) -> dict[str, Any]:
    gold_by_id = {_ticket_id(row): row for row in gold_rows}
    baseline_by_id = {_ticket_id(row): row for row in baseline_rows}
    outcomes = {
        "improved": [],
        "unchanged": [],
        "worsened": [],
        "both_miss": [],
        "not_evaluated": [],
    }
    for llm_row in llm_rows:
        ticket_id = _ticket_id(llm_row)
        gold = gold_by_id.get(ticket_id, {})
        baseline = baseline_by_id.get(ticket_id, {})
        if level == "file":
            gold_values = _gold_files(gold)
            def first_rank(prediction):
                if "ground_truth_schema" not in gold:
                    return _first_relevant_file_rank(_candidate_files(prediction), gold_values)
                return next((rank for rank, path in enumerate(_candidate_files(prediction), 1)
                             if _normalize_path(path) in {_normalize_path(value) for value in gold_values}), None)
            baseline_rank = first_rank(baseline)
            llm_rank = first_rank(llm_row)
        elif "ground_truth_schema" in gold:
            gold_values = [SymbolEvaluationItemV1.from_mapping(item) for item in mapped_symbols([gold])]
            baseline_symbols, _ = _formal_prediction_symbols(baseline, field="stage3_ranked_symbols")
            llm_symbols, _ = _formal_prediction_symbols(llm_row, field="stage3_ranked_symbols")
            baseline_rank = evaluate_ranked_symbols(baseline_symbols, gold_values)["exact"]["first_relevant_rank"]
            llm_rank = evaluate_ranked_symbols(llm_symbols, gold_values)["exact"]["first_relevant_rank"]
        else:
            gold_values = _gold_symbols(gold)
            baseline_rank = _first_relevant_symbol_rank(_candidate_symbols(baseline), gold_values)
            llm_rank = _first_relevant_symbol_rank(_candidate_symbols(llm_row), gold_values)
        if not gold_values:
            bucket = "not_evaluated"
        elif baseline_rank is None and llm_rank is None:
            bucket = "both_miss"
        elif llm_rank is not None and (baseline_rank is None or llm_rank < baseline_rank):
            bucket = "improved"
        elif baseline_rank is not None and (llm_rank is None or llm_rank > baseline_rank):
            bucket = "worsened"
        else:
            bucket = "unchanged"
        outcomes[bucket].append(
            {
                "ticket_id": ticket_id,
                "baseline_rank": baseline_rank,
                "llm_rank": llm_rank,
            }
        )
    return {key: {"count": len(values), "tickets": values} for key, values in outcomes.items()}


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    symbol_rows = int(metrics.get("rows_with_symbol_ground_truth", 0))
    return {
        "rows_with_file_ground_truth": metrics.get("rows_with_file_ground_truth", 0),
        "rows_with_symbol_ground_truth": symbol_rows,
        **{
            name: (
                None
                if name.startswith("symbol_") and symbol_rows == 0
                else metrics.get(name, 0.0)
            )
            for name in METRIC_NAMES
        },
    }


def _metric_delta(
    name: str,
    baseline_metrics: dict[str, Any],
    llm_metrics: dict[str, Any],
) -> float | None:
    if name.startswith("symbol_") and not int(
        baseline_metrics.get("rows_with_symbol_ground_truth", 0)
    ):
        return None
    return round(
        float(llm_metrics.get(name, 0.0)) - float(baseline_metrics.get(name, 0.0)),
        6,
    )


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or row.get("id") or "")


def _safe_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in normalized)


def _index_path(ticket: dict[str, Any], index_dir: Path) -> Path:
    repo = str(ticket.get("repo") or "")
    commit = str(ticket.get("base_commit") or "")
    path = index_dir / f"{_safe_name(repo)}__{_safe_name(commit[:16])}.json"
    if not path.exists():
        raise FileNotFoundError(f"Code index not found for {_ticket_id(ticket)}: {path}")
    return path


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
