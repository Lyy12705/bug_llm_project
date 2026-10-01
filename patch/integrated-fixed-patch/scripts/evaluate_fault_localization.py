#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.symbol_evaluation import (  # noqa: E402
    SYMBOL_EVALUATION_K_VALUES,
    SymbolEvaluationItemV1,
    evaluate_ranked_symbols,
    evaluation_contract,
)


CANDIDATE_K = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fault localization predictions.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL with fixed files or modified files.")
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL from scripts/fault_localization.py.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    parser.add_argument(
        "--repo-cache-dir",
        default=None,
        help="Optional repository cache used to report Recall@20 only for files present at base_commit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    reachable = (
        discover_base_commit_existing_gold_files(gold_rows, Path(args.repo_cache_dir))
        if args.repo_cache_dir
        else None
    )
    metrics = evaluate_records(
        gold_rows,
        pred_rows,
        reachable_gold_files_by_ticket=reachable,
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def evaluate_records(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    include_repository_breakdown: bool = True,
    reachable_gold_files_by_ticket: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    pairs = _align_rows(gold_rows, pred_rows)
    file_evaluated = 0
    symbol_evaluated = 0
    file_hits_at = {1: 0, 3: 0, 5: 0}
    symbol_hits_at = {1: 0, 3: 0, 5: 0}
    candidate_hits = 0
    candidate_recall_total = 0.0
    candidate_count_total = 0
    rows_with_stage1_candidates = 0
    rows_missing_stage1_output = 0
    candidate_outcomes = {"full_recall_rows": 0, "partial_recall_rows": 0, "miss_rows": 0}
    candidate_outcome_ticket_ids = {
        "full_recall": [],
        "partial_recall": [],
        "miss": [],
    }
    candidate_hit_values: list[float] = []
    candidate_recall_values: list[float] = []
    reachable_file_evaluated = 0
    reachable_candidate_hits = 0
    reachable_candidate_recall_total = 0.0
    reachable_hit_values: list[float] = []
    reachable_recall_values: list[float] = []
    unreachable_gold_files_by_ticket: dict[str, list[str]] = {}
    file_reciprocal_rank_total = 0.0
    symbol_reciprocal_rank_total = 0.0
    missing_file_ground_truth: list[str] = []
    missing_symbol_ground_truth: list[str] = []

    for gold, pred in pairs:
        gold_files = _gold_files(gold)
        gold_symbols = _gold_symbols(gold)
        ticket_id = _ticket_id(gold) or _ticket_id(pred)
        if gold_files:
            file_evaluated += 1
            file_rank = _first_relevant_file_rank(_candidate_files(pred), gold_files)
            if file_rank is not None:
                file_reciprocal_rank_total += 1.0 / file_rank
                for k in file_hits_at:
                    if file_rank <= k:
                        file_hits_at[k] += 1

            stage1_files = _stage1_candidate_files(pred)[:CANDIDATE_K]
            candidate_count_total += len(stage1_files)
            if stage1_files:
                rows_with_stage1_candidates += 1
            if "stage1_candidate_files" not in pred and "stage1_candidates" not in pred:
                rows_missing_stage1_output += 1
            recovered_count = sum(
                1
                for gold_file in gold_files
                if any(_file_matches(predicted, gold_file) for predicted in stage1_files)
            )
            recall = recovered_count / len(gold_files)
            candidate_recall_total += recall
            candidate_recall_values.append(recall)
            if recovered_count:
                candidate_hits += 1
                candidate_hit_values.append(1.0)
            else:
                candidate_hit_values.append(0.0)
            if recovered_count == len(gold_files):
                candidate_outcomes["full_recall_rows"] += 1
                candidate_outcome_ticket_ids["full_recall"].append(ticket_id)
            elif recovered_count:
                candidate_outcomes["partial_recall_rows"] += 1
                candidate_outcome_ticket_ids["partial_recall"].append(ticket_id)
            else:
                candidate_outcomes["miss_rows"] += 1
                candidate_outcome_ticket_ids["miss"].append(ticket_id)

            if reachable_gold_files_by_ticket is not None and ticket_id in reachable_gold_files_by_ticket:
                reachable_files = _unique_paths(reachable_gold_files_by_ticket[ticket_id])
                reachable_set = {_normalize_path(path) for path in reachable_files}
                unreachable = [
                    path for path in gold_files if _normalize_path(path) not in reachable_set
                ]
                if unreachable:
                    unreachable_gold_files_by_ticket[ticket_id] = unreachable
                if reachable_files:
                    reachable_file_evaluated += 1
                    reachable_recovered = sum(
                        1
                        for gold_file in reachable_files
                        if any(_file_matches(predicted, gold_file) for predicted in stage1_files)
                    )
                    reachable_recall = reachable_recovered / len(reachable_files)
                    reachable_candidate_recall_total += reachable_recall
                    reachable_recall_values.append(reachable_recall)
                    reachable_hit = 1.0 if reachable_recovered else 0.0
                    reachable_candidate_hits += int(reachable_hit)
                    reachable_hit_values.append(reachable_hit)
        else:
            missing_file_ground_truth.append(ticket_id)

        if gold_symbols:
            symbol_evaluated += 1
            symbol_rank = _first_relevant_symbol_rank(_candidate_symbols(pred), gold_symbols)
            if symbol_rank is not None:
                symbol_reciprocal_rank_total += 1.0 / symbol_rank
                for k in symbol_hits_at:
                    if symbol_rank <= k:
                        symbol_hits_at[k] += 1
        else:
            missing_symbol_ground_truth.append(ticket_id)

    metrics: dict[str, Any] = {
        "rows": len(pairs),
        "gold_rows": len(gold_rows),
        "prediction_rows": len(pred_rows),
        "matched_prediction_rows": sum(1 for _, pred in pairs if pred),
        "missing_prediction_rows": sum(1 for _, pred in pairs if not pred),
        "rows_with_file_ground_truth": file_evaluated,
        "rows_with_symbol_ground_truth": symbol_evaluated,
        "candidate_k": CANDIDATE_K,
        "rows_with_stage1_candidates": rows_with_stage1_candidates,
        "rows_missing_stage1_output": rows_missing_stage1_output,
        "candidate_hit_at_20": _ratio(candidate_hits, file_evaluated),
        "candidate_recall_at_20": _ratio(candidate_recall_total, file_evaluated),
        "average_candidate_count_at_20": _ratio(candidate_count_total, file_evaluated),
        "candidate_outcomes_at_20": candidate_outcomes,
        "candidate_outcome_ticket_ids_at_20": candidate_outcome_ticket_ids,
        "candidate_bootstrap_95_ci": {
            "samples": 2_000,
            "seed": 42,
            "candidate_hit_at_20": _bootstrap_mean_ci(candidate_hit_values, seed=42),
            "candidate_recall_at_20": _bootstrap_mean_ci(candidate_recall_values, seed=43),
        },
        "file_top_1_accuracy": _ratio(file_hits_at[1], file_evaluated),
        "file_top_3_accuracy": _ratio(file_hits_at[3], file_evaluated),
        "file_top_5_accuracy": _ratio(file_hits_at[5], file_evaluated),
        "file_mrr": _ratio(file_reciprocal_rank_total, file_evaluated),
        "symbol_top_1_accuracy": _ratio(symbol_hits_at[1], symbol_evaluated),
        "symbol_top_3_accuracy": _ratio(symbol_hits_at[3], symbol_evaluated),
        "symbol_top_5_accuracy": _ratio(symbol_hits_at[5], symbol_evaluated),
        "symbol_mrr": _ratio(symbol_reciprocal_rank_total, symbol_evaluated),
    }
    metrics["rows_with_ground_truth"] = file_evaluated
    metrics["top_1_accuracy"] = metrics["file_top_1_accuracy"]
    metrics["top_3_accuracy"] = metrics["file_top_3_accuracy"]
    metrics["top_5_accuracy"] = metrics["file_top_5_accuracy"]
    metrics["mrr"] = metrics["file_mrr"]
    if reachable_gold_files_by_ticket is not None:
        metrics["base_commit_reachable_evaluation"] = {
            "rows_with_reachable_file_ground_truth": reachable_file_evaluated,
            "candidate_hit_at_20": _ratio(reachable_candidate_hits, reachable_file_evaluated),
            "candidate_recall_at_20": _ratio(
                reachable_candidate_recall_total,
                reachable_file_evaluated,
            ),
            "unreachable_gold_file_count": sum(
                len(paths) for paths in unreachable_gold_files_by_ticket.values()
            ),
            "unreachable_gold_files_by_ticket": unreachable_gold_files_by_ticket,
            "bootstrap_95_ci": {
                "samples": 2_000,
                "seed": 44,
                "candidate_hit_at_20": _bootstrap_mean_ci(reachable_hit_values, seed=44),
                "candidate_recall_at_20": _bootstrap_mean_ci(reachable_recall_values, seed=45),
            },
            "note": (
                "Auxiliary metric only. Gold files absent at base_commit are excluded; "
                "the original Recall@20 remains the primary benchmark metric."
            ),
        }
    if missing_file_ground_truth or missing_symbol_ground_truth:
        metrics["missing_file_ground_truth_rows"] = len(missing_file_ground_truth)
        metrics["missing_symbol_ground_truth_rows"] = len(missing_symbol_ground_truth)
        metrics["note"] = (
            "Rows without fixed_files/modified_files/file ground truth are skipped for file-level metrics. "
            "Rows without fixed_symbols/function_name/class_name ground truth are skipped for symbol-level metrics."
        )

    formal_symbol_metrics = _evaluate_formal_symbol_records(gold_rows, pred_rows)
    if formal_symbol_metrics is not None:
        metrics["symbol_evaluation_contract"] = evaluation_contract()
        metrics["formal_symbol_evaluation"] = formal_symbol_metrics
        metrics["rows_with_symbol_ground_truth"] = formal_symbol_metrics["eligible_rows"]
        metrics["symbol_top_1_accuracy"] = formal_symbol_metrics["exact"]["hit_at_1"]
        metrics["symbol_top_3_accuracy"] = formal_symbol_metrics["exact"]["hit_at_3"]
        metrics["symbol_top_5_accuracy"] = formal_symbol_metrics["exact"]["hit_at_5"]
        metrics["symbol_mrr"] = formal_symbol_metrics["exact"]["mrr"]
    if include_repository_breakdown:
        repository_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for gold, pred in pairs:
            repository = _repository_name(gold) or _repository_name(pred) or "unknown"
            repository_rows.setdefault(repository, []).append((gold, pred))
        if len(repository_rows) > 1:
            metrics["per_repository"] = {}
            for repository, rows in sorted(repository_rows.items()):
                repository_metrics = evaluate_records(
                    [gold for gold, _ in rows],
                    [pred for _, pred in rows],
                    include_repository_breakdown=False,
                    reachable_gold_files_by_ticket=reachable_gold_files_by_ticket,
                )
                metrics["per_repository"][repository] = {
                    key: repository_metrics[key]
                    for key in (
                        "rows",
                        "candidate_hit_at_20",
                        "candidate_recall_at_20",
                        "average_candidate_count_at_20",
                        "candidate_outcomes_at_20",
                        "candidate_outcome_ticket_ids_at_20",
                        "candidate_bootstrap_95_ci",
                        "file_top_1_accuracy",
                        "file_top_3_accuracy",
                        "file_top_5_accuracy",
                        "file_mrr",
                    )
                }
                if "base_commit_reachable_evaluation" in repository_metrics:
                    metrics["per_repository"][repository]["base_commit_reachable_evaluation"] = (
                        repository_metrics["base_commit_reachable_evaluation"]
                    )
    return metrics


def _evaluate_formal_symbol_records(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Evaluate SymbolGoldRecordV1 rows once per ticket using the formal contract."""

    gold_by_ticket: dict[str, list[SymbolEvaluationItemV1]] = {}
    formal_input_seen = False
    for row in gold_rows:
        ticket_id = _ticket_id(row)
        if not ticket_id:
            continue
        candidates: list[dict[str, Any]] = []
        if "mapping_status" in row:
            formal_input_seen = True
            if str(row.get("mapping_status") or "").strip().casefold() == "mapped":
                candidates.append(row)
        for key in ("fixed_symbol_records", "symbol_gold_records"):
            nested = row.get(key)
            if isinstance(nested, list):
                formal_input_seen = True
                candidates.extend(item for item in nested if isinstance(item, dict))
        for candidate in candidates:
            try:
                item = SymbolEvaluationItemV1.from_mapping(candidate)
            except (TypeError, ValueError):
                continue
            if item not in gold_by_ticket.setdefault(ticket_id, []):
                gold_by_ticket[ticket_id].append(item)

    if not formal_input_seen:
        return None

    pred_by_ticket = {_ticket_id(row): row for row in pred_rows if _ticket_id(row)}
    totals = {
        mode: {
            "hit": {k: 0 for k in SYMBOL_EVALUATION_K_VALUES},
            "recall": {k: 0.0 for k in SYMBOL_EVALUATION_K_VALUES},
            "reciprocal_rank": 0.0,
        }
        for mode in ("exact", "relaxed_diagnostic")
    }
    conditional_totals = {
        "hit": {k: 0 for k in SYMBOL_EVALUATION_K_VALUES},
        "recall": {k: 0.0 for k in SYMBOL_EVALUATION_K_VALUES},
        "reciprocal_rank": 0.0,
    }
    candidate_totals = {
        "hit": {10: 0, 30: 0},
        "recall": {10: 0.0, 30: 0.0},
        "reciprocal_rank": 0.0,
    }
    baseline_totals = {
        "hit": {k: 0 for k in SYMBOL_EVALUATION_K_VALUES},
        "recall": {k: 0.0 for k in SYMBOL_EVALUATION_K_VALUES},
        "reciprocal_rank": 0.0,
    }
    paired_differences: dict[str, list[float]] = {
        **{f"hit_at_{k}": [] for k in SYMBOL_EVALUATION_K_VALUES},
        **{f"recall_at_{k}": [] for k in SYMBOL_EVALUATION_K_VALUES},
        "mrr": [],
    }
    paired_outcomes = {
        f"hit_at_{k}": {
            "improved": [],
            "unchanged": [],
            "worsened": [],
            "both_miss": [],
        }
        for k in SYMBOL_EVALUATION_K_VALUES
    }
    rank_outcomes = {
        "improved": [],
        "unchanged": [],
        "worsened": [],
        "both_miss": [],
    }
    missing_prediction_ticket_ids: list[str] = []
    missing_symbol_output_ticket_ids: list[str] = []
    missing_candidate_output_ticket_ids: list[str] = []
    invalid_prediction_symbol_count = 0
    invalid_candidate_symbol_count = 0
    invalid_retrieval_symbol_count = 0
    conditional_ticket_ids: list[str] = []
    coverage_counts = {
        "prediction_rows": 0,
        "final_output_rows": 0,
        "candidate_output_rows": 0,
        "retrieval_output_rows": 0,
        "llm_requested_rows": 0,
        "llm_eligible_rows": 0,
        "llm_attempted_rows": 0,
        "llm_valid_rows": 0,
        "fallback_rows": 0,
        "timeout_rows": 0,
    }
    fallback_reasons: dict[str, int] = {}
    per_ticket: list[dict[str, Any]] = []

    for ticket_id, gold_symbols in sorted(gold_by_ticket.items()):
        pred = pred_by_ticket.get(ticket_id, {})
        if not pred:
            missing_prediction_ticket_ids.append(ticket_id)
        else:
            coverage_counts["prediction_rows"] += 1
        ranked_symbols, invalid_count = _formal_prediction_symbols(
            pred,
            field="stage3_ranked_symbols",
        )
        retrieval_symbols, retrieval_invalid = _formal_prediction_symbols(
            pred,
            field="stage3_retrieval_symbols",
        )
        candidate_symbols, candidate_invalid = _formal_prediction_symbols(
            pred,
            field="stage3_candidate_symbols",
        )
        invalid_prediction_symbol_count += invalid_count
        invalid_retrieval_symbol_count += retrieval_invalid
        invalid_candidate_symbol_count += candidate_invalid
        if not ranked_symbols:
            missing_symbol_output_ticket_ids.append(ticket_id)
        else:
            coverage_counts["final_output_rows"] += 1
        if retrieval_symbols:
            coverage_counts["retrieval_output_rows"] += 1
        result = evaluate_ranked_symbols(ranked_symbols, gold_symbols)
        baseline_result = evaluate_ranked_symbols(retrieval_symbols, gold_symbols)
        for mode in ("exact", "relaxed_diagnostic"):
            for k in SYMBOL_EVALUATION_K_VALUES:
                totals[mode]["hit"][k] += result[mode]["hit_at"][str(k)]
                totals[mode]["recall"][k] += result[mode]["recall_at"][str(k)]
            totals[mode]["reciprocal_rank"] += result[mode]["reciprocal_rank"]
        for k in SYMBOL_EVALUATION_K_VALUES:
            baseline_totals["hit"][k] += baseline_result["exact"]["hit_at"][str(k)]
            baseline_totals["recall"][k] += baseline_result["exact"]["recall_at"][str(k)]
            final_hit = result["exact"]["hit_at"][str(k)]
            baseline_hit = baseline_result["exact"]["hit_at"][str(k)]
            paired_differences[f"hit_at_{k}"].append(float(final_hit - baseline_hit))
            paired_differences[f"recall_at_{k}"].append(
                result["exact"]["recall_at"][str(k)]
                - baseline_result["exact"]["recall_at"][str(k)]
            )
            paired_outcomes[f"hit_at_{k}"][
                _paired_hit_bucket(baseline_hit, final_hit)
            ].append(ticket_id)
        baseline_totals["reciprocal_rank"] += baseline_result["exact"]["reciprocal_rank"]
        paired_differences["mrr"].append(
            result["exact"]["reciprocal_rank"]
            - baseline_result["exact"]["reciprocal_rank"]
        )
        rank_outcomes[
            _paired_rank_bucket(
                baseline_result["exact"]["first_relevant_rank"],
                result["exact"]["first_relevant_rank"],
            )
        ].append(ticket_id)

        gold_files = [item.file_path for item in gold_symbols]
        conditional_eligible = any(
            _file_matches(predicted, gold_file)
            for predicted in _candidate_files(pred)
            for gold_file in gold_files
        )
        if conditional_eligible:
            conditional_ticket_ids.append(ticket_id)
            for k in SYMBOL_EVALUATION_K_VALUES:
                conditional_totals["hit"][k] += result["exact"]["hit_at"][str(k)]
                conditional_totals["recall"][k] += result["exact"]["recall_at"][str(k)]
            conditional_totals["reciprocal_rank"] += result["exact"]["reciprocal_rank"]
            if candidate_symbols:
                coverage_counts["candidate_output_rows"] += 1
            else:
                missing_candidate_output_ticket_ids.append(ticket_id)
            candidate_result = evaluate_ranked_symbols(
                candidate_symbols,
                gold_symbols,
                k_values=(10, 30),
            )
            for k in (10, 30):
                candidate_totals["hit"][k] += candidate_result["exact"]["hit_at"][str(k)]
                candidate_totals["recall"][k] += candidate_result["exact"]["recall_at"][str(k)]
            candidate_totals["reciprocal_rank"] += candidate_result["exact"]["reciprocal_rank"]

        diagnostics = pred.get("stage3_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        llm_requested = bool(diagnostics.get("requested"))
        llm_eligible = bool(diagnostics.get("eligible", candidate_symbols))
        llm_attempted = bool(diagnostics.get("llm_attempted"))
        llm_valid = bool(
            diagnostics.get("llm_valid", diagnostics.get("llm_rerank_used", False))
        )
        fallback_used = bool(
            diagnostics.get(
                "fallback_used",
                llm_requested and llm_eligible and not llm_valid,
            )
        )
        timed_out = bool(diagnostics.get("timed_out"))
        if llm_requested:
            coverage_counts["llm_requested_rows"] += 1
        if llm_requested and llm_eligible:
            coverage_counts["llm_eligible_rows"] += 1
            coverage_counts["llm_attempted_rows"] += int(llm_attempted)
            coverage_counts["llm_valid_rows"] += int(llm_valid)
            coverage_counts["fallback_rows"] += int(fallback_used)
            coverage_counts["timeout_rows"] += int(timed_out)
            if fallback_used:
                reason = str(diagnostics.get("fallback_reason") or "unspecified").strip()
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        per_ticket.append(
            {
                "ticket_id": ticket_id,
                "conditional_eligible": conditional_eligible,
                "gold_symbol_count": len(gold_symbols),
                "prediction_symbol_count": len(ranked_symbols),
                "candidate_symbol_count": len(candidate_symbols),
                "retrieval_symbol_count": len(retrieval_symbols),
                "exact": result["exact"],
                "retrieval_exact": baseline_result["exact"],
                "relaxed_diagnostic": result["relaxed_diagnostic"],
            }
        )

    eligible_rows = len(gold_by_ticket)
    conditional_rows = len(conditional_ticket_ids)
    final_exact = _aggregate_symbol_metrics(totals["exact"], eligible_rows)
    baseline_exact = _aggregate_symbol_metrics(baseline_totals, eligible_rows)
    return {
        "schema_version": evaluation_contract()["schema_version"],
        "eligible_rows": eligible_rows,
        "conditional_eligible_rows": conditional_rows,
        "conditional_eligible_ticket_ids": conditional_ticket_ids,
        "missing_prediction_rows": len(missing_prediction_ticket_ids),
        "missing_prediction_ticket_ids": missing_prediction_ticket_ids,
        "missing_symbol_output_rows": len(missing_symbol_output_ticket_ids),
        "missing_symbol_output_ticket_ids": missing_symbol_output_ticket_ids,
        "missing_candidate_output_rows": len(missing_candidate_output_ticket_ids),
        "missing_candidate_output_ticket_ids": missing_candidate_output_ticket_ids,
        "invalid_prediction_symbol_count": invalid_prediction_symbol_count,
        "invalid_candidate_symbol_count": invalid_candidate_symbol_count,
        "invalid_retrieval_symbol_count": invalid_retrieval_symbol_count,
        "exact": final_exact,
        "conditional_exact": _aggregate_symbol_metrics(
            conditional_totals,
            conditional_rows,
        ),
        "relaxed_diagnostic": _aggregate_symbol_metrics(
            totals["relaxed_diagnostic"],
            eligible_rows,
        ),
        "candidate_exact": {
            "eligible_rows": conditional_rows,
            **_aggregate_symbol_metrics(candidate_totals, conditional_rows),
        },
        "coverage": _symbol_coverage_metrics(
            coverage_counts,
            fallback_reasons=fallback_reasons,
            eligible_rows=eligible_rows,
            conditional_rows=conditional_rows,
        ),
        "paired_exact": _paired_symbol_metrics(
            baseline_exact=baseline_exact,
            final_exact=final_exact,
            paired_differences=paired_differences,
            paired_outcomes=paired_outcomes,
            rank_outcomes=rank_outcomes,
            rows=eligible_rows,
        ),
        "per_ticket": per_ticket,
    }


def _formal_prediction_symbols(
    row: dict[str, Any],
    *,
    field: str,
) -> tuple[list[SymbolEvaluationItemV1], int]:
    raw = row.get(field)
    if not isinstance(raw, list):
        return [], 0
    symbols: list[SymbolEvaluationItemV1] = []
    invalid = 0
    for candidate in raw:
        if not isinstance(candidate, dict):
            invalid += 1
            continue
        try:
            symbol = SymbolEvaluationItemV1.from_mapping(candidate)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols, invalid


def _aggregate_symbol_metrics(totals: dict[str, Any], denominator: int) -> dict[str, float]:
    k_values = tuple(sorted(int(k) for k in totals["hit"]))
    metrics = {
        f"hit_at_{k}": _ratio(totals["hit"][k], denominator)
        for k in k_values
    }
    metrics.update(
        {
            f"recall_at_{k}": _ratio(totals["recall"][k], denominator)
            for k in k_values
        }
    )
    metrics["mrr"] = _ratio(totals["reciprocal_rank"], denominator)
    return metrics


def _symbol_coverage_metrics(
    counts: dict[str, int],
    *,
    fallback_reasons: dict[str, int],
    eligible_rows: int,
    conditional_rows: int,
) -> dict[str, Any]:
    llm_denominator = counts["llm_eligible_rows"]
    return {
        **counts,
        "prediction_coverage": _ratio(counts["prediction_rows"], eligible_rows),
        "final_output_coverage": _ratio(counts["final_output_rows"], eligible_rows),
        "candidate_output_coverage": _ratio(
            counts["candidate_output_rows"],
            conditional_rows,
        ),
        "llm_attempt_coverage": _ratio(counts["llm_attempted_rows"], llm_denominator),
        "llm_valid_coverage": _ratio(counts["llm_valid_rows"], llm_denominator),
        "fallback_rate": _ratio(counts["fallback_rows"], llm_denominator),
        "timeout_rate": _ratio(counts["timeout_rows"], llm_denominator),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "rate_denominators": {
            "prediction_and_final_output": eligible_rows,
            "candidate_output": conditional_rows,
            "llm_attempt_valid_fallback_timeout": llm_denominator,
        },
    }


def _paired_symbol_metrics(
    *,
    baseline_exact: dict[str, float],
    final_exact: dict[str, float],
    paired_differences: dict[str, list[float]],
    paired_outcomes: dict[str, dict[str, list[str]]],
    rank_outcomes: dict[str, list[str]],
    rows: int,
) -> dict[str, Any]:
    metric_names = tuple(paired_differences)
    return {
        "rows": rows,
        "baseline_source": "stage3_retrieval_symbols",
        "final_source": "stage3_ranked_symbols",
        "baseline": baseline_exact,
        "final": final_exact,
        "delta": {
            name: _ratio(sum(paired_differences[name]), rows)
            for name in metric_names
        },
        "bootstrap_95_ci": {
            "samples": 2_000,
            "seed": 100,
            **{
                name: _bootstrap_mean_ci(values, seed=100 + index)
                for index, (name, values) in enumerate(paired_differences.items())
            },
        },
        "hit_outcomes": {
            metric: _outcome_summary(outcomes)
            for metric, outcomes in paired_outcomes.items()
        },
        "rank_outcomes": _outcome_summary(rank_outcomes),
    }


def _outcome_summary(outcomes: dict[str, list[str]]) -> dict[str, Any]:
    return {
        key: {"count": len(ticket_ids), "ticket_ids": ticket_ids}
        for key, ticket_ids in outcomes.items()
    }


def _paired_hit_bucket(baseline_hit: int, final_hit: int) -> str:
    if not baseline_hit and not final_hit:
        return "both_miss"
    if final_hit and not baseline_hit:
        return "improved"
    if baseline_hit and not final_hit:
        return "worsened"
    return "unchanged"


def _paired_rank_bucket(
    baseline_rank: int | None,
    final_rank: int | None,
) -> str:
    if baseline_rank is None and final_rank is None:
        return "both_miss"
    if final_rank is not None and (
        baseline_rank is None or final_rank < baseline_rank
    ):
        return "improved"
    if baseline_rank is not None and (
        final_rank is None or final_rank > baseline_rank
    ):
        return "worsened"
    return "unchanged"


def discover_base_commit_existing_gold_files(
    gold_rows: list[dict[str, Any]],
    repo_cache_dir: Path,
    *,
    ticket_rows: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Return gold files that physically exist in each ticket's base commit.

    This runs only during evaluation. The result is never passed to retrieval,
    so it cannot reveal the correct file to the model.
    """

    ticket_context = {
        _ticket_id(row): row for row in (ticket_rows or []) if _ticket_id(row)
    }
    result: dict[str, list[str]] = {}
    commit_available: dict[tuple[str, str], bool] = {}
    file_available: dict[tuple[str, str, str], bool] = {}
    for gold in gold_rows:
        ticket_id = _ticket_id(gold)
        if not ticket_id:
            continue
        context = ticket_context.get(ticket_id, {})
        repository = _repository_name(gold) or _repository_name(context)
        base_commit = str(gold.get("base_commit") or context.get("base_commit") or "").strip()
        local_repo = str(context.get("local_repo_path") or "").strip()
        repo_path = (
            Path(local_repo)
            if local_repo
            else repo_cache_dir / _safe_repository_name(repository)
        )
        if not repo_path.is_dir() or not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
            continue
        commit_key = (str(repo_path.resolve()), base_commit)
        if commit_key not in commit_available:
            commit_available[commit_key] = _git_object_exists(repo_path, f"{base_commit}^{{commit}}")
        if not commit_available[commit_key]:
            continue
        existing: list[str] = []
        for gold_file in _gold_files(gold):
            normalized = _normalize_path(gold_file)
            key = (*commit_key, normalized)
            if key not in file_available:
                file_available[key] = _git_object_exists(repo_path, f"{base_commit}:{normalized}")
            if file_available[key]:
                existing.append(normalized)
        result[ticket_id] = existing
    return result


def _git_object_exists(repo_path: Path, object_name: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), "cat-file", "-e", object_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _safe_repository_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in normalized)


def _align_rows(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pred_by_id = {_ticket_id(row): row for row in pred_rows if _ticket_id(row)}
    if pred_by_id and any(_ticket_id(gold) for gold in gold_rows):
        idless_predictions = iter(row for row in pred_rows if not _ticket_id(row))
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for gold in gold_rows:
            ticket_id = _ticket_id(gold)
            if ticket_id:
                pairs.append((gold, pred_by_id.get(ticket_id, {})))
            else:
                pairs.append((gold, next(idless_predictions, {})))
        return pairs

    # Positional data is still supported, but every gold row must remain in the
    # denominator. A short prediction file therefore counts as misses instead
    # of silently dropping the unpredicted tickets.
    return [
        (gold, pred_rows[index] if index < len(pred_rows) else {})
        for index, gold in enumerate(gold_rows)
    ]


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _repository_name(row: dict[str, Any]) -> str:
    return str(row.get("repo") or row.get("repository") or row.get("project") or "").strip()


def _gold_files(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths"):
        values.extend(_as_list(row.get(key)))
    for key in ("file_path", "file"):
        values.extend(_as_list(row.get(key)))
    for nested_key in ("ground_truth", "json_ground_truth", "bug_location"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            values.extend(_gold_files(nested))
    return _unique_paths(values)


def _candidate_files(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_files")
    if not isinstance(candidates, list):
        candidates = row.get("stage2_localized_files")
    if not isinstance(candidates, list):
        candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("candidates")
    values: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                values.extend(_as_list(candidate.get("file_path") or candidate.get("file")))
    if not values and isinstance(row.get("bug_location"), dict):
        location = row["bug_location"]
        if isinstance(location.get("bug_location"), dict):
            location = location["bug_location"]
        values.extend(_as_list(location.get("file_path") or location.get("file")))
    return _unique_paths(values)


def _stage1_candidate_files(row: dict[str, Any]) -> list[str]:
    """Read only the explicit pre-rerank output; never substitute final Top-K."""

    candidates = row.get("stage1_candidate_files")
    if not isinstance(candidates, list):
        candidates = row.get("stage1_candidates")
    values: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                values.extend(_as_list(candidate.get("file_path") or candidate.get("file")))
            elif isinstance(candidate, str):
                values.append(candidate)
    return _unique_paths(values)


def _gold_symbols(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "fixed_symbols",
        "modified_symbols",
        "changed_symbols",
        "bug_fix_symbols",
        "symbols",
        "symbol_names",
        "functions",
        "function_names",
        "methods",
        "method_names",
        "classes",
        "class_names",
    ):
        values.extend(_as_list(row.get(key)))
    for key in ("symbol_name", "symbol_qualified_name", "function_name", "function", "method", "class_name", "class"):
        values.extend(_as_list(row.get(key)))
    for nested_key in ("ground_truth", "json_ground_truth", "bug_location"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            values.extend(_gold_symbols(nested))
    return _unique_symbols(values)


def _candidate_symbols(row: dict[str, Any]) -> list[str]:
    candidates = row.get("stage3_ranked_symbols")
    if not isinstance(candidates, list):
        candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("candidates")
    values: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            values.extend(
                _as_list(
                    candidate.get("symbol_qualified_name")
                    or candidate.get("symbol_name")
                    or candidate.get("function_name")
                    or candidate.get("function")
                    or candidate.get("class_name")
                )
            )
    if not values and isinstance(row.get("bug_location"), dict):
        location = row["bug_location"]
        if isinstance(location.get("bug_location"), dict):
            location = location["bug_location"]
        values.extend(
            _as_list(
                location.get("symbol_qualified_name")
                or location.get("symbol_name")
                or location.get("function_name")
                or location.get("function")
                or location.get("class_name")
            )
        )
    return _unique_symbols(values)


def _first_relevant_file_rank(ranked_files: list[str], gold_files: list[str]) -> int | None:
    for index, predicted in enumerate(ranked_files, start=1):
        if any(_file_matches(predicted, gold) for gold in gold_files):
            return index
    return None


def _first_relevant_symbol_rank(ranked_symbols: list[str], gold_symbols: list[str]) -> int | None:
    for index, predicted in enumerate(ranked_symbols, start=1):
        if any(_symbol_matches(predicted, gold) for gold in gold_symbols):
            return index
    return None


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    if not left or not right:
        return False
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def _symbol_matches(predicted: str, gold: str) -> bool:
    left = _normalize_symbol(predicted)
    right = _normalize_symbol(gold)
    if not left or not right:
        return False
    return left == right or left.endswith("." + right) or right.endswith("." + left)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_as_list(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _unique_paths(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = _normalize_path(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _unique_symbols(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = _normalize_symbol(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def _normalize_symbol(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::", maxsplit=1)[1]
    text = text.split(":", maxsplit=1)[0]
    return text.replace("#", ".").strip().lstrip(".")


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0}
    if len(values) == 1:
        value = float(values[0])
        return {"lower": value, "upper": value}
    generator = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    return {
        "lower": round(float(means[lower_index]), 6),
        "upper": round(float(means[upper_index]), 6),
    }


if __name__ == "__main__":
    main()
